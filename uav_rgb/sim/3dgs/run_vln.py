#!/usr/bin/env python3
"""3DGS 上的视觉语言导航评测：SAGE-3D 指令或单条自由输入指令，由智能体决定停止。

从仓库根、用 3dgs-venv 跑：
  python -m sim.3dgs.run_vln --policy teacher --out $D/exp-vln/teacher --videos 3 \
      --scene $D/3dgs-scenes/sage3d_839992/scene.json [--scene ...]
  python -m sim.3dgs.run_vln --policy reactive --grounder vlm --scene <scene.json> \
      --instruction '飞到走廊尽头并悬停' --spawn 0 0 1.5 --videos 1 --out <new-run-dir>
每条航线缺省只取一句指令（句式按航线轮换），--all-instructions 取全部。

策略：
  teacher  占据 A* 教师，直接拿航线终点当目标：等于已经听懂了指令（特权），是上界，也是以后蒸馏的老师
  reactive  目标条件局部控制，只看当前深度、位姿和目标方向，不在线查询全局占据图或调用 A*
  forward  一直前进、从不喊停，不看指令：下界
数据集评测里的 teacher/reactive 使用 scene waypoint 验证控制层，不能作为语言成绩。
自由输入模式将用户指令交给 VLM Grounder 和零特权方位控制器，输出未评分演示轨迹。
student 模式只接受 prev_dim=15 的 VLN_NAV 权重，确保它实际接收 VLM 方位条件。

``--grounder lexical --catalog <JSON>`` 用独立地标目录验证指令换目标，仍不等同端到端语言学生。
自由输入结果只记录轨迹、碰撞和停止，不把 scene waypoint 当作真值目标，也不计算 SR / SPL。

产物：<out>/<场景>/<航线>_<指令序号>/（run_episode 的逐步记录；前 --videos 集另有 views.mp4）、
<out>/episodes.jsonl（每集一行）、<out>/vln_eval.json（总表，另按句式 / SAGE split / 场景分组）。
"""
import argparse
import json
import math
import os
import sys
import time

from . import collect_3dgs
from . import collision as collision_mod
from . import planner as planner_mod
from . import renderer as renderer_mod
from . import reactive
from . import scene as scene_mod
from . import video as video_mod
from . import vln
from . import grounder as grounder_mod
from .dynamics import CfSitlParams
from .eval_3dgs import reach_radius_of, route_shortest_path_m

METRIC_VERSION = "3dgs.vln.v3"
STEPS_HOLD_S = 0.5      # 航线 max_steps 是导入时按 hold 0.5 估的（scene_import_sage3d），别的 hold 按时长换算


def make_policy(name, occ_planner, waypoint, hold, stop_radius=1.0, clearance="ray"):
    if name == "teacher":
        return planner_mod.OccAstarTeacher(
            occ_planner, waypoint, stop_radius=stop_radius, hold_s=hold, clearance=clearance)
    if name == "reactive":
        return reactive.ReactiveGoalPolicy(waypoint, stop_radius=stop_radius)
    return collect_3dgs.ForwardPolicy()


class VlmBeliefTracker:
    """维护可见路标方位。任务 STOP 需要连续两次新鲜的同一近景目标证据。"""

    _ACT_SHORT = {0: "SAFE", 1: "FWD", 2: "LEFT", 3: "RIGHT"}

    def __init__(self, vlm_grounder, instruction, update_every=2, stage_instructions=None):
        self.vlm_grounder = vlm_grounder
        self.instruction = str(instruction or "")
        self.stage_instructions = [str(text).strip() for text in (stage_instructions or []) if str(text).strip()]
        self.stage_index = 0
        self.task_complete = False
        self.stage_confirmed = False
        self.stage_confirmed_index = None
        self.update_every = max(1, int(update_every))
        self.step_idx = 0
        self.history = []
        self.trace = []
        self.prev_vlm_bin = None
        self.last_seen_bin = None
        self.miss_streak = 0
        self.last_near_target = None
        self.confirmed_target = False
        self.turned_toward_target = False
        if vlm_grounder is not None and hasattr(vlm_grounder, "reset_episode"):
            vlm_grounder.reset_episode()
        self.last_vg = grounder_mod.VisualGrounding(grounder_mod.BEARING_SEARCH, 0, 0.0, "init", 0.0, "search")

    def note_step(self, action_executed: int):
        self.turned_toward_target |= (self.last_vg.bearing_bin, action_executed) in ((1, 2), (7, 3))
        if self.vlm_grounder is not None and hasattr(self.vlm_grounder, "note_action"):
            self.vlm_grounder.note_action(self._ACT_SHORT.get(int(action_executed), "ACT"))

    def __call__(self, rgb, _state):
        self.confirmed_target = False
        self.history.append(rgb)
        if len(self.history) > 4:
            self.history = self.history[-4:]
        if not self.instruction.strip() or self.vlm_grounder is None:
            self.last_vg = grounder_mod.VisualGrounding(grounder_mod.BEARING_SEARCH, 0, 0.0, "empty_instruction", 0.0, "search")
            self.step_idx += 1
            return self.last_vg.bearing_bin, 0
        if self.step_idx % self.update_every != 0:
            bb = 0 if self.turned_toward_target else self.last_vg.bearing_bin
            self.step_idx += 1
            return bb, 0

        instruction = self.instruction
        if self.stage_instructions:
            instruction = ("Treat the current stage as the active FINAL destination until confirmed reached.\n"
                           f"Overall instruction: {self.instruction}\n"
                           f"Current stage {self.stage_index + 1}/{len(self.stage_instructions)}: "
                           f"{self.stage_instructions[self.stage_index]}")
        vg = self.vlm_grounder.ground_step(self.history, instruction)
        self.stage_confirmed = False
        self.stage_confirmed_index = None
        self.turned_toward_target = False
        raw_bb = vg.bearing_bin
        if raw_bb in (0, 1, 7):
            eff_bb = 0 if {self.prev_vlm_bin, raw_bb} == {1, 7} else raw_bb
            self.miss_streak = 0
            self.last_seen_bin = raw_bb
        elif raw_bb == grounder_mod.BEARING_SEARCH and self.last_seen_bin is not None and self.miss_streak == 0:
            self.miss_streak = 1
            eff_bb = 0
        elif raw_bb == grounder_mod.BEARING_SEARCH:
            self.miss_streak += 1
            eff_bb = grounder_mod.BEARING_SEARCH
        else:
            eff_bb = raw_bb
        self.prev_vlm_bin = raw_bb
        near_target = raw_bb == 0 and vg.goal_mode == "final" and vg.near_goal == 1
        active_target_text = self.stage_instructions[self.stage_index] if self.stage_instructions else self.instruction
        target_id = (vg.reason.strip().casefold() if near_target and
                     grounder_mod.instruction_target_matches(active_target_text, vg.reason) else None)
        confirmed = bool(target_id and target_id == self.last_near_target)
        if self.stage_instructions and confirmed:
            self.stage_confirmed = True
            self.stage_confirmed_index = self.stage_index
            if self.stage_index + 1 < len(self.stage_instructions):
                self.stage_index += 1
            else:
                self.task_complete = True
        eff_ng = int(near_target and confirmed and (not self.stage_instructions or self.task_complete))
        self.confirmed_target = confirmed and (not self.stage_instructions or self.task_complete)
        self.last_near_target = None if self.stage_instructions and confirmed and not self.task_complete else target_id
        self.last_vg = grounder_mod.VisualGrounding(
            eff_bb, eff_ng, vg.conf, vg.reason, vg.latency_ms, goal_mode=vg.goal_mode,
        )
        self.trace.append({
            "step": self.step_idx,
            "raw_bearing_bin": raw_bb,
            "raw_near_goal": vg.near_goal,
            "bearing_bin": eff_bb,
            "goal_mode": vg.goal_mode,
            "near_goal": eff_ng,
            "confirmed_target": self.confirmed_target,
            "stage_index": self.stage_index,
            "stage_confirmed": self.stage_confirmed,
            "stage_confirmed_index": self.stage_confirmed_index,
            "task_complete": self.task_complete,
            "miss_streak": self.miss_streak,
            "conf": vg.conf,
            "reason": vg.reason,
            "latency_ms": vg.latency_ms,
        })
        self.step_idx += 1
        return eff_bb, eff_ng


def result_text(m):
    """视频结尾大字；以 SUCCESS 开头画绿色，其余红色（view_layout.panel）。"""
    if m.get("success"):
        return "SUCCESS stop %.2f m" % m["ne_m"]
    if m.get("collision"):
        return "COLLISION"
    if m["stopped"]:
        return "WRONG STOP %.1f m" % m["ne_m"]
    return "NO STOP %.1f m" % m["ne_m"]


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_grounding(selector, catalog, instruction):
    """把指令落到外部目录中的目标坐标；拒绝时返回结果和 ``None``。"""
    result = selector.ground(instruction, list(catalog or ()))
    if not result.accepted:
        return result, None
    return result, dict(catalog[result.target])


def _require_external_catalog(catalog_path, scene_path):
    """拒绝把 scene.json 自己当作语言地标目录，避免 waypoint 走后门。"""
    try:
        same = os.path.samefile(catalog_path, scene_path)
    except (FileNotFoundError, OSError):
        same = os.path.realpath(catalog_path) == os.path.realpath(scene_path)
    if same:
        raise ValueError("地标目录必须是独立文件，不能复用 scene.json")


def _grounding_summary(rows, failures):
    """报告指令落地覆盖、接受率和准确率；仿真失败仍保留已完成的落地结果。"""
    requested = len(rows) + len(failures)
    records = [item["grounding"] for item in [*rows, *failures]
               if isinstance(item.get("grounding"), dict)]
    accepted = [item for item in records if item.get("accepted")]
    correct = sum(bool(item.get("correct")) for item in records)
    return {
        "n_requested": requested,
        "n_grounded": len(records),
        "n_missing": requested - len(records),
        "n_accepted": len(accepted),
        "n_rejected": len(records) - len(accepted),
        "n_correct": correct,
        "coverage": len(accepted) / requested if requested else None,
        "accuracy_including_rejects": correct / requested if requested else None,
        "accuracy_among_accepted": correct / len(accepted) if accepted else None,
    }


def _manual_demo_metrics(actions, oracle, manifest):
    """汇报自由指令的飞行过程，不用 scene waypoint 计算成功率。"""
    points = vln.trajectory(actions, oracle)
    path_m = sum(math.dist(start, end) for start, end in zip(points, points[1:]))
    return {
        "success": None,
        "oracle_success": None,
        "ne_m": None,
        "spl": None,
        "ndtw": None,
        "sdtw": None,
        "path_m": path_m,
        "shortest_m": None,
        "stopped": bool(manifest.get("stopped_by_agent")),
        "collision": any(bool(action.get("collision")) for action in actions),
        "steps": len(actions),
        **vln._action_diagnostics(actions, path_m),
    }


def _spawn_within_scene_bounds(occ, spawn):
    """只接受落在已构建碰撞代理覆盖范围内的自由演示起点。"""
    raw = getattr(occ, "raw", None)
    if raw is None or raw.size == 0:
        return False
    low = occ.origin[:2]
    high = (low[0] + (occ.shape[0] - 1) * occ.voxel,
            low[1] + (occ.shape[1] - 1) * occ.voxel)
    xy = (float(spawn["x"]), float(spawn["y"]))
    return all(float(low[i]) <= xy[i] <= float(high[i]) for i in range(2))


def _manual_spawn_is_safe(occ, spawn):
    x, y, z = (float(spawn[key]) for key in ("x", "y", "z"))
    if getattr(occ, "collision_model", None) == "v2" and hasattr(occ, "spawn_ok"):
        return bool(occ.spawn_ok(x, y, z))
    if occ.collides(x, y, z):
        return False
    clearance = getattr(occ, "clearance", None)
    if clearance is None:
        return True
    required = float(getattr(occ, "body_radius", 0.0)) + float(
        getattr(occ, "spawn_drift_margin", 0.0))
    return float(clearance(x, y, z)) >= required


def _episode_route(scene, episode):
    route_index = episode.get("route_index")
    if not isinstance(route_index, int) or route_index < 0:
        return None
    routes = scene.get("routes") or []
    if route_index < len(routes):
        return routes[route_index]
    multi_index = route_index - len(routes)
    multi_routes = scene.get("multi_stage_routes") or []
    if multi_index >= len(multi_routes):
        return None
    return multi_routes[multi_index]


def _episode_spawn(scene, episode):
    route = _episode_route(scene, episode)
    if route is None:
        return None
    spawn = route.get("spawn") or {}
    if not all(axis in spawn for axis in ("x", "y", "z")):
        return None
    return tuple(float(spawn.get(axis, 0.0)) for axis in ("x", "y", "z", "yaw_deg"))


def _episode_budget(scene, episode):
    route = _episode_route(scene, episode)
    if route is None:
        return None
    return int(route.get("max_steps", scene.get("max_steps", 0)))


def _swapped_instruction(episodes, index, scene):
    episode = episodes[index]
    partner = index ^ 1
    candidates = episodes[partner:partner + 1] + episodes[index + 1:] + episodes[:index]
    goal = episode.get("goal_object")
    spawn = _episode_spawn(scene, episode)
    budget = _episode_budget(scene, episode)
    if not goal or spawn is None or budget is None:
        return None
    for candidate in candidates:
        if (candidate.get("goal_object") and candidate["goal_object"] != goal
                and candidate["text"] != episode["text"]
                and candidate.get("route_contract") == episode.get("route_contract")
                and candidate.get("split") == episode.get("split")
                and candidate.get("type") == episode.get("type")
                and len(candidate.get("stage_waypoints") or []) == len(episode.get("stage_waypoints") or [])
                and _episode_budget(scene, candidate) == budget
                and _episode_spawn(scene, candidate) == spawn):
            return candidate
    return None


def _episode_goal_waypoint(scene, episode):
    if episode is None or episode.get("manual_demo"):
        return None
    if episode.get("route_contract") == "sage.multi_stage.v2":
        routes = scene.get("multi_stage_routes") or []
        route_index = episode.get("multi_stage_index")
    else:
        routes = scene.get("routes") or []
        route_index = episode.get("route_index")
    if not routes:
        return dict(scene["waypoint"]) if route_index in (None, 0) and scene.get("waypoint") else None
    if not isinstance(route_index, int) or not 0 <= route_index < len(routes):
        return None
    waypoint = routes[route_index].get("waypoint")
    return dict(waypoint) if isinstance(waypoint, dict) else None


def _active_bearing_source(args):
    if args.grounder == "vlm":
        return "vlm_grounder"
    if args.grounder == "blind_unknown":
        return "blind_unknown"
    return f"g0_{args.g0_mode}"


def run_scene(path, args, params, lexical_grounder=None, catalog=None, catalog_path=None,
              vlm_grounder=None, student_proc=None):
    import numpy as np
    import random
    scene = scene_mod.load_scene(path)
    manual_instruction = getattr(args, "instruction", None)
    if manual_instruction is not None:
        manual_spawn = getattr(args, "spawn", None)
        spawn = dict(scene["spawn"])
        if manual_spawn is not None:
            spawn.update(dict(zip(("x", "y", "z"), manual_spawn)))
        if getattr(args, "spawn_yaw_deg", None) is not None:
            spawn["yaw_deg"] = float(args.spawn_yaw_deg)
        scene["spawn"] = spawn
        scene["waypoint"] = None
        scene["route"] = None
        if getattr(args, "manual_steps", None) is not None:
            scene["max_steps"] = int(args.manual_steps)
        eps = [{"key": f"{scene['name']}/manual", "route_index": None, "route_name": "manual",
                "instruction_index": 0, "route_key": "manual", "split": scene.get("split"),
                "type": "Manual instruction", "text": manual_instruction,
                "start_object": None, "goal_object": None, "manual_demo": True}]
    else:
        eps = vln.episodes(scene, splits=args.split)
        if not args.all_instructions:
            eps = vln.one_per_route(eps, seed=args.seed)
        if args.limit:
            eps = eps[:args.limit]
    if not eps:
        print(f"{path}: 没有带指令的航线（不是 SAGE-3D 导入的场景，或 --split 过滤光了）", file=sys.stderr)
        return [], []
    grounder_kind = getattr(args, "grounder", "none")
    collect_mode = bool(getattr(args, "collect", False))
    rows, failures = [], []
    runnable = []
    for ep in eps:
        if (ep.get("route_contract") == "sage.multi_stage.v2"
                and (grounder_kind != "vlm" or args.policy not in ("reactive", "student") or collect_mode)):
            ep_dir = os.path.join(args.out, scene["name"], f"{ep['route_name']}_{ep['instruction_index']}")
            failures.append({**ep, "scene": scene["name"], "dir": ep_dir,
                             "error": "ValueError: 多阶段任务需要 --grounder vlm、reactive/student 策略且不能使用 --collect"})
            continue
        runnable.append(ep)
    eps = runnable
    if not eps:
        return rows, failures
    identity = collect_3dgs.scene_identity(scene)
    gaussians = (scene_mod.read_gaussian_ply(scene["ply_path"])
                 if scene.get("collision", {}).get("source", "gaussians") == "gaussians" else None)
    occ = collision_mod.from_scene(scene, gaussians=gaussians)
    del gaussians
    if manual_instruction is not None:
        spawn = scene["spawn"]
        within_bounds = _spawn_within_scene_bounds(occ, spawn)
        is_safe = within_bounds and _manual_spawn_is_safe(occ, spawn)
        if not is_safe:
            return [], [{"key": f"{scene['name']}/manual", "scene": scene["name"],
                         "route_name": "manual", "type": "Manual instruction",
                         "text": manual_instruction,
                         "error": "ValueError: 起飞点不在碰撞代理范围内或与占据代理冲突（或起飞净空不足）"}]
    renderer = renderer_mod.make_renderer(scene)
    occ_planner = planner_mod.OccPlanner(occ)
    radius = reach_radius_of(scene)
    ref_paths = vln.sage_ref_paths(scene)
    clearance = getattr(args, "teacher_clearance", "ray")
    cf_mode = getattr(args, "counterfactual", "correct")
    blind_rgb = bool(getattr(args, "blind_rgb", False))
    rng = random.Random(args.seed)
    for n, ep in enumerate(eps):
        ep_dir = os.path.join(args.out, scene["name"], f"{ep['route_name']}_{ep['instruction_index']}")
        instruction_ep = _swapped_instruction(eps, n, scene) if cf_mode == "swap" else ep
        if instruction_ep is None:
            failures.append({**ep, "scene": scene["name"], "dir": ep_dir,
                             "error": "ValueError: swap 缺少同 split、同句式/阶段数/预算、同起点的异目标指令"})
            continue
        is_multi_stage = ep.get("route_contract") == "sage.multi_stage.v2"
        if is_multi_stage:
            route = (scene.get("multi_stage_routes") or [])[ep["multi_stage_index"]]
            scene.update({"spawn": dict(route["spawn"]), "waypoint": dict(route["waypoint"]),
                          "max_steps": int(route["max_steps"]),
                          "route": {"index": ep["route_index"], "name": route["name"],
                                    "contract": ep["route_contract"]}})
        elif not ep.get("manual_demo"):
            scene_mod.apply_route(scene, ep["route_index"])
        identity = collect_3dgs.scene_identity(scene, ply_sha256=identity["ply_sha256"])
        evaluation_goal = None if ep.get("manual_demo") else dict(scene["waypoint"])
        supplied_goal_waypoint = (None if cf_mode == "zero" else
                                  _episode_goal_waypoint(scene, instruction_ep))
        supplied_goal_object = instruction_ep.get("goal_object") if supplied_goal_waypoint else None
        if (supplied_goal_waypoint and instruction_ep.get("route_contract") == "sage.multi_stage.v2"
                and instruction_ep.get("stage_waypoints")):
            supplied_goal_object = instruction_ep["stage_waypoints"][-1].get("goal_object")
        supplied_goal_record = {"supplied_goal_object": supplied_goal_object,
                                "supplied_goal_waypoint": supplied_goal_waypoint}
        supplied_text = "" if cf_mode == "zero" else instruction_ep["text"]
        supplied_key = None if cf_mode == "zero" else instruction_ep["key"]

        grounding = None
        policy_goal = evaluation_goal
        if lexical_grounder is not None and not is_multi_stage:
            try:
                result, policy_goal = resolve_grounding(lexical_grounder, catalog, supplied_text)
            except Exception as e:  # noqa: BLE001 - 单集输入错误要保留在请求分母
                failures.append({**ep, **supplied_goal_record, "scene": scene["name"], "dir": ep_dir,
                                 "error": f"{type(e).__name__}: {e}"})
                continue
            grounding = {"target": result.target, "score": result.score, "margin": result.margin,
                         "accepted": result.accepted, "reason": result.reason}
            if not result.accepted:
                failures.append({**ep, **supplied_goal_record, "scene": scene["name"], "dir": ep_dir,
                                 "grounding": grounding, "error": f"ValueError: grounder 拒绝指令: {result.reason}"})
                continue
        if grounding is not None:
            grounding.update({
                "correct": grounding["target"] == supplied_goal_object,
                "policy_goal": policy_goal,
                "catalog_sha256": collect_3dgs._sha256(catalog_path) if catalog_path else None,
                "scene_json_sha256": collect_3dgs._sha256(scene.get("path")) if scene.get("path") else None,
                "geometry_gid": identity["geometry_gid"],
                "ply_sha256": identity["ply_sha256"],
            })
        # 始终保留真实评估目标作为 scene["waypoint"]，防止 grounder 落错地标时把 manifest 与视频伪装成到达真目标。
        if evaluation_goal is not None:
            scene["waypoint"] = dict(evaluation_goal)
        scene["vln"] = {**ep, **supplied_goal_record,
                        "supplied_text": supplied_text, "supplied_instruction_key": supplied_key,
                        "counterfactual": cf_mode, "blind_rgb": blind_rgb,
                        **({"grounding": grounding} if grounding is not None else {})}
        steps = math.ceil(int(scene["max_steps"]) * STEPS_HOLD_S / args.hold)

        vlm_tracker = None
        if grounder_kind == "vlm":
            stage_instructions = [stage.get("instruction") or "" for stage in instruction_ep.get("stage_waypoints", [])]
            vlm_tracker = VlmBeliefTracker(vlm_grounder, supplied_text,
                                           update_every=getattr(args, "vlm_every", 2),
                                           stage_instructions=stage_instructions if supplied_text else None)
            goal_fn = vlm_tracker
        elif grounder_kind == "blind_unknown":
            goal_fn = lambda _rgb, _st: (grounder_mod.BEARING_SEARCH, 0)
        elif is_multi_stage:
            goal_fn = lambda _rgb, _st: (grounder_mod.BEARING_SEARCH, 0)
        else:
            gx, gy = float(policy_goal["x"]), float(policy_goal["y"])
            g0_mode = getattr(args, "g0_mode", "subgoal")
            if g0_mode == "direct":
                def _g0_direct(_rgb, st, _gx=gx, _gy=gy, _r=radius):
                    px, py = float(st.x), float(st.y)
                    ng = int(math.hypot(px - _gx, py - _gy) <= _r)
                    return vln.bearing_bin(float(st.yaw), (px, py), (_gx, _gy)), ng

                goal_fn = _g0_direct
            else:
                def _g0_subgoal(_rgb, st, _gx=gx, _gy=gy, _r=radius, _pl=occ_planner):
                    px, py = float(st.x), float(st.y)
                    speed = math.hypot(float(getattr(st, "vx", 0.0)), float(getattr(st, "vy", 0.0)))
                    return vln.subgoal_bearing_bin(
                        _pl, float(st.yaw), (px, py), (_gx, _gy), speed, _r)

                goal_fn = _g0_subgoal

        rgb_tf = (lambda img: np.zeros_like(img)) if blind_rgb else None
        dagger_meta = None
        if args.policy == "student":
            if collect_mode:
                expert = planner_mod.OccAstarTeacher(
                    occ_planner, policy_goal, stop_radius=radius, hold_s=args.hold, clearance=clearance)
                beta = float(getattr(args, "dagger_beta", 0.5))
                policy = collect_3dgs.DaggerPolicy(student_proc, expert, beta=beta, rng=rng, goal_fn=goal_fn)
                dagger_meta = {
                    "executed_by": "student",
                    "dagger_round": int(getattr(args, "dagger_round", 1)),
                    "beta": beta,
                    "student_ckpt": os.path.abspath(args.ckpt),
                    "student_ckpt_sha256": collect_3dgs._sha256(args.ckpt),
                }
            else:
                policy = collect_3dgs.StudentAdapter(student_proc, goal_fn=goal_fn, rgb_transform=rgb_tf)
        elif args.policy == "reactive" and grounder_kind in ("vlm", "blind_unknown"):
            policy = reactive.ReactiveBearingPolicy(goal_fn)
        elif is_multi_stage:
            policy = collect_3dgs.StudentAdapter(student_proc, goal_fn=goal_fn, rgb_transform=rgb_tf)
        else:
            try:
                policy = make_policy(args.policy, occ_planner, policy_goal, args.hold,
                                     stop_radius=radius, clearance=clearance)
            except TypeError:
                policy = make_policy(args.policy, occ_planner, policy_goal, args.hold, stop_radius=radius)
        try:
            man, err = collect_3dgs.run_episode(
                scene, ep_dir, renderer, occ, params, policy, steps, args.hold,
                split=identity["split"], seed=args.seed + n, episode_id=n,
                stop_on_collision=not collect_mode,
                truncate_on_collision=collect_mode,
                identity=identity,
                split_policy=None if collect_mode else "diag-vln-eval",
                dynamics="cf_sitl",
                dynamics_path=getattr(args, "dynamics_params", None),
                save_depth=bool(getattr(args, "save_depth", False)),
                dagger=dagger_meta,
                stop_on_request=args.stop_mode == "explicit",
                stop_after_safe=vln.STOP_STREAK if args.stop_mode == "legacy-safe" else None)
            if vlm_tracker is not None and vlm_tracker.trace:
                collect_3dgs._write_jsonl(os.path.join(ep_dir, "grounding_trace.jsonl"), vlm_tracker.trace)
            if err or man.get("error"):
                raise RuntimeError(err or man["error"])
            validity = man.get("validity", {})
            if not validity.get("physics_verified") or not validity.get("camera_verified"):
                raise RuntimeError(f"无效仿真观测: validity={validity}, dynamics_anomaly={man.get('dynamics_anomaly')}")
            actions = _read_jsonl(os.path.join(ep_dir, "actions.jsonl"))
            oracle = _read_jsonl(os.path.join(ep_dir, "oracle.jsonl"))
            if not actions or len(actions) != len(oracle):
                raise ValueError("动作与位姿记录必须非空、长度一致")
            # p 包含第一步运动，l 也必须从同一个动作前位姿开始。
            if ep.get("manual_demo"):
                m = _manual_demo_metrics(actions, oracle, man)
            else:
                start = {**actions[0]["pose_before"], "yaw_deg": actions[0].get("yaw_before", 0.0)}
                route_scene = dict(scene, waypoint=evaluation_goal)
                shortest = (ep["total_path_m"]
                            if is_multi_stage else
                            route_shortest_path_m(occ_planner, [{"pose_gt": start}] + oracle, route_scene))
                m = vln.episode_metrics(actions, oracle, evaluation_goal, radius, man["stopped_by_agent"],
                                        shortest, ref_path=(ep.get("ref_path") or ref_paths.get(ep["route_index"])),
                                        stage_goals=([stage["waypoint"] for stage in ep["stage_waypoints"]]
                                                     if is_multi_stage else None))
            if is_multi_stage:
                m["stage_reached"] = m["stage_reached_count"]
                m["stage_ordered_complete"] = m["stage_reached_count"] == m["stage_count"]
                m["stage_names"] = [stage.get("goal_object") for stage in ep["stage_waypoints"]]
            supplied_metrics = None
            if supplied_goal_waypoint is not None:
                supplied_scene = dict(scene, waypoint=supplied_goal_waypoint)
                supplied_stages = instruction_ep.get("stage_waypoints") if is_multi_stage else None
                supplied_shortest = (instruction_ep["total_path_m"]
                                     if is_multi_stage else
                                     route_shortest_path_m(occ_planner, [{"pose_gt": start}] + oracle,
                                                           supplied_scene))
                supplied_metrics = vln.episode_metrics(
                    actions, oracle, supplied_goal_waypoint, radius, man["stopped_by_agent"],
                    supplied_shortest,
                    ref_path=(instruction_ep.get("ref_path") or
                              ref_paths.get(instruction_ep["route_index"])),
                    stage_goals=([stage["waypoint"] for stage in supplied_stages]
                                 if supplied_stages else None))
            target_metrics = None
            if grounding is not None:
                target_scene = dict(scene, waypoint=policy_goal)
                target_shortest = route_shortest_path_m(
                    occ_planner, [{"pose_gt": start}] + oracle, target_scene)
                target_metrics = vln.episode_metrics(
                    actions, oracle, policy_goal, radius, man["stopped_by_agent"], target_shortest)
        except Exception as e:  # noqa: BLE001 - 一集崩了记下来接着跑，报告里单列
            failures.append({**ep, **supplied_goal_record, "scene": scene["name"], "dir": ep_dir,
                             **({"grounding": grounding} if grounding is not None else {}),
                             "error": f"{type(e).__name__}: {e}"})
            continue
        row = {**ep, "seed": args.seed + n, "supplied_text": supplied_text,
               "supplied_instruction_key": supplied_key, "counterfactual": cf_mode,
               **supplied_goal_record,
               "scene": scene["name"], **m, "error": err, "dir": ep_dir,
               "terminal_reason": man.get("terminal_reason")}
        if supplied_metrics is not None:
            row.update({f"supplied_goal_{key}": value for key, value in supplied_metrics.items()})
        if ep.get("manual_demo"):
            row.update(start_pose=dict(scene["spawn"]), step_budget=steps)
        if vlm_tracker is not None:
            lats = [x["latency_ms"] for x in vlm_tracker.trace]
            row["vlm_calls"] = len(lats)
            row["vlm_latency_ms_p50"] = collect_3dgs._percentile(lats, 50) if lats else None
            row["vlm_latency_ms_p95"] = collect_3dgs._percentile(lats, 95) if lats else None
        if grounding is not None:
            row["grounding"] = grounding
            row.update({f"target_{key}": value for key, value in target_metrics.items()})
        rows.append(row)
        wrong_goal = grounding is not None and grounding.get("correct") is False
        result_metrics = supplied_metrics if cf_mode == "swap" and supplied_metrics is not None else m
        if ep.get("manual_demo"):
            badge = ("COLLISION | unscored manual demo" if m["collision"] else
                     "VLM STOP | unscored manual demo" if m["stopped"] else
                     "NO STOP | unscored manual demo")
        else:
            badge = (f"WRONG GOAL {result_metrics['ne_m']:.1f} m" if wrong_goal else
                     result_text(result_metrics))
        print(f"[{scene['name']} {n + 1}/{len(eps)}] {badge:24s} {ep['type']:22s} {supplied_text}", file=sys.stderr)
        if n < args.videos:
            try:
                video = video_mod.compose_views_video(
                    ep_dir, scene, renderer, params.camera_mount, occ,
                    title=f"{args.policy} | {grounder_kind} | {scene['name']} | {ep['type']}", result=badge,
                    instruction=f"\"{supplied_text}\"",
                    pred_goal=policy_goal if grounding is not None else None,
                    goal_visible=not ep.get("manual_demo", False),
                    log=lambda msg: print(msg, file=sys.stderr))
                row["video"] = os.path.relpath(video["video"], args.out)
                row["video_metadata"] = {key: value for key, value in video.items() if key != "video"}
            except Exception as e:  # noqa: BLE001 - 录像是诊断产物，失败不能丢掉已完成的评测
                row["video_error"] = f"{type(e).__name__}: {e}"
                print(f"[{scene['name']} {n + 1}/{len(eps)}] 录像失败: {row['video_error']}", file=sys.stderr)
    return rows, failures


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", action="append", required=True, help="SAGE-3D / Wanderland 导入的 scene.json，可给多个")
    p.add_argument("--policy", choices=("teacher", "reactive", "forward", "student"), required=True,
                   help="teacher = A* 特权上界；student = 机载 C++ CNN-FiLM+STOP 学生；reactive = 局部控制；forward = 盲飞下界")
    p.add_argument("--ckpt", default=None, help="--policy student 时的 C++ 权重文件 (.bin)")
    p.add_argument("--infer-bin", default=None, help="uavrgb_infer 二进制路径（缺省读 build/check/uavrgb_infer）")
    p.add_argument("--collect", action="store_true", help="采集正式训练/DAgger 数据集（split_policy=None，撞后截断不拒收）")
    p.add_argument("--save-depth", action="store_true", help="采集时落盘渲染深度图供 --depth-aux 训练")
    p.add_argument("--dagger-beta", type=float, default=0.5, help="--policy student --collect 时的教师接管概率 β")
    p.add_argument("--dagger-round", type=int, default=1, help="DAgger 轮次编号")
    p.add_argument("--grounder", choices=("none", "lexical", "phrase", "vlm", "blind_unknown"), default="none",
                   help="lexical/phrase：目录检索；vlm：零特权视觉语言 Grounder（1 Hz）；blind_unknown：方位恒置 UNKNOWN(8) 消融")
    p.add_argument("--g0-mode", choices=("subgoal", "direct"), default="subgoal",
                   help="G0 特权方位来源：subgoal=占据 A* 局部绕障子目标；direct=纯直线欧氏终点方位（S1-PureDirect 剥离机载视觉绕障）")
    p.add_argument("--vlm-model", default=None, help="--grounder vlm 的模型路径（默认 UAV-PUR-4B）")
    p.add_argument("--vlm-every", type=int, default=2, help="VLM 更新步间隔（hold=0.5s 时 2 步 = 1 Hz）")
    p.add_argument("--counterfactual", choices=("correct", "swap", "zero"), default="correct",
                   help="语言反事实模式：correct=原指令，swap=同场景异目标错配指令，zero=空指令")
    p.add_argument("--blind-rgb", action="store_true", help="将喂给 student 的图像全置黑（排除复读机/纯条件捷径）")
    p.add_argument("--catalog", action="append", default=None,
                   help="--grounder lexical/phrase 时必填，每场景一个 uavrgb.vln_landmark_catalog.v1 JSON；可重复")
    p.add_argument("--teacher-clearance", choices=planner_mod.CLEARANCE_MODES, default="ray",
                   help="teacher 净空与防抖模式：ray（默认基线）/ swept（hold=1.0）/ braking（速度感知刹停+转向滞回 v5）")
    p.add_argument("--dynamics-params", default=None,
                   help="动力学参数 JSON（默认 sim/dynamics/e1_mini.json；>30m 长航线传满电 e1_mini_soc100.json）")
    p.add_argument("--out", required=True)
    p.add_argument("--hold", type=float, default=0.5, help="每步动作窗口（秒）")
    p.add_argument("--stop-mode", choices=("explicit", "legacy-safe"), default="explicit",
                   help="explicit：独立 STOP 意图，执行一个 SAFE 窗口后判分；legacy-safe：仅复现 v2 连续三步 SAFE")
    p.add_argument("--split", action="append", default=None,
                   help="只取这些 SAGE split（train / test_Instruction_Unseen / test_Trajectory_Unseen / test_Scene_Unseen）")
    p.add_argument("--all-instructions", action="store_true", help="每条航线的每句指令都跑（缺省每条航线一句）")
    p.add_argument("--limit", type=int, default=0, help="每个场景最多跑几集（0 = 不限）")
    p.add_argument("--videos", type=int, default=0, help="每个场景前几集合成带指令横幅的四格视频")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--instruction", default=None,
                   help="运行一条自由输入指令；需要 --grounder vlm 和 reactive/student 策略")
    p.add_argument("--spawn", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   help="自由指令模式的起飞位置，使用 scene nav 坐标；缺省使用 scene 顶层 spawn")
    p.add_argument("--spawn-yaw-deg", type=float,
                   help="自由指令模式的机头朝向（nav 坐标系角度，+x 为 0°，逆时针为正）")
    p.add_argument("--manual-steps", type=int, default=None,
                   help="自由指令模式的飞行步数上限，缺省使用场景设置")
    args = p.parse_args(argv)
    use_grounder = args.grounder in ("lexical", "phrase")
    if args.policy == "student" and (not args.ckpt or not os.path.isfile(args.ckpt)):
        p.error("--policy student 必须提供存在的 --ckpt")
    if use_grounder and not args.catalog:
        p.error(f"--grounder {args.grounder} 必须同时提供 --catalog")
    if args.grounder in ("lexical", "phrase", "vlm", "blind_unknown") and args.policy == "forward":
        p.error(f"forward 不消费目标，不能与 --grounder {args.grounder} 一起使用")
    if args.grounder in ("vlm", "blind_unknown") and args.policy == "teacher":
        p.error(f"teacher 读取全局特权坐标，不能与零特权 --grounder {args.grounder} 混用")
    if args.teacher_clearance != "ray" and args.policy != "teacher" and not (args.policy == "student" and args.collect):
        p.error(f"--teacher-clearance {args.teacher_clearance} 仅适用于 --policy teacher 或 student --collect")
    if not (math.isfinite(args.hold) and args.hold > 0):
        p.error("--hold 必须是正数")
    if args.limit < 0 or args.videos < 0:
        p.error("--limit / --videos 不能为负数")
    if args.instruction is not None:
        if not args.instruction.strip():
            p.error("--instruction 不能为空")
        if len(args.scene) != 1:
            p.error("--instruction 自由演示模式只接受一个 --scene")
        if args.grounder != "vlm" or args.policy not in ("reactive", "student"):
            p.error("--instruction 需要 --grounder vlm 和 --policy reactive/student")
        if args.counterfactual != "correct" or args.collect or args.stop_mode != "explicit":
            p.error("--instruction 需要 correct 指令、评测模式和 --stop-mode explicit")
        if args.split or args.all_instructions or args.limit:
            p.error("--instruction 与 --split、--all-instructions、--limit 不兼容")
        if args.spawn is not None and not all(math.isfinite(value) for value in args.spawn):
            p.error("--spawn 的 X/Y/Z 必须是有限数")
        if args.spawn_yaw_deg is not None and not math.isfinite(args.spawn_yaw_deg):
            p.error("--spawn-yaw-deg 必须是有限数")
        if args.manual_steps is not None and args.manual_steps < 1:
            p.error("--manual-steps 必须大于 0")
    elif args.spawn is not None:
        p.error("--spawn 只能与 --instruction 一起使用")
    elif args.manual_steps is not None:
        p.error("--manual-steps 只能与 --instruction 一起使用")
    elif args.spawn_yaw_deg is not None:
        p.error("--spawn-yaw-deg 只能与 --instruction 一起使用")
    if os.path.isdir(args.out) and os.listdir(args.out):
        p.error(f"--out {args.out} 不是空目录")
    os.makedirs(args.out, exist_ok=True)
    params = CfSitlParams.from_json(args.dynamics_params)
    started = time.monotonic()
    rows, failures = [], []
    lexical_grounder = None
    if args.grounder == "lexical":
        lexical_grounder = grounder_mod.LexicalGrounder()
    elif args.grounder == "phrase":
        lexical_grounder = grounder_mod.RelationPhraseGrounder()
    catalogs = {}
    if lexical_grounder is not None:
        for catalog_path in args.catalog:
            with open(catalog_path, encoding="utf-8") as f:
                catalog_doc = json.load(f)
            scene_name = catalog_doc.get("scene") if isinstance(catalog_doc, dict) else None
            if not isinstance(scene_name, str) or not scene_name.strip():
                p.error(f"{catalog_path}: 地标目录缺少 scene")
            if scene_name in catalogs:
                p.error(f"重复的场景地标目录: {scene_name}")
            catalogs[scene_name] = catalog_path
        requested_names = {scene_mod.load_scene(path)["name"] for path in args.scene}
        unused = sorted(set(catalogs) - requested_names)
        if unused:
            p.error("未被 --scene 使用的 --catalog: " + ", ".join(unused))
    vlm_grounder = None
    student_proc = None
    ckpt_sha256 = None
    try:
        if args.policy == "student":
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
            import student as student_mod
            import nav_cond
            ckpt_sha256 = collect_3dgs._sha256(args.ckpt)
            student_proc = student_mod.StudentPolicy(args.ckpt, infer_bin=args.infer_bin)
            if args.grounder == "vlm" and student_proc.prev_dim != nav_cond.VLN_NAV_DIM:
                p.error("VLM student 模式需要 prev_dim=15 的 VLN_NAV checkpoint，当前权重不会消费 VLM 方位")
        if args.grounder == "vlm" and args.counterfactual != "zero":
            vlm_grounder = grounder_mod.VisualLanguageGrounder(model_path=args.vlm_model)
        for path in args.scene:
            scene_catalog, catalog_path = None, None
            if lexical_grounder is not None:
                scene_name = scene_mod.load_scene(path)["name"]
                catalog_path = catalogs.get(scene_name)
                if catalog_path is None:
                    p.error(f"场景 {scene_name} 缺少对应 --catalog 文件")
                _require_external_catalog(catalog_path, path)
                scene_catalog = grounder_mod.load_catalog(catalog_path, scene_name=scene_name, scene_path=path)
            r, f = run_scene(path, args, params, lexical_grounder=lexical_grounder,
                             catalog=scene_catalog, catalog_path=catalog_path,
                             vlm_grounder=vlm_grounder, student_proc=student_proc)
            rows += r
            failures += f
    finally:
        if student_proc is not None:
            student_proc.close()
        if vlm_grounder is not None:
            vlm_grounder.close()
    if not rows and not failures:
        p.error("没有选中任何 VLN 集，请检查场景、--split 和指令数据")
    with open(os.path.join(args.out, "episodes.jsonl"), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    vlm_lats = [r["vlm_latency_ms_p50"] for r in rows if r.get("vlm_latency_ms_p50") is not None]
    manual_demo = args.instruction is not None
    report = {"metric_version": METRIC_VERSION if args.stop_mode == "explicit" else "3dgs.vln.v2",
              "stage_metric_version": "sage.multi_stage.v1",
              "policy": args.policy, "ckpt": args.ckpt,
              "ckpt_sha256": ckpt_sha256,
              "seed": args.seed, "collect": bool(args.collect),
              "teacher_clearance": args.teacher_clearance,
              "counterfactual": args.counterfactual, "blind_rgb": bool(args.blind_rgb),
              "hold_s": args.hold, "stop_mode": args.stop_mode,
              "stop_streak": vln.STOP_STREAK if args.stop_mode == "legacy-safe" else None,
              "dynamics": "cf_sitl", "dynamics_profile": params.profile(),
              "scenes": args.scene, "split_filter": args.split, "all_instructions": args.all_instructions,
              "grounder": args.grounder, "g0_mode": args.g0_mode,
              "active_bearing_source": _active_bearing_source(args),
              "catalog": args.catalog,
              "vlm_latency_ms": ({"p50_median": collect_3dgs._percentile(vlm_lats, 50),
                                  "p50_max": max(vlm_lats)} if vlm_lats else None),
              "manual_demo": manual_demo,
              "overall_metric_scope": ("manual_unscored" if manual_demo else
                                       "route_waypoint_proxy" if use_grounder else "scene_route_waypoint"),
              "overall_goal_scope": "original_episode_goal",
              "overall": None if manual_demo else vln.summarize(rows, failures),
              "stage_metrics": None if manual_demo else vln.summarize_stages(rows, failures),
              "by_type": {} if manual_demo else vln.summarize_by(rows, "type", failures),
              "by_split": {} if manual_demo else vln.summarize_by(rows, "split", failures),
              "by_scene": {} if manual_demo else vln.summarize_by(rows, "scene", failures),
              "grounding": (_grounding_summary(rows, failures) if use_grounder and not manual_demo else None),
              "supplied_goal_metric_scope": "instruction_episode_waypoint",
              "supplied_goal_metrics": (None if manual_demo or args.counterfactual == "zero" else vln.summarize([
                  {"success": r["supplied_goal_success"],
                   "oracle_success": r["supplied_goal_oracle_success"],
                   "spl": r["supplied_goal_spl"], "ndtw": r["supplied_goal_ndtw"],
                   "sdtw": r["supplied_goal_sdtw"], "ne_m": r["supplied_goal_ne_m"],
                   "collision": r["collision"], "stopped": r["stopped"]} for r in rows
                  if "supplied_goal_success" in r],
                  failures)),
              "target_goal_metrics": (vln.summarize([
                  {"success": r["target_success"], "oracle_success": r["target_oracle_success"],
                   "spl": r["target_spl"], "ne_m": r["target_ne_m"],
                   "collision": r["collision"], "stopped": r["stopped"]} for r in rows
                  if "target_success" in r], failures) if use_grounder and not manual_demo else None),
              "failures": failures, "wall_s": round(time.monotonic() - started, 1)}
    with open(os.path.join(args.out, "vln_eval.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report["overall"], ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
