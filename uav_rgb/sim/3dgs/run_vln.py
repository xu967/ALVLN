#!/usr/bin/env python3
"""3DGS 上的视觉语言导航评测：每集一句 SAGE-3D 指令，智能体自己喊停，按 SR / OSR / NE / SPL 出报告（口径见 vln.py）。

从仓库根、用 3dgs-venv 跑：
  python -m sim.3dgs.run_vln --policy teacher --out $D/exp-vln/teacher --videos 3 \
      --scene $D/3dgs-scenes/sage3d_839992/scene.json [--scene ...]
每条航线缺省只取一句指令（句式按航线轮换），--all-instructions 取全部。

策略：
  teacher  占据 A* 教师，直接拿航线终点当目标：等于已经听懂了指令（特权），是上界，也是以后蒸馏的老师
  reactive  目标条件局部控制，只看当前深度、位姿和目标方向，不在线查询全局占据图或调用 A*
  forward  一直前进、从不喊停，不看指令：下界
teacher/reactive 都直接拿 waypoint，只用于验证控制层；它们都不是语言理解结果。
真正的语言学生尚未接入。

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
from .dynamics import CfSitlParams
from .eval_3dgs import reach_radius_of, route_shortest_path_m

METRIC_VERSION = "3dgs.vln.v3"
STEPS_HOLD_S = 0.5      # 航线 max_steps 是导入时按 hold 0.5 估的（scene_import_sage3d），别的 hold 按时长换算


def make_policy(name, occ_planner, waypoint, hold, stop_radius=1.0):
    if name == "teacher":
        return planner_mod.OccAstarTeacher(occ_planner, waypoint, hold_s=hold)
    if name == "reactive":
        return reactive.ReactiveGoalPolicy(waypoint, stop_radius=stop_radius)
    return collect_3dgs.ForwardPolicy()


def result_text(m):
    """视频结尾大字；以 SUCCESS 开头画绿色，其余红色（view_layout.panel）。"""
    if m["success"]:
        return "SUCCESS stop %.2f m" % m["ne_m"]
    if m["collision"]:
        return "COLLISION"
    if m["stopped"]:
        return "WRONG STOP %.1f m" % m["ne_m"]
    return "NO STOP %.1f m" % m["ne_m"]


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_scene(path, args, params):
    scene = scene_mod.load_scene(path)
    eps = vln.episodes(scene, splits=args.split)
    if not args.all_instructions:
        eps = vln.one_per_route(eps, seed=args.seed)
    if args.limit:
        eps = eps[:args.limit]
    if not eps:
        print(f"{path}: 没有带指令的航线（不是 SAGE-3D 导入的场景，或 --split 过滤光了）", file=sys.stderr)
        return [], []
    identity = collect_3dgs.scene_identity(scene)
    gaussians = scene_mod.read_gaussian_ply(scene["ply_path"])
    occ = collision_mod.from_scene(scene, gaussians=gaussians)
    del gaussians
    renderer = renderer_mod.make_renderer(scene)
    occ_planner = planner_mod.OccPlanner(occ)
    radius = reach_radius_of(scene)
    rows, failures = [], []
    for n, ep in enumerate(eps):
        scene_mod.apply_route(scene, ep["route_index"])
        identity = collect_3dgs.scene_identity(scene, ply_sha256=identity["ply_sha256"])
        scene["vln"] = ep
        ep_dir = os.path.join(args.out, scene["name"], f"{ep['route_name']}_{ep['instruction_index']}")
        steps = math.ceil(int(scene["max_steps"]) * STEPS_HOLD_S / args.hold)
        policy = make_policy(args.policy, occ_planner, scene["waypoint"], args.hold, stop_radius=radius)
        try:
            man, err = collect_3dgs.run_episode(
                scene, ep_dir, renderer, occ, params, policy, steps, args.hold,
                split=identity["split"], seed=args.seed + n, episode_id=n, stop_on_collision=True,
                identity=identity, split_policy="diag-vln-eval", dynamics="cf_sitl",
                stop_on_request=args.stop_mode == "explicit",
                stop_after_safe=vln.STOP_STREAK if args.stop_mode == "legacy-safe" else None)
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
            start = {**actions[0]["pose_before"], "yaw_deg": actions[0].get("yaw_before", 0.0)}
            shortest = route_shortest_path_m(occ_planner, [{"pose_gt": start}] + oracle, scene)
            m = vln.episode_metrics(actions, oracle, scene["waypoint"], radius, man["stopped_by_agent"], shortest)
        except Exception as e:  # noqa: BLE001 - 一集崩了记下来接着跑，报告里单列
            failures.append({**ep, "scene": scene["name"], "dir": ep_dir, "error": f"{type(e).__name__}: {e}"})
            continue
        row = {**ep, "scene": scene["name"], **m, "error": err, "dir": ep_dir,
               "terminal_reason": man.get("terminal_reason")}
        rows.append(row)
        print(f"[{scene['name']} {n + 1}/{len(eps)}] {result_text(m):24s} {ep['type']:22s} {ep['text']}", file=sys.stderr)
        if n < args.videos:
            video_mod.compose_views_video(
                ep_dir, scene, renderer, params.camera_mount, occ,
                title=f"{args.policy} | cf_sitl | {scene['name']} | {ep['type']}", result=result_text(m),
                instruction=f"\"{ep['text']}\"   goal: {ep['goal_object']}",
                log=lambda msg: print(msg, file=sys.stderr))
    return rows, failures


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", action="append", required=True, help="SAGE-3D 导入的 scene.json，可给多个")
    p.add_argument("--policy", choices=("teacher", "reactive", "forward"), required=True,
                   help="teacher = A* 特权上界；reactive = 不查全局地图的目标条件局部控制；forward = 盲目前进下界")
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
    args = p.parse_args(argv)
    if not (math.isfinite(args.hold) and args.hold > 0):
        p.error("--hold 必须是正数")
    if args.limit < 0 or args.videos < 0:
        p.error("--limit / --videos 不能为负数")
    if os.path.isdir(args.out) and os.listdir(args.out):
        p.error(f"--out {args.out} 不是空目录")
    os.makedirs(args.out, exist_ok=True)
    params = CfSitlParams.from_json(None)
    started = time.monotonic()
    rows, failures = [], []
    for path in args.scene:
        r, f = run_scene(path, args, params)
        rows += r
        failures += f
    if not rows and not failures:
        p.error("没有选中任何 VLN 集，请检查场景、--split 和指令数据")
    with open(os.path.join(args.out, "episodes.jsonl"), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"metric_version": METRIC_VERSION if args.stop_mode == "explicit" else "3dgs.vln.v2",
              "policy": args.policy, "hold_s": args.hold, "stop_mode": args.stop_mode,
              "stop_streak": vln.STOP_STREAK if args.stop_mode == "legacy-safe" else None,
              "dynamics": "cf_sitl", "dynamics_profile": params.profile(),
              "scenes": args.scene, "split_filter": args.split, "all_instructions": args.all_instructions,
              "overall": vln.summarize(rows, failures), "by_type": vln.summarize_by(rows, "type", failures),
              "by_split": vln.summarize_by(rows, "split", failures), "by_scene": vln.summarize_by(rows, "scene", failures),
              "failures": failures, "wall_s": round(time.monotonic() - started, 1)}
    with open(os.path.join(args.out, "vln_eval.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report["overall"], ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
