#!/usr/bin/env python3
"""数据集验收闸门：逐条检查采集产物的不变量，跑不过就非零退出。

一次采集是否正确，不靠肉眼看图，靠这几条能机械判定的不变量。每条检查
显式给出「覆盖 / 未覆盖」，缺字段的检查标 SKIP 并说明要补什么，不假装通过。

用法：
  python3 sim/tools/audit_dataset.py --dataset /tmp/ds200
  python3 sim/tools/audit_dataset.py --dataset /tmp/ds200 --out /tmp/audit.json

判定口径（阈值都在下面常量里，改阈值要同步改本文件顶部的说明）：

  manifest_validity  每集 validity 三项全 true      训练集必须是逐集跑满的
  instruction_coverage 每集 instruction 非空/无坐标 旧数据无字段 SKIP；部分缺 FAIL
  time_pairing       每步 pose 与 RGB 的 |gap| > 50 ms、depth > 100 ms
                     说明观测不同步（阈值与 collect.py 对齐；等于阈值算通过）
  window_semantics   每集 manifest.window_from       必须全包唯一；混用
                                                     dispatch/observation 结论不可比
  action_window      动作窗口 sim_dt 的变异系数       > 5% 说明 hold 随 RTF 漂移
  forward_motion     稳态 FORWARD_SLOW 步的位移       < 0.4×SPEED_FWD×dt 说明动作没生效
  transition_lag     偏航后第一步的位移中位数         < 50% 标称说明过渡步几乎不动
  yaw_direction      YAW 步的 dyaw 方向              需 actions 里有 dyaw_deg
                                                     反向偏航接续步跳过（惯性不可测）
  class_share        每类样本占比 / split 内缺席     < 5% 的类学不出来；
                                                      某 split 缺整类会让 recall 结构性为 0
  waypoint_side      waypoint 必须在目标柱外侧        目标柱按 relation 映射：
                                                     left → 右侧柱，right → 左侧柱；
                                                     waypoint 横向偏移必须 > 0.9 m
  provenance_uniform teacher_version 有值必须统一    task_version 全缺视为同一代
  split_geometry     train/test 的 geometry_gid 交集 必须为 0
  label_recompute    用 pillars+动作前位姿复算标签    需 teacher=plan_action 且 actions
                                                     里有 pose_before；不一致率必须为 0

campus.v1 用校园专用检查替代旧 waypoint_side/GEOM_CONST：地图拓扑组归属、
scene/manifest/生成器身份、目标区净空与 A* 可达、彩柱视野/遮挡、真实轨迹到达
（campus_teacher_reach）与整集无碰撞（campus_no_collision）分开判定。
可见性为几何分析，不代替 Gazebo 渲染验收；不声称地面圆盘始终在前视相机中。

反事实集（manifest.counterfactual_of 非空）复用原集 RGB/oracle/轨迹，只重写
action 与 waypoint；teacher_waypoint/teacher_stop/forward_motion/transition_lag/
yaw_direction/class_share 判 SKIP。label_recompute 照常执行。
"""
import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import landmarks as lm  # noqa: E402
import scene_gen as sg  # noqa: E402
import scene_forest as sf  # noqa: E402
import campus  # noqa: E402
from instruction import parse_target_color  # noqa: E402
from split_manifest import SPLIT_SALT, geometry_gid  # noqa: E402
from teacher import (TEACHER_VERSION, plan_action, plan_action_global_tiebreak,
                     astar_path, _point_blocked, set_vehicle,
                     teacher_version, task_of_scene)  # noqa: E402

# scene.json 里落盘的全局几何常量；与当前 scene_gen 不一致说明改了几何但没
# bump GEOMETRY_VERSION，gid 会全变、别的闸门查不出新旧混包（A2）。
GEOM_CONST = {
    "min_gap_required_m": sg.MIN_GAP,
    "corridor_y": list(sg.Y_RANGE),
    "pillar_x_range": list(sg.PILLAR_X_RANGE),
    "pillar_y_range": list(sg.PILLAR_Y_RANGE),
    "lane_sep_m": sg.LANE_SEP,
    "lane_gap_m": round(sg.LANE_SEP - sg.PILLAR_SIZE, 4),
    "pillar_count_range": list(sg.PILLAR_COUNT_RANGE),
    "gate_y_abs_max_m": sg.GATE_Y_ABS_MAX,
    "spawn_clear_before_x_m": sg.SPAWN_CLEAR_X,
    "pillar_height_m": sg.PILLAR_HEIGHT,
    "cruise_height_m": sg.CRUISE_HEIGHT,
}


def forest_geom_const(density):
    """forest_e1 scene.json clearance 里落盘的几何常量（按档位）；与 GEOM_CONST 同一用途。"""
    return {
        "min_gap_required_m": sf.DENSITY_GAP_M[density],
        "trunk_count_range": list(sf.DENSITY_COUNT[density]),
        "forest_x": list(sf.FOREST_X),
        "corridor_y": list(sf.FOREST_Y),
        "trunk_radius_m": list(sf.TRUNK_RADIUS_M),
        "trunk_height_m": sf.TRUNK_HEIGHT_M,
        "spawn_clear_before_x_m": sf.FOREST_X[0],
    }


# label_recompute 可复算的几何版本：生成器仍是当前版本，pillars 真值与采集时同源。
RECOMPUTE_GEOMETRY_VERSIONS = (SPLIT_SALT, campus.VERSION, sf.GEOMETRY_VERSION_FOREST)

NAMES = ("SAFE_NEUTRAL", "FORWARD_SLOW", "YAW_LEFT", "YAW_RIGHT")
SPEED_FWD = 0.3              # 与 collect.py 一致
MAX_POSE_GAP_NS = 50_000_000
# 与 collect.py 的 DEPTH_STALE_NS 一致：那边用 `>` 判超限，等于阈值算通过。
MAX_DEPTH_GAP_NS = 100_000_000
MAX_WINDOW_CV = 0.05
FWD_MIN_V = 0.4 * SPEED_FWD
MAX_FWD_FAIL_SHARE = 0.01
# 没有固定前进速度的动力学档（dynamics_profile 前缀）：定速判据 4/4b 跳过，偏航起始步免验。
NON_FIXED_SPEED_PREFIXES = ("cf_stab", "cf_sitl")
# T2 标定（2026-10-30，52 集 / 2080 步 / 164 个偏航接前进）：中位数 44%，
# 是速度控制器加速滞后，不是没动。旧阈值 50% 会把正常数据判死；20% 仍能抓
# 「几乎不动」的过渡步（stuck 级）。改阈值必须同步 HANDOFF §九。
MIN_TRANSITION_RATIO = 0.2
YAW_MIN_DEG = 0.5
MIN_CLASS_SHARE = 0.05
WAYPOINT_MIN_LATERAL = 0.9   # v4 waypoint 在目标柱外侧 standoff 0.8 以上
MAX_LABEL_MISMATCH = 0.0     # plan_action 是确定性的，同一输入必然同一标签
# 指令里出现坐标/带单位小数说明把 oracle 泄漏进了文本（L0/L1）
COORD_RE = re.compile(r"(?:\bx\s*=|\by\s*=|\(\s*-?\d|\d+\.\d+\s*m\b)", re.IGNORECASE)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dist3(a, b):
    return sum((a[k] - b[k]) ** 2 for k in ("x", "y", "z")) ** 0.5


def episode_facts(ep_dir):
    """读一集，返回逐集事实；缺文件时返回 {"error": 原因}。"""
    try:
        man = load_json(os.path.join(ep_dir, "episode_manifest.json"))
        rows = load_jsonl(os.path.join(ep_dir, man["actions_ref"]))
        oracle = load_jsonl(os.path.join(ep_dir, man["oracle_ref"]))
    except (OSError, KeyError, ValueError) as e:
        return {"dir": ep_dir, "error": f"{type(e).__name__}: {e}"}

    instruction = man.get("instruction")
    if instruction is not None and not isinstance(instruction, str):
        return {"dir": ep_dir, "error": "instruction 必须是字符串"}

    # 行数不一致时 zip 会静默截短：被截掉的尾步（例如 SAFE+碰撞）永远不会进
    # 任何检查，整个闸门会给出假 PASS。必须先于 zip 拒绝。
    if len(rows) != len(oracle):
        return {"dir": ep_dir,
                "error": f"actions {len(rows)} 行 != oracle {len(oracle)} 行"}
    declared = man.get("num_steps")
    if type(declared) is int and declared != len(rows):
        return {"dir": ep_dir,
                "error": f"manifest num_steps={declared} != actions {len(rows)} 行"}

    gid = None
    pillars = None
    clearance = None
    target_color = relation = waypoint = scene_spawn = stop_radius = None
    task_version = None
    scene = {}
    scene_path = os.path.join(ep_dir, "scene.json")
    if os.path.exists(scene_path):
        try:
            scene = load_json(scene_path)
            raw = scene["pillars"]
            # pillars 写成 null 时 geometry_gid(None) 会抛 TypeError，整个 audit
            # 直接崩；isinstance 挡住这种「文件在但内容是空的」。
            if isinstance(raw, list) and raw:
                pillars = raw
                gid = geometry_gid(pillars)
            if isinstance(scene.get("clearance"), dict):
                clearance = scene["clearance"]
            target_color = scene.get("target_color")
            relation = scene.get("relation")
            waypoint = scene.get("waypoint")
            scene_spawn = scene.get("spawn")
            stop_radius = scene.get("stop_radius")
            task_version = scene.get("task_version")
        except (KeyError, OSError, ValueError, TypeError):
            pillars = None

    steps = []
    try:
        for i, (row, orc) in enumerate(zip(rows, oracle)):
            # 行数相同也可能 step 错位（手工合并/重排）；不能只按行号硬配。
            if (type(row.get("step")) is int and type(orc.get("step")) is int
                    and row["step"] != orc["step"]):
                return {"dir": ep_dir,
                        "error": f"actions step[{i}]={row['step']} != oracle step {orc['step']}"}
            dt = (orc["action_end_ns"] - orc["action_start_ns"]) / 1e9
            disp = row.get("displacement_m")
            if disp is None and i > 0 and oracle[i - 1].get("pose_gt") and orc.get("pose_gt"):
                # 旧数据没记 displacement_m：用相邻动作后位姿差近似（误差 = 一个位姿周期）
                disp = _dist3(orc["pose_gt"], oracle[i - 1]["pose_gt"])
            depth_gap = row.get("depth_rgb_gap_ns")
            pose_gap = row.get("pose_rgb_gap_ns")
            steps.append({
                "action": row["action_executed"],
                "prev_action": rows[i - 1]["action_executed"] if i else None,
                "collision": bool(row.get("collision")),
                "dt": dt,
                "disp": disp,
                "disp_approx": row.get("displacement_m") is None,
                "dyaw": row.get("dyaw_deg"),
                # depth 可以晚于 RGB（_pick_depth 按 |Δt| 取最近帧），有符号 gap
                # 必须取绝对值。缺字段记 None，不能按 0 算 PASS（A1）。
                "depth_gap": abs(depth_gap) if depth_gap is not None else None,
                "pose_gap": abs(pose_gap) if pose_gap is not None else None,
                "has_pose_before": "pose_before" in row,
                "pose_before": row.get("pose_before"),
                "yaw_before": row.get("yaw_before"),
                "action_expert": row.get("action_expert"),
            })
    except (KeyError, TypeError, IndexError) as e:
        # 旧格式/坏行不能让整个 audit 崩掉；按「这集读不出来」计 FAIL。
        return {"dir": ep_dir, "error": f"{type(e).__name__}: {e}"}

    wp_min = None
    final_wp = None
    if isinstance(waypoint, dict):
        ds = []
        for o in oracle:
            pg = o.get("pose_gt")
            if isinstance(pg, dict) and "x" in pg and "y" in pg:
                ds.append(((pg["x"] - waypoint["x"]) ** 2 +
                           (pg["y"] - waypoint["y"]) ** 2) ** 0.5)
        wp_min = min(ds) if ds else None
        final_wp = ds[-1] if ds else None
    # 逐步的动作后真值 x（缺位姿记 None）：episode_length 用它核对 --stop-at-goal 的截断位置。
    oracle_x = [(o.get("pose_gt") or {}).get("x") if isinstance(o.get("pose_gt"), dict) else None
                for o in oracle]

    return {
        "dir": ep_dir,
        "manifest": man,
        "scene_metadata": scene,
        "split": man.get("split"),
        "teacher": man.get("teacher"),
        "planner": man.get("planner"),
        "teacher_version": man.get("teacher_version"),
        "geometry_version": man.get("geometry_version"),
        # make_counterfactual.py 写进 manifest 的标记；非空说明标签是重算的。
        "counterfactual_of": man.get("counterfactual_of"),
        "dynamics_profile": man.get("dynamics_profile"),
        "vehicle": man.get("vehicle") or "x3",   # 旧数据没写 vehicle，一律 X3
        # 动作窗口语义：collect.py 只写进 episode_manifest.json，actions.jsonl 行里没有。
        "window_from": man.get("window_from"),
        "gid": gid,
        "pillars": pillars,
        "clearance": clearance,
        "instruction": instruction,
        "target_color": target_color,
        "relation": relation,
        "waypoint": waypoint,
        "scene_spawn": scene_spawn,
        "task_version": task_version,
        "stop_radius": stop_radius,
        "stopped_on_goal": man.get("stopped_on_goal"),
        "stopped_on_collision": man.get("stopped_on_collision"),
        # collect.py --stop-at-goal 写的成功线 x（None = 没开）；--tiebreak 的口径；DAgger 执行者。
        "goal_line_x": man.get("goal_line_x"),
        "teacher_tiebreak": man.get("teacher_tiebreak"),
        "executed_by": man.get("executed_by"),
        "oracle_x": oracle_x,
        "waypoint_min_dist": wp_min,
        "final_waypoint_dist": final_wp,
        "last_action": steps[-1]["action"] if steps else None,
        "validity": man.get("validity") or {},
        "num_steps": len(steps),
        "requested_steps": man.get("requested_steps"),
        "steps": steps,
        "error": None,
    }


def check(name, status, detail, **extra):
    return {"name": name, "status": status, "detail": detail, **extra}


def _is_campus(f):
    return (f.get("geometry_version") == campus.VERSION
            or f.get("task_version") == campus.VERSION
            or f.get("scene_metadata", {}).get("geometry_version") == campus.VERSION)


def campus_checks(facts, quarantined=()):
    """校园新任务用自己的地图/目标合同；旧柱阵的外侧绕行规则不适用。

    quarantined：已隔离整集的目录。到达/碰撞是轨迹属性，隔离集单独记账，
    不参与这两条的 FAIL 判定。
    """
    qdirs = set(quarantined)
    failures = {name: [] for name in ("campus_identity", "campus_map_split",
                                    "campus_goal_geometry", "campus_teacher_reach",
                                    "campus_no_collision")}
    map_splits = {}
    for f in facts:
        scene, man = f.get("scene_metadata", {}), f.get("manifest", {})
        identity_checked = False
        try:
            seed = scene["seed"]
            if type(seed) is not int or seed < 0:
                raise ValueError("seed 必须是非负整数")
            family, boxes, color, relation, wp = campus.layout(seed)
            expected = {"geometry_version": campus.VERSION, "task_version": campus.VERSION,
                        "map_id": campus.map_id(seed), "family": family,
                        "split_policy": "campus_map_v1", "goal_radius": campus.RADIUS,
                        "stop_radius": campus.RADIUS,
                        "success_criterion": "visible_goal_reach_v1",
                        "waypoint": wp, "target_color": color, "relation": relation,
                        "instruction": campus.instruction(color, relation)}
            wrong = [k for k, v in expected.items() if scene.get(k) != v or man.get(k) != v]
            if scene.get("pillars") != boxes or man.get("geometry_gid") != geometry_gid(boxes):
                wrong.append("pillars/geometry_gid")
            with open(os.path.join(f["dir"], "scene.json"), "rb") as sf:
                scene_digest = hashlib.sha256(sf.read()).hexdigest()
            if man.get("scene_meta_sha256") != scene_digest:
                wrong.append("scene_meta_sha256")
            if (man.get("teacher") != "plan_action" or man.get("planner") != "astar"
                    or man.get("teacher_version") != TEACHER_VERSION):
                wrong.append("teacher/planner/version")
            if any(not isinstance(s.get("pose_before"), dict)
                   or any(k not in s["pose_before"] for k in ("x", "y"))
                   or s.get("yaw_before") is None
                   or s.get("action_expert") not in range(4) for s in f["steps"]):
                wrong.append("label_recompute inputs")
            if wrong:
                failures["campus_identity"].append(f"{f['dir']}: {', '.join(wrong)}")
            identity_checked = True
            identity = scene.get("map_id")
            map_splits.setdefault(identity, set()).add(f.get("split"))
            if (identity != campus.map_id(seed) or f.get("split") != campus.split(seed)
                    or man.get("map_id") != identity):
                failures["campus_map_split"].append(f["dir"])
            # 另做几何验证，不能只相信 clearance.ok 或生成器版本号。
            spawn = scene["spawn"]
            expected_spawn = dict(x=-.5, y=0., z=.15, yaw_deg=0., mode="default")
            if spawn != expected_spawn:
                raise ValueError("spawn 与 campus.v1 不一致")
            pillars = scene["pillars"]
            start = (spawn["x"], spawn["y"])
            target = next(p for p in pillars if p.get("color") == color
                          and (lm.lateral(p, spawn) > 0) == (relation == "left"))
            if (_point_blocked(*start, pillars, .356)
                    or _point_blocked(wp["x"], wp["y"], pillars, campus.RADIUS + .356)
                    or not astar_path(pillars, start, (wp["x"], wp["y"]))):
                raise ValueError("出生点/目标区净空不足或无 A* 路径")
            if math.hypot(wp["x"] - start[0], wp["y"] - start[1]) <= campus.RADIUS:
                raise ValueError("出生点已在目标区")
            # 检查出生视野与目标区中心朝 +X 的巡航切片可见性；不宣称地面圆盘可见。
            for pose in (spawn, dict(wp, yaw_deg=0.)):
                if abs(lm.h_angle(target, pose)) > lm.HALF_FOV_DEG - lm.FOV_MARGIN_DEG:
                    raise ValueError("目标彩柱超出相机水平视野")
                cx, cy = pose["x"] + lm.CAMERA_FORWARD_M, pose["y"]
                distance = math.hypot(target["x"] - cx, target["y"] - cy)
                others = [p for p in pillars if p is not target
                          and p["z"] - p["size"][2] / 2 <= sg.CRUISE_HEIGHT
                          <= p["z"] + p["size"][2] / 2]
                for i in range(1, max(2, int(distance / .02))):
                    t = i / max(2, int(distance / .02))
                    x, y = cx + t * (target["x"] - cx), cy + t * (target["y"] - cy)
                    if any(abs(x - p["x"]) < p["size"][0] / 2
                           and abs(y - p["y"]) < p["size"][1] / 2 for p in others):
                        raise ValueError("目标彩柱被其他碰撞体遮挡")
        except (KeyError, OSError, TypeError, ValueError, StopIteration) as e:
            failures["campus_goal_geometry"].append(f"{f['dir']}: {e}")
            if not identity_checked:
                failures["campus_identity"].append(f"{f['dir']}: 身份未完整核验: {e}")
        if f["dir"] not in qdirs:
            # 「到过目标区」与「整集无碰撞」是两件事，分开判定，避免到点但蹭到
            # island 的集把到达语义一起判死。
            if (f.get("counterfactual_of") is not None
                    or f.get("waypoint_min_dist") is None
                    or f["waypoint_min_dist"] > campus.RADIUS):
                failures["campus_teacher_reach"].append(f["dir"])
            if any(s["collision"] for s in f["steps"]):
                failures["campus_no_collision"].append(f["dir"])
    for identity, splits in map_splits.items():
        if len(splits) > 1:
            failures["campus_map_split"].append(f"{identity}: {sorted(splits)}")
    checks = []
    for name, bad in failures.items():
        detail = f"{len(bad)} 个问题 / {len(facts)} 集；" + {
            "campus_identity": "scene、manifest、可复算教师与确定性布局一致",
            "campus_map_split": "地图组归属与 split 一致（含排除集）",
            "campus_goal_geometry": "净空、A*可达、彩柱水平视野与巡航切片无遮挡",
            "campus_teacher_reach": "到过 0.5m 目标区（waypoint_min_dist ≤ goal_radius）",
            "campus_no_collision": "整集轨迹无 collision",
        }[name]
        extra = {}
        if qdirs and name in ("campus_teacher_reach", "campus_no_collision"):
            detail += f"；隔离 {len(qdirs)} 集不参与"
            extra["quarantined"] = len(qdirs)
        checks.append(check(name, "FAIL" if bad else "PASS", detail,
                            samples=bad[:5], **extra))
    return checks


def audit(facts, pre_excluded=(), quarantined=()):
    """把逐集事实折成检查项。status: PASS / FAIL / SKIP。

    pre_excluded：被排除的整集 facts。版本混包/validity/split 泄漏必须对
    **排除前**的全部集成立，否则排除碰撞集会把这些检查一起洗成 PASS。
    quarantined：其中被隔离的碰撞集。它们的 SAFE+碰撞步单独记入
    safe_collision 的 quarantined 计数，不参与 FAIL 判定，也不重复计入审计集。
    """
    ok = [f for f in facts if not f.get("error")]
    bad = [f for f in facts if f.get("error")]
    gate = ok + [f for f in pre_excluded if not f.get("error")]
    steps = [s for f in ok for s in f["steps"]]
    # 反事实集只重写 action/waypoint，轨迹仍是原集的：轨迹类检查对它必然假 FAIL。
    cf_ok = [f for f in ok if f.get("counterfactual_of") is not None]
    norm_ok = [f for f in ok if f.get("counterfactual_of") is None]
    traj_steps = [s for f in norm_ok for s in f["steps"]]
    CF_SKIP = ("反事实集复用原集 RGB/oracle/轨迹，只重写 action 与 waypoint，"
               "动作类检查不适用")
    checks = []
    campus_facts = [f for f in gate if _is_campus(f)]
    quarantine_dirs = [f["dir"] for f in quarantined if not f.get("error")]
    q_safe_steps = [s for f in quarantined if not f.get("error")
                    for s in f["steps"] if s["action"] == 0 and s["collision"]]

    if bad:
        checks.append(check("read_episodes", "FAIL",
                            f"{len(bad)} 集读不出来", samples=[b["dir"] for b in bad[:5]]))
    if not steps:
        checks.append(check("read_episodes", "FAIL", "没有任何步"))
        return checks

    # 1) manifest validity（对排除前的全部集成立）
    bad_val = [f["dir"] for f in gate if not all(f["validity"].get(k) for k in
               ("camera_verified", "physics_verified", "episode_executed"))]
    if not any(f["validity"] for f in gate):
        checks.append(check("manifest_validity", "SKIP", "manifest 里没有 validity 字段"))
    else:
        checks.append(check("manifest_validity", "FAIL" if bad_val else "PASS",
                            f"{len(bad_val)}/{len(gate)} 集 validity 不全 true"
                            f"（含已排除 {len(pre_excluded)} 集）",
                            samples=bad_val[:5]))

    # 1b) 指令覆盖（L0）：全有 → 非空/无坐标；部分缺 → FAIL；全无 → SKIP
    instr = [f for f in gate if isinstance(f.get("instruction"), str)]
    missing_instr = len(gate) - len(instr)
    nonempty = [f for f in instr if f["instruction"].strip()]
    leaks = [f["dir"] for f in nonempty if COORD_RE.search(f["instruction"])]
    if not instr:
        checks.append(check("instruction_coverage", "SKIP",
                            "全部 manifest 无 instruction（旧格式）"))
    elif missing_instr or len(nonempty) != len(instr):
        checks.append(check("instruction_coverage", "FAIL",
                            f"{missing_instr} 集缺 instruction，"
                            f"{len(instr) - len(nonempty)} 集为空"))
    elif leaks:
        checks.append(check("instruction_coverage", "FAIL",
                            f"{len(leaks)} 条指令含坐标/数值", samples=leaks[:5]))
    else:
        checks.append(check("instruction_coverage", "PASS",
                            f"{len(nonempty)}/{len(gate)} 集有非空指令，无坐标泄漏"))

    # 1c) L1：指令目标色/关系词 / 左右平衡 / teacher 是否经过 waypoint
    l1 = [f for f in gate if isinstance(f.get("instruction"), str)
          and f["instruction"].strip() and not _is_campus(f)]
    if campus_facts:
        checks.extend(campus_checks(campus_facts, quarantine_dirs))
    if not l1:
        reason = ("campus 使用 campus_identity/goal_geometry/teacher_reach 检查" if campus_facts
                  else "无 instruction（非 L1 数据）")
        checks.append(check("target_color", "SKIP", reason))
        checks.append(check("relation_balance", "SKIP", reason))
        checks.append(check("waypoint_side", "SKIP", reason))
        checks.append(check("teacher_waypoint", "SKIP", reason))
    else:
        bad = []
        for f in l1:
            want = parse_target_color(f["instruction"])
            # 同色柱对是任务定义的一部分：恰好 2 根彩色柱且同色，颜色就是目标色。
            colored = [p for p in (f.get("pillars") or []) if p.get("color")]
            colors = {p.get("color") for p in colored}
            same_pair = len(colored) == 2 and len(colors) == 1
            words = set(f["instruction"].lower().split())
            if (want is None or not same_pair or want not in colors
                    or want != f.get("target_color")
                    or f.get("relation") not in ("left", "right")
                    or f["relation"] not in words):
                bad.append(f["dir"])
        checks.append(check("target_color", "FAIL" if bad else "PASS",
                            f"{len(bad)}/{len(l1)} 集指令与 scene 的目标色/同色柱对/"
                            f"关系词不一致", samples=bad[:5]))

        # 同色柱对：两根彩色柱必须分居出生点左右，且 left/right 两类都要有。
        # 旧版查的「目标最近/最远」在同色下失效——两根都是目标色，恒为最近。
        left_n = right_n = no_data = 0
        bad_pair = []
        for f in l1:
            colored = [p for p in (f.get("pillars") or []) if p.get("color")]
            spawn = f.get("scene_spawn") or {}
            rel = f.get("relation")
            if (len(colored) != 2 or "x" not in spawn or "y" not in spawn
                    or rel not in ("left", "right")):
                no_data += 1
                continue
            lat = sorted(lm.lateral(p, spawn) for p in colored)
            if lat[0] > -lm.LATERAL_MIN or lat[1] < lm.LATERAL_MIN:
                bad_pair.append(f["dir"])
            elif rel == "left":
                left_n += 1
            else:
                right_n += 1
        if left_n + right_n == 0:
            checks.append(check("relation_balance", "SKIP",
                                f"无可计算集（缺数据 {no_data}）"))
        else:
            checks.append(check(
                "relation_balance",
                "FAIL" if (bad_pair or left_n == 0 or right_n == 0) else "PASS",
                f"left {left_n} / right {right_n}，两柱未分居左右 {len(bad_pair)} 集，"
                f"缺数据 {no_data}", samples=bad_pair[:5]))

        # waypoint 必须在目标柱外侧。目标柱按 assign_landmarks 的映射复算：
        # relation=left → 目标在右侧，relation=right → 目标在左侧。
        bad_side = []
        side_no_data = 0
        for f in l1:
            colored = [p for p in (f.get("pillars") or []) if p.get("color")]
            spawn, wp = f.get("scene_spawn"), f.get("waypoint")
            rel = f.get("relation")
            if (len(colored) != 2 or not isinstance(spawn, dict)
                    or "x" not in spawn or "y" not in spawn
                    or not isinstance(wp, dict) or "x" not in wp or "y" not in wp
                    or rel not in ("left", "right")):
                side_no_data += 1
                continue
            # lateral>0 为左，目标柱在 relation 的相反侧。
            hits = [p for p in colored
                    if (lm.lateral(p, spawn) > 0) == (rel == "right")]
            if len(hits) != 1:
                bad_side.append(f["dir"])
                continue
            target = hits[0]
            lat_t, lat_w = lm.lateral(target, spawn), lm.lateral(wp, spawn)
            if (wp["x"] != target["x"] or (lat_w > 0) != (lat_t > 0)
                    or abs(lat_w) <= WAYPOINT_MIN_LATERAL):
                bad_side.append(f["dir"])
        if not bad_side and side_no_data == len(l1):
            checks.append(check("waypoint_side", "SKIP",
                                f"无可计算集（缺数据 {side_no_data}）"))
        else:
            checks.append(check(
                "waypoint_side", "FAIL" if bad_side else "PASS",
                f"{len(bad_side)}/{len(l1)} 集 waypoint 不在目标柱外侧，"
                f"缺数据 {side_no_data}", samples=bad_side[:5]))

        # 反事实集的 oracle 轨迹沿用原集，teacher_waypoint/teacher_stop 必然假 FAIL。
        l1n = [f for f in l1 if f.get("counterfactual_of") is None]
        cf_note = (f"；另有 {len(l1) - len(l1n)} 集为反事实集未参与"
                   if len(l1n) < len(l1) else "")
        if not l1n:
            checks.append(check("teacher_waypoint", "SKIP",
                                f"{CF_SKIP}（{len(l1)} 集）"))
            checks.append(check("teacher_stop", "SKIP",
                                f"{CF_SKIP}（{len(l1)} 集）"))
        else:
            l1w = [f for f in l1n if isinstance(f.get("waypoint"), dict)]
            if not l1w:
                checks.append(check("teacher_waypoint", "SKIP", "无 waypoint"))
            else:
                miss = [f["dir"] for f in l1w if f.get("waypoint_min_dist") is None]
                badw = [f["dir"] for f in l1w if f.get("waypoint_min_dist") is not None
                        and f["waypoint_min_dist"] > 1.0]
                checks.append(check(
                    "teacher_waypoint", "FAIL" if (badw or miss) else "PASS",
                    f"{len(badw)} 集最近距离 >1.0 m，{len(miss)} 集无位姿{cf_note}",
                    samples=(badw + miss)[:5]))

            # goal-stop（v2）：teacher 必须停在 waypoint 半径内，且末动作是 SAFE。
            # 旧 route-choice 数据没有 stop_radius，SKIP，不把历史数据改判。
            l1s = [f for f in l1n if f.get("stop_radius") is not None
                   and f.get("planner") == "astar"]
            if not l1s:
                # 有 stop_radius 却没有任何可核验的 astar 集（例如全是 safe 出生集
                # 或 planner 字段缺失）时不能 SKIP：那等于门禁对这类数据整体失效。
                has_stop = any(f.get("stop_radius") is not None for f in l1n)
                checks.append(check(
                    "teacher_stop", "FAIL" if has_stop else "SKIP",
                    "有 stop_radius 但没有 planner=astar 的集可核验" if has_stop
                    else "无 stop_radius（旧 route-choice 数据）"))
            else:
                bad = [f["dir"] for f in l1s
                       if f.get("final_waypoint_dist") is None
                       or f["final_waypoint_dist"] > float(f["stop_radius"]) + 0.2
                       or f.get("last_action") != 0]
                checks.append(check(
                    "teacher_stop", "FAIL" if bad else "PASS",
                    f"{len(bad)}/{len(l1s)} 集未停在 waypoint 半径内或末动作不是 SAFE"
                    f"{cf_note}", samples=bad[:5]))

    # 2) 时间配对（缺 gap 字段的步单列；部分缺失不能让检查按 0 放过）
    covered = [s for s in steps
               if s["depth_gap"] is not None and s["pose_gap"] is not None]
    uncovered = len(steps) - len(covered)
    if not covered:
        checks.append(check("time_pairing", "SKIP",
                            f"{len(steps)} 步都缺 depth/pose gap 字段，无法核验"))
    else:
        over_depth = sum(1 for s in covered if s["depth_gap"] > MAX_DEPTH_GAP_NS)
        over_pose = sum(1 for s in covered if s["pose_gap"] > MAX_POSE_GAP_NS)
        detail = (f"max_depth_gap={max(s['depth_gap'] for s in covered)/1e6:.1f} ms，"
                  f"超 {MAX_DEPTH_GAP_NS/1e6:.0f} ms 的步 {over_depth}/{len(covered)}；"
                  f"max_pose_gap={max(s['pose_gap'] for s in covered)/1e6:.1f} ms，"
                  f"超 {MAX_POSE_GAP_NS/1e6:.0f} ms 的步 {over_pose}/{len(covered)}")
        if uncovered:
            detail += f"；{uncovered}/{len(steps)} 步缺 gap 字段，未覆盖"
        checks.append(check(
            "time_pairing",
            "FAIL" if (over_depth or over_pose or uncovered) else "PASS", detail))

    # 3) 动作窗口稳定
    dts = [s["dt"] for s in steps if s["dt"] > 0]
    if not dts:
        checks.append(check("action_window", "FAIL", "没有有效的 sim_dt"))
    else:
        mean = statistics.fmean(dts)
        cv = statistics.pstdev(dts) / mean if mean else 0.0
        checks.append(check("action_window", "FAIL" if cv > MAX_WINDOW_CV else "PASS",
                            f"sim_dt mean={mean:.3f}s min={min(dts):.3f} max={max(dts):.3f} cv={cv:.1%}",
                            mean_s=round(mean, 4), min_s=round(min(dts), 4),
                            max_s=round(max(dts), 4), cv=round(cv, 4)))

    # 4) 前进动作是否真的动。刚做完偏航的那一步有加速滞后，单独看，否则
    #    两群混在一起会得出「13% 动作没生效」的误判。
    approx = "（位移由位姿差近似，窗口错位 ±60 ms）" if any(
        s["disp_approx"] for s in traj_steps) else ""
    # 4/4b 按 x3 定速 SPEED_FWD 判定；cf_stab 是倾角加速+滑行、cf_sitl 是固件速度环
    # （起步/刹停曲线由固件定），都没有固定速度，不适用。
    cf_stab = [f for f in norm_ok
               if str(f.get("dynamics_profile") or "").startswith(NON_FIXED_SPEED_PREFIXES)]
    stab_ids = {id(f) for f in cf_stab}
    fixed_v = [s for f in norm_ok if id(f) not in stab_ids for s in f["steps"]]
    STAB_SKIP = (f"cf_stab/cf_sitl 动力学没有固定前进速度，定速判据不适用（{len(cf_stab)} 集）")
    steady = [s for s in fixed_v if s["action"] == 1 and s["prev_action"] == 1
              and s["disp"] is not None]
    if cf_stab and not fixed_v:
        checks.append(check("forward_motion", "SKIP", STAB_SKIP))
    elif cf_ok and not norm_ok:
        checks.append(check("forward_motion", "SKIP",
                            f"{CF_SKIP}（{len(cf_ok)} 集）"))
    elif not steady:
        checks.append(check("forward_motion", "SKIP", "没有连续两个前进步，无法判定"))
    else:
        bad_fwd = [s for s in steady if s["disp"] < FWD_MIN_V * s["dt"]]
        share = len(bad_fwd) / len(steady)
        checks.append(check("forward_motion", "FAIL" if share > MAX_FWD_FAIL_SHARE else "PASS",
                            f"{len(bad_fwd)}/{len(steady)} 个稳态前进步位移不足 "
                            f"{FWD_MIN_V:.2f}×dt（{share:.1%}，阈值 {MAX_FWD_FAIL_SHARE:.0%}）"
                            f"{approx}"))

    # 4b) 偏航→前进的过渡步几乎不动
    trans = [s for s in fixed_v if s["action"] == 1 and s["prev_action"] in (2, 3)
             and s["disp"] is not None and s["dt"] > 0]
    if cf_stab and not fixed_v:
        checks.append(check("transition_lag", "SKIP", STAB_SKIP))
    elif cf_ok and not norm_ok:
        checks.append(check("transition_lag", "SKIP",
                            f"{CF_SKIP}（{len(cf_ok)} 集）"))
    elif not trans:
        checks.append(check("transition_lag", "SKIP", "没有偏航接前进的过渡步"))
    else:
        ratios = sorted(s["disp"] / (SPEED_FWD * s["dt"]) for s in trans)
        med = ratios[len(ratios) // 2]
        checks.append(check("transition_lag", "FAIL" if med < MIN_TRANSITION_RATIO else "PASS",
                            f"偏航接前进的 {len(trans)} 步，位移中位数为标称的 {med:.0%}"
                            f"（阈值 {MIN_TRANSITION_RATIO:.0%}）{approx}",
                            median_ratio=round(med, 3), n=len(trans)))

    # 5) 偏航方向（逐步行容错：非首步 dyaw=null 不能让整个 audit 崩）
    # 反向偏航接续步：上一步是反向 YAW，窗口前半段要先抵消上一步的角速度，
    # 净 dyaw 反映不了新指令（T5 实测 17 个反转步里 3 个 |dyaw|<0.5°，且
    # 非反转步 0 异常）；按物理不可测跳过，但计入 detail。
    yaw_all = [s for s in traj_steps if s["action"] in (2, 3)]
    yaw = [s for s in yaw_all if s["dyaw"] is not None]
    def is_reversal(s):
        return (s["prev_action"], s["action"]) in ((3, 2), (2, 3))

    # cf_stab：0.15 s 指令延迟 + X3 偏航跟踪滞后，偏航头两步的 0.2 s 窗口里机体还没转起来
    # （2026-11-04 实测 14/14 个异常步都是头两步）；只核验同向连续第 3 步起。
    onset = {id(st) for f in cf_stab for i, st in enumerate(f["steps"])
             if st["action"] in (2, 3) and not (
                 i >= 2 and f["steps"][i - 1]["action"] == st["action"]
                 == f["steps"][i - 2]["action"])}
    checked = [s for s in yaw if not is_reversal(s) and id(s) not in onset]
    onset_note = f"；cf_stab 偏航起始步 {len(onset)} 个未验" if onset else ""
    if cf_ok and not norm_ok:
        checks.append(check("yaw_direction", "SKIP",
                            f"{CF_SKIP}（{len(cf_ok)} 集）"))
    elif not yaw_all:
        checks.append(check("yaw_direction", "SKIP", "没有偏航步，无法核验偏航方向"))
    elif not yaw:
        checks.append(check("yaw_direction", "SKIP",
                            "偏航步都缺 dyaw_deg（旧格式/全 null），无法离线核验"))
    elif not checked:
        # 「全部是反转步」和「没有 dyaw」是两件事，不能共用同一个误导性文案。
        checks.append(check("yaw_direction", "SKIP",
                            f"全部 {len(yaw)} 个偏航步都是反向接续步或 cf_stab 起始步，"
                            "方向受惯性影响不可测，无法离线核验"))
    else:
        wrong = [s for s in checked
                 if (s["action"] == 2 and s["dyaw"] <= YAW_MIN_DEG)
                 or (s["action"] == 3 and s["dyaw"] >= -YAW_MIN_DEG)]
        checks.append(check("yaw_direction", "FAIL" if wrong else "PASS",
                            f"{len(wrong)}/{len(checked)} 个偏航步转向与标签相反或过小"
                            f"（跳过 {sum(map(is_reversal, yaw))} 个反向接续步；"
                            f"另有 {len(yaw_all) - len(yaw)} 步 dyaw=null 未验{onset_note}）"))

    # 6) 类别分布（附 per-split 计数：T3 要确认 val/test 里 SAFE 有 support）
    if cf_ok and not norm_ok:
        checks.append(check("class_share", "SKIP",
                            f"{CF_SKIP}（{len(cf_ok)} 集）"))
    else:
        counts = Counter(s["action"] for s in traj_steps)
        total = len(traj_steps)
        dist = {NAMES[k]: counts.get(k, 0) for k in range(4)}
        thin = [NAMES[k] for k in range(4)
                if counts.get(k, 0) / total < MIN_CLASS_SHARE]
        per_split = {}
        for f in norm_ok:
            c = Counter(s["action"] for s in f["steps"])
            d = per_split.setdefault(f["split"], {n: 0 for n in NAMES})
            for k in range(4):
                d[NAMES[k]] += c.get(k, 0)
        # 全局占比达标不够：某个 split 里整类缺席时，该 split 的 recall 结构性为 0，
        # T3 的「四类 recall 全 >0」会静默变成只看 train/val 的结论。
        empty_split = [f"{sp}:{n}" for sp, d in per_split.items()
                       if sp in ("train", "val", "test")
                       for n, c in d.items() if c == 0]
        checks.append(check("class_share",
                            "FAIL" if (thin or empty_split) else "PASS",
                            f"每类占比低于 {MIN_CLASS_SHARE:.0%}：{thin or '无'}；"
                            f"split 内整类缺席：{empty_split or '无'}",
                            distribution=dist, per_split=per_split))

    # 6b) SAFE 步必须无碰撞：碰撞步的「停」标签不可信（T2 硬约束）。
    # 审计集里的命中才判 FAIL；已隔离（quarantine）的集单独记账，不重复计入。
    safe_hits = [s for s in steps if s["action"] == 0 and s["collision"]]
    safe_total = sum(1 for s in steps if s["action"] == 0)
    q_note = (f"；隔离集 {len(q_safe_steps)} 步单独记账，不参与判定"
              if q_safe_steps else "")
    q_extra = {"quarantined": {"episodes": len(quarantine_dirs),
                               "safe_collision_steps": len(q_safe_steps)}}
    if safe_hits:
        checks.append(check("safe_collision", "FAIL",
                            f"审计集 {len(safe_hits)}/{safe_total} 个 SAFE 步带 collision；"
                            f"应为 0{q_note}",
                            samples=[f["dir"] for f in ok
                                     if any(s["action"] == 0 and s["collision"]
                                            for s in f["steps"])][:5],
                            **q_extra))
    elif safe_total == 0:
        checks.append(check("safe_collision", "SKIP", "审计集没有 SAFE 步，无法核验",
                            **q_extra))
    else:
        checks.append(check("safe_collision", "PASS",
                            f"0/{safe_total} 个 SAFE 步带 collision{q_note}", **q_extra))

    # 6c) 几何版本一致性：v2/v3 混包时 gid 交集查不出来。
    #     必须含被排除的集，否则「排除碰撞集」会把旧版本集一起洗掉。
    all_versions = [f.get("geometry_version") for f in gate]
    present = {v for v in all_versions if v is not None}
    missing = sum(1 for v in all_versions if v is None)
    if not present:
        checks.append(check("geometry_version", "SKIP",
                            "全部 manifest 都没有 geometry_version（旧格式）"))
    elif missing:
        checks.append(check("geometry_version", "FAIL",
                            f"{missing}/{len(all_versions)} 集缺 geometry_version，"
                            f"与 {sorted(present)} 混在一起；gid 交集查不出这种混包"))
    elif len(present) > 1:
        checks.append(check("geometry_version", "FAIL",
                            f"混用多个几何版本：{sorted(present)}"))
    elif present == {campus.VERSION}:
        mismatch = next(c for c in checks if c["name"] == "campus_identity")
        checks.append(check("geometry_version", mismatch["status"],
                            "campus.v1 使用确定性地图布局合同；详见 campus_identity"))
    elif present not in ({SPLIT_SALT}, {sf.GEOMETRY_VERSION_FOREST}):
        checks.append(check("geometry_version", "FAIL",
                            f"数据版本 {sorted(present)} != 当前 {SPLIT_SALT} / {sf.GEOMETRY_VERSION_FOREST}"))
    else:
        # 版本字符串对还不够：改了 LANE_SEP/GATE_Y_ABS_MAX 等常量但忘 bump
        # GEOMETRY_VERSION 时，gid 全变、交集仍为 0，只有落盘常量能抓出来。
        # forest_e1 同理，常量按 scene.json 的 density 档取（缺 density 记一处不一致）。
        version = next(iter(present))
        clear = [f for f in gate if isinstance(f.get("clearance"), dict)]
        mism = []
        for f in clear:
            c = f["clearance"]
            if version == sf.GEOMETRY_VERSION_FOREST:
                density = f.get("scene_metadata", {}).get("density")
                if density not in sf.DENSITY_GAP_M:
                    mism.append(f"{os.path.basename(f['dir'])}:density={density!r}")
                    continue
                const = forest_geom_const(density)
            else:
                const = GEOM_CONST
            for key, want in const.items():
                got = c.get(key)
                if got != want:
                    mism.append(f"{os.path.basename(f['dir'])}:{key}={got!r}!={want!r}")
        if mism:
            checks.append(check("geometry_version", "FAIL",
                                f"{len(mism)} 处几何常量与当前生成器不一致："
                                f"{mism[:3]}"))
        else:
            detail = f"{version}；常量核对 {len(clear)}/{len(gate)} 集"
            if len(clear) < len(gate):
                detail += "（旧 scene.json 无 clearance）"
            checks.append(check("geometry_version", "PASS", detail))

    # 6d) 采集口径一致性：teacher/task 版本混包时，label_recompute 只跳过非当前
    #     teacher_version，门禁看起来全绿，但 SR/成功判据在包里已经有两种定义。
    #     两字段分开判定：task_version 是后加的，全缺（base 任务）视为同一代。
    t_vers = [f.get("teacher_version") for f in gate]
    k_vers = [f.get("task_version") for f in gate]
    t_have = {v for v in t_vers if v is not None}
    k_have = {v for v in k_vers if v is not None}
    t_missing = sum(1 for v in t_vers if v is None)
    k_missing = sum(1 for v in k_vers if v is None)
    problems = []
    if len(t_have) > 1:
        problems.append(f"teacher_version 混用 {sorted(t_have)}")
    if len(k_have) > 1:
        problems.append(f"task_version 混用 {sorted(k_have)}")
    if t_have and t_missing:
        problems.append(f"{t_missing}/{len(t_vers)} 集缺 teacher_version")
    if k_have and k_missing:
        problems.append(f"{k_missing}/{len(k_vers)} 集缺 task_version")
    if problems:
        checks.append(check("provenance_uniform", "FAIL", "；".join(problems)))
    elif not t_have and not k_have:
        checks.append(check("provenance_uniform", "SKIP",
                            "全部集缺 teacher_version/task_version（旧格式，无法判定）"))
    else:
        detail = [f"teacher_version={sorted(t_have)[0]}" if t_have
                  else "teacher_version 全部缺失"]
        detail.append(f"task_version={sorted(k_have)[0]}" if k_have
                      else "task_version 全部缺失（视为同一代）")
        checks.append(check("provenance_uniform", "PASS", "；".join(detail)))

    # 6e) 动作窗口语义：dispatch 从派发时刻起算，observation 从 RGB 观测时刻起算，
    #     差约 60 ms 仿真时间（同一模型闭环 SR 0.90 vs 0.55）。混包则结论失效。
    #     旧数据在字段引入前采集：全缺 SKIP（无法判定），既不判 PASS 也不判 FAIL。
    sem = [f.get("window_from") for f in gate]
    bad_sem = [f["dir"] for f in gate if f.get("window_from") is not None
               and not isinstance(f.get("window_from"), str)]
    present = {v for v in sem if isinstance(v, str)}
    missing = sum(1 for v in sem if v is None)
    if bad_sem:
        checks.append(check("window_semantics", "FAIL",
                            f"{len(bad_sem)} 集 window_from 不是字符串，无法判定",
                            samples=bad_sem[:5]))
    elif not present:
        checks.append(check("window_semantics", "SKIP",
                            "全部集无 window_from（旧数据无此字段，无法判定）"))
    elif len(present) > 1:
        counts = Counter("missing" if v is None else v for v in sem)
        checks.append(check("window_semantics", "FAIL",
                            f"混用多个动作窗口语义：{dict(sorted(counts.items()))}；"
                            "dispatch/observation 差约 60 ms 仿真时间，结论不可比",
                            samples=[f"{os.path.basename(f['dir'])}:{f.get('window_from')}"
                                     for f in gate][:5]))
    elif missing:
        checks.append(check("window_semantics", "FAIL",
                            f"{missing}/{len(sem)} 集缺 window_from，与 "
                            f"{next(iter(present))} 混在一起，无法确认全包同一口径",
                            samples=[f["dir"] for f in gate if f.get("window_from") is None][:5]))
    else:
        checks.append(check("window_semantics", "PASS",
                            f"{len(sem)}/{len(sem)} 集 window_from={next(iter(present))}"))

    # 7) split 几何隔离（含被排除的集：排除碰撞集不能把跨 split 泄漏洗掉）
    by_split = {}
    for f in gate:
        if f["gid"]:
            by_split.setdefault(f["split"], set()).add(f["gid"])
    if len(by_split) < 2:
        checks.append(check("split_geometry", "SKIP", "少于两个 split 且有 gid，无法核验"))
    else:
        leaks = []
        names = sorted(by_split)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                common = by_split[a] & by_split[b]
                if common:
                    leaks.append(f"{a}∩{b}={len(common)}")
        checks.append(check("split_geometry", "FAIL" if leaks else "PASS",
                            "; ".join(leaks) if leaks else
                            "交集为 0（" + ", ".join(f"{n}={len(g)}" for n, g in by_split.items()) + "）"))

    # 8) 标签复算：拿采集时记下的动作前位姿 + 场景柱子真值，重跑一遍 plan_action。
    # 只对 teacher=plan_action 的集做：深度专家和盲策略本来就不由柱子真值决定，
    # 拿它复算等于换了个教师，比出来的差异没有意义。
    given = [f for f in ok if f.get("teacher") == "plan_action"]
    others = Counter(f.get("teacher") for f in ok if f.get("teacher") != "plan_action")
    # 逐集/逐步统计覆盖：混了新旧格式时要能看出「验了多少」，不能因为一集是
    # 旧格式就把整条检查降级成 SKIP，也不能把没进 given 的集算作已验。
    no_pose = no_truth = old_version = 0  # 因缺字段/版本跳过的步数
    uncovered = sum(len(f["steps"]) for f in ok if f.get("teacher") != "plan_action")
    total = mism = 0
    samples = []
    for f in given:
        if not f["pillars"]:
            no_truth += len(f["steps"])
            continue
        # 旧几何/旧教师版本数据没有版本字段；用当前算法复算会假 FAIL。版本按任务档位比：
        # forest_e1 集要求 forest 档版本（v6），旧森林集（写 v5、按 0.40/0.30 方框算）自动跳过。
        teacher_task = task_of_scene(f.get("scene_metadata"))
        if (f.get("geometry_version") not in RECOMPUTE_GEOMETRY_VERSIONS or
                f.get("teacher_version") != teacher_version(teacher_task)):
            old_version += len(f["steps"])
            continue
        # 脱困段的机身方框与规划余量按采集时的 (载具, 任务) 算：E1 数据用 X3 半宽复算会在 0.36 m
        # 余量内判出不同标签（pilot 20 集实测 47/800 假不一致，按 e1_mini 复算为 0）
        set_vehicle(f["vehicle"], teacher_task)
        for s in f["steps"]:
            pb, yb = s["pose_before"], s["yaw_before"]
            if (not isinstance(pb, dict) or yb is None or
                    "x" not in pb or "y" not in pb):
                no_pose += 1
                continue
            total += 1
            # 普通集用 A*，safe 出生集用局部规划；manifest.planner 是唯一开关。
            # 近平局口径按 manifest.teacher_tiebreak（旧数据没有 = legacy，与旧 plan_action_global 逐位相同）：
            # visible 集的标签多了「按可见柱定左右」，用 legacy 复算会把 DISTILL 量到的 ~0.7% 帧报成不一致。
            if f.get("planner") == "astar":
                want, _margin = plan_action_global_tiebreak(
                    pb["x"], pb["y"], yb, f["pillars"],
                    tiebreak=f.get("teacher_tiebreak") or "legacy",
                    waypoint=f.get("waypoint"),
                    stop_at_waypoint=bool(f.get("stop_radius")),
                    stop_radius=f.get("stop_radius") or 0.5)
            else:
                want = plan_action(pb["x"], pb["y"], yb, f["pillars"])
            if want != s["action_expert"]:
                mism += 1
                if len(samples) < 5:
                    samples.append({"dir": f["dir"], "recorded": s["action_expert"],
                                    "recomputed": want})
    skipped = no_pose + no_truth + old_version
    if not ok:
        checks.append(check("label_recompute", "SKIP", "没有可读的集"))
    elif not given:
        checks.append(check("label_recompute", "SKIP",
                            f"没有 teacher=plan_action 的集（其它：{dict(others)}），"
                            "标签不由柱子真值决定，无法复算"))
    elif total == 0:
        # 「一步都没复算成」不等于「全错」。旧格式/缺真值走这里，别报成 100% 不一致。
        checks.append(check("label_recompute", "SKIP",
                            f"没有可复算的步：缺动作前位姿 {no_pose} 步、"
                            f"缺柱子真值 {no_truth} 步、旧几何或教师版本 {old_version} 步"
                            f"（共 {len(given)} 集）",
                            skipped=skipped))
    else:
        share = mism / total
        checks.append(check("label_recompute",
                            "FAIL" if share > MAX_LABEL_MISMATCH else "PASS",
                            f"{mism}/{total} 步复算标签与原标签不一致（{share:.1%}，"
                            f"阈值 {MAX_LABEL_MISMATCH:.0%}；跳过 {skipped} 步"
                            f"（缺位姿 {no_pose}、缺真值 {no_truth}、"
                            f"旧几何或教师版本 {old_version}）；"
                            f"另有 {uncovered} 步来自非 plan_action 的集未参与）",
                            samples=samples, mismatch=round(share, 4),
                            covered_steps=total, skipped=skipped))

    # 9) 集长度：短于 requested_steps 的集必须有 manifest 解释，不能悄悄混进训练集。
    #    --stop-at-goal（DISTILL §3.1）记 stopped_on_goal + goal_line_x，且末步动作后真值 x 必须是**第一次**
    #    到成功线（早一步已过线却没停、或没过线就停，都说明采集端判定坏了）；开了开关却过线不停同样 FAIL。
    #    waypoint 任务的 goal-stop 记 stopped_on_goal + stop_radius（停稳与否由 teacher_stop 查）；
    #    --stop-on-collision 记 stopped_on_collision。旧 manifest 没有 requested_steps 的集不参与。
    short = [f for f in gate if type(f.get("requested_steps")) is int
             and f["num_steps"] < f["requested_steps"]]
    goal_line = [f for f in gate if f.get("goal_line_x") is not None]
    n_line = n_wp = n_coll = 0
    unexplained, bad_line = [], []
    for f in short:
        if f.get("stopped_on_goal") and f.get("goal_line_x") is not None:
            n_line += 1
        elif f.get("stopped_on_goal") and f.get("stop_radius") is not None:
            n_wp += 1
        elif f.get("stopped_on_collision"):
            n_coll += 1
        else:
            unexplained.append(f["dir"])
    for f in goal_line:
        gx = float(f["goal_line_x"])
        crossed = [x is not None and x >= gx for x in f["oracle_x"]]
        if f.get("stopped_on_goal"):
            if not crossed or not crossed[-1] or any(crossed[:-1]):
                bad_line.append(f["dir"])
        elif any(crossed):
            bad_line.append(f["dir"])
    if not short and not goal_line:
        checks.append(check("episode_length", "PASS",
                            f"{len(gate)} 集都跑满 requested_steps（或旧 manifest 无该字段），无 --stop-at-goal 集"))
    else:
        checks.append(check(
            "episode_length", "FAIL" if (unexplained or bad_line) else "PASS",
            f"{len(short)}/{len(gate)} 集短于 requested_steps：过成功线提前结束 {n_line}、"
            f"waypoint goal-stop {n_wp}、碰撞终止 {n_coll}、无解释 {len(unexplained)}；"
            f"--stop-at-goal 集 {len(goal_line)} 中截断位置与成功线不符 {len(bad_line)}",
            samples=(unexplained + bad_line)[:5], short=len(short), goal_line=len(goal_line)))

    return checks


# 不属于采集数据的目录：评测 .work、并行采集 staging、失败集、模板/日志。
IGNORED_DIRS = {".work", ".staging", "failed", "worlds", "logs"}


def collection_dirs(root):
    """返回 (episode 目录列表, 扫描来源)。

    有 collection_manifest.json 时只审计它登记的 (split, scene_id)：否则在实验根
    目录里跑 audit 会把 eval 的 .work/ 当采集数据，统计被稀释或直接 FAIL。
    没有 manifest 时退回 os.walk，并剪掉非采集目录。
    """
    manifest = os.path.join(root, "collection_manifest.json")
    if os.path.isfile(manifest):
        with open(manifest, encoding="utf-8") as f:
            data = json.load(f)
        dirs = []
        for split, scene_ids in (data.get("splits") or {}).items():
            for sid in scene_ids:
                d = os.path.join(root, split, str(sid))
                if os.path.isdir(d):
                    dirs.append(d)
        if dirs:
            return sorted(dirs), "collection_manifest"
    dirs = []
    for dirpath, subdirs, files in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        parts = [] if rel == "." else rel.split(os.sep)
        if any(part in IGNORED_DIRS for part in parts):
            subdirs[:] = []
            continue
        if "episode_manifest.json" in files:
            dirs.append(dirpath)
    return sorted(dirs), "walk"


def main(argv=None):
    p = argparse.ArgumentParser(description="数据集验收闸门")
    p.add_argument("--dataset", required=True, help="采集输出根目录（递归找 episode_manifest.json）")
    p.add_argument("--out", help="写 audit.json 的路径")
    p.add_argument("--exclude-collision-episodes", action="store_true",
                   help="排除含 collision 步的整集后再审计（T2 硬约束：碰撞步标签不可信）")
    args = p.parse_args(argv)

    if not os.path.isdir(args.dataset):
        p.error(f"--dataset 不是目录: {args.dataset}")
    eps, scan_source = collection_dirs(args.dataset)
    if not eps:
        p.error(f"{args.dataset} 下没有 episode_manifest.json")

    facts = [episode_facts(e) for e in eps]
    excluded = []
    excluded_facts = []
    if args.exclude_collision_episodes:
        kept = []
        for f in facts:
            if not f.get("error") and any(s["collision"] for s in f["steps"]):
                excluded.append(f["dir"])
                excluded_facts.append(f)
            else:
                kept.append(f)
        facts = kept
    # 隔离集单独传给 quarantined，只记账、不参与 FAIL；同时仍作为 pre_excluded
    # 参与版本/validity/几何隔离等必须对排除前全集成立的检查。
    checks = audit(facts, pre_excluded=excluded_facts, quarantined=excluded_facts)
    failed = [c for c in checks if c["status"] == "FAIL"]

    result = {
        "schema": "uavrgb.audit.v1",
        "dataset": os.path.abspath(args.dataset),
        "scan_source": scan_source,
        "episodes_found": len(eps),
        # 闸门实际审计的集数；episodes_found 含被 --exclude-collision-episodes 排除的集。
        "episodes": len(facts),
        "steps": sum(f.get("num_steps", 0) for f in facts if not f.get("error")),
        "passed": not failed,
        # 被硬约束排除的整集：含碰撞步，标签不可信；不参与闸门，单独记账。
        "excluded_collision_episodes": len(excluded),
        "excluded_dirs": excluded[:20],
        "checks": checks,
    }

    print(f"数据集: {result['dataset']}")
    print(f"集数={result['episodes']}（发现 {result['episodes_found']}）"
          f" 步数={result['steps']}"
          + (f"（排除含碰撞整集 {len(excluded)} 个）" if excluded else ""))
    for c in checks:
        print(f"  [{c['status']:4}] {c['name']:18} {c['detail']}")
    print("结论: " + ("PASS" if not failed else f"FAIL（{len(failed)} 项）"))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        print(f"已写 {args.out}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
