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
  time_pairing       每步 depth/pose 与 RGB 的 |gap| > 50 ms 说明观测不同步
  action_window      动作窗口 sim_dt 的变异系数       > 5% 说明 hold 随 RTF 漂移
  forward_motion     稳态 FORWARD_SLOW 步的位移       < 0.4×SPEED_FWD×dt 说明动作没生效
  transition_lag     偏航后第一步的位移中位数         < 50% 标称说明过渡步几乎不动
  yaw_direction      YAW 步的 dyaw 方向              需 actions 里有 dyaw_deg
                                                     反向偏航接续步跳过（惯性不可测）
  class_share        每类样本占比 / split 内缺席     < 5% 的类学不出来；
                                                      某 split 缺整类会让 recall 结构性为 0
  split_geometry     train/test 的 geometry_gid 交集 必须为 0
  label_recompute    用 pillars+动作前位姿复算标签    需 teacher=plan_action 且 actions
                                                     里有 pose_before；不一致率必须为 0
"""
import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_gen as sg  # noqa: E402
from instruction import parse_target_color  # noqa: E402
from split_manifest import SPLIT_SALT, geometry_gid  # noqa: E402
from teacher import TEACHER_VERSION, plan_action, plan_action_global  # noqa: E402

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

NAMES = ("SAFE_NEUTRAL", "FORWARD_SLOW", "YAW_LEFT", "YAW_RIGHT")
SPEED_FWD = 0.3              # 与 collect.py 一致
MAX_GAP_NS = 50_000_000
MAX_WINDOW_CV = 0.05
FWD_MIN_V = 0.4 * SPEED_FWD
MAX_FWD_FAIL_SHARE = 0.01
# T2 标定（2026-10-30，52 集 / 2080 步 / 164 个偏航接前进）：中位数 44%，
# 是速度控制器加速滞后，不是没动。旧阈值 50% 会把正常数据判死；20% 仍能抓
# 「几乎不动」的过渡步（stuck 级）。改阈值必须同步 HANDOFF §九。
MIN_TRANSITION_RATIO = 0.2
YAW_MIN_DEG = 0.5
MIN_CLASS_SHARE = 0.05
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
    target_color = waypoint = scene_spawn = None
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
            waypoint = scene.get("waypoint")
            scene_spawn = scene.get("spawn")
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
            gap_covered = depth_gap is not None and pose_gap is not None
            steps.append({
                "action": row["action_executed"],
                "prev_action": rows[i - 1]["action_executed"] if i else None,
                "collision": bool(row.get("collision")),
                "dt": dt,
                "disp": disp,
                "disp_approx": row.get("displacement_m") is None,
                "dyaw": row.get("dyaw_deg"),
                # depth 可以晚于 RGB（_pick_depth 按 |Δt| 取最近帧），有符号 gap
                # 会被恒正的 pose_gap 盖住，必须取绝对值。缺字段记 None，
                # 不能按 0 算 PASS（A1）。
                "gap": max(abs(depth_gap), abs(pose_gap)) if gap_covered else None,
                "gap_covered": gap_covered,
                "has_pose_before": "pose_before" in row,
                "pose_before": row.get("pose_before"),
                "yaw_before": row.get("yaw_before"),
                "action_expert": row.get("action_expert"),
            })
    except (KeyError, TypeError, IndexError) as e:
        # 旧格式/坏行不能让整个 audit 崩掉；按「这集读不出来」计 FAIL。
        return {"dir": ep_dir, "error": f"{type(e).__name__}: {e}"}

    wp_min = None
    if isinstance(waypoint, dict):
        ds = []
        for o in oracle:
            pg = o.get("pose_gt")
            if isinstance(pg, dict) and "x" in pg and "y" in pg:
                ds.append(((pg["x"] - waypoint["x"]) ** 2 +
                           (pg["y"] - waypoint["y"]) ** 2) ** 0.5)
        wp_min = min(ds) if ds else None

    return {
        "dir": ep_dir,
        "split": man.get("split"),
        "teacher": man.get("teacher"),
        "planner": man.get("planner"),
        "teacher_version": man.get("teacher_version"),
        "geometry_version": man.get("geometry_version"),
        "gid": gid,
        "pillars": pillars,
        "clearance": clearance,
        "instruction": instruction,
        "target_color": target_color,
        "waypoint": waypoint,
        "scene_spawn": scene_spawn,
        "waypoint_min_dist": wp_min,
        "validity": man.get("validity") or {},
        "num_steps": len(steps),
        "requested_steps": man.get("requested_steps"),
        "steps": steps,
        "error": None,
    }


def check(name, status, detail, **extra):
    return {"name": name, "status": status, "detail": detail, **extra}


def audit(facts, pre_safe_collisions=0, pre_safe_samples=(), pre_excluded=()):
    """把逐集事实折成检查项。status: PASS / FAIL / SKIP。

    pre_safe_collisions：在 --exclude-collision-episodes 排除之前统计到的
    SAFE+碰撞步数。排除后 audit 看不到它们，必须单独传入，否则硬约束恒 PASS。
    pre_excluded：被排除的整集 facts。版本混包/validity/split 泄漏必须对
    **排除前**的全部集成立，否则排除碰撞集会把这些检查一起洗成 PASS。
    """
    ok = [f for f in facts if not f.get("error")]
    bad = [f for f in facts if f.get("error")]
    gate = ok + [f for f in pre_excluded if not f.get("error")]
    steps = [s for f in ok for s in f["steps"]]
    checks = []

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

    # 1c) L1：指令目标色 / 近远平衡 / teacher 是否经过 waypoint
    l1 = [f for f in gate if isinstance(f.get("instruction"), str)
          and f["instruction"].strip()]
    if not l1:
        checks.append(check("target_color", "SKIP", "无 instruction（非 L1 数据）"))
        checks.append(check("landmark_balance", "SKIP", "无 instruction（非 L1 数据）"))
        checks.append(check("teacher_waypoint", "SKIP", "无 instruction（非 L1 数据）"))
    else:
        bad = []
        for f in l1:
            want = parse_target_color(f["instruction"])
            colors = {p.get("color") for p in (f.get("pillars") or [])}
            if want is None or want != f.get("target_color") or want not in colors:
                bad.append(f["dir"])
        checks.append(check("target_color", "FAIL" if bad else "PASS",
                            f"{len(bad)}/{len(l1)} 集指令目标色与 scene 不一致",
                            samples=bad[:5]))

        near = far = no_data = 0
        for f in l1:
            colored = [p for p in (f.get("pillars") or []) if p.get("color")]
            spawn = f.get("scene_spawn") or {}
            if len(colored) < 2 or "x" not in spawn or "y" not in spawn:
                no_data += 1
                continue
            nearest = min(colored, key=lambda p: (p["x"] - spawn["x"]) ** 2 +
                          (p["y"] - spawn["y"]) ** 2)
            if nearest.get("color") == f.get("target_color"):
                near += 1
            else:
                far += 1
        if near + far == 0:
            checks.append(check("landmark_balance", "SKIP",
                                f"无可计算集（缺数据 {no_data}）"))
        else:
            checks.append(check(
                "landmark_balance", "FAIL" if (near == 0 or far == 0) else "PASS",
                f"目标最近 {near} / 非最近 {far}，缺数据 {no_data}"))

        # safe 出生集走局部 plan_action，本来就不跟 waypoint；不纳入该项检查。
        l1w = [f for f in l1 if isinstance(f.get("waypoint"), dict)
               and (f.get("scene_spawn") or {}).get("mode") != "safe"]
        if not l1w:
            checks.append(check("teacher_waypoint", "SKIP", "无 waypoint"))
        else:
            miss = [f["dir"] for f in l1w if f.get("waypoint_min_dist") is None]
            badw = [f["dir"] for f in l1w if f.get("waypoint_min_dist") is not None
                    and f["waypoint_min_dist"] > 1.0]
            checks.append(check(
                "teacher_waypoint", "FAIL" if (badw or miss) else "PASS",
                f"{len(badw)} 集最近距离 >1.0 m，{len(miss)} 集无位姿",
                samples=(badw + miss)[:5]))

    # 2) 时间配对（缺 gap 字段的步单列；部分缺失不能让检查按 0 放过）
    covered = [s["gap"] for s in steps if s["gap"] is not None]
    uncovered = len(steps) - len(covered)
    if not covered:
        checks.append(check("time_pairing", "SKIP",
                            f"{len(steps)} 步都缺 depth/pose gap 字段，无法核验"))
    else:
        over = sum(1 for g in covered if g > MAX_GAP_NS)
        detail = (f"max_gap={max(covered)/1e6:.1f} ms，超 {MAX_GAP_NS/1e6:.0f} ms "
                  f"的步 {over}/{len(covered)}")
        if uncovered:
            detail += f"；{uncovered}/{len(steps)} 步缺 gap 字段，未覆盖"
        checks.append(check("time_pairing", "FAIL" if (over or uncovered) else "PASS",
                            detail))

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
        s["disp_approx"] for s in steps) else ""
    steady = [s for s in steps if s["action"] == 1 and s["prev_action"] == 1
              and s["disp"] is not None]
    if not steady:
        checks.append(check("forward_motion", "SKIP", "没有连续两个前进步，无法判定"))
    else:
        bad_fwd = [s for s in steady if s["disp"] < FWD_MIN_V * s["dt"]]
        share = len(bad_fwd) / len(steady)
        checks.append(check("forward_motion", "FAIL" if share > MAX_FWD_FAIL_SHARE else "PASS",
                            f"{len(bad_fwd)}/{len(steady)} 个稳态前进步位移不足 "
                            f"{FWD_MIN_V:.2f}×dt（{share:.1%}，阈值 {MAX_FWD_FAIL_SHARE:.0%}）"
                            f"{approx}"))

    # 4b) 偏航→前进的过渡步几乎不动
    trans = [s for s in steps if s["action"] == 1 and s["prev_action"] in (2, 3)
             and s["disp"] is not None and s["dt"] > 0]
    if not trans:
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
    yaw_all = [s for s in steps if s["action"] in (2, 3)]
    yaw = [s for s in yaw_all if s["dyaw"] is not None]
    def is_reversal(s):
        return (s["prev_action"], s["action"]) in ((3, 2), (2, 3))

    checked = [s for s in yaw if not is_reversal(s)]
    if not yaw_all:
        checks.append(check("yaw_direction", "SKIP", "没有偏航步，无法核验偏航方向"))
    elif not yaw:
        checks.append(check("yaw_direction", "SKIP",
                            "偏航步都缺 dyaw_deg（旧格式/全 null），无法离线核验"))
    elif not checked:
        # 「全部是反转步」和「没有 dyaw」是两件事，不能共用同一个误导性文案。
        checks.append(check("yaw_direction", "SKIP",
                            f"全部 {len(yaw)} 个偏航步都是反向接续步，"
                            "方向受惯性影响不可测，无法离线核验"))
    else:
        wrong = [s for s in checked
                 if (s["action"] == 2 and s["dyaw"] <= YAW_MIN_DEG)
                 or (s["action"] == 3 and s["dyaw"] >= -YAW_MIN_DEG)]
        checks.append(check("yaw_direction", "FAIL" if wrong else "PASS",
                            f"{len(wrong)}/{len(checked)} 个偏航步转向与标签相反或过小"
                            f"（跳过 {len(yaw) - len(checked)} 个反向接续步；"
                            f"另有 {len(yaw_all) - len(yaw)} 步 dyaw=null 未验）"))

    # 6) 类别分布（附 per-split 计数：T3 要确认 val/test 里 SAFE 有 support）
    counts = Counter(s["action"] for s in steps)
    total = len(steps)
    dist = {NAMES[k]: counts.get(k, 0) for k in range(4)}
    thin = [NAMES[k] for k in range(4) if counts.get(k, 0) / total < MIN_CLASS_SHARE]
    per_split = {}
    for f in ok:
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
    # 已排除的碰撞集也要计入，否则 --exclude-collision-episodes 会把这条检查洗成 PASS。
    safe_hits = [s for s in steps if s["action"] == 0 and s["collision"]]
    safe_total = sum(1 for s in steps if s["action"] == 0)
    if safe_hits or pre_safe_collisions:
        checks.append(check("safe_collision", "FAIL",
                            f"审计集 {len(safe_hits)}/{safe_total} 个 SAFE 步带 collision；"
                            f"另有已排除集 {pre_safe_collisions} 步；应为 0",
                            samples=list(pre_safe_samples)[:5]))
    elif safe_total == 0:
        checks.append(check("safe_collision", "SKIP", "审计集没有 SAFE 步，无法核验"))
    else:
        checks.append(check("safe_collision", "PASS",
                            f"0/{safe_total} 个 SAFE 步带 collision"))

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
    elif present != {SPLIT_SALT}:
        checks.append(check("geometry_version", "FAIL",
                            f"数据版本 {sorted(present)} != 当前 {SPLIT_SALT}"))
    else:
        # 版本字符串对还不够：改了 LANE_SEP/GATE_Y_ABS_MAX 等常量但忘 bump
        # GEOMETRY_VERSION 时，gid 全变、交集仍为 0，只有落盘常量能抓出来。
        clear = [f for f in gate if isinstance(f.get("clearance"), dict)]
        mism = []
        for f in clear:
            c = f["clearance"]
            for key, want in GEOM_CONST.items():
                got = c.get(key)
                if got != want:
                    mism.append(f"{os.path.basename(f['dir'])}:{key}={got!r}!={want!r}")
        if mism:
            checks.append(check("geometry_version", "FAIL",
                                f"{len(mism)} 处几何常量与当前 scene_gen 不一致："
                                f"{mism[:3]}"))
        else:
            detail = f"{SPLIT_SALT}；常量核对 {len(clear)}/{len(gate)} 集"
            if len(clear) < len(gate):
                detail += "（旧 scene.json 无 clearance）"
            checks.append(check("geometry_version", "PASS", detail))

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
        # 旧几何/旧教师版本数据没有版本字段；用当前算法复算会假 FAIL。
        if (f.get("geometry_version") != SPLIT_SALT or
                f.get("teacher_version") != TEACHER_VERSION):
            old_version += len(f["steps"])
            continue
        for s in f["steps"]:
            pb, yb = s["pose_before"], s["yaw_before"]
            if (not isinstance(pb, dict) or yb is None or
                    "x" not in pb or "y" not in pb):
                no_pose += 1
                continue
            total += 1
            # 普通集用 A*，safe 出生集用局部规划；manifest.planner 是唯一开关。
            if f.get("planner") == "astar":
                want = plan_action_global(pb["x"], pb["y"], yb, f["pillars"],
                                          waypoint=f.get("waypoint"))
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
                            f"缺柱子真值 {no_truth} 步、旧几何版本 {old_version} 步"
                            f"（共 {len(given)} 集）",
                            skipped=skipped))
    else:
        share = mism / total
        checks.append(check("label_recompute",
                            "FAIL" if share > MAX_LABEL_MISMATCH else "PASS",
                            f"{mism}/{total} 步复算标签与原标签不一致（{share:.1%}，"
                            f"阈值 {MAX_LABEL_MISMATCH:.0%}；跳过 {skipped} 步"
                            f"（缺位姿 {no_pose}、缺真值 {no_truth}、"
                            f"旧几何版本 {old_version}）；"
                            f"另有 {uncovered} 步来自非 plan_action 的集未参与）",
                            samples=samples, mismatch=round(share, 4),
                            covered_steps=total, skipped=skipped))

    return checks


def main(argv=None):
    p = argparse.ArgumentParser(description="数据集验收闸门")
    p.add_argument("--dataset", required=True, help="采集输出根目录（递归找 episode_manifest.json）")
    p.add_argument("--out", help="写 audit.json 的路径")
    p.add_argument("--exclude-collision-episodes", action="store_true",
                   help="排除含 collision 步的整集后再审计（T2 硬约束：碰撞步标签不可信）")
    args = p.parse_args(argv)

    if not os.path.isdir(args.dataset):
        p.error(f"--dataset 不是目录: {args.dataset}")
    eps = []
    for dirpath, _dirs, files in os.walk(args.dataset):
        # collect_parallel 把重试两次仍失败的集搬进 <out>/failed/；它们不是训练
        # 数据，混进全量闸门会让「全量过闸门」永远失败，也会污染类别统计。
        if os.sep + "failed" + os.sep in dirpath + os.sep:
            continue
        if "episode_manifest.json" in files:
            eps.append(dirpath)
    eps.sort()
    if not eps:
        p.error(f"{args.dataset} 下没有 episode_manifest.json")

    facts = [episode_facts(e) for e in eps]
    # 排除前先统计 SAFE+碰撞：排除后 audit 看不到这些步，硬约束会恒 PASS。
    pre_safe_collisions = sum(
        1 for f in facts if not f.get("error")
        for s in f["steps"] if s["action"] == 0 and s["collision"])
    pre_safe_samples = [f["dir"] for f in facts if not f.get("error") and any(
        s["action"] == 0 and s["collision"] for s in f["steps"])]
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
    checks = audit(facts, pre_safe_collisions, pre_safe_samples, excluded_facts)
    failed = [c for c in checks if c["status"] == "FAIL"]

    result = {
        "schema": "uavrgb.audit.v1",
        "dataset": os.path.abspath(args.dataset),
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
