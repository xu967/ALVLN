#!/usr/bin/env python3
"""E6 闭环评测：学生在未知场景里自己飞，记录 SR / 碰撞率 / 误前向 / 推理延时。

每个 episode：生成固定 seed 的世界 -> 只保留指定 split 的几何 -> 起独立
GZ_PARTITION 的 gz sim -> collect.py --policy student --stop-on-collision ->
解析 actions.jsonl / oracle.jsonl。评测场景按几何组隔离，不会和训练几何重复。

用法（需要 .venv-tools 的 PIL；student 还要 uavrgb_infer 已构建）：
  python3 sim/tools/eval_closed_loop.py --policy student --ckpt <ckpt.bin> \
      --episodes 3 --out /tmp/eval1 --split test
  python3 sim/tools/eval_closed_loop.py --policy forward --episodes 3 \
      --out /tmp/eval-blind --split test
  # teacher-nn 用训练好的特权教师闭环（必须用带 torch 的解释器跑，见 --teacher-ckpt）
  $DATA_ROOT/uav_rgb/teacher-venv/bin/python sim/tools/eval_closed_loop.py \
      --policy teacher-nn --teacher-ckpt $DATA_ROOT/uav_rgb/teacher-run-pilot/ckpt_best.pth \
      --episodes 3 --out /tmp/eval-teacher --split test
  # T7 三个基线用同一批场景：把 student 的 seeds 回传给 --seeds。
  # 动作窗口默认 observation（唯一被采纳基线）；--window-from dispatch 仅用于复现
  # T7-R，其 SR/碰撞率不可与 observation 结果混报。
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.dirname(TOOLS)
REPO_ROOT = os.path.dirname(SIM_DIR)
SCENE_GEN = os.path.join(TOOLS, "scene_gen.py")
COLLECT = os.path.join(TOOLS, "collect.py")
RECORD = os.path.join(TOOLS, "record_views.py")
DEFAULT_TEMPLATE = os.path.join(SIM_DIR, "worlds", "uav_rgb.sdf")
from student import DEFAULT_INFER as DEFAULT_INFER_BIN
ACTION_NAMES = ("SAFE_NEUTRAL", "FORWARD_SLOW", "YAW_LEFT", "YAW_RIGHT")
WAYPOINT_RADIUS = 1.0
# L1 goal-reach 的到达半径。上限由几何定死，不是调出来的：
#   必须 < 两个候选 waypoint 最小间距 3.23 m 的一半（1.615），否则停在中间的无人机
#   对 keep 和 shuffle 都算成功，反事实就失效了；
#   必须 < 出生点到 waypoint 的最小距离 2.50 m，否则原地不动也算成功。
REACH_RADIUS = 1.5
# 新指标合同版本（SIM-09）。旧 success 定义不变，新字段单独按此版本解释。
METRIC_VERSION = "sim09.v1"
# cf_sitl（固件在环）下 FORWARD 到期后机体还会滑行，碰撞步的 action 多半已是 SAFE/YAW；
# false_forward 改成「碰撞前 1 s 仿真内下过 FORWARD」。metric_version 带后缀，数字不与旧档混报。
COAST_WINDOW_S = {"cf_sitl": 1.0}
# stable_termination：末步真值速度 < 0.05 m/s（只有 oracle 带 speed_mps 的档能算，其余 None）。
STABLE_SPEED_MPS = 0.05
# scene_gen/campus/asset_campus 目前都写死这个相机档；协议比对只用它做本次运行的实际值。
CAMERA_PROFILE = "rgb320x240_fov60_v1"
PROTOCOL_KEYS = ("window_from", "hold_s", "steps", "task",
                 "appearance_version", "dynamics", "camera_profile")

sys.path.insert(0, TOOLS)
import collect_parallel as cp  # noqa: E402
import instruction as ins  # noqa: E402
import split_manifest as sm  # noqa: E402
import campus  # noqa: E402
import asset_campus  # noqa: E402

# 与 collect_parallel 共用同一份默认矩阵（cf_sil 已退出；cf_blackbox 只留名字）。
DYNAMICS_PROFILES = cp.DYNAMICS_CHOICES


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(xs, p):
    """最近秩百分位；空列表返回 None。"""
    if not xs:
        return None
    ordered = sorted(xs)
    k = max(0, min(len(ordered) - 1, int(math.ceil(p / 100.0 * len(ordered))) - 1))
    return ordered[k]


def parse_seeds(text):
    """`--seeds` 文本（`23,25:28`）-> seed 列表；格式错误/重复 seed 抛 ValueError。"""
    seeds = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition(":")
        try:
            start = int(a)
            end = int(b) if b else start + 1
        except ValueError:
            raise ValueError(f"seed 不是整数: {part!r}（示例 23,25:28）") from None
        if start < 0 or end <= start:
            raise ValueError(f"seed 区间无效: {part!r}")
        seeds.extend(range(start, end))
    dup = sorted({s for s in seeds if seeds.count(s) > 1})
    if dup:
        raise ValueError(f"seed 重复: {dup}")
    return seeds


def dedup_key(meta, task):
    """唯一标识一个评测任务。

    base 用几何 gid；L1 必须把彩色柱的位置+颜色算进去。只用
    (gid, target_color) 会把「同几何、同目标色、但目标柱位置不同」的合法任务
    当成重复丢掉，也会让反事实对少一半。

    同色柱对之后 marks 不再能区分目标：左右两个任务的彩色柱完全一样，
    所以还要带上关系词，否则 left/right 两个任务会被去重成一个。
    """
    gid = sm.geometry_gid(meta["pillars"])
    if task not in ("l1", "campus", "campus_assets"):
        return gid
    marks = tuple(sorted(
        (round(float(p["x"]), 3), round(float(p["y"]), 3), p.get("color"))
        for p in meta["pillars"] if p.get("color")))
    if not marks:
        raise ValueError("L1 scene 没有彩色柱，无法确定评测任务")
    return (gid, marks, meta.get("relation"), meta.get("target_name"))


TEXT_MODES = ("keep", "blank", "shuffle")


def override_instruction(meta, mode):
    """L2 消融：keep=原指令；blank=空文本；shuffle=把 left/right 反过来。

    只改交给学生策略的 instruction，不改场景几何/标签/waypoint。
    同色柱对之后颜色两边一样，唯一的可控变量就是关系词——换色已经不可观测。
    """
    if mode == "keep":
        return meta
    out = dict(meta)
    if mode == "blank":
        out["instruction"] = ""
        return out
    color, relation = meta.get("target_color"), meta.get("relation")
    if color not in ins.COLORS or relation not in ins.RELATIONS:
        raise ValueError(
            f"shuffle 需要 target_color/relation，实际 {color!r}/{relation!r}")
    maker = ins.make_instruction
    # 按前缀判，不按具体版本：v1 与 v2 的指令生成器相同，旧数据仍要能评测。
    if str(meta.get("task_version", "")).startswith("campus."):
        from campus import instruction as maker
    out["instruction"] = maker(color, "right" if relation == "left" else "left")
    return out


def sha256_file(path):
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def episode_target(episodes, seed_list):
    """未显式给 --episodes 时，显式 seed 列表就是目标集数。

    否则传 20 个 seed 会按默认 3 集跑完就退出，SR 只在 3 集上算。
    """
    return episodes if episodes is not None else (len(seed_list) or 3)


def appearance_version_for(task, appearance):
    return ("campus.v1" if task == "campus" else
            asset_campus.VERSION if task == "campus_assets" else appearance)


def load_protocol(path):
    """读协议快照 JSON；必须是对象，坏文件直接报错而不是静默忽略。"""
    with open(path, "r", encoding="utf-8") as f:
        protocol = json.load(f)
    if not isinstance(protocol, dict):
        raise ValueError("protocol JSON 顶层必须是对象")
    return protocol


def protocol_mismatches(protocol, actual):
    """协议中出现的字段与本次实际参数逐项比对，返回 (key, expected, got)。

    未知字段忽略：协议文件可以带 protocol_id 等备注，不能因此判 OOD。
    """
    return [(key, protocol[key], actual.get(key)) for key in PROTOCOL_KEYS
            if key in protocol and protocol[key] != actual.get(key)]


def wilson_ci(successes, n, z=1.96):
    """Wilson 95% 区间；n=0 返回 None。用于 requested 与 done-set 两个分母。"""
    if not n:
        return None
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def _src(a):
    """collision 来源；旧 episode 没有 src 字段时返回空串（按 contact 处理，保持旧口径）。"""
    return a.get("src") or ""


def coast_false_forward(actions, window_s):
    """碰撞（contact）步之前 window_s 仿真秒内是否下过 FORWARD（含碰撞步本身）。

    按 action_start_ns 计时；缺时间戳的行退回「碰撞步动作 == FORWARD」。
    """
    starts = [a.get("action_start_ns") for a in actions]
    for k, a in enumerate(actions):
        if not (bool(a.get("collision")) and _src(a) != "stuck"):
            continue
        if starts[k] is None:
            if int(a.get("action_executed", -1)) == 1:
                return True
            continue
        for j in range(k, -1, -1):
            if starts[j] is None:
                continue
            if (int(starts[k]) - int(starts[j])) > window_s * 1e9:
                break
            if int(actions[j].get("action_executed", -1)) == 1:
                return True
    return False


def summarize_episode(actions, oracle, goal_x, waypoint=None,
                      waypoint_radius=WAYPOINT_RADIUS, stop_radius=None, goal_radius=None,
                      timed_out=False, coast_window_s=None):
    """单集指标；纯函数，便于单测。actions/oracle 是 jsonl 行。

    EVAL_STANDARD §5.1：stuck（FORWARD 位移 < 阈值）不是碰撞，碰撞率只看 contact。
    新 collect 在 actions.jsonl 写 src（contact/stuck/contact+stuck）；旧数据没有该
    字段时退回原 collision 口径，不会把历史结果改成另一个数。

    stop_radius 非空（L1）：同时算两个口径，都写进每集记录。

    success（goal_reach_ever，2026-11-02 起的判据）= 无碰撞 + **轨迹上曾**进入
    waypoint 的 REACH_RADIUS 球内。理由：waypoint 是柱阵外侧一个没有视觉标记的点，
    把「末位姿落在 1.5 m 内」当判据实际在考「停的时机」，而 fit 到 0.03 m 却飞走的
    集会被记成完全没到，和「根本没走过去」混为一谈，掩盖了导航能力。
    stop_success（旧 goal_reach）= 无碰撞 + **末位姿**在 REACH_RADIUS 内，保留用于
    对照与历史可比，不再作为主指标。
    两者都不要求末动作 SAFE_NEUTRAL（原因见下方分支注释）；`stopped` 仍是诊断字段。
    旧 route-choice 数据仍走“max_x≥goal 且经过 waypoint”的旧口径，保持历史结果可复现。

    SIM-09 起另写一组字段（旧 success/stop_success 逐位不变）：
      reach_ever = 无 contact 且轨迹曾进入 goal_radius_reach（1.5 m）
      final_in_radius = 无 contact 且末位姿在 goal_radius_stop（scene.meta.stop_radius）
      stop_action_issued = 最近 waypoint 点前后各 2 步内出现过 SAFE_NEUTRAL；
        不只看最后一步。没有可靠速度数据，因此不用速度阈值。
      task_success = reach_ever + final_in_radius + stop_action_issued + 无 contact + 未超时；
        stable_termination：oracle 末行带 speed_mps（cf_sitl）时 = 速度 < STABLE_SPEED_MPS，
        否则 None（不知道，不等于通过）。
    coast_window_s（cf_sitl）：false_forward 改为「碰撞前该窗口内下过 FORWARD」，
    metric_version 加后缀 .coast<N>s。
    """
    goal_radius_stop = float(stop_radius) if stop_radius is not None else None
    metric_version = (METRIC_VERSION if coast_window_s is None
                      else f"{METRIC_VERSION}.coast{coast_window_s:g}s")
    if not actions:
        return {"steps": 0, "collision": False, "stuck": False, "false_forward": False,
                "success": False, "stop_success": False, "max_x": None,
                "waypoint_min_dist": None, "final_waypoint_dist": None,
                "stopped": False, "infer_ms": [],
                "reach_ever": False, "final_in_radius": False,
                "stop_action_issued": False, "collision_free": True,
                "stuck_free": True, "task_success": False,
                "stable_termination": None, "timed_out": bool(timed_out),
                "metric_version": metric_version,
                "goal_radius_reach": REACH_RADIUS,
                "goal_radius_stop": goal_radius_stop}
    collision = any(bool(a.get("collision")) and _src(a) != "stuck" for a in actions)
    stuck = any(bool(a.get("collision")) and "stuck" in _src(a) for a in actions)
    if coast_window_s is None:
        false_forward = any(bool(a.get("collision")) and _src(a) != "stuck" and
                            int(a.get("action_executed", -1)) == 1 for a in actions)
    else:
        false_forward = coast_false_forward(actions, coast_window_s)
    final_speed = oracle[-1].get("speed_mps") if oracle and isinstance(oracle[-1], dict) else None
    stable_termination = (bool(float(final_speed) < STABLE_SPEED_MPS)
                          if final_speed is not None else None)
    xs = [float(o["pose_gt"]["x"]) for o in oracle
          if isinstance(o.get("pose_gt"), dict) and "x" in o["pose_gt"]]
    max_x = max(xs) if xs else None
    wp_min = wp_final = None
    nearest_i = None
    if isinstance(waypoint, dict):
        ds = []
        best = None
        for i, o in enumerate(oracle):
            pg = o.get("pose_gt")
            if isinstance(pg, dict) and "x" in pg and "y" in pg:
                d = ((pg["x"] - waypoint["x"]) ** 2 +
                     (pg["y"] - waypoint["y"]) ** 2) ** 0.5
                ds.append(d)
                if best is None or d < best:
                    best, nearest_i = d, i
        wp_min = min(ds) if ds else None
        wp_final = ds[-1] if ds else None
    stopped = int(actions[-1].get("action_executed", -1)) == 0
    # 最近点前后各 2 步：SAFE 常出现在到达点前一两步，只看末步会把「到点后继续飞走」
    # 全判成没停。窗口按 actions 行与 oracle 行一一对应取索引。
    stop_action_issued = bool(nearest_i is not None and any(
        int(a.get("action_executed", -1)) == 0
        for a in actions[max(0, nearest_i - 2):nearest_i + 3]))
    if stop_radius is not None:
        # 2026-11-01 起不再要求末动作 SAFE_NEUTRAL：waypoint 在柱阵外侧 ±2.4 m，
        # 相机 60° 视场在真停止点上 40% 的帧一根柱子都看不到，"停"不可观测
        # （worklog 2026-11-01-2130-stop-point-unobservable.md）。改成"到达"。
        radius = REACH_RADIUS if goal_radius is None else goal_radius
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("goal_radius 必须有限且为正")
        success = bool(not collision and wp_min is not None
                       and wp_min <= radius)
        stop_success = bool(not collision and wp_final is not None
                            and wp_final <= radius)
    else:
        reached = wp_min is None or wp_min <= waypoint_radius
        success = bool(not collision and max_x is not None and max_x >= goal_x and reached)
        stop_success = success
    collision_free = not collision
    reach_ever = bool(collision_free and wp_min is not None
                      and wp_min <= REACH_RADIUS)
    final_in_radius = bool(collision_free and wp_final is not None
                           and goal_radius_stop is not None
                           and wp_final <= goal_radius_stop)
    task_success = bool(reach_ever and final_in_radius and stop_action_issued
                        and collision_free and not timed_out)
    infer = [float(a["infer_ms"]) for a in actions if a.get("infer_ms") is not None]
    return {"steps": len(actions), "collision": collision, "stuck": stuck,
            "false_forward": false_forward, "success": success,
            "stop_success": stop_success,
            "max_x": max_x, "waypoint_min_dist": wp_min,
            "final_waypoint_dist": wp_final, "stopped": stopped, "infer_ms": infer,
            "reach_ever": reach_ever, "final_in_radius": final_in_radius,
            "stop_action_issued": stop_action_issued,
            "collision_free": collision_free, "stuck_free": not stuck,
            "task_success": task_success,
            # 只有 oracle 带真值速度（cf_sitl）才能判；其余档 None，不等于通过。
            "stable_termination": stable_termination, "timed_out": bool(timed_out),
            "metric_version": metric_version,
            "goal_radius_reach": REACH_RADIUS,
            "goal_radius_stop": goal_radius_stop}


def aggregate(episodes, requested=None, attempted=None, failed_count=0, skipped_count=0):
    """汇总。分母写清楚：
    - success_rate / 新指标率 = episodes_done；
    - requested_success_rate = episodes_requested（固定请求集，含 failed/skipped 的非成功）；
    - completion_rate = 实际尝试数（done+failed，skipped 不算）；
    - execution_rate = episodes_done / episodes_requested（skipped 表现为未执行）。
    比率都落在 [0,1]：失败后补扫 seed 不会把 execution_rate 推过 1。
    """
    done = [e for e in episodes if e.get("steps", 0) > 0]
    n = len(done)
    if attempted is None:
        attempted = n
    infer_all = [ms for e in done for ms in e["infer_ms"]]
    infer_warm = [ms for e in done for ms in e["infer_ms"][1:]]  # 首帧含 CUDA 初始化
    dist = {name: 0 for name in ACTION_NAMES}
    for e in done:
        for k, v in e.get("actions", {}).items():
            if 0 <= k < len(ACTION_NAMES):
                dist[ACTION_NAMES[k]] += v

    def rate(key):
        return (sum(1 for e in done if e.get(key)) / n) if n else None

    successes = sum(1 for e in done if e.get("success"))
    return {
        "episodes_done": n,
        "success_rate": rate("success"),
        "requested_success_rate": (successes / requested) if requested else None,
        # skipped 不进 completion 分母：只看真正尝试过的集（失败算尝试）。
        "completion_rate": (n / attempted) if attempted else None,
        "execution_rate": (n / requested) if requested else None,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
        "unique_gids": len({e["geometry_gid"] for e in done
                            if e.get("geometry_gid") is not None}),
        "wilson_ci": {
            "success": wilson_ci(successes, n),
            "requested": wilson_ci(successes, requested) if requested else None,
        },
        # 旧「末位姿在 1.5 m 内」口径，保留对照；主指标改看 success_rate。
        "stop_success_rate": rate("stop_success"),
        # SIM-09 新口径：分别报告，旧 success_rate 不动。
        "reach_ever_rate": rate("reach_ever"),
        "final_in_radius_rate": rate("final_in_radius"),
        "stop_action_issued_rate": rate("stop_action_issued"),
        "task_success_rate": rate("task_success"),
        "collision_free_rate": rate("collision_free"),
        "stuck_free_rate": rate("stuck_free"),
        # 只在有真值速度的集上算（cf_sitl）；一集都没有就是 None，不是 0。
        "stable_termination_rate": (
            (sum(1 for e in done if e.get("stable_termination")) /
             sum(1 for e in done if e.get("stable_termination") is not None))
            if any(e.get("stable_termination") is not None for e in done) else None),
        "collision_rate": rate("collision"),
        # stuck 是「被挡住没动」而非接触；EVAL_STANDARD 要求单列，不能混进碰撞率。
        "stuck_rate": rate("stuck"),
        "false_forward_rate": rate("false_forward"),
        "inference_ms": {
            "p50": percentile(infer_warm, 50),
            "p95": percentile(infer_warm, 95),
            "n_warm": len(infer_warm),
            "cold_first_ms": [round(e["infer_ms"][0], 3) for e in done if e["infer_ms"]],
            "p50_all": percentile(infer_all, 50),
            "p95_all": percentile(infer_all, 95),
        },
        "action_distribution": dist,
    }


def run_episode(seed, args, workdir, seen_gids):
    """返回 (summary 或 None, 跳过原因)。None 表示该 seed 不在目标 split/已评测。"""
    world = os.path.join(workdir, f"scene_{seed}.sdf")
    meta_path = os.path.join(workdir, f"scene_{seed}.json")
    cmd = [sys.executable, SCENE_GEN, "--seed", str(seed), "--scene-id", str(seed),
           "--out", world, "--template", args.template, "--task", args.task,
           "--appearance", args.appearance] + cp.scene_gen_extra(args)
    if args.task == "l1":
        # v5 关系任务要求两根同色柱横跨出生点，默认 spawn 在 y=0 时
        # assign_landmarks 找不到柱对会直接抛错（采集也是 random）。
        cmd += ["--spawn-mode", "random"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"scene_gen seed={seed} 失败：{r.stderr.strip()}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    gid = sm.geometry_gid(meta["pillars"])
    split = sm.split_for_scene(meta)
    if split != args.split:
        return None, f"split={split}"
    # 在起 sim 之前去重：key 必须唯一标识「几何 + 彩色柱位置/颜色」，
    # 否则同几何不同目标柱的合法任务会被当成重复，反事实对也会少一半。
    dedup = dedup_key(meta, args.task)
    if dedup in seen_gids:
        return None, f"已评测 {dedup}"

    eval_meta = override_instruction(meta, args.text_mode)
    eval_meta_path = meta_path
    if args.text_mode != "keep":
        eval_meta_path = os.path.join(workdir, f"scene_{seed}.text.json")
        with open(eval_meta_path, "w", encoding="utf-8") as f:
            json.dump(eval_meta, f, ensure_ascii=False)

    goal_x = max(float(p["x"]) for p in meta["pillars"]) + args.goal_margin
    partition = f"uavrgb_eval_{os.getpid()}_{seed}"
    log_path = os.path.join(workdir, "logs", f"sim_{seed}.log")
    ep_dir = os.path.join(workdir, f"ep_{seed}")
    os.makedirs(ep_dir, exist_ok=True)
    sim = cp._start_sim(world, partition, log_path)
    t0 = time.monotonic()
    rec = None
    frames_dir = os.path.join(args.video_dir, f".frames_{seed}") if args.record_video else None
    try:
        cp.wait_ready(sim, partition, log_path, args.sim_timeout)
        time.sleep(cp.SIM_SETTLE_S)
        if frames_dir:
            # 录像与 collect 并行订阅相机；只读话题，不影响动作窗口（窗口按仿真时间计）。
            rec = subprocess.Popen([sys.executable, RECORD, "record", "--out", frames_dir],
                                   env=cp.gz_env(partition), stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, start_new_session=True)
            cp._register(rec)
        cmd = [sys.executable, COLLECT, "--out", ep_dir,
               "--steps", str(args.steps), "--hold", str(args.hold),
               "--policy", args.policy,
               "--stop-on-collision",
               "--scene-id", str(seed), "--seed", str(seed),
               "--scene-meta", eval_meta_path, "--split", split,
               "--window-from", args.window_from, "--dynamics", args.dynamics]
        if args.save_depth:
            cmd += ["--save-depth"]
        if args.policy == "student":
            cmd += ["--student-ckpt", args.ckpt]
            if args.infer_bin:
                cmd += ["--infer-bin", args.infer_bin]
        elif args.policy == "teacher-nn":
            cmd += ["--teacher-ckpt", args.teacher_ckpt]
        elif args.policy == "teacher-rl":
            cmd += ["--rl-checkpoint", args.rl_checkpoint]
        if args.policy == "teacher" and args.teacher_planner != "astar":
            cmd += ["--teacher-planner", args.teacher_planner]
        if getattr(args, "vehicle", None) and args.vehicle != "x3":
            cmd += ["--vehicle", args.vehicle]   # x3 不传：旧命令行逐字不变
        env = cp.gz_env(partition)
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, start_new_session=True)
        cp._register(proc)   # Ctrl-C/SIGTERM 时 stop_group 能兜底杀掉 collect
        try:
            out, err = proc.communicate(timeout=args.episode_timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, err = proc.communicate()
            rc = "timeout"
        finally:
            cp.stop_group(proc, f"eval collect seed {seed}")
    finally:
        cp.stop_group(rec, f"eval recorder seed {seed}")
        cp.stop_group(sim, f"eval seed {seed}")

    actions_path = os.path.join(ep_dir, "actions.jsonl")
    oracle_path = os.path.join(ep_dir, "oracle.jsonl")
    actions = load_jsonl(actions_path) if os.path.isfile(actions_path) else []
    oracle = load_jsonl(oracle_path) if os.path.isfile(oracle_path) else []
    # 默认会删除临时采集目录，先将性能与请求后端带入持久报告。
    # 旧集/采集失败没有这些字段时保留 null，不推断实际 renderer。
    try:
        with open(os.path.join(ep_dir, "episode_manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
    except FileNotFoundError:
        manifest = {}
    summary = summarize_episode(actions, oracle, goal_x, meta.get("waypoint"),
                                WAYPOINT_RADIUS, meta.get("stop_radius"), meta.get("goal_radius"),
                                timed_out=(rc == "timeout"),
                                coast_window_s=COAST_WINDOW_S.get(args.dynamics))
    summary.update({
        "seed": seed, "geometry_gid": gid, "split": split, "policy": args.policy,
        "task": args.task, "window_from": args.window_from,
        "dynamics": args.dynamics,
        "vehicle": getattr(args, "vehicle", None) or "x3",
        "text_mode": args.text_mode,
        "task_version": meta.get("task_version"),
        **{key: meta.get(key) for key in sm.PROVENANCE_KEYS},
        # 场景只写 x3_v1；cf_stab 的真实档名在采集 manifest 里。
        "dynamics_profile": manifest.get("dynamics_profile") or meta.get("dynamics_profile"),
        "target_color": meta.get("target_color"),
        "instruction": eval_meta.get("instruction"),
        "waypoint": meta.get("waypoint"),
        "waypoint_radius": WAYPOINT_RADIUS,
        "stop_radius": meta.get("stop_radius"),
        "pillars": meta.get("pillar_count"), "goal_x": goal_x,
        "collect_rc": rc, "wall_s": round(time.monotonic() - t0, 1),
        "performance": manifest.get("performance"),
        "render_backend_requested": manifest.get("render_backend_requested"),
        "actions": {},
        "stderr_tail": err.strip().splitlines()[-5:],
        "dedup_key": dedup,
        # 地图标识与评测任务标识不是一回事：campus 每 8 个 seed 是同一张图，
        # dedup_key 还含颜色/左右，不能拿来数地图覆盖。
        "geometry_gid": sm.geometry_gid(meta["pillars"]),
    })
    for a in actions:
        k = int(a.get("action_executed", -1))
        summary["actions"][k] = summary["actions"].get(k, 0) + 1
    if frames_dir:
        _write_video(summary, frames_dir, ep_dir, world, args, seed, goal_x, meta)
    return summary, None


def scan_episodes(seed_list, episodes, run_fn, log=print):
    """固定请求集扫描（AUDIT #11）：成功与采集失败都占一个名额，不再为失败补扫新 seed。

    之前按 len(results) 停：1 集失败就多扫 1 个 seed 补上，requested_SR 的分母名义上是
    请求数、实际成功集却来自更大的集合（4 请求 1 失败 → SR 1.0 / execution 1.0 /
    completion 0.75 自相矛盾）。现在按 len(attempted) 停：失败进分母、单列 failed。
    跳过（split 不符 / 重复几何）不占名额。run_fn(seed, seen) -> (summary, skip_reason)。
    返回 (results, failures, skips, scanned, attempted)。
    """
    results, seen, failures, skips, scanned, attempted = [], set(), [], [], [], []
    for seed in seed_list:
        if len(attempted) >= episodes:
            break
        scanned.append(seed)
        try:
            summary, skip = run_fn(seed, seen)
        except Exception as e:  # noqa: BLE001 - 单集失败要记录，不能拖垮整轮
            log(f"[seed {seed}] 失败: {type(e).__name__}: {e}", flush=True)
            attempted.append(seed)
            failures.append({"seed": seed, "error": f"{type(e).__name__}: {e}"})
            continue
        if skip:
            log(f"[seed {seed}] 跳过: {skip}", flush=True)
            skips.append({"seed": seed, "reason": skip})
            continue
        attempted.append(seed)
        if summary["collect_rc"] != 0 or summary["steps"] == 0:
            # 采集中途失败不是「学生飞得差」，不计进 SR 分子，但占请求名额。
            failures.append({"seed": seed, "rc": summary["collect_rc"],
                             "stderr_tail": summary.get("stderr_tail"),
                             "performance": summary.get("performance"),
                             "render_backend_requested": summary.get("render_backend_requested")})
            log(f"[seed {seed}] 采集失败 rc={summary['collect_rc']}，占请求名额、不计入成功",
                flush=True)
            continue
        seen.add(summary["dedup_key"])
        results.append(summary)
        log(f"[seed {seed}] steps={summary['steps']} "
            f"collision={summary['collision']} success={summary['success']} "
            f"max_x={summary['max_x']} rc={summary['collect_rc']}", flush=True)
    return results, failures, skips, scanned, attempted


def _write_video(summary, frames_dir, ep_dir, world, args, seed, goal_x, meta):
    """三视图视频 -> <out>/videos/seed_<seed>.mp4。失败只记 video_error，不影响评测结论。"""
    import record_views
    result = ("SUCCESS" if summary.get("success") else
              "COLLISION" if summary.get("collision") else
              "STUCK" if summary.get("stuck") else "NOT REACHED")
    if args.task in ("campus", "campus_assets"):
        criterion, goal_radius = "visible_goal_reach_v1", summary.get("goal_radius_stop")
    elif meta.get("stop_radius") is not None:
        criterion, goal_radius = "goal_reach_ever", summary.get("goal_radius_reach")
    else:
        criterion, goal_radius = "route_choice", None
    path = os.path.join(args.video_dir, f"seed_{seed}.mp4")
    try:
        n = record_views.compose(frames_dir, ep_dir, world, path,
                                 title=f"{args.policy} | {args.dynamics} | seed {seed}",
                                 result=result, goal_x=goal_x, waypoint=meta.get("waypoint"),
                                 success_criterion=criterion, goal_radius=goal_radius)
        summary["video"] = os.path.relpath(path, os.path.dirname(args.video_dir))
        summary["video_frames"] = n
    except Exception as e:  # noqa: BLE001 - 视频是诊断产物，不能让评测失败
        summary["video_error"] = f"{type(e).__name__}: {e}"
    if not args.keep_work:
        shutil.rmtree(frames_dir, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None, help="学生 checkpoint（--policy student 必填）")
    p.add_argument("--policy", choices=("student", "forward", "teacher-nn", "teacher", "teacher-rl"),
                   default="student",
                   help="student=模型闭环；forward=盲前进基线（T7）；"
                        "teacher-nn=特权教师网络闭环（可解性闸门）；"
                        "teacher=柱子真值 A* 专家闭环（新动力学下的可解性上界）")
    p.add_argument("--teacher-planner", choices=("astar", "predictive"), default="astar")
    p.add_argument("--rl-checkpoint", default="")
    p.add_argument("--teacher-ckpt", default=None,
                   help="PrivilegedTeacher checkpoint（--policy teacher-nn 必填）")
    p.add_argument("--episodes", type=int, default=None,
                   help="请求评测的 episode 数 = SR 分母（失败集也占名额并计入分母，列在 failed；"
                        "缺省：给了 --seeds 用其个数，否则 3）")
    p.add_argument("--steps", type=int, default=80, help="每集最大步数")
    p.add_argument("--hold", type=float, default=1.0, help="每动作仿真时间秒")
    # default=None 只为识别「用户显式给了 dispatch」；缺省解析成 observation。
    p.add_argument("--window-from", choices=("dispatch", "observation"),
                   default=None,
                   help="动作窗口起点：observation=观测位姿（默认，唯一被采纳基线）；"
                        "dispatch=派发位姿（仅复现 T7-R，数字不可与 observation 混报）")
    p.add_argument("--dynamics", choices=DYNAMICS_PROFILES, default="x3_velocity",
                   help="透传给 collect.py：x3_velocity=旧行为（唯一历史基线）；"
                        "cf_stab=CF-Drone 摇杆模型（建议配 --hold 0.1~0.2）；"
                        "cf_blackbox=纯 Python 黑盒（3DGS collector 直接使用；Gazebo "
                        "闭环会明确拒绝，不拿 x3 物理冒充）；"
                        "cf_sitl=E1 MINI 固件在环（必须 --vehicle e1_mini；false_forward 按滑行窗口"
                        f"{COAST_WINDOW_S['cf_sitl']:g}s 判，metric_version 带 .coast 后缀）")
    p.add_argument("--vehicle", choices=("x3", "e1_mini"), default=None,
                   help="世界载具：缺省 x3（旧行为）；e1_mini 选 E1 模板并透传给 collect.py")
    p.add_argument("--camera-rate", type=float, default=None,
                   help="透传给 scene_gen：学生 RGB/深度相机 update_rate（Hz）；缺省用模板值")
    p.add_argument("--no-aux-cameras", action="store_true",
                   help="透传给 scene_gen：不生成 chase/top 辅助相机（仅 E1 模板；--record-video 时不要加）")
    p.add_argument("--protocol", default=None,
                   help="可选协议快照 JSON；比对 window_from/hold_s/steps/task/"
                        "appearance_version/dynamics/camera_profile，不一致写 ood=true 并警告")
    p.add_argument("--strict-protocol", action="store_true",
                   help="协议与本次参数不一致时直接失败（默认只警告，保证旧 run 可复现）")
    p.add_argument("--text-mode", choices=TEXT_MODES, default="keep",
                   help="L2 消融：keep=原指令；blank=空文本；shuffle=把 left/right 反过来")
    p.add_argument("--task", choices=("base", "l1", "campus", "campus_assets", "forest_e1"), default="base",
                   help="base=原柱阵；l1=彩色 landmark + instruction/waypoint；"
                        "forest_e1=E1 随机圆柱森林（需 --density 与 --vehicle e1_mini）")
    p.add_argument("--density", choices=("sparse", "medium", "dense"), default=None,
                   help="透传给 scene_gen：forest_e1 密度档（最窄缝 0.45/0.35/0.28 m）；forest_e1 必填")
    p.add_argument("--appearance", choices=("legacy", "v1"), default="v1",
                   help="透传给 scene_gen：v1 删固定地标并随机化光照/天空（几何不变）；旧数据复现请显式 legacy")
    p.add_argument("--split", choices=("train", "val", "test"), default="test")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--max-seeds", type=int, default=500,
                   help="最多扫描多少个 seed 来凑够 --episodes（test split 约 10%%）")
    p.add_argument("--seeds", default="",
                   help="固定 seed 列表（逗号分隔或 a:b）；T7 三个基线用同一批场景")
    p.add_argument("--out", required=True)
    p.add_argument("--template", default=None,
                   help="模板 SDF（缺省按 --vehicle 选；x3 为 sim/worlds/uav_rgb.sdf）")
    p.add_argument("--infer-bin", default=None)
    p.add_argument("--goal-margin", type=float, default=1.0,
                   help="成功 = 无碰撞且 max_x >= 最远柱 x + 该余量")
    p.add_argument("--sim-timeout", type=float, default=120.0)
    p.add_argument("--episode-timeout", type=float, default=600.0)
    p.add_argument("--keep-work", action="store_true", help="保留 world/每集原始文件")
    p.add_argument("--record-video", action="store_true",
                   help="每集录 机载/追尾/顶视 三视图视频到 <out>/videos/seed_<seed>.mp4"
                        "（原始帧约 10 MB/仿真秒，暂存 <out>/videos/.frames_*，合成后删除）")
    p.add_argument("--save-depth", action="store_true",
                   help="保留每步 16 位深度 PNG（诊断闭环失败用；配合 --keep-work）")
    args = p.parse_args()
    if args.window_from is None:
        args.window_from = "dispatch" if args.task in ("campus", "campus_assets") else "observation"
    elif args.window_from == "dispatch" and args.task not in ("campus", "campus_assets"):
        print("警告: --window-from dispatch 不是被采纳的基线语义（唯一基线是 "
              "observation）；本次 SR/碰撞率不可与 observation 结果混报。",
              file=sys.stderr, flush=True)

    actual_protocol = {
        "window_from": args.window_from, "hold_s": args.hold, "steps": args.steps,
        "task": args.task,
        "appearance_version": appearance_version_for(args.task, args.appearance),
        "dynamics": args.dynamics, "camera_profile": CAMERA_PROFILE,
    }
    protocol, mismatches = None, []
    if args.protocol:
        try:
            protocol = load_protocol(args.protocol)
        except (OSError, ValueError) as e:
            p.error(f"--protocol 无法读取: {e}")
        mismatches = protocol_mismatches(protocol, actual_protocol)
        if mismatches:
            detail = "; ".join(f"{k}: protocol={v!r} actual={g!r}"
                               for k, v, g in mismatches)
            if args.strict_protocol:
                p.error(f"protocol 与本次参数不一致: {detail}")
            print(f"警告: protocol 不一致（本次结果 ood=true）: {detail}",
                  file=sys.stderr, flush=True)
    ood = bool(mismatches)

    try:
        seed_list = parse_seeds(args.seeds) if args.seeds else []
    except ValueError as e:
        p.error(str(e))
    if args.seeds and not seed_list:
        p.error("--seeds 没有有效 seed")
    args.episodes = episode_target(args.episodes, seed_list)
    if args.episodes < 1 or args.steps < 1:
        p.error("episodes/steps 必须 >=1")
    if not seed_list and args.max_seeds < args.episodes:
        p.error("max-seeds >= episodes")
    if not (math.isfinite(args.hold) and args.hold > 0):
        p.error("--hold 必须是有限正数")
    if args.dynamics == "cf_blackbox":
        p.error("cf_blackbox 是纯 Python 模型档，Gazebo 闭环未接入；"
                "请让 3DGS collector 直接调用 cf_blackbox，或改用 cf_stab/x3_velocity")
    cp.resolve_vehicle(args, p.error)
    if (args.task == "forest_e1") != (args.density is not None):
        p.error("--density 与 --task forest_e1 必须同时给")
    if args.task == "forest_e1" and args.vehicle != "e1_mini":
        p.error("--task forest_e1 需要 --vehicle e1_mini（--dynamics cf_sitl）")
    if args.record_video and args.no_aux_cameras:
        p.error("--record-video 需要 chase/top 相机，不能同时 --no-aux-cameras")
    for name in ("goal_margin", "sim_timeout", "episode_timeout"):
        value = getattr(args, name)
        if not (math.isfinite(value) and value > 0):
            p.error(f"--{name.replace('_', '-')} 必须是有限正数")
    if args.policy == "teacher-nn":
        if not args.teacher_ckpt:
            p.error("--policy teacher-nn 需要 --teacher-ckpt")
        if not os.path.isfile(args.teacher_ckpt):
            p.error(f"teacher checkpoint 不存在: {args.teacher_ckpt}")
        # collect.py 由 run_episode 用 sys.executable 起，所以这里探的就是子进程解释器。
        # 不能只看 find_spec：本机 conda 的 torch 装了但缺 libcudnn，import 会失败。
        try:
            torch_ok = subprocess.run([sys.executable, "-c", "import torch"],
                                      capture_output=True,
                                      timeout=120).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            torch_ok = False
        if not torch_ok:
            p.error("teacher-nn 需要带 torch 的解释器，例如 "
                    "$DATA_ROOT/uav_rgb/teacher-venv/bin/python")
    if (args.teacher_planner == "predictive" or args.policy == "teacher-rl") and (
            args.dynamics != "cf_stab" or args.task not in ("campus", "campus_assets")
            or abs(args.hold-0.2)>1e-6):
        p.error("新 Teacher 需要 --dynamics cf_stab --hold 0.2 --task campus_assets/campus")
    if args.policy == "teacher-rl" and not args.rl_checkpoint:
        p.error("teacher-rl 需要 --rl-checkpoint")
    if args.text_mode != "keep" and args.policy != "student":
        p.error("--text-mode 只对 --policy student 有意义")
    infer_bin = None
    text_dim = None
    if args.policy == "student":
        if not args.ckpt:
            p.error("--policy student 需要 --ckpt")
        if not os.path.isfile(args.ckpt):
            p.error(f"checkpoint 不存在: {args.ckpt}")
        infer_bin = os.path.abspath(args.infer_bin or DEFAULT_INFER_BIN)
        if not os.path.isfile(infer_bin):
            p.error(f"推理程序不存在: {infer_bin}")
        # 无文本分支的 ckpt 上 blank/shuffle 与 keep 完全等价：不先探一次，
        # 会把「消融没接上」读成「语言没有增益」。
        import student as stu
        try:
            probe = stu.StudentPolicy(args.ckpt, infer_bin)
        except RuntimeError as e:
            p.error(str(e))
        try:
            text_dim = probe.text_dim
        finally:
            probe.close()
        if args.text_mode != "keep" and text_dim <= 0:
            p.error(f"ckpt text_dim={text_dim}，--text-mode {args.text_mode} 不会改变推理")
        if args.text_mode != "keep" and args.task not in ("l1", "campus", "campus_assets"):
            p.error("--text-mode 只对 --task l1/campus 有意义")
        args.infer_bin = infer_bin  # 让 run_episode 与 report 用同一个路径
    out = os.path.abspath(args.out)
    if os.path.isdir(out) and os.listdir(out):
        p.error(f"--out {out} 非空；请用新目录")
    os.makedirs(out, exist_ok=True)
    args.video_dir = os.path.join(out, "videos")
    if args.record_video:
        os.makedirs(args.video_dir, exist_ok=True)
    try:
        os.makedirs(cp.XDG_RUNTIME_DIR, mode=0o700, exist_ok=True)
        os.chmod(cp.XDG_RUNTIME_DIR, 0o700)
    except OSError as e:
        print(f"警告: 无法准备 XDG_RUNTIME_DIR: {e}", flush=True)

    workdir = (os.path.join(out, ".work") if args.keep_work
               else tempfile.mkdtemp(prefix="uavrgb_eval_"))
    os.makedirs(os.path.join(workdir, "logs"), exist_ok=True)

    def _term(_sig, _frame):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, _term)
    if not seed_list:
        seed_list = list(range(args.base_seed, args.base_seed + args.max_seeds))
    started = time.monotonic()
    try:
        results, failures, skips, scanned_seeds, attempted_seeds = scan_episodes(
            seed_list, args.episodes,
            lambda seed, seen: run_episode(seed, args, workdir, seen))
    finally:
        cp.cleanup_all()

    agg = aggregate(results, requested=args.episodes,
                    attempted=len(attempted_seeds),
                    failed_count=len(failures), skipped_count=len(skips))
    stop_radius_used = next((e.get("stop_radius") for e in results
                             if e.get("stop_radius") is not None), None)
    report = {"schema": "uavrgb.eval.v1", "policy": args.policy,
              "teacher_planner": args.teacher_planner,
              "rl_checkpoint": args.rl_checkpoint if args.policy == "teacher-rl" else None,
              "task": args.task, "window_from": args.window_from,
              "dynamics": args.dynamics,
              "vehicle": args.vehicle or "x3",
              "camera_rate_hz": args.camera_rate,   # None = 模板默认（X3 30 Hz / E1 10 Hz）
              "aux_cameras": not args.no_aux_cameras,
              "text_mode": args.text_mode, "text_dim": text_dim,
              # 新指标合同（SIM-09）；旧 success/stop_success 定义不变。
              # cf_sitl 带 .coast 后缀：false_forward 按滑行窗口判，数字不与旧档混报。
              "metric_version": (METRIC_VERSION if args.dynamics not in COAST_WINDOW_S else
                                 f"{METRIC_VERSION}.coast{COAST_WINDOW_S[args.dynamics]:g}s"),
              "coast_window_s": COAST_WINDOW_S.get(args.dynamics),
              # 成功语义跟着数据走：L1 是 goal_reach_ever「轨迹上曾进入 waypoint
              # 1.5 m 且无碰撞」，旧数据是 route_choice「越过终点」。不写死会让新旧
              # SR 被直接混报。历史口径：2026-11-01 前是 goal_stop（含 SAFE 判定），
              # 2026-11-01~11-02 是 goal_reach（**末位姿**在 1.5 m 内），三者都不可比。
              "success_criterion": ("visible_goal_reach_v1" if args.task in ("campus", "campus_assets") else
                                    "goal_reach_ever" if stop_radius_used is not None
                                    else "route_choice"),
              "appearance_version": appearance_version_for(args.task, args.appearance),
              "goal_radius": (0.5 if args.task in ("campus", "campus_assets") else
                              REACH_RADIUS if stop_radius_used is not None else None),
              # L1 训练目标是 0.5 m 停、评测按 1.5 m 到达：两个半径同时写。
              "goal_radius_reach": REACH_RADIUS,
              "goal_radius_stop": stop_radius_used,
              "template_sha256": sha256_file(args.template),
              # requested_seeds 只含真正尝试过的 seed；扫描过的另存，避免把未处理
              # 的扫描尾巴当成已请求集。
              "requested_seeds": list(attempted_seeds),
              "scanned_seeds": scanned_seeds,
              "attempted_seeds": list(attempted_seeds),
              "protocol": protocol,
              "protocol_mismatches": [{"key": k, "protocol": v, "actual": g}
                                      for k, v, g in mismatches],
              "ood": ood,
              "strict_protocol": args.strict_protocol,
              "goal_margin": args.goal_margin,
              "waypoint_radius": WAYPOINT_RADIUS,
              "stop_radius": stop_radius_used,
              "geometry_version": cp.task_geometry_version(args.task),
              "density": args.density,   # forest_e1 的密度档；其它任务 None
              "ckpt": os.path.abspath(args.ckpt) if args.ckpt else None,
              "ckpt_sha256": sha256_file(args.ckpt) if args.ckpt else None,
              "infer_bin": infer_bin,
              "infer_bin_sha256": (sha256_file(infer_bin)
                                   if args.policy == "student" else None),
              "split": args.split, "episodes_requested": args.episodes,
              "steps": args.steps, "hold_s": args.hold,
              "seeds": [e["seed"] for e in results],
              # 集数不等于独立地图数：campus 每 8 个 seed 是同一张图，扫出来的
              # 20 集可能只覆盖 3 张图（2026-11-03 审计）。不写出来会被当 20 个样本。
              "unique_maps": len({e["geometry_gid"] for e in results}),
              "wall_s": round(time.monotonic() - started, 1),
              **agg, "failed": failures, "skipped": skips,
              "per_episode": [{k: v for k, v in e.items() if k != "infer_ms"}
                              for e in results]}
    path = os.path.join(out, "eval.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    successes = sum(1 for e in results if e.get("success"))
    # 首行先报 requested（分母 = 本次请求集数），再报 done-set（分母 = 完成集数）。
    print(f"requested: SR={agg['requested_success_rate']} "
          f"({successes}/{args.episodes}); "
          f"completion={agg['completion_rate']} "
          f"({agg['episodes_done']}/{len(attempted_seeds)} attempted); "
          f"execution={agg['execution_rate']} "
          f"({agg['episodes_done']}/{args.episodes} requested); "
          f"failed={agg['failed_count']} skipped={agg['skipped_count']} "
          f"ood={ood}")
    print(f"done-set: SR={agg['success_rate']} ({successes}/{agg['episodes_done']}); "
          f"SR_stop={agg['stop_success_rate']}  collision={agg['collision_rate']}  "
          f"stuck={agg['stuck_rate']}  "
          f"false_forward={agg['false_forward_rate']}")
    print(f"new ({METRIC_VERSION}): reach_ever={agg['reach_ever_rate']} "
          f"final_in_radius={agg['final_in_radius_rate']} "
          f"stop_action={agg['stop_action_issued_rate']} "
          f"task_success={agg['task_success_rate']}  "
          f"radii reach={REACH_RADIUS}m stop={stop_radius_used}m")
    print(f"wilson95: success={agg['wilson_ci']['success']} "
          f"requested={agg['wilson_ci']['requested']}")
    im = agg["inference_ms"]
    print(f"inference p50={im['p50']}ms p95={im['p95']}ms "
          f"(warm n={im['n_warm']}; cold first={im['cold_first_ms']})")
    print(f"action distribution: {agg['action_distribution']}")
    print(f"-> {path}")

    if args.keep_work:
        print(f"原始文件保留在 {workdir}")
    else:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0 if agg["episodes_done"] == args.episodes else 1


if __name__ == "__main__":
    sys.exit(main())
