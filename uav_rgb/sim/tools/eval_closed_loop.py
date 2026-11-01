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
  # T7 三个基线用同一批场景：把 student 的 seeds 回传给 --seeds。
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
DEFAULT_TEMPLATE = os.path.join(SIM_DIR, "worlds", "uav_rgb.sdf")
DEFAULT_INFER_BIN = os.path.join(REPO_ROOT, "src", "build", "uavrgb_infer")
ACTION_NAMES = ("SAFE_NEUTRAL", "FORWARD_SLOW", "YAW_LEFT", "YAW_RIGHT")
WAYPOINT_RADIUS = 1.0
# L1 goal-reach 的到达半径。上限由几何定死，不是调出来的：
#   必须 < 两个候选 waypoint 最小间距 3.23 m 的一半（1.615），否则停在中间的无人机
#   对 keep 和 shuffle 都算成功，反事实就失效了；
#   必须 < 出生点到 waypoint 的最小距离 2.50 m，否则原地不动也算成功。
REACH_RADIUS = 1.5

sys.path.insert(0, TOOLS)
import collect_parallel as cp  # noqa: E402
import instruction as ins  # noqa: E402
import split_manifest as sm  # noqa: E402


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
    if task != "l1":
        return gid
    marks = tuple(sorted(
        (round(float(p["x"]), 3), round(float(p["y"]), 3), p.get("color"))
        for p in meta["pillars"] if p.get("color")))
    if not marks:
        raise ValueError("L1 scene 没有彩色柱，无法确定评测任务")
    return (gid, marks, meta.get("relation"))


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
    out["instruction"] = ins.make_instruction(
        color, "right" if relation == "left" else "left")
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


def _src(a):
    """collision 来源；旧 episode 没有 src 字段时返回空串（按 contact 处理，保持旧口径）。"""
    return a.get("src") or ""


def summarize_episode(actions, oracle, goal_x, waypoint=None,
                      waypoint_radius=WAYPOINT_RADIUS, stop_radius=None):
    """单集指标；纯函数，便于单测。actions/oracle 是 jsonl 行。

    EVAL_STANDARD §5.1：stuck（FORWARD 位移 < 阈值）不是碰撞，碰撞率只看 contact。
    新 collect 在 actions.jsonl 写 src（contact/stuck/contact+stuck）；旧数据没有该
    字段时退回原 collision 口径，不会把历史结果改成另一个数。

    stop_radius 非空（L1 goal-reach）：success = 无碰撞 + 最终位姿距 waypoint
    ≤ REACH_RADIUS。不再要求末动作 SAFE_NEUTRAL（原因见该分支注释）。
    `stopped` 仍作为诊断字段返回，但不参与判定。
    旧 route-choice 数据仍走“max_x≥goal 且经过 waypoint”的旧口径，保持历史结果可复现。
    """
    if not actions:
        return {"steps": 0, "collision": False, "stuck": False, "false_forward": False,
                "success": False, "max_x": None, "waypoint_min_dist": None,
                "final_waypoint_dist": None, "stopped": False, "infer_ms": []}
    collision = any(bool(a.get("collision")) and _src(a) != "stuck" for a in actions)
    stuck = any(bool(a.get("collision")) and "stuck" in _src(a) for a in actions)
    false_forward = any(bool(a.get("collision")) and _src(a) != "stuck" and
                        int(a.get("action_executed", -1)) == 1 for a in actions)
    xs = [float(o["pose_gt"]["x"]) for o in oracle
          if isinstance(o.get("pose_gt"), dict) and "x" in o["pose_gt"]]
    max_x = max(xs) if xs else None
    wp_min = wp_final = None
    if isinstance(waypoint, dict):
        ds = []
        for o in oracle:
            pg = o.get("pose_gt")
            if isinstance(pg, dict) and "x" in pg and "y" in pg:
                ds.append(((pg["x"] - waypoint["x"]) ** 2 +
                           (pg["y"] - waypoint["y"]) ** 2) ** 0.5)
        wp_min = min(ds) if ds else None
        wp_final = ds[-1] if ds else None
    stopped = int(actions[-1].get("action_executed", -1)) == 0
    if stop_radius is not None:
        # 2026-11-01 起不再要求末动作 SAFE_NEUTRAL：waypoint 在柱阵外侧 ±2.4 m，
        # 相机 60° 视场在真停止点上 40% 的帧一根柱子都看不到，"停"不可观测
        # （worklog 2026-11-01-2130-stop-point-unobservable.md）。改成"到达"。
        success = bool(not collision and wp_final is not None
                       and wp_final <= REACH_RADIUS)
    else:
        reached = wp_min is None or wp_min <= waypoint_radius
        success = bool(not collision and max_x is not None and max_x >= goal_x and reached)
    infer = [float(a["infer_ms"]) for a in actions if a.get("infer_ms") is not None]
    return {"steps": len(actions), "collision": collision, "stuck": stuck,
            "false_forward": false_forward, "success": success,
            "max_x": max_x, "waypoint_min_dist": wp_min,
            "final_waypoint_dist": wp_final, "stopped": stopped, "infer_ms": infer}


def aggregate(episodes):
    done = [e for e in episodes if e.get("steps", 0) > 0]
    n = len(done)
    infer_all = [ms for e in done for ms in e["infer_ms"]]
    infer_warm = [ms for e in done for ms in e["infer_ms"][1:]]  # 首帧含 CUDA 初始化
    dist = {name: 0 for name in ACTION_NAMES}
    for e in done:
        for k, v in e.get("actions", {}).items():
            if 0 <= k < len(ACTION_NAMES):
                dist[ACTION_NAMES[k]] += v

    def rate(key):
        return (sum(1 for e in done if e[key]) / n) if n else None

    return {
        "episodes_done": n,
        "success_rate": rate("success"),
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
           "--out", world, "--template", args.template, "--task", args.task]
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
    split = sm.split_for_gid(gid)
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
    try:
        cp.wait_ready(sim, partition, log_path, args.sim_timeout)
        time.sleep(cp.SIM_SETTLE_S)
        cmd = [sys.executable, COLLECT, "--out", ep_dir,
               "--steps", str(args.steps), "--hold", str(args.hold),
               "--policy", args.policy,
               "--stop-on-collision",
               "--scene-id", str(seed), "--seed", str(seed),
               "--scene-meta", eval_meta_path, "--split", split,
               "--window-from", args.window_from]
        if args.policy == "student":
            cmd += ["--student-ckpt", args.ckpt]
            if args.infer_bin:
                cmd += ["--infer-bin", args.infer_bin]
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
        cp.stop_group(sim, f"eval seed {seed}")

    actions_path = os.path.join(ep_dir, "actions.jsonl")
    oracle_path = os.path.join(ep_dir, "oracle.jsonl")
    actions = load_jsonl(actions_path) if os.path.isfile(actions_path) else []
    oracle = load_jsonl(oracle_path) if os.path.isfile(oracle_path) else []
    summary = summarize_episode(actions, oracle, goal_x, meta.get("waypoint"),
                                WAYPOINT_RADIUS, meta.get("stop_radius"))
    summary.update({
        "seed": seed, "geometry_gid": gid, "split": split, "policy": args.policy,
        "task": args.task, "window_from": args.window_from,
        "text_mode": args.text_mode,
        "task_version": meta.get("task_version"),
        "target_color": meta.get("target_color"),
        "instruction": eval_meta.get("instruction"),
        "waypoint": meta.get("waypoint"),
        "waypoint_radius": WAYPOINT_RADIUS,
        "stop_radius": meta.get("stop_radius"),
        "pillars": meta.get("pillar_count"), "goal_x": goal_x,
        "collect_rc": rc, "wall_s": round(time.monotonic() - t0, 1),
        "actions": {},
        "stderr_tail": err.strip().splitlines()[-5:],
        "dedup_key": dedup,
    })
    for a in actions:
        k = int(a.get("action_executed", -1))
        summary["actions"][k] = summary["actions"].get(k, 0) + 1
    return summary, None


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None, help="学生 checkpoint（--policy student 必填）")
    p.add_argument("--policy", choices=("student", "forward"), default="student",
                   help="student=模型闭环；forward=盲前进基线（T7）")
    p.add_argument("--episodes", type=int, default=None,
                   help="成功评测的 episode 数（缺省：给了 --seeds 用其个数，否则 3）")
    p.add_argument("--steps", type=int, default=80, help="每集最大步数")
    p.add_argument("--hold", type=float, default=1.0, help="每动作仿真时间秒")
    p.add_argument("--window-from", choices=("dispatch", "observation"),
                   default="dispatch",
                   help="动作窗口起点：dispatch=派发位姿（默认，R01）；"
                        "observation=观测位姿（旧语义，仅诊断/兼容旧数据）")
    p.add_argument("--text-mode", choices=TEXT_MODES, default="keep",
                   help="L2 消融：keep=原指令；blank=空文本；shuffle=把 left/right 反过来")
    p.add_argument("--task", choices=("base", "l1"), default="base",
                   help="base=原柱阵；l1=彩色 landmark + instruction/waypoint")
    p.add_argument("--split", choices=("train", "val", "test"), default="test")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--max-seeds", type=int, default=500,
                   help="最多扫描多少个 seed 来凑够 --episodes（test split 约 10%%）")
    p.add_argument("--seeds", default="",
                   help="固定 seed 列表（逗号分隔或 a:b）；T7 三个基线用同一批场景")
    p.add_argument("--out", required=True)
    p.add_argument("--template", default=DEFAULT_TEMPLATE)
    p.add_argument("--infer-bin", default=None)
    p.add_argument("--goal-margin", type=float, default=1.0,
                   help="成功 = 无碰撞且 max_x >= 最远柱 x + 该余量")
    p.add_argument("--sim-timeout", type=float, default=120.0)
    p.add_argument("--episode-timeout", type=float, default=600.0)
    p.add_argument("--keep-work", action="store_true", help="保留 world/每集原始文件")
    args = p.parse_args()

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
    for name in ("goal_margin", "sim_timeout", "episode_timeout"):
        value = getattr(args, name)
        if not (math.isfinite(value) and value > 0):
            p.error(f"--{name.replace('_', '-')} 必须是有限正数")
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
        if args.text_mode != "keep" and args.task != "l1":
            p.error("--text-mode 只对 --task l1 有意义")
        args.infer_bin = infer_bin  # 让 run_episode 与 report 用同一个路径
    out = os.path.abspath(args.out)
    if os.path.isdir(out) and os.listdir(out):
        p.error(f"--out {out} 非空；请用新目录")
    os.makedirs(out, exist_ok=True)
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
    results, seen_gids, failures, skips = [], set(), [], []
    if not seed_list:
        seed_list = list(range(args.base_seed, args.base_seed + args.max_seeds))
    started = time.monotonic()
    try:
        for seed in seed_list:
            if len(results) >= args.episodes:
                break
            try:
                summary, skip = run_episode(seed, args, workdir, seen_gids)
            except Exception as e:  # noqa: BLE001 - 单集失败要记录，不能拖垮整轮
                print(f"[seed {seed}] 失败: {type(e).__name__}: {e}", flush=True)
                failures.append({"seed": seed,
                                 "error": f"{type(e).__name__}: {e}"})
                continue
            if skip:
                print(f"[seed {seed}] 跳过: {skip}", flush=True)
                skips.append({"seed": seed, "reason": skip})
                continue
            if summary["collect_rc"] != 0 or summary["steps"] == 0:
                # 采集中途失败不是「学生飞得差」，不能计进 SR/碰撞率。
                failures.append({"seed": seed, "rc": summary["collect_rc"],
                                 "stderr_tail": summary["stderr_tail"]})
                print(f"[seed {seed}] 采集失败 rc={summary['collect_rc']}，不计入指标",
                      flush=True)
                continue
            seen_gids.add(summary["dedup_key"])
            results.append(summary)
            print(f"[seed {seed}] steps={summary['steps']} "
                  f"collision={summary['collision']} success={summary['success']} "
                  f"max_x={summary['max_x']} rc={summary['collect_rc']}", flush=True)
    finally:
        cp.cleanup_all()

    agg = aggregate(results)
    stop_radius_used = next((e.get("stop_radius") for e in results
                             if e.get("stop_radius") is not None), None)
    report = {"schema": "uavrgb.eval.v1", "policy": args.policy,
              "task": args.task, "window_from": args.window_from,
              "text_mode": args.text_mode, "text_dim": text_dim,
              # 成功语义跟着数据走：L1 是 goal-reach「末态在 waypoint 1.5 m 内」，
              # 旧数据是 route-choice「越过终点」。不写死会让新旧 SR 被直接混报。
              # 2026-11-01 之前的 L1 产物是 goal_stop（含 SAFE 判定），两者不可比。
              "success_criterion": ("goal_reach" if stop_radius_used is not None
                                    else "route_choice"),
              "goal_margin": args.goal_margin,
              "waypoint_radius": WAYPOINT_RADIUS,
              "stop_radius": stop_radius_used,
              "geometry_version": sm.SPLIT_SALT,
              "ckpt": os.path.abspath(args.ckpt) if args.ckpt else None,
              "ckpt_sha256": sha256_file(args.ckpt) if args.ckpt else None,
              "infer_bin": infer_bin,
              "infer_bin_sha256": (sha256_file(infer_bin)
                                   if args.policy == "student" else None),
              "split": args.split, "episodes_requested": args.episodes,
              "steps": args.steps, "hold_s": args.hold,
              "seeds": [e["seed"] for e in results],
              "wall_s": round(time.monotonic() - started, 1),
              **agg, "failed": failures, "skipped": skips,
              "per_episode": [{k: v for k, v in e.items() if k != "infer_ms"}
                              for e in results]}
    path = os.path.join(out, "eval.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n== 闭环评测 ==")
    print(f"episodes={agg['episodes_done']}/{args.episodes}  "
          f"SR={agg['success_rate']}  collision={agg['collision_rate']}  "
          f"stuck={agg['stuck_rate']}  "
          f"false_forward={agg['false_forward_rate']}")
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
