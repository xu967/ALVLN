#!/usr/bin/env python3
"""E6 闭环评测：学生在未知场景里自己飞，记录 SR / 碰撞率 / 误前向 / 推理延时。

每个 episode：生成固定 seed 的世界 -> 只保留指定 split 的几何 -> 起独立
GZ_PARTITION 的 gz sim -> collect.py --policy student --stop-on-collision ->
解析 actions.jsonl / oracle.jsonl。评测场景按几何组隔离，不会和训练几何重复。

用法（需要 .venv-tools 的 PIL；uavrgb_infer 已构建）：
  python3 sim/tools/eval_closed_loop.py --ckpt <ckpt.bin> --episodes 3 \
      --out /tmp/eval1 --split test
"""
import argparse
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
ACTION_NAMES = ("SAFE_NEUTRAL", "FORWARD_SLOW", "YAW_LEFT", "YAW_RIGHT")

sys.path.insert(0, TOOLS)
import collect_parallel as cp  # noqa: E402
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


def summarize_episode(actions, oracle, goal_x):
    """单集指标；纯函数，便于单测。actions/oracle 是 jsonl 行。"""
    if not actions:
        return {"steps": 0, "collision": False, "false_forward": False,
                "success": False, "max_x": None, "infer_ms": []}
    collision = any(bool(a.get("collision")) for a in actions)
    false_forward = any(bool(a.get("collision")) and
                        int(a.get("action_executed", -1)) == 1 for a in actions)
    xs = [float(o["pose_gt"]["x"]) for o in oracle
          if isinstance(o.get("pose_gt"), dict) and "x" in o["pose_gt"]]
    max_x = max(xs) if xs else None
    success = bool(not collision and max_x is not None and max_x >= goal_x)
    infer = [float(a["infer_ms"]) for a in actions if a.get("infer_ms") is not None]
    return {"steps": len(actions), "collision": collision,
            "false_forward": false_forward, "success": success,
            "max_x": max_x, "infer_ms": infer}


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


def run_episode(seed, args, workdir):
    """返回 (summary 或 None, 跳过原因)。None 表示该 seed 不在目标 split。"""
    world = os.path.join(workdir, f"scene_{seed}.sdf")
    meta_path = os.path.join(workdir, f"scene_{seed}.json")
    r = subprocess.run(
        [sys.executable, SCENE_GEN, "--seed", str(seed), "--scene-id", str(seed),
         "--out", world, "--template", args.template],
        capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"scene_gen seed={seed} 失败：{r.stderr.strip()}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    gid = sm.geometry_gid(meta["pillars"])
    split = sm.split_for_gid(gid)
    if split != args.split:
        return None, f"split={split}"

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
               "--policy", "student", "--student-ckpt", args.ckpt,
               "--stop-on-collision",
               "--scene-id", str(seed), "--seed", str(seed),
               "--scene-meta", meta_path, "--split", split]
        if args.infer_bin:
            cmd += ["--infer-bin", args.infer_bin]
        env = cp.gz_env(partition)
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            out, err = proc.communicate(timeout=args.episode_timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, err = proc.communicate()
            rc = "timeout"
    finally:
        cp.stop_group(sim, f"eval seed {seed}")

    actions_path = os.path.join(ep_dir, "actions.jsonl")
    oracle_path = os.path.join(ep_dir, "oracle.jsonl")
    actions = load_jsonl(actions_path) if os.path.isfile(actions_path) else []
    oracle = load_jsonl(oracle_path) if os.path.isfile(oracle_path) else []
    summary = summarize_episode(actions, oracle, goal_x)
    summary.update({
        "seed": seed, "geometry_gid": gid, "split": split,
        "pillars": meta.get("pillar_count"), "goal_x": goal_x,
        "collect_rc": rc, "wall_s": round(time.monotonic() - t0, 1),
        "actions": {},
        "stderr_tail": err.strip().splitlines()[-5:],
    })
    for a in actions:
        k = int(a.get("action_executed", -1))
        summary["actions"][k] = summary["actions"].get(k, 0) + 1
    return summary, None


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="学生 checkpoint")
    p.add_argument("--episodes", type=int, default=3, help="成功评测的 episode 数")
    p.add_argument("--steps", type=int, default=80, help="每集最大步数")
    p.add_argument("--hold", type=float, default=1.0, help="每动作仿真时间秒")
    p.add_argument("--split", choices=("train", "val", "test"), default="test")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--max-seeds", type=int, default=500,
                   help="最多扫描多少个 seed 来凑够 --episodes（test split 约 10%）")
    p.add_argument("--out", required=True)
    p.add_argument("--template", default=DEFAULT_TEMPLATE)
    p.add_argument("--infer-bin", default=None)
    p.add_argument("--goal-margin", type=float, default=1.0,
                   help="成功 = 无碰撞且 max_x >= 最远柱 x + 该余量")
    p.add_argument("--sim-timeout", type=float, default=120.0)
    p.add_argument("--episode-timeout", type=float, default=600.0)
    p.add_argument("--keep-work", action="store_true", help="保留 world/每集原始文件")
    args = p.parse_args()

    if args.episodes < 1 or args.steps < 1 or args.max_seeds < args.episodes:
        p.error("episodes/steps 必须 >=1，max-seeds >= episodes")
    if not (math.isfinite(args.hold) and args.hold > 0):
        p.error("--hold 必须是有限正数")
    if not os.path.isfile(args.ckpt):
        p.error(f"checkpoint 不存在: {args.ckpt}")
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
    results, seen_gids, failures, seed = [], set(), [], args.base_seed
    started = time.monotonic()
    try:
        while len(results) < args.episodes and seed < args.base_seed + args.max_seeds:
            try:
                summary, skip = run_episode(seed, args, workdir)
            except Exception as e:  # noqa: BLE001 - 单集失败要记录，不能拖垮整轮
                print(f"[seed {seed}] 失败: {type(e).__name__}: {e}", flush=True)
                seed += 1
                continue
            if skip:
                seed += 1
                continue
            if summary["collect_rc"] != 0 or summary["steps"] == 0:
                # 采集中途失败不是「学生飞得差」，不能计进 SR/碰撞率。
                failures.append({"seed": seed, "rc": summary["collect_rc"],
                                 "stderr_tail": summary["stderr_tail"]})
                print(f"[seed {seed}] 采集失败 rc={summary['collect_rc']}，不计入指标",
                      flush=True)
                seed += 1
                continue
            if summary["geometry_gid"] in seen_gids:
                print(f"[seed {seed}] 跳过：几何 {summary['geometry_gid']} 已评测", flush=True)
                seed += 1
                continue
            seen_gids.add(summary["geometry_gid"])
            results.append(summary)
            print(f"[seed {seed}] steps={summary['steps']} "
                  f"collision={summary['collision']} success={summary['success']} "
                  f"max_x={summary['max_x']} rc={summary['collect_rc']}", flush=True)
            seed += 1
    finally:
        cp.cleanup_all()

    agg = aggregate(results)
    report = {"schema": "uavrgb.eval.v1", "ckpt": os.path.abspath(args.ckpt),
              "split": args.split, "episodes_requested": args.episodes,
              "steps": args.steps, "hold_s": args.hold,
              "wall_s": round(time.monotonic() - started, 1),
              **agg, "failed": failures,
              "per_episode": [{k: v for k, v in e.items() if k != "infer_ms"}
                              for e in results]}
    path = os.path.join(out, "eval.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n== 闭环评测 ==")
    print(f"episodes={agg['episodes_done']}/{args.episodes}  "
          f"SR={agg['success_rate']}  collision={agg['collision_rate']}  "
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
