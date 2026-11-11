#!/usr/bin/env python3
"""Gazebo DAgger 流水线（PLAN W2 第 2 层；DISTILL §2.5 协议）：round0 = 截断臂（教师集 + 学生 ckpt）
→ 第 k 轮学生自采（A* 影子教师打标、β 混合）→ 审计 → 专家不一致率 → 全部轮次并集从零重训
→ more_demos 对照臂（教师再采帧数对齐的示范，同开关）→ 20 seed 闭环（dagger / more_demos / 基线行复用或重评）
→ 60 seed 清单（只评最终臂）→ 汇总。守护进程跑。

运行（在自己的 worktree 根；Python 用主仓库 .venv-tools）：
  export TMPDIR=$DATA_ROOT/uav_rgb/tmp/agent10-dagger-gz/tmp UAV_RGB_BUILD_DIR=$DATA_ROOT/uav_rgb/exp-dagger-gz-20261109/bin
  OUT=$DATA_ROOT/uav_rgb/exp-dagger-gz-20261109/full-$(date +%m%d-%H%M); mkdir -p "$OUT"
  setsid nohup $UAV_RGB_PYTHON tools/dagger_gz_pipeline.py --out "$OUT" > "$OUT/run.log" 2>&1 < /dev/null &
  echo $! > "$OUT/run.pid"; ps -o pid,pgid,etime,cmd -p "$(cat "$OUT/run.pid")"    # 停：kill -- -<PGID>
看进度：tail -f $OUT/run.log；每步 $OUT/status.json；结束落 $OUT/DONE（rc + 时间 + 最后一步）。
续跑：同一 --out 重跑；产物齐的步骤跳过（采集 = collection_manifest.json；训练 = run/metrics.jsonl 末行 done:；
评测 = eval.json 且 failed_count==0 且 episodes_done>0 且不是 partial）。半成品的采集/训练目录要手动删（脚本会指出是哪个），
半成品评测目录自动挪到 <目录>.failed-<时间> 再重评。
--dry-run 只打印命令、不建目录；--stop-after-preflight 只查缺件、写 inputs.json 与两份 dynamics_profile。
交付权重默认 calswa（每次训练后加一步 average_ckpt：末 10 轮平均 + 校准，要 GPU，见 e1_pipeline.CFG["deliver"]）；E1_DELIVER=best 回到 ckpt_best。
smoke（--epochs 2 不满 10 轮，calswa 取不到窗口，要带 E1_DELIVER=best）：
  E1_DELIVER=best … tools/dagger_gz_pipeline.py --rounds 1 --eps-per-round 4 --steps 12 --instances 2 --eval-seeds 1,113 --epochs 2 --baselines trunc --eval60-arms ""
改参数只改 CFG，不改函数。
预计用时（4 实例、RTF 锁 1.0、软件渲染，2080 Ti）：每轮采 100 集 × ≤40 步约 35 min + 训练 30 轮约 8 min；
more_demos 同量级；20 seed × 80 步每臂约 40 min（eval3 实测 36 min），两臂并行；60 seed 最终臂约 2 h；
3 轮全程约 6–7 h（基线行全部重评再加约 2 h）。
"""
import argparse
import glob
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("DATA_ROOT", "/root/rivermind-data")
BUILD = os.environ.get("UAV_RGB_BUILD_DIR", os.path.join(REPO, "build", "check"))   # 冻结快照：各臂同一 train/infer
PY = sys.executable
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "sim", "tools"))
import e1_pipeline  # noqa: E402  评测 seed 单源、GZ_ENV、训练配置的出处
import eval_seeds as es  # noqa: E402  60 seed 清单单源
import pipeline_common as pc  # noqa: E402
import split_manifest as sm  # noqa: E402
import cf_sitl_exec  # noqa: E402  当前 e1_mini.json 的 dynamics_profile，与 collect.py 同一函数
import stats_report  # noqa: E402  McNemar / 配对结果

D = f"{DATA_ROOT}/uav_rgb"
TRUNC_CKPT = f"{D}/e1-pipeline-20261107-0417-eval3/ckpt_trunc_distill.bin"   # = tmp/agent6-distill/trunc/run/ckpt_best.bin（sha ee37c41f）
EVAL4 = f"{D}/e1-pipeline-20261107-0417-eval4"   # 主会话 FIX-A 合并后起的基准重评；目录名按实际为准

# ───────────────────────────── 配置区（改这里） ─────────────────────────────
CFG = dict(
    # 第 0 轮 = 截断臂（PLAN 11-09 决策 2；sgvis 不用）。col/ 没有顶层 collection_manifest.json，
    # 采集开关（hold / window_from / dynamics / vehicle / tiebreak / stop_at_goal）从每集 episode_manifest 聚合读并强制一致
    round0_collection=f"{D}/tmp/agent6-distill/trunc/col",
    round0_ckpt=TRUNC_CKPT,
    # DAgger 轮次：Ross 2011 β_i=p^i，第 1 轮一半步改执行教师（学生刚训出来还不稳），之后全学生执行（DISTILL §2.5）
    rounds=3,
    betas=(0.5, 0.0, 0.0),
    eps_per_round=100,         # 4 实例 × 25；≈ round0 300 集的 1/3，3 轮后并集翻倍
    steps=40,                  # 与 round0 的 requested_steps 相同；round0 是截断集 ⇒ 新轮次开 --stop-at-goal，过线即止
    round_seeds=(5100, 6100, 7100),   # 每轮 seed 段不重叠、不与 0417/sgvis 的 3000–3399 重叠；按 gid 排除 20 个 test seed 的几何后顺延补足
    more_demos_seed=8100,      # 对照臂教师再采的 seed 段
    instances=4,               # gz agent 实测 4 实例 6.8 核（cgroup 12 核）；smoke 用 2
    # round0 是 lib v1 采的（dynamics_profile cf_sitl_v1:bd0dcd43），新轮次是 v3.1（683e7400）：标签来自 A* 只依赖位姿、
    # 不依赖动力学，混包混的是状态分布；more_demos 对照臂同样混包，两臂公平。关掉则 preflight 遇到不一致直接退出
    allow_mixed_dynamics=True,
    no_aux_cameras=True,       # 采集不要 chase/top 相机：每步快 17%
    collect_min_ok=0.95,       # 同 e1_pipeline：偶发丢首条起飞指令属正常；低于 95% 说明环境坏了，停
    audit_ignore=("class_share", "safe_collision"),   # 同 e1_pipeline（X3 口径项）；DAgger 集的 class_share 按 action_executed 统计
    # 训练 = e1_pipeline.CFG 的冻结配置（BRIEF_ALGO §0.2）：30 轮 / batch 128 / lr 1e-3 / seed 0 / class-weights auto / --drop-collision
    drop_collision=True, epochs=30, batch=128, lr=1e-3, seed=0, class_weights="auto",
    # 交付权重（W1-5）：三项都读 e1_pipeline.CFG（单源，环境变量 E1_DELIVER / E1_GPU 两条流水线共用；取值含义与 UUID 来历写在那里）
    deliver=e1_pipeline.CFG["deliver"],   # calswa = 每轮与 more_demos 训练后末 avg_last_k 轮平均 + 校准（run/ckpt_calswa.bin），下一轮学生自采、20/60 seed 评测都用它；
                                          # best = ckpt_best.bin（改前行为）。--epochs < avg_last_k 时 preflight 报错；smoke（--epochs 2）要带 E1_DELIVER=best
    avg_last_k=e1_pipeline.CFG["avg_last_k"],
    gpu=e1_pipeline.CFG["gpu"],           # 校准推理钉的那张卡（CUDA_VISIBLE_DEVICES）
    # 评测：20 seed 从 e1_pipeline.CFG 读（单源，不复制字面量）；80 步；每臂 1 gz
    eval_seeds=e1_pipeline.CFG["eval_seeds"],
    eval_steps=e1_pipeline.CFG["eval_steps"],
    corridor_half_width=float(os.environ.get("E1_CORRIDOR_HALF_WIDTH", "3.0")),
                               # 柱阵评测的横向边界（m），等于 scene_gen.Y_RANGE；11-10 发现不加它时 20/20 里有 8 集是绕出柱阵；
                               # 设 0 回到旧口径。与 e1_pipeline.CFG 同名同环境变量，两条流水线的评测口径一致。
                               # 基线行复用要求 metric_version 相同：带走廊后 EVAL4（无走廊）的基线不再复用，会重评
    eval_parallel=2,           # 同时起的评测数：BRIEF §0 每个 agent ≤2 个 gz；正式跑 eval4 结束后可改 3
    gz_total_max=6,            # 起 gz 前全机 `gz sim` 总数超过它就等（BRIEF §0 闸门）；0 = 不等
    eval60_count=60,           # 60 seed 清单 = eval_seeds.EVAL_SEEDS_BASE_60 的前 eval60_count 个（用户 2026-11-09 定的 E1 口径）
    eval60_arms=("dagger",),   # 60 seed 只评最终臂（约 2 h/臂/gz）；"" 跳过
    # 基线行 名 -> (policy, ckpt, 可复用的 eval 目录)。复用四条件：seeds 同 + metric_version 同 + 插件 sha 同
    # （eval 目录或其父目录的 inputs.json / sha256sum 格式 inputs.sha256）+ infer sha 同（学生臂）；e1_mini.json 用 per_episode.dynamics_profile 比。缺一就重评
    baselines={
        "teacher": ("teacher", None, f"{EVAL4}/eval-teacher"),
        "forward": ("forward", None, f"{EVAL4}/eval-forward"),
        "x3-old": ("student", e1_pipeline.CFG["old_x3_ckpt"], f"{EVAL4}/eval-student-x3-old"),
        "bc": ("student", f"{D}/e1-pipeline-20261107-0417/run/ckpt_best.bin", f"{EVAL4}/eval-student-e1-new"),
        "trunc": ("student", TRUNC_CKPT, f"{EVAL4}/eval-student-e1-trunc"),
        "sgvis": ("student", f"{D}/e1-pipeline-20261107-1620-sgvis/run/ckpt_best.bin", f"{EVAL4}/eval-student-e1-sgvis"),
    },
    plugin_dir=e1_pipeline.CFG["plugin_dir"],   # 真库插件目录（lib v3.1）；sha 记进 inputs.json
    episode_timeout=600.0, sim_timeout=120.0,
)
ACTIONS = ("SAFE_NEUTRAL", "FORWARD_SLOW", "YAW_LEFT", "YAW_RIGHT")
# ──────────────────────────────────────────────────────────────────────────

DRY = False
CHILDREN = set()   # 在跑的子进程：SIGTERM 时转发，DONE 写真实 rc
log, read_json, write_json = pc.log, pc.read_json, pc.write_json
# run/sh/mark/need 与 pipeline_common 同名不同义，不能换：run 的子进程留在本进程组、由 _on_term 转发 SIGTERM（pc.run 另起进程组）；
# mark 合并同一步的旧字段（preflight 的 dynamics_profiles 靠它保留）；need 在 dry-run 下不打印提醒


def run(cmd, env_extra=None, log_path=None):
    """跑一条命令；stdout/stderr 进 run.log（或 log_path）；返回退出码，由调用方决定要不要停。"""
    log("$ " + " ".join(cmd) + (f"  > {log_path}" if log_path else ""))
    if DRY:
        return 0
    env = dict(os.environ, **(env_extra or {}))
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)   # 子命令自己建 --out，旁边的日志目录得先有
    f = open(log_path, "ab") if log_path else None
    proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT if f else None)
    CHILDREN.add(proc)
    try:
        return proc.wait()
    finally:
        CHILDREN.discard(proc)
        if f:
            f.close()


def sh(cmd, **kw):
    rc = run(cmd, **kw)
    if rc != 0:
        raise RuntimeError(f"退出码 {rc}: {' '.join(os.path.basename(c) for c in cmd[:3])}")


def mark(out, step, state, **extra):
    if DRY:
        return
    path = os.path.join(out, "status.json")
    status = read_json(path) if os.path.isfile(path) else {}
    entry = dict(status.get(step) or {}, state=state, at=time.strftime("%F %T"), **extra)   # 同一步多次 mark 合并（preflight 的 profile 不丢）
    if state in ("running", "ok"):
        entry.pop("error", None)   # 续跑成功后不留上一次的失败原因
    status[step] = entry
    write_json(path, status)


def need(path, hint):
    if not DRY and not os.path.exists(path):
        raise RuntimeError(f"缺 {path}（{hint}）")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gz_env():
    return dict(e1_pipeline.GZ_ENV, UAVRGB_CF_SITL_PLUGIN_DIR=CFG["plugin_dir"])


def wait_gz_gate():
    """BRIEF §0 闸门：全机 gz sim 总数 > gz_total_max 就等。shortcut: 只数已出现的进程，刚起的 gz 几秒后才可见"""
    while not DRY and CFG["gz_total_max"]:
        r = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True)
        n = sum(1 for line in r.stdout.splitlines() if "gz sim" in line)
        if n <= CFG["gz_total_max"]:
            return
        log(f"闸门：全机 gz sim {n} > {CFG['gz_total_max']}，等 30 s")
        time.sleep(30)


# ───────────────────────────── 续跑判定（BRIEF §3 第 4 条） ─────────────────────────────
def collect_done(col):
    return os.path.isfile(os.path.join(col, "collection_manifest.json"))


def train_done(run_dir):
    """训完 = run/metrics.jsonl 末行是 done: 状态行；只看 ckpt 文件会把中途被杀的 run 当成功。"""
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.isfile(path):
        return False
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f.read().splitlines() if line.strip()]
    try:
        return bool(lines) and str(json.loads(lines[-1]).get("message", "")).startswith("done:")
    except ValueError:
        return False


def eval_done(edir):
    """评完 = pipeline_common.eval_finished：eval.json 在、failed_count == 0、episodes_done > 0、不是中断写出的 partial。"""
    return pc.eval_finished(edir)


# ───────────────────────────── round0 / seed / 命令行 ─────────────────────────────
def episode_dirs(col):
    """collect_parallel 布局下可进训练集的集目录（train/val/test；failed/ 不算）。"""
    return sorted(d.rstrip("/") for s in ("train", "val", "test") for d in glob.glob(f"{col}/{s}/*/")
                  if os.path.isfile(os.path.join(d, "episode_manifest.json")))


def round0_settings(col):
    """round0 的采集开关：有 collection_manifest.json 就以它为准，否则从每集 manifest 聚合（trunc/col 没有顶层清单）。
    任一键缺失或不一致 → 退出（preprocess 的 STRICT_KEYS 会在几小时后才炸）。"""
    dirs = episode_dirs(col)
    if not dirs:
        raise RuntimeError(f"{col} 下没有带 episode_manifest.json 的 train/val/test 集目录")
    mans = [read_json(os.path.join(d, "episode_manifest.json")) for d in dirs]

    def uniform(key, default=None):
        vals = {json.dumps(m.get(key, default), sort_keys=True) for m in mans}
        if len(vals) != 1:
            raise RuntimeError(f"round0 各集的 {key} 不一致：{sorted(vals)}（不同口径不能当同一轮）")
        v = json.loads(vals.pop())
        if v is None:
            raise RuntimeError(f"round0 的 {key} 缺失（{col}）")
        return v

    if uniform("teacher") != "plan_action" or uniform("planner") != "astar":
        raise RuntimeError("round0 的 teacher/planner 必须是 plan_action/astar：DAgger 纠正标签来自同一个 A* 教师")
    s = dict(hold=uniform("hold_s"), window_from=uniform("window_from"), dynamics=uniform("dynamics"),
             vehicle=uniform("vehicle"), tiebreak=uniform("teacher_tiebreak", "legacy"),
             # 采集时过线即止，或离线按成功线截断（trunc/col 的 truncated_at_success_line）⇒ 新轮次同样过线即止
             stop_at_goal=any(m.get("stopped_on_goal") or m.get("truncated_at_success_line") for m in mans),
             dynamics_profiles=sorted({m.get("dynamics_profile") for m in mans} - {None}),
             gids=sorted({m.get("geometry_gid") for m in mans} - {None}),
             episodes=len(mans), steps=sum(int(m.get("num_steps") or 0) for m in mans), dirs=dirs)
    cm = os.path.join(col, "collection_manifest.json")
    if os.path.isfile(cm):
        c = read_json(cm)
        s.update(hold=c["hold_s"], window_from=c["window_from"], dynamics=c["dynamics"], vehicle=c["vehicle"],
                 stop_at_goal=bool(c.get("stop_at_goal")), tiebreak=c.get("teacher_tiebreak") or "legacy")
    return s


def test_gids(seeds_csv):
    return {sm.group_id_for_task(int(s), "base") for s in seeds_csv.split(",") if s.strip()}


def pick_round_seeds(base, count, exclude_gids):
    """从 base 起顺序取 count 个 seed，几何落在 exclude_gids（评测几何）里的跳过并顺延补足。"""
    seeds, seed = [], base
    while len(seeds) < count:
        if seed > base + 100_000:
            raise RuntimeError(f"从 {base} 扫了 100000 个 seed 仍凑不齐 {count} 个")
        if sm.group_id_for_task(seed, "base") not in exclude_gids:
            seeds.append(seed)
        seed += 1
    return seeds


def collect_cmd(col, seeds, settings, instances, steps, policy="teacher", ckpt=None, beta=0.0, dagger_round=0):
    """collect_parallel 命令行：开关从 round0 继承；student 是 DAgger 自采（影子教师 A* 打标、β 混合）。"""
    cmd = [PY, f"{REPO}/sim/tools/collect_parallel.py", "--instances", str(instances),
           "--seeds", ",".join(map(str, seeds)), "--steps", str(steps), "--task", "base", "--keep-worlds",
           "--out", col, "--dynamics", settings["dynamics"], "--vehicle", settings["vehicle"],
           "--hold", str(settings["hold"]), "--window-from", settings["window_from"]]
    if CFG["no_aux_cameras"]:
        cmd.append("--no-aux-cameras")
    if settings["stop_at_goal"]:
        cmd.append("--stop-at-goal")
    if settings["tiebreak"] != "legacy":
        cmd += ["--tiebreak", settings["tiebreak"]]
    if policy == "student":
        cmd += ["--policy", "student", "--student-ckpt", ckpt]
        if beta:
            cmd += ["--beta", str(beta)]   # 0 是 collect.py 默认值，不传
        cmd += ["--dagger-round", str(dagger_round), "--infer-bin", f"{BUILD}/uavrgb_infer"]
    return cmd


def more_demos_n(new_frames, frames_per_episode, instances):
    """对照臂教师再采的集数：按 DAgger 各轮新增帧数 / round0 每集平均帧数，补到实例数整数倍。
    按采集帧（含碰撞步）对齐：学生集碰撞步更多、--drop-collision 后有效帧更少，对照臂拿到的有效帧只多不少。"""
    if frames_per_episode <= 0:
        raise RuntimeError("round0 每集平均帧数必须 > 0")
    n = math.ceil(new_frames / frames_per_episode)
    return math.ceil(n / instances) * instances if n else 0


def reusable(edir, seeds, metric_version, plugin_sha, infer_sha, profile):
    """基线行能否直接用现成 eval.json：四项缺一不复用。返回 (bool, 原因)。"""
    if not eval_done(edir):
        return False, "没有完整的 eval.json（failed_count==0 且 episodes_done>0 且不是 partial）"
    e = read_json(os.path.join(edir, "eval.json"))
    if set(e.get("seeds") or []) != set(seeds):
        return False, "seeds 不同"
    if e.get("metric_version") != metric_version:
        return False, f"metric_version {e.get('metric_version')} ≠ {metric_version}"
    got = {ep.get("dynamics_profile") for ep in e.get("per_episode") or []}
    if got != {profile}:
        return False, f"dynamics_profile {sorted(map(str, got))} ≠ {profile}（e1_mini.json 变了）"
    if infer_sha is not None and e.get("infer_bin_sha256") != infer_sha:
        return False, "uavrgb_infer sha 不同"
    got_plugin = recorded_plugin_sha(edir)
    if got_plugin is None:
        return False, "没有 inputs.json / inputs.sha256 记录插件 sha（旧 eval 目录按 lib 不同处理）"
    return (got_plugin == plugin_sha), ("ok" if got_plugin == plugin_sha else "libCfSitlSystem.so sha 不同")


def recorded_plugin_sha(edir):
    """eval.json 不记插件 sha，只能看 eval 目录或其父目录旁的记录：本流水线写 inputs.json（键 libCfSitlSystem.so），
    主会话的 eval 目录写 sha256sum 格式的 inputs.sha256（eval3-gpucheck-20261109）。两种都认；都没有返回 None。"""
    for d in (edir, os.path.dirname(os.path.abspath(edir))):
        p = os.path.join(d, "inputs.json")
        if os.path.isfile(p):
            return read_json(p).get("libCfSitlSystem.so")
        p = os.path.join(d, "inputs.sha256")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2 and os.path.basename(parts[-1]) == "libCfSitlSystem.so":
                        return parts[0]
    return None


def disagreement_stats(col):
    """学生分布上的专家不一致率：executed_by=student 的步里 action_expert≠action_executed 的比例（分专家动作）。
    这是离闭环最近的离线代理（DISTILL §2.5 第 3 条）。顺带记到线率与碰撞。"""
    st = dict(episodes=0, steps=0, student_steps=0, expert_steps=0, disagree=0, reached_goal=0,
              collision_episodes=0, collision_steps=0,
              by_expert={a: {"n": 0, "disagree": 0} for a in ACTIONS},
              confusion=[[0] * len(ACTIONS) for _ in ACTIONS])   # 行 = 专家标签，列 = 学生执行
    for d in episode_dirs(col):
        man = read_json(os.path.join(d, "episode_manifest.json"))
        st["episodes"] += 1
        st["reached_goal"] += bool(man.get("stopped_on_goal"))
        hit = False
        with open(os.path.join(d, "actions.jsonl"), encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                st["steps"] += 1
                if r.get("collision") is True:
                    st["collision_steps"] += 1
                    hit = True
                who = r.get("executed_by")
                if who == "expert":
                    st["expert_steps"] += 1
                if who != "student":
                    continue
                a, x = int(r["action_expert"]), int(r["action_executed"])
                st["student_steps"] += 1
                st["by_expert"][ACTIONS[a]]["n"] += 1
                st["confusion"][a][x] += 1
                if a != x:
                    st["disagree"] += 1
                    st["by_expert"][ACTIONS[a]]["disagree"] += 1
        st["collision_episodes"] += hit
    st["disagree_rate"] = st["disagree"] / st["student_steps"] if st["student_steps"] else None
    st["reached_goal_rate"] = st["reached_goal"] / st["episodes"] if st["episodes"] else None
    return st


# ───────────────────────────── 各步 ─────────────────────────────
def step(out, name, fn, *args, **kw):
    """一步：status.json 记 running/ok/failed（失败原因进去再抛出）。"""
    log(f"==== {name} 开始")
    mark(out, name, "running")
    t0 = time.monotonic()
    try:
        r = fn(*args, **kw)
    except BaseException as e:   # noqa: BLE001 - SIGTERM 的 SystemExit 也要记进 status
        mark(out, name, "interrupted" if isinstance(e, SystemExit) else "failed", error=f"{type(e).__name__}: {e}")
        raise
    mark(out, name, "ok", seconds=round(time.monotonic() - t0, 1))
    log(f"==== {name} 完成 {time.monotonic() - t0:.0f}s")
    return r


def preflight(out, settings, baseline_names):
    """开跑前把会在几小时后才炸的缺件一次查完；记 inputs.json（各臂同一套二进制/库/参数）与两份 dynamics_profile。"""
    deliver_check()
    need(f"{BUILD}/uavrgb_train", "先 bash tools/check.sh 构建并复制到快照目录，设 UAV_RGB_BUILD_DIR")
    need(f"{BUILD}/uavrgb_infer", "同上")
    plugin = os.path.join(CFG["plugin_dir"], "libCfSitlSystem.so")
    need(plugin, "真库插件（sim/cf_sitl/gz/README.md）；不要指向 build-gz-stub 桩库")
    need(CFG["round0_ckpt"], "round0 学生 ckpt（截断臂）")
    for name in baseline_names:
        ckpt = CFG["baselines"][name][1]
        if ckpt:
            need(ckpt, f"基线 {name} 的 ckpt")
    params = os.path.join(REPO, "sim", "dynamics", "e1_mini.json")
    current = cf_sitl_exec.CfSitlParams.from_json(params).profile()
    if settings["dynamics_profiles"] != [current]:
        msg = f"round0 dynamics_profile {settings['dynamics_profiles']} ≠ 当前 {current}"
        if not CFG["allow_mixed_dynamics"]:
            raise RuntimeError(msg + "：CFG.allow_mixed_dynamics 关着，不混包")
        log(msg + "：按 CFG.allow_mixed_dynamics 混包（标签来自 A*，只依赖位姿）")
    profiles = {"round0": settings["dynamics_profiles"], "current": current}
    if DRY:
        return profiles
    inputs = {os.path.basename(p): sha256_file(p) for p in (plugin, params, f"{BUILD}/uavrgb_train", f"{BUILD}/uavrgb_infer")}
    lib = os.path.join(os.path.dirname(CFG["plugin_dir"]), "libcf_sitl.so")
    if os.path.isfile(lib):
        inputs["libcf_sitl.so"] = sha256_file(lib)
    inputs.update(round0_ckpt=sha256_file(CFG["round0_ckpt"]), round0_ckpt_path=CFG["round0_ckpt"],
                  dynamics_profiles=profiles, plugin_dir=CFG["plugin_dir"])
    write_json(os.path.join(out, "inputs.json"), inputs)
    return profiles


def collect_into(col, seeds, settings, **kw):
    """采一个集合；manifest 成功率低于 collect_min_ok 就停。collect_parallel 一集失败 rc=1 是常态，按 manifest 判。"""
    if collect_done(col):
        return log(f"跳过采集（已有 {col}/collection_manifest.json）")
    if os.path.isdir(col) and os.listdir(col):
        raise RuntimeError(f"{col} 非空且没有 collection_manifest.json（半成品）：删掉再跑")
    wait_gz_gate()
    run(collect_cmd(col, seeds, settings, CFG["instances"], CFG["steps"], **kw), env_extra=gz_env(),
        log_path=None if DRY else col + ".log")
    if DRY:
        return
    need(os.path.join(col, "collection_manifest.json"), f"collect_parallel 没写出汇总，看 {col}.log 与 {col}/logs/")
    m = read_json(os.path.join(col, "collection_manifest.json"))
    ratio = m["episodes_ok"] / max(m["episodes_requested"], 1)
    log(f"采集 ok {m['episodes_ok']}/{m['episodes_requested']}，{m['steps_completed_total']} 步，"
        f"split train/val/test = {[len(m['splits'][s]) for s in ('train', 'val', 'test')]}，端到端 {m['end_to_end_wall_s']:.0f} s")
    if ratio < CFG["collect_min_ok"]:
        raise RuntimeError(f"采集成功率 {ratio:.2%} < {CFG['collect_min_ok']:.0%}，见 {col}/failed/ 与 logs/")


def audit_dir(col):
    report = os.path.join(col, "audit.json")
    if not os.path.isfile(report):
        run([PY, f"{REPO}/sim/tools/audit_dataset.py", "--dataset", col, "--out", report],
            log_path=None if DRY else col + ".audit.log")   # 有 FAIL 时 rc=1
    if DRY:
        return
    need(report, "audit_dataset 没写出 audit.json")
    failed = [c["name"] for c in read_json(report)["checks"] if c["status"] == "FAIL"]
    if [n for n in failed if n in CFG["audit_ignore"]]:
        log(f"审计 FAIL 但按 CFG.audit_ignore 放行: {[n for n in failed if n in CFG['audit_ignore']]}")
    if [n for n in failed if n not in CFG["audit_ignore"]]:
        raise RuntimeError(f"审计 FAIL: {failed}，见 {report}")   # 不让坏数据进训练
    log("审计通过")


def write_stats(col, path):
    if DRY:
        return
    st = disagreement_stats(col)
    write_json(path, st)
    rate = "-" if st["disagree_rate"] is None else f"{st['disagree_rate']:.3f}"
    by_expert = ", ".join(f"{a}: {v['disagree']}/{v['n']}" for a, v in st["by_expert"].items())
    log(f"专家不一致率 {st['disagree']}/{st['student_steps']} = {rate}；到线 {st['reached_goal']}/{st['episodes']}；"
        f"碰撞集 {st['collision_episodes']}；分专家动作 {{{by_expert}}}")
    return st


def preprocess_into(ds, eps):
    """并集打包：round0 + 各轮自采。动力学档不一致按 CFG.allow_mixed_dynamics 放行（meta 记 dynamics_profiles）。"""
    if not DRY and os.path.isfile(os.path.join(ds, "dataset.uavr")):
        return log(f"跳过打包（已有 {ds}/dataset.uavr）")
    if not DRY and not eps:
        raise RuntimeError("没有可打包的集目录")
    cmd = [PY, f"{REPO}/tools/preprocess.py", "--out", ds]
    if CFG["drop_collision"]:
        cmd.append("--drop-collision")
    if CFG["allow_mixed_dynamics"]:
        cmd.append("--allow-mixed-dynamics")
    sh(cmd + eps, log_path=None if DRY else ds + ".log")


def deliver_check():
    """交付权重配置查错（同 e1_pipeline.deliver_check）：值不认识、calswa 的轮数不够（取不到末 avg_last_k 轮）都在起任何命令之前报错。
    preflight 与 train_into 都调：前者让它在开跑时就失败，不等采集几小时后训练步才炸。"""
    if CFG["deliver"] not in ("calswa", "best"):
        raise RuntimeError(f"CFG deliver={CFG['deliver']!r}（环境变量 E1_DELIVER）只能是 calswa 或 best")
    if CFG["deliver"] == "calswa" and CFG["epochs"] < CFG["avg_last_k"]:
        raise RuntimeError(f"deliver=calswa 要末 {CFG['avg_last_k']} 轮的权重平均，但 epochs={CFG['epochs']} 取不到；"
                           "改 E1_DELIVER=best（用 ckpt_best.bin），或把 --epochs 加到不小于 avg_last_k")


def deliver_ckpt(run_dir):
    """下一轮学生自采与评测用的权重文件。"""
    return os.path.join(run_dir, "ckpt_calswa.bin" if CFG["deliver"] == "calswa" else "ckpt_best.bin")


def calibrate_into(ds, run_dir):
    """deliver=calswa：末 avg_last_k 轮平均 + 校准 → run/ckpt_calswa.bin（真实 uavrgb_infer 跑 GPU）。
    失败（工具非零退出）直接抛出，流水线停下，不退回 ckpt_best。average_ckpt 拒绝覆盖已有产物，
    续跑时 .bin 与旁边的 .json 都在才算做完；只剩 .bin 时再起一次，由工具拒绝并停下，人来处理。"""
    if CFG["deliver"] != "calswa":
        return
    swa = deliver_ckpt(run_dir)
    if os.path.isfile(swa) and os.path.isfile(swa + ".json"):
        return log(f"跳过校准平均：{swa} 已有")
    sh([PY, f"{REPO}/tools/average_ckpt.py", "--run", run_dir, "--data", f"{ds}/dataset.uavr", "--out", swa,
        "--last-k", str(CFG["avg_last_k"]), "--infer-bin", f"{BUILD}/uavrgb_infer"],
       env_extra={"CUDA_VISIBLE_DEVICES": CFG["gpu"]}, log_path=None if DRY else run_dir + ".average.log")
    need(swa, "average_ckpt 退出码 0 但没写出产物")


def train_into(ds, run_dir):
    deliver_check()
    if train_done(run_dir):
        log(f"跳过训练（{run_dir}/metrics.jsonl 已 done）")
        return calibrate_into(ds, run_dir)   # 训练做完、校准平均没做完（被打断）时在这里补
    if os.path.isdir(run_dir) and os.listdir(run_dir):
        raise RuntimeError(f"{run_dir} 非空且没训完：uavrgb_train 的 RunOwner 拒绝复用，删掉再跑")
    need(os.path.join(ds, "dataset.uavr"), "先打包")
    cmd = [f"{BUILD}/uavrgb_train", "--data", f"{ds}/dataset.uavr", "--out", run_dir,
           "--epochs", str(CFG["epochs"]), "--batch", str(CFG["batch"]), "--lr", repr(float(CFG["lr"])),
           "--seed", str(CFG["seed"]), "--class-weights", CFG["class_weights"]]
    if CFG["drop_collision"]:
        cmd.append("--drop-collision")
    if CFG["deliver"] == "calswa":
        cmd.append("--save-every-epoch")   # 平均要用末 avg_last_k 轮的 ckpt_epNN.bin（每个约 1.2 MB）
    sh(cmd, log_path=None if DRY else run_dir + ".log")
    need(f"{run_dir}/ckpt_best.bin", "训练结束但没有 best ckpt，看 run/metrics.jsonl")
    if not DRY and not train_done(run_dir):
        raise RuntimeError(f"{run_dir}/metrics.jsonl 末行不是 done:，训练没正常结束")
    calibrate_into(ds, run_dir)


def run_round(out, k, settings, exclude_gids, prev_ckpt):
    """第 k 轮：学生自采 → 审计 → 不一致率 → 并集打包 → 从零重训（→ calswa 时再做校准平均）；返回本轮的交付权重。"""
    rd = os.path.join(out, f"round_{k}")
    beta = CFG["betas"][min(k - 1, len(CFG["betas"]) - 1)]
    base = CFG["round_seeds"][k - 1] if k <= len(CFG["round_seeds"]) else CFG["round_seeds"][-1] + 1000 * (k - len(CFG["round_seeds"]))
    col = os.path.join(rd, "collect")
    step(out, f"round_{k}.collect", collect_into, col, pick_round_seeds(base, CFG["eps_per_round"], exclude_gids), settings,
         policy="student", ckpt=prev_ckpt, beta=beta, dagger_round=k)
    step(out, f"round_{k}.audit", audit_dir, col)
    step(out, f"round_{k}.stats", write_stats, col, os.path.join(rd, "stats.json"))
    new = [d for j in range(1, k + 1) for d in episode_dirs(os.path.join(out, f"round_{j}", "collect"))]
    if not DRY and not episode_dirs(col):
        raise RuntimeError(f"第 {k} 轮没有可用的自采集（{col}/train|val|test 为空）")
    step(out, f"round_{k}.preprocess", preprocess_into, os.path.join(rd, "ds"), settings["dirs"] + new)
    step(out, f"round_{k}.train", train_into, os.path.join(rd, "ds"), os.path.join(rd, "run"))
    return deliver_ckpt(os.path.join(rd, "run"))


def run_more_demos(out, settings, exclude_gids, rounds):
    """对照臂：教师再采 N 集（N 按 DAgger 新增帧数对齐）→ 审计 → round0 + 新示范打包 → 训练（同样的交付权重配方）；返回交付权重。"""
    md = os.path.join(out, "more_demos")
    if DRY:
        n = CFG["eps_per_round"] * rounds   # 真跑按各轮 collection_manifest 的 steps_completed_total 算
    else:
        new_frames = sum(read_json(os.path.join(out, f"round_{k}", "collect", "collection_manifest.json"))["steps_completed_total"]
                         for k in range(1, rounds + 1))
        per_ep = settings["steps"] / settings["episodes"]
        n = more_demos_n(new_frames, per_ep, CFG["instances"])
        os.makedirs(md, exist_ok=True)
        write_json(os.path.join(md, "plan.json"), {"dagger_new_frames": new_frames, "round0_frames_per_episode": per_ep,
                                                   "episodes": n, "instances": CFG["instances"]})
        log(f"more_demos：DAgger 新增 {new_frames} 帧 / round0 每集 {per_ep:.1f} 帧 → 教师再采 {n} 集")
        if n == 0:
            raise RuntimeError("DAgger 各轮没有新增帧，对照臂无从对齐")
    col = os.path.join(md, "collect")
    step(out, "more_demos.collect", collect_into, col, pick_round_seeds(CFG["more_demos_seed"], n, exclude_gids), settings)
    step(out, "more_demos.audit", audit_dir, col)
    step(out, "more_demos.stats", write_stats, col, os.path.join(md, "stats.json"))
    step(out, "more_demos.preprocess", preprocess_into, os.path.join(md, "ds"), settings["dirs"] + episode_dirs(col))
    step(out, "more_demos.train", train_into, os.path.join(md, "ds"), os.path.join(md, "run"))
    return deliver_ckpt(os.path.join(md, "run"))


def eval_cmd(policy, ckpt, seeds_csv, edir, settings):
    cmd = [PY, f"{REPO}/sim/tools/eval_closed_loop.py", "--policy", policy, "--task", "base", "--split", "test",
           "--seeds", seeds_csv, "--episodes", str(len(seeds_csv.split(","))), "--steps", str(CFG["eval_steps"]),
           "--out", edir, "--sim-timeout", str(CFG["sim_timeout"]), "--episode-timeout", str(CFG["episode_timeout"]),
           "--dynamics", settings["dynamics"], "--vehicle", settings["vehicle"],
           "--hold", str(settings["hold"]), "--window-from", settings["window_from"]]
    if CFG["corridor_half_width"] > 0:
        cmd += ["--corridor-half-width", f"{CFG['corridor_half_width']:g}"]
    if ckpt:
        cmd += ["--ckpt", ckpt, "--infer-bin", f"{BUILD}/uavrgb_infer"]
    return cmd


def evaluate(out, arms, seeds_csv, prefix, settings):
    """arms = [(名, policy, ckpt)]；同时 ≤ eval_parallel 个（各 1 gz）；等齐所有组再查产物（e1_pipeline 4e44e70 的教训）。"""
    running, launched, logs = {}, [], []
    for name, policy, ckpt in arms:
        edir = os.path.join(out, f"{prefix}-{name}")
        if eval_done(edir):
            log(f"跳过 {prefix}-{name}：已有完整 eval.json")
            continue
        if not DRY and os.path.isdir(edir) and os.listdir(edir):
            aside = f"{edir}.failed-{time.strftime('%m%d-%H%M%S')}"
            os.replace(edir, aside)   # eval_closed_loop 拒绝非空 --out；半成品留着看日志
            log(f"{edir} 不完整，挪到 {aside} 后重评")
        if ckpt:
            need(ckpt, f"{name} 的 ckpt")
        while len(running) >= CFG["eval_parallel"]:
            for n, p in list(running.items()):
                if p.poll() is not None:
                    log(f"{prefix}-{n} rc={p.returncode}")
                    running.pop(n)
            time.sleep(5)
        wait_gz_gate()
        cmd, log_path = eval_cmd(policy, ckpt, seeds_csv, edir, settings), f"{edir}.log"
        log("$ " + " ".join(cmd) + f"  > {log_path}")
        launched.append(name)
        if not DRY:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            logs.append(open(log_path, "ab"))
            running[name] = subprocess.Popen(cmd, env=dict(os.environ, **gz_env()), stdout=logs[-1], stderr=subprocess.STDOUT)
            CHILDREN.add(running[name])
    for n, p in running.items():
        rc = p.wait()
        CHILDREN.discard(p)
        log(f"{prefix}-{n} rc={rc}" + ("（有失败/跳过集，见 eval.json 的 failed）" if rc else ""))
    for f in logs:
        f.close()
    bad = [n for n in launched if not DRY and not eval_done(os.path.join(out, f"{prefix}-{n}"))]
    if bad:
        raise RuntimeError(f"{bad} 没评完（eval.json 缺失、failed_count>0 或 partial），看 {out}/{prefix}-<名>.log；"
                           "同一 --out 重跑会把该臂目录挪开重评")


def baselines(out, names, seeds_csv, settings):
    """基线行：能复用就记路径，不能就重评。复用判据以本次 dagger 臂的 eval.json（metric_version / dynamics_profile）为准。"""
    decisions, todo = {}, []
    ref_path = os.path.join(out, "eval20-dagger", "eval.json")
    ref = read_json(ref_path) if not DRY else None
    inputs = read_json(os.path.join(out, "inputs.json")) if not DRY else {}
    for name in names:
        policy, ckpt, reuse_dir = CFG["baselines"][name]
        edir = os.path.join(out, f"eval20-{name}")
        if eval_done(edir):
            decisions[name] = {"source": "self", "path": edir}
            continue
        if DRY:
            ok, why = False, "dry-run 不判复用"
        else:
            want = dict(seeds=[int(s) for s in seeds_csv.split(",")], metric_version=ref["metric_version"],
                        plugin_sha=inputs.get("libCfSitlSystem.so"), infer_sha=inputs.get("uavrgb_infer") if ckpt else None,
                        profile=(ref["per_episode"] or [{}])[0].get("dynamics_profile"))
            ok, why = reusable(reuse_dir, **want) if reuse_dir and os.path.isdir(reuse_dir) else (False, f"没有可复用目录 {reuse_dir}")
        if ok:
            log(f"基线 {name} 复用 {reuse_dir}")
            decisions[name] = {"source": "reused", "path": reuse_dir}
        else:
            log(f"基线 {name} 不复用（{why}），重评")
            decisions[name] = {"source": "re-eval", "path": edir, "reuse_rejected": why}
            todo.append((name, policy, ckpt))
    evaluate(out, todo, seeds_csv, "eval20", settings)
    if not DRY:
        write_json(os.path.join(out, "baselines.json"), decisions)
    return decisions


def seeds60(out):
    """60 seed 评测清单 = eval_seeds.EVAL_SEEDS_BASE_60（用户 2026-11-09 定的 E1 口径：71 个 test 分区几何里的 60 个，含冻结 20）。
    test 分区几何按几何组分区从不进训练，各轮 DAgger / more_demos 采到其中几何也只落各自数据集的 test 分区，所以不按采集集合
    排除（按集合排除只剩 24 个）。out 里已有 eval_seeds60.json 就用它（续跑与主会话预写的清单）。"""
    path = os.path.join(out, "eval_seeds60.json")
    if DRY:
        return "<eval_seeds60>"
    if os.path.isfile(path):
        return ",".join(map(str, read_json(path)["seeds"]))
    seeds = list(es.EVAL_SEEDS_BASE_60[:CFG["eval60_count"]])
    write_json(path, {"seeds": seeds, "gids": [sm.group_id_for_task(s, "base") for s in seeds],
                      "requested": CFG["eval60_count"], "count": len(seeds), "rule": "eval_seeds.EVAL_SEEDS_BASE_60（E1 口径）",
                      "eval_seeds20": CFG["eval_seeds"]})
    return ",".join(map(str, seeds))


def summary(out, rounds, arms, decisions, seeds60_csv):
    """每轮统计 + 各臂 SR/碰撞/乱飞 + 与 dagger 臂的 McNemar；stats_report 的完整表另起子进程（失败不致命）。"""
    rows, per = {}, {}
    sources = {name: os.path.join(out, f"eval20-{name}") for name, _p, _c in arms}
    sources.update({n: d["path"] for n, d in decisions.items()})
    for name, edir in sources.items():
        path = os.path.join(edir, "eval.json")
        if not os.path.isfile(path):
            continue
        e = read_json(path)
        eps = e.get("per_episode") or []
        per[name] = {ep["seed"]: bool(ep.get("success")) for ep in eps}
        rows[name] = {"n": e.get("episodes_done"), "success": sum(1 for ep in eps if ep.get("success")),
                      "SR": e.get("requested_success_rate"), "wilson95": (e.get("wilson_ci") or {}).get("requested"),
                      "collision": sum(1 for ep in eps if ep.get("collision")),
                      "stray": sum(1 for ep in eps if not ep.get("success") and not ep.get("collision")),   # 无碰撞的失败 = 乱飞/超时
                      "stuck": sum(1 for ep in eps if ep.get("stuck")), "false_forward": sum(1 for ep in eps if ep.get("false_forward")),
                      "spl": e.get("spl"), "failed": e.get("failed_count"), "metric_version": e.get("metric_version"),
                      "source": decisions.get(name, {}).get("source", "self"), "path": path}
    pairs = {}
    for name in rows:
        if name != "dagger" and "dagger" in per:
            b, c, n = stats_report.paired_outcomes(per["dagger"], per[name])
            pairs[name] = dict(stats_report.mcnemar(b, c), n_common=n)
    round_rows = {}
    for k in range(1, rounds + 1):
        rd = os.path.join(out, f"round_{k}")
        m, st = os.path.join(rd, "collect", "collection_manifest.json"), os.path.join(rd, "stats.json")
        if os.path.isfile(m) and os.path.isfile(st):
            m, st = read_json(m), read_json(st)
            round_rows[k] = {"beta": CFG["betas"][min(k - 1, len(CFG["betas"]) - 1)], "episodes_ok": m["episodes_ok"],
                             "episodes_requested": m["episodes_requested"], "steps": m["steps_completed_total"],
                             "disagree_rate": st["disagree_rate"], "student_steps": st["student_steps"],
                             "reached_goal": st["reached_goal"], "collision_episodes": st["collision_episodes"],
                             "by_expert": st["by_expert"]}
    eval60 = {}
    for name in CFG["eval60_arms"]:
        p = os.path.join(out, f"eval60-{name}", "eval.json")
        if os.path.isfile(p):
            e = read_json(p)
            eval60[name] = {"n": e.get("episodes_done"), "SR": e.get("requested_success_rate"),
                            "wilson95": (e.get("wilson_ci") or {}).get("requested"), "collision_rate": e.get("collision_rate")}
    md_plan = os.path.join(out, "more_demos", "plan.json")
    res = {"cfg": {k: v for k, v in CFG.items() if k != "baselines"}, "rounds": round_rows, "arms": rows, "mcnemar_vs_dagger": pairs,
           "eval60": eval60, "eval_seeds60": seeds60_csv, "baselines": decisions,
           "more_demos_plan": read_json(md_plan) if os.path.isfile(md_plan) else None,
           "inputs": read_json(os.path.join(out, "inputs.json")) if os.path.isfile(os.path.join(out, "inputs.json")) else None}
    write_json(os.path.join(out, "summary.json"), res)
    lines = ["| round | β | 集 ok/req | 步 | 不一致率（学生步） | 到线 | 碰撞集 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for k, r in round_rows.items():
        dr = "-" if r["disagree_rate"] is None else f"{r['disagree_rate']:.3f} ({r['student_steps']})"
        lines.append(f"| {k} | {r['beta']} | {r['episodes_ok']}/{r['episodes_requested']} | {r['steps']} | {dr} | {r['reached_goal']} | {r['collision_episodes']} |")
    lines += ["", "| 臂 | SR | Wilson95 | 碰撞 | 乱飞/超时 | SPL | McNemar vs dagger (b,c,p) | 来源 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in rows.items():
        pr = pairs.get(name)
        spl = "-" if r["spl"] is None else f"{r['spl']:.2f}"
        mc = "-" if not pr else f"{pr['b']},{pr['c']},p={pr['p_value']:.3f}"
        lines.append(f"| {name} | {r['success']}/{r['n']} | {_ci(r['wilson95'])} | {r['collision']} | {r['stray']} | {spl} | {mc} | {r['source']} |")
    n60 = len([s for s in (seeds60_csv or "").split(",") if s.strip().isdigit()])
    for name, r in eval60.items():
        lines.append(f"| {name} ({n60} seed 清单) | {r['SR']} ({r['n']}) | {_ci(r['wilson95'])} | - | - | - | - | self |")
    if seeds60_csv and n60 < CFG["eval60_count"]:
        lines.append(f"\n注意：独立 test 几何只有 {n60} 个（要 {CFG['eval60_count']}），见 eval_seeds60.json；「60 seed 区间」按 {n60} 个看。")
    lines.append(f"\n口径：SR 分母 = 请求集；{settings_line(res)}；n=20 的 Wilson 半宽约 ±0.2，结论要带区间与配对 p 值。")
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    for line in lines:
        log(line)
    evals = [x for name, r in rows.items() for x in ("--eval", name, "gazebo", r["path"])]
    if evals:
        rc = run([PY, f"{REPO}/tools/stats_report.py", "--out-dir", os.path.join(out, "stats"), *evals],
                 log_path=os.path.join(out, "stats.log"))
        if rc:
            log(f"stats_report rc={rc}（不致命，summary.md 已有主数字），见 {out}/stats.log")
    return res


def _ci(w):
    return "-" if not w else f"[{w[0]:.2f}, {w[1]:.2f}]"


def settings_line(res):
    inp = res.get("inputs") or {}
    prof = inp.get("dynamics_profiles") or {}
    return (f"dynamics_profile round0={prof.get('round0')} 新轮次={prof.get('current')}（混包，标签只依赖位姿）；"
            f"插件 {str(inp.get('libCfSitlSystem.so'))[:8]}，infer {str(inp.get('uavrgb_infer'))[:8]}")


def _on_term(signum, _frame):
    for p in list(CHILDREN):
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)   # collect_parallel / eval_closed_loop 自己收 gz
    raise SystemExit(128 + signum)


def main(argv=None):
    global DRY
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="输出根目录（数据盘，如 $DATA_ROOT/uav_rgb/exp-dagger-gz-20261109/full-<时间>）")
    p.add_argument("--rounds", type=int, default=CFG["rounds"])
    p.add_argument("--eps-per-round", type=int, default=CFG["eps_per_round"])
    p.add_argument("--steps", type=int, default=CFG["steps"])
    p.add_argument("--instances", type=int, default=CFG["instances"])
    p.add_argument("--epochs", type=int, default=CFG["epochs"])
    p.add_argument("--eval-seeds", default=CFG["eval_seeds"], help="逗号分隔；默认 e1_pipeline.CFG 的 20 个")
    p.add_argument("--baselines", default=None, help=f"要带的基线行，逗号分隔（默认全部 {list(CFG['baselines'])}；\"\" = 不带）")
    p.add_argument("--eval60-arms", default=None, help="60 seed 评测哪些臂（dagger / more_demos；\"\" = 跳过）")
    p.add_argument("--dry-run", action="store_true", help="只打印命令、不建目录")
    p.add_argument("--stop-after-preflight", action="store_true", help="只查缺件并写 inputs.json / status.json")
    a = p.parse_args(argv)
    DRY = a.dry_run
    if a.rounds < 1:
        p.error("--rounds 至少 1")
    CFG.update(rounds=a.rounds, eps_per_round=a.eps_per_round, steps=a.steps, instances=a.instances,
               epochs=a.epochs, eval_seeds=a.eval_seeds)
    names = [n for n in a.baselines.split(",") if n.strip()] if a.baselines is not None else list(CFG["baselines"])
    if set(names) - set(CFG["baselines"]):
        p.error(f"未知基线 {sorted(set(names) - set(CFG['baselines']))}")
    if a.eval60_arms is not None:
        CFG["eval60_arms"] = tuple(n for n in a.eval60_arms.split(",") if n.strip())
    if set(CFG["eval60_arms"]) - {"dagger", "more_demos"}:
        p.error("--eval60-arms 只能是 dagger / more_demos")
    out = os.path.abspath(a.out)
    if not DRY:
        os.makedirs(out, exist_ok=True)
        write_json(os.path.join(out, "config.json"),
                   {"cfg": CFG, "baselines": names, "argv": sys.argv, "repo": REPO, "build": BUILD, "python": PY,
                    "gz_env": gz_env(), "git": subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                                                              capture_output=True, text=True).stdout.strip()})
        if os.path.exists(os.path.join(out, "DONE")):
            os.remove(os.path.join(out, "DONE"))
        signal.signal(signal.SIGTERM, _on_term)
        signal.signal(signal.SIGINT, _on_term)
    rc, last = 0, "preflight"
    try:
        settings = step(out, "preflight", lambda: _preflight_all(out, names))
        if a.stop_after_preflight:
            return 0
        exclude = test_gids(CFG["eval_seeds"])
        ckpt = CFG["round0_ckpt"]
        for k in range(1, CFG["rounds"] + 1):
            last = f"round_{k}"
            ckpt = run_round(out, k, settings, exclude, ckpt)
        last = "more_demos"
        md_ckpt = run_more_demos(out, settings, exclude, CFG["rounds"])
        arms = [("dagger", "student", ckpt), ("more_demos", "student", md_ckpt)]
        last = "eval20"
        step(out, "eval20", evaluate, out, arms, CFG["eval_seeds"], "eval20", settings)
        last = "baselines"
        decisions = step(out, "baselines", baselines, out, names, CFG["eval_seeds"], settings)
        last = "eval60"
        csv60 = step(out, "eval60.seeds", seeds60, out)
        arm60 = [(n, "student", ckpt if n == "dagger" else md_ckpt) for n in CFG["eval60_arms"]]
        if arm60:
            step(out, "eval60", evaluate, out, arm60, csv60, "eval60", settings)
        last = "summary"
        if not DRY:
            step(out, "summary", summary, out, CFG["rounds"], arms, decisions, csv60)
    except SystemExit as e:   # SIGTERM/SIGINT（_on_term）或 pick_test_seeds 的 SystemExit
        rc = e.code if isinstance(e.code, int) else 1
        log(f"!!!! {last} 中断：rc={rc}")
    except Exception as e:   # noqa: BLE001 - 任一步失败就停，原因已由 step() 进 status
        rc = 1
        log(f"!!!! {last} 失败：{e}")
    finally:
        if not DRY:
            with open(os.path.join(out, "DONE"), "w", encoding="utf-8") as f:
                f.write(f"rc={rc} {time.strftime('%F %T')} last_step={last}\n")
    return rc


def _preflight_all(out, names):
    settings = round0_settings(CFG["round0_collection"])
    profiles = preflight(out, settings, names)
    mark(out, "preflight", "ok", dynamics_profiles=profiles, round0_episodes=settings["episodes"], round0_steps=settings["steps"],
         stop_at_goal=settings["stop_at_goal"], tiebreak=settings["tiebreak"])
    log(f"round0 {settings['episodes']} 集 {settings['steps']} 步，stop_at_goal={settings['stop_at_goal']} tiebreak={settings['tiebreak']}，"
        f"dynamics_profile {profiles}")
    return settings


if __name__ == "__main__":
    sys.exit(main())
