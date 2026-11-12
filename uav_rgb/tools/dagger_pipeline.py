#!/usr/bin/env python3
"""DAgger 流水线（P1 信念软标签教师）：第 0 轮教师数据 → 打标/打包/训练 → 第 k 轮学生自采（教师打标）
→ 全部轮次并集重打包 → 训练 → 闭环评测。守护进程跑。

每轮训练三个学生（同 seed、同 epoch、同数据）：
  hard       硬标签 BC + 逆频率类权重（项目标准配方）
  soft       P1 信念软标签（teacher_belief，--soft-targets --distill-alpha 1.0；trainer 不允许同时给类权重）
  hard_plain 硬标签、无类权重（对照：把「类权重」和「软标签」两个因素拆开）
第 k 轮自采由 soft 学生执行（DAgger 的被纠正对象是 P1 学生）；hard/hard_plain 用同一份并集数据训，
三者看到的状态完全一样，差的只是标签。

运行（仓库根，3DGS 域）：
  OUT=$DATA_ROOT/uav_rgb/tmp/agent4-teacher-p1/dagger-$(date +%Y%m%d-%H%M); mkdir -p "$OUT"
  export TMPDIR=$DATA_ROOT/uav_rgb/tmp UAV_RGB_BUILD_DIR=$DATA_ROOT/uav_rgb/tmp/agent4-teacher-p1/bin   # 冻结的二进制快照：各臂同一 uavrgb_train
  setsid nohup .venv-tools/bin/python tools/dagger_pipeline.py --out "$OUT" > "$OUT/run.log" 2>&1 < /dev/null &
  sleep 3; cat "$OUT/run.pid"   # 进程自己写 run.pid：pid= pgid= start=（/proc/<pid>/stat 第 22 列）cmd=；不要再 echo $! > run.pid，重复起会冲掉第一个的记录
  # 停：先核对 start 与该 PID 现在的启动时刻一致（容器重启后 PID 可能已被别的进程拿走），再 kill -- -<pgid>
看进度：tail -f $OUT/run.log；每步 $OUT/status.json；结束落 $OUT/DONE。--dry-run 只打印命令。
同一个 --out 只许一个实例：起跑先拿 $OUT/.pipeline.lock（flock），拿不到就退出 3、不改 $OUT 里任何文件；锁随进程消失，被杀后原命令可直接再起。
断点续跑：同一 --out 重跑，完成的步骤跳过（标注 npz、ds、训练 = run/metrics.jsonl 末行 done:、评测 = eval.json 完整且
无失败集；没完成的评测目录改名 .failed-<时间> 后重评），汇总重算（tools/pipeline_common）。
--collect-round0：第 0 轮用 astar 教师现采（与 DAgger 轮同一份固件参数；pilot 的 dynamics_profile 与新采集不同，不能混包）。
--domain 只接受 3dgs（旗标保留做兼容）：Gazebo 域的学生自采/训练/评测走 tools/e1_pipeline.py 与
sim/tools/collect_parallel.py --policy student，这里不再打印那套命令。
SIGTERM / Ctrl-C：收掉子进程组，status.json 当前步标 interrupted，DONE 写真实 rc（143 / 130）。
改参数只改 CFG，不改函数。
"""
import argparse
import glob
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("DATA_ROOT", "/root/rivermind-data")
BUILD = os.environ.get("UAV_RGB_BUILD_DIR", os.path.join(REPO, "build", "check"))
PY = sys.executable                                                   # .venv-tools：preprocess / join / teacher_belief
PY3DGS = os.environ.get("UAV_RGB_3DGS_PYTHON", f"{DATA_ROOT}/uav_rgb/3dgs-venv/bin/python")   # torch + gsplat
sys.path.insert(0, os.path.join(REPO, "tools"))
import pipeline_common as pc  # noqa: E402  - 三条流水线共用：helper / 续跑判定 / 信号

# ───────────────────────────── 配置区（改这里） ─────────────────────────────
CFG = dict(
    domain="3dgs",                       # 3dgs = 真跑；gazebo = 只打印命令
    # 第 0 轮教师数据：现成的 3DGS pilot（astar_occ 教师，FOV 60°，20 集 1457 步）的 episodes.txt；
    # 设为 None 则用 astar 教师现采 round0_eps_per_route × 路线（和 DAgger 轮同一套固件参数——
    # preprocess 的 dynamics_profile 是严格键，pilot 的 e1_mini.json 指纹 bd0dcd43 与之后的 cec5d6b0（11-07 06:59）、
    # 683e7400（v3.1，vel_ki 0.30，11-09 起）互不能混包）
    round0_episodes=f"{DATA_ROOT}/uav_rgb/datasets/e1-3dgs-callinstrasse-pilot-v3-fov60/episodes.txt",
    round0_eps_per_route=3, round0_seed=1000,
    # cf_sitl 固件参数：本仓库的 e1_mini.json（2026-11-07 06:59 起含 tilt_int_max/pos_kp 三项，cf_sitl_proc 需要；
    # 以前写死主仓库绝对路径是因为当时 worktree 落后，现在 main 已含该版本）
    dynamics_params=f"{REPO}/sim/dynamics/e1_mini.json",
    scene=f"{DATA_ROOT}/uav_rgb/3dgs-scenes/callinstrasse/scene.json",
    occ_npz=f"{DATA_ROOT}/uav_rgb/tmp/agent4-teacher-p1/occ2d_callinstrasse.npz",   # build_occ_cache.py 的 2D 占据缓存
    routes=(0, 1, 2, 3, 4, 5),
    # DAgger 轮次：第 1 轮 β=0.5（一半步改执行教师，学生刚训出来还不稳），之后 β=0（Ross 2011 §4 的建议）
    rounds=2,
    betas=(0.5, 0.0),
    dagger_eps_per_route=2,              # 每轮每路线自采集数：12 集 ≈ 800 步，约等于第 0 轮的一半
    dagger_seed=5000,                    # 自采 seed 段（pilot 用 1000–1512，评测用 9000+）
    spawn_jitter=0.5,                    # 同 pilot：出生 ±0.5 m / ±10°，不然 12 集全是同一条轨迹
    # 信念教师
    k=8, alpha_steps=1.0, aggregate="qmdp", label_workers=10,
    # 训练（uavrgb_train；数据只有 ~1.5k–3k 步，20 轮足够，best ckpt 按 val）
    epochs=20, batch=64, lr=1e-3, seed=0, class_weights="auto", distill_alpha=1.0, drop_collision=True,
    arms=("hard", "soft", "hard_plain"),
    # 评测（eval_3dgs：成功 = 无碰撞且进入 reach_radius 2 m；每路线 3 集 = 18 集/组）
    eval_eps_per_route=3, eval_seed=9000, hold=0.5, dynamics="cf_sitl",
    # Gazebo 域（只打印）：collect_parallel 现有 --policy 只有 teacher/forward/teacher-rl，student 要 STUDENT-P2 加
    gazebo_task="base", gazebo_instances=4, gazebo_eps=40,
)
# gsplat JIT 缓存目录在 main 里按本机 GPU 计算能力解析（torch_extensions_dir）：不同卡的 .so 不混用。
ENV_3DGS = dict(UAV_RGB_BUILD_DIR=BUILD)   # student.py 按 UAV_RGB_BUILD_DIR 找 uavrgb_infer
EVAL_METRIC_VERSION = "3dgs.pilot.v2"      # 与 sim/3dgs/eval_3dgs.METRIC_VERSION 同步（test_pipeline_common 对账）
# ──────────────────────────────────────────────────────────────────────────

DRY = False   # main 里与 pipeline_common.DRY 一起置位；helper 都在 tools/pipeline_common.py（A4）
log, sh, read_json, write_json, mark, need = pc.log, pc.sh, pc.read_json, pc.write_json, pc.mark, pc.need


def run(cmd, env_extra=None, log_path=None, cwd=REPO):
    """3DGS 侧命令都在仓库根跑（-m sim.3dgs.*），并去掉 TORCH_CUDA_ARCH_LIST：gsplat 按本机卡 JIT，带旧变量会重编/编错。"""
    return pc.run_cmd(cmd, env_extra=env_extra, log_path=log_path, cwd=cwd, env_drop=("TORCH_CUDA_ARCH_LIST",))


def torch_extensions_dir(capability=None, python=None):
    """gsplat JIT 缓存目录（pipeline_common.torch_extensions_dir），算力缺省问 PY3DGS 里的 torch；问不到返回 None，
    main 就不设 TORCH_EXTENSIONS_DIR，用 torch 默认缓存。"""
    return pc.torch_extensions_dir(capability, python or PY3DGS)


# ───────────────────────────── 各步 ─────────────────────────────
def round0_dirs(out):
    """第 0 轮教师数据：给了 episodes.txt 就用现成的，否则 astar 教师现采。"""
    if CFG["round0_episodes"]:
        with open(CFG["round0_episodes"], encoding="utf-8") as f:
            dirs = [l.strip() for l in f if l.strip()]
        for d in dirs:
            need(os.path.join(d, "actions.jsonl"), "第 0 轮 episode 目录")
        return dirs
    root = os.path.join(out, "round_0", "collect")
    for r in CFG["routes"]:
        rdir = os.path.join(root, f"route_{r}")
        if done(os.path.join(rdir, "run.json")):
            continue
        if not DRY and os.path.isdir(rdir) and os.listdir(rdir):
            moved = pc.retire_dir(rdir, "partial", "没有 run.json")
            log(f"round_0 route_{r} 上次采集没跑完，旧目录挪到 {moved}，重采")
        sh([PY3DGS, "-m", "sim.3dgs.collect_3dgs", "--scene", CFG["scene"], "--route", str(r),
            "--out", rdir, "--policy", "astar", "--episodes", str(CFG["round0_eps_per_route"]),
            "--seed", str(CFG["round0_seed"] + r * 100), "--hold", str(CFG["hold"]), "--dynamics", CFG["dynamics"],
            "--dynamics-params", CFG["dynamics_params"], "--spawn-jitter", str(CFG["spawn_jitter"])],
           env_extra=ENV_3DGS, log_path=None if DRY else os.path.join(root, f"collect_route_{r}.log"))
    dirs = good_episode_dirs(root)
    if not DRY and not dirs:
        raise RuntimeError("第 0 轮没有可用的教师集")
    return dirs


def good_episode_dirs(collect_root):
    """自采产物里可进训练集的集：manifest 无 error、validity 全 true、没有动力学异常。"""
    out = []
    for d in sorted(glob.glob(os.path.join(collect_root, "route_*", "ep_*"))):
        if "." in os.path.basename(os.path.dirname(d)):
            continue   # 排除 route_<r>.partial-<时刻> 等退居目录里的残余集
        mp = os.path.join(d, "episode_manifest.json")
        if not os.path.isfile(mp):
            continue
        m = read_json(mp)
        if m.get("error") or m.get("dynamics_anomaly") or not all((m.get("validity") or {}).values()):
            log(f"跳过 {d}: error={m.get('error')} anomaly={m.get('dynamics_anomaly')} validity={m.get('validity')}")
            continue
        out.append(d)
    return out


def done(*paths):
    """断点续跑：产物都在就跳过（同一 --out 重跑只补缺的步骤、重算汇总）。"""
    return not DRY and all(os.path.exists(x) for x in paths)


def label(dirs, out_npz, out_stats):
    """信念软标签：teacher_belief 按位姿离线重标（多进程）。"""
    if done(out_npz, out_stats):
        return log(f"跳过标注（已有 {out_npz}）")
    sh([PY, f"{REPO}/sim/tools/teacher_belief.py", "--occ-npz", CFG["occ_npz"], "--out", out_npz,
        "--stats", out_stats, "--k", str(CFG["k"]), "--seed", str(CFG["seed"]),
        "--alpha-steps", str(CFG["alpha_steps"]), "--aggregate", CFG["aggregate"],
        "--workers", str(CFG["label_workers"]), *dirs])


def pack(dirs, npzs, ds):
    """preprocess 打包 + join_soft_targets（无侧车：flags=0，按 UAVR 记录的 source_dir 找 episode）。"""
    if done(f"{ds}/dataset.uavr", f"{ds}/soft_targets.bin"):
        return log(f"跳过打包（已有 {ds}）")
    if not DRY and os.path.isdir(ds) and os.listdir(ds):
        moved = pc.retire_dir(ds, "partial", "缺少 dataset.uavr 或 soft_targets.bin")
        log(f"{ds} 上次打包没跑完，旧目录挪到 {moved}，重打")
    cmd = [PY, f"{REPO}/tools/preprocess.py", "--out", ds]
    if CFG["drop_collision"]:
        cmd.append("--drop-collision")
    sh(cmd + dirs)
    sh([PY, f"{REPO}/tools/join_soft_targets.py", "--uavr-dir", ds, "--soft", *npzs,
        "--temperature", "1.0", "--out", os.path.join(ds, "soft_targets.bin")])


def train(ds, run_dir, arm):
    if not DRY and pc.train_finished(run_dir):   # 完成 = metrics.jsonl 末行 done:（ckpt.bin 每轮都写，不算）
        log(f"跳过训练（{run_dir}/metrics.jsonl 已记 done）")
        return f"{run_dir}/ckpt_best.bin" if os.path.isfile(f"{run_dir}/ckpt_best.bin") else f"{run_dir}/ckpt.bin"
    if not DRY and os.path.isdir(run_dir) and os.listdir(run_dir):
        moved = pc.retire_dir(run_dir, "partial", "训练没跑完（metrics.jsonl 末行不是 done:）")
        log(f"{run_dir} 上次训练没跑完，旧目录挪到 {moved}，重训")
    cmd = [f"{BUILD}/uavrgb_train", "--data", f"{ds}/dataset.uavr", "--out", run_dir,
           "--epochs", str(CFG["epochs"]), "--batch", str(CFG["batch"]), "--lr", repr(float(CFG["lr"])),
           "--seed", str(CFG["seed"])]
    if arm == "hard":
        cmd += ["--class-weights", CFG["class_weights"]]
    elif arm == "soft":
        cmd += ["--soft-targets", f"{ds}/soft_targets.bin", "--distill-alpha", str(CFG["distill_alpha"])]
    if CFG["drop_collision"]:
        cmd.append("--drop-collision")
    sh(cmd, log_path=None if DRY else run_dir + ".log")
    # 单几何场景（3DGS pilot）按几何组分 split 后没有 val：trainer 不选 best（避免用训练集挑权重），
    # 两臂都用末轮 ckpt.bin（同 epoch、无选择），有 val 时才用 ckpt_best.bin。
    if DRY or os.path.isfile(f"{run_dir}/ckpt_best.bin"):
        return f"{run_dir}/ckpt_best.bin"
    need(f"{run_dir}/ckpt.bin", "训练结束但没有 ckpt，看 metrics.jsonl")
    log(f"{run_dir}: 无 val split，用末轮 ckpt.bin")
    return f"{run_dir}/ckpt.bin"


def collect_dagger(k, ckpt, beta, out_root):
    """第 k 轮学生自采（3DGS）：每条路线 N 集，astar 教师打标，β 混合。"""
    for r in CFG["routes"]:
        rdir = os.path.join(out_root, f"route_{r}")
        if done(os.path.join(rdir, "run.json")):
            continue
        if not DRY and os.path.isdir(rdir) and os.listdir(rdir):
            moved = pc.retire_dir(rdir, "partial", "没有 run.json")
            log(f"round_{k} route_{r} 上次采集没跑完，旧目录挪到 {moved}，重采")
        sh([PY3DGS, "-m", "sim.3dgs.collect_3dgs", "--scene", CFG["scene"], "--route", str(r),
            "--out", rdir, "--policy", "student", "--ckpt", ckpt,
            "--beta", str(beta), "--dagger-round", str(k), "--episodes", str(CFG["dagger_eps_per_route"]),
            "--seed", str(CFG["dagger_seed"] + k * 1000 + r * 100), "--hold", str(CFG["hold"]),
            "--dynamics", CFG["dynamics"], "--dynamics-params", CFG["dynamics_params"],
            "--spawn-jitter", str(CFG["spawn_jitter"])],
           env_extra=ENV_3DGS, log_path=None if DRY else os.path.join(out_root, f"collect_route_{r}.log"))


def evaluate(ckpt, out_root):
    """6 路线 × N 集；返回汇总 dict。每路线一个 eval_3dgs 进程（串行，共用一张卡）。
    E-7：成功类指标（success / task_success）与对应 Wilson 区间分母取请求集数 n_req（失败/剔除集计入分母、成功记 0），
    越低越好的碰撞率分母取完成集 n_done（与 W1-2 / stats_report 同口径），failed 单独列出。"""
    rows, failed, n_req = [], 0, 0
    route_req, route_failed = {}, {}
    for r in CFG["routes"]:
        edir = os.path.join(out_root, f"route_{r}")
        state, moved, why = pc.settle_eval(edir, metric_version=EVAL_METRIC_VERSION, ckpt=ckpt)
        if DRY or state != "done":
            if moved:
                kind_msg = "与当前配置不一致" if state == "stale" else "有失败集或没跑完"
                log(f"route {r} 上次评测{kind_msg}（{why}），旧目录挪到 {moved}，重跑")
            # eval_3dgs 有集被 physics_verified=false（cf_sitl 失控保护）剔除时退出码 1 但 eval.json 照写：
            # 本轮只要 eval.json 在就继续汇总，剔除数记进 failed 并计入请求集数分母；文件都没有才算真失败。续跑时这条路线会重评。
            rc = run([PY3DGS, "-m", "sim.3dgs.eval_3dgs", "--scene", CFG["scene"], "--route", str(r), "--out", edir,
                      "--policy", "student", "--ckpt", ckpt, "--episodes", str(CFG["eval_eps_per_route"]),
                      "--seed", str(CFG["eval_seed"] + r * 10), "--hold", str(CFG["hold"]),
                      "--dynamics", CFG["dynamics"], "--dynamics-params", CFG["dynamics_params"],
                      "--spawn-jitter", str(CFG["spawn_jitter"])],
                     env_extra=ENV_3DGS, log_path=None if DRY else edir + ".log")
            if rc != 0 and not DRY:
                if not os.path.isfile(os.path.join(edir, "eval.json")):
                    raise RuntimeError(f"eval_3dgs 退出码 {rc} 且没有 eval.json：{edir}.log")
                log(f"eval_3dgs 退出码 {rc}（有集被剔除），继续：{edir}")
        if DRY:
            continue
        rep = read_json(os.path.join(edir, "eval.json"))
        eps = rep.get("per_episode") or []
        r_failed = rep.get("failed_count", len(rep.get("failed") or []))
        r_skipped = rep.get("skipped_count", len(rep.get("skipped") or []))
        r_req = rep["episodes_requested"] if rep.get("episodes_requested") is not None else (len(eps) + r_failed + r_skipped)
        failed += r_failed
        n_req += r_req
        route_req[r], route_failed[r] = r_req, r_failed
        # eval.json（uavrgb.eval.v1）的逐集记录在 per_episode：success = 轨迹曾进 reach 半径且无碰撞，
        # task_success 还要求在半径内发 SAFE 停住（3dgs.pilot.v2 口径，都不是 Gazebo/AvoidBench 的正式数字）
        for e in eps:
            rows.append({"route": r, "success": bool(e.get("success")), "task_success": bool(e.get("task_success")),
                         "collision": bool(e.get("collision")), "stuck": bool(e.get("stuck")),
                         "steps": e.get("steps")})
    n_done = len(rows)
    cnt = {key: sum(x[key] for x in rows) for key in ("success", "task_success", "collision", "stuck")}

    def ci95(k, denom):   # [lo, hi] 保留 3 位（summary.json 的历史格式）；公式只在 pipeline_common.wilson_ci
        w = pc.wilson_ci(k, denom)
        return None if w is None else [round(w[0], 3), round(w[1], 3)]

    return {"n": n_req, "episodes_done": n_done, "failed": failed, **cnt,
            "success_rate": cnt["success"] / n_req if n_req else None,
            "task_success_rate": cnt["task_success"] / n_req if n_req else None,
            "collision_rate": cnt["collision"] / n_done if n_done else None,
            "success_ci95": ci95(cnt["success"], n_req),
            "task_success_ci95": ci95(cnt["task_success"], n_req),
            "collision_ci95": ci95(cnt["collision"], n_done),
            "per_route": {str(r): {"n": route_req.get(r, 0),
                                   "episodes_done": sum(1 for x in rows if x["route"] == r),
                                   "failed": route_failed.get(r, 0),
                                   **{key: sum(x[key] for x in rows if x["route"] == r)
                                      for key in ("success", "task_success", "collision", "stuck")}}
                          for r in CFG["routes"]}}


def main():
    global DRY
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="输出根目录（数据盘）")
    p.add_argument("--dry-run", action="store_true", help="只打印命令")
    p.add_argument("--domain", choices=("3dgs", "gazebo"), default=CFG["domain"],
                   help="只支持 3dgs；gazebo 请用 tools/e1_pipeline.py（旗标保留做兼容）")
    p.add_argument("--rounds", type=int, default=CFG["rounds"], help="DAgger 轮数（0 = 只做第 0 轮训练+评测）")
    p.add_argument("--collect-round0", action="store_true",
                   help="第 0 轮不用 CFG.round0_episodes，改用 astar 教师现采（与 DAgger 轮同固件参数）")
    # 消融用的覆盖项（标签温度 / 蒸馏权重 / 只训哪些臂）；默认值都在 CFG
    p.add_argument("--arms", default=None, help="逗号分隔，如 soft 或 hard,soft（默认 CFG.arms）")
    p.add_argument("--alpha-steps", type=float, default=None, help="信念教师温度（步数），默认 CFG.alpha_steps")
    p.add_argument("--distill-alpha", type=float, default=None, help="软标签项权重，默认 CFG.distill_alpha")
    a = p.parse_args()
    if a.domain == "gazebo":
        p.error("--domain gazebo 不再由本脚本打印命令：Gazebo 域的学生自采用 sim/tools/collect_parallel.py --policy student，"
                "训练/评测走 tools/e1_pipeline.py（评测 seed 见 tools/eval_seeds.py）")
    pc.DRY = DRY = a.dry_run
    CFG["rounds"], CFG["domain"] = a.rounds, a.domain
    if a.collect_round0:
        CFG["round0_episodes"] = None
    if a.arms:
        CFG["arms"] = tuple(x.strip() for x in a.arms.split(",") if x.strip())
        if "soft" not in CFG["arms"] and CFG["rounds"] > 0:
            p.error("DAgger 轮由 soft 学生自采，--arms 必须含 soft")
    if a.alpha_steps is not None:
        CFG["alpha_steps"] = a.alpha_steps
    if a.distill_alpha is not None:
        CFG["distill_alpha"] = a.distill_alpha
    out = os.path.abspath(a.out)
    if not DRY:   # dry-run 不建目录、不写文件（A6）
        os.makedirs(out, exist_ok=True)
        pc.lock_out(out)   # E-3：已有实例在跑就退出 3；要赶在 config.json / DONE 之前，第二个实例不能碰第一个的状态
        pc.write_run_pid(out)   # E-4：run.pid 记 pid / pgid / 启动时刻 / 命令行，停进程前能核对 PID 还是不是这个进程
        ext_dir = torch_extensions_dir()   # 按本机卡命名的 gsplat JIT 缓存；问不到 torch 就用默认缓存
        if ext_dir:
            ENV_3DGS["TORCH_EXTENSIONS_DIR"] = ext_dir
        log(f"TORCH_EXTENSIONS_DIR={ext_dir or '(torch 默认)'}")
        write_json(os.path.join(out, "config.json"),
                   {"cfg": CFG, "env_3dgs": ENV_3DGS, "argv": sys.argv, "repo": REPO, "build": BUILD, "python": PY,
                    "python_3dgs": PY3DGS,
                    "git": subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                                          capture_output=True, text=True).stdout.strip()})
        if os.path.exists(os.path.join(out, "DONE")):
            os.remove(os.path.join(out, "DONE"))
    pc.install_signal_handlers()   # SIGTERM/Ctrl-C → Interrupted：收子进程组、status 标 interrupted、DONE 写真实 rc
    rc, last = 0, None
    summary = {"rounds": {}}
    try:
        need(f"{BUILD}/uavrgb_train", "构建目录；设 UAV_RGB_BUILD_DIR")
        need(f"{BUILD}/uavrgb_infer", "同上")
        need(CFG["occ_npz"], "tmp/agent4-teacher-p1/build_occ_cache.py 先落 2D 占据缓存")
        all_dirs = round0_dirs(out)
        npzs, ckpts = [], {}
        for k in range(0, CFG["rounds"] + 1):
            rd = os.path.join(out, f"round_{k}")
            if not DRY:
                os.makedirs(rd, exist_ok=True)
            if k > 0:
                last = f"round_{k}.collect"
                mark(out, last, "running")
                beta = CFG["betas"][min(k - 1, len(CFG["betas"]) - 1)]
                # 用上一轮软标签臂的学生跑闭环（train() 决定是 ckpt_best 还是末轮 ckpt）
                collect_dagger(k, ckpts["soft"], beta, os.path.join(rd, "collect"))
                new_dirs = good_episode_dirs(os.path.join(rd, "collect"))
                if not DRY and not new_dirs:
                    raise RuntimeError(f"第 {k} 轮没有可用的自采集")
                all_dirs = all_dirs + new_dirs
                mark(out, last, "ok", episodes=len(new_dirs), beta=beta)
                label_dirs = new_dirs
            else:
                label_dirs = all_dirs
            last = f"round_{k}.label"
            mark(out, last, "running")
            npz = os.path.join(rd, "soft.npz")
            label(label_dirs, npz, os.path.join(rd, "soft_stats.json"))
            npzs.append(npz)
            mark(out, last, "ok")
            last = f"round_{k}.pack"
            mark(out, last, "running")
            ds = os.path.join(rd, "ds")
            pack(all_dirs, npzs, ds)
            mark(out, last, "ok", episodes=len(all_dirs))
            for arm in CFG["arms"]:
                last = f"round_{k}.train_{arm}"
                mark(out, last, "running")
                ckpts[arm] = train(ds, os.path.join(rd, f"run_{arm}"), arm)
                mark(out, last, "ok")
            res = {}
            for arm in CFG["arms"]:
                last = f"round_{k}.eval_{arm}"
                mark(out, last, "running")
                res[arm] = evaluate(ckpts[arm], os.path.join(rd, f"eval_{arm}"))
                mark(out, last, "ok", **{kk: res[arm][kk] for kk in ("n", "episodes_done", "failed", "success", "task_success", "collision")})
                log(f"round {k} {arm}: SR {res[arm]['success']}/{res[arm]['n']} CI95 {res[arm]['success_ci95']} "
                    f"task {res[arm]['task_success']}/{res[arm]['n']} collision {res[arm]['collision']}/{res[arm]['episodes_done']}"
                    + (f" failed {res[arm]['failed']}" if res[arm]["failed"] else ""))
            stats = read_json(os.path.join(rd, "soft_stats.json"))["summary"] if not DRY else None
            summary["rounds"][str(k)] = {"episodes": len(all_dirs), "eval": res, "label_stats": stats}
            if not DRY:
                retired = pc.read_retired(out)
                if retired:
                    summary["retired"] = retired
                write_json(os.path.join(out, "summary.json"), summary)
        if not DRY:
            for line in pc.retired_lines(out):
                log(line)
    except pc.Interrupted as e:
        rc = e.rc
        n = pc.terminate_children()
        mark(out, last or "preflight", "interrupted", signal=e.name, children_terminated=n)
        log(f"!!!! {e}：已收掉 {n} 个子进程组，{last or 'preflight'} 标 interrupted，rc={rc}")
    except Exception as e:   # noqa: BLE001 - 任一步失败就停，原因进 status/DONE
        rc = 1
        mark(out, last or "preflight", "failed", error=f"{type(e).__name__}: {e}")
        log(f"!!!! {last or 'preflight'} 失败：{e}")
    finally:
        if not DRY:
            with open(os.path.join(out, "DONE"), "w", encoding="utf-8") as f:
                f.write(f"rc={rc} {time.strftime('%F %T')} last_step={last}\n")
        pc.unlock_out()   # DONE 写完才放锁，别的实例起得来时标记已经是最新的
    return rc


if __name__ == "__main__":
    sys.exit(main())
