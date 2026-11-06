#!/usr/bin/env python3
"""两数据集对比（骨架）：A=Gazebo 柱阵+森林、B=3DGS 转弯路线、AB=合并 → 训 3 个学生 → 两域测试集交叉闭环 → summary.md。

矩阵见 docs/experiments/2026-11-07-dataset-compare-plan.md §2：行 = {teacher, forward, student-x3-old, S_A, S_B, S_AB}，
列 = {gazebo（20 个固定 seed）, 3dgs（留出路线）}。学生同结构同超参，只换训练集。

运行（仓库根，先 source env.sh；守护进程，与 e1_pipeline 同一套约定）：
  OUT=$DATA_ROOT/uav_rgb/compare-$(date +%Y%m%d-%H%M); mkdir -p "$OUT"
  setsid nohup "$UAV_RGB_PYTHON" tools/compare_datasets.py --out "$OUT" > "$OUT/run.log" 2>&1 < /dev/null &
  echo $! > "$OUT/run.pid"
看进度：tail -f $OUT/run.log；每步状态 $OUT/status.json；结束落 $OUT/DONE。续跑 --from train。--dry-run 只打印命令。
改参数只改 CFG，不改函数。

骨架状态（2026-11-07）：只保证 --dry-run 能把全部命令打印出来；真跑前要确认：
  ① CFG["sources"] 里 A/B 的 episode 目录（B 放量后的目录名）；
  ② 3DGS 评测的 teacher 组 CLI（eval_3dgs 现只有 student/forward；教师组数字要从 collect_3dgs --policy teacher 的统计取）；
  ③ hold：A 1.0 s、B pilot 0.5 s，preprocess 拒绝混 hold（AB 打包会停），B 放量要用 hold 1.0。
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("DATA_ROOT", "/root/rivermind-data")
BUILD = os.environ.get("UAV_RGB_BUILD_DIR", os.path.join(REPO, "build", "check"))
PY = sys.executable

# ───────────────────────────── 配置区（改这里） ─────────────────────────────
CFG = dict(
    # 数据来源：每个集合 = 若干「含 train/val/test 子目录的采集目录」或「直接是 episode 目录的 glob」。
    # A 的 base 与森林是两个 geometry_version、B 是 3dgs.*：同一 .uavr 里混族必须显式 --allow-mixed-geometry。
    sources=dict(
        A=[f"{DATA_ROOT}/uav_rgb/e1-pipeline-20261107-0417/collection/{{train,val,test}}/*/",       # 柱阵 base 300 集
           f"{DATA_ROOT}/uav_rgb/tmp/agent-forest/teacher3/{{sparse,medium,dense}}/{{train,val,test}}/*/"],  # 森林三档（教师能过的档位）
        B=[f"{DATA_ROOT}/uav_rgb/datasets/e1-3dgs-callinstrasse-pilot/route_*/ep_*/"],              # ② 放量后换目录
    ),
    datasets=dict(A=("A",), B=("B",), AB=("A", "B")),   # 三个训练集 = 来源集合的并
    drop_collision=True,     # 与 e1_pipeline 一致：擦柱后的标签不进训练
    # 训练（与 tools/e1_pipeline.py CFG 相同，学生只换数据）
    epochs=30, batch=128, lr=1e-3, seed=0, class_weights="auto", text_dim=0,
    old_x3_ckpt=f"{DATA_ROOT}/uav_rgb/t6-full-1640/run/ckpt_best.bin",   # 旧 X3 学生（载具迁移差距）
    # 评测域
    gazebo=dict(   # E1 机体、observation、hold 1.0，20 个双 salt 都是 test 的 seed（同 e1_pipeline）
        dynamics="cf_sitl", vehicle="e1_mini", hold=1.0, window_from="observation",
        seeds="180,206,250,485,503,555,586,708,715,721,799,855,932,992,1113,1116,1123,1145,1246,1282",
        steps=80, sim_timeout=120.0, episode_timeout=600.0,
        env={"UAVRGB_RENDER_BACKEND": "software", "DISPLAY": ":20", "XDG_RUNTIME_DIR": "/tmp/xdg",
             "UAVRGB_CF_SITL_PLUGIN_DIR": f"{DATA_ROOT}/uav_rgb/tmp/build-cf-sitl/gz"},
    ),
    gs=dict(       # 3DGS 留出路线：scene.json 带 routes[]，--route 选留出路线（split=test 的）
        scene=f"{DATA_ROOT}/uav_rgb/3dgs-scenes/callinstrasse/scene.json",
        routes=(3,),           # ② 留出路线编号，等 3DGS agent 定
        episodes_per_route=10, steps=80, hold=1.0, dynamics="cf_sitl",
        dynamics_params=f"{REPO}/sim/dynamics/e1_mini.json",
    ),
)
# 评测组：(名字, policy, ckpt 来源)。ckpt 来源 None=无模型；"old_x3_ckpt"=CFG 里的路径；"S_A" 等=本脚本训出的学生。
GROUPS = (
    ("teacher", "teacher", None),
    ("forward", "forward", None),
    ("student-x3-old", "student", "old_x3_ckpt"),
    ("S_A", "student", "A"),
    ("S_B", "student", "B"),
    ("S_AB", "student", "AB"),
)
EVAL_DOMAINS = ("gazebo", "3dgs")
STEPS = ("pack", "train", "eval", "summary")
# ──────────────────────────────────────────────────────────────────────────

DRY = False


def log(msg):
    print(time.strftime("%F %T"), msg, flush=True)


def run(cmd, env_extra=None, log_path=None):
    log("$ " + " ".join(cmd) + (f"  > {log_path}" if log_path else ""))
    if DRY:
        return 0
    env = dict(os.environ, **(env_extra or {}))
    if log_path:
        with open(log_path, "ab") as f:
            return subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    return subprocess.run(cmd, env=env).returncode


def sh(cmd, env_extra=None):
    rc = run(cmd, env_extra)
    if rc != 0:
        raise RuntimeError(f"退出码 {rc}: {' '.join(cmd[:3])}")


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def mark(out, step, state, **extra):
    if DRY:
        return
    path = os.path.join(out, "status.json")
    status = json.load(open(path, encoding="utf-8")) if os.path.isfile(path) else {}
    status[step] = dict(state=state, at=time.strftime("%F %T"), **extra)
    write_json(path, status)


def need(path, hint):
    if not DRY and not os.path.exists(path):
        raise RuntimeError(f"缺 {path}（{hint}）")


def expand_braces(pattern):
    """把 {a,b} 展开成多条 glob（subprocess 不经 shell，这里自己展开）。"""
    if "{" not in pattern:
        return [pattern]
    head, _, rest = pattern.partition("{")
    alts, _, tail = rest.partition("}")
    return [p for alt in alts.split(",") for p in expand_braces(head + alt + tail)]


def episode_dirs(source_names):
    eps = []
    for name in source_names:
        for pat in CFG["sources"][name]:
            for g in expand_braces(pat):
                eps += sorted(d for d in glob.glob(g) if os.path.isfile(os.path.join(d, "episode_manifest.json")))
    return eps


# ───────────────────────────── 四个步骤，按顺序 ─────────────────────────────
def pack(out):
    """三个 .uavr。AB（以及 A 自身的柱阵+森林）跨族，必须显式 --allow-mixed-geometry；hold/window 不一致仍会停（③）。"""
    for ds_name, srcs in CFG["datasets"].items():
        ds = os.path.join(out, "ds", ds_name)
        eps = episode_dirs(srcs)
        if not DRY and not eps:
            raise RuntimeError(f"{ds_name}: {srcs} 下没有 episode 目录")
        cmd = [PY, f"{REPO}/tools/preprocess.py", "--out", ds, "--allow-mixed-geometry"]
        if CFG["drop_collision"]:
            cmd.append("--drop-collision")
        sh(cmd + (eps or [f"<{'+'.join(srcs)} 的 episode 目录>"]))


def train(out):
    for ds_name in CFG["datasets"]:
        ds, run_dir = os.path.join(out, "ds", ds_name, "dataset.uavr"), os.path.join(out, "run", ds_name)
        need(ds, f"先跑 pack（{ds_name}）")
        if os.path.isdir(run_dir) and os.listdir(run_dir):
            raise RuntimeError(f"{run_dir} 非空：uavrgb_train 拒绝复用；删掉或换 --out")
        cmd = [f"{BUILD}/uavrgb_train", "--data", ds, "--out", run_dir,
               "--epochs", str(CFG["epochs"]), "--batch", str(CFG["batch"]), "--lr", repr(float(CFG["lr"])),
               "--seed", str(CFG["seed"]), "--class-weights", CFG["class_weights"]]
        if CFG["drop_collision"]:
            cmd.append("--drop-collision")
        if CFG["text_dim"] > 0:
            cmd += ["--text-dim", str(CFG["text_dim"])]
        elif os.path.isfile(os.path.join(out, "ds", ds_name, "instructions.jsonl")):
            cmd.append("--vision-only")
        sh(cmd)
        need(f"{run_dir}/ckpt_best.bin", "训练结束但没有 best ckpt")


def ckpt_of(out, key):
    if key is None:
        return None
    if key == "old_x3_ckpt":
        return CFG["old_x3_ckpt"]
    return os.path.join(out, "run", key, "ckpt_best.bin")


def eval_gazebo_cmd(out, name, policy, ckpt):
    g = CFG["gazebo"]
    edir = os.path.join(out, "eval-gazebo", name)
    cmd = [PY, f"{REPO}/sim/tools/eval_closed_loop.py", "--policy", policy, "--task", "base",
           "--split", "test", "--seeds", g["seeds"], "--episodes", str(len(g["seeds"].split(","))),
           "--steps", str(g["steps"]), "--out", edir,
           "--sim-timeout", str(g["sim_timeout"]), "--episode-timeout", str(g["episode_timeout"]),
           "--dynamics", g["dynamics"], "--vehicle", g["vehicle"], "--hold", str(g["hold"]),
           "--window-from", g["window_from"]]
    if ckpt:
        cmd += ["--ckpt", ckpt, "--infer-bin", f"{BUILD}/uavrgb_infer"]
    return cmd, g["env"], edir


def eval_3dgs_cmd(out, name, policy, ckpt, route):
    g = CFG["gs"]
    edir = os.path.join(out, "eval-3dgs", f"{name}-route{route}")
    if policy == "teacher":
        # ② eval_3dgs 没有 teacher 策略；教师组用采集器的到达/碰撞统计（collect_3dgs --policy teacher）。
        cmd = [PY, "-m", "sim.3dgs.collect_3dgs", "--scene", g["scene"], "--route", str(route),
               "--policy", "teacher", "--episodes", str(g["episodes_per_route"]), "--steps", str(g["steps"]),
               "--hold", str(g["hold"]), "--dynamics", g["dynamics"], "--dynamics-params", g["dynamics_params"],
               "--out", edir]
        return cmd, {}, edir
    cmd = [PY, "-m", "sim.3dgs.eval_3dgs", "--scene", g["scene"], "--route", str(route), "--policy", policy,
           "--episodes", str(g["episodes_per_route"]), "--steps", str(g["steps"]), "--hold", str(g["hold"]),
           "--dynamics", g["dynamics"], "--dynamics-params", g["dynamics_params"], "--out", edir]
    if ckpt:
        cmd += ["--ckpt", ckpt, "--infer-bin", f"{BUILD}/uavrgb_infer"]
    return cmd, {}, edir


def evaluate(out):
    """gazebo 域 6 组并行各 1 个 gz（采集已结束，核是空的）；3dgs 域串行（共用一块 GPU 渲染）。"""
    for _name, _policy, key in GROUPS:
        if key:
            need(ckpt_of(out, key), f"{key} 的 ckpt")
    procs = []
    if "gazebo" in EVAL_DOMAINS:
        for name, policy, key in GROUPS:
            cmd, env, edir = eval_gazebo_cmd(out, name, policy, ckpt_of(out, key))
            if os.path.isfile(os.path.join(edir, "eval.json")):
                log(f"跳过 gazebo/{name}：已有 eval.json")
                continue
            log_path = os.path.join(out, f"eval-gazebo-{name}.log")
            log("$ " + " ".join(cmd) + f"  > {log_path}")
            if not DRY:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                procs.append((name, subprocess.Popen(cmd, env=dict(os.environ, **env),
                                                     stdout=open(log_path, "ab"), stderr=subprocess.STDOUT)))
    if "3dgs" in EVAL_DOMAINS:
        for name, policy, key in GROUPS:
            for route in CFG["gs"]["routes"]:
                cmd, env, edir = eval_3dgs_cmd(out, name, policy, ckpt_of(out, key), route)
                if os.path.isfile(os.path.join(edir, "eval.json")):
                    log(f"跳过 3dgs/{name} route {route}：已有 eval.json")
                    continue
                run(cmd, env, log_path=os.path.join(out, f"eval-3dgs-{name}-route{route}.log"))
    for name, proc in procs:
        rc = proc.wait()
        log(f"eval-gazebo/{name} rc={rc}" + ("（有失败/跳过集，见 eval.json 的 failed）" if rc else ""))


def summary(out):
    """每域每组一行：SR（分母=请求集）、碰撞、stuck、false_forward、n、Wilson95。3DGS 的 eval.json 字段以 eval_3dgs 实际输出为准（②）。"""
    rows = []
    for domain in EVAL_DOMAINS:
        for name, _policy, key in GROUPS:
            paths = ([os.path.join(out, "eval-gazebo", name, "eval.json")] if domain == "gazebo" else
                     [os.path.join(out, "eval-3dgs", f"{name}-route{r}", "eval.json") for r in CFG["gs"]["routes"]])
            for path in paths:
                if not os.path.isfile(path):
                    rows.append((domain, name, os.path.basename(os.path.dirname(path)), None))
                    continue
                e = json.load(open(path, encoding="utf-8"))
                rows.append((domain, name, os.path.basename(os.path.dirname(path)), {
                    "SR": e.get("requested_success_rate", e.get("success_rate")),
                    "coll": e.get("collision_rate"), "stuck": e.get("stuck_rate"),
                    "ffwd": e.get("false_forward_rate"), "n": e.get("episodes_done"),
                    "wilson95": (e.get("wilson_ci") or {}).get("requested"),
                    "metric_version": e.get("metric_version")}))
    fmt = "%-7s %-15s %-22s %6s %6s %6s %6s %4s"
    lines = [fmt % ("domain", "group", "run", "SR", "coll", "stuck", "ffwd", "n")]
    f = lambda v, d=2: "-" if v is None else f"{v:.{d}f}"   # noqa: E731
    for domain, name, run_name, r in rows:
        lines.append(fmt % ((domain, name, run_name) + (("缺", "", "", "", "") if r is None else
                            (f(r["SR"]), f(r["coll"]), f(r["stuck"]), f(r["ffwd"]), r["n"]))))
    lines.append("口径：SR 分母 = 请求集；gazebo 域 E1 机体 cf_sitl / observation / hold 1.0，20 个双 salt test seed；"
                 "3dgs 域留出路线，hold 以 eval.json 的 hold_s 为准（A/B hold 不同时表里必须标注）；详见 docs/EVAL_STANDARD.md")
    for line in lines:
        log(line)
    if not DRY:
        write_json(os.path.join(out, "summary.json"), {"cfg": CFG, "groups": GROUPS, "rows": rows})
        with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as fp:
            fp.write("```\n" + "\n".join(lines) + "\n```\n")


RUN = dict(pack=pack, train=train, eval=evaluate, summary=summary)


def preflight():
    need(f"{BUILD}/uavrgb_train", "先 bash tools/check.sh 构建，或设 UAV_RGB_BUILD_DIR")
    need(f"{BUILD}/uavrgb_infer", "同上")
    need(CFG["old_x3_ckpt"], "旧 X3 学生 ckpt")
    for name, _p, key in GROUPS:
        if key not in (None, "old_x3_ckpt") and key not in CFG["datasets"]:
            raise RuntimeError(f"GROUPS 里 {name} 的 ckpt 来源 {key} 不在 CFG.datasets")


def main():
    global DRY
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="输出根目录（数据盘）")
    p.add_argument("--from", dest="start", choices=STEPS, default="pack", help="从哪一步开始（续跑）")
    p.add_argument("--dry-run", action="store_true", help="只打印命令，不执行、不写文件")
    a = p.parse_args()
    DRY = a.dry_run
    out = os.path.abspath(a.out)
    if not DRY:
        os.makedirs(out, exist_ok=True)
        write_json(os.path.join(out, "config.json"),
                   {"cfg": CFG, "groups": GROUPS, "argv": sys.argv, "repo": REPO, "build": BUILD, "python": PY,
                    "git": subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                                          capture_output=True, text=True).stdout.strip()})
        if os.path.exists(os.path.join(out, "DONE")):
            os.remove(os.path.join(out, "DONE"))
    rc, last = 0, None
    try:
        preflight()
        for step in STEPS[STEPS.index(a.start):]:
            last = step
            log(f"==== {step} 开始")
            mark(out, step, "running")
            t0 = time.monotonic()
            RUN[step](out)
            mark(out, step, "ok", seconds=round(time.monotonic() - t0, 1))
            log(f"==== {step} 完成 {time.monotonic() - t0:.0f}s")
    except Exception as e:   # noqa: BLE001
        rc = 1
        mark(out, last or "preflight", "failed", error=f"{type(e).__name__}: {e}")
        log(f"!!!! {last or 'preflight'} 失败：{e}")
    finally:
        if not DRY:
            with open(os.path.join(out, "DONE"), "w", encoding="utf-8") as fp:
                fp.write(f"rc={rc} {time.strftime('%F %T')} last_step={last}\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
