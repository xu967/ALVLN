#!/usr/bin/env python3
"""E1 MINI 流水线：pilot 采集+审计 → 放量采集 → 审计 → 打包 → 训练 → 20 seed 闭环四组对照 → 汇总。守护进程跑。

运行（仓库根，先 source env.sh）：
  OUT=$DATA_ROOT/uav_rgb/e1-pipeline-$(date +%Y%m%d-%H%M); mkdir -p "$OUT"
  setsid nohup "$UAV_RGB_PYTHON" tools/e1_pipeline.py --out "$OUT" > "$OUT/run.log" 2>&1 < /dev/null &
  echo $! > "$OUT/run.pid"; ps -o pid,pgid,etime,cmd -p "$(cat "$OUT/run.pid")"   # 停：kill -- -<PGID>
看进度：tail -f $OUT/run.log；每步状态 $OUT/status.json；结束落 $OUT/DONE（rc + 时间 + 最后一步）。
续跑：--from train（跳过前面的步骤，要求它们的产物已在 OUT 里）。--dry-run 只打印命令不执行。
改参数只改 CFG，不改函数。
预计耗时（RTF 锁 1.0、软件渲染、4 实例，gz agent 11-07 实测 ≈ 50 s/集）：pilot 5 min；采集 300 集 ≈ 65 min；
训练 30 轮 ≈ 5 min（设备常驻 2669 样本/秒）；评测 4 组并行 ≈ 40 min。
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("DATA_ROOT", "/root/rivermind-data")
BUILD = os.environ.get("UAV_RGB_BUILD_DIR", os.path.join(REPO, "build", "check"))  # check.sh 的构建目录
PY = sys.executable   # 用 $UAV_RGB_PYTHON 启动本脚本，子进程继承它（preprocess/eval 需要 PIL、numpy、cv2）

# ───────────────────────────── 配置区（改这里） ─────────────────────────────
CFG = dict(
    # 载具 / 动力学（采集与四组评测共用，保证训练数据和评测协议是同一套）
    dynamics="cf_sitl",      # E1 MINI 固件在环；复现旧 X3 数字用 "x3_velocity" 并把 vehicle 改 "x3"
    vehicle="e1_mini",       # 选 sim/worlds/uav_rgb_e1.sdf；cf_sitl 必须显式给
    hold=1.0,                # 每动作仿真秒。沿用 X3 基线 1.0 便于对比；固件里杆量保持到下一次派发
    window_from="observation",   # 唯一被采纳的基线语义；dispatch 只用于复现 T7-R，数字不可混报
    plugin_dir=f"{DATA_ROOT}/uav_rgb/tmp/build-cf-sitl/gz",   # 真库插件（sim/cf_sitl/gz/README 的构建路径）。
                                                                # 显式给，免得 collect_parallel 兜底找到桩库 build-gz-stub
    # 采集（pilot 先过 audit 再放量；seed 段都避开下面 eval_seeds，它们最大 1282）
    pilot_episodes=20,       # 4 实例 × 5 集；只做闸门，不进训练集
    pilot_seed=3000,
    # 采集场景集合 (task[:density], 集数, base_seed)：逐集合采（collect/、collection-<集合>/）、各自 audit、preprocess 合并打包。
    # 集数取 instances 整数倍；T6 学生用了 599 集 / 23960 步，300 集 ≈ 12000 步。seed 段互不重叠、都避开 pilot 3000–3019。
    collect_sets=(
        ("base", 300, 3100),             # 旧柱阵 + E1 机体：当前唯一默认集
        # ("forest_e1:sparse", 150, 4100),   # E1 随机圆柱森林稀疏档（缝 0.45 m）。教师按 (e1_mini, forest_e1) 档 0.20/0.15 + 圆判定
        #                                    # （teacher.py v6 档，离线 sparse 20/20 穿林）；多集合时 preprocess 自动加 --allow-mixed-geometry。
        #                                    # 启用前仍要：Gazebo 实测森林教师 SR（旧 0.20/0.15 方框试跑 19/20 擦柱）、评测加 forest 域（evaluate 现只跑 base）
    ),
    steps=40,                # 每集步数：A* 每步约 0.27 m（X3）/ E1 实测前飞 0.25 m，29–38 步过柱区；80 步会让一半帧在柱区外原地转
    stop_at_goal=False,      # DISTILL §3.1：真值 x 过成功线（最远柱 x + 1.0 m，与 eval 同口径）即结束本集。0417 的 12000 帧里
                             # 过线后 2155 帧（18%）93% 视锥内无柱、38% 标签是 YAW（看不见的目标点的函数）；离线截断重训无柱帧误转率
                             # 22%→5%。开了后集长约 27–38 步，audit episode_length 按 manifest 放行。默认关：与 0417 基线逐位可比
    tiebreak="legacy",       # DISTILL §3.3：A* 教师左右近平局（|Q_L−Q_R|<0.5 步，转弯帧 ~10%）的口径。"visible" 改向学生视锥内
                             # 最近柱的反侧（0417 的 12000 帧里会改 85 帧）；"legacy" 与旧标签逐位相同。默认 legacy
    instances=4,             # gz agent 实测：4 实例 6.8 核 0.55 s/步；6 实例 8.6 核吞吐只多 10%（cgroup 12 核）
    no_aux_cameras=True,     # 采集不要 chase/top 相机：RTF 锁 1.0 时每步快 17%；录像只在评测时开
    collect_min_ok=0.95,     # 偶发丢首条起飞指令属正常（collect_parallel 自带重试一次）；低于 95% 说明环境坏了，停
    # 审计
    audit_ignore=("class_share", "safe_collision"),
                             # 其余任一 FAIL 即停。这两项是 X3 口径、E1 上放行（pilot 2 轮实测）：E1 教师脱困 v2 后几乎不再 SAFE 死锁，
                             # SAFE 只占 3.9% (<5%)；仍有 ~15% 集在柱旁左右摆后擦柱，擦柱步 preprocess --drop-collision 会丢掉。
                             # 注意（2026-11-07 DATA 复核 .staging 245 集）：base 任务没有 waypoint，教师只在被围死时 SAFE，SAFE 结构性 <5%；
                             # 擦柱前的标签能复算对上（0/9800 不一致），但多是余量内悬停 SAFE / 左右反转，不是想教学生的东西；
                             # 要干净训练集用 --exclude-collision-episodes（丢 6.5% 集）。教师真实 SR 由评测 teacher 组报，不靠这里兜
    # 打包 / 训练（uavrgb_train；没列的 flag 用 trainer.hpp 默认：max-grad-norm 5.0，A/B 开关全关）
    drop_collision=True,     # 专家撞柱后的标签本身是错的；T6 起一直开。打包与训练同步传
    epochs=30,               # T6 best 落在 epoch 8，20 轮后 dev F1 不再涨；30 轮给 best-ckpt 余量
    batch=128,               # DeviceCnn 常驻 ≈ 0.83 GB 显存；与用户 YOLO 共卡也够
    lr=1e-3,                 # Adam；HANDOFF §十二 示例值。l1/FiLM 用 5e-4（v4）
    seed=0,                  # 与 src-algo 基准 bench-after 同 seed，便于对照
    class_weights="auto",    # 训练集逆频率。FORWARD 占 ~75%，不加权 SAFE/YAW 召回归零（EVAL_STANDARD §0）
    text_dim=0,              # base 任务无指令。l1 用 16
    # 评测（四组同一批 test 场景；E1 机体、observation 语义、hold 1.0）
    eval_seeds="1,113,122,153,172,176,180,230,250,536,555,628,683,798,929,990,992,997,1113,1654",
                             # tools/pick_test_seeds.py 挑的 20 个：几何组两两不同、v5 下都是 test、与新 E1 学生的 collection/（300 集）
                             # 和旧 X3 学生的 t5-full-1155（600 集）的几何都零重叠（tools/test_pick_test_seeds.py 锁定）。
                             # 0417-eval2 用的 1,58,113,…,536 只排除了 E1 集合，10/20 的几何在 X3 学生的 train 里，x3-old 那行不是留出集。
                             # 更早的 180,206,…,1282 虽然都是 test，但按 gid 去重只剩 11 个独立几何（eval 会「跳过: 已评测」），
                             # 0417 那轮四组对照实际 n=11（docs/experiments/2026-11-07-e1-demo-plan.md §3 已注明）
    eval_steps=80,           # 给足全程余量（T7 起 80）；碰撞/到达会提前结束
    old_x3_ckpt=f"{DATA_ROOT}/uav_rgb/t6-full-1640/run/ckpt_best.bin",   # T6 best（epoch 8）：X3 / x3_velocity / hold 1.0 训的
    record_groups=("teacher", "student-e1-new"),   # 这两组录三视角视频（≈10 MB/仿真秒暂存，合成后删）
    extra_students=os.environ.get("E1_EXTRA_STUDENTS", ""),   # 探索臂："名=ckpt路径;名=路径"，每臂多起 1 个 gz；默认空。
                             # 例：DISTILL 2026-11-07 建议 1 的「成功线截断重训」ckpt 与四组同 seed 同动力学对照
    episode_timeout=600.0,   # 80 步 × ~1.1 s + 起飞 15 s，留 5 倍余量
    sim_timeout=120.0,       # 等相机/深度首帧
)
EVAL_GROUPS = (   # (目录名, --policy, ckpt 的键)。四组并行各起 1 个 gz（评测时采集已结束，核是空的）
    ("teacher", "teacher", None),              # 柱子真值 A* 专家：新动力学下的可解性上界
    ("student-x3-old", "student", "old_x3_ckpt"),   # 旧 X3 学生放到 E1 机体上：载具迁移差距
    ("student-e1-new", "student", "new_ckpt"),      # 本轮产物
    ("forward", "forward", None),              # 盲前进下限
)


def eval_groups():
    """四个固定组 + CFG["extra_students"] 的探索臂（(目录名, "student", ckpt 路径)）。"""
    extra = []
    for item in filter(None, (x.strip() for x in CFG["extra_students"].split(";"))):
        name, _, path = item.partition("=")
        if not name or not path or name in {g[0] for g in EVAL_GROUPS}:
            raise RuntimeError(f"extra_students 项格式应为 名=ckpt路径 且名字不与固定组重复：{item!r}")
        extra.append((name.strip(), "student", path.strip()))
    return EVAL_GROUPS + tuple(extra)


GZ_ENV = {"UAVRGB_RENDER_BACKEND": "software", "DISPLAY": ":20",
          "XDG_RUNTIME_DIR": "/tmp/xdg",      # 本容器 gpu 后端（EGL）段错误；换机器删掉这三项即回到默认 EGL
          "UAVRGB_CF_SITL_PLUGIN_DIR": CFG["plugin_dir"]}
STEPS = ("pilot", "collect", "audit", "preprocess", "train", "eval", "summary")
# ──────────────────────────────────────────────────────────────────────────

DRY = False


def log(msg):
    print(time.strftime("%F %T"), msg, flush=True)


def run(cmd, env_extra=None, log_path=None):
    """跑一条命令；stdout/stderr 进 run.log（或 log_path）；返回退出码，由调用方决定要不要停。"""
    log("$ " + " ".join(cmd) + (f"  > {log_path}" if log_path else ""))
    if DRY:
        return 0
    env = dict(os.environ, **(env_extra or {}))
    if log_path:
        with open(log_path, "ab") as f:
            return subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    return subprocess.run(cmd, env=env).returncode


def sh(cmd, env_extra=None):
    """非零退出就抛异常 → 流水线停。"""
    rc = run(cmd, env_extra)
    if rc != 0:
        raise RuntimeError(f"退出码 {rc}: {os.path.basename(cmd[0])} {cmd[1] if len(cmd) > 1 else ''}")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def mark(out, step, state, **extra):
    """每步落一次 status.json：新会话一看就知道跑到哪。"""
    if DRY:
        return
    path = os.path.join(out, "status.json")
    status = read_json(path) if os.path.isfile(path) else {}
    status[step] = dict(state=state, at=time.strftime("%F %T"), **extra)
    write_json(path, status)


def need(path, hint):
    if not DRY and not os.path.exists(path):
        raise RuntimeError(f"缺 {path}（{hint}）")


def vehicle_flags():
    """采集和评测共用的动力学/载具参数。"""
    return ["--dynamics", CFG["dynamics"], "--vehicle", CFG["vehicle"],
            "--hold", str(CFG["hold"]), "--window-from", CFG["window_from"]]


def set_dir(out, spec):
    """集合的采集目录。base 仍叫 collection/：正在跑的 e1-pipeline-* 目录与 --from 续跑不用改名。"""
    return os.path.join(out, "collection" if spec == "base" else "collection-" + spec.replace(":", "-"))


def collect_into(col, episodes, base_seed, spec="base"):
    """collect_parallel 一集失败就 rc=1，300 集里丢 1 集是常态；按 manifest 的成功率判，不拿 rc 当成败。"""
    if os.path.isdir(col) and os.listdir(col):
        raise RuntimeError(f"{col} 非空：collect_parallel 拒绝复用目录。删掉它或换 --out；想跳过用 --from")
    per = math.ceil(episodes / CFG["instances"])
    task, _, density = spec.partition(":")   # "forest_e1:sparse" -> --task forest_e1 --density sparse
    cmd = [PY, f"{REPO}/sim/tools/collect_parallel.py",
           "--instances", str(CFG["instances"]), "--episodes-per-instance", str(per),
           "--steps", str(CFG["steps"]), "--base-seed", str(base_seed),
           "--task", task, "--keep-worlds", "--out", col, *vehicle_flags()]
    if density:
        cmd += ["--density", density]
    if CFG["no_aux_cameras"]:
        cmd.append("--no-aux-cameras")
    if CFG["stop_at_goal"]:
        cmd.append("--stop-at-goal")
    if CFG["tiebreak"] != "legacy":
        cmd += ["--tiebreak", CFG["tiebreak"]]
    run(cmd, env_extra=GZ_ENV)
    if DRY:
        return
    man = os.path.join(col, "collection_manifest.json")
    need(man, "collect_parallel 没写出汇总，看 <col>/logs/")
    m = read_json(man)
    ratio = m["episodes_ok"] / max(m["episodes_requested"], 1)
    log(f"采集 ok {m['episodes_ok']}/{m['episodes_requested']}，"
        f"split train/val/test = {[len(m['splits'][s]) for s in ('train', 'val', 'test')]}，"
        f"端到端 {m['end_to_end_wall_s']:.0f} s")
    if ratio < CFG["collect_min_ok"]:
        raise RuntimeError(f"采集成功率 {ratio:.2%} < {CFG['collect_min_ok']:.0%}，见 {col}/failed/ 与 logs/")


def audit_dir(col):
    report = os.path.join(col, "audit.json")
    run([PY, f"{REPO}/sim/tools/audit_dataset.py", "--dataset", col, "--out", report])   # 有 FAIL 时 rc=1
    if DRY:
        return
    need(report, "audit_dataset 没写出 audit.json")
    failed = [c["name"] for c in read_json(report)["checks"] if c["status"] == "FAIL"]
    ignored = [n for n in failed if n in CFG["audit_ignore"]]
    if ignored:
        log(f"审计 FAIL 但按 CFG.audit_ignore 放行: {ignored}")
    if [n for n in failed if n not in CFG["audit_ignore"]]:
        raise RuntimeError(f"审计 FAIL: {failed}，见 {report}")   # 不让坏数据进训练
    log("审计全部 PASS")


# ───────────────────────────── 七个步骤，按顺序 ─────────────────────────────
def pilot(out):
    """20 集闸门：执行器在柱阵里撞柱/漂移、ToF 误判这类问题 5 分钟就暴露，不等 300 集白跑。"""
    col = os.path.join(out, "pilot")
    collect_into(col, CFG["pilot_episodes"], CFG["pilot_seed"])
    audit_dir(col)


def collect(out):
    for spec, episodes, base_seed in CFG["collect_sets"]:
        collect_into(set_dir(out, spec), episodes, base_seed, spec)


def audit(out):
    for spec, _n, _seed in CFG["collect_sets"]:
        col = set_dir(out, spec)
        need(os.path.join(col, "collection_manifest.json"), f"先跑 collect（集合 {spec}）")
        audit_dir(col)


def preprocess(out):
    """各集合的集目录合并进一个 dataset.uavr（preprocess 自己按 manifest 复算 split、拒绝混口径）。"""
    cols, ds = [set_dir(out, spec) for spec, _n, _s in CFG["collect_sets"]], os.path.join(out, "ds")
    eps = sorted(d for col in cols for s in ("train", "val", "test") for d in glob.glob(f"{col}/{s}/*/"))
    if not DRY and not eps:
        raise RuntimeError(f"{cols} 下没有 train/val/test 集目录")
    cmd = [PY, f"{REPO}/tools/preprocess.py", "--out", ds]
    if CFG["drop_collision"]:
        cmd.append("--drop-collision")
    if len(CFG["collect_sets"]) > 1:
        cmd.append("--allow-mixed-geometry")   # 柱阵 + 森林是两个 geometry_version；hold/window 仍由 preprocess 严格校验
    sh(cmd + (eps or [f"{col}/{{train,val,test}}/*/" for col in cols]))   # 集目录在 Python 里枚举，不依赖 shell 展开


def train(out):
    ds, run_dir = os.path.join(out, "ds"), os.path.join(out, "run")
    need(os.path.join(ds, "dataset.uavr"), "先跑 preprocess")
    if os.path.isdir(run_dir) and os.listdir(run_dir):
        raise RuntimeError(f"{run_dir} 非空：uavrgb_train 的 RunOwner 拒绝复用；删掉或换 --out")
    cmd = [f"{BUILD}/uavrgb_train", "--data", f"{ds}/dataset.uavr", "--out", run_dir,
           "--epochs", str(CFG["epochs"]), "--batch", str(CFG["batch"]), "--lr", repr(float(CFG["lr"])),
           "--seed", str(CFG["seed"]), "--class-weights", CFG["class_weights"]]
    if CFG["drop_collision"]:
        cmd.append("--drop-collision")
    if CFG["text_dim"] > 0:
        cmd += ["--text-dim", str(CFG["text_dim"])]
    elif os.path.isfile(f"{ds}/instructions.jsonl"):
        cmd.append("--vision-only")   # 数据带指令却要纯视觉基线时 trainer 要求显式声明
    sh(cmd)
    need(f"{run_dir}/ckpt_best.bin", "训练结束但没有 best ckpt，看 run/metrics.jsonl")


def evaluate(out):
    """四组并行，各自 1 个 gz；eval_closed_loop 有失败集时 rc=1 但 eval.json 已写，四组全跑完再一起看。"""
    ckpts = dict(old_x3_ckpt=CFG["old_x3_ckpt"], new_ckpt=os.path.join(out, "run", "ckpt_best.bin"))
    for name, _policy, ckpt_key in eval_groups():
        if ckpt_key and ckpt_key not in ckpts:
            ckpts[ckpt_key] = ckpt_key      # 探索臂：键就是 ckpt 路径
    for key, path in ckpts.items():
        need(path, f"{key} 不存在")
    procs = {}
    for name, policy, ckpt_key in eval_groups():
        edir = os.path.join(out, f"eval-{name}")
        if os.path.isfile(os.path.join(edir, "eval.json")):
            log(f"跳过 {name}：已有 eval.json")
            continue
        cmd = [PY, f"{REPO}/sim/tools/eval_closed_loop.py", "--policy", policy, "--task", "base",
               "--split", "test", "--seeds", CFG["eval_seeds"],
               "--episodes", str(len(CFG["eval_seeds"].split(","))),
               "--steps", str(CFG["eval_steps"]), "--out", edir,
               "--sim-timeout", str(CFG["sim_timeout"]), "--episode-timeout", str(CFG["episode_timeout"]),
               *vehicle_flags()]
        if ckpt_key:
            cmd += ["--ckpt", ckpts[ckpt_key], "--infer-bin", f"{BUILD}/uavrgb_infer"]
        if name in CFG["record_groups"]:
            cmd.append("--record-video")   # 需要 chase/top 相机：评测模板不加 --no-aux-cameras
        log_path = os.path.join(out, f"eval-{name}.log")
        log("$ " + " ".join(cmd) + f"  > {log_path}")
        if not DRY:
            procs[name] = subprocess.Popen(cmd, env=dict(os.environ, **GZ_ENV),
                                           stdout=open(log_path, "ab"), stderr=subprocess.STDOUT)
    for name, proc in procs.items():
        rc = proc.wait()
        log(f"eval-{name} rc={rc}" + ("（有失败/跳过集，见 eval.json 的 failed）" if rc else ""))
    for name in procs:   # 等齐了再查：中途抛出会让 main 写 DONE 退出，其余组的 gz 留成孤儿（tools/test_e1_pipeline.py）
        need(os.path.join(out, f"eval-{name}", "eval.json"), f"{name} 没写出 eval.json，看 {out}/eval-{name}.log")


def summary(out):
    rows = {}
    for name, _policy, _k in eval_groups():
        path = os.path.join(out, f"eval-{name}", "eval.json")
        if DRY and not os.path.isfile(path):
            continue
        e = read_json(path)
        rows[name] = {k: e.get(k) for k in (
            "success_rate", "requested_success_rate", "collision_rate", "stuck_rate",
            "false_forward_rate", "stable_termination_rate", "episodes_done", "failed_count",
            "skipped_count", "metric_version", "dynamics", "hold_s", "steps", "ood")}
        rows[name]["wilson95"] = (e.get("wilson_ci") or {}).get("requested")
        rows[name]["infer_p50_ms"] = (e.get("inference_ms") or {}).get("p50")
        rows[name]["infer_p95_ms"] = (e.get("inference_ms") or {}).get("p95")
    if not DRY:
        write_json(os.path.join(out, "summary.json"), {"cfg": CFG, "groups": rows})
    fmt = "%-16s %6s %6s %6s %6s %5s %5s %8s"
    lines = [fmt % ("group", "SR", "coll", "stuck", "ffwd", "done", "fail", "p50ms")]
    for name, r in rows.items():
        f = lambda v, d=2: "-" if v is None else f"{v:.{d}f}"   # noqa: E731
        lines.append(fmt % (name, f(r["requested_success_rate"]), f(r["collision_rate"]), f(r["stuck_rate"]),
                            f(r["false_forward_rate"]), r["episodes_done"], r["failed_count"], f(r["infer_p50_ms"], 1)))
    lines.append(f"口径：SR 分母 = 请求集 {len(CFG['eval_seeds'].split(','))}（失败集计入分母）；四组同一批 seed、"
                 f"E1 机体 {CFG['dynamics']} / {CFG['window_from']} / hold {CFG['hold']}；"
                 "E1 外径 0.20 m vs X3 0.71 m，与历史 X3 数字不可比；详见 docs/EVAL_STANDARD.md")
    for line in lines:
        log(line)
    if not DRY:
        with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
            f.write("```\n" + "\n".join(lines) + "\n```\n")


RUN = dict(pilot=pilot, collect=collect, audit=audit, preprocess=preprocess,
           train=train, eval=evaluate, summary=summary)


def preflight():
    """开跑前把会在几小时后才炸的缺件一次查完。"""
    need(f"{BUILD}/uavrgb_train", "先 bash tools/check.sh 构建，或设 UAV_RGB_BUILD_DIR")
    need(f"{BUILD}/uavrgb_infer", "同上")
    need(CFG["old_x3_ckpt"], "旧 X3 学生 ckpt")
    if CFG["dynamics"] == "cf_sitl":
        need(os.path.join(CFG["plugin_dir"], "libCfSitlSystem.so"),
             "按 sim/cf_sitl/gz/README.md 构建真库插件；不要指向 build-gz-stub 桩库")
    for name in CFG["record_groups"]:
        if name not in {g[0] for g in eval_groups()}:
            raise RuntimeError(f"record_groups 里的 {name} 不在 EVAL_GROUPS / extra_students")


def main():
    global DRY
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="输出根目录（数据盘，如 $DATA_ROOT/uav_rgb/e1-pipeline-<时间>）")
    p.add_argument("--from", dest="start", choices=STEPS, default="pilot", help="从哪一步开始（续跑）")
    p.add_argument("--dry-run", action="store_true", help="只打印命令")
    a = p.parse_args()
    DRY = a.dry_run
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    if not DRY:
        write_json(os.path.join(out, "config.json"),   # 这次实验到底用了哪套参数，事后不用翻 manifest
                   {"cfg": CFG, "groups": eval_groups(), "gz_env": GZ_ENV, "argv": sys.argv,
                    "repo": REPO, "build": BUILD, "python": PY,
                    "git": subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                                          capture_output=True, text=True).stdout.strip()})
        if os.path.exists(os.path.join(out, "DONE")):
            os.remove(os.path.join(out, "DONE"))   # 续跑时先摘掉旧标记，免得外面的等待脚本误判
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
    except Exception as e:   # noqa: BLE001 - 任一步失败就停，原因进 status/DONE
        rc = 1
        mark(out, last or "preflight", "failed", error=f"{type(e).__name__}: {e}")
        log(f"!!!! {last or 'preflight'} 失败：{e}")
    finally:
        if not DRY:
            with open(os.path.join(out, "DONE"), "w", encoding="utf-8") as f:
                f.write(f"rc={rc} {time.strftime('%F %T')} last_step={last}\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
