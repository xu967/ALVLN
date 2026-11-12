#!/usr/bin/env python3
"""E1 MINI 流水线：pilot 采集+审计 → 放量采集 → 审计 → 打包 → 训练 → 20 seed 闭环四组对照 → 汇总。守护进程跑。

运行（仓库根，先 source env.sh）：
  OUT=$DATA_ROOT/uav_rgb/e1-pipeline-$(date +%Y%m%d-%H%M); mkdir -p "$OUT"
  setsid nohup "$UAV_RGB_PYTHON" tools/e1_pipeline.py --out "$OUT" >> "$OUT/run.log" 2>&1 < /dev/null &
  # 用 >> 不用 >：误起第二个实例或续跑时，外层 shell 会在 Python 拿锁退出前先执行重定向，> 会把第一个实例正在写的 run.log 截断
  sleep 3; cat "$OUT/run.pid"   # 进程自己写 run.pid：pid= pgid= start=（/proc/<pid>/stat 第 22 列）cmd=；不要再 echo $! > run.pid，重复起会冲掉第一个的记录
  # 停：先核对 start 与该 PID 现在的启动时刻一致（容器重启后 PID 可能已被别的进程拿走），再 kill -- -<pgid>
看进度：tail -f $OUT/run.log；每步状态 $OUT/status.json；结束落 $OUT/DONE（rc + 时间 + 最后一步）。
续跑：同一 --out 原命令重跑自动跳过已完成的步骤，半成品的采集/打包/训练目录挪到 <目录>.partial-<时刻> 后重做；也可 --from train 指定起点。--dry-run 只打印命令不执行。
同一个 --out 只许一个实例：起跑先拿 $OUT/.pipeline.lock（flock），拿不到就退出 3、不改 $OUT 里任何文件；锁随进程消失，被杀后原命令可直接再起。
改参数只改 CFG，不改函数。
交付权重（CFG deliver，环境变量 E1_DELIVER）：默认 calswa = 训练后再做末 10 轮权重平均 + FORWARD 先验校准（tools/average_ckpt.py，
产物 run/ckpt_calswa.bin，校准推理占 GPU，卡由 CFG gpu / E1_GPU 指定），评测评它；E1_DELIVER=best 回到 trainer 存的 ckpt_best.bin。
calswa 要求 epochs 不小于 avg_last_k（10），否则 preflight 报错。
预计耗时（RTF 锁 1.0、软件渲染、4 实例，gz agent 11-07 实测 ≈ 50 s/集）：pilot 5 min；采集 300 集 ≈ 65 min；
训练 30 轮 ≈ 5 min（设备常驻 2669 样本/秒）；评测 4 组并行 ≈ 40 min。
"""
import argparse
import glob
import math
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("DATA_ROOT", "/root/rivermind-data")
BUILD = os.environ.get("UAV_RGB_BUILD_DIR", os.path.join(REPO, "build", "check"))  # check.sh 的构建目录
PY = sys.executable   # 用 $UAV_RGB_PYTHON 启动本脚本，子进程继承它（preprocess/eval 需要 PIL、numpy、cv2）
sys.path.insert(0, os.path.join(REPO, "tools"))
import eval_seeds  # noqa: E402  - 冻结评测 seed 的唯一来源
import pipeline_common as pc  # noqa: E402  - 三条流水线共用：续跑判定等

def _opt_float_env(name, default=None):
    raw = os.environ.get(name, "").strip()
    return default if not raw else float(raw)


# ───────────────────────────── 配置区（改这里） ─────────────────────────────
CFG = dict(
    # 载具 / 动力学（采集与四组评测共用，保证训练数据和评测协议是同一套）
    dynamics="cf_sitl",      # E1 MINI 固件在环；复现旧 X3 数字用 "x3_velocity" 并把 vehicle 改 "x3"
    vehicle="e1_mini",       # 选 sim/worlds/uav_rgb_e1.sdf；cf_sitl 必须显式给
    hold=1.0,                # 每动作仿真秒。沿用 X3 基线 1.0 便于对比；固件里杆量保持到下一次派发
    window_from="observation",   # 唯一被采纳的基线语义；dispatch 只用于复现 T7-R，数字不可混报
    action_latency=_opt_float_env("E1_ACTION_LATENCY", None),
                             # 观测到派发的固定仿真秒（--action-latency L，0 < L < hold 且要求 observation 语义）；缺省 None = 旧协议不传该开关；
                             # W1-0 第二轮标定值为 0.20（E1_ACTION_LATENCY 可覆盖；采集端依赖卡 W1-13 的 collect_parallel 支持）
    rtf=_opt_float_env("E1_RTF", None),
                             # Gazebo 世界实时因子（--rtf R，有限正数）；缺省 None = 用世界模板原值不传该开关（长课程/密林慢策略可用 0.5，E1_RTF 可覆盖）
    plugin_dir=os.environ.get("E1_CF_SITL_PLUGIN_DIR", f"{DATA_ROOT}/uav_rgb/tmp/build-cf-sitl/gz"),
                             # 真库插件（sim/cf_sitl/gz/README 的构建路径）。显式给，免得 collect_parallel 兜底找到桩库 build-gz-stub。
                             # 环境变量覆盖用于换库过渡期：库升版要重建这个目录，但正在跑的旧代码实验还在加载旧插件时，
                             # 新实验先指向另建的目录（2026-11-09 lib31：tmp/build-cf-sitl-lib31/gz）
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
    stop_at_goal=os.environ.get("E1_STOP_AT_GOAL", "0") == "1",
                             # DISTILL §3.1：真值 x 过成功线（最远柱 x + 1.0 m，与 eval 同口径）即结束本集。0417 的 12000 帧里
                             # 过线后 2155 帧（18%）93% 视锥内无柱、38% 标签是 YAW（看不见的目标点的函数）；离线截断重训无柱帧误转率
                             # 22%→5%，eval3 闭环 10/20→13/20（乱飞 6→2，碰撞 2→5）。开了后集长约 27–38 步，audit episode_length
                             # 按 manifest 放行。默认关：与 0417 基线逐位可比；对照实验用 E1_STOP_AT_GOAL=1 启动，不改这里
    tiebreak=os.environ.get("E1_TIEBREAK", "legacy"),
                             # DISTILL §3.3：A* 教师左右近平局（|Q_L−Q_R|<0.5 步，转弯帧 ~10%）的口径。"visible" 改向学生视锥内
                             # 最近柱的反侧（0417 的 12000 帧里会改 85 帧）；"legacy" 与旧标签逐位相同。默认 legacy；对照实验用
                             # E1_TIEBREAK=visible 启动
    teacher_planner=os.environ.get("E1_TEACHER_PLANNER", "astar"),
                             # 示范教师（采集与评测 teacher 组同一个）：astar = 真值 A* + 纯追踪（现有全部数据）；costmap = 测地场
                             # 代价 argmin（TEACHER-COST 2026-11-09：闭环 20/20、碰撞 0；用它重标的学生 14/20 vs 截断臂 12/20，
                             # p=0.754）。默认 astar 时命令行逐字不变；对照实验用 E1_TEACHER_PLANNER=costmap 启动
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
    # 交付权重（训练步之后；评测与后续步骤用它）
    deliver=os.environ.get("E1_DELIVER", "calswa"),
                             # calswa = 末 avg_last_k 轮权重平均 + FORWARD 先验校准（tools/average_ckpt.py → run/ckpt_calswa.bin），训练命令加 --save-every-epoch；
                             # best = trainer 按 dev 宏 F1 存的 run/ckpt_best.bin（改前的行为，训练命令不加 --save-every-epoch）。
                             # 11-10 闭环 60 集（走廊内）：ckpt_best 27 → calswa 39；只平均不校准 20（tools/average_ckpt.py 文件头）。
                             # epochs < avg_last_k 时 calswa 取不到窗口，preflight 与 train 都报错并提示改 best。环境变量 E1_DELIVER 切换。
                             # average_ckpt 只支持单帧纯视觉 ckpt（text_dim=0）：l1 那种 text_dim>0 的配置校准会报错停下，要设 best
    avg_last_k=10,           # calswa 平均末几轮；11-10 闭环只测过 10（average_ckpt.LAST_K），改它之前先在闭环上测
    gpu=os.environ.get("E1_GPU", "GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740"),
                             # 校准推理（average_ckpt 起 uavrgb_infer）钉的那张卡，写进 CUDA_VISIBLE_DEVICES。11-11 19:23 起 nvidia-smi -L 只列这一张
                             # （RTX 2080 Ti），用户指示 uav_rgb 放在它上。换机器或换卡用 E1_GPU 覆盖
    # 评测（四组同一批 test 场景；E1 机体、observation 语义、hold 1.0）
    eval_seeds=eval_seeds.csv(eval_seeds.EVAL_SEEDS_BASE_20),
                             # 20 个独立几何的冻结 test seed；数字与来源（pick_test_seeds 排除清单、淘汰的旧清单）只写在
                             # tools/eval_seeds.py。这里保持逗号串：其它 agent/脚本从 CFG["eval_seeds"] 读。
    eval_steps=80,           # 给足全程余量（T7 起 80）；碰撞/到达会提前结束
    corridor_half_width=float(os.environ.get("E1_CORRIDOR_HALF_WIDTH", "3.0")),
                             # 柱阵评测的横向边界（m），等于 scene_gen.Y_RANGE；11-10 发现不加它时 20/20 里有 8 集是绕出柱阵；
                             # 设 0 回到旧口径（命令不带该开关，metric_version 也不带 .corridor3m，新旧 SR 不可混报）
    old_x3_ckpt=f"{DATA_ROOT}/uav_rgb/t6-full-1640/run/ckpt_best.bin",   # T6 best（epoch 8）：X3 / x3_velocity / hold 1.0 训的
    record_groups=("teacher", "student-e1-new"),   # 这两组录三视角视频（≈10 MB/仿真秒暂存，合成后删）
    VIDEO_QUALITY="medium",  # 录像画质档（sim/tools/view_layout.VIDEO_QUALITY）：low=1280×720/CRF 28（预览）；medium=1280×720/CRF 22；
                             # high=2560×1440 画布、显示视角 2× 渲染、CRF 18（demo 用；编码约 3× 慢，追尾相机 1280×720 软件渲染
                             # 会把 gz RTF 砍一半，评测批量别用）。环境变量 UAVRGB_VIDEO_QUALITY 可覆盖
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

DRY = False   # main 里与 pipeline_common.DRY 一起置位；helper 都在 tools/pipeline_common.py（A4）
log, run, sh, read_json, write_json, mark, need = pc.log, pc.run_cmd, pc.sh, pc.read_json, pc.write_json, pc.mark, pc.need


def protocol_check(hold=None, window_from=None):
    """校验 action_latency 与 rtf（与 eval_closed_loop 参数约束一致，起跑前即拦下非法取值）。"""
    hold = CFG["hold"] if hold is None else hold
    window_from = CFG["window_from"] if window_from is None else window_from
    lat, rtf = CFG["action_latency"], CFG["rtf"]
    if lat is not None:
        if not (isinstance(lat, (int, float)) and not isinstance(lat, bool) and math.isfinite(lat) and lat > 0):
            raise RuntimeError(f"CFG action_latency={lat!r}（环境变量 E1_ACTION_LATENCY）必须是有限正数")
        if lat >= float(hold):
            raise RuntimeError(f"CFG action_latency={lat:g} 必须小于 hold={float(hold):g}")
        if window_from != "observation":
            raise RuntimeError(f"CFG action_latency 只能配 window_from='observation'，当前是 {window_from!r}")
    if rtf is not None:
        if not (isinstance(rtf, (int, float)) and not isinstance(rtf, bool) and math.isfinite(rtf) and rtf > 0):
            raise RuntimeError(f"CFG rtf={rtf!r}（环境变量 E1_RTF）必须是有限正数")


def vehicle_flags():
    """采集和评测共用的动力学/载具/协议参数。"""
    protocol_check()
    flags = ["--dynamics", CFG["dynamics"], "--vehicle", CFG["vehicle"],
             "--hold", str(CFG["hold"]), "--window-from", CFG["window_from"]]
    if CFG["action_latency"] is not None:
        flags += ["--action-latency", f"{CFG['action_latency']:g}"]
    if CFG["rtf"] is not None:
        flags += ["--rtf", f"{CFG['rtf']:g}"]
    return flags


def set_dir(out, spec):
    """集合的采集目录。base 仍叫 collection/：正在跑的 e1-pipeline-* 目录与 --from 续跑不用改名。"""
    return os.path.join(out, "collection" if spec == "base" else "collection-" + spec.replace(":", "-"))


def collect_into(col, episodes, base_seed, spec="base"):
    """collect_parallel 一集失败就 rc=1，300 集里丢 1 集是常态；按 manifest 的成功率判，不拿 rc 当成败。"""
    man = os.path.join(col, "collection_manifest.json")
    if not DRY and os.path.isfile(man):
        return log(f"跳过采集（已有 {man}）")
    if not DRY and os.path.isdir(col) and os.listdir(col):
        moved = pc.retire_dir(col, "partial", "没有 collection_manifest.json")
        log(f"{col} 上次采集没跑完，旧目录挪到 {moved}，重采")
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
    if CFG["teacher_planner"] != "astar":
        cmd += ["--teacher-planner", CFG["teacher_planner"]]
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
    if DRY or not os.path.isfile(report):
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
    if not DRY and os.path.isfile(os.path.join(ds, "dataset.uavr")):
        return log(f"跳过 preprocess（已有 {ds}/dataset.uavr）")
    if not DRY and os.path.isdir(ds) and os.listdir(ds):
        moved = pc.retire_dir(ds, "partial", "没有 dataset.uavr")
        log(f"{ds} 上次打包没跑完，旧目录挪到 {moved}，重打")
    eps = sorted(d for col in cols for s in ("train", "val", "test") for d in glob.glob(f"{col}/{s}/*/"))
    if not DRY and not eps:
        raise RuntimeError(f"{cols} 下没有 train/val/test 集目录")
    cmd = [PY, f"{REPO}/tools/preprocess.py", "--out", ds]
    if CFG["drop_collision"]:
        cmd.append("--drop-collision")
    if len(CFG["collect_sets"]) > 1:
        cmd.append("--allow-mixed-geometry")   # 柱阵 + 森林是两个 geometry_version；hold/window 仍由 preprocess 严格校验
    sh(cmd + (eps or [f"{col}/{{train,val,test}}/*/" for col in cols]))   # 集目录在 Python 里枚举，不依赖 shell 展开


def deliver_check():
    """交付权重配置查错：值不认识、calswa 的轮数不够（取不到末 avg_last_k 轮）都在起任何命令之前报错。preflight 与 train 都调。"""
    if CFG["deliver"] not in ("calswa", "best"):
        raise RuntimeError(f"CFG deliver={CFG['deliver']!r}（环境变量 E1_DELIVER）只能是 calswa 或 best")
    if CFG["deliver"] == "calswa" and CFG["epochs"] < CFG["avg_last_k"]:
        raise RuntimeError(f"deliver=calswa 要末 {CFG['avg_last_k']} 轮的权重平均，但 epochs={CFG['epochs']} 取不到；"
                           "改 E1_DELIVER=best（用 ckpt_best.bin），或把 CFG epochs 加到不小于 avg_last_k")


def deliver_ckpt(run_dir):
    """评测与后续步骤用的权重文件。"""
    return os.path.join(run_dir, "ckpt_calswa.bin" if CFG["deliver"] == "calswa" else "ckpt_best.bin")


def calibrate_swa(ds, run_dir):
    """deliver=calswa：末 avg_last_k 轮平均 + 校准 → run/ckpt_calswa.bin（要 GPU 跑真实 uavrgb_infer）。
    失败（工具非零退出）直接抛出，流水线停下，不退回 ckpt_best。average_ckpt 拒绝覆盖已有产物，
    续跑时 .bin 与旁边的 .json 都在才算做完；只剩 .bin 时再起一次，由工具拒绝并停下，人来处理。"""
    if CFG["deliver"] != "calswa":
        return
    swa = deliver_ckpt(run_dir)
    if os.path.isfile(swa) and os.path.isfile(swa + ".json"):
        return log(f"跳过校准平均：{swa} 已有")
    sh([PY, f"{REPO}/tools/average_ckpt.py", "--run", run_dir, "--data", f"{ds}/dataset.uavr", "--out", swa,
        "--last-k", str(CFG["avg_last_k"]), "--infer-bin", f"{BUILD}/uavrgb_infer"],
       env_extra={"CUDA_VISIBLE_DEVICES": CFG["gpu"]})
    need(swa, "average_ckpt 退出码 0 但没写出产物")


def train(out):
    ds, run_dir = os.path.join(out, "ds"), os.path.join(out, "run")
    need(os.path.join(ds, "dataset.uavr"), "先跑 preprocess")
    deliver_check()
    if not DRY and pc.train_finished(run_dir):   # 续跑：完成 = metrics.jsonl 末行 done:（ckpt 文件每轮都写，不算）
        log(f"跳过 train：{run_dir}/metrics.jsonl 已记 done")
        return calibrate_swa(ds, run_dir)   # 训练做完、校准平均没做完（被打断）时在这里补
    if not DRY and os.path.isdir(run_dir) and os.listdir(run_dir):
        moved = pc.retire_dir(run_dir, "partial", "训练没跑完（metrics.jsonl 末行不是 done:）")
        log(f"{run_dir} 上次训练没跑完，旧目录挪到 {moved}，重训")
    cmd = [f"{BUILD}/uavrgb_train", "--data", f"{ds}/dataset.uavr", "--out", run_dir,
           "--epochs", str(CFG["epochs"]), "--batch", str(CFG["batch"]), "--lr", repr(float(CFG["lr"])),
           "--seed", str(CFG["seed"]), "--class-weights", CFG["class_weights"]]
    if CFG["drop_collision"]:
        cmd.append("--drop-collision")
    if CFG["text_dim"] > 0:
        cmd += ["--text-dim", str(CFG["text_dim"])]
    elif os.path.isfile(f"{ds}/instructions.jsonl"):
        cmd.append("--vision-only")   # 数据带指令却要纯视觉基线时 trainer 要求显式声明
    if CFG["deliver"] == "calswa":
        cmd.append("--save-every-epoch")   # 平均要用末 avg_last_k 轮的 ckpt_epNN.bin（每个约 1.2 MB）
    sh(cmd)
    need(f"{run_dir}/ckpt_best.bin", "训练结束但没有 best ckpt，看 run/metrics.jsonl")
    calibrate_swa(ds, run_dir)


def eval_metric_version():
    """本次配置下 eval_closed_loop 会写进 eval.json 的 metric_version（续跑时据此识别旧口径并挪 .stale-）。"""
    return pc.metric_version_for(CFG["dynamics"], CFG["corridor_half_width"] if CFG["corridor_half_width"] > 0 else None)


def evaluate(out):
    """四组并行，各自 1 个 gz；eval_closed_loop 有失败集时 rc=1 但 eval.json 已写，四组全跑完再一起看。"""
    ckpts = dict(old_x3_ckpt=CFG["old_x3_ckpt"], new_ckpt=deliver_ckpt(os.path.join(out, "run")))
    for name, _policy, ckpt_key in eval_groups():
        if ckpt_key and ckpt_key not in ckpts:
            ckpts[ckpt_key] = ckpt_key      # 探索臂：键就是 ckpt 路径
    for key, path in ckpts.items():
        need(path, f"{key} 不存在")
    mv = eval_metric_version()
    procs = {}
    for name, policy, ckpt_key in eval_groups():
        edir = os.path.join(out, f"eval-{name}")
        ckpt = ckpts[ckpt_key] if ckpt_key else None
        state, moved, why = pc.settle_eval(edir, metric_version=mv, ckpt=ckpt)
        if state == "done":
            log(f"跳过 {name}：eval.json 完整")
            continue
        if moved:
            kind_msg = "与当前配置不一致" if state == "stale" else "上次没跑完"
            log(f"{name} {kind_msg}（{why}），旧目录挪到 {moved}，重跑")
        cmd = [PY, f"{REPO}/sim/tools/eval_closed_loop.py", "--policy", policy, "--task", "base",
               "--split", "test", "--seeds", CFG["eval_seeds"],
               "--episodes", str(len(CFG["eval_seeds"].split(","))),
               "--steps", str(CFG["eval_steps"]), "--out", edir,
               "--sim-timeout", str(CFG["sim_timeout"]), "--episode-timeout", str(CFG["episode_timeout"]),
               *vehicle_flags()]
        if CFG["corridor_half_width"] > 0:
            cmd += ["--corridor-half-width", f"{CFG['corridor_half_width']:g}"]
        if ckpt_key:
            cmd += ["--ckpt", ckpts[ckpt_key], "--infer-bin", f"{BUILD}/uavrgb_infer"]
        if policy == "teacher" and CFG["teacher_planner"] != "astar":
            cmd += ["--teacher-planner", CFG["teacher_planner"]]
        if name in CFG["record_groups"]:
            cmd += ["--record-video", "--video-quality", CFG["VIDEO_QUALITY"]]   # 需要 chase/top 相机：评测模板不加 --no-aux-cameras
        log_path = os.path.join(out, f"eval-{name}.log")
        log("$ " + " ".join(cmd) + f"  > {log_path}")
        if not DRY:
            procs[name] = pc.popen(cmd, env=dict(os.environ, **GZ_ENV),   # 登记：SIGTERM 时连 gz 一起收
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
    retired = pc.read_retired(out)
    if not DRY:
        doc = {"cfg": CFG, "groups": rows}
        if retired:
            doc["retired"] = retired
        write_json(os.path.join(out, "summary.json"), doc)
    fmt = "%-16s %6s %6s %6s %6s %5s %5s %8s  %s"
    versions = sorted({str(r["metric_version"]) for r in rows.values()})
    lines = []
    if len(versions) > 1:
        # sim09.v1 过线后继续飞（碰撞可能发生在过线之后）、v2 成功线即停：SR/coll 不可直接比。
        lines.append(f"⚠ metric_version 混合：{' / '.join(versions)}（v1 过线后继续飞，v2 成功线即停；"
                     "各组按自己那列的版本读，见 docs/EVAL_STANDARD.md §5）")
    lines.append(fmt % ("group", "SR", "coll", "stuck", "ffwd", "done", "fail", "p50ms", "metric_version"))
    for name, r in rows.items():
        f = lambda v, d=2: "-" if v is None else f"{v:.{d}f}"   # noqa: E731
        lines.append(fmt % (name, f(r["requested_success_rate"]), f(r["collision_rate"]), f(r["stuck_rate"]),
                            f(r["false_forward_rate"]), r["episodes_done"], r["failed_count"], f(r["infer_p50_ms"], 1),
                            r["metric_version"]))
    lines.append(f"口径：SR 分母 = 请求集 {len(CFG['eval_seeds'].split(','))}（失败集计入分母）；四组同一批 seed、"
                 f"E1 机体 {CFG['dynamics']} / {CFG['window_from']} / hold {CFG['hold']}；"
                 "E1 外径 0.20 m vs X3 0.71 m，与历史 X3 数字不可比；详见 docs/EVAL_STANDARD.md")
    lines.extend(pc.retired_lines(out))
    for line in lines:
        log(line)
    if not DRY:
        with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
            f.write("```\n" + "\n".join(lines) + "\n```\n")


RUN = dict(pilot=pilot, collect=collect, audit=audit, preprocess=preprocess,
           train=train, eval=evaluate, summary=summary)


def preflight():
    """开跑前把会在几小时后才炸的缺件一次查完。"""
    deliver_check()
    protocol_check()
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
    pc.DRY = DRY = a.dry_run
    out = os.path.abspath(a.out)
    if not DRY:   # dry-run 不建目录、不写文件（A6）
        os.makedirs(out, exist_ok=True)
        pc.lock_out(out)   # E-3：已有实例在跑就退出 3；要赶在 config.json / DONE 之前，第二个实例不能碰第一个的状态
        pc.write_run_pid(out)   # E-4：run.pid 记 pid / pgid / 启动时刻 / 命令行，停进程前能核对 PID 还是不是这个进程
        write_json(os.path.join(out, "config.json"),   # 这次实验到底用了哪套参数，事后不用翻 manifest
                   {"cfg": CFG, "groups": eval_groups(), "gz_env": GZ_ENV, "argv": sys.argv,
                    "repo": REPO, "build": BUILD, "python": PY,
                    "git": subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                                          capture_output=True, text=True).stdout.strip()})
        if os.path.exists(os.path.join(out, "DONE")):
            os.remove(os.path.join(out, "DONE"))   # 续跑时先摘掉旧标记，免得外面的等待脚本误判
    pc.install_signal_handlers()   # SIGTERM/Ctrl-C → Interrupted：收子进程组、status 标 interrupted、DONE 写真实 rc
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
