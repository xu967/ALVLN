#!/usr/bin/env python3
# uav_rgb 唯一实验、训练与闭环录像入口（已移除所有散落的 run*.py 包装脚本与废弃的 VLM 线性判停探针）。
# 所有实验超参数集中在下方 ENV_REGISTRY，直接运行即可：
#   python3 train.py --check          仅检查配置、路径与防覆盖守卫并打印分阶段预计耗时，不碰 GPU
#   python3 train.py                  默认主推路线 vln_3dgs（S1 三带列池化+深度门控绕障训练闭环 + G3 阶段路标分解与几何语义联合判停闭环，全集保存 MP4 视频与 index.html）
#   python3 train.py s1_direct        仅运行 S1-PureDirect（纯直线方位打包 -> R0 训练 -> 在线 DAgger -> R1 训练 -> 室内外闭环对比与录像）
#   python3 train.py s1_subgoal       运行 S1 A* 局部子目标特权上界训练与闭环对照
#   python3 train.py g3_vlm           仅运行 G3 层次化阶段路标 VLM + 机载深度联合判停（未见公寓 840865 反事实 12 集 + 841980 跨房间 5 集闭环录像）
#   python3 train.py openfly_eval     仅运行 OpenFly v2 多场景 UE4 AirSim（env16 / env23 / env18 seen20）在线闭环与全程 MP4 录像
#   python3 train.py openfly_v2       从头训练 OpenFly 6 场景 v2 模型 + 20 集时序对拍 + 多场景 UE4 闭环录像
#   python3 train.py full_all         串联运行 vln_3dgs + openfly_eval 全套实验与录像
#   UAVRGB_NAME=<新run名> python3 train.py   自定义重跑时换名，默认禁止覆盖已完成 run 目录
import difflib
import glob
import html
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

# ======================================================================= 配置区
# 代码读取的每个 UAVRGB_* 环境变量都登记在 ENV_REGISTRY：名字 -> 字符串字面量默认值，行尾写取值依据。
# 环境里出现表外的 UAVRGB_* 立即列出名字并拒绝启动，防止拼错开关导致对照实验白跑。
ENV_REGISTRY = {
    # ---- 运行标识与防覆盖 ----
    "UAVRGB_NAME": "exp-vln-3dgs-g3-20261117",                  # 默认实验目录名；已有 DONE(rc=0) 时拒绝覆盖
    "UAVRGB_OUT_ROOT": "/root/rivermind-data/uav_rgb",          # 所有数据集、权重、日志与 MP4 统一落数据盘（根分区仅剩 6.6G，rivermind-fs Inode 已满禁写）
    "UAVRGB_MODE": "vln_3dgs",                                  # vln_3dgs | s1_direct | s1_subgoal | g3_vlm | openfly_eval | openfly_v2 | full_all
    "UAVRGB_OVERWRITE": "0",                                    # 0 = 严禁覆盖已完成目录；1 = 允许重跑覆盖
    "UAVRGB_RESUME": "1",                                       # 1 = 守护进程意外中断重拉起时跳过已完成子阶段；0 = 拒绝进入非空目录
    # ---- S1 机载双帧 C++/CUDA 三带列池化 + 深度扇区门控导航学生 ----
    "UAVRGB_EXISTING_S1_COLLECT": "/root/rivermind-data/uav_rgb/exp-full_1019-1117-0828/s1_direct",  # 默认直接复用已校验的 90 集教师 RGB+Depth 示范（不存在时自动重采）
    "UAVRGB_STUDENT_CKPT": "",           # g3_vlm 单跑时可选指定已训好的三带列池化 C++ 权重；空串时自动先训 S1
    "UAVRGB_S1_DEMO_LIMIT": "25",        # 无现成采集时每个训练公寓采 25 集（3 场景共 75 集）
    "UAVRGB_S1_VAL_LIMIT": "15",         # 验证公寓 839992 采 15 集 train split 选模（严格隔离 test split）
    "UAVRGB_S1_DAGGER_LIMIT": "15",      # 每个训练公寓采 15 集在线 DAgger 纠偏（3 场景共 45 集，暴露近墙偏航恢复状态）
    "UAVRGB_S1_DAGGER_BETA": "0.3",      # DAgger 30% 步执行刹停教师 v5、70% 步由学生自主飞，全部帧记录专家避障标签
    "UAVRGB_S1_SEARCH_DROPOUT": "0.45",  # 将 45% 左转帧（出生首步放大 1.8x）映射为 BEARING_UNKNOWN(8)，消除出生悬停死锁
    "UAVRGB_S1_EPOCHS_R0": "15",         # 三带列池化瓶颈(3840->5)参数减 5 倍，15 轮余弦退火收敛且不过拟合
    "UAVRGB_S1_EPOCHS_DAGGER": "25",     # DAgger 聚合轮数（约 11,000 帧，配合最佳宏 F1 选模）
    "UAVRGB_S1_BATCH": "32",             # 显存占用 < 1.5 GiB，梯度方差与更新步数平衡点
    "UAVRGB_S1_LR": "0.001",             # C++ AdamW 初始学习率 1e-3 配 --cosine-lr
    "UAVRGB_S1_WEIGHT_DECAY": "0.0001",  # 解耦权重衰减，抑制分类头后期验证集损失回升
    "UAVRGB_S1_AUGMENT": "all",          # 同时开启光度扰动 + ±4px 平移 + 50% 水平镜像（自动同步翻转动作、方位桶与深度列）
    "UAVRGB_S1_AUX_LAMBDA": "1.5",       # 未受 FiLM 调制的 p3 8 带物理深度辅助损失权重，支撑在线前视三扇区避障与判停
    "UAVRGB_S1_CLASS_WEIGHTS": "1.0,0.75,1.35,1.35",  # 温和抬高左右偏航避障权重(1.8x)，纠正纯直线方位下遇墙仍直冲偏置
    "UAVRGB_SEED": "20261117",           # 全局确定性随机种子
    # ---- 视频全量录制开关（用户要求：实验必须保存好视频供直接查看效果）----
    "UAVRGB_VIDEOS_PER_SCENE": "999",    # 999 = 评测集中每一集都渲染合成 10 FPS 四格指令横幅 MP4 视频（不抽样漏录）
    # ---- G3 零特权层次化视觉语言大模型（UAV-PUR-4B：FINAL/SUBGOAL 阶段路标分解 + 机载深度联合判停）----
    "UAVRGB_VLM_MODEL": "/root/rivermind-fs/data/Drone-PUR/models/UAV-PUR-4B",  # 只读挂载 Qwen3-VL-4B（共享卡自动分层）
    "UAVRGB_VLM_EVERY": "2",             # hold=0.5s 下每 2 步（1 Hz）更新一次 VLM 阶段路标信念
    # ---- OpenFly v2 多场景训练与 UE4 AirSim 在线闭环录像 ----
    "UAVRGB_OPENFLY_CKPT_DIR": "/root/rivermind-data/uav_rgb/exp-openfly-v2-20261116/training",  # 已训好的 v2_cma_rgb_language / v2_cma_rgb_only 权重目录
    "UAVRGB_OPENFLY_MANIFEST": "/root/rivermind-data/uav_rgb/downloads/OpenFly/subsets/six_scene_causal_bc_v1/manifest.json",  # 6 场景 100 路线空间簇隔离清单
    "UAVRGB_OPENFLY_SEEN_TASKS": "/root/rivermind-data/uav_rgb/exp-openfly-optimization-20261116/official_seen_tasks.json",    # 官方 env18 seen20 任务清单（含 episode_014）
    "UAVRGB_OPENFLY_SCENES": "env_airsim_16,env_airsim_23,env_airsim_18",  # 依次拉起 UE4 跑在线闭环并录视频的三个城市场景
    "UAVRGB_OPENFLY_SEEN_LIMIT": "20",   # env_airsim_18 跑满全部 20 条 seen20 任务并逐集生成 decisions.mp4
    "UAVRGB_OPENFLY_EPOCHS": "15",       # openfly_v2 重训模式：80 train × 15 epochs = 1,200 updates
    "UAVRGB_OPENFLY_STOP_WEIGHT": "3.0", # OpenFly v2 末步 STOP 乘性损失权重
    "UAVRGB_OPENFLY_PREV_DROPOUT": "0.2",  # 因果 prev_action 20% 整步置零 dropout，打破原地摆头吸引子
    "UAVRGB_DISPLAY": ":20",             # 容器内已运行的 Xorg 显示服务号（UE4 Vulkan 离屏渲染依赖）
    "UAVRGB_AIRSIM_PORT": "41458",       # AirSim RPC 端口（单实例串行复用，退出必清理进程组）
    # ---- 底层运行时开关（兼容 env.sh、C++ 与仿真子模块读取的 UAVRGB_*）----
    "UAVRGB_CUDA_BLOCKING_SYNC": "1",    # 训练期开启同步错误定位与确定性统计
    "UAVRGB_RENDER_BACKEND": "",         # 空串由各子模块用默认后端（3DGS 用 gsplat CUDA）
    "UAVRGB_GZ_PYTHON": "",              # env.sh 注入的 Gazebo 系统 Python 路径
    "UAVRGB_3DGS_PYTHON": "",            # 可选覆盖 3dgs-venv Python 路径
    "UAVRGB_FFMPEG": "",                 # 可选覆盖静态 ffmpeg 二进制路径
    "UAVRGB_VIDEO_QUALITY": "",          # 可选覆盖录像画质档 low|medium|high
    "UAVRGB_TRAIN_BIN": "",              # Studio 可选覆盖 uavrgb_train 路径
    "UAVRGB_SOURCE_GIT": "",             # C++ trainer 可选覆盖记录的源码 git 版本
    "UAVRGB_REPO_ROOT": "",              # 子模块可选覆盖仓库根目录
    "UAVRGB_XDG_RUNTIME_DIR": "",        # Gazebo XDG 运行时目录
    "UAVRGB_CF_SITL_PLUGIN_DIR": "",     # Gazebo CfSitlSystem 插件目录
}

REPO_ROOT = Path(__file__).resolve().parent
TRAIN_SCENES = ("sage3d_840139", "sage3d_841012", "sage3d_841150")  # 严格只用这 3 个公寓训练（留 840865 / 841980 作完全未见测试公寓）
VAL_SCENE = "sage3d_839992"                                         # 仅取其 train split 作选模，留 test_Instruction_Unseen 作测试
VALID_MODES = ("vln_3dgs", "s1_direct", "s1_subgoal", "g3_vlm", "openfly_eval", "openfly_v2", "full_all")
PRESETS = {
    "vln_3dgs": {"UAVRGB_MODE": "vln_3dgs", "UAVRGB_NAME": "exp-vln-3dgs-g3-20261117"},
    "s1_direct": {"UAVRGB_MODE": "s1_direct", "UAVRGB_NAME": "exp-s1-colpool-direct-20261117"},
    "s1_subgoal": {"UAVRGB_MODE": "s1_subgoal", "UAVRGB_NAME": "exp-s1-colpool-subgoal-20261117"},
    "g3_vlm": {"UAVRGB_MODE": "g3_vlm", "UAVRGB_NAME": "exp-g3-subgoal-vlm-20261117"},
    "openfly_eval": {"UAVRGB_MODE": "openfly_eval", "UAVRGB_NAME": "exp-openfly-v2-closedloop-20261117"},
    "openfly_v2": {"UAVRGB_MODE": "openfly_v2", "UAVRGB_NAME": "exp-openfly-v2-retrain-20261117"},
    "full_all": {"UAVRGB_MODE": "full_all", "UAVRGB_NAME": "exp-full-3dgs-g3-openfly-20261117"},
}
# =======================================================================


def _parse_cli_and_validate_env(argv):
    check_only = "--check" in argv
    pos_args = [a for a in argv if a != "--check"]
    if len(pos_args) > 1:
        raise SystemExit(f"[train.py] 最多接受 1 个预设模式参数 {VALID_MODES}，收到: {pos_args}")

    unknown = sorted(k for k in os.environ if k.startswith("UAVRGB_") and k not in ENV_REGISTRY)
    if unknown:
        hints = []
        for k in unknown:
            near = difflib.get_close_matches(k, list(ENV_REGISTRY), n=1, cutoff=0.6)
            hints.append(f"  {k}" + (f"（疑似 {near[0]}）" if near else ""))
        raise SystemExit(
            f"[train.py] 拒绝启动：环境里有 {len(unknown)} 个未登记的 UAVRGB_* 变量：\n"
            + "\n".join(hints)
            + "\n  拼错的开关不会生效。请核对变量名，或先登记到 train.py 顶部 ENV_REGISTRY。"
        )

    explicit_env = set(os.environ)
    preset = pos_args[0].lower() if pos_args else ""
    if preset and preset not in PRESETS:
        raise SystemExit(f"[train.py] 未知预设模式 {preset!r}，可选: {', '.join(VALID_MODES)}")

    for k, v in ENV_REGISTRY.items():
        if v:
            os.environ.setdefault(k, v)
    if preset:
        for k, v in PRESETS[preset].items():
            if k not in explicit_env:
                os.environ[k] = v

    mode = os.environ["UAVRGB_MODE"].strip()
    if mode not in VALID_MODES:
        raise SystemExit(f"[train.py] UAVRGB_MODE={mode!r} 非法，可选: {', '.join(VALID_MODES)}")
    for flag in ("UAVRGB_OVERWRITE", "UAVRGB_RESUME"):
        if os.environ.get(flag, "0") not in ("0", "1"):
            raise SystemExit(f"[train.py] {flag} 只接受 0 | 1，收到 {os.environ.get(flag)!r}")
    name = os.environ.get("UAVRGB_NAME", "").strip()
    if not name:
        raise SystemExit("[train.py] UAVRGB_NAME 不能为空")
    out_root = Path(os.environ["UAVRGB_OUT_ROOT"]).resolve()
    if str(out_root).startswith("/root/rivermind-fs"):
        raise SystemExit("[train.py] 严禁将输出目录设在 /root/rivermind-fs（JuiceFS Inode 已 100% 耗尽）")
    run_dir = out_root / name

    done_file = run_dir / "DONE"
    overwrite = os.environ["UAVRGB_OVERWRITE"] == "1"
    resume = os.environ["UAVRGB_RESUME"] == "1"
    if done_file.is_file() and "rc=0" in done_file.read_text(errors="ignore") and not overwrite:
        raise SystemExit(
            f"[train.py] 拒绝启动：目标目录已完成 ({done_file})；请换用新的 UAVRGB_NAME，或显式设 UAVRGB_OVERWRITE=1。"
        )
    if run_dir.exists() and any(run_dir.iterdir()) and not (overwrite or resume):
        raise SystemExit(
            f"[train.py] 拒绝启动：目标目录非空 ({run_dir}) 且 UAVRGB_RESUME=0, UAVRGB_OVERWRITE=0。"
        )
    return check_only, mode, run_dir


def _has_complete_s1_collect(base: Path) -> bool:
    col_tr = base / "collect_train"
    col_val = base / "collect_val"
    if not (base and str(base) != "." and (col_tr / "vln_eval.json").is_file() and (col_val / "vln_eval.json").is_file()):
        return False
    sample_depths = glob.glob(str(col_tr / "*/*/depth/depth_00000.npy"))
    return len(sample_depths) > 0


def estimate_eta_minutes(mode: str):
    """基于本机 RTX 2080 Ti + 12 核 cgroup 实测吞吐给出各子阶段预计耗时（分钟）。"""
    stages = []
    ext_raw = os.environ.get("UAVRGB_EXISTING_S1_COLLECT", "").strip()
    has_existing = _has_complete_s1_collect(Path(ext_raw)) if ext_raw else False
    has_ckpt = bool(os.environ.get("UAVRGB_STUDENT_CKPT", "").strip()) and Path(os.environ["UAVRGB_STUDENT_CKPT"]).is_file()
    need_s1 = mode in ("vln_3dgs", "full_all", "s1_direct", "s1_subgoal") or (mode == "g3_vlm" and not has_ckpt)
    if need_s1:
        if not has_existing:
            stages.append(("S1 教师 v5 初始示范采集 (90 集含渲染深度)", 7.5))
        stages.append(("S1 三带列池化+深度头训练 (R0 15轮 + 45集在线DAgger + R1 25轮)", 6.5))
        if mode != "g3_vlm":
            stages.append(("S1 室内留出 10 集四臂闭环 + 户外长航线 + 全量四格 MP4 (35 段视频)", 7.5))
    if mode in ("vln_3dgs", "full_all", "g3_vlm"):
        stages.append(("G3 阶段路标分解 VLM + 机载深度联合判停：未见公寓 840865 (12集) + 841980 (5集) 全录像", 8.5))
    if mode == "openfly_v2":
        stages.append(("OpenFly 6 场景 v2 模型训练 (2×1200 updates) + 20 集时序对拍", 7.0))
    if mode in ("full_all", "openfly_eval", "openfly_v2"):
        seen_lim = int(os.environ.get("UAVRGB_OPENFLY_SEEN_LIMIT", "20"))
        of_min = 4.5 + 0.65 * seen_lim
        stages.append((f"OpenFly v2 多场景 UE4 在线闭环与全量录像 (env16/env23/env18 seen{seen_lim})", round(of_min, 1)))
    stages.append(("统一视频展厅汇总 (videos/ + index.html)", 0.5))
    total_min = round(sum(m for _, m in stages) * 1.12, 1)
    return stages, total_min


def _print_effective_config(mode: str, run_dir: Path):
    print(f"[train.py] 模式={mode} | 输出目录={run_dir}", flush=True)
    for k, default_v in ENV_REGISTRY.items():
        cur = os.environ.get(k, "")
        mark = "*" if cur != default_v else " "
        note = f" (默认 {default_v!r})" if mark == "*" else ""
        print(f"  {mark} {k}={cur}{note}", flush=True)
    stages, total_min = estimate_eta_minutes(mode)
    print("[train.py] 预计分阶段耗时（RTX 2080 Ti 实测基准 + 12% 余量）：", flush=True)
    for title, mins in stages:
        print(f"  - {title}: ~{mins:.1f} 分钟", flush=True)
    print(f"[train.py] 总预计耗时: 约 {total_min:.0f} 分钟", flush=True)


def _paths():
    data_uav = Path("/root/rivermind-data/uav_rgb")
    return {
        "data_uav": data_uav,
        "py_3dgs": data_uav / "3dgs-venv/bin/python",
        "py_tools": REPO_ROOT / ".venv-tools/bin/python",
        "py_vlm": data_uav / "envs/vlm-venv/bin/python",
        "py_airvln": data_uav / "envs/airvln/bin/python",
        "train_bin": REPO_ROOT / "build/check/uavrgb_train",
        "infer_bin": REPO_ROOT / "build/check/uavrgb_infer",
        "cf_sitl_proc": data_uav / "tmp/build-cf-sitl/cf_sitl_proc",
        "gsplat_ext": data_uav / "torch-cache/extensions-sm75-20261109",
        "cache_root": data_uav / "cache",
        "airvln_root": data_uav / "airvln-runtime-ubuntu/AirVLN",
        "asset_inv": data_uav / "openfly-runtime-ubuntu/asset_inventory.json",
        "soc100_json": data_uav / "exp-wanderland-10nvb-20261114/soc100/e1_mini_soc100.json",
    }


def _preflight_files(mode: str):
    p = _paths()
    req = [p["py_tools"]]
    if mode in ("vln_3dgs", "full_all", "s1_direct", "s1_subgoal", "g3_vlm"):
        req += [p["py_3dgs"], p["train_bin"], p["infer_bin"], p["cf_sitl_proc"]]
    if mode in ("vln_3dgs", "full_all", "g3_vlm"):
        req += [p["py_vlm"], Path(os.environ["UAVRGB_VLM_MODEL"])]
    if mode in ("full_all", "openfly_eval", "openfly_v2"):
        req += [p["py_airvln"], p["airvln_root"], p["asset_inv"], Path(os.environ["UAVRGB_OPENFLY_MANIFEST"])]
    missing = [str(x) for x in req if not x.exists()]
    if missing:
        raise SystemExit("[train.py] 预检失败，缺少关键依赖或资产:\n  " + "\n  ".join(missing))


def _base_env():
    p = _paths()
    env = os.environ.copy()
    env.update({
        "DATA_ROOT": "/root/rivermind-data",
        "UAV_RGB_BUILD_DIR": str(REPO_ROOT / "build/check"),
        "CF_SITL_PROC": str(p["cf_sitl_proc"]),
        "TORCH_EXTENSIONS_DIR": str(p["gsplat_ext"]),
        "UAV_RGB_VLM_PYTHON": str(p["py_vlm"]),
        "UAV_RGB_CACHE_ROOT": str(p["cache_root"]),
        "TORCH_HOME": str(p["data_uav"] / "torch-cache"),
        "HF_HOME": str(p["data_uav"] / "hf-cache"),
        "TRANSFORMERS_CACHE": str(p["data_uav"] / "hf-cache/hub"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TMPDIR": str(p["data_uav"] / "tmp"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
    })
    return env


def _run_cmd(cmd, env, allow_rc1=False):
    print(f"\n[CMD {time.strftime('%H:%M:%S')}] {' '.join(str(c) for c in cmd)}", flush=True)
    res = subprocess.run([str(c) for c in cmd], cwd=str(REPO_ROOT), env=env, check=False)
    if res.returncode != 0 and not (allow_rc1 and res.returncode == 1):
        raise RuntimeError(f"命令失败 rc={res.returncode}: {' '.join(str(c) for c in cmd)}")


def _ensure_s1_base_demos(s1_dir: Path, env: dict):
    """复用已有含完整 RGB+Depth 的 75 train + 15 val 教师 v5 示范集；否则自动采集。"""
    p = _paths()
    ext_raw = os.environ.get("UAVRGB_EXISTING_S1_COLLECT", "").strip()
    if ext_raw and _has_complete_s1_collect(Path(ext_raw)):
        existing = Path(ext_raw)
        col_tr = existing / "collect_train"
        col_val = existing / "collect_val"
        print(f"[S1] 复用已有完整教师 v5 初始采集: {col_tr} & {col_val}", flush=True)
        return col_tr, col_val

    col_tr = s1_dir / "collect_train"
    col_val = s1_dir / "collect_val"
    tr_flags = []
    for s in TRAIN_SCENES:
        tr_flags += ["--scene", p["data_uav"] / f"3dgs-scenes/{s}/scene.json"]
    if not (col_tr / "vln_eval.json").is_file():
        _run_cmd([
            p["py_3dgs"], "-m", "sim.3dgs.run_vln", *tr_flags, "--split", "train",
            "--policy", "teacher", "--teacher-clearance", "braking",
            "--collect", "--save-depth", "--limit", os.environ["UAVRGB_S1_DEMO_LIMIT"],
            "--seed", "101", "--out", col_tr,
        ], env)
    if not (col_val / "vln_eval.json").is_file():
        _run_cmd([
            p["py_3dgs"], "-m", "sim.3dgs.run_vln",
            "--scene", p["data_uav"] / f"3dgs-scenes/{VAL_SCENE}/scene.json", "--split", "train",
            "--policy", "teacher", "--teacher-clearance", "braking",
            "--collect", "--save-depth", "--limit", os.environ["UAVRGB_S1_VAL_LIMIT"],
            "--seed", "201", "--out", col_val,
        ], env)
    return col_tr, col_val


def run_stage_s1(run_dir: Path, env: dict, g0_mode: str) -> dict:
    """运行 S1 机载双帧 C++/CUDA 学生训练、在线 DAgger 与全视频闭环评测（g0_mode = 'direct' 或 'subgoal'）。"""
    p = _paths()
    tag = "s1_direct" if g0_mode == "direct" else "s1_subgoal"
    s1_dir = run_dir / tag
    s1_dir.mkdir(parents=True, exist_ok=True)
    col_tr, col_val = _ensure_s1_base_demos(s1_dir, env)

    bearing_src = "direct" if g0_mode == "direct" else "recorded"
    dropout = os.environ["UAVRGB_S1_SEARCH_DROPOUT"]
    seed = os.environ["UAVRGB_SEED"]
    batch = os.environ["UAVRGB_S1_BATCH"]
    lr = os.environ["UAVRGB_S1_LR"]
    videos_n = os.environ["UAVRGB_VIDEOS_PER_SCENE"]

    ds_r0 = s1_dir / "ds_r0"
    run_r0 = s1_dir / "run_r0"
    col_dag = s1_dir / "collect_dagger"
    ds_dag = s1_dir / "ds_dagger"
    run_dag = s1_dir / "run_dagger"

    if not (ds_r0 / "dataset.uavr").is_file():
        eps = [d for d in sorted(glob.glob(str(col_tr / "*/*")) + glob.glob(str(col_val / "*/*")))
               if os.path.isfile(os.path.join(d, "episode_manifest.json"))]
        _run_cmd([
            p["py_tools"], "tools/preprocess.py",
            "--drop-collision", "--cond", "vln-nav",
            "--vln-bearing-source", bearing_src,
            "--vln-search-dropout", dropout,
            "--out", ds_r0, *eps,
        ], env)

    aux_lam = os.environ["UAVRGB_S1_AUX_LAMBDA"]
    wd = os.environ["UAVRGB_S1_WEIGHT_DECAY"]
    aug = os.environ["UAVRGB_S1_AUGMENT"]
    cw_raw = os.environ["UAVRGB_S1_CLASS_WEIGHTS"].strip()
    cw_flags = ["--class-weights", cw_raw] if cw_raw else []
    common_train_flags = [
        "--batch", batch, "--lr", lr, "--seed", seed,
        "--two-frame", "--vln-nav", "--stop-head",
        "--depth-aux", "--aux-lambda", aux_lam,
        "--augment", aug, "--weight-decay", wd, *cw_flags,
        "--cosine-lr", "--drop-collision", "--cuda-blocking-sync",
    ]

    if not (run_r0 / "ckpt_best.bin").is_file():
        _run_cmd([
            p["train_bin"], "--data", ds_r0 / "dataset.uavr", "--out", run_r0,
            "--epochs", os.environ["UAVRGB_S1_EPOCHS_R0"], *common_train_flags,
        ], env)

    if not (col_dag / "vln_eval.json").is_file():
        tr_flags = []
        for s in TRAIN_SCENES:
            tr_flags += ["--scene", p["data_uav"] / f"3dgs-scenes/{s}/scene.json"]
        _run_cmd([
            p["py_3dgs"], "-m", "sim.3dgs.run_vln", *tr_flags, "--split", "train",
            "--policy", "student", "--ckpt", run_r0 / "ckpt_best.bin", "--infer-bin", p["infer_bin"],
            "--g0-mode", g0_mode, "--teacher-clearance", "braking",
            "--collect", "--save-depth",
            "--dagger-beta", os.environ["UAVRGB_S1_DAGGER_BETA"], "--dagger-round", "1",
            "--limit", os.environ["UAVRGB_S1_DAGGER_LIMIT"], "--seed", "301", "--out", col_dag,
        ], env, allow_rc1=True)

    if not (ds_dag / "dataset.uavr").is_file():
        eps = [d for d in sorted(
            glob.glob(str(col_tr / "*/*")) + glob.glob(str(col_dag / "*/*")) + glob.glob(str(col_val / "*/*"))
        ) if os.path.isfile(os.path.join(d, "episode_manifest.json"))]
        _run_cmd([
            p["py_tools"], "tools/preprocess.py",
            "--drop-collision", "--cond", "vln-nav",
            "--vln-bearing-source", bearing_src,
            "--vln-search-dropout", dropout,
            "--out", ds_dag, *eps,
        ], env)

    ckpt_final = run_dag / "ckpt_best.bin"
    if not ckpt_final.is_file():
        _run_cmd([
            p["train_bin"], "--data", ds_dag / "dataset.uavr", "--out", run_dag,
            "--epochs", os.environ["UAVRGB_S1_EPOCHS_DAGGER"], *common_train_flags,
        ], env)

    eval_scenes = [
        "--scene", p["data_uav"] / "3dgs-scenes/sage3d_839992/scene.json",
        "--scene", p["data_uav"] / "3dgs-scenes/sage3d_841980/scene.json",
        "--split", "test_Instruction_Unseen",
        "--split", "test_Scene_Unseen",
    ]
    eval_arms = {
        f"student_g0_{g0_mode}": [
            "--policy", "student", "--ckpt", ckpt_final, "--infer-bin", p["infer_bin"],
            "--g0-mode", g0_mode, "--videos", videos_n,
        ],
        f"student_blind_rgb_{g0_mode}": [
            "--policy", "student", "--ckpt", ckpt_final, "--infer-bin", p["infer_bin"],
            "--g0-mode", g0_mode, "--blind-rgb", "--videos", videos_n,
        ],
        "student_blind_unknown": [
            "--policy", "student", "--ckpt", ckpt_final, "--infer-bin", p["infer_bin"],
            "--grounder", "blind_unknown", "--videos", "2",
        ],
        "teacher_v5": [
            "--policy", "teacher", "--teacher-clearance", "braking", "--videos", "4",
        ],
    }
    heldout = {}
    for arm, flags in eval_arms.items():
        adir = s1_dir / f"eval_{arm}"
        if not (adir / "vln_eval.json").is_file():
            _run_cmd([p["py_3dgs"], "-m", "sim.3dgs.run_vln", *eval_scenes, *flags, "--out", adir], env, allow_rc1=True)
        rep = json.loads((adir / "vln_eval.json").read_text())
        heldout[arm] = {"overall": rep["overall"], "by_split": rep.get("by_split")}

    wl_dir = s1_dir / f"eval_wanderland_{g0_mode}"
    if not (wl_dir / "vln_eval.json").is_file():
        _run_cmd([
            p["py_3dgs"], "-m", "sim.3dgs.run_vln",
            "--scene", p["data_uav"] / "3dgs-scenes/wanderland_1MlUucug/scene.json",
            "--policy", "student", "--ckpt", ckpt_final, "--infer-bin", p["infer_bin"],
            "--g0-mode", g0_mode, "--dynamics-params", p["soc100_json"],
            "--limit", "3", "--videos", videos_n, "--out", wl_dir,
        ], env, allow_rc1=True)
    wl_eval = json.loads((wl_dir / "vln_eval.json").read_text())["overall"]

    return {
        "g0_mode": g0_mode,
        "bearing_source": bearing_src,
        "ckpt_best": str(ckpt_final),
        "col_tr": str(col_tr),
        "col_val": str(col_val),
        "heldout_eval": heldout,
        "wanderland_outdoor_long_eval": wl_eval,
    }


def _build_unseen_vlm_cf_scene(out_json: Path) -> Path:
    """未见公寓 sage3d_840865 两组同起点异目标真实指令对（与 §10.18 基线严格同任务对比）。"""
    data_uav = _paths()["data_uav"]
    src = json.loads((data_uav / "3dgs-scenes/sage3d_840865/scene.json").read_text())
    src["ply"] = str((data_uav / "3dgs-scenes/sage3d_840865/scene.ply").resolve())
    src["collision"]["geometry"] = str((data_uav / "3dgs-scenes/sage3d_840865/collision_mesh.ply").resolve())
    routes = src["routes"]
    p1a, p1b = json.loads(json.dumps(routes[53])), json.loads(json.dumps(routes[71]))
    p1b["spawn"] = dict(p1a["spawn"])
    p2a, p2b = json.loads(json.dumps(routes[17])), json.loads(json.dumps(routes[67]))
    p2b["spawn"] = dict(p2a["spawn"])
    picked = [p1a, p1b, p2a, p2b]
    src["routes"] = picked
    src["spawn"] = picked[0]["spawn"]
    src["waypoint"] = picked[0]["waypoint"]
    src["max_steps"] = max(int(r["max_steps"]) for r in picked)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(src, indent=2, ensure_ascii=False) + "\n")
    return out_json


def _summarize_vln_dir(cdir: Path) -> dict:
    rep = json.loads((cdir / "vln_eval.json").read_text())
    eps = [json.loads(ln) for ln in (cdir / "episodes.jsonl").read_text().splitlines() if ln.strip()]
    return {
        "overall": rep["overall"],
        "vlm_latency_ms": rep.get("vlm_latency_ms"),
        "episodes": [
            {
                "route": r["route_name"],
                "goal_object": r.get("goal_object"),
                "supplied": r["supplied_text"][:80],
                "success": r["success"],
                "oracle_success": r["oracle_success"],
                "ne_m": round(r["ne_m"], 3),
                "spl": round(r["spl"], 3) if isinstance(r.get("spl"), (int, float)) else None,
                "stopped": r["stopped"],
                "collision": r["collision"],
                "video": r.get("video"),
            }
            for r in eps
        ],
    }


def run_stage_g3_vlm(run_dir: Path, env: dict, student_ckpt: Path) -> dict:
    """运行 G3 零特权阶段路标分解 VLM + 机载几何深度联合判停：未见公寓 840865 反事实 12 集 + 841980 5 集全录像。"""
    p = _paths()
    g3_dir = run_dir / "g3_subgoal_vlm"
    cf_scene = _build_unseen_vlm_cf_scene(g3_dir / "sage3d_840865_cf4.json")
    videos_n = os.environ["UAVRGB_VIDEOS_PER_SCENE"]
    vlm_every = os.environ["UAVRGB_VLM_EVERY"]
    vlm_model = os.environ["UAVRGB_VLM_MODEL"]

    cf_results = {}
    for cf in ("correct", "swap", "zero"):
        cdir = g3_dir / f"vlm_student_{cf}"
        if not (cdir / "vln_eval.json").is_file():
            _run_cmd([
                p["py_3dgs"], "-m", "sim.3dgs.run_vln",
                "--scene", cf_scene, "--limit", "4",
                "--policy", "student", "--ckpt", student_ckpt, "--infer-bin", p["infer_bin"],
                "--grounder", "vlm", "--vlm-model", vlm_model, "--vlm-every", vlm_every,
                "--counterfactual", cf, "--videos", videos_n, "--out", cdir,
            ], env, allow_rc1=True)
        cf_results[cf] = _summarize_vln_dir(cdir)

    unseen_841980_dir = g3_dir / "unseen_841980_vlm_correct"
    if not (unseen_841980_dir / "vln_eval.json").is_file():
        _run_cmd([
            p["py_3dgs"], "-m", "sim.3dgs.run_vln",
            "--scene", p["data_uav"] / "3dgs-scenes/sage3d_841980/scene.json",
            "--split", "test_Scene_Unseen", "--limit", "5",
            "--policy", "student", "--ckpt", student_ckpt, "--infer-bin", p["infer_bin"],
            "--grounder", "vlm", "--vlm-model", vlm_model, "--vlm-every", vlm_every,
            "--videos", videos_n, "--out", unseen_841980_dir,
        ], env, allow_rc1=True)

    return {
        "student_ckpt": str(student_ckpt),
        "unseen_840865_counterfactual": cf_results,
        "unseen_841980_scene_eval": _summarize_vln_dir(unseen_841980_dir),
    }


def _wait_tcp(port: int, timeout_s: float, proc=None) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(1.0)
    return False


def _wait_port_closed(port: int, timeout_s: float = 15.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return
        time.sleep(0.5)


def _write_airsim_settings(dst: Path, port: int):
    ref = Path("/root/rivermind-data/uav_rgb/exp-openfly-optimization-20261116/settings.json")
    doc = json.loads(ref.read_text())
    doc["ApiServerPort"] = int(port)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(doc, indent=2) + "\n")
    os.chmod(dst, 0o644)


def summarize_openfly_job(job_name: str, scene: str, sdoc: dict) -> dict:
    """汇总本地运动学诊断。OSR 分子取整段有效观测最近距离。"""
    metric_protocol = "uavrgb.openfly_local_metrics.v2"
    if sdoc.get("metric_protocol") != metric_protocol:
        raise ValueError(f"OpenFly 汇总协议不匹配或过旧，拒绝复用运行结果: {job_name}")
    eps = sdoc.get("episodes", [])
    requested = int(sdoc.get("n_requested", len(eps)))
    observed = [r for r in eps if isinstance(r.get("min_goal_distance_m"), (int, float))]
    osr_count = sum(1 for r in eps if bool(r.get("ever_within_20m_diagnostic")))
    final_count = sum(1 for r in eps if bool(r.get("within_20m_diagnostic")))
    stop_count = sum(1 for r in eps if bool(r.get("stopped_within_20m_diagnostic")))
    collisions = sum(1 for r in eps if str(r.get("termination", "")).startswith("collision"))
    videos_ok = sum(1 for r in eps if (Path(r["output"]) / "decisions.mp4").is_file()
                    and r.get("video_error") is None)
    final_distances = [r["final_goal_distance_m"] for r in eps
                       if isinstance(r.get("final_goal_distance_m"), (int, float))]
    return {
        "metric_protocol": metric_protocol,
        "metric_note": "本地估算诊断，不是官方 OpenFly 指标",
        "scene": scene,
        "n_requested": requested,
        "n_completed": len(eps),
        "n_with_valid_observation": len(observed),
        "errors": sdoc.get("errors", 0),
        "stop_lt_20m": stop_count,
        "sr_stop20": round(stop_count / requested, 4) if requested else 0.0,
        "estimated_local_osr_lt_20m": osr_count,
        "estimated_local_osr": round(osr_count / requested, 4) if requested else 0.0,
        "estimated_local_osr_valid_observation_fraction":
            round(osr_count / len(observed), 4) if observed else None,
        "final_frame_lt_20m_diagnostic": final_count,
        "final_frame_fraction": round(final_count / requested, 4) if requested else 0.0,
        "collisions": collisions,
        "mean_ne_m": round(sum(final_distances) / len(final_distances), 2) if final_distances else None,
        "videos_ok": videos_ok,
        "protocol_errors": sdoc.get("protocol_errors", 0),
        "video_errors": sdoc.get("video_errors", 0),
    }


def run_stage_openfly(run_dir: Path, env: dict, retrain: bool = False) -> dict:
    """运行 OpenFly v2（可选重训）与多场景 UE4 AirSim 在线闭环评测，逐集保存 decisions.mp4 录像。"""
    p = _paths()
    of_dir = run_dir / "openfly_v2"
    of_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(os.environ["UAVRGB_OPENFLY_MANIFEST"])

    if retrain:
        train_root = of_dir / "training"
        train_root.mkdir(parents=True, exist_ok=True)
        arms = [
            ("v2_cma_rgb_only", [
                "--policy-type", "cma", "--training-mode", "rgb_only",
                "--imagenet-norm", "--inflection-weight", "auto",
                "--stop-weight", os.environ["UAVRGB_OPENFLY_STOP_WEIGHT"], "--progress-monitor",
                "--use-prev-action", "--prev-action-dropout", os.environ["UAVRGB_OPENFLY_PREV_DROPOUT"],
            ]),
            ("v2_cma_rgb_language", [
                "--policy-type", "cma", "--training-mode", "rgb_language",
                "--imagenet-norm", "--use-bert", "--inflection-weight", "auto",
                "--stop-weight", os.environ["UAVRGB_OPENFLY_STOP_WEIGHT"], "--progress-monitor",
                "--use-prev-action", "--prev-action-dropout", os.environ["UAVRGB_OPENFLY_PREV_DROPOUT"],
            ]),
        ]
        for arm_name, extra in arms:
            adir = train_root / arm_name
            if not (adir / "DONE").is_file():
                _run_cmd([
                    p["py_airvln"], "tools/train_openfly_subset.py",
                    "--manifest", manifest_path, "--airvln-root", p["airvln_root"],
                    "--out", adir, "--epochs", os.environ["UAVRGB_OPENFLY_EPOCHS"],
                    "--seed", os.environ["UAVRGB_SEED"], *extra,
                ], env)
    else:
        train_root = Path(os.environ["UAVRGB_OPENFLY_CKPT_DIR"])

    ckpt_lang = train_root / "v2_cma_rgb_language/model.pt"
    ckpt_rgb = train_root / "v2_cma_rgb_only/model.pt"
    for ck in (ckpt_lang, ckpt_rgb):
        if not ck.is_file():
            raise FileNotFoundError(f"找不到 OpenFly v2 权重: {ck}")

    inv = json.loads(p["asset_inv"].read_text())
    scene_bins = {row["scene"]: (Path(row["binaries"][0]["path"]), Path(row["start_script"]).parent) for row in inv["rows"]}
    port = int(os.environ["UAVRGB_AIRSIM_PORT"])
    display = os.environ["UAVRGB_DISPLAY"]
    settings_path = of_dir / "airsim_settings.json"
    _write_airsim_settings(settings_path, port)
    os.chmod(of_dir, 0o755)

    scenes = [s.strip() for s in os.environ["UAVRGB_OPENFLY_SCENES"].split(",") if s.strip()]
    seen_tasks_orig = Path(os.environ["UAVRGB_OPENFLY_SEEN_TASKS"])
    seen_tasks = of_dir / "official_seen_tasks_bound.json"
    if seen_tasks_orig.is_file():
        import hashlib
        mh = hashlib.sha256()
        with open(manifest_path, "rb") as mf:
            for chunk in iter(lambda: mf.read(1 << 20), b""):
                mh.update(chunk)
        sdoc_bind = json.loads(seen_tasks_orig.read_text())
        sdoc_bind["training_manifest"] = str(manifest_path)
        sdoc_bind["training_manifest_sha256"] = mh.hexdigest()
        seen_tasks.write_text(json.dumps(sdoc_bind, indent=2) + "\n")
    seen_limit = int(os.environ["UAVRGB_OPENFLY_SEEN_LIMIT"])
    closedloop_summary = {}

    for scene in scenes:
        if scene not in scene_bins:
            raise ValueError(f"asset_inventory.json 未包含场景 {scene}")
        if scene == "env_airsim_18" and seen_tasks.is_file():
            jobs = [
                ("seen20_v2_cma_rgb_language", ckpt_lang, "correct", ["--evaluation-tasks", seen_tasks, "--limit", str(seen_limit), "--max-steps", "100", "--timeout", "360"]),
                ("seen20_v2_cma_rgb_language_swap", ckpt_lang, "swap", ["--evaluation-tasks", seen_tasks, "--limit", str(seen_limit), "--max-steps", "100", "--timeout", "360"]),
                ("seen20_v2_cma_rgb_language_zero", ckpt_lang, "zero", ["--evaluation-tasks", seen_tasks, "--limit", str(seen_limit), "--max-steps", "100", "--timeout", "360"]),
                ("seen20_v2_cma_rgb_only", ckpt_rgb, "correct", ["--evaluation-tasks", seen_tasks, "--limit", str(seen_limit), "--max-steps", "100", "--timeout", "360"]),
                ("seen20_blind_forward", None, "correct", ["--evaluation-tasks", seen_tasks, "--limit", str(seen_limit), "--max-steps", "100", "--timeout", "360"]),
            ]
        else:
            jobs = [
                (f"{scene}_val_v2_cma_rgb_language_correct", ckpt_lang, "correct", ["--scene", scene, "--limit", "20", "--max-steps", "40", "--timeout", "240"]),
                (f"{scene}_val_v2_cma_rgb_language_swap", ckpt_lang, "swap", ["--scene", scene, "--limit", "20", "--max-steps", "40", "--timeout", "240"]),
                (f"{scene}_val_v2_cma_rgb_language_zero", ckpt_lang, "zero", ["--scene", scene, "--limit", "20", "--max-steps", "40", "--timeout", "240"]),
                (f"{scene}_val_v2_cma_rgb_only", ckpt_rgb, "correct", ["--scene", scene, "--limit", "20", "--max-steps", "40", "--timeout", "240"]),
                (f"{scene}_val_blind_forward", None, "correct", ["--scene", scene, "--policy", "blind_forward", "--limit", "20", "--max-steps", "40", "--timeout", "240"]),
            ]

        pending_jobs = []
        for job in jobs:
            summary_path = of_dir / "closedloop" / job[0] / "summary.json"
            if not summary_path.is_file():
                pending_jobs.append(job)
                continue
            summary_doc = json.loads(summary_path.read_text())
            if summary_doc.get("metric_protocol") != "uavrgb.openfly_local_metrics.v2":
                raise ValueError(
                    f"OpenFly 旧 summary 不可复用: {summary_path}。请使用新的运行名称，或先备份并移走旧评测目录。"
                )
        if pending_jobs:
            bin_path, ue_cwd = scene_bins[scene]
            ue_log = of_dir / f"ue4_{scene}.log"
            _wait_port_closed(port, timeout_s=10.0)
            ue_cmd = [
                "runuser", "-u", "ubuntu", "--", "env",
                f"DISPLAY={display}", "QT_X11_NO_MITSHM=1", "SDL_HINT_CUDA_DEVICE=0",
                str(bin_path), "-RenderOffscreen", "-NoSound", "-NoVSync", "-GraphicsAdapter=0",
                "--settings", str(settings_path),
            ]
            print(f"\n[UE4] 启动场景 {scene}: {' '.join(ue_cmd)}", flush=True)
            with open(ue_log, "ab") as lf:
                ue_proc = subprocess.Popen(ue_cmd, cwd=str(ue_cwd), stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                if not _wait_tcp(port, timeout_s=120.0, proc=ue_proc):
                    raise RuntimeError(f"UE4 场景 {scene} 启动失败或端口 {port} 超时，详见 {ue_log}")
                time.sleep(7.0)  # 等待首帧纹理解码与自动曝光稳定（防止 env23 首帧全黑）
                for job_name, ckpt, lang_mode, extra_flags in pending_jobs:
                    jout = of_dir / "closedloop" / job_name
                    if jout.exists():
                        shutil.rmtree(jout)
                    eval_cmd = [p["py_airvln"], "tools/eval_openfly_subset.py"]
                    if ckpt is not None:
                        eval_cmd.extend(["--checkpoint", ckpt, "--airvln-root", p["airvln_root"]])
                    elif lang_mode == "correct":
                        eval_cmd.extend(["--policy", "blind_forward"])
                    _run_cmd([
                        *eval_cmd,
                        "--manifest", manifest_path,
                        "--language", lang_mode,
                        "--port", str(port),
                        "--record-video",
                        *extra_flags,
                        "--out", jout,
                    ], env, allow_rc1=True)
            finally:
                try:
                    os.killpg(ue_proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    ue_proc.wait(timeout=12.0)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(ue_proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    ue_proc.wait(timeout=5.0)
                _wait_port_closed(port, timeout_s=15.0)

        for job_name, _, _, _ in jobs:
            sfile = of_dir / "closedloop" / job_name / "summary.json"
            sdoc = json.loads(sfile.read_text())
            closedloop_summary[job_name] = summarize_openfly_job(job_name, scene, sdoc)
    return closedloop_summary


def build_video_gallery(run_dir: Path) -> dict:
    """将本次实验生成的全部 3DGS views.mp4 与 OpenFly decisions.mp4 汇总链接至 videos/ 并生成可点播 index.html。"""
    vid_dir = run_dir / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)
    cards = []

    # 1) 收集 3DGS 四格闭环视频
    for eval_json in sorted(run_dir.glob("*/*/vln_eval.json")):
        arm_dir = eval_json.parent
        stage_name = arm_dir.parent.name
        arm_name = arm_dir.name
        ep_jsonl = arm_dir / "episodes.jsonl"
        if not ep_jsonl.is_file():
            continue
        for row in (json.loads(ln) for ln in ep_jsonl.read_text().splitlines() if ln.strip()):
            rel_v = row.get("video")
            if not rel_v:
                continue
            src_mp4 = arm_dir / rel_v
            if not src_mp4.is_file():
                continue
            flat_name = f"3dgs__{stage_name}__{arm_name}__{row['scene']}__{row['route_name']}_{row['instruction_index']}.mp4"
            dst_mp4 = vid_dir / flat_name
            if dst_mp4.is_symlink() or dst_mp4.exists():
                dst_mp4.unlink()
            os.symlink(os.path.relpath(src_mp4, vid_dir), dst_mp4)
            cards.append({
                "domain": "3DGS 机载闭环",
                "group": f"{stage_name} / {arm_name}",
                "scene": row["scene"],
                "episode": f"{row['route_name']} ({row.get('goal_object', '')})",
                "badge": "SUCCESS" if row.get("success") else ("COLLISION" if row.get("collision") else ("STOPPED" if row.get("stopped") else "TIMEOUT")),
                "metrics": f"SR={int(bool(row.get('success')))} | OSR={int(bool(row.get('oracle_success')))} | NE={row.get('ne_m', 0):.2f}m | SPL={row.get('spl', 0):.2f}",
                "instruction": row.get("supplied_text", ""),
                "rel_video": f"videos/{flat_name}",
            })

    # 2) 收集 OpenFly UE4 在线闭环视频
    for sum_json in sorted(run_dir.glob("openfly_v2/closedloop/*/summary.json")):
        job_dir = sum_json.parent
        job_name = job_dir.name
        sdoc = json.loads(sum_json.read_text())
        for idx, r in enumerate(sdoc.get("episodes", [])):
            src_mp4 = Path(r["output"]) / "decisions.mp4"
            if not src_mp4.is_file():
                continue
            flat_name = f"openfly__{job_name}__ep{idx:03d}.mp4"
            dst_mp4 = vid_dir / flat_name
            if dst_mp4.is_symlink() or dst_mp4.exists():
                dst_mp4.unlink()
            os.symlink(os.path.relpath(src_mp4, vid_dir), dst_mp4)
            ne = r.get("final_goal_distance_m")
            ok20 = bool(r.get("stopped_within_20m_diagnostic"))
            steps = r.get("executed_actions")
            cards.append({
                "domain": "OpenFly UE4 闭环",
                "group": job_name,
                "scene": r.get("route_id", "").split("/")[0],
                "episode": f"ep_{idx:03d} ({r.get('route_id', '')})",
                "badge": "SUCCESS" if ok20 else str(r.get("termination", "done")).upper(),
                "metrics": f"term={r.get('termination')} | NE={ne:.2f}m | stop<20m={int(ok20)} | steps={steps}" if isinstance(ne, (int, float)) else str(r.get("termination")),
                "instruction": r.get("supplied_instruction", ""),
                "rel_video": f"videos/{flat_name}",
            })

    rows_html = []
    for c in cards:
        color = "#1b7f3b" if c["badge"] == "SUCCESS" else ("#b91c1c" if "COLL" in c["badge"] else "#92400e")
        rows_html.append(
            f'<div class="card">'
            f'<div class="hdr"><span class="tag">{html.escape(c["domain"])}</span> '
            f'<b>{html.escape(c["group"])}</b> — {html.escape(c["scene"])} / {html.escape(c["episode"])} '
            f'<span class="badge" style="background:{color}">{html.escape(c["badge"])}</span></div>'
            f'<div class="meta">{html.escape(c["metrics"])}</div>'
            f'<div class="ins">{html.escape(c["instruction"])}</div>'
            f'<video controls preload="metadata" src="{html.escape(c["rel_video"])}"></video>'
            f'</div>'
        )
    page = (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        f"<title>UAV VLN 闭环视频展厅 ({html.escape(run_dir.name)})</title>"
        "<style>body{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;margin:24px}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(460px,1fr));gap:16px}"
        ".card{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px}"
        ".hdr{font-size:14px;margin-bottom:6px}.tag{background:#334155;padding:2px 6px;border-radius:4px;font-size:12px}"
        ".badge{color:#fff;padding:2px 8px;border-radius:4px;font-size:12px;float:right}"
        ".meta{font-size:12px;color:#94a3b8;margin-bottom:6px}"
        ".ins{font-size:12px;color:#cbd5e1;margin-bottom:8px;max-height:48px;overflow:auto}"
        "video{width:100%;border-radius:6px;background:#000}</style></head><body>"
        f"<h2>UAV VLN 实验视频总览：{html.escape(run_dir.name)}（共 {len(cards)} 段 MP4）</h2>"
        f"<div class='grid'>{''.join(rows_html)}</div></body></html>\n"
    )
    index_path = run_dir / "index.html"
    index_path.write_text(page, encoding="utf-8")
    return {"total_videos": len(cards), "videos_dir": str(vid_dir), "index_html": str(index_path)}


def main(argv=None):
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    check_only, mode, run_dir = _parse_cli_and_validate_env(raw_argv)
    _print_effective_config(mode, run_dir)
    _preflight_files(mode)
    if check_only:
        print("[train.py] --check 预检全部通过（未导入 torch，未触碰 GPU）。", flush=True)
        return 0

    t0 = time.time()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.pid").write_text(f"{os.getpid()}\n")
    done_file = run_dir / "DONE"
    if done_file.exists():
        done_file.unlink()
    (run_dir / "run_config.json").write_text(
        json.dumps({k: os.environ.get(k, "") for k in ENV_REGISTRY}, indent=2, ensure_ascii=False) + "\n"
    )

    env = _base_env()
    summary = {"schema": "uavrgb.unified_train_summary.v1", "mode": mode, "run_dir": str(run_dir)}

    s1_res = None
    if mode in ("vln_3dgs", "full_all", "s1_direct"):
        s1_res = run_stage_s1(run_dir, env, g0_mode="direct")
        summary["s1_pure_direct"] = s1_res
    elif mode == "s1_subgoal":
        s1_res = run_stage_s1(run_dir, env, g0_mode="subgoal")
        summary["s1_subgoal"] = s1_res

    if mode in ("vln_3dgs", "full_all", "g3_vlm"):
        if s1_res is not None:
            student_ckpt = Path(s1_res["ckpt_best"])
        else:
            ckpt_raw = os.environ.get("UAVRGB_STUDENT_CKPT", "").strip()
            if ckpt_raw and Path(ckpt_raw).is_file():
                student_ckpt = Path(ckpt_raw)
            else:
                s1_res = run_stage_s1(run_dir, env, g0_mode="direct")
                summary["s1_pure_direct"] = s1_res
                student_ckpt = Path(s1_res["ckpt_best"])
        summary["g3_subgoal_vlm"] = run_stage_g3_vlm(run_dir, env, student_ckpt)

    if mode in ("full_all", "openfly_eval", "openfly_v2"):
        summary["openfly_v2_closedloop"] = run_stage_openfly(run_dir, env, retrain=(mode == "openfly_v2"))

    summary["video_gallery"] = build_video_gallery(run_dir)
    summary["elapsed_s"] = round(time.time() - t0, 2)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    done_file.write_text(f"rc=0 finished_at={time.strftime('%Y-%m-%dT%H:%M:%S%z')} elapsed_s={summary['elapsed_s']}\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        out_root = Path(os.environ.get("UAVRGB_OUT_ROOT", "/root/rivermind-data/uav_rgb"))
        name = os.environ.get("UAVRGB_NAME", "exp-failed")
        rdir = out_root / name
        if rdir.is_dir():
            (rdir / "DONE").write_text(
                f"rc=1 failed_at={time.strftime('%Y-%m-%dT%H:%M:%S%z')} error={type(exc).__name__}: {exc}\n"
            )
        raise
