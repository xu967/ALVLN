"""卡 E2E-1 的四个臂与路径约定（summarize_e2e.py、offline_report.py 共用；lib.sh 里有一份同样的 bash 版，test_e2e_arms.py 逐项核对两边一致）。

臂（卡 E2E-1 的表）：
  a0  img/demo     现成权重 $SWA/cal_swa_s{s}_k10.bin，训练 run = exp-student-select-20261109/e1-noise/seed{s}/run
  a1  code/demo    demo 用 --cond heading-code 重打包，训练带 --prev-action，评测带 --student-cond heading-code
  a2  code/dagger  dagger 聚合集重打包，同上
  a3  img/dagger   现有 dagger 数据集，图像输入
"""
import os

D = os.environ.get("UAVRGB_DATA", "/root/rivermind-data/uav_rgb")
ARMS = ("a0", "a1", "a2", "a3")
SEEDS_TRAIN = (0, 1, 2)
NAME = {"a0": "a0-img-demo", "a1": "a1-code-demo", "a2": "a2-code-dagger", "a3": "a3-img-dagger"}
LABEL = {"a0": "A0 img/demo", "a1": "A1 code/demo", "a2": "A2 code/dagger", "a3": "A3 img/dagger"}
IS_CODE = {"a0": False, "a1": True, "a2": True, "a3": False}
DATASET = {"a0": "demo", "a1": "demo", "a2": "dagger", "a3": "dagger"}
SWA = os.path.join(D, "exp-swa-calib-20261110", "ckpt")
A0_RUN_DIR = os.path.join(D, "exp-student-select-20261109", "e1-noise")
DEMO_DS = os.path.join(D, "tmp", "agent6-distill", "trunc", "ds", "dataset.uavr")
DAGGER_DS = os.path.join(D, "exp-dagger-gz-20261109", "full-1109", "round_3", "ds", "dataset.uavr")


def data_path(out, arm):
    """训练数据集（也是校准与离线核对用的数据集）。"""
    return {"a0": DEMO_DS, "a1": os.path.join(out, "data", "demo-hc", "dataset.uavr"),
            "a2": os.path.join(out, "data", "dagger-hc", "dataset.uavr"), "a3": DAGGER_DS}[arm]


def run_dir(out, arm, seed):
    return os.path.join(A0_RUN_DIR, f"seed{seed}", "run") if arm == "a0" else os.path.join(out, "runs", f"{NAME[arm]}-s{seed}")


def ckpt_path(out, arm, seed):
    """交付权重：末 10 轮平均 + FORWARD 先验校准。"""
    return os.path.join(SWA, f"cal_swa_s{seed}_k10.bin") if arm == "a0" else os.path.join(out, "ckpt", f"cal_swa_{NAME[arm]}_s{seed}_k10.bin")


def evdir(out, arm, seed):
    """评测臂目录（含重跑的 <臂>-retry1 另算，在名字后加 -retry1）。"""
    return os.path.join(out, f"{NAME[arm]}-s{seed}")
