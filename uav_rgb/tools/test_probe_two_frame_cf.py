#!/usr/bin/env python3
"""probe_two_frame_cf 的纯 Python 部分：三个反事实 prev_index 的映射、反事实数据集的侧车重写、方向计数。
不需要 GPU / uavrgb_infer。运行（仓库根）：.venv-tools/bin/python tools/test_probe_two_frame_cf.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import preprocess as pp  # noqa: E402
import probe_two_frame_cf as probe  # noqa: E402
import select_ckpt as sc  # noqa: E402
from test_select_ckpt import write_uavr  # noqa: E402


def test_cf_prev_index():
    # ep A：step 0,1,3（step 2 被 --drop-collision 丢掉）→ 样本 0,1,2；ep B：step 0,1 → 样本 3,4
    episodes = [{"sample_start": 0, "sample_count": 3}, {"sample_start": 3, "sample_count": 2}]
    step = np.array([0, 1, 3, 0, 1])
    assert list(probe.cf_prev_index("self", episodes, step)) == [0, 1, 2, 3, 4]
    # next：step+1 存在才指过去；step 1 的下一步 2 被丢了 → 自身；末帧 → 自身；不跨集
    assert list(probe.cf_prev_index("next", episodes, step)) == [1, 1, 2, 4, 4]
    # 没有 step 时按集内位置
    assert list(probe.cf_prev_index("next", episodes, None)) == [1, 2, 2, 4, 4]
    sh = probe.cf_prev_index("shuffle", episodes, step, seed=0)
    assert sorted(sh[:3]) == [0, 1, 2] and sorted(sh[3:]) == [3, 4], f"shuffle 必须是集内置换：{list(sh)}"
    assert list(sh) == list(probe.cf_prev_index("shuffle", episodes, step, seed=0)), "同 seed 应可复现"
    try:
        probe.cf_prev_index("flip", episodes, step)
    except ValueError:
        pass
    else:
        raise AssertionError("未知 mode 应报错")


def test_make_cf_dataset(tmp):
    n, h, w = 5, 8, 8
    images = np.random.default_rng(1).integers(0, 256, size=(n, 3, h, w), dtype=np.uint8)
    meta = {"num_samples": n, "collision_flags": "0" * n, "collision_filter": {"drop_collision_steps": False},
            "episodes": [{"split": "val", "sample_start": 0, "sample_count": 3, "source_dir": "/nonexistent"},
                         {"split": "test", "sample_start": 3, "sample_count": 2, "source_dir": "/nonexistent"}]}
    src = os.path.join(tmp, "src")
    os.makedirs(src)
    uavr = os.path.join(src, "dataset.uavr")
    write_uavr(uavr, images, [1, 2, 3, 1, 1], meta)
    orig_prev = np.array([0, 0, 1, 3, 3], dtype=np.uint32)
    prev_action = np.array([0, 1, 2, 0, 1], dtype=np.uint8)
    pp.write_aux_sidecar(src, uavr, orig_prev, prev_action, None, (h // 8, w // 8), 0.102)
    aux = sc.read_aux(os.path.join(src, "dataset.aux"))
    assert list(aux["prev_index"]) == [0, 0, 1, 3, 3] and list(aux["prev_action"]) == [0, 1, 2, 0, 1]
    assert aux["depth"] is None and abs(aux["envelope_radius_m"] - 0.102) < 1e-6

    dst = os.path.join(tmp, "cf-next")
    new_prev = probe.cf_prev_index("next", meta["episodes"], None)
    probe.make_cf_dataset(src, dst, aux, new_prev)
    assert os.stat(os.path.join(dst, "dataset.uavr")).st_ino == os.stat(uavr).st_ino, "dataset.uavr 应是硬链接，不复制"
    aux2 = sc.read_aux(os.path.join(dst, "dataset.aux"))
    assert list(aux2["prev_index"]) == [1, 2, 2, 4, 4], list(aux2["prev_index"])
    assert list(aux2["prev_action"]) == [0, 1, 2, 0, 1], "prev_action 列必须原样"
    # 侧车头绑定的 uavr SHA 与原文件一致（硬链接同一份字节）
    with open(os.path.join(dst, "dataset.aux"), "rb") as f1, open(os.path.join(src, "dataset.aux"), "rb") as f2:
        assert f1.read(128)[28:60] == f2.read(128)[28:60]


def test_summary():
    y = np.array([2, 2, 2, 3, 3, 1, 1])
    pred = np.array([2, 2, 3, 3, 2, 1, 2])
    assert probe.direction_ratio(y, pred, 2) == [2, 1] and probe.direction_ratio(y, pred, 3) == [1, 1]
    ref = np.array([2, 2, 2, 3, 3, 1, 1])
    d = probe.condition_summary(y, pred, ref)
    assert d["n"] == 7 and abs(d["changed_rate"] - 3 / 7) < 1e-12
    assert abs(d["changed_rate_yaw_label"] - 2 / 5) < 1e-12 and abs(d["changed_rate_fwd_label"] - 1 / 2) < 1e-12
    assert d["yawl_frames_pred_L_R"] == [2, 1] and "changed_rate" not in probe.condition_summary(y, pred)


def main():
    tmp = tempfile.mkdtemp(prefix="test_probe_two_frame_cf_")
    try:
        test_cf_prev_index()
        test_make_cf_dataset(tmp)
        test_summary()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS: probe_two_frame_cf 反事实映射 / 侧车重写 / 方向计数")


if __name__ == "__main__":
    main()
