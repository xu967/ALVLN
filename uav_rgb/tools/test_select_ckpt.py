#!/usr/bin/env python3
"""select_ckpt 的纯 Python 部分：UAVR 读取与 split/碰撞过滤、样本→位姿映射、空画面判定、四件套、选模规则。
不需要 GPU / uavrgb_infer。运行（仓库根）：.venv-tools/bin/python tools/test_select_ckpt.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import select_ckpt as sc  # noqa: E402


def write_uavr(path, images, labels, meta):
    """按 DATASET_FORMAT §1 写一个小 dataset.uavr；images (N,3,H,W) uint8。"""
    n, c, h, w = images.shape
    mb = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    io = sc.HEADER.size
    lo = io + images.nbytes
    mo = lo + n
    with open(path, "wb") as f:
        f.write(sc.HEADER.pack(b"UAVR", 1, sc.HEADER.size, n, h, w, c, 1, 1, 0, io, lo, mo, len(mb)))
        f.write(images.tobytes())
        f.write(np.asarray(labels, dtype=np.uint8).tobytes())
        f.write(mb)


def make_episode_dir(root, name, steps, collisions, pose_fn, pillars):
    d = os.path.join(root, name)
    os.makedirs(d)
    with open(os.path.join(d, "actions.jsonl"), "w", encoding="utf-8") as f:
        for s, col in zip(steps, collisions):
            x, y, yaw = pose_fn(s)
            f.write(json.dumps({"step": s, "action_expert": 1, "collision": col,
                                "pose_before": {"x": x, "y": y, "z": 2.0}, "yaw_before": yaw}) + "\n")
    with open(os.path.join(d, "scene.json"), "w", encoding="utf-8") as f:
        json.dump({"pillars": pillars}, f)
    return d


def test_metrics():
    # 行 = 真值，列 = 预测；val 没有 SAFE 真值 → recall_safe 记 0，macro_f1 上限 0.75。
    y = np.array([1, 1, 1, 2, 2, 3, 3, 3])
    pred = np.array([1, 1, 2, 2, 3, 3, 3, 1])
    m = sc.metrics_from_confusion(sc.confusion(y, pred))
    assert abs(m["acc"] - 5 / 8) < 1e-12, m
    # recall: SAFE 0（无真值）、FWD 2/3、L 1/2、R 2/3；precision: FWD 2/3、L 1/2、R 2/3
    assert np.allclose(m["recall"], [0, 2 / 3, 0.5, 2 / 3]), m
    f1 = [0, 2 / 3, 0.5, 2 / 3]
    assert abs(m["macro_f1"] - sum(f1) / 4) < 1e-12, m
    assert abs(m["balanced_acc"] - sum([0, 2 / 3, 0.5, 2 / 3]) / 4) < 1e-12, m
    # 全对
    m2 = sc.metrics_from_confusion(sc.confusion(y, y))
    assert abs(m2["macro_f1"] - 0.75) < 1e-12 and abs(m2["acc"] - 1.0) < 1e-12, m2


def test_uavr_and_poses(tmp):
    n, h, w = 7, 8, 8
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(n, 3, h, w), dtype=np.uint8)
    labels = [1, 2, 1, 3, 1, 1, 2]
    pillars = [{"x": 3.0, "y": 0.0, "size": [0.4, 0.4, 3.0]}]
    # ep A（train）：step 0,1,2，step 1 碰撞被打包时丢掉 → 2 个样本；ep B（val）：step 0..4，全保留 → 5 个样本
    ep_a = make_episode_dir(tmp, "epA", [0, 1, 2], [False, True, False], lambda s: (0.1 * s, 0.0, 0.0), pillars)
    ep_b = make_episode_dir(tmp, "epB", [0, 1, 2, 3, 4], [False] * 5,
                            lambda s: (0.0, 0.0, 180.0 if s >= 3 else 0.0), pillars)   # step 3/4 背对柱子
    meta = {"num_samples": n, "collision_flags": "0000100",   # 样本 4（ep B step 2）标碰撞但未在打包时丢
            "collision_filter": {"drop_collision_steps": True},
            "episodes": [{"split": "train", "sample_start": 0, "sample_count": 2, "source_dir": ep_a},
                         {"split": "val", "sample_start": 2, "sample_count": 5, "source_dir": ep_b}]}
    path = os.path.join(tmp, "dataset.uavr")
    write_uavr(path, images, labels, meta)
    ds = sc.Uavr(path)
    assert (ds.n, ds.h, ds.w, ds.c) == (n, h, w, 3)
    assert np.array_equal(ds.images[3], images[3]) and list(ds.labels) == labels
    assert ds.hwc_bytes(3) == np.ascontiguousarray(images[3].transpose(1, 2, 0)).tobytes()
    assert list(ds.split_indices("val", False)) == [2, 3, 4, 5, 6]
    assert list(ds.split_indices("val", True)) == [2, 3, 5, 6], "drop_collision 应剔除 collision_flags=='1' 的样本"
    assert list(ds.split_indices("train", True)) == [0, 1]

    pose, step, pil = sc.sample_poses(ds.meta, True)
    assert list(step) == [0, 2, 0, 1, 2, 3, 4], f"碰撞步 1 被丢后 ep A 的样本应映射到 step 0/2：{list(step)}"
    assert abs(pose[1, 0] - 0.2) < 1e-12 and pil[1] is pil[0]
    empty = sc.empty_frame_mask(pose, pil, ds.split_indices("val", False))
    assert list(empty) == [False, False, False, True, True], f"背对柱子的两帧应判空画面：{list(empty)}"
    # 映射核对：meta 说 2 个样本但 actions.jsonl 不丢碰撞时有 3 行 → 报错而不是错位
    try:
        sc.sample_poses(ds.meta, False)
    except ValueError as e:
        assert "sample_count" in str(e)
    else:
        raise AssertionError("样本数与 actions.jsonl 不符时应报错")
    # source_dir 不在 → None（调用方记 NaN，不猜）
    meta2 = dict(meta, episodes=[dict(meta["episodes"][0], source_dir=os.path.join(tmp, "gone"))])
    assert sc.sample_poses(meta2, True) is None


def rows_from(mf1, empty_yaw, lr_gap, label_yaw=0.2):
    return [{"epoch": i + 1, "macro_f1": m, "empty_yaw_rate": e, "empty_label_yaw_rate": label_yaw, "lr_recall_gap": g}
            for i, (m, e, g) in enumerate(zip(mf1, empty_yaw, lr_gap))]


def test_select_epoch():
    # ep3 单轮 mF1 最高但空画面 YAW 率 0.6 远超标签 0.2+0.05 → 排除；ep5 L/R 差 0.3 → 排除；
    # 余下 ep1/2/4/6 里按 ±1 轮平滑取最大：ep4 的邻居是 ep3(0.50)/ep5(0.42) → 平滑 (0.50+0.40+0.42)/3=0.44 最高。
    rows = rows_from(mf1=[0.30, 0.35, 0.50, 0.40, 0.42, 0.38],
                     empty_yaw=[0.10, 0.20, 0.60, 0.22, 0.20, 0.24],
                     lr_gap=[0.05, 0.10, 0.05, 0.10, 0.30, 0.10])
    ep, why = sc.select_epoch(rows, empty_yaw_margin=0.05, lr_gap_max=0.15, smooth_half=1)
    assert ep == 4, (ep, why)
    assert why["candidates"] == [1, 2, 4, 6] and not why["relaxed_constraints"], why
    assert why["argmax_raw_macro_f1_epoch"] == 3
    # 约束全空 → 放宽并标明，而不是返回 None。边界轮的窗口只平均存在的邻居：
    # smooth(1)=(0.3+0.4)/2=0.35、smooth(2)=0.35、smooth(3)=(0.4+0.35)/2=0.375 → 3。
    rows2 = rows_from(mf1=[0.3, 0.4, 0.35], empty_yaw=[0.9, 0.9, 0.9], lr_gap=[0.0, 0.0, 0.0])
    ep2, why2 = sc.select_epoch(rows2, empty_yaw_margin=0.05, lr_gap_max=0.15, smooth_half=1)
    assert ep2 == 3 and why2["relaxed_constraints"] and why2["argmax_raw_macro_f1_epoch"] == 2, (ep2, why2)
    # 空画面列 NaN（source_dir 不可读）时只按 L/R 与平滑分选
    rows3 = rows_from(mf1=[0.3, 0.4, 0.35], empty_yaw=[float("nan")] * 3, lr_gap=[0.0, 0.0, 0.5])
    assert sc.select_epoch(rows3, lr_gap_max=0.15)[0] == 2


def test_epoch_row_and_ckpts(tmp):
    y = np.array([1, 1, 2, 3, 1, 2])
    pred = np.array([1, 2, 2, 3, 3, 1])
    prev = np.array([1, 1, 2, 2, 3, 1])
    empty = np.array([True, True, False, False, True, False])
    r = sc.epoch_row(7, y, pred, empty, prev)
    assert r["epoch"] == 7 and r["n"] == 6 and r["empty_n"] == 3
    assert abs(r["empty_yaw_rate"] - 2 / 3) < 1e-12 and abs(r["empty_label_yaw_rate"] - 0.0) < 1e-12
    assert abs(r["flip_rate"] - 2 / 6) < 1e-12 and abs(r["flip_rate_yaw_label"] - 1 / 3) < 1e-12
    assert abs(r["lr_recall_gap"] - abs(0.5 - 1.0)) < 1e-12
    run = os.path.join(tmp, "run")
    os.makedirs(run)
    for name in ("ckpt_ep10.bin", "ckpt_ep02.bin", "ckpt_best.bin", "ckpt.bin", "ckpt_ep1.bin"):
        open(os.path.join(run, name), "wb").close()
    assert [e for e, _ in sc.epoch_ckpts(run)] == [1, 2, 10]


def main():
    tmp = tempfile.mkdtemp(prefix="test_select_ckpt_")
    try:
        test_metrics()
        test_uavr_and_poses(tmp)
        test_select_epoch()
        test_epoch_row_and_ckpts(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS: select_ckpt 四件套 / UAVR 读取与过滤 / 样本→位姿 / 空画面 / 选模规则")


if __name__ == "__main__":
    main()
