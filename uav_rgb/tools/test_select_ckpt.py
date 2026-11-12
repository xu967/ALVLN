#!/usr/bin/env python3
"""select_ckpt 的纯 Python 部分：UAVR 读取与 split/碰撞过滤、样本→位姿映射、空画面判定、四件套、选模规则。
不需要 GPU / uavrgb_infer。运行（仓库根）：.venv-tools/bin/python tools/test_select_ckpt.py
"""
import json
import os
import shutil
import struct
import subprocess
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


def write_ckpt_header(path, version=6, drop_collision=True, zero_contract=False, ds_sha="ab" * 32, lr=1e-3, batch=128):
    """按 src/checkpoint.cpp 布局写最小 ckpt_epNN.bin 头（足够解析 v5 训练契约）。"""
    nw = 4
    buf = bytearray(struct.pack("<8sIIiiiiIII", b"UAVRCK01", version, 1, 8, 8, 3, 4, 10, 1, nw))
    buf += struct.pack("<4f", 1.0, 1.0, 1.0, 1.0)
    if version >= 3:
        buf += struct.pack("<f", 0.5)
    if version >= 4:
        buf += struct.pack("<i", 0)
    if version >= 5:
        if zero_contract:
            buf += b"\x00" * 113
        else:
            buf += (bytes.fromhex(ds_sha) + struct.pack("<B", 1 if drop_collision else 0) + b"\x00" * 32
                    + struct.pack("<ffIi", 0.0, lr, 42, batch) + b"\x00" * 32)
    if version >= 6:
        buf += struct.pack("<ii", 0, 0)
    with open(path, "wb") as f:
        f.write(bytes(buf))


def test_resolve_drop_collision_and_cli(tmp):
    """C6：select_ckpt 没有 config.json 时从 ckpt v5 契约读 drop_collision；先看 config.resume.json；
    config 与契约矛盾或两处均无记录时拒绝（对齐 src/trainer_eval.cpp:194-212）。"""
    # 1) 无 config.json，仅 ckpt 契约有记录（True / False）
    for want in (True, False):
        rdir = os.path.join(tmp, f"run_nocfg_{want}")
        os.makedirs(rdir)
        p1 = os.path.join(rdir, "ckpt_ep01.bin")
        write_ckpt_header(p1, drop_collision=want)
        assert sc.resolve_drop_collision(rdir, [(1, p1)]) is want

    # 2) 同目录续训：config.resume.json 优先于旧 config.json
    r_res = os.path.join(tmp, "run_res")
    os.makedirs(r_res)
    p_res = os.path.join(r_res, "ckpt_ep02.bin")
    write_ckpt_header(p_res, drop_collision=True)
    with open(os.path.join(r_res, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"drop_collision": False}, f)
    with open(os.path.join(r_res, "config.resume.json"), "w", encoding="utf-8") as f:
        json.dump({"drop_collision": True}, f)
    assert sc.resolve_drop_collision(r_res, [(2, p_res)]) is True

    # 3) config.json 与 ckpt 契约矛盾 → 报错拒绝
    r_conflict = os.path.join(tmp, "run_conflict")
    os.makedirs(r_conflict)
    p_conf = os.path.join(r_conflict, "ckpt_ep01.bin")
    write_ckpt_header(p_conf, drop_collision=True)
    with open(os.path.join(r_conflict, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"drop_collision": False}, f)
    try:
        sc.resolve_drop_collision(r_conflict, [(1, p_conf)])
    except ValueError as e:
        assert "矛盾" in str(e), e
    else:
        raise AssertionError("config 与 ckpt 契约矛盾时应报错")

    # 4) 无含 drop_collision 的 config，且 ckpt 为全零契约或 v4（无契约）→ 报错拒绝，不静默回落 False
    for tag, kw in (("zero", {"version": 6, "zero_contract": True}), ("v4", {"version": 4})):
        r_none = os.path.join(tmp, f"run_none_{tag}")
        os.makedirs(r_none)
        p_none = os.path.join(r_none, "ckpt_ep01.bin")
        write_ckpt_header(p_none, **kw)
        with open(os.path.join(r_none, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"lr": 1e-3}, f)   # 无 drop_collision 键
        try:
            sc.resolve_drop_collision(r_none, [(1, p_none)])
        except ValueError as e:
            assert "drop_collision" in str(e) or "契约" in str(e), e
        else:
            raise AssertionError(f"{tag}: 两处都无记录时应报错拒绝")

    # 5) 端到端 CLI：无 config.json、ckpt 契约 drop_collision=True 时剔除 val 碰撞帧（4 帧而非 5 帧）；全零契约时非零退出
    fake_infer = os.path.join(tmp, "fake_infer")
    with open(fake_infer, "w", encoding="utf-8") as f:
        f.write(f"#!{sys.executable}\nimport json, sys\n"
                "print(json.dumps({'in_h':8,'in_w':8,'in_c':3,'num_classes':4,'text_dim':0,"
                "'prev_action_dim':0,'aux_classes':0}), flush=True)\n"
                "while sys.stdin.buffer.read(8*8*3):\n"
                "    print(json.dumps({'action':1,'probs':[0.1,0.7,0.1,0.1]}), flush=True)\n")
    os.chmod(fake_infer, 0o755)
    ds_path = os.path.join(tmp, "dataset.uavr")   # test_uavr_and_poses 已写出（val 5 帧含 1 帧碰撞）
    out_ok = os.path.join(tmp, "sel_ok")
    res_ok = subprocess.run([sys.executable, os.path.join(REPO, "tools", "select_ckpt.py"),
                             "--run", os.path.join(tmp, "run_nocfg_True"), "--data", ds_path,
                             "--out", out_ok, "--infer-bin", fake_infer, "--no-check-metrics"],
                            capture_output=True, text=True)
    assert res_ok.returncode == 0, res_ok.stderr
    ep_json = json.load(open(os.path.join(out_ok, "epochs.json"), encoding="utf-8"))
    assert ep_json["n"] == 4 and ep_json["rows"][0]["n"] == 4, f"应按契约 drop_collision=True 剔除 1 个碰撞帧：{ep_json}"
    assert ep_json.get("drop_collision") is True

    res_bad = subprocess.run([sys.executable, os.path.join(REPO, "tools", "select_ckpt.py"),
                              "--run", os.path.join(tmp, "run_none_zero"), "--data", ds_path,
                              "--out", os.path.join(tmp, "sel_bad"), "--infer-bin", fake_infer, "--no-check-metrics"],
                             capture_output=True, text=True)
    assert res_bad.returncode != 0, "无 config 且全零契约时 CLI 必须非零退出"


def main():
    tmp = tempfile.mkdtemp(prefix="test_select_ckpt_")
    try:
        test_metrics()
        test_uavr_and_poses(tmp)
        test_select_epoch()
        test_epoch_row_and_ckpts(tmp)
        test_resolve_drop_collision_and_cli(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS: select_ckpt 四件套 / UAVR 读取与过滤 / 样本→位姿 / 空画面 / 选模规则")


if __name__ == "__main__":
    main()
