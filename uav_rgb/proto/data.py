#!/usr/bin/env python3
"""UAVR v1 数据集与 UAVRAUX1 侧车的读取（对照 tools/DATASET_FORMAT.md、src/dataset.cpp）。

读端规则与 C++ 的 load_uavr 一致：头与偏移必须自洽，标签 0..3，episodes[] 的区间不重叠且覆盖全部样本，
同一 scene_id / geometry_gid 不跨 split；split 只从 meta 的 episodes[] 推导，顺序是 meta 里的 episode 顺序
（训练打乱前的 order 就是这个顺序）；drop_collision 按 meta.collision_flags 从三个 split 里删碰撞步。
图像区用 memmap 打开，按下标取批时才复制。
"""
import hashlib
import json
import os
import struct

import numpy as np

HEADER_BYTES = 48
NUM_CLASSES = 4


class DatasetError(ValueError):
    pass


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


class Uavr:
    def __init__(self, path, drop_collision=True):
        self.path = path
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            hb = f.read(HEADER_BYTES)
        if len(hb) != HEADER_BYTES or hb[:4] != b"UAVR":
            raise DatasetError("UAVR: magic 不是 UAVR")
        version, header_size, n, h, w, c, idt, ldt, _res, img_off, lab_off, meta_off, meta_len = \
            struct.unpack("<HHIHHBBBBQQQI", hb[4:])
        if version != 1 or header_size != HEADER_BYTES:
            raise DatasetError("UAVR: version/header_size 不是 1/48")
        if c != 3 or idt != 1 or ldt != 1:
            raise DatasetError("UAVR: channels/dtype 不是 3/uint8/uint8")
        if n <= 0 or h <= 0 or w <= 0:
            raise DatasetError("UAVR: N/H/W 非正")
        img_bytes = n * c * h * w
        if img_off != HEADER_BYTES or lab_off != img_off + img_bytes or meta_off != lab_off + n:
            raise DatasetError("UAVR: 偏移与 N*C*H*W 不自洽")
        if size != meta_off + meta_len:
            raise DatasetError(f"UAVR: 文件长 {size} != meta_offset+meta_len {meta_off + meta_len}")
        self.n, self.h, self.w, self.c = n, h, w, c
        self.images = np.memmap(path, dtype=np.uint8, mode="r", offset=img_off, shape=(n, c, h, w))
        self.labels = np.fromfile(path, dtype=np.uint8, count=n, offset=lab_off)
        if int(self.labels.max()) >= NUM_CLASSES:
            raise DatasetError("UAVR: 标签越界")
        with open(path, "rb") as f:
            f.seek(meta_off)
            self.meta = json.loads(f.read(meta_len).decode("utf-8"))
        self.split_scheme = self.meta.get("split_scheme") or self.meta.get("split_policy")
        self.splits = self._assign_splits()
        self.drop_collision = drop_collision
        if drop_collision:
            flags = self.meta.get("collision_flags")
            if not isinstance(flags, str) or len(flags) != n or set(flags) - {"0", "1"}:
                raise DatasetError("UAVR: meta 缺合规的 collision_flags，无法按碰撞过滤")
            keep = np.frombuffer(flags.encode("ascii"), dtype=np.uint8) == ord("0")
            self.splits = {k: v[keep[v]] for k, v in self.splits.items()}
        self._sha = None

    def _assign_splits(self):
        eps = self.meta.get("episodes")
        if not eps:
            raise DatasetError("UAVR: meta 缺 episodes[]")
        assigned = np.zeros(self.n, dtype=bool)
        out = {"train": [], "val": [], "test": []}
        scenes, gids = {}, {}
        need_gid = self.split_scheme == "geometry_gid"
        for e in eps:
            sp = e["split"]
            if sp not in out:
                raise DatasetError(f"UAVR: episode split 非法: {sp}")
            if scenes.setdefault(e["scene_id"], sp) != sp:
                raise DatasetError(f"UAVR: scene_id 跨 split（{e['scene_id']}）")
            gid = e.get("geometry_gid")
            if gid is None:
                if need_gid:
                    raise DatasetError("UAVR: split_scheme=geometry_gid 但 episode 缺 geometry_gid")
            elif gids.setdefault(gid, sp) != sp:
                raise DatasetError(f"UAVR: geometry_gid 跨 split（{gid}）")
            a, b = int(e["sample_start"]), int(e["sample_start"]) + int(e["sample_count"])
            if b > self.n:
                raise DatasetError("UAVR: episode 区间越界")
            if assigned[a:b].any():
                raise DatasetError("UAVR: episode 区间重叠")
            assigned[a:b] = True
            out[sp].append(np.arange(a, b, dtype=np.int64))
        if not assigned.all():
            raise DatasetError("UAVR: episode 区间未覆盖全部样本")
        return {k: (np.concatenate(v) if v else np.zeros(0, dtype=np.int64)) for k, v in out.items()}

    @property
    def sha256(self):
        if self._sha is None:
            self._sha = sha256_file(self.path)
        return self._sha

    def episode_spans(self, split):
        """该 split 的 episode 列表：[(episode_id, sample_start, sample_count, source_dir)]。"""
        return [(e.get("episode_id"), int(e["sample_start"]), int(e["sample_count"]), e.get("source_dir"))
                for e in self.meta["episodes"] if e["split"] == split]

    def class_counts(self, split="train"):
        return np.bincount(self.labels[self.splits[split]], minlength=NUM_CLASSES)

    def auto_class_weights(self):
        """src/class_weights.cpp 的 auto：w_c = N/(C·n_c)，double 算完转 float32；没有样本的类记 0。"""
        cnt = self.class_counts("train")
        n = float(len(self.splits["train"]))
        w = np.zeros(NUM_CLASSES, dtype=np.float32)
        for k in range(NUM_CLASSES):
            if cnt[k] > 0:
                w[k] = np.float32(n / (float(NUM_CLASSES) * float(cnt[k])))
        return w

    def take(self, idx):
        """按全局下标取图像 [B,C,H,W] uint8（拷贝）与标签 int64。"""
        idx = np.asarray(idx, dtype=np.int64)
        return np.ascontiguousarray(self.images[idx]), self.labels[idx].astype(np.int64)


def read_aux(uavr_path):
    """UAVRAUX1 侧车的 prev_index / prev_action（同目录 dataset.aux）；没有侧车返回 None。只读这两段。"""
    path = os.path.splitext(uavr_path)[0] + ".aux"
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        hdr = f.read(128)
    if len(hdr) != 128 or hdr[:8] != b"UAVRAUX1":
        raise DatasetError("UAVRAUX1: magic 不符")
    n = struct.unpack("<Q", hdr[8:16])[0]
    prev_index = np.fromfile(path, dtype="<u4", count=n, offset=128)
    prev_action = np.fromfile(path, dtype=np.uint8, count=n, offset=128 + 4 * n)
    if len(prev_index) != n or len(prev_action) != n:
        raise DatasetError("UAVRAUX1: payload 不足")
    return {"prev_index": prev_index.astype(np.int64), "prev_action": prev_action.astype(np.int64)}
