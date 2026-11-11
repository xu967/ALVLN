#!/usr/bin/env python3
"""打包的标签来源与图像对齐回归（纯 CPU，临时目录，约 2 s）。

锁住两条打包契约。在此之前 `tools/check.sh --python-only` 里没有测试守它们（交叉审核 F3，2026-11-12）：
  1. 标签 = actions.jsonl 的 action_expert（教师标签）。DAgger 集（executed_by=student）里
     action_executed 与教师标签逐步不同，标签改取执行动作就是「学生学自己」。
  2. UAVR 第 g 个样本的图像 = 该样本 step 对应的 rgb_<step>.png。帧号错一位，标签与画面就错位。

夹具：三集；每一帧是常量色，R 通道编码 step、G 通道编码集序号；双线性缩放常量图仍精确等于该常量，
所以由打包后的像素直接解出 (集, step)，不调用 preprocess 里的任何函数，也不复用它的 HEADER。
每集多写一张 rgb_<步数>.png（采集末帧常见），「帧号 +1」就不会因缺帧报错，只会错位。

变异检查（在副本里改 preprocess.py，不碰工作树）：标签取 action_executed、帧号 +1、帧号取
「已保留样本数」、标签取上一步，都应让本文件变红。
"""
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
# tools/DATASET_FORMAT.md §1.1 的 48 字节小端头；读端不借用 preprocess.HEADER，写端改了布局这里会红。
HEADER = struct.Struct("<4sHHIHHBBBBQQQI")
SRC_W, SRC_H = 320, 240
# 每集相对教师标签的执行动作偏移：集 0 = DAgger 学生自采，集 1 = 旧格式教师集（没有 action_executed 键），
# 集 2 = 差 3（即 -1）。集 0、集 2 的执行动作逐步不同于教师标签，取错来源一定改变标签。
EXEC_OFFSET = (1, None, 3)


def frame_color(ep_idx, step):
    """R 编码 step（0..9），G 编码集序号（0..3），B 固定。"""
    return (30 + 25 * step, 60 + 50 * ep_idx, 77)


def decode_color(rgb):
    """frame_color 的逆：返回 (集序号, step)；不是该编码的色值返回 None。"""
    r, g, _ = (int(v) for v in rgb)
    if (r - 30) % 25 or (g - 60) % 50:
        return None
    return (g - 60) // 50, (r - 30) // 25


def read_uavr(path):
    raw = Path(path).read_bytes()
    magic, ver, hsize, n, h, w, c, _i, _l, _r, io, lo, mo, ml = HEADER.unpack_from(raw, 0)
    assert (magic, ver, hsize, c) == (b"UAVR", 1, 48, 3), "文件头不是 UAVR v1"
    images = np.frombuffer(raw, np.uint8, n * c * h * w, io).reshape(n, c, h, w)
    labels = np.frombuffer(raw, np.uint8, n, lo)
    meta = json.loads(raw[mo:mo + ml].decode("utf-8"))
    return images, labels, meta


class LabelAlignmentTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        subprocess.run([PY, str(REPO / "tools/make_synthetic_episode.py"), "--out", str(self.tmp / "ep")],
                       check=True, capture_output=True)
        self.eps = [self.tmp / "ep" / n for n in ("ep1", "ep2", "ep3")]
        self.rows = []  # 每集按 step 序的 actions 行，即打包前落盘的真值
        for k, ep in enumerate(self.eps):
            rows = [json.loads(x) for x in (ep / "actions.jsonl").read_text().splitlines() if x]
            for r in rows:
                if EXEC_OFFSET[k] is None:
                    r.pop("action_executed")
                else:
                    r["action_executed"] = (r["action_expert"] + EXEC_OFFSET[k]) % 4
                    self.assertNotEqual(r["action_executed"], r["action_expert"], "夹具自检：执行动作须不同于标签")
            self.rows.append(rows)
            for step in range(len(rows) + 1):
                Image.new("RGB", (SRC_W, SRC_H), frame_color(k, step)).save(ep / ("rgb_%05d.png" % step))
        man = json.loads((self.eps[0] / "episode_manifest.json").read_text())
        man.update(executed_by="student", dagger_round=1, beta=0.0)
        (self.eps[0] / "episode_manifest.json").write_text(json.dumps(man))
        self.write_rows()

    def tearDown(self):
        self._tmp.cleanup()

    def write_rows(self):
        """落盘 actions.jsonl；集 2 的行序倒着写，打包必须按 step 而不是行序排样本。"""
        for k, ep in enumerate(self.eps):
            rows = self.rows[k][::-1] if k == 2 else self.rows[k]
            (ep / "actions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def pack(self, *flags):
        out = self.tmp / "ds"
        r = subprocess.run([PY, str(REPO / "tools/preprocess.py"), "--out", str(out), *flags, *map(str, self.eps)],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return read_uavr(out / "dataset.uavr")

    def expected(self, dropped=()):
        """输入顺序（集 0、1、2，各集 step 升序）减去被丢的 (集序号, step)，每项带教师标签。"""
        return [(k, r["step"], r["action_expert"]) for k, rows in enumerate(self.rows) for r in rows
                if (k, r["step"]) not in dropped]

    def assert_samples(self, images, labels, want):
        self.assertEqual(len(labels), len(want), "样本数")
        for g, (k, step, label) in enumerate(want):
            where = "样本 %d（集 %d，step %d）" % (g, k, step)
            self.assertEqual(int(labels[g]), label, where + "：标签必须是教师标签 action_expert")
            planes = images[g]
            self.assertTrue(all(np.all(planes[ch] == planes[ch, 0, 0]) for ch in range(3)), where + "：常量帧缩放后应仍为常量")
            self.assertEqual(decode_color(planes[:, 0, 0]), (k, step), where + "：像素必须来自 rgb_%05d.png" % step)

    def test_labels_are_teacher_and_pixels_follow_step(self):
        images, labels, meta = self.pack()
        self.assert_samples(images, labels, self.expected())
        starts = [e["sample_start"] for e in meta["episodes"]]
        self.assertEqual(starts, [0, len(self.rows[0]), len(self.rows[0]) + len(self.rows[1])])

    def test_alignment_survives_drop_collision(self):
        # 集 0 的 step 2 与集 2 的 step 0 是碰撞步，--drop-collision 丢掉它们；
        # 其后的样本仍取自己 step 的帧与标签（帧号不能按「已保留样本数」数）。
        drops = {(0, 2), (2, 0)}
        for k, step in drops:
            self.rows[k][step]["collision"] = True
        self.write_rows()
        images, labels, meta = self.pack("--drop-collision")
        self.assertEqual(meta["collision_dropped"], len(drops))
        self.assert_samples(images, labels, self.expected(drops))


if __name__ == "__main__":
    unittest.main()
