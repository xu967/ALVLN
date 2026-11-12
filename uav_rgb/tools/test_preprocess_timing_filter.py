#!/usr/bin/env python3
"""preprocess --drop-late-and-stall（第十二阶段 W1-13 的 A2/A3）回归测试（纯 CPU，临时目录）。

覆盖：
  1. 停顿判据 is_stall_obs_gap 与 sim/tools/collect.py 的 is_stall_gap 逐点一致（含 hold != 1.0）。
  2. 默认关闭时：含迟到步与停顿步的集照常保留全部行，meta 与侧车不加任何新键，dataset.uavr 与 dataset.aux 逐字节不变。
  3. 打开 --drop-late-and-stall 时：丢掉 late_dispatch=true 的迟到步与 obs_gap_s（缺省回退相邻 rgb_ts 差）超阈值的停顿步；
     保留样本的图像与 step 对齐、标签取 action_expert；被丢步之后的 prev_action 与 heading_code 仍反映真实已执行动作；
     meta.collision_filter 记 drop_late_and_stall / late_dropped / stall_dropped / timing_dropped。
  4. 非法类型的 late_dispatch（如字符串 "false"）拒绝打包。
"""
import contextlib
import io
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim" / "tools"))
sys.path.insert(0, str(REPO / "tools"))
import collect  # noqa: E402
import preprocess as pre  # noqa: E402

PY = sys.executable
HEADER = struct.Struct("<4sHHIHHBBBBQQQI")
AUX_HEADER = struct.Struct("<8sQIHHBBH")
SRC_W, SRC_H = 320, 240
NS = 10**9


def frame_color(ep_idx, step):
    return (20 + 25 * step, 40 + 60 * ep_idx, 99)


def decode_color(rgb):
    r, g, _ = (int(v) for v in rgb)
    return (g - 40) // 60, (r - 20) // 25


def read_uavr_and_aux(ds_dir):
    raw = (ds_dir / "dataset.uavr").read_bytes()
    magic, ver, hsize, n, h, w, c, _i, _l, _r, io_off, lo_off, mo_off, ml = HEADER.unpack_from(raw, 0)
    assert (magic, ver, hsize, c) == (b"UAVR", 1, 48, 3)
    images = np.frombuffer(raw, np.uint8, n * c * h * w, io_off).reshape(n, c, h, w)
    labels = np.frombuffer(raw, np.uint8, n, lo_off)
    meta = json.loads(raw[mo_off:mo_off + ml].decode("utf-8"))
    aux_raw = (ds_dir / "dataset.aux").read_bytes()
    amagic, an, *_ = AUX_HEADER.unpack_from(aux_raw, 0)
    assert amagic == b"UAVRAUX1" and an == n
    prev_index = np.frombuffer(aux_raw, "<u4", n, 128)
    prev_action = np.frombuffer(aux_raw, np.uint8, n, 128 + 4 * n)
    return images, labels, meta, prev_index, prev_action


class PreprocessTimingFilterTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        subprocess.run([PY, str(REPO / "tools/make_synthetic_episode.py"), "--out", str(self.tmp / "ep")],
                       check=True, capture_output=True)
        self.eps = [self.tmp / "ep" / n for n in ("ep1", "ep2", "ep3")]
        self.rows = []
        for k, ep in enumerate(self.eps):
            man = json.loads((ep / "episode_manifest.json").read_text())
            man["action_latency_s"] = 0.2
            man["window_from"] = "observation"
            man["hold_s"] = 1.0
            (ep / "episode_manifest.json").write_text(json.dumps(man))
            rows = [json.loads(x) for x in (ep / "actions.jsonl").read_text().splitlines() if x]
            ts = 10 * NS
            for i, r in enumerate(rows):
                if i > 0:
                    ts += int(round(0.9 * NS))
                r["rgb_ts"] = ts
                r["obs_gap_s"] = None if i == 0 else 0.9
                r["late_dispatch"] = False
                r["action_executed"] = (r["action_expert"] + 1) % 4
                Image.new("RGB", (SRC_W, SRC_H), frame_color(k, r["step"])).save(ep / ("rgb_%05d.png" % r["step"]))
            self.rows.append(rows)
        self.flush_rows()

    def tearDown(self):
        self._tmp.cleanup()

    def flush_rows(self):
        for ep, rows in zip(self.eps, self.rows):
            (ep / "actions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def run_inproc(self, out_dir, extra_flags=()):
        fixed_dt = datetime(2026, 11, 12, 7, 0, 0, tzinfo=timezone.utc)

        class _FixedDT:
            @staticmethod
            def now(_tz=None):
                return fixed_dt

        argv = ["preprocess.py", "--out", str(out_dir), *extra_flags, *map(str, self.eps)]
        with patch.object(sys, "argv", argv), patch.object(pre, "datetime", _FixedDT), \
                contextlib.redirect_stdout(io.StringIO()):
            rc = pre.main()
        self.assertEqual(rc, 0)

    def test_stall_predicate_matches_collect_single_source(self):
        for hold in (None, 0.5, 1.0, 2.0):
            for gap in (None, 0.4, 0.6, 0.6001, 0.9, 1.1, 1.1 + 1e-9, 1.101, 1.6, 1.9, 2.1, 2.2):
                self.assertEqual(pre.is_stall_obs_gap(gap, hold=hold),
                                 collect.is_stall_gap(gap, hold=hold), (hold, gap))

    def test_default_off_is_byte_identical_even_when_rows_have_late_and_stall(self):
        out_clean = self.tmp / "ds_clean"
        self.run_inproc(out_clean)
        # 在行里标上迟到步与停顿步，默认不给开关再打一包：两个输出文件必须逐字节一致
        self.rows[0][1]["late_dispatch"] = True
        self.rows[1][2]["obs_gap_s"] = 1.6
        self.rows[2][1]["late_dispatch"] = True
        self.rows[2][1]["obs_gap_s"] = 2.6
        self.flush_rows()
        out_default = self.tmp / "ds_default"
        self.run_inproc(out_default)
        self.assertEqual((out_default / "dataset.uavr").read_bytes(), (out_clean / "dataset.uavr").read_bytes())
        self.assertEqual((out_default / "dataset.aux").read_bytes(), (out_clean / "dataset.aux").read_bytes())

    def test_flag_drops_late_and_stall_rows_and_preserves_alignment_and_cond(self):
        # ep0 step 1: 迟到；ep0 step 2: 正常（它的上一步 step 1 被丢，prev_action 仍取 step 1 的 action_executed）
        self.rows[0][1]["late_dispatch"] = True
        self.rows[0][1]["action_executed"] = 2  # YAW_LEFT
        self.rows[0][0]["action_executed"] = 2  # YAW_LEFT：到 step 2 前累计了两次左转 → heading_code = 2
        # ep1 step 2: 显式 obs_gap_s 停顿（1.6 s）
        self.rows[1][2]["obs_gap_s"] = 1.6
        # ep2 step 1: 旧格式（删掉 obs_gap_s 键），靠相邻 rgb_ts 差 1.8 s 识别出停顿；同一步还迟到（只计丢 1 个样本）
        for r in self.rows[2]:
            r.pop("obs_gap_s", None)
        self.rows[2][1]["rgb_ts"] = self.rows[2][0]["rgb_ts"] + int(round(1.8 * NS))
        for j in range(2, len(self.rows[2])):
            self.rows[2][j]["rgb_ts"] = self.rows[2][j - 1]["rgb_ts"] + int(round(0.9 * NS))
        self.rows[2][1]["late_dispatch"] = True
        self.flush_rows()

        dropped_pairs = {(0, 1), (1, 2), (2, 1)}
        want = [(k, r["step"], r["action_expert"]) for k, rows in enumerate(self.rows) for r in rows
                if (k, r["step"]) not in dropped_pairs]

        out_prev = self.tmp / "ds_prev"
        self.run_inproc(out_prev, ("--drop-late-and-stall",))
        images, labels, meta, prev_idx, prev_act = read_uavr_and_aux(out_prev)
        cf = meta["collision_filter"]
        self.assertIs(cf["drop_late_and_stall"], True)
        self.assertEqual(cf["late_dropped"], 2)
        self.assertEqual(cf["stall_dropped"], 2)
        self.assertEqual(cf["timing_dropped"], 3)
        self.assertEqual(meta["collision_dropped"], 0, "不混入碰撞步丢弃计数")
        self.assertEqual(len(labels), len(want))
        for g, (k, step, label) in enumerate(want):
            self.assertEqual(int(labels[g]), label)
            self.assertEqual(decode_color(images[g, :, 0, 0]), (k, step))
        # ep0 保留样本下标 0 -> step 0，下标 1 -> step 2（上一步 step 1 被丢：prev_index 退回自身 1，prev_action 仍是 step 1 的 2）
        self.assertEqual(int(prev_idx[1]), 1)
        self.assertEqual(int(prev_act[1]), 2)

        # --cond heading-code：被丢步的转向照常累计，ep0 step 2 的码是 2
        for ep in self.eps:
            man = json.loads((ep / "episode_manifest.json").read_text())
            man["instruction"] = None
            man["waypoint"] = None
            (ep / "episode_manifest.json").write_text(json.dumps(man))
        out_hc = self.tmp / "ds_hc"
        self.run_inproc(out_hc, ("--drop-late-and-stall", "--cond", "heading-code"))
        _, _, _, _, hc = read_uavr_and_aux(out_hc)
        self.assertEqual(int(hc[1]), 2)

    def test_hold_two_seconds_respects_scaled_nominal_period(self):
        for k, ep in enumerate(self.eps):
            man = json.loads((ep / "episode_manifest.json").read_text())
            man["hold_s"] = 2.0
            (ep / "episode_manifest.json").write_text(json.dumps(man))
            for i, r in enumerate(self.rows[k]):
                r["obs_gap_s"] = None if i == 0 else 1.9
        self.rows[1][1]["obs_gap_s"] = 2.25   # 只有这一步超过 1.9 + 0.2 = 2.1
        self.flush_rows()
        out = self.tmp / "ds_hold2"
        self.run_inproc(out, ("--drop-late-and-stall",))
        _, labels, meta, _, _ = read_uavr_and_aux(out)
        self.assertEqual(meta["collision_filter"]["stall_dropped"], 1)
        self.assertEqual(meta["collision_filter"]["timing_dropped"], 1)
        self.assertEqual(len(labels), sum(len(r) for r in self.rows) - 1)

    def test_non_bool_late_dispatch_is_rejected(self):
        self.rows[0][1]["late_dispatch"] = "false"
        self.flush_rows()
        r = subprocess.run([PY, str(REPO / "tools/preprocess.py"), "--drop-late-and-stall",
                            "--out", str(self.tmp / "ds_bad"), *map(str, self.eps)],
                           capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("late_dispatch", r.stderr)


if __name__ == "__main__":
    unittest.main()
