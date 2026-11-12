#!/usr/bin/env python3
"""tools/average_ckpt.py 的测试。纯 Python 部分不需要 GPU：合成小 ckpt / 小数据集，推理进程用同协议的假二进制。
复现性验收（真实 GPU、真实数据盘）默认跳过，开关 UAVRGB_REPRO_AVGCKPT=1：

  python tools/test_average_ckpt.py                                  # 进 check.sh 的部分
  UAVRGB_REPRO_AVGCKPT=1 CUDA_VISIBLE_DEVICES=<UUID> python tools/test_average_ckpt.py TestReproduction

复现性验收的产物落在 UAVRGB_REPRO_OUT（父目录，默认 TMPDIR）下新建的目录里；推理二进制用 UAVRGB_REPRO_INFER 换。
"""
import ast
import hashlib
import inspect
import json
import math
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_TMP = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb", "tmp")
if "TMPDIR" not in os.environ and os.path.isdir(_DATA_TMP):
    tempfile.tempdir = _DATA_TMP
sys.path.insert(0, os.path.join(REPO, "tools"))
import average_ckpt as ac  # noqa: E402

# c1.W c1.b c2.W c2.b c3.W c3.b fc.W fc.b（顺序同 cnn_params）；形状取小值，工具对形状只做一致性检查
SHAPES = [(2, 3, 3, 3), (2,), (3, 2, 3, 3), (3,), (4, 3, 3, 3), (4,), (4, 12), (4,)]
H, W = 4, 6
FWD = 1


def rand_params(seed):
    rng = np.random.default_rng(seed)
    return [(rng.standard_normal(s) * 3).astype(np.float32) for s in SHAPES]


def write_ckpt(path, params, epoch, version=6, text_dim=0, prev=0, in_c=3, in_h=H, trailing=b"", ds_sha=None,
               drop_collision=None, zero_contract=False):
    """独立于工具的 v6 写法（布局见 src/include/uavrgb/checkpoint.hpp）。契约 113 B 与 Adam m/v 随轮次变，用来核对「沿用最后一轮」。"""
    b = bytearray(b"UAVRCK01" + struct.pack("<II", version, len(params)))
    b += struct.pack("<4i", in_h, W, in_c, 4) + struct.pack("<3I", 100 + epoch, epoch, 4)
    b += struct.pack("<4f", 36.6, 0.3, 3.2, 3.3) + struct.pack("<f", 0.4) + struct.pack("<i", text_dim)
    if zero_contract:
        contract = b"\x00" * 113
    else:
        c_buf = bytearray((epoch + i) % 256 for i in range(113))
        if ds_sha:   # 契约开头 32 B 是训练数据集 sha256
            c_buf[:32] = bytes.fromhex(ds_sha)
        if drop_collision is not None:   # 偏移 32 的 u8 最低位为 drop_collision（src/checkpoint.cpp:136）
            c_buf[32] = (c_buf[32] & ~1) | (1 if drop_collision else 0)
        contract = bytes(c_buf)
    b += contract + struct.pack("<2i", 0, prev)
    for p in params:
        b += struct.pack("<I", p.ndim) + struct.pack(f"<{p.ndim}q", *p.shape) + p.astype("<f4").tobytes()
    rng = np.random.default_rng(1000 + epoch)
    for _ in range(2):
        for p in params:
            b += rng.random(p.size, dtype=np.float32).tobytes()
    with open(path, "wb") as f:
        f.write(bytes(b) + trailing)


def make_run(root, epochs, params_of, name="run", **kw):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    for e in epochs:
        write_ckpt(os.path.join(d, f"ckpt_ep{e:02d}.bin"), params_of(e), e, **kw)
    return d


def write_uavr(path, n_per_split, labels, collisions, images, extra_meta=None):
    """按 tools/DATASET_FORMAT.md §1 写小数据集：每个 split 一集；collisions 为 '0'/'1' 串或 None（不写该字段）。"""
    n = sum(n_per_split.values())
    meta = {"episodes": [], "split_policy": "scene_id", **(extra_meta or {})}
    start = 0
    for sp, cnt in n_per_split.items():
        meta["episodes"].append({"split": sp, "sample_start": start, "sample_count": cnt, "scene_id": start})
        start += cnt
    if collisions is not None:
        meta["collision_flags"] = collisions
    mb = json.dumps(meta, separators=(",", ":")).encode()
    io = 48
    lo = io + images.nbytes
    with open(path, "wb") as f:
        f.write(struct.pack("<4sHHIHHBBBBQQQI", b"UAVR", 1, 48, n, H, W, 3, 1, 1, 0, io, lo, lo + n, len(mb)))
        f.write(images.tobytes() + np.asarray(labels, np.uint8).tobytes() + mb)


FAKE_INFER = """#!{py}
import json, math, sys
sys.path.insert(0, {tools!r})
import average_ckpt as ac
ck = ac.read_ckpt(sys.argv[sys.argv.index("--ckpt") + 1])
fc_b = ck["params"][7]
print(json.dumps({{"in_h": ck["in_h"], "in_w": ck["in_w"], "in_c": ck["in_c"], "num_classes": 4, "text_dim": ck["text_dim"],
                  "prev_action_dim": ck["prev_dim"], "aux_classes": 0}}), flush=True)
n = ck["in_h"] * ck["in_w"] * 3
seen = 0
while True:
    buf = sys.stdin.buffer.read(n)
    if not buf:
        break
    z = [float(fc_b[k]) + buf[k] / 64.0 for k in range(4)]   # 每类 logit = fc.b[k] + 该帧第 k 个字节/64：帧间有稳定差异
    z[1] += 1e-3 * seen   # 每帧给 FORWARD 加一个不重复的小量：校准阈值互不并列（并列时占比一次跳多帧）
    seen += 1
    m = max(z)
    e = [math.exp(x - m) for x in z]
    p = [x / sum(e) for x in e]
    print('{{"action":%d,"probs":[%s],"infer_ms":0.1}}' % (z.index(m), ",".join("%g" % x for x in p)), flush=True)
"""


COND_COEF = (0.0, 0.4, -0.3, 0.2)   # 条件字节对各类 logit 的影响：条件值不同概率就不同（FAKE_INFER_COND 与测试里的独立算法共用）
FAKE_INFER_COND = """#!{py}
import json, math, os, sys
sys.path.insert(0, {tools!r})
import average_ckpt as ac
ck = ac.read_ckpt(sys.argv[sys.argv.index("--ckpt") + 1])
fc_b = ck["params"][7]
pd = ck["prev_dim"]
print(json.dumps({{"in_h": ck["in_h"], "in_w": ck["in_w"], "in_c": ck["in_c"], "num_classes": 4, "text_dim": ck["text_dim"],
                  "prev_action_dim": int(os.environ.get("FAKE_HELLO_PREV", pd)), "aux_classes": 0}}), flush=True)
n = ck["in_h"] * ck["in_w"] * 3
frame = n + (1 if pd else 0)          # 同 src/infer.cpp：prev_dim>0 时一帧 = HWC 字节 + 1 字节条件
log = os.environ.get("COND_LOG")
seen = 0
while True:
    buf = sys.stdin.buffer.read(frame)
    if not buf:
        break
    if len(buf) != frame:
        sys.exit(4)
    c = buf[n] if pd else 0
    if pd and c >= pd:                # 同 src/infer.cpp：条件字节 >= prev_dim 报错退出
        sys.exit(3)
    if log:
        with open(log, "a") as f:
            f.write("%d %d\\n" % (len(buf), c))
    z = [float(fc_b[k]) + buf[k] / 64.0 + c * {coef!r}[k] for k in range(4)]
    z[1] += 1e-3 * seen
    seen += 1
    m = max(z)
    e = [math.exp(x - m) for x in z]
    p = [x / sum(e) for x in e]
    print('{{"action":%d,"probs":[%s],"infer_ms":0.1}}' % (z.index(m), ",".join("%g" % x for x in p)), flush=True)
"""


def write_aux(data_path, prev_action, *, depth_hw=None, magic=b"UAVRAUX1", n_override=None, flags_override=None,
              unknown=255, uavr_sha=None, corrupt_payload=False, trailing=b"", drop_tail=0):
    """按 tools/DATASET_FORMAT.md §6 写 <data>.aux（独立于工具的写法）；各个 *_override / corrupt 参数造单项损坏。"""
    n = len(prev_action)
    flags = 1 if depth_hw else 0
    dh, dw = depth_hw or (0, 0)
    payload = np.arange(n, dtype="<u4").tobytes() + np.asarray(prev_action, np.uint8).tobytes()   # prev_index 全取自身（首帧口径），本工具不读
    if depth_hw:
        payload += bytes(n * dh * dw)
    pay_sha = hashlib.sha256(payload).digest()
    if corrupt_payload:
        payload = payload[:-1] + bytes([payload[-1] ^ 1])
    head = struct.pack("<8sQIHHBBH", magic, n if n_override is None else n_override, flags if flags_override is None else flags_override,
                       dh, dw, 8, unknown, 0)
    head += bytes.fromhex(uavr_sha or sha(data_path)) + pay_sha + struct.pack("<7f", 0.3, 0.5, 0.8, 1.2, 1.8, 2.5, 4.0) + struct.pack("<fI", 0.102, 0)
    assert len(head) == 128
    spit(os.path.splitext(data_path)[0] + ".aux", (head + payload + trailing)[:len(head + payload + trailing) - drop_tail])


def slurp(path):
    with open(path, "rb") as f:
        return f.read()


def spit(path, data):
    with open(path, "wb") as f:
        f.write(data)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha(path):
    return hashlib.sha256(slurp(path)).hexdigest()


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()


class TestAverage(Base):
    def test_identical_files_give_bitwise_identical_params(self):
        p = rand_params(1)
        for k in (1, 3, 7, 10):
            run = make_run(self.tmp, range(11 - k, 11), lambda e: p, name=f"run{k}")
            out = os.path.join(self.tmp, f"avg{k}.bin")
            ac.main(["--run", run, "--out", out, "--last-k", str(k), "--no-calibrate"])
            got = ac.read_ckpt(out)["params"]
            for a, b in zip(p, got):
                self.assertEqual(a.tobytes(), b.tobytes(), f"k={k}: 相同文件的平均必须逐位等于原参数")
            # 参数相同时整个文件应与最后一轮的文件逐字节相同（头、契约、Adam 全沿用最后一轮）
            self.assertEqual(sha(out), sha(os.path.join(run, "ckpt_ep10.bin")), f"k={k}")

    def test_two_files_average_is_hand_value_and_rest_comes_from_last(self):
        # a = 0.5*i，b = 1.5*i → 平均恰为 1.0*i；另加 0.1f 与 0.2f 检查 float64 累加后再转 float32
        a = [(0.5 * np.arange(int(np.prod(s)))).reshape(s).astype(np.float32) for s in SHAPES]
        b = [(1.5 * np.arange(int(np.prod(s)))).reshape(s).astype(np.float32) for s in SHAPES]
        a[1][0], b[1][0] = np.float32(0.1), np.float32(0.2)
        run = make_run(self.tmp, [4, 5], lambda e: a if e == 4 else b)
        out = os.path.join(self.tmp, "avg.bin")
        ac.main(["--run", run, "--out", out, "--last-k", "2", "--no-calibrate"])
        got = ac.read_ckpt(out)
        for i, s in enumerate(SHAPES):
            want = np.arange(int(np.prod(s))).reshape(s).astype(np.float32)
            if i == 1:
                want[0] = ((np.float64(np.float32(0.1)) + np.float64(np.float32(0.2))) / 2).astype(np.float32)
            self.assertEqual(want.tobytes(), got["params"][i].tobytes(), f"张量 {i}")
        self.assertEqual(got["epoch"], 5, "头里的 epoch 沿用最后一轮")
        last = slurp(os.path.join(run, "ckpt_ep05.bin"))
        mine = slurp(out)
        self.assertEqual(len(last), len(mine))
        ends = [off + 4 * n for off, n in ac.read_ckpt(out)["spans"]]
        first = ac.read_ckpt(out)["spans"][0][0]
        self.assertEqual(mine[:first], last[:first], "头与契约沿用最后一轮")
        self.assertEqual(mine[ends[-1]:], last[ends[-1]:], "Adam m/v 沿用最后一轮")

    def test_window_is_the_last_k_of_the_run(self):
        # 窗口之外的轮（ep01..03）参数故意取极端值：进了平均就会看出来
        run = make_run(self.tmp, range(1, 7), lambda e: [np.full(s, 1000.0 if e < 4 else float(e), np.float32) for s in SHAPES])
        out = os.path.join(self.tmp, "avg.bin")
        ac.main(["--run", run, "--out", out, "--last-k", "3", "--no-calibrate"])
        for p in ac.read_ckpt(out)["params"]:
            self.assertTrue((p == 5.0).all(), "末 3 轮 = ep04/05/06，平均应为 5")
        meta = load_json(out + ".json")
        self.assertEqual([f["epoch"] for f in meta["inputs"]], [4, 5, 6])
        self.assertEqual([f["sha256"] for f in meta["inputs"]],
                         [sha(os.path.join(run, f"ckpt_ep{e:02d}.bin")) for e in (4, 5, 6)])
        self.assertEqual((meta["last_k"], meta["calibrated"], meta["out_sha256"]), (3, False, sha(out)))

    def test_missing_file_is_an_error(self):
        run = make_run(self.tmp, [8, 10], lambda e: rand_params(e))
        with self.assertRaises(SystemExit) as cm:
            ac.main(["--run", run, "--out", os.path.join(self.tmp, "o.bin"), "--last-k", "3", "--no-calibrate"])
        self.assertIn("ckpt_ep09.bin", str(cm.exception.code))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "o.bin")))
        run2 = make_run(self.tmp, [1, 2], lambda e: rand_params(e), name="short")
        with self.assertRaises(SystemExit):   # k 比 run 的轮数还大
            ac.main(["--run", run2, "--out", os.path.join(self.tmp, "o2.bin"), "--last-k", "3", "--no-calibrate"])
        with self.assertRaises(SystemExit):   # 目录里没有 ckpt_epNN.bin
            ac.main(["--run", self.tmp, "--out", os.path.join(self.tmp, "o3.bin"), "--no-calibrate"])

    def test_shape_mismatch_is_an_error(self):
        def params_of(e):
            ps = rand_params(e)
            if e == 9:
                ps[2] = np.zeros((3, 2, 3, 4), np.float32)   # 与其它轮同元素数以外的形状不同
            return ps
        run = make_run(self.tmp, [8, 9, 10], params_of)
        with self.assertRaises(SystemExit) as cm:
            ac.main(["--run", run, "--out", os.path.join(self.tmp, "o.bin"), "--last-k", "3", "--no-calibrate"])
        self.assertIn("ckpt_ep09.bin", str(cm.exception.code))
        self.assertIn("形状", str(cm.exception.code))
        run2 = make_run(self.tmp, [8, 9], lambda e: rand_params(e), name="hdr")
        write_ckpt(os.path.join(run2, "ckpt_ep09.bin"), rand_params(9), 9, in_h=H + 2)   # 头里的输入尺寸不同
        with self.assertRaises(SystemExit) as cm:
            ac.main(["--run", run2, "--out", os.path.join(self.tmp, "o2.bin"), "--last-k", "2", "--no-calibrate"])
        self.assertIn("结构", str(cm.exception.code))

    def test_header_epoch_must_match_file_name(self):
        run = make_run(self.tmp, [8, 9], lambda e: rand_params(e))
        write_ckpt(os.path.join(run, "ckpt_ep09.bin"), rand_params(9), 7)   # 文件名 ep09，头里 epoch=7
        with self.assertRaises(SystemExit) as cm:
            ac.main(["--run", run, "--out", os.path.join(self.tmp, "o.bin"), "--last-k", "2", "--no-calibrate"])
        self.assertIn("epoch", str(cm.exception.code))

    def test_unknown_formats_are_rejected(self):
        ps = rand_params(0)
        for name, kw in (("v5", {"version": 5}), ("trailing", {"trailing": b"x"})):
            p = os.path.join(self.tmp, name + ".bin")
            write_ckpt(p, ps, 1, **kw)
            with self.assertRaises(ValueError, msg=name):
                ac.read_ckpt(p)
        p = os.path.join(self.tmp, "short.bin")
        write_ckpt(p, ps, 1)
        data = slurp(p)
        spit(p, data[:-4])
        with self.assertRaises(ValueError):
            ac.read_ckpt(p)
        spit(p, b"NOTACKPT" + data[8:])
        with self.assertRaises(ValueError):
            ac.read_ckpt(p)
        bad = [q.copy() for q in ps]
        bad[7][2] = np.nan
        p = os.path.join(self.tmp, "nan.bin")
        write_ckpt(p, bad, 1)
        with self.assertRaises(ValueError):
            ac.read_ckpt(p)

    def test_refuses_to_overwrite_existing_output(self):
        run = make_run(self.tmp, [1], lambda e: rand_params(e))
        out = os.path.join(self.tmp, "o.bin")
        spit(out, b"keep")
        with self.assertRaises(SystemExit):
            ac.main(["--run", run, "--out", out, "--last-k", "1", "--no-calibrate"])
        self.assertEqual(slurp(out), b"keep")


class TestCalibrate(Base):
    def test_bisection_hits_target_within_one_frame(self):
        rng = np.random.default_rng(7)
        n = 500
        for base_bias, target_count in ((0.0, 400), (0.0, 100), (-2.0, 250), (3.0, 20)):
            logits = rng.standard_normal((n, 4)) * 2
            logits[:, FWD] += base_bias
            p = np.exp(logits - logits.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            delta = ac.calibrate(p, target_count / n)
            got = ac.forward_frac(p, delta) * n
            self.assertLessEqual(abs(got - target_count), 1, f"base_bias={base_bias} target={target_count} got={got}")

    def test_unreachable_target_is_an_error(self):
        p = np.full((20, 4), 0.25)
        p[:10] = [0.5, 0.0, 0.5, 0.0]    # 一半帧 FORWARD 概率为 0：log 截在 -690，加 30 也翻不过来
        with self.assertRaises(ValueError):
            ac.calibrate(p, 1.0)

    def test_only_forward_bias_changes(self):
        ps = rand_params(3)
        run = make_run(self.tmp, [10], lambda e: ps)
        avg = os.path.join(self.tmp, "avg.bin")
        ac.main(["--run", run, "--out", avg, "--last-k", "1", "--no-calibrate"])
        base = ac.read_ckpt(avg)
        new = ac.shift_forward_bias(base["params"], 1.7000000001)
        cal = os.path.join(self.tmp, "cal.bin")
        ac.write_ckpt(base, new, cal)
        got = ac.read_ckpt(cal)
        for i, (a, b) in enumerate(zip(base["params"], got["params"])):
            if i != 7:
                self.assertEqual(a.tobytes(), b.tobytes(), f"张量 {i} 必须逐位不变")
        self.assertEqual(got["params"][7][FWD], base["params"][7][FWD] + np.float32(1.7000000001))
        for k in (0, 2, 3):
            self.assertEqual(got["params"][7][k], base["params"][7][k])
        off7 = base["spans"][7][0]
        diff = [i for i, (x, y) in enumerate(zip(slurp(avg), slurp(cal))) if x != y]
        self.assertTrue(diff and all(off7 + 4 * FWD <= i < off7 + 4 * FWD + 4 for i in diff), f"字节差异应只在 fc.b[1]：{diff[:8]}")


class TestEndToEnd(Base):
    """假推理二进制（同行协议；logit = fc.b + 帧内容）跑完整流程：只有 fc.b[FORWARD] 变，校准后占比等于目标。"""

    def setUp(self):
        super().setUp()
        self.fake = os.path.join(self.tmp, "fake_infer")
        with open(self.fake, "w") as f:
            f.write(FAKE_INFER.format(py=sys.executable, tools=os.path.join(REPO, "tools")))
        os.chmod(self.fake, 0o755)
        rng = np.random.default_rng(5)
        self.n_split = {"train": 30, "val": 80, "test": 20}
        n = sum(self.n_split.values())
        self.images = rng.integers(0, 256, size=(n, 3, H, W), dtype=np.uint8)
        self.labels = rng.integers(0, 4, size=n)
        self.labels[30:110] = np.where(rng.random(80) < 0.7, 1, rng.integers(2, 4, size=80))
        flags = ["0"] * n
        for i in (35, 36, 37, 38):     # val 里 4 帧碰撞：不进校准
            flags[i] = "1"
        self.flags = "".join(flags)
        self.data = os.path.join(self.tmp, "dataset.uavr")
        write_uavr(self.data, self.n_split, self.labels, self.flags, self.images)
        self.run_dir = make_run(self.tmp, range(1, 6), lambda e: rand_params(e), ds_sha=sha(self.data))

    def test_calibrated_output(self):
        out = os.path.join(self.tmp, "cal.bin")
        rc = ac.main(["--run", self.run_dir, "--data", self.data, "--out", out, "--last-k", "3", "--infer-bin", self.fake])
        self.assertEqual(rc, 0)
        meta = load_json(out + ".json")
        val = [i for i in range(30, 110) if self.flags[i] == "0"]
        self.assertEqual(meta["n_frames"], len(val))
        target = float((self.labels[val] == 1).mean())
        self.assertAlmostEqual(meta["forward_target"], target, places=12)
        self.assertLessEqual(abs(meta["forward_frac_after_real"] - target), 2.0 / len(val) + 1e-12)
        self.assertNotAlmostEqual(meta["forward_frac_before"], target, places=3, msg="造的数据校准前占比应偏离目标")
        # 输出 = 只平均的产物 + 偏置的 FORWARD 分量
        avg = os.path.join(self.tmp, "avg.bin")
        ac.main(["--run", self.run_dir, "--out", avg, "--last-k", "3", "--no-calibrate"])
        a, c = ac.read_ckpt(avg), ac.read_ckpt(out)
        for i in range(8):
            if i != 7:
                self.assertEqual(a["params"][i].tobytes(), c["params"][i].tobytes())
        self.assertEqual(c["params"][7][FWD], a["params"][7][FWD] + np.float32(meta["delta"]))
        self.assertEqual(sha(out), meta["out_sha256"])
        self.assertEqual(meta["infer_bin_sha256"], sha(self.fake))
        self.assertEqual(meta["data"], os.path.abspath(self.data))
        self.assertEqual(meta["data_sha256"], sha(self.data))
        self.assertEqual(meta["split"], "val")
        self.assertFalse(os.path.exists(out + ".tmp"))
        self.assertEqual([n for n in os.listdir(self.tmp) if n.startswith(".average_ckpt")], [], "临时目录要清掉")

    def test_collision_flags_required_and_validated(self):
        for name, flags in (("missing", None), ("short", "0" * 5), ("badchar", "x" * len(self.flags))):
            data = os.path.join(self.tmp, f"{name}.uavr")
            write_uavr(data, self.n_split, self.labels, flags, self.images)
            with self.assertRaises(SystemExit, msg=name) as cm:
                ac.main(["--run", self.run_dir, "--data", data, "--out", os.path.join(self.tmp, name + ".bin"),
                         "--last-k", "2", "--infer-bin", self.fake])
            self.assertIn("collision_flags", str(cm.exception.code))

    def test_data_must_be_the_one_the_ckpt_was_trained_on(self):
        other = os.path.join(self.tmp, "other.uavr")
        labels = self.labels.copy()
        labels[40] = (labels[40] + 1) % 4     # 只差一个 val 标签：格式全合法，sha256 不同
        write_uavr(other, self.n_split, labels, self.flags, self.images)
        out = os.path.join(self.tmp, "x.bin")
        with self.assertRaises(SystemExit) as cm:
            ac.main(["--run", self.run_dir, "--data", other, "--out", out, "--last-k", "2", "--infer-bin", self.fake])
        self.assertIn("sha256", str(cm.exception.code))
        self.assertFalse(os.path.exists(out) or os.path.exists(out + ".json"))

    def test_unsupported_inputs_are_refused_not_guessed(self):
        # 文本条件 / 双帧 / 认不得的 prev_dim：行协议无法从数据集还原这些输入，不能静默算出错的 δ。
        # prev_dim=4（prev-action 与航向码）自卡 E2E-1 起支持，见 TestEndToEndCond。
        for name, kw in (("text", {"text_dim": 16}), ("prev_dim3", {"prev": 3}), ("two_frame", {"in_c": 6}),
                         ("prev_two_frame", {"prev": 4, "in_c": 6}), ("prev_text", {"prev": 4, "text_dim": 16})):
            run = make_run(self.tmp, [1, 2], lambda e: rand_params(e), name="run_" + name, **kw)
            with self.assertRaises(SystemExit, msg=name) as cm:
                ac.main(["--run", run, "--data", self.data, "--out", os.path.join(self.tmp, name + ".bin"),
                         "--last-k", "2", "--infer-bin", self.fake])
            self.assertIn("暂不支持", str(cm.exception.code))
            # 只平均不需要推理，照常可用
            out = os.path.join(self.tmp, name + "_avg.bin")
            ac.main(["--run", run, "--out", out, "--last-k", "2", "--no-calibrate"])
            self.assertTrue(os.path.isfile(out))

    def test_data_required_unless_no_calibrate(self):
        with self.assertRaises(SystemExit):
            ac.main(["--run", self.run_dir, "--out", os.path.join(self.tmp, "o.bin"), "--last-k", "2"])

    def test_calibration_follows_contract_drop_collision_and_locks_delivered_bytes(self):
        """C6 / I2-5 第 2 点：average_ckpt 按最后一轮 ckpt 契约的 drop_collision 位决定是否剔除碰撞步，
        在 .json 记下 drop_collision，全零契约报错拒绝，且 drop_collision=True 校准产物与改前逐字节一致。"""
        # 1) 锁死改前 drop_collision=True 的校准产物逐字节不变
        out_drop = os.path.join(self.tmp, "cal_drop_true.bin")
        rc = ac.main(["--run", self.run_dir, "--data", self.data, "--out", out_drop, "--last-k", "3", "--infer-bin", self.fake])
        self.assertEqual(rc, 0)
        self.assertEqual(sha(out_drop), "aa39f3cd40e6887ebf69b38dbb60afe3c337286e85a3908d27b5e9a519c00c51",
                         "drop_collision=True 的校准产物字节不得改变")
        meta_drop = load_json(out_drop + ".json")
        self.assertIs(meta_drop.get("drop_collision"), True)
        self.assertEqual(meta_drop["n_frames"], 76)

        # 2) drop_collision=False：保留 val 全部 80 帧（令 4 个碰撞帧标签全为 0 以拉大与 76 帧目标的差距）
        labels2 = self.labels.copy()
        for i in (35, 36, 37, 38):
            labels2[i] = 0
        data_keep = os.path.join(self.tmp, "keep.uavr")
        write_uavr(data_keep, self.n_split, labels2, self.flags, self.images)
        run_keep = make_run(self.tmp, range(1, 6), lambda e: rand_params(e), name="run_keep",
                            ds_sha=sha(data_keep), drop_collision=False)
        out_keep = os.path.join(self.tmp, "cal_drop_false.bin")
        self.assertEqual(ac.main(["--run", run_keep, "--data", data_keep, "--out", out_keep,
                                  "--last-k", "3", "--infer-bin", self.fake]), 0)
        meta_keep = load_json(out_keep + ".json")
        self.assertIs(meta_keep.get("drop_collision"), False)
        self.assertEqual(meta_keep["n_frames"], 80)
        self.assertAlmostEqual(meta_keep["forward_target"], float((labels2[30:110] == 1).mean()), places=12)

        # 3) drop_collision=False 时即使老数据集没有 collision_flags 字段也可正常按全量 val 校准（同 src/dataset.cpp:519）
        data_noflags = os.path.join(self.tmp, "noflags.uavr")
        write_uavr(data_noflags, self.n_split, labels2, None, self.images)
        run_noflags = make_run(self.tmp, range(1, 6), lambda e: rand_params(e), name="run_noflags",
                               ds_sha=sha(data_noflags), drop_collision=False)
        out_noflags = os.path.join(self.tmp, "cal_noflags.bin")
        self.assertEqual(ac.main(["--run", run_noflags, "--data", data_noflags, "--out", out_noflags,
                                  "--last-k", "3", "--infer-bin", self.fake]), 0)
        self.assertEqual(load_json(out_noflags + ".json")["n_frames"], 80)

        # 4) 全零契约（read_v5_contract 返回 false）：无训练记录，拒绝静默回落
        run_zero = make_run(self.tmp, range(1, 6), lambda e: rand_params(e), name="run_zero", zero_contract=True)
        with self.assertRaises(SystemExit) as cm:
            ac.main(["--run", run_zero, "--data", self.data, "--out", os.path.join(self.tmp, "zero.bin"),
                     "--last-k", "3", "--infer-bin", self.fake])
        self.assertIn("训练契约全零", str(cm.exception.code))

        # 5) 若数据盘上存在现有交付权重，只读核验其契约全部解析为 present=True, drop_collision=True
        real_dir = os.path.join(_DATA_TMP, "..", "exp-swa-calib-20261110", "ckpt")
        if os.path.isdir(real_dir):
            checked = 0
            for fn in sorted(os.listdir(real_dir)):
                if fn.endswith(".bin"):
                    c = ac.parse_v5_contract(ac.read_ckpt(os.path.join(real_dir, fn))["contract"])
                    self.assertTrue(c["present"] and c["drop_collision"], f"{fn}: {c}")
                    checked += 1
            self.assertGreater(checked, 0)

    def test_read_val_call_shape_matches_verify_infer_vs_evalonly(self):
        """experiments/2026-11-12-e2e-heading-code/verify_infer_vs_evalonly.py 这样用：`images, idx, y = ac.read_val(args.data)`。
        用 ast 取出那个脚本里对 ac.read_val 的全部调用：每个都能绑到当前签名，结果被解成三个名字；
        再按脚本的形态（一个位置参数）真调一次，缺省仍剔除碰撞帧（改前的行为）。"""
        script = os.path.join(REPO, "experiments", "2026-11-12-e2e-heading-code", "verify_infer_vs_evalonly.py")
        with open(script, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=script)

        def is_read_val(call):
            return (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "read_val"
                    and isinstance(call.func.value, ast.Name) and call.func.value.id == "ac")
        calls = [n for n in ast.walk(tree) if is_read_val(n)]
        widths = [len(n.targets[0].elts) for n in ast.walk(tree)
                  if isinstance(n, ast.Assign) and is_read_val(n.value) and isinstance(n.targets[0], ast.Tuple)]
        self.assertTrue(calls, "脚本里应有 ac.read_val 的调用")
        sig = inspect.signature(ac.read_val)
        for call in calls:
            self.assertFalse(any(isinstance(a, ast.Starred) for a in call.args))
            sig.bind(*[None] * len(call.args), **{k.arg: None for k in call.keywords})   # 绑不上就 TypeError
        self.assertTrue(widths and all(w == 3 for w in widths), f"脚本应把 read_val 的结果解成三个名字：{widths}")
        images, idx, y = ac.read_val(self.data)
        self.assertEqual((len(idx), len(y)), (76, 76), "缺省 drop_collision=True：80 帧 val 剔掉 4 个碰撞帧")
        self.assertEqual(len(images), len(self.images))


class TestAuxCond(Base):
    """UAVRAUX1 侧车的 prev_action 列读取：逐项绑定核对，任何不符都报错（同 src/aux_sidecar.cpp）。"""

    def setUp(self):
        super().setUp()
        rng = np.random.default_rng(11)
        self.n_split = {"train": 12, "val": 9, "test": 4}
        n = sum(self.n_split.values())
        images = rng.integers(0, 256, size=(n, 3, H, W), dtype=np.uint8)
        self.data = os.path.join(self.tmp, "dataset.uavr")
        write_uavr(self.data, self.n_split, rng.integers(0, 4, size=n), "0" * n, images)
        self.sha = sha(self.data)
        self.cond = rng.choice([0, 2, 3], size=n).astype(np.uint8)

    def test_returns_the_prev_action_column_with_and_without_depth(self):
        for depth in (None, (1, 2)):
            write_aux(self.data, self.cond, depth_hw=depth)
            got, path = ac.read_aux_cond(self.data, self.sha)
            self.assertEqual(got.tolist(), self.cond.tolist(), f"depth={depth}")
            self.assertEqual(path, os.path.join(self.tmp, "dataset.aux"))

    def test_sidecar_name_follows_the_dataset_name(self):
        other = os.path.join(self.tmp, "other.uavr")
        spit(other, slurp(self.data))
        write_aux(other, self.cond)
        self.assertEqual(ac.read_aux_cond(other, self.sha)[1], os.path.join(self.tmp, "other.aux"))

    def test_every_binding_check_fires(self):
        n = len(self.cond)
        bad_value = self.cond.copy()
        bad_value[3] = 4                      # prev_dim=4 → 合法值 0..3
        cases = [
            ("missing", None, {}, "不存在"),
            ("magic", self.cond, {"magic": b"UAVRAUX2"}, "头不合法"),
            ("n", self.cond, {"n_override": n + 1}, "头不合法"),
            ("flags", self.cond, {"flags_override": 2}, "头不合法"),
            ("unknown_code", self.cond, {"unknown": 254}, "头不合法"),
            ("uavr_sha", self.cond, {"uavr_sha": "00" * 32}, "uavr sha256"),
            ("payload_sha", self.cond, {"corrupt_payload": True}, "payload sha256"),
            ("trailing", self.cond, {"trailing": b"x"}, "文件长"),
            ("truncated", self.cond, {"drop_tail": 1}, "文件长"),
            ("out_of_range", bad_value, {}, "越界"),
        ]
        for name, col, kw, want in cases:
            aux = os.path.splitext(self.data)[0] + ".aux"
            if os.path.exists(aux):
                os.remove(aux)
            if col is not None:
                write_aux(self.data, col, **kw)
            with self.assertRaises(ValueError, msg=name) as cm:
                ac.read_aux_cond(self.data, self.sha)
            self.assertIn(want, str(cm.exception), name)


class TestEndToEndCond(Base):
    """prev_dim=4 的 ckpt：条件字节取自侧车 prev_action 列、只对 val 非碰撞帧、按帧序发出；δ 与独立算法一致；不带条件的 ckpt 不多发字节。"""

    def setUp(self):
        super().setUp()
        self.fake = os.path.join(self.tmp, "fake_infer_cond")
        with open(self.fake, "w") as f:
            f.write(FAKE_INFER_COND.format(py=sys.executable, tools=os.path.join(REPO, "tools"), coef=COND_COEF))
        os.chmod(self.fake, 0o755)
        rng = np.random.default_rng(5)
        self.n_split = {"train": 30, "val": 80, "test": 20}
        n = sum(self.n_split.values())
        self.images = rng.integers(0, 256, size=(n, 3, H, W), dtype=np.uint8)
        self.labels = rng.integers(0, 4, size=n)
        self.labels[30:110] = np.where(rng.random(80) < 0.7, 1, rng.integers(2, 4, size=80))
        flags = ["0"] * n
        for i in (35, 36, 37, 38):     # val 里 4 帧碰撞：不进校准，条件也不能错位
            flags[i] = "1"
        self.flags = "".join(flags)
        self.cond = rng.choice([0, 2, 3], size=n, p=[0.5, 0.25, 0.25]).astype(np.uint8)   # 码 0/2/3，帧间无规律：错位取值必被发现
        self.data = os.path.join(self.tmp, "dataset.uavr")
        write_uavr(self.data, self.n_split, self.labels, self.flags, self.images)
        write_aux(self.data, self.cond)
        self.val = [i for i in range(30, 110) if self.flags[i] == "0"]
        self.run_dir = make_run(self.tmp, range(1, 6), lambda e: rand_params(e), ds_sha=sha(self.data), prev=4)

    def expected_delta(self, avg_fc_b, cond):
        """独立于工具与假二进制脚本之外重算一遍：与 FAKE_INFER_COND 同一个 logit 公式，概率按 %g 取 6 位，再用工具的二分求 δ。"""
        probs = []
        for seen, (g, c) in enumerate(zip(self.val, cond)):
            buf = np.ascontiguousarray(self.images[g].transpose(1, 2, 0)).tobytes()
            z = [float(avg_fc_b[k]) + buf[k] / 64.0 + int(c) * COND_COEF[k] for k in range(4)]
            z[1] += 1e-3 * seen
            m = max(z)
            e = [math.exp(x - m) for x in z]
            probs.append([float("%g" % (x / sum(e))) for x in e])
        return ac.calibrate(np.array(probs), float((self.labels[self.val] == 1).mean()))

    def calibrate(self, run=None, name="cal", **env):
        out = os.path.join(self.tmp, name + ".bin")
        log = os.path.join(self.tmp, name + ".condlog")
        old = {k: os.environ.get(k) for k in ("COND_LOG", *env)}
        os.environ["COND_LOG"] = log
        os.environ.update(env)
        try:
            ac.main(["--run", run or self.run_dir, "--data", self.data, "--out", out, "--last-k", "3", "--infer-bin", self.fake])
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        lines = [tuple(map(int, ln.split())) for ln in slurp(log).decode().splitlines()] if os.path.exists(log) else []
        return out, lines

    def test_condition_bytes_reach_inference_in_val_order_and_delta_matches_independent_value(self):
        out, lines = self.calibrate()
        want = [int(self.cond[g]) for g in self.val]
        n = H * W * 3
        self.assertEqual(len(lines), 2 * len(self.val), "平均权重与校准后权重各推理一遍 val")
        self.assertEqual({ln[0] for ln in lines}, {n + 1}, "每帧 h*w*3 字节 + 1 字节条件")
        self.assertEqual([ln[1] for ln in lines[:len(self.val)]], want, "第一遍：条件取自侧车 val 非碰撞帧，按帧序")
        self.assertEqual([ln[1] for ln in lines[len(self.val):]], want, "第二遍（校准后核对）同样")
        self.assertGreater(len(set(want)), 1, "造的条件列要有不止一个取值，否则测不出错位")
        avg = os.path.join(self.tmp, "avg.bin")
        ac.main(["--run", self.run_dir, "--out", avg, "--last-k", "3", "--no-calibrate"])
        fc_b = ac.read_ckpt(avg)["params"][7]
        meta = load_json(out + ".json")
        self.assertEqual(meta["delta"], self.expected_delta(fc_b, want), "δ 必须等于按真实条件列独立算出的值")
        self.assertNotAlmostEqual(meta["delta"], self.expected_delta(fc_b, [0] * len(self.val)), places=4,
                                  msg="条件全置 0 的 δ 应不同，否则这个测试对条件不敏感")
        target = float((self.labels[self.val] == 1).mean())
        self.assertLessEqual(abs(meta["forward_frac_after_real"] - target), 2.0 / len(self.val) + 1e-12)
        self.assertEqual(meta["cond"]["prev_dim"], 4)
        self.assertEqual(meta["cond"]["aux"], os.path.join(self.tmp, "dataset.aux"))
        self.assertEqual(meta["cond"]["aux_sha256"], sha(os.path.join(self.tmp, "dataset.aux")))
        self.assertEqual(meta["cond"]["val_value_counts"], [want.count(v) for v in range(4)])
        # 输出仍是「只平均 + fc.b[FORWARD] 加 δ」：别的张量逐位等于只平均的产物
        a, c = ac.read_ckpt(avg), ac.read_ckpt(out)
        for i in range(8):
            if i != 7:
                self.assertEqual(a["params"][i].tobytes(), c["params"][i].tobytes())
        self.assertEqual(c["params"][7][FWD], a["params"][7][FWD] + np.float32(meta["delta"]))
        self.assertEqual(c["prev_dim"], 4)

    def test_missing_or_foreign_sidecar_is_refused_and_nothing_is_written(self):
        os.remove(os.path.join(self.tmp, "dataset.aux"))
        out = os.path.join(self.tmp, "x.bin")
        with self.assertRaises(SystemExit) as cm:
            ac.main(["--run", self.run_dir, "--data", self.data, "--out", out, "--last-k", "2", "--infer-bin", self.fake])
        self.assertIn("dataset.aux", str(cm.exception.code))
        write_aux(self.data, self.cond, uavr_sha="11" * 32)       # 别的数据集的侧车
        with self.assertRaises(SystemExit) as cm:
            ac.main(["--run", self.run_dir, "--data", self.data, "--out", out, "--last-k", "2", "--infer-bin", self.fake])
        self.assertIn("uavr sha256", str(cm.exception.code))
        self.assertFalse(os.path.exists(out) or os.path.exists(out + ".json"))
        self.assertEqual([n for n in os.listdir(self.tmp) if n.startswith(".average_ckpt")], [])

    def test_infer_process_that_does_not_take_a_condition_is_refused(self):
        # hello 报 prev_action_dim=0 而 ckpt 需要条件：每帧长度对不上，后面全部错位，必须在第一帧之前拦下
        with self.assertRaises(SystemExit) as cm:
            self.calibrate(name="hello", FAKE_HELLO_PREV="0")
        self.assertIn("prev_action_dim", str(cm.exception.code))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "hello.bin")))

    def test_ckpt_without_condition_sends_no_extra_byte_and_has_no_cond_record(self):
        run0 = make_run(self.tmp, range(1, 6), lambda e: rand_params(e), name="run_plain", ds_sha=sha(self.data))
        out, lines = self.calibrate(run=run0, name="plain")
        self.assertEqual({ln[0] for ln in lines}, {H * W * 3}, "不带条件的 ckpt 每帧只发图像字节")
        self.assertEqual({ln[1] for ln in lines}, {0})
        meta = load_json(out + ".json")
        self.assertNotIn("cond", meta)
        # 与原假二进制（只认 h*w*3 字节一帧）算出同一个 δ：协议没有为这类 ckpt 改动
        fake_old = os.path.join(self.tmp, "fake_infer_old")
        with open(fake_old, "w") as f:
            f.write(FAKE_INFER.format(py=sys.executable, tools=os.path.join(REPO, "tools")))
        os.chmod(fake_old, 0o755)
        out_old = os.path.join(self.tmp, "plain_old.bin")
        ac.main(["--run", run0, "--data", self.data, "--out", out_old, "--last-k", "3", "--infer-bin", fake_old])
        self.assertEqual(load_json(out_old + ".json")["delta"], meta["delta"])
        self.assertEqual(sha(out_old), sha(out), "不带条件的 ckpt：两种假二进制得到逐字节相同的产物")

    def test_only_average_still_works_for_condition_ckpt_without_sidecar(self):
        os.remove(os.path.join(self.tmp, "dataset.aux"))
        out = os.path.join(self.tmp, "avg.bin")
        ac.main(["--run", self.run_dir, "--out", out, "--last-k", "2", "--no-calibrate"])
        self.assertTrue(os.path.isfile(out))

    def test_json_records_dataset_cond_abs_data_and_writes_atomically(self):
        """I2-5 第 1、3 点：.json 记录数据集 meta.cond（heading_code.v1 或 None）与数据集绝对路径（即使 --data 给相对路径），
        且通过 .json.tmp -> os.replace 原子落盘（写 JSON 中途失败不留半截 .json）。"""
        hc_data = os.path.join(self.tmp, "sub", "hc.uavr")
        os.makedirs(os.path.dirname(hc_data), exist_ok=True)
        write_uavr(hc_data, self.n_split, self.labels, self.flags, self.images, extra_meta={"cond": "heading_code.v1"})
        write_aux(hc_data, self.cond)
        hc_run = make_run(self.tmp, range(1, 6), lambda e: rand_params(e), name="run_hc", ds_sha=sha(hc_data), prev=4)
        out_hc = os.path.join(self.tmp, "cal_hc.bin")
        old_cwd = os.getcwd()
        replaced = []
        real_replace = ac.os.replace
        ac.os.replace = lambda s, d: (replaced.append((s, d)), real_replace(s, d))[1]
        try:
            os.chdir(self.tmp)
            rc = ac.main(["--run", hc_run, "--data", os.path.join("sub", "hc.uavr"),
                          "--out", out_hc, "--last-k", "3", "--infer-bin", self.fake])
        finally:
            ac.os.replace = real_replace
            os.chdir(old_cwd)
        self.assertEqual(rc, 0)
        meta_hc = load_json(out_hc + ".json")
        self.assertEqual(meta_hc["data"], os.path.abspath(hc_data))
        self.assertEqual(meta_hc["dataset_cond"], "heading_code.v1")
        self.assertEqual(meta_hc["cond"]["dataset_cond"], "heading_code.v1")
        self.assertIn((out_hc + ".json.tmp", out_hc + ".json"), replaced, "写 .json 必须先写 .json.tmp 再 os.replace")
        self.assertFalse(os.path.exists(out_hc + ".json.tmp"))

        # 无 cond 键的 prev-action 数据集：dataset_cond 显式记为 None（键存在）
        out_pa, _ = self.calibrate(name="cal_pa_meta")
        meta_pa = load_json(out_pa + ".json")
        self.assertIn("dataset_cond", meta_pa)
        self.assertIsNone(meta_pa["dataset_cond"])
        self.assertIn("dataset_cond", meta_pa["cond"])
        self.assertIsNone(meta_pa["cond"]["dataset_cond"])


# 今天闭环实验用的权重（exp-swa-calib-20261110/inputs.txt 的 sha256）
REF_SHA = {
    "swa_s0_k10": "15570258caf075802f1054482ad359c3fcfefff000db5407765d7f6b4eba6534",
    "swa_s1_k10": "b00ddf1756a514229f4739a09f6598739d2ee0b054d05e0aca70b15c68080e2c",
    "swa_s2_k10": "5ac4aa9aead3f16c609cb1e3bcd275032c5a9cbaf46de84a6f2631a32d6fe436",
    "cal_swa_s0_k10": "2fb25b30b337589999ad2385e2372bb38ae1ca6e9b0b28d6889d25bf4a327928",
    "cal_swa_s1_k10": "5cfe51f58f4f4a1d1afb4f79db0f11e777fd67271b9605f722b6bdf8372a6900",
    "cal_swa_s2_k10": "abcae7cf0d09a475f68356535e14c0afeaf201aefa0fd5f0375ceff7d4b17734",
}


@unittest.skipUnless(os.environ.get("UAVRGB_REPRO_AVGCKPT") == "1", "需要 GPU 与数据盘；UAVRGB_REPRO_AVGCKPT=1 才跑")
class TestReproduction(unittest.TestCase):
    """用 average_ckpt.py 重新生成 11-10 闭环用的 swa / cal_swa 权重，对 sha256；对不上时给出参数最大绝对差与 δ 差。"""

    def test_regenerate_swa_and_calswa(self):
        d = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb")
        runs = f"{d}/exp-student-select-20261109/e1-noise"
        ref = f"{d}/exp-swa-calib-20261110/ckpt"
        data = f"{d}/tmp/agent6-distill/trunc/ds/dataset.uavr"
        infer = os.environ.get("UAVRGB_REPRO_INFER", f"{d}/tmp/cursor-1110/build-check/uavrgb_infer")
        parent = os.environ.get("UAVRGB_REPRO_OUT")   # 父目录；每次在里面新建一个目录，工具不覆盖已有产物
        if parent:
            os.makedirs(parent, exist_ok=True)
        out_dir = tempfile.mkdtemp(prefix="repro_avgckpt_", dir=parent)
        print(f"产物目录 {out_dir}；推理二进制 {infer}")
        bad = []
        for s in (0, 1, 2):
            run = f"{runs}/seed{s}/run"
            swa, cal = f"{out_dir}/swa_s{s}_k10.bin", f"{out_dir}/cal_swa_s{s}_k10.bin"
            ac.main(["--run", run, "--out", swa, "--last-k", "10", "--no-calibrate"])
            ac.main(["--run", run, "--data", data, "--out", cal, "--last-k", "10", "--infer-bin", infer])
            for tag, got in ((f"swa_s{s}_k10", swa), (f"cal_swa_s{s}_k10", cal)):
                print(f"{tag}: sha256 {sha(got)} 参考 {REF_SHA[tag]} {'相同' if sha(got) == REF_SHA[tag] else '不同'}")
                if sha(got) != REF_SHA[tag]:
                    a, b = ac.read_ckpt(got)["params"], ac.read_ckpt(f"{ref}/{tag}.bin")["params"]
                    bad.append(f"{tag}: 参数最大绝对差 {max(float(np.abs(x - y).max()) for x, y in zip(a, b)):.3e}")
            ref_delta = float(ac.read_ckpt(f"{ref}/cal_swa_s{s}_k10.bin")["params"][7][FWD]
                              - ac.read_ckpt(f"{ref}/swa_s{s}_k10.bin")["params"][7][FWD])
            mine = load_json(cal + ".json")["delta"]
            print(f"seed{s}: δ 本工具 {mine!r} 参考（两个 ckpt 的 fc.b[1] 之差，float32）{ref_delta!r}")
        self.assertEqual(bad, [], "sha256 对不上")


if __name__ == "__main__":
    unittest.main(verbosity=2)
