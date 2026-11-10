#!/usr/bin/env python3
"""tools/average_ckpt.py 的测试。纯 Python 部分不需要 GPU：合成小 ckpt / 小数据集，推理进程用同协议的假二进制。
复现性验收（真实 GPU、真实数据盘）默认跳过，开关 UAVRGB_REPRO_AVGCKPT=1：

  python tools/test_average_ckpt.py                                  # 进 check.sh 的部分
  UAVRGB_REPRO_AVGCKPT=1 CUDA_VISIBLE_DEVICES=<UUID> python tools/test_average_ckpt.py TestReproduction

复现性验收的产物落在 UAVRGB_REPRO_OUT（父目录，默认 TMPDIR）下新建的目录里；推理二进制用 UAVRGB_REPRO_INFER 换。
"""
import hashlib
import json
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


def write_ckpt(path, params, epoch, version=6, text_dim=0, prev=0, in_c=3, in_h=H, trailing=b"", ds_sha=None):
    """独立于工具的 v6 写法（布局见 src/include/uavrgb/checkpoint.hpp）。契约 113 B 与 Adam m/v 随轮次变，用来核对「沿用最后一轮」。"""
    b = bytearray(b"UAVRCK01" + struct.pack("<II", version, len(params)))
    b += struct.pack("<4i", in_h, W, in_c, 4) + struct.pack("<3I", 100 + epoch, epoch, 4)
    b += struct.pack("<4f", 36.6, 0.3, 3.2, 3.3) + struct.pack("<f", 0.4) + struct.pack("<i", text_dim)
    contract = bytes((epoch + i) % 256 for i in range(113))
    if ds_sha:   # 契约开头 32 B 是训练数据集 sha256
        contract = bytes.fromhex(ds_sha) + contract[32:]
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


def write_uavr(path, n_per_split, labels, collisions, images):
    """按 tools/DATASET_FORMAT.md §1 写小数据集：每个 split 一集；collisions 为 '0'/'1' 串或 None（不写该字段）。"""
    n = sum(n_per_split.values())
    meta = {"episodes": [], "split_policy": "scene_id"}
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
        # 文本条件 / prev-action / 双帧：行协议无法从数据集还原这些输入，不能静默算出错的 δ
        for name, kw in (("text", {"text_dim": 16}), ("prev", {"prev": 4}), ("two_frame", {"in_c": 6})):
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
