#!/usr/bin/env python3
"""proto/infer_wrap.py 与启动器 proto/uavrgb_infer_proto 的协议测试。

用合成的小 checkpoint（16×24 输入，CPU）起真实子进程，检查行协议的每一条：hello 字段、payload 解析（HWC→CHW、
双帧顺序、prev-action 字节、文本 id）、回复格式、EOF/截断/越界/用法错误的退出码、stdout 不被库打印污染。
GPU 路径与 C++ 逐步对拍见 proto/replay_compare.py 与 proto/test_forward_parity.py。没有 torch 的解释器只跑命令行解析测试。
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ckpt_io as cio  # noqa: E402
import infer_wrap as iw  # noqa: E402

HAVE_TORCH = importlib.util.find_spec("torch") is not None
LAUNCHER = os.path.join(HERE, "uavrgb_infer_proto")
H, W = 16, 24

if HAVE_TORCH:
    import torch
    import model as pm


def env():
    e = dict(os.environ)
    e["UAV_RGB_PROTO_PYTHON"] = sys.executable
    e["UAV_RGB_PROTO_DEVICE"] = "cpu"
    e["PYTHONDONTWRITEBYTECODE"] = "1"
    return e


def make_model(seed=0, **st):
    """Xavier 量级的随机权重加小偏置（同 parity_forward.random_model），概率不会饱和，对拍才有区分力。"""
    import parity_forward as pf
    structure = dict(in_h=H, in_w=W, in_c=3, num_classes=4, text_dim=0, aux_classes=0, prev_dim=0)
    structure.update(st)
    return pf.random_model(structure, seed)


def write_ckpt(path, model):
    cio.write_checkpoint(path, pm.ckpt_from_model(model, epoch=1))


def hwc(frame_chw):
    return np.ascontiguousarray(frame_chw.transpose(1, 2, 0)).tobytes()


def run(args, payload=b"", timeout=120):
    return subprocess.run([LAUNCHER] + args, input=payload, capture_output=True, env=env(), timeout=timeout)


class TestParse(unittest.TestCase):
    def test_text_tokens(self):
        self.assertEqual(iw.parse_text(""), [])
        self.assertEqual(iw.parse_text("1,2,21"), [1, 2, 21])
        self.assertEqual(iw.parse_text(" 3,+4,0"), [3, 4, 0])     # strtol 接受前导空白与 +
        for bad in ("a", "22", "1,,2", "-1", ",", "1,", "3 ", "0x10", "1.5", "٣", "1_0"):
            with self.subTest(bad):
                with self.assertRaises(iw.UsageError):
                    iw.parse_text(bad)

    def test_args(self):
        self.assertEqual(iw.parse_args(["--ckpt", "a.bin"]), ("a.bin", []))
        self.assertEqual(iw.parse_args(["--text", "1,2", "--ckpt", "a.bin"]), ("a.bin", [1, 2]))
        for bad in ([], ["--ckpt"], ["--ckpt", "a", "--bogus"], ["--text", "1"], ["a.bin"], ["--ckpt", "a", "--text"]):
            with self.subTest(bad):
                with self.assertRaises(iw.UsageError):
                    iw.parse_args(bad)

    def test_hello_and_reply_format(self):
        ck = cio.Ckpt(in_h=120, in_w=160, in_c=6, num_classes=4, params=[], adam_m=[], adam_v=[], text_dim=8,
                      aux_classes=2, prev_dim=4)
        self.assertEqual(iw.hello_line(ck).decode(),
                         '{"in_h":120,"in_w":160,"in_c":6,"num_classes":4,"text_dim":8,"prev_action_dim":4,"aux_classes":2}\n')
        line = iw.reply_line(1, [0.25, 0.5, 0.125, 0.125], 0.75).decode()
        self.assertEqual(line, '{"action":1,"probs":[0.25,0.5,0.125,0.125],"infer_ms":0.75}\n')


@unittest.skipUnless(HAVE_TORCH, "没有 torch，跳过子进程测试")
class TestProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.rng = np.random.RandomState(0)

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def ckpt(self, name, model):
        path = os.path.join(self.td.name, name + ".bin")
        write_ckpt(path, model)
        return path

    def frames(self, n):
        return [self.rng.randint(0, 256, size=(3, H, W)).astype(np.uint8) for _ in range(n)]

    def check_against_model(self, model, path, steps, text_ids=None, tol=2e-6):
        """steps：[(cur, prev_or_None, prev_action_or_None)]。逐步与同一模型在进程内的前向比较。"""
        from proto_client import InferClient, make_payload
        text = None
        if model.text_dim:
            ids = list(text_ids or []) + [0] * (cio.TEXT_LEN - len(text_ids or []))
            text = torch.tensor([ids], dtype=torch.uint8)
        with InferClient(LAUNCHER, path, text_ids=text_ids, env=env()) as cli:
            self.assertEqual(cli.hello, {"in_h": H, "in_w": W, "in_c": model.in_c, "num_classes": 4,
                                         "text_dim": model.text_dim, "prev_action_dim": model.prev_dim,
                                         "aux_classes": model.aux_classes})
            for cur, prv, pa in steps:
                r = cli.step(make_payload(cur, prv, pa))
                x = np.concatenate([cur, prv], axis=0) if prv is not None else cur
                with torch.no_grad():
                    z = model(torch.from_numpy(x[None]), text,
                              None if pa is None else torch.tensor([pa])).float().numpy()
                p = pm.softmax_probs(z)[0]
                self.assertEqual(list(r.keys()), ["action", "probs", "infer_ms"])
                self.assertLess(np.abs(np.array(r["probs"]) - p).max(), tol)
                self.assertEqual(r["action"], int(np.argmax(z[0])))
                self.assertAlmostEqual(sum(r["probs"]), 1.0, places=12)
                self.assertLess(max(r["probs"]), 0.999)      # 概率不饱和，下面的数值比较才有区分力
                self.assertGreater(r["infer_ms"], 0.0)
            rc, err = cli.close()
        self.assertEqual(rc, 0)
        self.assertIn(f"{len(steps)} 帧", err)

    def test_single_frame(self):
        m = make_model(1)
        self.check_against_model(m, self.ckpt("base", m), [(f, None, None) for f in self.frames(5)])

    def test_two_frame_order(self):
        m = make_model(2, in_c=6)
        path = self.ckpt("two", m)
        fr = self.frames(4)
        self.check_against_model(m, path, [(fr[i], fr[i + 1], None) for i in range(3)])
        # 当前帧与上一帧换位必须改变输出：证明顺序是（当前，上一帧）而不是对称处理
        from proto_client import InferClient, make_payload
        with InferClient(LAUNCHER, path, env=env()) as cli:
            a = cli.step(make_payload(fr[0], fr[1]))["probs"]
            b = cli.step(make_payload(fr[1], fr[0]))["probs"]
            cli.close()
        self.assertGreater(np.abs(np.array(a) - np.array(b)).max(), 1e-4)

    def test_prev_action(self):
        m = make_model(3, prev_dim=4)
        path = self.ckpt("prev", m)
        fr = self.frames(4)
        self.check_against_model(m, path, [(fr[i], None, i) for i in range(4)])

    def test_text_and_all(self):
        m = make_model(4, in_c=6, text_dim=8, prev_dim=4, aux_classes=3)
        path = self.ckpt("all", m)
        fr = self.frames(4)
        self.check_against_model(m, path, [(fr[i], fr[(i + 1) % 4], (i + 2) % 4) for i in range(3)], text_ids=[3, 7, 21, 1])
        self.check_against_model(m, path, [(fr[0], fr[1], 0)], text_ids=[])        # 无指令：全 pad，不报错

    def test_clean_eof_and_truncation(self):
        m = make_model(5)
        path = self.ckpt("eof", m)
        one = hwc(self.frames(1)[0])
        r = run(["--ckpt", path], b"")
        self.assertEqual((r.returncode, r.stdout.count(b"\n")), (0, 1))         # 只有 hello
        self.assertIn("0 帧", r.stderr.decode())
        r = run(["--ckpt", path], one + one[:100])
        self.assertEqual(r.returncode, 1)
        self.assertIn("截断", r.stderr.decode())
        self.assertEqual(r.stdout.count(b"\n"), 2)                              # hello + 第一步的回复
        self.assertTrue(all(json.loads(line) for line in r.stdout.splitlines()))

    def test_prev_action_byte_out_of_range(self):
        m = make_model(6, prev_dim=4)
        path = self.ckpt("prevbad", m)
        r = run(["--ckpt", path], hwc(self.frames(1)[0]) + bytes([4]))
        self.assertEqual(r.returncode, 1)
        self.assertIn("越界", r.stderr.decode())
        self.assertEqual(r.stdout.count(b"\n"), 1)

    def test_usage_errors_exit_2(self):
        m = make_model(7)
        path = self.ckpt("usage", m)
        for args in ([], ["--ckpt"], ["--bogus"], ["--ckpt", path, "--text", "a"], ["--ckpt", path, "--text", "22"],
                     ["--ckpt", path, "--text", "1,,2"], ["--ckpt", path, "--text", "-1"]):
            with self.subTest(args):
                r = run(args)
                self.assertEqual(r.returncode, 2)
                self.assertEqual(r.stdout, b"")
        self.assertEqual(run(["--ckpt", path, "--text", ""]).returncode, 0)

    def test_text_overflow_after_hello_only_for_text_models(self):
        ids = ",".join(str(i) for i in range(1, 11))        # 10 个 token
        mt = make_model(8, text_dim=4)
        r = run(["--ckpt", self.ckpt("txt", mt), "--text", ids], hwc(self.frames(1)[0]))
        self.assertEqual(r.returncode, 1)
        self.assertEqual(r.stdout.count(b"\n"), 1)           # hello 已发出，错误在第一帧之前
        self.assertIn("token 数超过", r.stderr.decode())
        mb = make_model(9)
        r = run(["--ckpt", self.ckpt("notxt", mb), "--text", ids], hwc(self.frames(1)[0]))
        self.assertEqual(r.returncode, 0)                    # 无文本分支：忽略 --text
        self.assertEqual(r.stdout.count(b"\n"), 2)

    def test_bad_checkpoints_fail_before_hello(self):
        m = make_model(10)
        path = self.ckpt("good", m)
        r = run(["--ckpt", os.path.join(self.td.name, "nope.bin")])
        self.assertEqual((r.returncode, r.stdout), (1, b""))
        with open(path, "rb") as f:
            raw = f.read()
        for name, data in (("trail", raw + b"\0"), ("short", raw[:-4]), ("magic", b"X" + raw[1:])):
            with self.subTest(name):
                bad = os.path.join(self.td.name, name + ".bin")
                with open(bad, "wb") as f:
                    f.write(data)
                r = run(["--ckpt", bad])
                self.assertEqual((r.returncode, r.stdout), (1, b""))
                self.assertIn("infer_wrap:", r.stderr.decode())

    def test_stdout_is_protocol_only(self):
        code = ("import os, sys; sys.path.insert(0, %r); import infer_wrap as iw; fd = iw.protect_stdout();"
                "print('库的打印'); os.write(1, b'raw\\n'); os.write(fd, b'proto\\n')") % HERE
        r = subprocess.run([sys.executable, "-c", code], capture_output=True)
        self.assertEqual(r.stdout, b"proto\n")
        self.assertIn("库的打印".encode(), r.stderr)
        self.assertIn(b"raw", r.stderr)


if __name__ == "__main__":
    unittest.main()
