#!/usr/bin/env python3
"""proto/train.py、proto/export.py 的测试。默认只用 CPU 与合成数据（约 1 分钟内）；
UAVRGB_PROTO_REAL=1 时加测数据盘上的真实数据与 C++ 参照 run（训练契约逐字节一致、前几步 loss 与参照 metrics.jsonl 对拍），
再加 UAVRGB_PROTO_DEVICE=cuda 时跑 1 个 epoch 并用主线 uavrgb_train --eval-only 读回导出的 checkpoint 核对混淆矩阵。
运行：nice -n 10 python -m unittest proto.test_train -v
"""
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

try:
    import torch
except ImportError:      # 没有 torch 的环境整份跳过
    torch = None

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
import ckpt_io as cio  # noqa: E402
import cpp_rng  # noqa: E402

if torch is not None:
    import data as dataio  # noqa: E402
    import export as exporter  # noqa: E402
    import model as pm  # noqa: E402
    import train as tr  # noqa: E402

D = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb")
REAL = os.environ.get("UAVRGB_PROTO_REAL") == "1"
DEVICE = os.environ.get("UAVRGB_PROTO_DEVICE", "cpu")
TRUNC = os.path.join(D, "tmp/agent6-distill/trunc/ds/dataset.uavr")
REF_RUN = os.path.join(D, "exp-student-select-20261109/e1-noise/seed0/run")
TRAIN_BIN = os.path.join(D, "tmp/build-check-main-1111/uavrgb_train")
F32 = np.float32


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def read_jsonl(path):
    return [json.loads(line) for line in read_text(path).splitlines() if line.strip()]


def make_synthetic(path, h=16, w=16, per_episode=24, seed=0):
    """合成 UAVR：类别决定整幅图的亮度（可分），标签不均衡（类 0 少），train/val/test 按 episode 分，无碰撞。"""
    rs = np.random.RandomState(seed)
    eps_spec = [("train", i + 1, f"g{i + 1}", per_episode) for i in range(6)] + \
               [("val", 10, "gv1", 16), ("val", 11, "gv2", 16), ("test", 20, "gt1", 16), ("test", 21, "gt2", 16)]
    n = sum(e[3] for e in eps_spec)
    labels = rs.choice(4, size=n, p=[0.05, 0.6, 0.2, 0.15]).astype(np.uint8)
    labels[:4] = [0, 1, 2, 3]
    imgs = np.zeros((n, 3, h, w), np.uint8)
    for i in range(n):
        imgs[i] = np.clip(40 * int(labels[i]) + 30 + rs.randint(0, 20, size=(3, h, w)), 0, 255).astype(np.uint8)
    eps, start = [], 0
    for k, (split, scene, gid, count) in enumerate(eps_spec):
        eps.append({"episode_id": k, "scene_id": scene, "geometry_gid": gid, "split": split,
                    "sample_start": start, "sample_count": count, "source_dir": f"/x/{k}"})
        start += count
    meta = {"schema": "uavrgb.dataset.v1", "split_policy": "geometry_gid", "split_scheme": "geometry_gid",
            "episodes": eps, "collision_flags": "0" * n}
    mb = json.dumps(meta, separators=(",", ":")).encode()
    lab_off = 48 + imgs.size
    hdr = b"UAVR" + struct.pack("<HHIHHBBBBQQQI", 1, 48, n, h, w, 3, 1, 1, 0, 48, lab_off, lab_off + n, len(mb))
    with open(path, "wb") as f:
        f.write(hdr + imgs.tobytes() + labels.tobytes() + mb)
    return n


@unittest.skipIf(torch is None, "没有 torch")
class TestKernels(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.rs = np.random.RandomState(1)

    def test_weighted_loss_and_grad_match_float64_formula(self):
        n, c = 64, 4
        logits64 = self.rs.randn(n, c) * 2.0
        y = self.rs.randint(0, c, size=n)
        w = np.array([36.6294632, 0.29793027, 3.21009398, 3.28200006], dtype=np.float32)
        loss, g = tr.weighted_loss_and_grad(torch.from_numpy(logits64.astype(np.float32)), torch.from_numpy(y), torch.from_numpy(w))
        lg = logits64.astype(np.float32).astype(np.float64)
        p = np.exp(lg - lg.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        wy = w.astype(np.float64)[y]
        ref_loss = float((wy * -np.log(p[np.arange(n), y])).sum() / wy.sum())
        onehot = np.eye(c)[y]
        ref_g = wy[:, None] * (p - onehot) / wy.sum()
        self.assertAlmostEqual(loss, ref_loss, places=6)
        self.assertEqual(g.dtype, torch.float32)
        np.testing.assert_allclose(g.numpy(), ref_g, rtol=2e-6, atol=1e-9)
        # 与 autograd 的梯度一致：dlogits 就是加权 loss 对 logits 的导数
        t = torch.from_numpy(lg).requires_grad_(True)
        lw = (torch.from_numpy(wy) * torch.nn.functional.cross_entropy(t, torch.from_numpy(y), reduction="none")).sum() / float(wy.sum())
        lw.backward()
        np.testing.assert_allclose(g.numpy(), t.grad.numpy(), rtol=2e-5, atol=1e-8)

    def test_plain_ce_matches_torch(self):
        logits = torch.from_numpy((self.rs.randn(50, 4) * 3).astype(np.float32))
        y = torch.from_numpy(self.rs.randint(0, 4, size=50))
        ref = float(torch.nn.functional.cross_entropy(logits.double(), y))
        self.assertAlmostEqual(tr.plain_ce(logits, y), ref, places=6)

    def test_softmax_parts_sum_is_sequential_float32(self):
        logits = torch.from_numpy((self.rs.randn(100, 4) * 5).astype(np.float32))
        z, e, s = tr.softmax_parts(logits)
        ez = np.exp(z.numpy())
        ref = ((ez[:, 0] + ez[:, 1]) + ez[:, 2]) + ez[:, 3]
        np.testing.assert_allclose(s.numpy(), ref, rtol=1e-6)
        self.assertEqual(s.dtype, torch.float32)

    def test_count_correct_first_max_on_ties(self):
        logits = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 2.0, 2.0, 2.0], [0.0, 0.0, 0.0, 3.0]])
        y = torch.tensor([0, 1, 3])
        self.assertEqual(tr.count_correct(logits, y), 3)      # 平局取小下标：第 1 行预测 0，第 2 行预测 1
        self.assertEqual(tr.count_correct(logits, torch.tensor([1, 2, 3])), 1)

    def test_clip_global_norm(self):
        g = torch.from_numpy(self.rs.randn(1000).astype(np.float32) * 3)
        norm = math.sqrt(float((g.double() ** 2).sum()))
        g0 = g.clone()
        got = tr.clip_global_norm(g, 5.0)
        self.assertAlmostEqual(got, norm, places=9)
        coef = F32(5.0) / (F32(norm) + F32(1e-6))
        self.assertLess(coef, 1.0)
        np.testing.assert_array_equal(g.numpy(), (g0.numpy() * coef).astype(np.float32))
        small = torch.from_numpy(self.rs.randn(1000).astype(np.float32) * 0.01)
        s0 = small.clone()
        tr.clip_global_norm(small, 5.0)
        np.testing.assert_array_equal(small.numpy(), s0.numpy())      # 范数 < 5：不缩放
        with self.assertRaises(RuntimeError):
            tr.clip_global_norm(torch.tensor([1.0, float("nan")]), 5.0)

    def numpy_adam_reference(self, p, grads, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        """nn_cuda.cu adam_step_kernel 的逐元素 float32 写法（无 FMA）。"""
        b1, b2, eps, lr = F32(b1), F32(b2), F32(eps), F32(lr)
        m = np.zeros_like(p)
        v = np.zeros_like(p)
        p = p.copy()
        for t, g in enumerate(grads, start=1):
            bc1 = F32(1.0 - math.pow(float(b1), t))
            bc2 = F32(1.0 - math.pow(float(b2), t))
            mi = b1 * m + (F32(1.0) - b1) * g
            vi = b2 * v + (F32(1.0) - b2) * g * g
            m, v = mi.astype(np.float32), vi.astype(np.float32)
            p = (p - lr * ((m / bc1) / (np.sqrt(v / bc2) + eps))).astype(np.float32)
        return p, m, v

    def test_adam_matches_float32_reference(self):
        p0 = self.rs.randn(5000).astype(np.float32) * 0.1
        grads = [(self.rs.randn(5000) * 10.0 ** self.rs.uniform(-8, 0, size=5000)).astype(np.float32) for _ in range(8)]
        ref_p, ref_m, ref_v = self.numpy_adam_reference(p0, grads)
        flat = torch.from_numpy(p0.copy())
        opt = tr.FlatAdam(flat)
        for g in grads:
            opt.step(torch.from_numpy(g.copy()))
        self.assertEqual(opt.t, 8)
        # 同样的逐元素 float32 步骤，CPU 上应当逐位相同；留 1 ulp 余量给不同的向量化实现
        np.testing.assert_array_max_ulp(flat.numpy(), ref_p, maxulp=1)
        np.testing.assert_array_max_ulp(opt.m.numpy(), ref_m, maxulp=1)
        np.testing.assert_array_max_ulp(opt.v.numpy(), ref_v, maxulp=1)

    def test_adam_bias_corrections_and_first_step(self):
        opt = tr.FlatAdam(torch.zeros(3))
        bc1, bc2 = opt.bias_corrections(1)
        self.assertEqual(bc1, F32(1.0 - float(F32(0.9))))
        self.assertEqual(bc2, F32(1.0 - float(F32(0.999))))
        # 第一步：p 的变化 ≈ lr·g/(|g|+eps)·(bias 修正后恒为 ±1)，g 远大于 eps 时就是 −lr·sign(g)
        flat = torch.zeros(4)
        opt = tr.FlatAdam(flat)
        opt.step(torch.tensor([1.0, -2.0, 3.0, -0.5]))
        np.testing.assert_allclose(flat.numpy(), -0.001 * np.array([1, -1, 1, -1]), rtol=1e-5)

    def test_flatten_params_shares_storage(self):
        m = pm.ProtoCnn()
        flat, params = tr.flatten_params(m)
        self.assertEqual(flat.numel(), 100388)
        self.assertEqual([tuple(p.shape) for p in params], [tuple(p.shape) for p in m.cpp_params()])
        before = m.c1.weight.detach().clone()
        flat[0] += 1.0
        self.assertAlmostEqual(float(m.c1.weight.detach().reshape(-1)[0] - before.reshape(-1)[0]), 1.0, places=6)
        flat.sub_(1.0)
        # 反向传播产生的梯度形状与参数一致，且参数仍是叶子
        x = torch.randint(0, 255, (2, 3, 120, 160), dtype=torch.uint8)
        grads = torch.autograd.grad(m(x), params, grad_outputs=torch.ones(2, 4))
        self.assertEqual([tuple(g.shape) for g in grads], [tuple(p.shape) for p in params])


@unittest.skipIf(torch is None, "没有 torch")
class TestInitAndShuffle(unittest.TestCase):
    def test_init_uses_cpp_xavier_stream_in_memory_order(self):
        m = pm.ProtoCnn(in_h=16, in_w=16)
        m.init_xavier(cpp_rng.CppXavier(5))
        rng = cpp_rng.CppMt19937(5)
        for w, fan_in, fan_out in m.xavier_plan():
            lim = F32(np.sqrt(F32(6.0) / F32(fan_in + fan_out)))
            expect = rng.uniform_float(-lim, lim, w.numel())
            np.testing.assert_array_equal(w.detach().numpy().reshape(-1), expect)
        for b in (m.c1.bias, m.c2.bias, m.c3.bias, m.fc.bias):
            self.assertEqual(float(b.detach().abs().max()), 0.0)


@unittest.skipIf(torch is None, "没有 torch")
class TestExport(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp(prefix="proto_export_", dir=os.environ.get("TMPDIR") or None)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def make_state(self, epoch=3, seed=1):
        m = pm.ProtoCnn(in_h=16, in_w=16)
        m.init_xavier(cpp_rng.CppXavier(seed))
        rs = np.random.RandomState(seed)
        params = [p.detach().numpy().copy() for p in m.cpp_params()]
        mm = [rs.randn(*a.shape).astype(np.float32) * 0.01 for a in params]
        vv = [np.abs(rs.randn(*a.shape)).astype(np.float32) * 0.01 for a in params]
        contract = cio.make_contract(hashlib.sha256(b"x").hexdigest(), True, F32(0.001), seed, 128, hashlib.sha256(b"geometry_gid").digest())
        return dict(structure=m.structure(), params=params, adam_m=mm, adam_v=vv, adam_t=24, epoch=epoch,
                    class_weights=np.array([36.6, 0.3, 3.2, 3.3], np.float32), best_macro_f1=0.4257, contract=contract)

    def test_pt_roundtrip_and_cpp_layout(self):
        st = self.make_state()
        pt = os.path.join(self.td, "a.pt")
        exporter.save_pt(pt, **st)
        out = os.path.join(self.td, "a.bin")
        self.assertEqual(exporter.export_pt(pt, out), "written")
        ck = cio.read_checkpoint(out)
        self.assertEqual((ck.epoch, ck.adam_t, ck.in_h, ck.in_w, ck.in_c), (3, 24, 16, 16, 3))
        self.assertAlmostEqual(ck.best_macro_f1, 0.4257, places=6)
        self.assertEqual(ck.contract, st["contract"])
        np.testing.assert_array_equal(ck.class_weights, st["class_weights"])
        for a, b in zip(ck.params, st["params"]):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(ck.adam_m, st["adam_m"]):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(ck.adam_v, st["adam_v"]):
            np.testing.assert_array_equal(a, b)
        # 主线工具的读端（tools/average_ckpt.py，只依赖 numpy）接受导出的文件
        import average_ckpt as avg
        mine = avg.read_ckpt(out)
        self.assertEqual((mine["epoch"], mine["in_h"], mine["param_count"]), (3, 16, 8))
        self.assertEqual(mine["contract"], st["contract"])

    def test_export_run_names_and_idempotence(self):
        run = os.path.join(self.td, "run")
        os.makedirs(os.path.join(run, "pt"))
        for name, ep in (("ep01.pt", 1), ("ep02.pt", 2), ("last.pt", 2), ("best.pt", 1)):
            exporter.save_pt(os.path.join(run, "pt", name), **self.make_state(epoch=ep))
        res = exporter.export_run(run)
        self.assertEqual(sorted(d for _, d, _ in res), ["ckpt.bin", "ckpt_best.bin", "ckpt_ep01.bin", "ckpt_ep02.bin"])
        self.assertTrue(all(r == "written" for _, _, r in res))
        self.assertEqual(cio.read_checkpoint(os.path.join(run, "ckpt_best.bin")).epoch, 1)
        self.assertEqual(cio.read_checkpoint(os.path.join(run, "ckpt.bin")).epoch, 2)
        self.assertTrue(all(r == "same" for _, _, r in exporter.export_run(run)))
        # 目标已存在且内容不同：报错，--overwrite 才覆盖
        exporter.save_pt(os.path.join(run, "pt", "ep01.pt"), **self.make_state(epoch=1, seed=9))
        with self.assertRaises(cio.CkptError):
            exporter.export_run(run)
        self.assertIn("written", [r for _, _, r in exporter.export_run(run, overwrite=True)])

    def test_rejects_non_finite_and_foreign_files(self):
        st = self.make_state()
        st["params"][0][0, 0, 0, 0] = np.nan
        pt = os.path.join(self.td, "bad.pt")
        exporter.save_pt(pt, **st)
        with self.assertRaises(cio.CkptError):
            exporter.export_pt(pt, os.path.join(self.td, "bad.bin"))
        self.assertFalse(os.path.exists(os.path.join(self.td, "bad.bin")))
        torch.save({"format": "other"}, os.path.join(self.td, "foreign.pt"))
        with self.assertRaises(cio.CkptError):
            exporter.load_pt(os.path.join(self.td, "foreign.pt"))


@unittest.skipIf(torch is None, "没有 torch")
class TestTrainerSynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.mkdtemp(prefix="proto_train_", dir=os.environ.get("TMPDIR") or None)
        cls.path = os.path.join(cls.td, "ds.uavr")
        make_synthetic(cls.path)
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.td, ignore_errors=True)

    def train(self, name, seed=3, epochs=3):
        ds = dataio.Uavr(self.path, drop_collision=True)
        run = os.path.join(self.td, name)
        os.makedirs(run)
        t = tr.Trainer(ds, seed, "cpu", epochs=epochs, batch=32, run_dir=run)
        res = t.run()
        return ds, t, run, res

    def test_artifacts_and_contents(self):
        ds, t, run, res = self.train("a")
        names = set(os.listdir(run))
        for n in ("config.json", "metrics.jsonl", "results.csv", "test_eval.jsonl", "ckpt.bin", "ckpt_best.bin",
                  "ckpt_ep01.bin", "ckpt_ep02.bin", "ckpt_ep03.bin"):
            self.assertIn(n, names)
        events = read_jsonl(os.path.join(run, "metrics.jsonl"))
        steps = [e for e in events if e["event"] == "train_step"]
        per_epoch = math.ceil(len(ds.splits["train"]) / 32)          # 144 训练样本、批 32：末批 16，不丢
        self.assertEqual(len(steps), 3 * per_epoch)
        self.assertEqual([e["step"] for e in steps], list(range(1, 3 * per_epoch + 1)))
        self.assertEqual(steps[0]["total_steps"], 3 * per_epoch)
        self.assertEqual(len([e for e in events if e["event"] == "validation" and e["split"] == "val"]), 3)
        self.assertEqual(len([e for e in events if e["event"] == "validation" and e["split"] == "test"]), 1)
        rows = read_text(os.path.join(run, "results.csv")).strip().split("\n")
        self.assertEqual(rows[0], "epoch,train_ce,dev_ce,dev_accuracy")
        self.assertEqual([r.split(",")[0] for r in rows[1:]], ["1", "2", "3"])
        # 训练在可分的合成数据上有效：训练 CE 逐轮下降
        ce = [float(r.split(",")[1]) for r in rows[1:]]
        self.assertLess(ce[-1], ce[0])
        # checkpoint：epoch、Adam 步数、契约字段、末轮与内存里的权重一致
        last = cio.read_checkpoint(os.path.join(run, "ckpt.bin"))
        self.assertEqual((last.epoch, last.adam_t), (3, 3 * per_epoch))
        self.assertEqual(read_bytes(os.path.join(run, "ckpt_ep03.bin")), read_bytes(os.path.join(run, "ckpt.bin")))
        c = cio.parse_contract(last.contract)
        self.assertEqual((c["dataset_sha256"], c["seed"], c["batch"], c["drop_collision"], c["lr"]),
                         (ds.sha256, 3, 32, True, float(F32(0.001))))
        self.assertEqual(c["split_scheme_sha256"], hashlib.sha256(b"geometry_gid").hexdigest())
        for a, b in zip(last.params, tr.split_flat(t.flat, t.params)):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(last.class_weights, ds.auto_class_weights())
        # best 是 val 宏 F1 最高的一轮，严格大于才更新（并列取先出现的）
        val = [e for e in events if e["event"] == "validation" and e["split"] == "val"]
        f1 = [e["macro_f1"] for e in val]
        # 并列时 C++ 的比较是 double 对 float(best)，取舍随舍入方向而定：只断言 best 落在最大值的并列集合里
        candidates = [k + 1 for k, v in enumerate(f1) if v >= max(f1) - 1e-7]
        self.assertIn(cio.read_checkpoint(os.path.join(run, "ckpt_best.bin")).epoch, candidates)
        # test 用的是交付的 ckpt_best.bin
        te = read_jsonl(os.path.join(run, "test_eval.jsonl"))[-1]
        self.assertTrue(te["ckpt"].endswith("ckpt_best.bin"))
        self.assertEqual(te["samples"], len(ds.splits["test"]))
        # 主线的 average_ckpt 能读这个 run 目录的窗口（末 2 轮）
        import average_ckpt as avg
        files, cks = avg.load_window(run, 2)
        self.assertEqual([e for e, _ in files], [2, 3])

    def test_same_seed_is_reproducible_and_seeds_differ(self):
        _, _, ra, _ = self.train("r1", seed=4, epochs=2)
        _, _, rb, _ = self.train("r2", seed=4, epochs=2)
        _, _, rc, _ = self.train("r3", seed=5, epochs=2)
        read = lambda d: read_bytes(os.path.join(d, "ckpt_ep02.bin"))  # noqa: E731
        self.assertEqual(read(ra), read(rb))        # CPU 上同种子逐字节相同
        self.assertNotEqual(read(ra), read(rc))

    def test_shuffle_state_persists_across_epochs(self):
        ds = dataio.Uavr(self.path, drop_collision=True)
        run = os.path.join(self.td, "sh")
        os.makedirs(run)
        t = tr.Trainer(ds, 7, "cpu", epochs=2, batch=32, run_dir=run)
        seen = []
        for _ in range(2):
            seen.append(torch.cat(list(t.epoch_batches())).cpu().numpy())
        rng = cpp_rng.CppMt19937(7)
        order = ds.splits["train"].astype(np.int64).copy()
        for k in range(2):
            rng.shuffle(order)
            np.testing.assert_array_equal(seen[k], order)
        self.assertFalse(np.array_equal(seen[0], seen[1]))

    def test_refuses_nonfinite_loss(self):
        ds = dataio.Uavr(self.path, drop_collision=True)
        run = os.path.join(self.td, "nan")
        os.makedirs(run)
        t = tr.Trainer(ds, 1, "cpu", epochs=1, batch=32, run_dir=run)
        with torch.no_grad():
            t.flat[:] = float("inf")
        with self.assertRaises(RuntimeError):
            t.train_step(torch.arange(32))


@unittest.skipUnless(torch is not None and REAL and os.path.isfile(TRUNC) and os.path.isdir(REF_RUN), "UAVRGB_PROTO_REAL=1 且数据盘上有 trunc 数据集与参照 run 时才跑")
class TestRealParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.ds = dataio.Uavr(TRUNC, drop_collision=True)
        cls.ref_events = read_jsonl(os.path.join(REF_RUN, "metrics.jsonl"))
        cls.ref_config = json.loads(read_text(os.path.join(REF_RUN, "config.json")))

    def test_class_weights_and_counts_match_reference_config(self):
        w = self.ds.auto_class_weights()
        self.assertEqual([float("%.9g" % float(x)) for x in w], self.ref_config["class_weights"])
        self.assertEqual((len(self.ds.splits["train"]), len(self.ds.splits["val"]), len(self.ds.splits["test"])), (8205, 574, 884))

    def test_contract_bytes_equal_reference_checkpoint(self):
        ref = cio.read_checkpoint(os.path.join(REF_RUN, "ckpt_ep30.bin"))
        scheme = hashlib.sha256(self.ds.split_scheme.encode()).digest()
        mine = cio.make_contract(self.ds.sha256, True, F32(0.001), 0, 128, scheme, None, 0.5)
        self.assertEqual(mine, ref.contract)
        self.assertEqual(self.ds.sha256, self.ref_config["dataset_sha256"])

    def test_first_steps_match_reference_train_step_losses(self):
        ref = [e for e in self.ref_events if e["event"] == "train_step"][:3]
        t = tr.Trainer(self.ds, 0, DEVICE, epochs=1, batch=128, run_dir=None)
        got = []
        for idx in t.epoch_batches():
            got.append(t.train_step(idx))
            if len(got) == 3:
                break
        for r, (loss, acc, _, _) in zip(ref, got):
            self.assertAlmostEqual(loss, r["loss"], delta=2e-5, msg=f"step {r['step']}")
            self.assertEqual(acc, r["action_accuracy"], f"step {r['step']}")

    @unittest.skipUnless(DEVICE == "cuda" and os.path.isfile(TRAIN_BIN), "需要 UAVRGB_PROTO_DEVICE=cuda 与 uavrgb_train")
    def test_one_epoch_then_cpp_eval_only_reads_exported_checkpoint(self):
        td = tempfile.mkdtemp(prefix="proto_real_", dir=os.environ.get("TMPDIR") or None)
        try:
            run = os.path.join(td, "run")
            os.makedirs(run)
            t = tr.Trainer(self.ds, 0, "cuda", epochs=1, batch=128, run_dir=run)
            res = t.run()
            out = os.path.join(td, "evalonly")
            subprocess.run([TRAIN_BIN, "--data", TRUNC, "--eval-only", os.path.join(run, "ckpt_best.bin"), "--out", out,
                            "--split", "all", "--drop-collision"], check=True, capture_output=True, text=True)
            events = read_jsonl(os.path.join(out, "metrics.jsonl"))
            cpp = {e["split"]: e for e in events if e["event"] == "validation"}
            mine = {e["split"]: e for e in read_jsonl(os.path.join(run, "metrics.jsonl")) if e["event"] == "validation"}
            for split in ("val", "test"):
                self.assertEqual(cpp[split]["confusion"], mine[split]["confusion"], split)
                self.assertAlmostEqual(cpp[split]["loss"], mine[split]["loss"], delta=1e-5)
            self.assertEqual(res["steps"], 65)
        finally:
            shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
