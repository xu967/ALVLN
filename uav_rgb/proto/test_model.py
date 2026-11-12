#!/usr/bin/env python3
"""proto/model.py 的测试（需要 torch；没有 torch 时整体跳过，check.sh 的 .venv-tools 解释器没有 torch）。

  $D/teacher-venv/bin/python proto/test_model.py

对照物是一份独立的 numpy 前向（double 精度、显式循环语义）：直接按 src/model.cpp 的文字描述写
（卷积 = 互相关、pad 1，池化 2×2，FiLM y'=(1+γ)y+β，展平顺序 [c][h][w]），不经过 torch，
用来防止 PyTorch 的布局假设与 C++ 不一致。与真实 uavrgb_infer 的对拍在 test_forward_parity.py。
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ckpt_io as cio  # noqa: E402

try:
    import torch
    import model as pm
except ImportError:   # pragma: no cover
    torch = None

STRUCTURES = {
    "base": dict(in_h=16, in_w=24, in_c=3, num_classes=4),
    "two_frame": dict(in_h=16, in_w=24, in_c=6, num_classes=4),
    "prev": dict(in_h=16, in_w=24, in_c=3, num_classes=4, prev_dim=4),
    "text": dict(in_h=16, in_w=24, in_c=3, num_classes=4, text_dim=5),
    "aux": dict(in_h=16, in_w=24, in_c=3, num_classes=4, aux_classes=8),
    "all": dict(in_h=16, in_w=24, in_c=6, num_classes=4, text_dim=3, prev_dim=4, aux_classes=8),
}


def np_conv(x, w, b, pad):
    n, c, h, wd = x.shape
    k = w.shape[2]
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out = np.zeros((n, w.shape[0], h, wd))
    for ky in range(k):
        for kx in range(k):
            out += np.einsum("nchw,oc->nohw", xp[:, :, ky:ky + h, kx:kx + wd], w[:, :, ky, kx])
    return out + b[None, :, None, None]


def np_pool(x):
    n, c, h, w = x.shape
    return x.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))


def np_forward(p, st, x, text=None, prev=None):
    """p：按 cnn_params 顺序的 float64 数组列表。返回 (logits, aux_logits 或 None)。"""
    i = 0
    y = x
    for _ in range(3):
        y = np_pool(np.maximum(np_conv(y, p[i], p[i + 1], 1), 0.0))
        i += 2
    fcw, fcb = p[6], p[7]
    i = 8
    td, pd, aux = st.get("text_dim", 0), st.get("prev_dim", 0), st.get("aux_classes", 0)
    emb = None
    if td:
        emb, i = p[i], i + 1
    n = x.shape[0]
    if td or pd:
        cond = np.zeros((n, cio.TEXT_LEN * td + pd))
        for s in range(n):
            for l in range(cio.TEXT_LEN if td else 0):
                tid = int(text[s, l])
                if tid != 0 and tid < cio.TEXT_VOCAB:
                    cond[s, l * td:(l + 1) * td] = emb[tid]
            if pd:
                cond[s, cio.TEXT_LEN * td + int(prev[s])] = 1.0
        gb = cond @ p[i].T + p[i + 1]
        i += 2
        gamma, beta = gb[:, :64], gb[:, 64:]
        y = (1.0 + gamma)[:, :, None, None] * y + beta[:, :, None, None]
    logits = y.reshape(n, -1) @ fcw.T + fcb
    auxl = np_conv(y, p[i], p[i + 1], 0) if aux else None
    return logits, auxl


def random_inputs(st, n=3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 256, size=(n, st["in_c"], st["in_h"], st["in_w"]), dtype=np.uint8)
    text = rng.integers(0, cio.TEXT_VOCAB + 3, size=(n, cio.TEXT_LEN)).astype(np.uint8) if st.get("text_dim") else None
    prev = rng.integers(0, st.get("prev_dim", 1) or 1, size=n).astype(np.int64) if st.get("prev_dim") else None
    return x, text, prev


def random_model(st, seed=0):
    g = torch.Generator().manual_seed(seed)
    m = pm.ProtoCnn(**st)
    with torch.no_grad():
        for p in m.cpp_params():
            p.copy_(torch.randn(p.shape, generator=g) * (0.3 if p.ndim > 1 else 0.1))
    return m.eval()


@unittest.skipIf(torch is None, "没有 torch")
class TestModel(unittest.TestCase):
    def test_param_count_and_order(self):
        m = pm.ProtoCnn(120, 160, 3, 4)
        self.assertEqual(sum(p.numel() for p in m.cpp_params()), 100388)
        for name, st in STRUCTURES.items():
            with self.subTest(name):
                shapes = [tuple(p.shape) for p in pm.ProtoCnn(**st).cpp_params()]
                self.assertEqual(shapes, cio.expected_shapes(**st))

    def test_matches_numpy_reference(self):
        for name, st in STRUCTURES.items():
            with self.subTest(name):
                m = random_model(st, seed=len(name))
                x, text, prev = random_inputs(st)
                with torch.no_grad():
                    t_text = None if text is None else torch.from_numpy(text)
                    t_prev = None if prev is None else torch.from_numpy(prev)
                    out = m(torch.from_numpy(x), t_text, t_prev, want_aux=bool(st.get("aux_classes")))
                logits, aux = out if st.get("aux_classes") else (out, None)
                p64 = [p.detach().numpy().astype(np.float64) for p in m.cpp_params()]
                ref, ref_aux = np_forward(p64, st, x.astype(np.float64) / 255.0, text, prev)
                np.testing.assert_allclose(logits.numpy(), ref, rtol=2e-4, atol=2e-4)
                if ref_aux is not None:
                    np.testing.assert_allclose(aux.numpy(), ref_aux, rtol=2e-4, atol=2e-4)

    def test_text_pad_and_unknown_ids_act_as_zero(self):
        st = STRUCTURES["text"]
        m = random_model(st, seed=3)
        x, _, _ = random_inputs(st, n=1)
        ids = np.array([[3, 0, 7, 0, 0, 0, 0, 0, 0]], np.uint8)
        alt = np.array([[3, 22, 7, 200, 255, 22, 0, 22, 99]], np.uint8)   # 0 与 ≥22 都按 pad：整段为 0
        with torch.no_grad():
            a = m(torch.from_numpy(x), torch.from_numpy(ids))
            b = m(torch.from_numpy(x), torch.from_numpy(alt))
            moved = m(torch.from_numpy(x), torch.from_numpy(np.array([[7, 0, 3, 0, 0, 0, 0, 0, 0]], np.uint8)))
        np.testing.assert_array_equal(a.numpy(), b.numpy())
        self.assertGreater(float((a - moved).abs().max()), 1e-6)   # 位置是有意义的：交换 token 位置结果不同

    def test_uint8_is_divided_by_255(self):
        st = STRUCTURES["base"]
        m = random_model(st)
        x, _, _ = random_inputs(st)
        with torch.no_grad():
            a = m(torch.from_numpy(x))
            b = m(torch.from_numpy(x).float() / 255.0)
        np.testing.assert_array_equal(a.numpy(), b.numpy())

    def test_input_validation(self):
        base = random_model(STRUCTURES["base"])
        x, _, _ = random_inputs(STRUCTURES["base"])
        xt = torch.from_numpy(x)
        with self.assertRaises(ValueError):
            base(xt, text=torch.zeros(3, 9, dtype=torch.uint8))
        with self.assertRaises(ValueError):
            base(xt, prev=torch.zeros(3, dtype=torch.int64))
        with self.assertRaises(ValueError):
            base(xt[:, :, :8])
        with self.assertRaises(ValueError):
            base(xt, want_aux=True)
        prev_m = random_model(STRUCTURES["prev"])
        with self.assertRaises(ValueError):
            prev_m(xt)
        with self.assertRaises(ValueError):
            prev_m(xt, prev=torch.tensor([0, 1, 4]))   # 越界
        text_m = random_model(STRUCTURES["text"])
        with self.assertRaises(ValueError):
            text_m(xt)

    def test_checkpoint_roundtrip_keeps_logits(self):
        import tempfile
        for name, st in STRUCTURES.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as td:
                m = random_model(st, seed=11)
                path = os.path.join(td, "m.bin")
                cio.write_checkpoint(path, pm.ckpt_from_model(m, epoch=2))
                back = pm.model_from_ckpt(cio.read_checkpoint(path))
                x, text, prev = random_inputs(st)
                args = (torch.from_numpy(x), None if text is None else torch.from_numpy(text),
                        None if prev is None else torch.from_numpy(prev))
                with torch.no_grad():
                    np.testing.assert_array_equal(m(*args).numpy(), back(*args).numpy())

    def test_xavier_plan_order_and_limits(self):
        st = dict(in_h=120, in_w=160, in_c=6, num_classes=4, text_dim=3, prev_dim=4, aux_classes=8)
        m = pm.ProtoCnn(**st)
        calls = []

        def fake(numel, lim):
            calls.append((numel, float(lim)))
            return np.full(numel, 0.5 * lim, np.float32)

        m.init_xavier(fake)
        fan = [(6 * 9, 16 * 9), (16 * 9, 32 * 9), (32 * 9, 64 * 9), (19200, 4), (22, 3), (9 * 3 + 4, 128), (64, 8)]
        want = [(p.numel(), float(np.float32(np.sqrt(np.float32(6.0) / np.float32(a + b)))))
                for p, (a, b) in zip([m.c1.weight, m.c2.weight, m.c3.weight, m.fc.weight, m.text_emb,
                                      m.film.weight, m.aux.weight], fan)]
        self.assertEqual(calls, want)
        for p in m.cpp_params():
            if p.ndim == 1:
                self.assertEqual(float(p.detach().abs().max()), 0.0)   # 偏置清零

    def test_softmax_and_argmax(self):
        lg = np.array([[1.0, 2.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0], [5.0, -1.0, 4.0, 3.0]], np.float32)
        pr = pm.softmax_probs(lg)
        np.testing.assert_allclose(pr.sum(axis=1), 1.0, rtol=1e-12)
        self.assertEqual(pm.first_argmax(lg).tolist(), [1, 0, 0])   # 平局取小下标，同 C++ 的 logits[k] > logits[best]


if __name__ == "__main__":
    unittest.main()
