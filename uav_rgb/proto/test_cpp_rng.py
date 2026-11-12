#!/usr/bin/env python3
"""cpp_rng 的对拍：Python 复刻的随机数流与 libstdc++ 的 std::mt19937 / uniform_real_distribution<float> / std::shuffle 逐位相同。

常量测试不需要编译器。对拍测试编译 proto/cpp/rng_dump.cpp（g++，输出到 TMPDIR），没有 g++ 时跳过。
运行：nice -n 10 python -m unittest proto.test_cpp_rng -v      （不需要 torch、GPU）
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cpp_rng  # noqa: E402

GXX = shutil.which("g++")


class KnownConstants(unittest.TestCase):
    def test_mt19937_known_outputs(self):
        # C++ 标准给的第 10000 个输出是 4123659995（默认种子 5489）；前几个值是公开的常量
        self.assertEqual(cpp_rng.CppMt19937(5489).take(3).tolist(), [3499211612, 581869302, 3890346734])
        self.assertEqual(cpp_rng.CppMt19937(0).take(3).tolist(), [2357136044, 2546248239, 3071714933])
        self.assertEqual(int(cpp_rng.CppMt19937(5489).take(10000)[-1]), 4123659995)

    def test_buffering_does_not_change_stream(self):
        a = cpp_rng.CppMt19937(7, block=3)
        b = cpp_rng.CppMt19937(7, block=1 << 14)
        xs = [a.next() for _ in range(50)]
        self.assertEqual(xs, b.take(50).tolist())
        # 混用 next 与 take 之后仍接着同一条流
        c = cpp_rng.CppMt19937(7, block=5)
        ys = [c.next(), c.next()] + c.take(10).tolist() + [c.next()]
        self.assertEqual(ys, cpp_rng.CppMt19937(7).take(13).tolist())

    def test_uniform_float_range_and_edge(self):
        v = cpp_rng.CppMt19937(1).uniform_float(-np.float32(0.25), np.float32(0.25), 20000)
        self.assertEqual(v.dtype, np.float32)
        self.assertTrue((v >= -0.25).all() and (v <= 0.25).all())
        # u 接近 2^32 时 float(u) 会舍入到 2^32，必须按 nextafter(1,0) 处理，不能得到 1.0
        class Fixed(cpp_rng.CppMt19937):
            def take(self, n):
                return np.full(n, 4294967295, dtype=np.uint32)
        top = Fixed(0).uniform_float(np.float32(0.0), np.float32(1.0), 2)
        self.assertTrue((top < 1.0).all())
        self.assertEqual(float(top[0]), float(np.nextafter(np.float32(1.0), np.float32(0.0))))

    def test_shuffle_is_permutation(self):
        for n in (2, 3, 7, 1000, 8205):
            order = np.arange(n, dtype=np.uint32)
            cpp_rng.CppMt19937(3).shuffle(order)
            self.assertEqual(sorted(order.tolist()), list(range(n)))


@unittest.skipUnless(GXX, "没有 g++，跳过与 libstdc++ 的对拍")
class AgainstLibstdcxx(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = os.environ.get("TMPDIR") or tempfile.gettempdir()
        cls.dir = tempfile.mkdtemp(prefix="rng_dump_", dir=tmp)
        cls.exe = os.path.join(cls.dir, "rng_dump")
        subprocess.run([GXX, "-O3", "-std=c++17", "-o", cls.exe, os.path.join(HERE, "cpp", "rng_dump.cpp")],
                       check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def run_dump(self, *args):
        return subprocess.run([self.exe, *map(str, args)], check=True, capture_output=True, text=True).stdout

    def test_raw_stream(self):
        for seed in (0, 1, 2, 12345, 4294967295):
            c = np.array(self.run_dump("raw", seed, 5000).split(), dtype=np.uint64)
            p = cpp_rng.CppMt19937(seed).take(5000).astype(np.uint64)
            self.assertTrue((c == p).all(), f"seed {seed}")

    def test_uniform_float_bits(self):
        for seed, lim in ((0, 0.1), (1, 0.0123), (2, 0.5), (7, 1.0 / 3.0), (0, 0.0003)):
            lim32 = np.float32(lim)
            n = 20000
            c = np.array([int(x, 16) for x in self.run_dump("uniform", seed, repr(float(lim32)), n).split()], dtype=np.uint32)
            p = cpp_rng.CppMt19937(seed).uniform_float(-lim32, lim32, n).view(np.uint32)
            self.assertTrue((c == p).all(), f"seed {seed} lim {lim}: 不同位 {int((c != p).sum())}")

    def check_shuffle(self, n, epochs, seed=42):
        lines = self.run_dump("shuffle", seed, n, epochs).strip().split("\n")
        self.assertEqual(len(lines), epochs)
        rng = cpp_rng.CppMt19937(seed)
        order = np.arange(n, dtype=np.uint32)
        for e in range(epochs):
            rng.shuffle(order)          # 同一个引擎连续打乱，同 trainer 里每个 epoch 就地再打乱一次
            c = np.array(lines[e].split(), dtype=np.uint32)
            self.assertTrue((c == order).all(), f"N={n} 第 {e + 1} 次打乱不同")

    def test_shuffle_fast_path_odd_and_even(self):
        # 8205 是 e1-noise 的训练集大小（奇数，走两个位置一次抽的快路径）；偶数长度多一次单独的交换
        self.check_shuffle(8205, 6)
        self.check_shuffle(8204, 4)
        self.check_shuffle(1000, 3)
        for n in (2, 3, 7):
            self.check_shuffle(n, 5)

    def test_shuffle_slow_path_large_n(self):
        # (2^32-1)/N < N 即 N > 65535 时逐元素抽；边界两侧各测一个
        for n in (65535, 65536, 70000):
            self.check_shuffle(n, 2)


if __name__ == "__main__":
    unittest.main()
