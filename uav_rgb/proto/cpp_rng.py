#!/usr/bin/env python3
"""C++ 随机数流的 Python 复刻：让 PyTorch 训练与 uavrgb_train 用同一个初值、同一个逐轮打乱顺序。

复刻的是 libstdc++（GCC 13，主线 uavrgb_train 的编译器）的三个调用，对拍程序 proto/cpp/rng_dump.cpp 与
proto/test_cpp_rng.py 逐项核对过：
  std::mt19937(seed)                           numpy 的 MT19937 用 init_genrand 种子化（_legacy_seeding），输出流与 C++ 逐个相同
  std::uniform_real_distribution<float>(a, b)  generate_canonical<float,24>：每个值恰好消耗 1 个 32 位输出；
                                               ret = float(u)/2^32（u 先舍入到 float），ret>=1 时取 nextafter(1,0)；值 = ret*(b-a)+a，全程 float
  std::shuffle(first, last, g)                 N=urange；若 (2^32-1)/N >= N，走「每次抽一个数产生两个交换位置」的快路径，
                                               否则逐元素抽；区间内均匀整数用 Lemire 的 nearly divisionless（32 位范围、64 位乘积）
Xavier 抽样顺序与 (fan_in, fan_out) 见 model.ProtoCnn.xavier_plan（对应 src/model.cpp 的 make_cnn）。
引擎是有状态的：CppMt19937 对象在整个训练里只有一个，每个 epoch 的 shuffle 接着上一次的位置继续（同 trainer.cpp 里的 `std::mt19937 rng(seed)`）。
"""
import numpy as np

_TWO32 = 1 << 32
_MASK32 = _TWO32 - 1
_ONE_MINUS = np.nextafter(np.float32(1.0), np.float32(0.0))   # float，约 0.99999994


class CppMt19937:
    """std::mt19937 的输出流。内部按块从 numpy 取，块只是缓冲，不改变输出顺序。"""

    def __init__(self, seed, block=1 << 14):
        self._bg = np.random.MT19937()
        self._bg._legacy_seeding(int(seed) & _MASK32)
        self._buf = np.zeros(0, dtype=np.uint64)
        self._pos = 0
        self._block = block
        self.consumed = 0

    def _refill(self):
        self._buf = self._bg.random_raw(self._block)
        self._pos = 0

    def next(self):
        """下一个 32 位输出（Python int）。"""
        if self._pos >= len(self._buf):
            self._refill()
        v = int(self._buf[self._pos])
        self._pos += 1
        self.consumed += 1
        return v

    def take(self, n):
        """接下来 n 个输出，uint32 数组。"""
        out = np.empty(n, dtype=np.uint32)
        got = 0
        while got < n:
            if self._pos >= len(self._buf):
                self._refill()
            k = min(n - got, len(self._buf) - self._pos)
            out[got:got + k] = self._buf[self._pos:self._pos + k]
            self._pos += k
            got += k
        self.consumed += n
        return out

    # ---- uniform_real_distribution<float> ----
    def uniform_float(self, a, b, n):
        """n 个 std::uniform_real_distribution<float>(a, b) 的值（float32 数组）。a、b 是 float32。"""
        u = self.take(n)
        ret = u.astype(np.float32) / np.float32(_TWO32)           # float(u) 舍入到 24 位尾数；除以 2^32 是精确的
        ret = np.where(ret >= np.float32(1.0), _ONE_MINUS, ret).astype(np.float32)
        a32, b32 = np.float32(a), np.float32(b)
        return (ret * np.float32(b32 - a32)).astype(np.float32) + a32   # (ret*(b-a)) 先舍入，再加 a：同 C++ 无 FMA 的两步

    # ---- std::uniform_int_distribution 的 Lemire 路径（32 位引擎、范围 <= 2^32）----
    def _nd(self, rng_range):
        """[0, rng_range) 内均匀的整数，rng_range 是 uint32（1 <= rng_range <= 2^32-1）。"""
        product = self.next() * rng_range
        low = product & _MASK32
        if low < rng_range:
            threshold = ((_TWO32 - rng_range) & _MASK32) % rng_range
            while low < threshold:
                product = self.next() * rng_range
                low = product & _MASK32
        return product >> 32

    # ---- std::shuffle ----
    def shuffle(self, order):
        """就地打乱一维数组（行为同 std::shuffle(order.begin(), order.end(), *this)）。"""
        n = len(order)
        if n <= 1:
            return
        a = order.tolist() if isinstance(order, np.ndarray) else order
        urange = n
        if _MASK32 // urange >= urange:
            i = 1
            if urange % 2 == 0:
                j = self._nd(2)                       # uniform_int_distribution{0,1}
                a[i], a[j] = a[j], a[i]
                i += 1
            while i != n:
                b0 = i + 1                            # swap_range = (i - first) + 1
                b1 = b0 + 1
                x = self._nd(b0 * b1)                 # uniform_int_distribution{0, b0*b1-1}
                p0, p1 = x // b1, x % b1
                a[i], a[p0] = a[p0], a[i]
                i += 1
                a[i], a[p1] = a[p1], a[i]
                i += 1
        else:
            for i in range(1, n):
                j = self._nd(i + 1)                   # uniform_int_distribution{0, i}
                a[i], a[j] = a[j], a[i]
        if isinstance(order, np.ndarray):
            order[:] = a


def xavier_values(rng, numel, lim):
    """一个权重张量的 Xavier 抽样：uniform_real_distribution<float>(-lim, lim)，每个元素一次。"""
    return rng.uniform_float(-np.float32(lim), np.float32(lim), numel)


class CppXavier:
    """给 model.ProtoCnn.init_xavier 用的 uniform(numel, lim) 回调；seed 同 uavrgb_train 的 --seed。"""

    def __init__(self, seed):
        self.rng = CppMt19937(seed)

    def __call__(self, numel, lim):
        return xavier_values(self.rng, numel, lim)
