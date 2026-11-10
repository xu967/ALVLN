#!/usr/bin/env python3
"""DroNet 的 npz 与 Keras 原 h5 权重逐键逐元素比对（解释器：$D/baselines/venv-h5/bin/python，只有它带 h5py）。

npz 的键 = h5 数据集路径把 '/' 换成 '|'（同一份转换见 sim/tools/baselines/README.md「权重」）。
不等 / 缺键 / 多键都以非 0 退出，输出一行汇总。
"""
import sys

import h5py
import numpy as np

BASE = "/root/rivermind-data/uav_rgb/baselines/dronet"   # 第三方权重留在数据盘
h5 = h5py.File(f"{BASE}/model/dronet/best_weights.h5", "r")
z = np.load(f"{BASE}/dronet_best_weights.npz")
found = {}
h5.visititems(lambda name, obj: found.__setitem__(name, obj[()]) if isinstance(obj, h5py.Dataset) else None)
want = {k.replace("|", "/"): z[k] for k in z.files}
missing, extra = sorted(set(found) - set(want)), sorted(set(want) - set(found))
bad = [k for k in sorted(set(found) & set(want)) if not np.array_equal(found[k], want[k])]
print(f"h5 数据集 {len(found)} 个，npz 键 {len(want)} 个；h5 有而 npz 无 {len(missing)}，npz 有而 h5 无 {len(extra)}，数值不等 {len(bad)}")
for k in missing + extra + bad:
    print("  不一致:", k)
sys.exit(1 if (missing or extra or bad) else 0)
