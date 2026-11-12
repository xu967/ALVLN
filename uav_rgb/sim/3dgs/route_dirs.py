#!/usr/bin/env python3
"""读 3DGS 实验输出里的路线目录（任务卡 S1b，审核 E-12）：新汇总脚本都用这里的函数，不再自己写 r* 通配。

输出布局（experiments/2026-11-10-bearing-guard-3dgs/driver.py）：<根>/<若干层目录>/r<N>/ep_NNNNN/<文件>。
续跑时驱动把没跑完的路线目录改名成 r<N>.incomplete-<时刻> 再重跑；通配 r*/ep_*/… 会把改名后的半成品一起读进来，
所以这里只收名字恰为 r<数字> 的目录。

depth = 根与路线目录之间的层数：旧 3DGS 实验 <臂>/<场景>/r<N> 是 2（缺省）；teacher-tol 的 <场景>/r<N> 是 1；根下直接是 r<N> 是 0。

已跑完的实验自带的汇总脚本不改（它们读的目录里没有半成品，见 docs/worklog 里 S1b 的清单）。
"""
import glob
import os
import re

ROUTE_DIR = re.compile(r"^r\d+$")      # 完整的路线目录名；r3.incomplete-020101、r3.bak、rX 都不匹配


def route_dirs(root, depth=2):
    """<root>/<depth 层任意目录>/r<数字> 的目录路径，字典序；root 不存在返回空列表。"""
    pattern = os.path.join(root, *(["*"] * depth), "r[0-9]*")
    return sorted(p for p in glob.glob(pattern) if ROUTE_DIR.match(os.path.basename(p)) and os.path.isdir(p))


def episode_files(root, name="metrics.json", depth=2):
    """每个完整路线目录下 ep_*/<name> 的路径，字典序（没有半成品时与旧通配 sorted(glob(r*/ep_*/<name>)) 逐项相同）。"""
    out = []
    for d in route_dirs(root, depth):
        out.extend(glob.glob(os.path.join(d, "ep_*", name)))
    return sorted(out)
