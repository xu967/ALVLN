#!/usr/bin/env python3
"""事后分析（不在预注册之内，不改卡的设计）：解析路径每格射线数 rays 取不同值时，与 Gazebo 深度图（file 包）的柱子格分带一致率。

卡与 tools/preprocess.py 固定 rays = 3；本脚本只回答「采样是不是一致率的主要限制」，数字不用于判定通过与否。
rays = 16 时子射线中心恰是 16×16 块里每个像素的中心，等于稠密逐像素渲染再取块内最小值。

  <解释器> posthoc_rays.py <episode 目录> <file 包目录>

输入与 compare.py 相同：episode 目录带 actions.jsonl、manifest、scene.json；file 包由 run.sh 的 file 路径打出（带深度）。输出一张 markdown 表。
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools"))
sys.path.insert(0, str(HERE))
import preprocess as pre  # noqa: E402
from compare import read_depth_bands  # noqa: E402

RAYS_SWEEP = (1, 2, 3, 4, 5, 8, 16)
DENSE = 16


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    ep = pre.load_episode(sys.argv[1])
    bf = read_depth_bands(sys.argv[2])
    mount, pillars = pre.analytic_episode_inputs(ep)
    focal = float(pre.episode_focal_px(ep))
    src_wh = (ep["src_w"], ep["src_h"])
    grid = bf.shape[1:]
    poses = [pre._analytic_pose(ep, s) for s in ep["samples"]]
    # 柱子格 = 稠密渲染里至少有 1 个柱面像素的格，与 compare.py 同一定义
    is_pillar = []
    for pose in poses:
        _, pid = pre.analytic_ray_depth(pose, pillars, focal, src_wh, grid, mount, DENSE, want_pid=True)
        is_pillar.append((pid.reshape(grid[0], DENSE, grid[1], DENSE) >= 0).any(axis=(1, 3)))
    is_pillar = np.stack(is_pillar)
    print("| 每格射线数 | 射线数/帧 | 柱子格一致率 | 全部格一致率 | 比 file 近的柱子格 | 比 file 远的柱子格 |")
    print("|---|---|---|---|---|---|")
    for rays in RAYS_SWEEP:
        cells = np.stack([pre.analytic_cell_depth(p, pillars, focal, src_wh, grid, mount, rays) for p in poses])
        bands = pre.depth_bands(pre.dilate_by_envelope(cells, focal, src_wh[0] / grid[1], pre.DEFAULT_ENVELOPE_RADIUS_M))
        same = bands == bf
        d = bands.astype(int) - bf.astype(int)
        mark = "（卡与提交版）" if rays == pre.ANALYTIC_RAYS else ""
        print("| %d%s | %d | %.2f%% (%d/%d) | %.2f%% | %d | %d |" % (
            rays, mark, (grid[0] * rays) * (grid[1] * rays), 100.0 * same[is_pillar].mean(), same[is_pillar].sum(),
            is_pillar.sum(), 100.0 * same.mean(), int(((d < 0) & is_pillar).sum()), int(((d > 0) & is_pillar).sum())))


if __name__ == "__main__":
    main()
