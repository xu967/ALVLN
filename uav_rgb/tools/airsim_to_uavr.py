#!/usr/bin/env python3
"""Kaggle AirSim 10k -> .uavr：把连续速度指令离散成我们的 4 类动作接口。

数据集：`lukpellant/droneflight-obs-avoidanceairsimrgbdepth10k-320x320`
（AirSim，10000 帧 RGB 320x320 + depth + 每帧一条 (vx, vy, vz, yaw_rate) 指令，
标签由确定性势场法生成，CC BY 4.0）。解压后目录形如

  <src>/rgb/000000.png      <src>/depth/000000.npy     <src>/commands/000000.npy

关于单位：.npy 里存的是**原始指令**，不是归一化值——四列的均值/标准差
（vx 2.43/0.87、vy 0/0、vz 0.025/0.32、yaw_rate -1.17/20.56）正好就是原值
的统计量；若再按这组参数反归一化，yaw_rate 会变成 ±1447 deg/s（物理上不可能）。
实测取值范围：vx 1.00~3.50 m/s、vy 恒为 0、vz -0.93~1.88 m/s、
yaw_rate -70.3~+70.4 deg/s。

已知硬伤（本工具改不了，只能照实写进 meta 和文档）：
  - **vx 最小 1.00 m/s，永远在前进 => SAFE_NEUTRAL 一条都没有**。
  - 没有场景/位置元数据，也没有碰撞标记。
  - 世界是 AirSim 郊区街道，不是我们的柱阵；320x320 方形，与本机 320x240 不同。

离散化规则（确定性，可复算，改阈值等于改标签口径）见 action_of()。

用法：
  python3 tools/airsim_to_uavr.py --src <解压目录> --out <输出目录>
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import ACTION_NAMES, die, write_dataset  # noqa: E402

SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT = range(4)
SRC_SIZE = (320, 320)
# 方形原图中心裁成 4:3 再缩放，避免 320x320 -> 160x120 的纵向拉伸。
CROP = (0, 40, 320, 280)          # left, top, right, bottom -> 320x240
CROPPED_SIZE = (320, 240)
YAW_DEG = 10.0                    # |yaw_rate| >= 此值判为转向（deg/s）
VX_MIN = 0.05                     # vx <= 此值判为停（m/s）


def action_of(vx, vy, vz, yaw_deg):
    """连续指令 -> 4 类。先判转向，再判前进，都不满足才是停。

    yaw 优先：同一帧既前进又转向时，取「更该先做的事」。本数据里两者
    永远同时发生（vx 恒 >= 1 m/s），这条规则等价于「转得够明显就算转向」。
    """
    if yaw_deg >= YAW_DEG:
        return YAW_LEFT
    if yaw_deg <= -YAW_DEG:
        return YAW_RIGHT
    if vx > VX_MIN:
        return FORWARD_SLOW
    return SAFE_NEUTRAL


def check_labels(src):
    """标签自洽性检验：转向标签应与深度图的左右几何相关。

    势场法避障下「想左转」不会凭空出现。若标签与图像错位，两组分布会重合。
    指标：prox = 1/depth（越近越大），取下半幅（天空在远裁剪上，没有信息）
    的左半均值减右半均值；L-R > 0 表示左边更近。
    """
    rgb_dir, dep_dir = os.path.join(src, "rgb"), os.path.join(src, "depth")
    cmd_dir = os.path.join(src, "commands")
    for d in (rgb_dir, cmd_dir, dep_dir):
        if not os.path.isdir(d):
            die(f"{src} 下缺 {os.path.basename(d)}/（本检验要用 depth）")
    idx = [int(f[:6]) for f in sorted(os.listdir(cmd_dir)) if f.endswith(".npy")]

    yaw, asym = [], []
    for i in idx:
        c = np.load(os.path.join(cmd_dir, f"{i:06d}.npy")).ravel()
        d = np.load(os.path.join(dep_dir, f"{i:06d}.npy")).astype(np.float32)
        prox = 1.0 / np.clip(d, 0.2, 1e4)
        h, w = prox.shape
        yaw.append(float(c[3]))
        asym.append(float(prox[h // 2:, :w // 2].mean() - prox[h // 2:, w // 2:].mean()))
    yaw, asym = np.array(yaw), np.array(asym)
    lab = np.where(yaw >= YAW_DEG, YAW_LEFT,
                   np.where(yaw <= -YAW_DEG, YAW_RIGHT, FORWARD_SLOW))

    print(f"样本 {len(idx)}，下半幅 left-right 接近度差（1/m）：")
    for k in (FORWARD_SLOW, YAW_LEFT, YAW_RIGHT):
        a = asym[lab == k]
        if not len(a):
            continue
        print(f"  {ACTION_NAMES[k]:<13} n={len(a):>5} median={np.median(a):+.4f} "
              f"mean={a.mean():+.4f} 左更近占比={np.mean(a > 0):.1%}")
    a2, a3 = asym[lab == YAW_LEFT], asym[lab == YAW_RIGHT]
    if not (len(a2) and len(a3)):
        print("缺少转向样本，无法比较")
        return 1
    se = float(np.sqrt(a2.var(ddof=1) / len(a2) + a3.var(ddof=1) / len(a3)))
    delta = float(a2.mean() - a3.mean())
    sigma = delta / se if se else 0.0
    print(f"\nYAW_LEFT 与 YAW_RIGHT 均值差 = {delta:+.4f} 1/m，"
          f"合并标准误 {se:.4f}，标准化差 {sigma:+.1f} sigma")
    print("判据：|sigma| < 2 说明转向标签与深度左右几何无关（图像/标签可能错位）")
    return 0 if abs(sigma) >= 2.0 else 1


def self_check():
    """离散化规则与裁剪常数的内置用例（阈值是标签口径，改了必须重跑）。"""
    cases = [
        ((2.0, 0.0, 0.0, 0.0), FORWARD_SLOW),
        ((2.0, 0.0, 0.0, 9.9), FORWARD_SLOW),      # 差一点不算转
        ((2.0, 0.0, 0.0, 10.0), YAW_LEFT),         # 边界含等号
        ((2.0, 0.0, 0.0, -10.0), YAW_RIGHT),
        ((1.0, 0.0, 0.0, -70.0), YAW_RIGHT),
        ((0.0, 0.0, 0.0, 0.0), SAFE_NEUTRAL),      # 本数据不会出现，规则得给得出
        ((0.0, 0.0, 0.0, 30.0), YAW_LEFT),         # 停着也判转向：转向优先
        ((0.05, 0.0, 0.0, 0.0), SAFE_NEUTRAL),     # vx 等于阈值算停
        ((0.06, 0.0, 0.0, 0.0), FORWARD_SLOW),
    ]
    for (vx, vy, vz, yaw), want in cases:
        got = action_of(vx, vy, vz, yaw)
        if got != want:
            print(f"FAIL: action_of({vx},{vy},{vz},{yaw}) = {ACTION_NAMES[got]}，"
                  f"期望 {ACTION_NAMES[want]}")
            return 1
    left, top, right, bottom = CROP
    if (right - left, bottom - top) != CROPPED_SIZE:
        print(f"FAIL: CROP {CROP} 与 CROPPED_SIZE {CROPPED_SIZE} 不一致")
        return 1
    if CROPPED_SIZE[0] * 3 != CROPPED_SIZE[1] * 4:
        print(f"FAIL: 裁剪后 {CROPPED_SIZE} 不是 4:3")
        return 1
    print(f"airsim_to_uavr: {len(cases) + 2} 项通过")
    return 0


def main(argv=None):
    global YAW_DEG, VX_MIN
    p = argparse.ArgumentParser(description="AirSim 10k -> uavr 数据集")
    p.add_argument("--self-check", action="store_true", help="跑内置用例后退出")
    p.add_argument("--check-labels", action="store_true",
                   help="只做标签自洽性检验（标签 vs 深度左右几何）")
    p.add_argument("--src", help="解压目录（含 rgb/ depth/ commands/）")
    p.add_argument("--out", help="输出目录（写 dataset.uavr）")
    p.add_argument("--yaw-deg", type=float, default=YAW_DEG,
                   help=f"|yaw_rate| >= 此值判为转向（默认 {YAW_DEG}，单位 deg/s）")
    p.add_argument("--vx-min", type=float, default=VX_MIN,
                   help=f"vx <= 此值判为停（默认 {VX_MIN}，单位 m/s）")
    p.add_argument("--blocks", default="0.70,0.85",
                   help="train/val/test 按序号连续切分的两个分界（默认 0.70,0.85）")
    args = p.parse_args(argv)
    if args.self_check:
        return self_check()
    if args.check_labels:
        if not args.src:
            p.error("--check-labels 需要 --src")
        return check_labels(args.src)
    if not args.src or not args.out:
        p.error("需要 --src 和 --out（或 --self-check）")
    YAW_DEG, VX_MIN = args.yaw_deg, args.vx_min

    rgb_dir = os.path.join(args.src, "rgb")
    cmd_dir = os.path.join(args.src, "commands")
    if not os.path.isdir(rgb_dir) or not os.path.isdir(cmd_dir):
        die(f"{args.src} 下缺 rgb/ 或 commands/")

    # 必须整名解析：f[:6] 会把 7 位的 1000000 静默截成 100000，和别的帧配错标签。
    names = sorted(f for f in os.listdir(rgb_dir) if f.endswith(".png"))
    if not names:
        die(f"{rgb_dir} 里没有 PNG")
    bad = [f for f in names if len(f) != 10 or not f[:6].isdigit()]
    if bad:
        die(f"文件名不是 6 位数字.png：{bad[:3]}")
    idx = [int(f[:6]) for f in names]
    if idx != list(range(len(idx))):
        die(f"PNG 序号必须是 0..{len(idx)-1} 连续，实际 {idx[0]}..{idx[-1]}"
            "（标签按下标配对，不从 0 开始会错位）")

    missing = [i for i in idx
               if not os.path.exists(os.path.join(cmd_dir, f"{i:06d}.npy"))]
    if missing:
        die(f"{len(missing)} 帧缺指令，例如 {missing[:3]}")

    rows = [np.load(os.path.join(cmd_dir, f"{i:06d}.npy")).ravel() for i in idx]
    shapes = sorted({r.shape for r in rows})
    if shapes != [(4,)]:
        die(f"指令应全是 (4,) 一维，实际形状 {shapes[:5]}")
    cmds = np.stack(rows).astype(np.float64)
    labels = [action_of(*row) for row in cmds]

    counts = [labels.count(k) for k in range(4)]
    print("离散化后类别分布（yaw>=%.1f deg/s 判转向，vx<=%.2f m/s 判停）："
          % (YAW_DEG, VX_MIN))
    for name, n in zip(ACTION_NAMES, counts):
        print(f"  {name:<13} {n:>6} ({n / len(labels):.2%})")
    if counts[SAFE_NEUTRAL] == 0:
        print("  警告: SAFE_NEUTRAL 为 0 条——这份数据学不出「停」，只能当预训练/链路用")

    cuts = [float(x) for x in args.blocks.split(",")]
    if len(cuts) != 2:
        die(f"--blocks 需要正好两个分界，实际 {len(cuts)} 个：{args.blocks!r}")
    if not (0.0 < cuts[0] < cuts[1] < 1.0):
        die(f"--blocks 必须满足 0 < a < b < 1，实际 {cuts}")
    n = len(idx)
    b0, b1 = int(round(cuts[0] * n)), int(round(cuts[1] * n))
    ranges = [("train", 0, b0), ("val", b0, b1), ("test", b1, n)]

    episodes, sample_start = [], 0
    for eid, (split, lo, hi) in enumerate(ranges):
        sub = labels[lo:hi]
        if not sub:
            die(f"split={split} 的序号区间 [{lo},{hi}) 为空")
        episodes.append({
            "dir": f"airsim:{lo}-{hi}",
            "episode_id": eid,
            # 用块号当 scene_id：build_meta 要求同一 scene 不跨 split，
            # 一个连续块天然满足。这不是真实场景 id（数据里没有）。
            "scene_id": eid,
            "seed": eid,
            "version": "airsim10k",
            "split": split,
            "geometry_gid": f"airsim-block-{lo}-{hi}",
            "src_w": CROPPED_SIZE[0],
            "src_h": CROPPED_SIZE[1],
            "crop": CROP,
            # 裁剪前必须还是 320x320，否则 im.crop 会静默取错区域。
            "src_expected": SRC_SIZE,
            "dropped": 0,
            "sample_start": sample_start,
            "num_samples": hi - lo,
            "samples": [{"png": os.path.join(rgb_dir, f"{i:06d}.png"),
                         "label": labels[i], "collision": False}
                        for i in idx[lo:hi]],
        })
        sample_start += hi - lo

    # split_policy 用默认的 scene_id：这里 scene_id 就是连续块编号，块本身不跨
    # split，所以 C++ 读端要求的「同 scene_id 必须同 split」是满足的。数据里
    # 没有真实场景元数据这件事写进文档，不靠 meta 字段表达。
    write_dataset(args.out, episodes, n, (160, 120))
    out_path = os.path.join(args.out, "dataset.uavr")
    print(f"\n已写 {out_path}（{n} 样本，{os.path.getsize(out_path) / 2**20:.0f} MiB）")
    print(f"  split: train[0,{b0}) val[{b0},{b1}) test[{b1},{n})")
    print("  注意: 数据里没有场景元数据，split 只按序号连续切，"
          "无法保证 test 是新场景")
    return 0


if __name__ == "__main__":
    sys.exit(main())
