#!/usr/bin/env python3
""".splat (antimatter15 32B/point) -> INRIA 3DGS PLY。

输入不是训练数据，只是把公开/自建 .splat 转成 gsplat 可读的 PLY。

坐标变换只允许**正交旋转（行列式 +1）+ 各向同性尺度 + z 平移**，同一个旋转同时作用到
高斯中心和朝向四元数（Σ' = R Σ Rᵀ ⇔ q' = q_R ⊗ q），scales 乘尺度：
  --up y      源 y-up  -> nav z-up：nav = (x, -z, y)，绕 x 转 +90°（默认，多数 Unity/glTF 导出）
  --up=-y     源 y-down -> nav z-up：nav = (x, z, -y)，绕 x 转 -90°（COLMAP/INRIA 训练出的 3DGS；
              argparse 会把裸的 "-y" 当选项，写 --up=-y 或 --up ydown）
  --up z      不旋转
以前的 --flip-height（只翻 z）是行列式 -1 的镜像，场景左右反了，已删除；
y-down 的场景要用 --up -y（= y-up 旋转后再绕 x 转 180°，仍是正常旋转）。

.splat 字节布局：pos f32×3、scale f32×3（线性）、rgba u8×4、rot u8×4。rot 四个字节是
(w, x, y, z)，解码 (b-128)/128，与 PLY rot_0..3 和 gsplat 的 wxyz 同序（antimatter15 的
convert.py 直接按 rot_0..3 顺序量化；实测按 xyzw 解读会把 bicycle 渲糊，lapvar 16.8→4.2）。
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim", "3dgs"))

import scene as scene_mod  # noqa: E402

DT = np.dtype([("pos", "<f4", (3,)), ("scale", "<f4", (3,)),
               ("rgba", "u1", (4,)), ("rot", "u1", (4,))])

# 源 up 轴 -> 把它转到 nav z-up 的旋转矩阵（行向量约定：nav = R @ src）。
UP_ROTATIONS = {
    "y": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),   # 绕 x +90°
    "-y": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),  # 绕 x -90°
    "z": np.eye(3),
}


UP_ALIASES = {"ydown": "-y", "y-down": "-y", "yup": "y", "y-up": "y"}


def rotation_for_up(up):
    """返回正交、行列式 +1 的 3x3 旋转；非法 up 或矩阵不是正常旋转时抛 ValueError。"""
    up = UP_ALIASES.get(str(up).lower(), up)
    if up not in UP_ROTATIONS:
        raise ValueError(f"--up 只能是 {sorted(UP_ROTATIONS)}（或别名 {sorted(UP_ALIASES)}），实际 {up!r}")
    R = UP_ROTATIONS[up]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-12) or not np.isclose(np.linalg.det(R), 1.0):
        raise ValueError("旋转矩阵必须正交且行列式 +1（镜像不允许）")
    return R


def rotmat_to_quat(R):
    """3x3 正常旋转 -> 单位四元数 (w, x, y, z)。"""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_multiply(p, q):
    """Hamilton 积 p ⊗ q，p 是 (4,)，q 是 [N,4]，都为 (w, x, y, z)。"""
    pw, px, py, pz = (float(v) for v in p)
    q = np.asarray(q, dtype=np.float64)
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    ], axis=1)


def read_splat(src):
    """.splat -> dict(means, scales, colors, opacities, quats[wxyz 单位化])，均为源坐标系。"""
    a = np.fromfile(src, dtype=DT)
    if a.size == 0:
        raise ValueError(f"{src}: 空 .splat")
    quats = (a["rot"].astype(np.float32) - 128.0) / 128.0
    quats = quats / np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-9)
    return {"means": a["pos"].astype(np.float32),
            "scales": a["scale"].astype(np.float32),
            "colors": a["rgba"][:, :3].astype(np.float32) / 255.0,
            "opacities": a["rgba"][:, 3].astype(np.float32) / 255.0,
            "quats": quats.astype(np.float32)}


def transform_gaussians(g, up="y", scale=1.0, z_offset=0.0):
    """同一个正常旋转作用到 means 与 quats，scales 乘尺度，再加 z 平移。返回新 dict。"""
    scale = float(scale)
    if not (np.isfinite(scale) and scale > 0):
        raise ValueError(f"--scale 必须是正数，实际 {scale}")
    R = rotation_for_up(up)
    q_r = rotmat_to_quat(R)
    means = (np.asarray(g["means"], dtype=np.float64) @ R.T) * scale
    means[:, 2] += float(z_offset)
    quats = quat_multiply(q_r, g["quats"])
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-9)
    return {"means": means.astype(np.float32),
            "scales": (np.asarray(g["scales"], dtype=np.float32) * scale).astype(np.float32),
            "colors": g["colors"], "opacities": g["opacities"],
            "quats": quats.astype(np.float32), "rotation": R, "rotation_quat": q_r}


def convert(src, dst, up="y", scale=1.0, z_offset=0.0):
    g = transform_gaussians(read_splat(src), up=up, scale=scale, z_offset=z_offset)
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    scene_mod.write_gaussian_ply(dst, g["means"], g["colors"], g["scales"], g["quats"],
                                 g["opacities"])
    means = g["means"]
    print("wrote %s n=%d up=%s scale=%g z_offset=%g bounds min=%s max=%s"
          % (dst, len(means), up, scale, z_offset, means.min(axis=0).round(3).tolist(),
             means.max(axis=0).round(3).tolist()))
    return means


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--up", choices=sorted(UP_ROTATIONS) + ["ydown"], default="y",
                   help="源文件的 up 轴：y（默认，绕 x +90°）、-y（COLMAP y-down，绕 x -90°；"
                        "argparse 把 -y 当选项，要写 --up=-y 或 --up ydown）、z（不转）")
    p.add_argument("--scale", type=float, default=1.0, help="各向同性尺度，同时乘 means 与 scales")
    p.add_argument("--z-offset", type=float, default=0.0, help="旋转+缩放后加到 z 的平移（米）")
    p.add_argument("--flip-height", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--no-rotate-y-up", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.flip_height:
        p.error("--flip-height 已删除：它只翻 z，是行列式 -1 的镜像。y-down 的源用 --up -y。")
    if a.no_rotate_y_up:
        p.error("--no-rotate-y-up 已改为 --up z")
    convert(a.src, a.dst, up=a.up, scale=a.scale, z_offset=a.z_offset)


if __name__ == "__main__":
    main()
