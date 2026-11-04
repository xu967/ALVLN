#!/usr/bin/env python3
"""3DGS 场景：PLY 高斯读写、合成场景生成、scene.json、位姿 -> viewmat。

坐标约定：nav 世界系 x/y 水平、z 向上、yaw 从 +x 逆时针（度）。
gsplat 的 viewmat 是 world->camera（相机看 -z）。COLMAP/3DGS 场景的尺度和
上轴未知，scene.json 用 nav_to_world（4x4 刚体）手工对齐；合成场景为 I。
"""
import json
import math
import os

import numpy as np

C0 = 0.28209479177387814   # SH DC -> RGB 的系数

PLY_PROPS = ("x", "y", "z", "nx", "ny", "nz",
             "f_dc_0", "f_dc_1", "f_dc_2") + tuple(
    f"f_rest_{i}" for i in range(45)) + (
    "opacity", "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3")


def _ply_dtype(props, endian="<"):
    """属性名 -> 结构化 dtype；未知属性按 float32 读（INRIA 3DGS 全是 float）。"""
    return np.dtype([(name, endian + "f4") for name in props])


def read_gaussian_ply(path):
    """读 INRIA 3DGS PLY（binary_little_endian/ascii），返回 numpy 字段字典。

    只保证返回 means/quats/scales/opacities/colors（RGB 0..1）与 points；
    缺 f_rest 也能读（sh_degree=0 渲染）。
    """
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} 不是 PLY")
        fmt = None
        props, count, header_done = [], 0, False
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path} PLY 头不完整")
            text = line.decode("ascii", "replace").strip()
            if text.startswith("format "):
                fmt = text.split()[1]
            elif text.startswith("element vertex "):
                count = int(text.split()[2])
            elif text.startswith("property "):
                parts = text.split()
                if len(parts) == 3 and parts[1] == "float":
                    props.append(parts[2])
            elif text == "end_header":
                header_done = True
                break
        if not header_done:
            raise ValueError(f"{path} 缺 end_header")
        if fmt == "binary_little_endian":
            need = {"x", "y", "z", "f_dc_0", "opacity"}
            if not need <= set(props):
                raise ValueError(f"{path} 缺 3DGS 属性: {sorted(need - set(props))}")
            table = np.fromfile(f, dtype=_ply_dtype(props), count=count)
        elif fmt == "ascii":
            raw = np.loadtxt(f, dtype=np.float32, ndmin=2)
            if raw.shape[0] != count or raw.shape[1] != len(props):
                raise ValueError(f"{path}: ascii 行数/列数与头不一致")
            table = np.zeros(count, dtype=_ply_dtype(props))
            for i, name in enumerate(props):
                table[name] = raw[:, i]
        else:
            raise ValueError(f"{path}: 不支持的 PLY 格式 {fmt!r}")

    means = np.stack([table["x"], table["y"], table["z"]], axis=1).astype(np.float32)
    dc = np.stack([table[f"f_dc_{i}"] for i in range(3)], axis=1).astype(np.float32)
    colors = np.clip(0.5 + C0 * dc, 0.0, 1.0).astype(np.float32)
    opac = 1.0 / (1.0 + np.exp(-table["opacity"].astype(np.float32)))
    if "scale_0" in props:
        scales = np.exp(np.stack([table[f"scale_{i}"] for i in range(3)],
                                 axis=1).astype(np.float32))
    else:
        scales = np.full((count, 3), 0.02, dtype=np.float32)
    if "rot_0" in props:
        quats = np.stack([table[f"rot_{i}"] for i in range(4)],
                         axis=1).astype(np.float32)
        quats /= np.linalg.norm(quats, axis=1, keepdims=True) + 1e-9
    else:
        quats = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (count, 1))
    return {"means": means, "colors": colors, "opacities": opac,
            "scales": scales, "quats": quats}


def write_gaussian_ply(path, means, colors, scales=None, quats=None, opacities=None):
    """写 INRIA 3DGS binary PLY（f_rest 补零），供 gsplat/外部查看器使用。"""
    n = len(means)
    colors = np.asarray(colors, dtype=np.float32)
    scales = (np.full((n, 3), 0.03, np.float32) if scales is None
              else np.asarray(scales, dtype=np.float32))
    quats = (np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)) if quats is None
             else np.asarray(quats, dtype=np.float32))
    opac = (np.full(n, 4.6, np.float32) if opacities is None
            else np.asarray(opacities, dtype=np.float32))
    # f_dc = (rgb - 0.5) / C0；opacity/scale 写 sigmoid/exp 前的值。
    dc = (np.clip(colors, 0, 1) - 0.5) / C0
    pos = np.zeros(n, dtype=_ply_dtype(PLY_PROPS))
    pos["x"], pos["y"], pos["z"] = means[:, 0], means[:, 1], means[:, 2]
    pos["f_dc_0"], pos["f_dc_1"], pos["f_dc_2"] = dc[:, 0], dc[:, 1], dc[:, 2]
    pos["opacity"] = np.log(np.clip(opac, 1e-6, 1 - 1e-6) /
                            np.clip(1 - opac, 1e-6, 1))
    for i in range(3):
        pos[f"scale_{i}"] = np.log(np.maximum(scales[:, i], 1e-6))
    for i in range(4):
        pos[f"rot_{i}"] = quats[:, i]
    header = ["ply", "format binary_little_endian 1.0",
              f"element vertex {n}", "property float x", "property float y",
              "property float z", "property float nx", "property float ny",
              "property float nz", "property float f_dc_0", "property float f_dc_1",
              "property float f_dc_2"]
    header += [f"property float f_rest_{i}" for i in range(45)]
    header += ["property float opacity", "property float scale_0",
               "property float scale_1", "property float scale_2",
               "property float rot_0", "property float rot_1",
               "property float rot_2", "property float rot_3", "end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        pos.tofile(f)


def _scatter_surface(rng, n, lo, hi, color, jitter=0.01, color_jitter=0.06):
    """在轴对齐盒表面撒点；lo/hi 为 (3,)。返回 means, colors。"""
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    pts = rng.uniform(lo, hi, size=(n, 3))
    # 每个点贴到最近的一个盒面上（随机选面）。
    axis = rng.integers(0, 3, size=n)
    side = rng.integers(0, 2, size=n)
    pts[np.arange(n), axis] = np.where(side == 0, lo[axis], hi[axis])
    pts += rng.normal(0, jitter, size=(n, 3))
    base = np.asarray(color, np.float64)
    cols = np.clip(base + rng.normal(0, color_jitter, size=(n, 3)), 0, 1)
    return pts.astype(np.float32), cols.astype(np.float32)


def make_corridor(out_dir, seed=0, length=32.0, width=8.0, height=4.0):
    """生成 3DGS 合成走廊（盒面高斯）+ scene.json；返回 scene 字典。

    这是**合成场景**，不是真实重建；用于在没有公共场景的情况下打通渲染/碰撞/闭环。
    几何：地面 + 两侧墙 + 若干立柱/箱子，沿 +x。点密度约 0.15 m。
    """
    rng = np.random.default_rng(seed)
    pts, cols = [], []

    def add(lo, hi, color, density=0.15, jitter=0.01):
        size = (np.asarray(hi) - np.asarray(lo))
        # 只按两个最大面估算点数，避免薄板产生过多点。
        area = 2 * (size[0] * size[1] + size[0] * size[2] + size[1] * size[2])
        n = max(50, int(area / density ** 2))
        p, c = _scatter_surface(rng, n, lo, hi, color, jitter)
        pts.append(p)
        cols.append(c)

    add((-2.0, -width / 2, -0.05), (length, width / 2, 0.0), (0.42, 0.38, 0.30), 0.18)
    for sign in (-1, 1):
        y = sign * width / 2
        add((-2.0, y - 0.05, 0.0), (length, y + 0.05, height), (0.62, 0.60, 0.55), 0.16)
    # 立柱/箱子：给碰撞和偏航留出真实障碍。
    boxes = [(7.0, 1.2, 0.45, 2.4, (0.75, 0.30, 0.15)),
             (7.0, -1.2, 0.45, 2.4, (0.75, 0.30, 0.15)),
             (14.0, 0.0, 0.55, 2.6, (0.20, 0.40, 0.70)),
             (21.0, 1.5, 0.45, 2.2, (0.66, 0.62, 0.12)),
             (21.0, -1.6, 0.45, 2.2, (0.66, 0.62, 0.12)),
             (27.0, 0.0, 0.60, 2.3, (0.16, 0.55, 0.24))]
    for x, y, half, h, color in boxes:
        add((x - half, y - half, 0.0), (x + half, y + half, h), color, 0.10)

    means = np.concatenate(pts).astype(np.float32)
    colors = np.concatenate(cols).astype(np.float32)
    os.makedirs(out_dir, exist_ok=True)
    ply = os.path.join(out_dir, "scene.ply")
    write_gaussian_ply(ply, means, colors,
                       scales=np.full((len(means), 3), 0.035, np.float32))
    scene = {
        "schema": "uavrgb.3dgs_scene.v1",
        "name": f"synthetic_corridor_s{seed}",
        "source": "self-generated synthetic 3DGS corridor (not a real reconstruction)",
        "license": "self-authored, MIT-equivalent",
        "ply": "scene.ply",
        "nav_to_world": np.eye(4).tolist(),
        "world_up": "z",
        "ground_z": 0.0,
        "spawn": {"x": 0.5, "y": 0.0, "z": 2.0, "yaw_deg": 0.0},
        "waypoint": {"x": 30.0, "y": 0.0, "z": 2.0},
        "reach_radius_m": 1.5,
        "camera": {"width": 320, "height": 240, "fov_deg": 60.0, "pitch_deg": 0.0},
        "collision": {"z_slice": [1.65, 2.35], "voxel_m": 0.1,
                      "inflate_m": 0.35, "drone_radius_m": 0.3},
        "max_steps": 60,
        "step_s": 0.5,
        "notes": [
            "合成走廊，尺度/上轴已知（米，z 向上），非 COLMAP 重建。",
            "碰撞代理：2 m 高度切片点云体素占据 + 地面平面，不是 3DGS 求交。",
        ],
    }
    save_scene(scene, os.path.join(out_dir, "scene.json"))
    return scene


def save_scene(scene, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(scene, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_scene(path):
    """读 scene.json；ply 路径相对 json 所在目录。"""
    with open(path, "r", encoding="utf-8") as f:
        scene = json.load(f)
    if scene.get("schema") != "uavrgb.3dgs_scene.v1":
        raise ValueError(f"{path}: schema 不是 uavrgb.3dgs_scene.v1")
    scene["path"] = os.path.abspath(path)
    scene["dir"] = os.path.dirname(scene["path"])
    scene["ply_path"] = os.path.join(scene["dir"], scene["ply"])
    if not os.path.isfile(scene["ply_path"]):
        raise FileNotFoundError(f"scene.ply 不存在: {scene['ply_path']}")
    m = np.asarray(scene.get("nav_to_world", np.eye(4).tolist()), dtype=np.float64)
    if m.shape != (4, 4) or not np.isfinite(m).all():
        raise ValueError(f"{path}: nav_to_world 必须是有限 4x4")
    scene["_nav_to_world"] = m
    return scene


def viewmat_from_pose(x, y, z, yaw_deg, pitch_deg=0.0, nav_to_world=None):
    """nav 位姿 -> gsplat world->camera 4x4（float64）。

    pitch 正为抬头；相机轴：右、下、前（+z 朝前，COLMAP/OpenCV 约定）。
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    fwd = np.array([cy * cp, sy * cp, sp])          # nav 前向
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    norm = np.linalg.norm(right)
    if norm < 1e-8:                                  # 垂直俯仰退化
        right = np.array([sy, -cy, 0.0])
    else:
        right /= norm
    down = np.cross(fwd, right)
    # gsplat/COLMAP 约定：相机 +z 朝前（不是 OpenGL 的 -z）。
    r_cn = np.stack([right, down, fwd])              # world->cam 旋转（nav 系）
    c_nav = np.array([x, y, z], dtype=np.float64)
    if nav_to_world is None:
        nav_to_world = np.eye(4)
    m = np.asarray(nav_to_world, dtype=np.float64)
    r_m, t_m = m[:3, :3], m[:3, 3]
    r_cw = r_cn @ r_m.T
    c_world = r_m @ c_nav + t_m
    view = np.eye(4)
    view[:3, :3] = r_cw
    view[:3, 3] = -r_cw @ c_world
    return view


def intrinsics(width, height, fov_deg):
    """水平 FOV -> K；方形像素（fy=fx），主点在中心。"""
    fx = 0.5 * width / math.tan(math.radians(fov_deg) / 2.0)
    return np.array([[fx, 0.0, width / 2.0],
                     [0.0, fx, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def main():
    import argparse
    p = argparse.ArgumentParser(description="3DGS 场景工具")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("make-synthetic", help="生成合成走廊 + scene.json")
    g.add_argument("--out", required=True)
    g.add_argument("--seed", type=int, default=0)
    info = sub.add_parser("info", help="打印场景摘要")
    info.add_argument("scene")
    args = p.parse_args()
    if args.cmd == "make-synthetic":
        scene = make_corridor(args.out, seed=args.seed)
        print(json.dumps({k: v for k, v in scene.items() if not k.startswith("_")},
                         indent=2, ensure_ascii=False))
    else:
        s = load_scene(args.scene)
        print(json.dumps({k: v for k, v in s.items()
                          if not k.startswith("_")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
