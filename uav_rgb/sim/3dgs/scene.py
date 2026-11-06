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

# 渲染/碰撞共用的默认值（scene.json 的 "render" 段可覆盖）。数字来自
# docs/worklog/2026-11-06-*-3dgs-geometry-render.md 的 Callinstrasse 扫描：
#   max_scale_m=3：去掉最大轴 1σ > 3 m 的 6226 个（0.3%）"天空壳/雾"高斯后，机载帧
#     暗通道 0.51→0.40、拉普拉斯方差 36→437，alpha 均值仍 0.96；再往下（2/1.5/1 m）
#     背景洞 >17%。
#   supersample=2：2 倍渲染再盒滤波缩小，细结构不闪且比 antialiased 模式清楚
#     （lapvar 83 vs 38）。
#   sh_degree=3：读 f_rest，视角相关颜色恢复（暗通道 0.51→0.49，lapvar 36→68）。
RENDER_DEFAULTS = {"sh_degree": 3, "max_scale_m": 3.0, "supersample": 2}

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

    返回 means/quats(w,x,y,z)/scales/opacities/colors（DC 项换算的 RGB 0..1），以及
    sh：[N,(L+1)^2,3] 的完整球谐系数（DC + f_rest），sh_degree=L；缺 f_rest 或
    数量不是 3*((L+1)^2-1) 时 sh=None、sh_degree=0（只能按 DC 渲染）。
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
    # 高阶球谐：INRIA 布局 f_rest 按 [3 通道][(L+1)^2-1 系数] 排，gsplat 要 [N,K,3]。
    n_rest = sum(1 for p in props if p.startswith("f_rest_"))
    sh, sh_degree = None, 0
    if n_rest and n_rest % 3 == 0:
        k = n_rest // 3 + 1
        deg = int(round(math.sqrt(k))) - 1
        if (deg + 1) ** 2 == k and deg > 0:
            rest = np.stack([table[f"f_rest_{i}"] for i in range(n_rest)],
                            axis=1).astype(np.float32).reshape(count, 3, k - 1)
            sh = np.concatenate([dc[:, None, :], rest.transpose(0, 2, 1)], axis=1)
            sh_degree = deg
    return {"means": means, "colors": colors, "opacities": opac,
            "scales": scales, "quats": quats, "sh": sh, "sh_degree": sh_degree}


def scale_keep_mask(scales, max_scale_m):
    """尺度过滤：保留最大轴 1σ <= max_scale_m 的高斯；None 表示不过滤。

    巨型高斯（Callinstrasse 里 1σ 可到 6500 m）是训练时拟合天空/远景的"壳"，
    机载视角里它们盖在近景前面形成白雾；去掉后被它们遮住的洞会露出背景色。
    """
    scales = np.asarray(scales)
    if max_scale_m is None:
        return np.ones(len(scales), dtype=bool)
    return scales.max(axis=1) <= float(max_scale_m)


def render_options(scene):
    """scene.json 的 render 段 + RENDER_DEFAULTS -> dict（渲染端与碰撞端同一来源）。"""
    opts = dict(RENDER_DEFAULTS)
    opts.update({k: v for k, v in (scene or {}).get("render", {}).items() if k in opts})
    return opts


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
    几何：地面 + 两侧墙 + 若干立柱/箱子，沿 +x。点间距约 0.1 m、σ=0.06 m，
    相邻高斯在 1.5σ 处重叠，碰撞代理 v2（3D 椭球占据）才能把面补严。
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

    add((-2.0, -width / 2, -0.05), (length, width / 2, 0.0), (0.42, 0.38, 0.30), 0.12)
    for sign in (-1, 1):
        y = sign * width / 2
        add((-2.0, y - 0.05, 0.0), (length, y + 0.05, height), (0.62, 0.60, 0.55), 0.10)
    # 立柱/箱子：给碰撞和偏航留出真实障碍。
    boxes = [(7.0, 1.2, 0.45, 2.4, (0.75, 0.30, 0.15)),
             (7.0, -1.2, 0.45, 2.4, (0.75, 0.30, 0.15)),
             (14.0, 0.0, 0.55, 2.6, (0.20, 0.40, 0.70)),
             (21.0, 1.5, 0.45, 2.2, (0.66, 0.62, 0.12)),
             (21.0, -1.6, 0.45, 2.2, (0.66, 0.62, 0.12)),
             (27.0, 0.0, 0.60, 2.3, (0.16, 0.55, 0.24))]
    for x, y, half, h, color in boxes:
        add((x - half, y - half, 0.0), (x + half, y + half, h), color, 0.07)

    means = np.concatenate(pts).astype(np.float32)
    colors = np.concatenate(cols).astype(np.float32)
    os.makedirs(out_dir, exist_ok=True)
    ply = os.path.join(out_dir, "scene.ply")
    write_gaussian_ply(ply, means, colors,
                       scales=np.full((len(means), 3), 0.06, np.float32))
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
        # 合成点是随机撒的，局部会有 0.2–0.3 m 的空隙；用 3σ + 0.1 m 余量把面补严
        # （真实场景用 collision.py 的默认 1.5σ、无余量）。
        "collision": {"z_slice": [1.65, 2.35], "voxel_m": 0.1,
                      "sigma_k": 3.0, "margin_m": 0.1, "envelope": "e1_mini"},
        "max_steps": 60,
        "step_s": 0.5,
        "notes": [
            "合成走廊，尺度/上轴已知（米，z 向上），非 COLMAP 重建。",
            "碰撞代理：导航系 3D 体素占据（高斯椭球）+ 地面平面，沿轨迹扫掠；不是 3DGS 求交。",
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


def _quat_rotmat(q):
    """单个 (w,x,y,z) -> 3x3（机体->世界）。"""
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        raise ValueError("四元数不能为零")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def viewmat_from_quat(pos, quat, mount=None, nav_to_world=None):
    """机体位姿（nav 系位置 + FLU->nav 四元数 w,x,y,z）⊗ 相机安装 -> gsplat world->camera 4x4。

    mount：e1_mini.json camera_mount.value，{"xyz": 机体系偏移 m, "pitch_deg": 抬头为正}；
    None = 机体原点、不俯仰。相机轴：右=-y_body、下=-z_body、前=+x_body，再绕机体 y 轴抬头 pitch。
    四元数只含 yaw、mount 为 None 时与 viewmat_from_pose(x, y, z, yaw) 相同；roll/pitch 直接进画面。
    """
    r_nb = _quat_rotmat(quat)                                   # body -> nav
    xyz = np.zeros(3) if mount is None else np.asarray(mount.get("xyz", (0, 0, 0)), np.float64)
    pitch = math.radians(0.0 if mount is None else float(mount.get("pitch_deg", 0.0)))
    cp, sp = math.cos(pitch), math.sin(pitch)
    r_cb = np.array([[0.0, -1.0, 0.0],       # 相机右 = 机体 -y
                     [sp, 0.0, -cp],         # 相机下 = 机体 -z 抬头 pitch 后
                     [cp, 0.0, sp]])         # 相机前 = 机体 +x 抬头 pitch 后
    r_cn = r_cb @ r_nb.T
    c_nav = np.asarray(pos, dtype=np.float64) + r_nb @ xyz
    m = np.asarray(np.eye(4) if nav_to_world is None else nav_to_world, dtype=np.float64)
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
