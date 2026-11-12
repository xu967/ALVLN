#!/usr/bin/env python3
"""碰撞代理 v2：3DGS 高斯 -> 导航坐标系 3D 体素占据 + 地面平面，沿轨迹扫掠。

与 v1（REVIEW #3 / AUDIT #2、#3）的差别：
  (a) 高斯先按 scene.nav_to_world 的逆变换到**导航系**再建占据，和渲染端同一坐标；
  (b) 只用不透明度 >= opacity_min 的高斯（Callinstrasse 2 m 薄片里 58% 的高斯
      不透明度 < 0.1，是看不见的幽灵墙），再去掉 smax > max_scale_m 的巨型"天空"高斯；
  (c) 每个高斯按 sigma_k 倍标准差的椭球落到体素，不只用中心；
  (d) 膨胀用固定的圆柱结构元素（xy 圆盘 + z 层），纯 numpy，不再依赖 scipy；
  (e) sweep() 沿线段按 <= 半体素步距采样，终点在墙另一侧也判碰。

仍是**近似代理**：高斯椭球不是真实表面，玻璃/薄叶/背面都可能漏；不能当 Gazebo
contact 或真实碰撞真值。

v3（2026-11-07）：同一套 Occupancy 接口多两条**真值几何**输入路径，给带 LiDAR/网格的场景
（Wanderland）当碰撞真值、也给高斯代理做对照：
  from_points()：米制点云（LiDAR raw_pcd.ply / COLMAP points3D）按体素计数占据，min_points 滤掉
      空中噪点；from_mesh()：三角网格按 <= 半体素间距在面上撒点后走同一条路。
  scene.json collision.source = "gaussians"（默认，v2）| "points" | "mesh"，collision.geometry 给文件
  （相对 scene.json 目录），from_scene() 按它切换，collect_3dgs / planner / video 不用改。

两套编号别混：上面的 v1 / v2 / v3 是代理的代际（PROXY_VERSION = 3dgs.occ.v2，v3 = 真值几何输入）；
collision_model 的 v1 / v2（卡 S1b 第一轮立、第二轮定稿，交叉审核 B 与 J）是碰撞口径，两者正交：
  口径 v1 = 本文件在 S1b 之前的判法（缺省，金标锁死不变）；
  口径 v2 = 定稿几何（第一轮的未定稿 v2 没有出过任何正式数字，v2 与 +coll.v2 后缀直接沿用给定稿口径）：
            机体视为圆柱（半径 = body_radius + inflate，半高 = body_half_height），与任一原始占据体素方块相交即碰，判定不再加高度余量；
            规划硬栅格与出生检查按整集高度包络取层（盖住起飞过冲与飞行波动，使规划看到的障碍为判定可能碰到障碍的超集），出生检查另加起飞水平漂移余量；
            每步碰撞沿 SITL 窗口内的 10 Hz trace 折线逐段扫。
"""
import json
import math
import os
import sys

import numpy as np

from . import scene as scene_mod

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
# 参数单源：sim/dynamics/e1_mini.json（collision_envelope.value）；没有别的回退。
E1_MINI_PARAMS = os.path.join(_REPO, "sim", "dynamics", "e1_mini.json")
# 旧 Gazebo X3 的机体尺寸（转子半径 0.1 + 机臂，外径约 0.71 m -> 以前代理用 0.3 m），
# 只给旧数据复现：scene.json collision.envelope = {"radius_m":0.3,"half_height_m":0.3}。
X3_ENVELOPE = {"radius_m": 0.3, "half_height_m": 0.3, "source": "legacy x3 proxy"}

# opacity_min=0.3：单个高斯中心处的像素 alpha 就是 opacity，0.3 以下单独看是半透明雾。
# Callinstrasse z∈[1.8,2.2] 的 58030 个高斯里不透明度是双峰的：<0.05 占 55%、
# 0.95–1.0 占 24%，中间每 0.05 一档只有 1–2%；0.3 落在谷底（<0.3 累计 65.6%，
# 0.25–0.30 一档 738 个），阈值在 0.2–0.5 之间挪对结果影响 <4%。
DEFAULT_OPACITY_MIN = 0.3
DEFAULT_MAX_SCALE_M = scene_mod.RENDER_DEFAULTS["max_scale_m"]   # 与渲染端同值
# sigma_k=1.5：高斯在 1.5σ 处的 alpha 是 exp(-1.125)=0.32·opacity，和 opacity_min 用同一个
# "看得见就算实体"的 0.3 口径；1σ（0.61）会让相邻高斯之间留缝，3σ（0.01）把薄墙画成 3 倍厚。
DEFAULT_SIGMA_K = 1.5
PROXY_VERSION = "3dgs.occ.v2"
# 碰撞口径（任务卡 S1b 第一轮立开关，第二轮按审核 J 定稿）：collision_model 选连续查询的形状、高度口径与每步扫掠的形状。
#   v1 = 改动前的现状（缺省，逐位不变，test_collision_model.py 的金标锁定）
#   v2 = 定稿口径：连续查询（collides / sweep / sweep_path / 规则臂列距离）取圆柱与原始占据体素方块的精确相交，
#        水平为点到方块的精确欧氏距离 <= 机体半径 + 余量，竖直为机体半高碰到的层，不加高度余量（J-1、J-2）；
#        规划硬栅格与出生检查取整集高度包络的层（J-1）；每步碰撞沿 SITL 窗口内的轨迹点逐段扫（B3，在 collect_3dgs.run_episode）。
#        第一轮的格中心欧氏结构元素（_dilate_euclid）只留给规划硬栅格用。
# 同一份几何（原始占据 raw 与口径无关）两种口径的结果不可混报：v2 的 proxy_version / metric_version 带 +coll.v2 后缀。
COLLISION_MODELS = ("v1", "v2")
DEFAULT_COLLISION_MODEL = "v1"
COLLISION_VERSION_SUFFIX = {"v1": "", "v2": "+coll.v2"}


def metric_version(base, collision_model=DEFAULT_COLLISION_MODEL):
    """评测口径版本串：v1 原样（旧产物的串不变），v2 加 +coll.v2。同一批轨迹在两种碰撞口径下的成败不可混报，汇总表头以这个串区分。"""
    if collision_model not in COLLISION_MODELS:
        raise ValueError(f"collision_model 必须是 {COLLISION_MODELS} 之一，得到 {collision_model!r}")
    return base + COLLISION_VERSION_SUFFIX[collision_model]


# v2 定稿的高度包络与起飞水平漂移余量（卡 S1b 第二轮，审核 J-1；先量后定，脚本 experiments/2026-11-12-3dgs-rebaseline/measure_z_margin.py）：
# 数据来源：W1-11（teacher-tol 室内外各 120 集）、W1-12（swept 室内外各 120 集）、K1（forward+route 与 rule-occ+route 室内外各 120 集）、
#   K3（holdout 四臂各 96 集、formal40 两臂各 120 集、outdoor 两臂各 120 集），共 16 个已完成来源、1,824 集、92,826 步（含第 0 步动作前位姿
#   pose_before 与步末 pose_gt）及 186,850 个 10 Hz trace.jsonl 轨迹点。
# 实测极值与取整规则（各极值向外取到 5 mm 的整数倍）：
#   1. 整集高度相对名义出生高度 z − spawn.z：最小 −0.034197 m（teacher-tol / swept 室内 trace）→ Z_ENVELOPE_DOWN_M = 0.035 m；
#      最大 +0.070023 m（室内第 0 步起飞过冲 pose_before.z）→ Z_ENVELOPE_UP_M = 0.075 m。
#   2. 起飞段水平漂移 hypot(step0.pose_before.xy − candidate_spawn.xy)：室内最大 0.0824 m，室外最大 0.102369 m（r5 ep3）
#      → SPAWN_DRIFT_MARGIN_M = 0.105 m。
# 用途区分：连续碰撞判定（layer_span / _grid_hit）只看机体自身圆柱半高 [z − 半高, z + 半高]，不再加高度余量；
#   按名义出生高度做的规划硬栅格（planning_grid）与出生检查（spawn_ok）取包络 [z_nom − Z_ENVELOPE_DOWN_M − 半高, z_nom + Z_ENVELOPE_UP_M + 半高]，
#   保证规划与出生检查看到的障碍层始终是全集碰撞判定可能查到的层的超集。
Z_ENVELOPE_DOWN_M = 0.035
Z_ENVELOPE_UP_M = 0.075
SPAWN_DRIFT_MARGIN_M = 0.105
# 真值几何路径的版本串（summary().proxy_version 按 source 取；高斯路径保持 v2 串，旧 manifest 可比）。
TRUTH_VERSION = {"points": "3dgs.occ.v3.points", "mesh": "3dgs.occ.v3.mesh"}
# min_points=3：LiDAR 稠密云里 0.1–0.2 m 体素内真实表面有几十到上千个点，1–2 个点的体素多是
# 行人/多路径回波之类的空中噪声。Wanderland 1MlUucug 实测（0.2 m 体素、飞行层 1.8–2.2 m、抽稀 2）：
# 19% 的占据格只有 1–2 个点，其中 84% 紧贴 ≥3 点的格（表面边缘，滤掉不影响障碍形状），其余孤立格
# 离路线 >3 m；取 3 后路线最小净空不变。COLMAP 稀疏点云（每格常只有 1 个点）用 1。
DEFAULT_MIN_POINTS = 3

_PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
              "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
              "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


def read_ply_vertices(path, stride=1):
    """通用 PLY：读顶点 x/y/z（任意数值类型）与三角面（若有 element face）。

    返回 {"xyz": float64 [N,3], "faces": int64 [M,3] 或 None, "n_vertices": 文件内顶点数}。
    其它顶点属性（颜色/法线）忽略；stride>1 时每 stride 个顶点取 1（有 faces 时不允许抽稀）。
    支持 binary_little_endian / binary_big_endian / ascii；面列表只接受三角形（LiDAR 网格导出
    都是三角面；四边形请先三角化）。
    """
    stride = int(stride)
    if stride < 1:
        raise ValueError("stride 必须 >= 1")
    elements = []          # [(name, count, [(prop_name, dtype_str) | ("list", count_t, item_t, name)])]
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} 不是 PLY")
        fmt = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path} PLY 头不完整")
            parts = line.decode("ascii", "replace").split()
            if not parts or parts[0] == "comment" or parts[0] == "obj_info":
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                elements.append((parts[1], int(parts[2]), []))
            elif parts[0] == "property":
                if parts[1] == "list":
                    elements[-1][2].append(("list", _PLY_TYPES[parts[2]], _PLY_TYPES[parts[3]], parts[4]))
                else:
                    elements[-1][2].append((parts[2], _PLY_TYPES[parts[1]]))
            elif parts[0] == "end_header":
                break
        if fmt not in ("binary_little_endian", "binary_big_endian", "ascii"):
            raise ValueError(f"{path}: 不支持的 PLY 格式 {fmt!r}")
        endian = ">" if fmt == "binary_big_endian" else "<"
        xyz, faces, n_vertices = None, None, 0
        for name, count, props in elements:
            if name == "vertex":
                n_vertices = count
                names = [p[0] for p in props]
                if any(p[0] == "list" for p in props) or not {"x", "y", "z"} <= set(names):
                    raise ValueError(f"{path}: vertex 元素必须有标量 x/y/z")
                dt = np.dtype([(p[0], endian + p[1]) for p in props])
                if fmt == "ascii":
                    rows = np.loadtxt(_ascii_lines(f, count), dtype=np.float64, ndmin=2)
                    if rows.shape[0] != count:
                        raise ValueError(f"{path}: ascii 顶点行数与头不一致")
                    cols = [names.index(k) for k in ("x", "y", "z")]
                    xyz = rows[::stride][:, cols]
                else:
                    table = np.fromfile(f, dtype=dt, count=count)
                    if len(table) != count:
                        raise ValueError(f"{path}: 顶点数据不足（文件截断？）")
                    table = table[::stride]
                    xyz = np.stack([table["x"], table["y"], table["z"]], axis=1).astype(np.float64)
            elif name == "face":
                lists = [p for p in props if p[0] == "list"]
                if len(props) != 1 or not lists:
                    raise ValueError(f"{path}: 只支持单个 list 属性的 face 元素")
                _, count_t, item_t, _ = lists[0]
                if fmt == "ascii":
                    tri = []
                    for line in _ascii_lines(f, count):
                        v = line.split()
                        if int(v[0]) != 3:
                            raise ValueError(f"{path}: 面不是三角形（{v[0]} 顶点）")
                        tri.append([int(v[1]), int(v[2]), int(v[3])])
                    faces = np.asarray(tri, dtype=np.int64).reshape(-1, 3)
                else:
                    dt = np.dtype([("n", endian + count_t), ("idx", endian + item_t, (3,))])
                    table = np.fromfile(f, dtype=dt, count=count)
                    if len(table) != count or (table["n"] != 3).any():
                        raise ValueError(f"{path}: 面数据不足或含非三角形（只支持全三角网格）")
                    faces = table["idx"].astype(np.int64)
            else:
                raise ValueError(f"{path}: 不支持的元素 {name!r}")
        if xyz is None:
            raise ValueError(f"{path}: 没有 vertex 元素")
        if faces is not None:
            if stride != 1:
                raise ValueError("带 faces 的 PLY 不能 stride 抽稀（索引会错位）")
            if len(faces) and (faces.min() < 0 or faces.max() >= n_vertices):
                raise ValueError(f"{path}: 面索引越界")
    return {"xyz": xyz, "faces": faces, "n_vertices": n_vertices}


def _ascii_lines(f, count):
    out = []
    while len(out) < count:
        line = f.readline()
        if not line:
            raise ValueError("ascii PLY 数据行不足")
        if line.strip():
            out.append(line.decode("ascii", "replace"))
    return out


MESH_SAMPLE_MAX_POINTS = 200_000_000     # 撒点上限（约 4.8 GB float64），超过说明 spacing 太小或网格太大


def sample_mesh_surface(xyz, faces, spacing):
    """三角网格 -> 面上确定性格点撒点（含原顶点），相邻点间距 <= spacing（米）。

    每个三角形按 n = ceil(最长边 / spacing) 等分，取重心格点 {(i/n, j/n) : i + j <= n}
    （(n+1)(n+2)/2 个，含三个顶点和三条边），格点沿三条边方向的步长都 <= spacing；没有随机数，
    同输入同输出。以前是固定种子的随机重心撒点（每格期望 4 点，按泊松分布 e^-4 ≈ 2% 的格一个点
    都没有，R2 实测 20×1 m 墙 voxel 0.1 漏 1.9%、0.2 漏 3.75%）；格点撒法下轴对齐面在
    voxel = 2·spacing 时每个被穿过的体素至少 4 个点，漏格 0；斜面只在体素内的切片窄于 spacing 时
    才可能漏。只用于占据体素化，不是表面重建。
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if spacing <= 0:
        raise ValueError("spacing 必须为正")
    if len(faces) == 0:
        return xyz.copy()
    a, b, c = xyz[faces[:, 0]], xyz[faces[:, 1]], xyz[faces[:, 2]]
    longest = np.max([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                      np.linalg.norm(a - c, axis=1)], axis=0)
    ok = np.isfinite(longest)
    a, b, c = a[ok], b[ok], c[ok]
    n_div = np.maximum(np.ceil(longest[ok] / spacing).astype(np.int64), 1)
    total = int(((n_div + 1) * (n_div + 2) // 2).sum())
    if total > MESH_SAMPLE_MAX_POINTS:
        raise ValueError(f"网格撒点 {total} 个超过上限 {MESH_SAMPLE_MAX_POINTS}；放大 spacing/voxel_m 或裁剪网格")
    chunks = [xyz]
    for n in np.unique(n_div):
        sel = n_div == n
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = (i + j) <= n
        u, v = i[keep] / n, j[keep] / n                                   # 重心坐标格点 [K]
        pts = (a[sel][:, None, :] + u[None, :, None] * (b[sel] - a[sel])[:, None, :]
               + v[None, :, None] * (c[sel] - a[sel])[:, None, :])        # [M, K, 3]
        chunks.append(pts.reshape(-1, 3))
    return np.concatenate(chunks, axis=0)


def load_body_envelope(path=None):
    """读 E1 MINI collision_envelope -> {"radius_m","half_height_m","source"}。

    键：collision_envelope.value.radius_from_center / z_min / z_max（米）；默认文件 E1_MINI_PARAMS。
    """
    cand = path or E1_MINI_PARAMS
    if not os.path.isfile(cand):
        raise FileNotFoundError(
            f"找不到 E1 MINI 参数 {cand}；也可在 scene.json collision.envelope 里显式给 radius_m/half_height_m")
    with open(cand, "r", encoding="utf-8") as f:
        doc = json.load(f)
    env = doc.get("collision_envelope")
    val = env.get("value") if isinstance(env, dict) else None
    if not isinstance(val, dict):
        raise ValueError(f"{cand}: 缺 collision_envelope.value")
    radius = float(val["radius_from_center"])
    half_h = max(abs(float(val["z_min"])), abs(float(val["z_max"])))
    if not (radius > 0 and half_h > 0):
        raise ValueError(f"{cand}: collision_envelope 数值无效 {val}")
    return {"radius_m": radius, "half_height_m": half_h, "source": cand}


def quat_to_rotmat(q):
    """[N,4] (w,x,y,z) -> [N,3,3]；与 gsplat / INRIA PLY 的 rot_0..3 顺序一致。"""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = np.empty((len(q), 3, 3), dtype=np.float64)
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - w * z)
    r[:, 0, 2] = 2 * (x * z + w * y)
    r[:, 1, 0] = 2 * (x * y + w * z)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - w * x)
    r[:, 2, 0] = 2 * (x * z - w * y)
    r[:, 2, 1] = 2 * (y * z + w * x)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def _rigid_parts(nav_to_world):
    """4x4 -> (R, t)，只接受刚体变换（渲染端 viewmat_from_pose 也只按刚体用它）。"""
    m = np.asarray(nav_to_world, dtype=np.float64)
    if m.shape != (4, 4) or not np.isfinite(m).all():
        raise ValueError("nav_to_world 必须是有限 4x4")
    r = m[:3, :3]
    if not np.allclose(r @ r.T, np.eye(3), atol=1e-6) or np.linalg.det(r) < 0:
        raise ValueError("nav_to_world 的旋转块必须正交且 det=+1（不支持缩放/镜像）")
    return r, m[:3, 3]


class Occupancy:
    """导航系 3D 体素占据 + 地面；collides()/sweep() 查膨胀栅格，clearance() 查原始栅格。"""

    def __init__(self, means, scales=None, quats=None, opacities=None, *,
                 nav_to_world=None, z_range=(1.65, 2.35), voxel_m=0.1,
                 body_radius_m=0.102, body_half_height_m=0.028, inflate_m=0.0,
                 opacity_min=DEFAULT_OPACITY_MIN, max_scale_m=DEFAULT_MAX_SCALE_M,
                 sigma_k=DEFAULT_SIGMA_K, ground_z=0.0, pad_m=1.0,
                 min_points=1, source="gaussians", collision_model=DEFAULT_COLLISION_MODEL):
        if collision_model not in COLLISION_MODELS:
            raise ValueError(f"collision_model 必须是 {COLLISION_MODELS} 之一，得到 {collision_model!r}")
        pts = np.asarray(means, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError("means 必须是 Nx3")
        if not (z_range[0] < z_range[1]) or voxel_m <= 0 or inflate_m < 0:
            raise ValueError("z_range/voxel/inflate 参数无效")
        if not (body_radius_m > 0 and body_half_height_m > 0 and sigma_k > 0):
            raise ValueError("机体包络与 sigma_k 必须为正")
        if int(min_points) < 1 or (int(min_points) > 1 and scales is not None):
            raise ValueError("min_points 必须 >= 1，且只对纯点输入（scales=None）有意义")
        self.z_range = (float(z_range[0]), float(z_range[1]))
        self.voxel = float(voxel_m)
        self.body_radius = float(body_radius_m)
        self.body_half_height = float(body_half_height_m)
        self.inflate = float(inflate_m)
        self.opacity_min = float(opacity_min)
        self.max_scale = None if max_scale_m is None else float(max_scale_m)
        self.sigma_k = float(sigma_k)
        self.ground_z = float(ground_z)
        self.min_points = int(min_points)
        self.source = str(source)
        self.collision_model = collision_model
        self.proxy_version = (TRUTH_VERSION.get(self.source, PROXY_VERSION)
                              + COLLISION_VERSION_SUFFIX[collision_model])
        # cruise_z = 名义出生高度（from_scene 取 scene.spawn.z），供 v2 的规划硬栅格缺省取层包络；v1 不用
        self.z_envelope_down = Z_ENVELOPE_DOWN_M
        self.z_envelope_up = Z_ENVELOPE_UP_M
        self.spawn_drift_margin = SPAWN_DRIFT_MARGIN_M
        self.cruise_z = None
        self.n_points = len(pts)
        self.n_kept = 0
        self.n_in_range = 0
        self.n_occupied = 0
        pts = pts[:, :3]

        # (a) 世界 -> 导航：m_nav = R^T (m_world - t)，高斯朝向 R_nav = R^T R_g。
        r_m, t_m = _rigid_parts(np.eye(4) if nav_to_world is None else nav_to_world)
        pts = (pts - t_m) @ r_m
        keep = np.isfinite(pts).all(axis=1)
        if opacities is not None:
            op = np.asarray(opacities, dtype=np.float64).reshape(-1)
            keep &= op >= self.opacity_min                       # (b) 幽灵墙
        if scales is not None:
            sc = np.asarray(scales, dtype=np.float64)
            keep &= np.isfinite(sc).all(axis=1)
            if self.max_scale is not None:
                keep &= sc.max(axis=1) <= self.max_scale         # (b) 巨型天空高斯
        self.n_kept = int(keep.sum())
        pts = pts[keep]
        if scales is None:
            half = np.zeros_like(pts)                            # 纯点：只占中心体素
            rot = None
            sc = None
        else:
            # shortcut: 薄高斯（墙面/地面 σ 常只有 1 cm）按最小半厚 0.6 体素处理，否则
            # 体素中心采样会漏掉整面墙；代价是薄墙最多被画成 1 体素厚。
            sc = np.maximum(sc[keep], 0.6 * self.voxel / self.sigma_k)
            q = (np.tile([1.0, 0, 0, 0], (len(pts), 1)) if quats is None
                 else np.asarray(quats, dtype=np.float64)[keep])
            rot = np.einsum("ji,njk->nik", r_m, quat_to_rotmat(q))  # R_m^T R_g
            # 轴对齐半宽 e_i = k*sqrt(Σ_ii)，Σ = R diag(s²) R^T。
            half = self.sigma_k * np.sqrt(np.einsum("nij,nj->ni", rot ** 2, sc ** 2))
        # 只留 z 方向伸进栅格高度范围的高斯。
        zlo, zhi = self.z_range
        in_range = (pts[:, 2] + half[:, 2] >= zlo) & (pts[:, 2] - half[:, 2] <= zhi)
        self.n_in_range = int(in_range.sum())
        if self.n_in_range == 0:
            self._empty()
            return
        pts, half = pts[in_range], half[in_range]
        if rot is not None:
            rot, sc = rot[in_range], sc[in_range]
        lo = (pts - half)[:, :2].min(axis=0) - pad_m
        hi = (pts + half)[:, :2].max(axis=0) + pad_m
        self.origin = np.array([lo[0], lo[1], zlo])
        nxy = np.ceil((hi - lo) / self.voxel).astype(int) + 1
        nz = max(1, int(math.ceil((zhi - zlo) / self.voxel - 1e-9)))
        self.shape = (int(nxy[0]), int(nxy[1]), nz)
        if self.shape[0] * self.shape[1] * self.shape[2] > 40_000_000:
            raise ValueError(f"占据栅格过大 {self.shape}；提高 voxel_m、缩小 z_range 或裁剪场景")
        occ = np.zeros(self.shape, dtype=bool)
        self._rasterize(occ, pts, half, rot, sc)
        self.raw = occ
        self.n_occupied = int(occ.sum())
        ii, jj = np.nonzero(occ.any(axis=2))
        self._occ_xy = self.origin[:2] + (np.stack([ii, jj], 1) + 0.5) * self.voxel
        self.grid = self._dilate(occ)

    # -- 建栅格 ------------------------------------------------------------
    def _cell(self, p):
        return np.floor((np.asarray(p, dtype=np.float64) - self.origin) / self.voxel).astype(int)

    def _mark(self, occ, cells):
        shape = np.array(self.shape)
        ok = (cells >= 0).all(axis=1) & (cells < shape).all(axis=1)
        cells = cells[ok]
        occ[cells[:, 0], cells[:, 1], cells[:, 2]] = True

    def _rasterize(self, occ, pts, half, rot, sc):
        """(c) 椭球落体素：对每个高斯轴对齐包围盒内的体素中心做马氏距离 <= k 测试。

        均值所在体素总是占据（再小的高斯也至少占一格）。纯点输入（rot None）按体素计数，
        点数 >= min_points 的体素才占据（v3 真值点云的噪点过滤）。
        """
        c0 = self._cell(pts)
        if rot is None:
            if self.min_points > 1:
                shape = np.array(self.shape)
                ok = (c0 >= 0).all(axis=1) & (c0 < shape).all(axis=1)
                flat = np.ravel_multi_index(c0[ok].T, self.shape)
                uniq, cnt = np.unique(flat, return_counts=True)
                occ.reshape(-1)[uniq[cnt >= self.min_points]] = True
            else:
                self._mark(occ, c0)
            return
        self._mark(occ, c0)
        ncell = np.ceil(half / self.voxel - 1e-9).astype(int)    # 每轴半宽体素数
        nmax = ncell.max(axis=1)
        inv_s = 1.0 / sc
        k2 = self.sigma_k ** 2
        nz = self.shape[2]
        # 小高斯（绝大多数）：按 nmax 分桶，整桶用同一张偏移表做向量化测试。
        n_small = 6
        for n in range(1, n_small + 1):
            idx = np.nonzero(nmax == n)[0]
            if len(idx) == 0:
                continue
            rng = np.arange(-n, n + 1)
            rz = np.arange(-min(n, nz - 1), min(n, nz - 1) + 1)   # z 层数有限，偏移表裁掉
            off = np.stack(np.meshgrid(rng, rng, rz, indexing="ij"), -1).reshape(-1, 3)
            chunk = max(1, 4_000_000 // len(off))
            for s in range(0, len(idx), chunk):
                sel = idx[s:s + chunk]
                cells = c0[sel][:, None, :] + off[None, :, :]            # [G,P,3]
                centers = self.origin + (cells + 0.5) * self.voxel
                d = centers - pts[sel][:, None, :]
                local = np.einsum("nji,npj->npi", rot[sel], d) * inv_s[sel][:, None, :]
                self._mark(occ, cells[(local ** 2).sum(-1) <= k2])
        # 大高斯（少数：墙面、地面大片）：逐个按自己的包围盒（裁到栅格）测试。
        for g in np.nonzero(nmax > n_small)[0]:
            lo = np.maximum(c0[g] - ncell[g], 0)
            hi = np.minimum(c0[g] + ncell[g], np.array(self.shape) - 1)
            if (hi < lo).any():
                continue
            ax = [np.arange(lo[i], hi[i] + 1) for i in range(3)]
            cells = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
            centers = self.origin + (cells + 0.5) * self.voxel
            local = (centers - pts[g]) @ rot[g] * inv_s[g]
            self._mark(occ, cells[(local ** 2).sum(-1) <= k2])

    def _dilate(self, occ):
        """(d) 固定圆柱结构元素：xy 半径 = 机体半径+余量，z 半高 = 机体半高；纯 numpy。

        体素数按四舍五入：占据体素本身已含 ±半格，再 ceil 会把 E1 的 0.102 m 在
        0.1 m 体素下翻倍成 0.2 m。

        这是 v1 的结构元素。r_cells = 1 时 di² + dj² <= 1 只留 4 个轴向邻格，斜向邻格（中心离原始体素边角 0.0707 m）
        不膨胀（审核 B 发现 1）；collision_model="v2" 改走 _dilate_euclid。
        """
        if self.collision_model == "v2":
            return self._dilate_euclid(occ)
        r_cells = int(round((self.body_radius + self.inflate) / self.voxel))
        h_cells = int(round(self.body_half_height / self.voxel))
        self.dilation_cells = (r_cells, h_cells)
        out = occ.copy()
        nx, ny, nz = occ.shape
        for di in range(-r_cells, r_cells + 1):
            for dj in range(-r_cells, r_cells + 1):
                if di * di + dj * dj > r_cells * r_cells:
                    continue
                for dk in range(-h_cells, h_cells + 1):
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    src = occ[max(0, -di):nx - max(0, di),
                              max(0, -dj):ny - max(0, dj),
                              max(0, -dk):nz - max(0, dk)]
                    out[max(0, di):nx - max(0, -di),
                        max(0, dj):ny - max(0, -dj),
                        max(0, dk):nz - max(0, -dk)] |= src
        return out

    def _dilate_euclid(self, occ):
        """v2 的规划格中心膨胀栅格（B1 / J-2）：xy 偏移 (di, dj) 保留 <=> 该格中心到原始占据体素（边长 1 格的方块）的欧氏距离
        <= 机体半径 + 余量；z 方向不膨胀。

        注意（审核 J-2，卡 S1b 第二轮定稿）：这张 self.grid 只供离散格中心上的 A* 规划硬栅格（planning_grid）用
        （格中心处与精确距离恒等）；连续坐标查询（collides / sweep / sweep_path / 规则臂 occ_columns）在 _grid_hit 里
        不查 self.grid，而是查点到原始占据体素方块的精确水平欧氏距离（_exact_xy_hit），消除 0.1 m 体素 3×3 方核斜向多判
        3.94 cm 与 0.2 m 体素十字核斜向漏判 10.2 cm / 轴向多判 9.8 cm 的误差。
        """
        r = (self.body_radius + self.inflate) / self.voxel
        reach = int(math.floor(r + 0.5 + 1e-9))
        self.dilation_cells = (reach, 0)
        out = occ.copy()
        nx, ny, _ = occ.shape
        for di in range(-reach, reach + 1):
            for dj in range(-reach, reach + 1):
                gx, gy = max(abs(di) - 0.5, 0.0), max(abs(dj) - 0.5, 0.0)
                if gx * gx + gy * gy > r * r + 1e-9:
                    continue
                if di == 0 and dj == 0:
                    continue
                out[max(0, di):nx - max(0, -di), max(0, dj):ny - max(0, -dj), :] |= \
                    occ[max(0, -di):nx - max(0, di), max(0, -dj):ny - max(0, dj), :]
        self._prepare_exact_xy(occ)
        return out

    def _prepare_exact_xy(self, occ):
        """预计算连续点到原始占据体素方块精确水平距离查询的粗筛表（J-2）：
        - _exact_inner：格内任意点到某占据方块的最大可能距离 <= R（落入即必碰，O(1) 返回 True）；
        - _exact_shell：格内至少存在一点到某占据方块的最小可能距离 <= R，但未必碰（只对这一薄壳里的候选点算精确距离）。
        """
        r = (self.body_radius + self.inflate) / self.voxel
        reach = int(math.ceil(r + 1e-12))
        nx, ny, _ = occ.shape
        inner = occ.copy()
        outer = occ.copy()
        shell_offs, all_offs = [], []
        for di in range(-reach, reach + 1):
            for dj in range(-reach, reach + 1):
                min_d2 = max(abs(di) - 1, 0) ** 2 + max(abs(dj) - 1, 0) ** 2
                if min_d2 > r * r + 1e-12:
                    continue
                all_offs.append((di, dj))
                max_d2 = abs(di) ** 2 + abs(dj) ** 2
                if di != 0 or dj != 0:
                    src = occ[max(0, -di):nx - max(0, di), max(0, -dj):ny - max(0, dj), :]
                    outer[max(0, di):nx - max(0, -di), max(0, dj):ny - max(0, -dj), :] |= src
                    if max_d2 <= r * r - 1e-12:
                        inner[max(0, di):nx - max(0, -di), max(0, dj):ny - max(0, -dj), :] |= src
                if max_d2 > r * r - 1e-12 and (di != 0 or dj != 0):
                    shell_offs.append((di, dj))
        self._exact_inner = inner
        self._exact_shell = outer & ~inner
        self._exact_shell_offsets = np.asarray(shell_offs, dtype=np.int64).reshape(-1, 2)
        self._exact_all_offsets = np.asarray(all_offs, dtype=np.int64).reshape(-1, 2)
        self._exact_reach = reach

    def _empty(self):
        self.shape = (0, 0, 0)
        self.origin = np.zeros(3)
        self.raw = self.grid = np.zeros((0, 0, 0), dtype=bool)
        self._occ_xy = np.zeros((0, 2))
        self.dilation_cells = (0, 0)
        self._exact_inner = self._exact_shell = np.zeros((0, 0, 0), dtype=bool)
        self._exact_shell_offsets = self._exact_all_offsets = np.zeros((0, 2), dtype=np.int64)
        self._exact_reach = 0

    def _exact_xy_hit(self, p_xy, k0, k1, radius_m=None, strict_lt=False):
        """p_xy:[M,2] 在各自层区间 [k0, k1] 内，到原始占据体素方块的水平欧氏距离是否 <= radius_m（strict_lt=True 时判 <）。"""
        p_xy = np.asarray(p_xy, dtype=np.float64).reshape(-1, 2)
        hit = np.zeros(len(p_xy), dtype=bool)
        if self.raw.size == 0 or len(p_xy) == 0:
            return hit
        v = self.voxel
        nx, ny, _ = self.raw.shape
        finite = np.isfinite(p_xy).all(axis=1) & (k0 <= k1)
        f_idx = np.nonzero(finite)[0]
        if len(f_idx) == 0:
            return hit
        c_xy = np.floor((p_xy[f_idx] - self.origin[:2]) / v).astype(int)

        def check_subset(sub_f_idx, sub_c, offs, thr2, strict):
            if len(sub_f_idx) == 0 or len(offs) == 0:
                return
            ni = sub_c[:, 0:1] + offs[None, :, 0]
            nj = sub_c[:, 1:2] + offs[None, :, 1]
            ok = (ni >= 0) & (ni < nx) & (nj >= 0) & (nj < ny)
            bx0 = self.origin[0] + ni * v
            by0 = self.origin[1] + nj * v
            px, py = p_xy[sub_f_idx, 0:1], p_xy[sub_f_idx, 1:2]
            dx = np.maximum(np.maximum(bx0 - px, 0.0), px - (bx0 + v))
            dy = np.maximum(np.maximum(by0 - py, 0.0), py - (by0 + v))
            d2 = dx * dx + dy * dy
            in_r = ok & ((d2 < thr2) if strict else (d2 <= thr2))
            mm, kk_off = np.nonzero(in_r)
            if len(mm):
                nni, nnj = ni[mm, kk_off], nj[mm, kk_off]
                sa, sb = k0[sub_f_idx][mm], k1[sub_f_idx][mm]
                occ_z = self.raw[nni, nnj, sa].copy()
                max_d = int((sb - sa).max())
                for d in range(1, max_d + 1):
                    lv = sa + d <= sb
                    if lv.any():
                        occ_z[lv] |= self.raw[nni[lv], nnj[lv], (sa + d)[lv]]
                if occ_z.any():
                    hit[sub_f_idx[mm[occ_z]]] = True

        if radius_m is None and not strict_lt:
            reach = self._exact_reach
            r_tot = self.body_radius + self.inflate
            r2 = r_tot * r_tot + 1e-18
            in_b = (c_xy[:, 0] >= 0) & (c_xy[:, 0] < nx) & (c_xy[:, 1] >= 0) & (c_xy[:, 1] < ny)
            if in_b.all():
                ib_idx, ic = f_idx, c_xy
                ia, ib = k0[ib_idx], k1[ib_idx]
                c0, c1 = ic[:, 0], ic[:, 1]
                sure = self._exact_inner[c0, c1, ia].copy()
                maybe = self._exact_shell[c0, c1, ia].copy()
                for d in range(1, int((ib - ia).max()) + 1):
                    lv = ia + d <= ib
                    kk = (ia + d)[lv]
                    sure[lv] |= self._exact_inner[c0[lv], c1[lv], kk]
                    maybe[lv] |= self._exact_shell[c0[lv], c1[lv], kk]
                hit[ib_idx[sure]] = True
                sh = maybe & ~sure
                if sh.any():
                    check_subset(ib_idx[sh], ic[sh], self._exact_shell_offsets, r2, False)
                return hit
            near = ((c_xy[:, 0] >= -reach) & (c_xy[:, 0] < nx + reach)
                    & (c_xy[:, 1] >= -reach) & (c_xy[:, 1] < ny + reach))
            if not near.any():
                return hit
            idx, c_xy = f_idx[near], c_xy[near]
            in_b = (c_xy[:, 0] >= 0) & (c_xy[:, 0] < nx) & (c_xy[:, 1] >= 0) & (c_xy[:, 1] < ny)
            if in_b.any():
                ib_idx, ic = idx[in_b], c_xy[in_b]
                ia, ib = k0[ib_idx], k1[ib_idx]
                sure = np.zeros(len(ib_idx), dtype=bool)
                maybe = np.zeros(len(ib_idx), dtype=bool)
                for d in range(int((ib - ia).max()) + 1):
                    lv = ia + d <= ib
                    kk = (ia + d)[lv]
                    sure[lv] |= self._exact_inner[ic[lv, 0], ic[lv, 1], kk]
                    maybe[lv] |= self._exact_shell[ic[lv, 0], ic[lv, 1], kk]
                hit[ib_idx[sure]] = True
                sh = maybe & ~sure
                if sh.any():
                    check_subset(ib_idx[sh], ic[sh], self._exact_shell_offsets, r2, False)
            if (~in_b).any():
                check_subset(idx[~in_b], c_xy[~in_b], self._exact_all_offsets, r2, False)
            return hit

        r_tot = (self.body_radius + self.inflate) if radius_m is None else float(radius_m)
        r = r_tot / v
        reach = int(math.ceil(r + 1e-12))
        near = ((c_xy[:, 0] >= -reach) & (c_xy[:, 0] < nx + reach)
                & (c_xy[:, 1] >= -reach) & (c_xy[:, 1] < ny + reach))
        if not near.any():
            return hit
        offs = np.asarray(
            [(di, dj) for di in range(-reach, reach + 1) for dj in range(-reach, reach + 1)
             if max(abs(di) - 1, 0) ** 2 + max(abs(dj) - 1, 0) ** 2 <= r * r + 1e-12],
            dtype=np.int64,
        ).reshape(-1, 2)
        thr2 = (max(r_tot - 1e-12, 0.0) ** 2) if strict_lt else (r_tot * r_tot + 1e-18)
        check_subset(f_idx[near], c_xy[near], offs, thr2, strict_lt)
        return hit

    # -- 查询 ---------------------------------------------------------------
    def _interval_layers(self, z_lo, z_hi):
        """区间 [z_lo, z_hi] 与栅格各层 [origin_z + k·v, origin_z + (k+1)·v] 相交的层下标范围 (k0, k1)；
        完全在栅格上方或下方、或非有限数时 k0 > k1（空）。

        边界约定：竖直方向两端各向内缩 1e-12 m，区间只贴着层面（只差浮点误差）不算碰到那一层；水平方向 _exact_xy_hit 则是
        距离 <= 半径算碰（贴着也算）。两处的约定只在测度为零的边界上不同。"""
        zlo_g, v = float(self.origin[2]), self.voxel
        lo = np.nan_to_num(np.floor((np.asarray(z_lo, dtype=np.float64) - zlo_g) / v + 1e-12), nan=1e9, posinf=1e9, neginf=-1e9)
        hi = np.nan_to_num(np.ceil((np.asarray(z_hi, dtype=np.float64) - zlo_g) / v - 1e-12) - 1.0, nan=-1e9, posinf=1e9, neginf=-1e9)
        return np.maximum(lo, 0).astype(int), np.minimum(hi, self.shape[2] - 1).astype(int)

    def layer_span(self, z):
        """连续碰撞判定的竖直层范围（J-1 定稿）：机体高度区间 [z − body_half_height, z + body_half_height]（无余量）碰到的层 (k0, k1)。"""
        z = np.asarray(z, dtype=np.float64)
        return self._interval_layers(z - self.body_half_height, z + self.body_half_height)

    def envelope_layer_span(self, z):
        """规划硬栅格与出生检查按名义出生高度 z 取的整集高度包络层范围（J-1 定稿）：
        [z − z_envelope_down − body_half_height, z + z_envelope_up + body_half_height] 碰到的层 (k0, k1)。"""
        z = np.asarray(z, dtype=np.float64)
        return self._interval_layers(z - self.z_envelope_down - self.body_half_height,
                                     z + self.z_envelope_up + self.body_half_height)

    def planning_grid(self, z=None):
        """教师规划的 2D 硬栅格（B4 / J-1）。v1：各层取并（不看 z，改动前的做法）；v2：名义高度 z 处整集高度包络层的并（self.grid
        已按 _dilate_euclid 做格中心欧氏膨胀），z 缺省取 cruise_z（from_scene 取 scene.spawn.z），都没有就报错，不猜高度。"""
        if self.collision_model != "v2":
            return self.grid.any(axis=2)
        if z is None:
            z = self.cruise_z
        if z is None:
            raise ValueError("collision_model='v2' 的规划硬栅格要知道名义出生高度：给 z，或经 from_scene 建占据（取 scene.spawn.z）")
        if self.grid.size == 0:
            return np.zeros(self.shape[:2], dtype=bool)
        k0, k1 = (int(k) for k in self.envelope_layer_span(float(z)))
        if k0 > k1:
            return np.zeros(self.shape[:2], dtype=bool)
        return self.grid[:, :, k0:k1 + 1].any(axis=2)

    def spawn_ok(self, x, y, z):
        """v2 的出生候选点检查（J-1）：在名义高度 z 的整集包络层上，候选点到原始占据体素方块的精确水平距离不小于
        body_radius + inflate + spawn_drift_margin，且 z 不低于地面包络；v1 等价于 not self.collides(x, y, z)。"""
        if self.collision_model != "v2":
            return not self.collides(x, y, z)
        if bool(self._ground_hit(float(z) - self.z_envelope_down)):
            return False
        if self.raw.size == 0:
            return True
        k0, k1 = self.envelope_layer_span(np.array([float(z)]))
        if int(k0[0]) > int(k1[0]):
            return True
        need = self.body_radius + self.inflate + self.spawn_drift_margin
        return not bool(self._exact_xy_hit(np.array([[float(x), float(y)]]), k0, k1, radius_m=need, strict_lt=True)[0])

    def _grid_hit(self, p):
        """p:[M,3] -> [M] bool。

        v1：点所在的那一层查膨胀栅格 self.grid；栅格外 / 高度范围外为 False（z 在层边界上差 1 cm 就换一层，审核 B 发现 2）。
        v2：竖直方向取 layer_span(z) 覆盖到的层，水平方向取点到原始占据体素方块的精确欧氏距离 <= 机体半径 + 余量（审核 J-2）。
        """
        p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
        hit = np.zeros(len(p), dtype=bool)
        if self.grid.size == 0:
            return hit
        if self.collision_model == "v2":
            k0, k1 = self.layer_span(p[:, 2])
            return self._exact_xy_hit(p[:, :2], k0, k1)
        inz = (p[:, 2] >= self.z_range[0]) & (p[:, 2] < self.z_range[1])
        c = self._cell(p)
        ok = inz & (c >= 0).all(axis=1) & (c < np.array(self.shape)).all(axis=1)
        hit[ok] = self.grid[c[ok, 0], c[ok, 1], c[ok, 2]]
        return hit

    def _ground_hit(self, z):
        return np.asarray(z, dtype=np.float64) - self.body_half_height <= self.ground_z

    def collides(self, x, y, z=2.0):
        """机体包络（圆柱）碰到占据体素，或离地低于机体半高。v1：z 在高度范围外只查地面；v2：层带与栅格不相交时只查地面。"""
        if self._ground_hit(z):
            return True
        return bool(self._grid_hit([[x, y, z]])[0])

    def _segment_samples(self, p0, p1):
        """线段 p0->p1 上按 <= 半体素步距的采样点 [n, 3]（含两端）。"""
        n = int(math.ceil(np.linalg.norm(p1 - p0) / (0.5 * self.voxel))) + 1
        ts = np.linspace(0.0, 1.0, max(2, n))
        return p0[None, :] + ts[:, None] * (p1 - p0)[None, :]

    def sweep(self, p0, p1):
        """(e) 沿线段 p0->p1 按 <= 半体素步距采样；返回 (是否碰, 首个碰撞点或 None)。

        shortcut: 动作窗口内的真实轨迹是弧线（偏航+前进过渡），这里按直线段近似。hold 0.5 s、位移 <= 0.7 m 时弦与弧的偏差
        不到体素；hold 1.0 s 的 cf_sitl 实测弦外横偏 p90 0.0415 m、max 0.0679 m（审核 B 发现 3，trace_vs_chord.py），
        已不比半个体素小。升级条件：窗口内有中间轨迹点（cf_sitl 的 trace，每 0.1 s 一个）时用 sweep_path，
        collision_model="v2" 的 collect_3dgs.run_episode 就是这样做的。
        """
        pts = self._segment_samples(np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64))
        hit = self._ground_hit(pts[:, 2]) | self._grid_hit(pts)
        if not hit.any():
            return False, None
        return True, pts[int(np.argmax(hit))].tolist()

    def sweep_path(self, points):
        """（B3）沿折线 points（[N, 3]，按顺序）逐段 sweep，返回 (是否碰, 沿折线首个碰撞点或 None)。

        每段的采样与 sweep 完全相同（拼起来再一次查栅格），所以与「按段调 sweep 取第一个命中」逐位一致；
        N = 1 只查那一个点，N = 0 返回 (False, None)。
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(pts) == 0:
            return False, None
        if len(pts) == 1:
            pts = np.repeat(pts, 2, axis=0)
        samples = np.concatenate([self._segment_samples(a, b) for a, b in zip(pts[:-1], pts[1:])])
        hit = self._ground_hit(samples[:, 2]) | self._grid_hit(samples)
        if not hit.any():
            return False, None
        return True, samples[int(np.argmax(hit))].tolist()

    def clearance(self, x, y, z=2.0):
        """到最近**原始**占据体素中心的水平距离（米，未减机体半径）与离地高度取小。"""
        ground = float(z - self.ground_z)
        if len(self._occ_xy) == 0:
            return ground
        d = float(np.min(np.hypot(self._occ_xy[:, 0] - x, self._occ_xy[:, 1] - y)))
        return float(min(ground, d))

    def summary(self):
        if self.source == "gaussians":
            note = ("approximate proxy: opacity-filtered Gaussian ellipsoids in nav frame, "
                    "swept along segments; not collision ground truth")
        else:
            note = (f"metric {self.source} geometry voxelised in nav frame (min_points={self.min_points}), "
                    "swept along segments; sensor-grade truth, not a physics contact")
        out = {"proxy_version": self.proxy_version, "collision_model": self.collision_model, "source": self.source,
                "geometry": getattr(self, "geometry_source", None),
                "n_points": self.n_points, "n_kept": self.n_kept,
                "n_in_z_range": self.n_in_range, "n_occupied_cells": self.n_occupied,
                "z_range": list(self.z_range), "voxel_m": self.voxel,
                "body_radius_m": self.body_radius,
                "body_half_height_m": self.body_half_height,
                "envelope_source": getattr(self, "envelope_source", None),
                "margin_m": self.inflate, "dilation_cells": list(self.dilation_cells),
                "opacity_min": self.opacity_min, "max_scale_m": self.max_scale,
                "sigma_k": self.sigma_k, "min_points": self.min_points, "ground_z": self.ground_z,
                "note": note}
        if self.collision_model == "v2":
            out["z_envelope_down_m"] = self.z_envelope_down
            out["z_envelope_up_m"] = self.z_envelope_up
            out["spawn_drift_margin_m"] = self.spawn_drift_margin
            out["cruise_z"] = self.cruise_z
        return out


def resolve_envelope(spec):
    """"e1_mini" | "x3" | {"radius_m","half_height_m"} -> 机体包络 dict。"""
    if spec in (None, "e1_mini"):
        return load_body_envelope()
    if spec == "x3":
        return dict(X3_ENVELOPE)
    if isinstance(spec, dict) and "radius_m" in spec and "half_height_m" in spec:
        return {"radius_m": float(spec["radius_m"]),
                "half_height_m": float(spec["half_height_m"]),
                "source": spec.get("source", "scene.json collision.envelope")}
    raise ValueError(f"collision.envelope 无效: {spec!r}")


def from_points(points, *, nav_to_world=None, z_range=(1.65, 2.35), voxel_m=0.1,
                body_radius_m=0.102, body_half_height_m=0.028, inflate_m=0.0,
                min_points=DEFAULT_MIN_POINTS, ground_z=0.0, source="points",
                collision_model=DEFAULT_COLLISION_MODEL):
    """米制点云（世界系，LiDAR / COLMAP points3D）-> 体素占据真值；接口与高斯代理同。"""
    return Occupancy(points, None, None, None, nav_to_world=nav_to_world, z_range=z_range,
                     voxel_m=voxel_m, body_radius_m=body_radius_m,
                     body_half_height_m=body_half_height_m, inflate_m=inflate_m,
                     max_scale_m=None, ground_z=ground_z, min_points=min_points, source=source,
                     collision_model=collision_model)


def from_mesh(xyz, faces, *, voxel_m=0.1, **kw):
    """三角网格（世界系，米）-> 面上按半体素间距确定性撒点 -> from_points；min_points 固定 1
    （撒点密度由我们控制，不存在噪点）。"""
    pts = sample_mesh_surface(xyz, faces, spacing=0.5 * float(voxel_m))
    kw.pop("min_points", None)
    return from_points(pts, voxel_m=voxel_m, min_points=1, source="mesh", **kw)


def from_scene(scene, envelope=None, gaussians=None, source=None, collision_model=DEFAULT_COLLISION_MODEL):
    """按 scene.json 的 collision 段建占据；机体包络默认 E1 MINI（e1_mini.json）。

    collision_model（S1b）：只由调用方显式给，不进 scene.json（场景文件是各实验共用的数据）；缺省 v1 = 改动前的口径。

    collision 段键：z_slice（栅格高度范围）、voxel_m、margin_m（机体半径外的余量，默认 0）、
    opacity_min、sigma_k、envelope（"e1_mini" | "x3" | {"radius_m","half_height_m"}）、
    max_scale_m（默认跟 render.max_scale_m 同值）；
    v3：source（"gaussians" 默认 | "points" | "mesh"）、geometry（点云/网格 PLY，相对 scene.json
    目录；与 3DGS 同一世界系，nav_to_world 一并作用）、min_points（points 的体素计数阈值，默认 3）、
    geometry_stride（点云抽稀步长，默认 1）。参数 source 覆盖 scene.json，对照脚本用。
    v1 的 inflate_m / drone_radius_m 把机体尺寸混在膨胀里，v2 一律忽略并提示。
    gaussians 可传 read_gaussian_ply 的结果，避免渲染端已读过的 511 MB PLY 再读一遍。
    """
    from . import scene as scene_mod
    c = dict(scene.get("collision", {}))
    legacy = {k: c[k] for k in ("inflate_m", "drone_radius_m") if k in c}
    if legacy:
        print(f"collision: 忽略 v1 键 {legacy}；v2 机体包络来自 collision.envelope"
              "（默认 E1 MINI），余量用 margin_m", file=sys.stderr)
    env = resolve_envelope(envelope if envelope is not None else c.get("envelope"))
    src = source if source is not None else c.get("source", "gaussians")
    if src not in ("gaussians", "points", "mesh"):
        raise ValueError(f"collision.source 无效: {src!r}（gaussians | points | mesh）")
    common = dict(nav_to_world=scene.get("_nav_to_world"),
                  z_range=c.get("z_slice", (1.65, 2.35)), voxel_m=c.get("voxel_m", 0.1),
                  body_radius_m=env["radius_m"], body_half_height_m=env["half_height_m"],
                  inflate_m=c.get("margin_m", 0.0), ground_z=scene.get("ground_z", 0.0),
                  collision_model=collision_model)
    if src == "gaussians":
        g = gaussians or scene_mod.read_gaussian_ply(scene["ply_path"])
        occ = Occupancy(g["means"], g["scales"], g["quats"], g["opacities"],
                        opacity_min=c.get("opacity_min", DEFAULT_OPACITY_MIN),
                        max_scale_m=c.get("max_scale_m",
                                          scene_mod.render_options(scene)["max_scale_m"]),
                        sigma_k=c.get("sigma_k", DEFAULT_SIGMA_K), **common)
        occ.geometry_source = scene.get("ply")
    else:
        if not c.get("geometry"):
            raise ValueError(f"collision.source={src} 需要 collision.geometry（点云/网格 PLY 路径）")
        path = os.path.join(scene["dir"], c["geometry"])
        if src == "points":
            ply = read_ply_vertices(path, stride=int(c.get("geometry_stride", 1)))
            occ = from_points(ply["xyz"], min_points=int(c.get("min_points", DEFAULT_MIN_POINTS)), **common)
        else:
            ply = read_ply_vertices(path)
            if ply["faces"] is None:
                raise ValueError(f"{path} 没有 face 元素，不是网格；点云请用 source=points")
            occ = from_mesh(ply["xyz"], ply["faces"], **common)
        occ.geometry_source = c["geometry"]
    occ.envelope_source = env.get("source")
    spawn = scene.get("spawn")
    occ.cruise_z = float(spawn["z"]) if spawn and "z" in spawn else None   # v2 的规划硬栅格取这个高度的层带
    return occ


# ToF 地形层下沿（nav 米）：SITL 的 ToF 是平地模型（(z − ground_z)/cos），飞行层以下的桌面/长椅在仿真里不存在，
# 真机 ToF 到它们上方会读短、执行器按「离地 2 m」爬升。planner.route_check 用 [TOF_TERRAIN_Z_MIN, z_slice 下沿)
# 这一层占据沿路径查有没有东西，只打标（tof_terrain_m / agl_model_unverified），不改 ToF 输入（AUDIT_SIM #2）。
# 0.3 m：避开地面本身与草、路缘（Wanderland LiDAR 地面 nav z −0.13…+0.06，garden 草坪 0.1–0.2 m）。
TOF_TERRAIN_Z_MIN = 0.3


def terrain_from_scene(scene, gaussians=None, z_min=TOF_TERRAIN_Z_MIN, source=None):
    """同一份几何、同一 voxel/source/min_points，只把 z 切层换成 [z_min, z_slice 下沿)：飞行层以下的「地形」占据。

    返回 Occupancy（可能为空：grid.size == 0）；z_slice 下沿 <= z_min 时报错（没有可切的层）。
    """
    c = dict(scene.get("collision", {}))
    z_lo = float(c.get("z_slice", (1.65, 2.35))[0])
    if not (z_min < z_lo):
        raise ValueError(f"地形层 [{z_min}, {z_lo}) 为空：z_slice 下沿必须高于 TOF_TERRAIN_Z_MIN")
    c["z_slice"] = (float(z_min), z_lo)
    c.pop("margin_m", None)                                   # 地形层只看原始占据，不做机体膨胀
    scene2 = dict(scene)
    scene2["collision"] = c
    return from_scene(scene2, gaussians=gaussians, source=source)


def _selftest():
    # 一堵 x=5 的薄墙（扁高斯，厚 1 cm、宽 0.6 m）+ 同位置一堵不透明度 0.05 的幽灵墙。
    ys = np.linspace(-1.0, 1.0, 11)
    means = np.array([(5.0, y, 2.0) for y in ys])
    scales = np.tile([0.01, 0.3, 0.3], (len(ys), 1))
    occ = Occupancy(means, scales, None, np.full(len(ys), 0.95),
                    z_range=(1.65, 2.35), voxel_m=0.1,
                    body_radius_m=0.102, body_half_height_m=0.028)
    assert occ.collides(5.0, 0.0, 2.0)
    assert occ.collides(5.0, 1.25, 2.0)          # 墙端 1σ=0.3 + 机体 0.1
    assert not occ.collides(5.0, 1.6, 2.0)
    assert not occ.collides(4.6, 0.0, 2.0) and not occ.collides(5.3, 0.0, 2.0)
    hit, pt = occ.sweep((4.6, 0.0, 2.0), (5.3, 0.0, 2.0))   # 终点在墙另一侧
    assert hit and abs(pt[0] - 4.9) < 0.11, pt
    assert occ.collides(5.0, 0.0, 0.02)                     # 地面
    ghost = Occupancy(means, scales, None, np.full(len(ys), 0.05),
                      z_range=(1.65, 2.35), voxel_m=0.1,
                      body_radius_m=0.102, body_half_height_m=0.028)
    assert not ghost.collides(5.0, 0.0, 2.0) and not ghost.sweep((4.6, 0, 2), (5.3, 0, 2))[0]
    print("collision selftest PASS", occ.summary())


if __name__ == "__main__":
    _selftest()
