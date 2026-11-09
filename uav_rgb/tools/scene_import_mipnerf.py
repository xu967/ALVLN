#!/usr/bin/env python3
"""Mip-NeRF 360（Inria 3DGS 官方预训练 30k）单场景接入：重力对齐 + 定尺度 -> 米制 PLY -> scene.json + 路线校验
-> 沿路线清晰度探测（+ 俯视高度复核）-> A* 教师冷烟 -> 三视角视频 -> 冷烟汇总。

SCENE-B（2026-11-07）用 tmp/agent4-scene-mipnerf/ 下 probe_align / ply_scale / build_scene / finalize_routes /
height_check / smoke.sh / smoke_summary 七个一次性脚本接了 garden、bicycle；这里收成一份，每场景一段配置（SCENES：
原始件目录、尺度取值与依据、路线）。每步落盘、产物在就跳过（ply 步按 PLY 头注释里的缩放倍数判断要不要重做），
任一步失败即停，结束落 OUT/DONE（rc + 时间）。

运行（守护进程；先 cd 到仓库根，再单独一行起，否则 $! 拿到的是子 shell）：
  OUT=/root/rivermind-data/uav_rgb/tmp/agent10-scene-3dgs/import-stump; mkdir -p $OUT
  SCENE=stump setsid nohup /root/rivermind-data/uav_rgb/3dgs-venv/bin/python tools/scene_import_mipnerf.py \
      > $OUT/run.log 2>&1 < /dev/null & echo $! > $OUT/run.pid
只跑部分步骤：IMPORT_STEPS=align,ply,scene（顺序固定，见 STEPS）。SCENE=garden / bicycle 会按配置重写该场景
scene.json（不是本工具写的旧文件先备份成 scene.json.bak-scene-import），已发布场景没有必要重跑。
"""
import glob
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np

# ---------------------------------------------------------------- 配置（改这里）
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根（本文件在 tools/）
PY = os.environ.get("UAV_RGB_3DGS_PYTHON", "/root/rivermind-data/uav_rgb/3dgs-venv/bin/python")   # torch + gsplat 1.5.3
DATA_ROOT = os.environ.get("DATA_ROOT", "/root/rivermind-data")
DOWNLOADS = os.path.join(DATA_ROOT, "uav_rgb", "downloads", "inria-3dgs-pretrained")   # <scene>/{point_cloud_30000.ply, cameras.json, input.ply}
SCENES_ROOT = os.path.join(DATA_ROOT, "uav_rgb", "3dgs-scenes")
SCENE = os.environ.get("SCENE", "stump")
OUT = os.environ.get("IMPORT_OUT", os.path.join(DATA_ROOT, "uav_rgb", "tmp", "agent10-scene-3dgs", f"import-{SCENE}"))
SMOKE_ROOT = os.path.join(DATA_ROOT, "uav_rgb", "3dgs-smoke-20261109")   # 冷烟 <SMOKE_ROOT>/<scene>/r<i>/ep_0000k/
STEPS = ("align", "ply", "scene", "probe", "smoke", "video", "summary")
RUN_STEPS = tuple(s for s in os.environ.get("IMPORT_STEPS", ",".join(STEPS)).split(",") if s)
FLY_Z = 2.0                      # nav 系飞行高度（米）：项目口径（Gazebo / Callinstrasse / garden / bicycle 同值）
CAMERA = {"width": 320, "height": 240, "fov_deg": 60.0, "pitch_deg": 0.0}   # 用户 11-07 拍板：与 Gazebo 同 60° HFOV / 320×240
COLLISION = {"z_slice": [FLY_Z - 0.2, FLY_Z + 0.2], "voxel_m": 0.1, "envelope": "e1_mini"}   # 可飞区只有 ±6 m，0.1 m 体素（SCENE-B 同）
REACH_RADIUS_M = 1.0             # SCENE-B 口径：场景小，到达圈比 Callinstrasse（1.5）/ Wanderland（2.0）小
HOLD_S = 0.5                     # 与 scene.step_s / Gazebo 训练数据窗口一致
RANSAC_ITERS, RANSAC_SEED = 2000, 0   # SCENE-B 同值；固定种子，同一输入逐位复现（test_scene_import_mipnerf 对拍 garden/bicycle）
RANSAC_THRESH_FRAC = 0.01        # 地面内点阈值 = 1% 轨道半径（轨道半径 3.4–3.7 单位 → 3–4 cm 级，草坪起伏量级）
UP_AGREE_MAX_DEG = 5.0           # 相机 down 轴均值 / 轨道平面法线 / RANSAC 地面法线两两夹角上限；SCENE-B 三场景实测 ≤ 2.7°
REF_MIN_POINTS = 50              # object_height：猜测高 ±band 内少于 50 个稀疏点 = 没找到参照面，报错不给尺度
HEIGHT_CHECK_Z = (1.0, 2.0, 3.0)  # 俯视复核：每条路线出生点 / 航点在这几个 nav 高度垂直向下渲染，nav z − 深度 = 地面高（应 ≈ 0）
SMOKE_EPISODES = 2               # 每路线 2 集：过线 = ≥1 集到达、0 碰撞（BRIEF §9，与已有四场景冷烟同口径）
SEED0 = 1000                     # 路线 r 第 k 集 seed = SEED0 + 10*r + k（SCENE-B 冷烟同规则）
SMOKE_PARALLEL = 2               # 同时跑的 collect_3dgs 个数：GPU 上主会话已有 2 个 3DGS 进程，本 agent ≤ 2
VIDEO_ROUTES = (1, 5)            # 出三视角视频的路线（各取第 0 集），挑带转弯/绕障的（stump：r1 左转绕树苗 B、r5 绕树苗 A）
VIDEO_QUALITY = "medium"         # sim/tools/view_layout.VIDEO_QUALITY：1280×720 / CRF 22（high 编码约 3× 慢）
CF_SITL_PROC = os.environ.get("CF_SITL_PROC", os.path.join(DATA_ROOT, "uav_rgb", "tmp", "build-cf-sitl", "cf_sitl_proc"))   # lib v3.1
LICENSE = ("Mip-NeRF 360 photos (Barron et al. CVPR 2022): CC-BY 4.0 as declared by the authors on the SMERF page "
           "(smerf-3d.github.io, 'Both datasets and checkpoints are released under CC-BY 4.0'); the Mip-NeRF 360 project page "
           "itself carries no license text (checked 2026-11-07). Pretrained 3DGS PLY: Inria/MPII gaussian-splatting pretrained "
           "package models.zip (repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip, no license stated "
           "for the models; the gaussian-splatting software license is non-commercial research). Use: academic research only, "
           "attribute both Barron et al. 2022 and Kerbl et al. 2023.")
# 每场景一段：src = Inria 包里该场景的原始件目录；scale.method：object_height（已知尺寸物的水平面在稀疏点高度直方图里的峰）|
# camera_height（相机离地中位 ↔ 手持高度，低置信）| fixed（另行量好的 m/单位）；confidence=low 时 scene.json 记 scale_unverified。
# routes：nav 系（米，z 固定 FLY_Z），max_steps 由 planner.route_check 回填。garden / bicycle 的值与已发布 scene.json 一致。
SCENES = {
    "garden": dict(
        src=os.path.join(DOWNLOADS, "garden"),
        scale=dict(method="object_height", ref="桌面", ref_h_m=0.75, ref_h_guess=1.35, center_r=1.5, band=0.08,
                   confidence="medium", source="园桌桌面标准高 0.72–0.76 m 取 0.75（±4%）；稀疏点水平面峰 [1.30,1.40) 单位（SCENE-B 11-07）"),
        routes=[
            {"name": "r0_south_lane_east", "spawn": {"x": -3.5, "y": -2.0, "yaw_deg": 0.0}, "waypoint": {"x": 4.0, "y": -3.5}, "note": "南侧直线向东，树在左手，终点离房屋南墙 1.5 m"},
            {"name": "r1_sw_ne_around_tree", "spawn": {"x": -3.0, "y": -4.0, "yaw_deg": 45.0}, "waypoint": {"x": 6.0, "y": 1.0}, "note": "直线会撞树，A* 绕树南侧再北上"},
            {"name": "r2_nw_se_past_table", "spawn": {"x": -3.0, "y": 3.0, "yaw_deg": -45.0}, "waypoint": {"x": 4.0, "y": -3.5}, "note": "西北→东南斜穿，擦树西南缘到房屋前"},
            {"name": "r3_north_south_over_table", "spawn": {"x": 0.0, "y": 4.0, "yaw_deg": -90.0}, "waypoint": {"x": 0.0, "y": -5.0}, "note": "北→南直线飞越桌面上方"},
            {"name": "r4_east_return_west", "spawn": {"x": 6.0, "y": -1.0, "yaw_deg": 90.0}, "waypoint": {"x": -2.0, "y": 3.0}, "note": "东侧朝北出生，左转绕树回桌子西北"},
            {"name": "r5_west_edge_north", "spawn": {"x": -3.0, "y": -5.0, "yaw_deg": 90.0}, "waypoint": {"x": -2.0, "y": 4.0}, "note": "沿草坪西缘北行，桌子在右手"},
        ]),
    "bicycle": dict(
        src=os.path.join(DOWNLOADS, "bicycle"),
        scale=dict(method="object_height", ref="长椅座面", ref_h_m=0.45, ref_h_guess=0.475, center_r=1.5, band=0.08,
                   confidence="medium", source="公园长椅座高 0.43–0.46 m 取 0.45；靠背顶 ≈0.95 单位 → 0.89 m 自洽（SCENE-B 11-07）"),
        routes=[
            {"name": "r0_path_west_east", "spawn": {"x": -5.0, "y": -3.0, "yaw_deg": 0.0}, "waypoint": {"x": 3.5, "y": -3.5}, "note": "西→东沿南侧小径，长椅在左前，终点在树丛南侧"},
            {"name": "r1_west_ese_past_bench", "spawn": {"x": -6.0, "y": 1.0, "yaw_deg": -20.0}, "waypoint": {"x": 3.0, "y": -3.5}, "note": "西侧出发朝东南偏东，从长椅南侧经过"},
            {"name": "r2_east_north_turn_west", "spawn": {"x": 5.0, "y": -5.0, "yaw_deg": 90.0}, "waypoint": {"x": -2.0, "y": 4.0}, "note": "东南角朝北出生，前方 3 m 是树丛，必须左转绕其南缘"},
            {"name": "r3_north_south_over_bench", "spawn": {"x": 2.0, "y": 5.0, "yaw_deg": -90.0}, "waypoint": {"x": -1.0, "y": -5.0}, "note": "北→南从长椅东侧上方掠过，树丛在左"},
            {"name": "r4_south_north_turn_west", "spawn": {"x": 3.0, "y": -5.0, "yaw_deg": 90.0}, "waypoint": {"x": -4.0, "y": 3.0}, "note": "南侧朝北，左转向西北，长椅从右侧经过"},
            {"name": "r5_se_nw_around_tree", "spawn": {"x": 7.0, "y": -4.0, "yaw_deg": 135.0}, "waypoint": {"x": -3.0, "y": 2.0}, "note": "东南角朝西北，树丛在右前，绕其南侧后直指长椅"},
        ]),
    "stump": dict(
        src=os.path.join(DOWNLOADS, "stump"),
        # 林间空地 + 爬满常春藤的树桩（顶 ≈1.55 单位，非标准尺寸）+ 树苗木桩护栏 + 倒木：稀疏点高度直方图在中心 1.5 单位内
        # 从地面到 1.6 单位平铺，没有水平面峰；相机离地 0.35–2.79 m（按 0.632），顶端按「举过头顶 ≈2.2 m」算是 0.50 m/单位
        scale=dict(method="camera_height", cam_h_m=1.5, confidence="low",
                   source="场景里没有已知尺寸物体，按「手持采集相机离地中位 ≈ 1.5 m」假设（BRIEF §9）"),
        # 飞行层两处占据都是带木桩护栏的树苗冠：A (0..2.3, 2.5..5.3)、B (2.8..5.3, −2.8..0.3)；r > 5 m 朝外看是雾 / 拉丝，
        # 东、东北外围最差（候选 c0/c1/c5 因此淘汰），路线收在 r ≲ 5.5 m、视线穿过空地。出生点离外围枝叶 ≥1 m：
        # 低不透明度的枝叶不进占据代理，贴着它出生开局几帧是满屏虚化（r0 原 (6.5,2)、r5 原 (3.5,5.5) 因此内移）
        routes=[
            {"name": "r0_east_west_between", "spawn": {"x": 5.0, "y": 1.2, "yaw_deg": 180.0}, "waypoint": {"x": -5.5, "y": -2.0}, "note": "东→西从树苗 A（北）与 B（东南）之间穿过，飞越树桩北侧"},
            {"name": "r1_se_north_turn_west", "spawn": {"x": 4.5, "y": -5.0, "yaw_deg": 90.0}, "waypoint": {"x": -3.5, "y": 4.0}, "note": "东南角朝北出生，正前方是树苗 B 的冠，必须左转绕其西侧再向西北"},
            {"name": "r2_south_north_past_stump", "spawn": {"x": 0.0, "y": -5.5, "yaw_deg": 90.0}, "waypoint": {"x": -1.5, "y": 5.0}, "note": "南→北从树桩上方掠过，树苗 A 在右前"},
            {"name": "r3_west_east_north_of_b", "spawn": {"x": -5.0, "y": -1.0, "yaw_deg": 0.0}, "waypoint": {"x": 4.0, "y": 1.5}, "note": "西→东飞越树桩，终点在树苗 A、B 之间"},
            {"name": "r4_north_south_past_a", "spawn": {"x": -2.0, "y": 5.0, "yaw_deg": -90.0}, "waypoint": {"x": 1.5, "y": -4.5}, "note": "北侧朝南（树苗 A 在左手），斜向东南到 B 的西南"},
            {"name": "r5_ne_sw_around_a", "spawn": {"x": 4.0, "y": 4.4, "yaw_deg": -140.0}, "waypoint": {"x": -4.0, "y": -3.0}, "note": "东北角朝西南，直线擦树苗 A 的冠东南角，A* 绕其东南侧"},
        ]),
}
CFG = SCENES[SCENE]
SRC_DIR = CFG["src"]
SCENE_DIR = os.path.join(SCENES_ROOT, SCENE)
SCENE_JSON = os.path.join(SCENE_DIR, "scene.json")
ENV = dict(os.environ)           # main() 补 TORCH_EXTENSIONS_DIR / CF_SITL_PROC，子进程与进程内渲染用同一份

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run(cmd, what):
    log(f"### {what}: {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=REPO, env=ENV)
    log(f"### {what} rc={rc} wall={time.time() - t0:.1f}s")
    if rc != 0:
        raise SystemExit(f"{what} 失败 rc={rc}")


def torch_ext_dir():
    """gsplat JIT 缓存按 GPU 架构分目录（目录里混了别的架构会重编或加载失败）：extensions-sm<compute_cap>。"""
    if os.environ.get("TORCH_EXTENSIONS_DIR"):
        return os.environ["TORCH_EXTENSIONS_DIR"]
    try:
        cap = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=60).stdout.split()[0].replace(".", "")
    except (OSError, IndexError, subprocess.SubprocessError):
        cap = "unknown"
    return os.path.join(DATA_ROOT, "uav_rgb", "torch-cache", f"extensions-sm{cap}")


# ---------------------------------------------------------------- 1 对齐：重力 / 地面 / 尺度 / nav_to_world
def read_xyz_ply(path):
    """Inria 包里的 input.ply（COLMAP 稀疏点，Open3D 导出：double x/y/z/nx/ny/nz + uchar rgb）-> xyz float64 [N,3]。"""
    types = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4"}
    with open(path, "rb") as f:
        props, n = [], 0
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[2])
            elif line.startswith("property"):
                props.append((line.split()[2], types[line.split()[1]]))
            elif line == "end_header":
                break
        arr = np.fromfile(f, dtype=np.dtype(props), count=n)
    return np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float64)


def fit_plane_ransac(pts, n_iter=RANSAC_ITERS, thresh=0.02, seed=RANSAC_SEED, normal_hint=None, max_tilt_deg=25.0):
    """RANSAC 平面 + 内点最小二乘精修：返回 (单位法线 n, d, 内点数)，n·p + d = 0，n 朝 normal_hint 一侧且与之夹角 ≤ max_tilt_deg。"""
    rng = np.random.default_rng(seed)
    best, best_n, best_d = -1, None, None
    for _ in range(n_iter):
        i = rng.choice(len(pts), 3, replace=False)
        a, b, c = pts[i]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if normal_hint is not None:
            if np.dot(n, normal_hint) < 0:
                n = -n
            if np.degrees(np.arccos(np.clip(np.dot(n, normal_hint), -1, 1))) > max_tilt_deg:
                continue
        d = -np.dot(n, a)
        cnt = int((np.abs(pts @ n + d) < thresh).sum())
        if cnt > best:
            best, best_n, best_d = cnt, n, d
    if best_n is None:
        raise ValueError(f"RANSAC {n_iter} 次没有一个平面法线落在 hint ±{max_tilt_deg}° 内")
    inl = np.abs(pts @ best_n + best_d) < thresh
    q = pts[inl]
    cen = q.mean(0)
    _, _, vt = np.linalg.svd(q - cen, full_matrices=False)
    n = vt[2]
    if np.dot(n, best_n) < 0:
        n = -n
    return n, -np.dot(n, cen), int(inl.sum())


def scale_from(cfg, hg, hc, rad):
    """尺度（米/COLMAP 单位）与依据文字。hg/hc：稀疏点/相机离地高（单位）；rad：稀疏点到轨道中心的水平距离（单位）。"""
    method = cfg.get("method")
    if method == "object_height":
        sel = (rad < cfg["center_r"]) & (np.abs(hg - cfg["ref_h_guess"]) < cfg["band"])
        if int(sel.sum()) < REF_MIN_POINTS:
            raise ValueError(f"{cfg['ref']}：猜测高 {cfg['ref_h_guess']}±{cfg['band']} 单位内只有 {int(sel.sum())} 个稀疏点，没找到参照面")
        h_ref = float(np.median(hg[sel]))
        return cfg["ref_h_m"] / h_ref, (f"{cfg['ref']} {cfg['ref_h_m']} m ↔ {h_ref:.3f} COLMAP 单位（轨道中心 {cfg['center_r']} 单位内、"
                                        f"猜测高 ±{cfg['band']} 的 {int(sel.sum())} 个稀疏点中位）")
    if method == "camera_height":
        h_ref = float(np.median(hc))
        return cfg["cam_h_m"] / h_ref, f"相机离地中位 {h_ref:.3f} 单位（{len(hc)} 台）↔ 手持高度 {cfg['cam_h_m']} m"
    if method == "fixed":
        return float(cfg["m_per_unit"]), f"给定 {cfg['m_per_unit']} m/单位"
    raise ValueError(f"scale.method 无效: {method!r}（object_height | camera_height | fixed）")


def align_scene(src_dir, scale):
    """cameras.json + input.ply -> 对齐结果 dict（不写盘）。

    重力：input.ply 里最低相机以下、3 倍轨道半径内的稀疏点 RANSAC 地面，法线 = up；与相机 down 轴均值、相机轨道平面法线
    交叉核对。nav_to_world：列 = [世界 x 的水平投影, up × 它, up]，平移 = 轨道中心投到（缩放后）地面，所以 nav z = 离地高、
    ground_z = 0。尺度不进 nav_to_world（collision._rigid_parts 只收刚体），由 ply 步烘进 PLY。
    """
    with open(os.path.join(src_dir, "cameras.json"), encoding="utf-8") as f:
        cams = json.load(f)
    C = np.array([c["position"] for c in cams], dtype=np.float64)
    R = np.array([c["rotation"] for c in cams], dtype=np.float64)      # camera->world，列 = 相机 右/下/前
    up_cam = -R[:, :, 1].mean(0)
    up_cam /= np.linalg.norm(up_cam)
    cen = C.mean(0)
    _, _, vt = np.linalg.svd(C - cen, full_matrices=False)
    up_orbit = vt[2] if np.dot(vt[2], up_cam) > 0 else -vt[2]
    xyz = read_xyz_ply(os.path.join(src_dir, "input.ply"))

    def horiz(p, u):
        return (p - cen) - np.outer((p - cen) @ u, u)

    def ang(a, b):
        return round(float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1)))), 2)

    r_orbit = float(np.median(np.linalg.norm(horiz(C, up_cam), axis=1)))
    h_all, h_cam = xyz @ up_cam, C @ up_cam
    rad = np.linalg.norm(horiz(xyz, up_cam), axis=1)
    cand = (h_all < h_cam.min()) & (rad < 3 * r_orbit) & (h_all > h_cam.min() - 3 * r_orbit)
    up, d, n_inl = fit_plane_ransac(xyz[cand], thresh=RANSAC_THRESH_FRAC * r_orbit, normal_hint=up_cam)
    hg, hc = xyz @ up + d, C @ up + d
    rad = np.linalg.norm(horiz(xyz, up), axis=1)
    s, basis = scale_from(scale, hg, hc, rad)
    ex = np.array([1.0, 0.0, 0.0]) - up[0] * up
    ex /= np.linalg.norm(ex)
    rm = np.stack([ex, np.cross(up, ex), up], axis=1)
    c_w = cen * s
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = rm, c_w - (up @ c_w + d * s) * up
    center_r = scale.get("center_r", 1.5)          # 中心物高度直方图（找参照面 / 写文档「没有水平面峰」用）
    h_c = hg[rad < center_r]
    hist, edges = np.histogram(h_c, bins=np.arange(-0.1, max(0.0, float(np.percentile(h_c, 99.5))) + 0.1, 0.05))
    conf = scale.get("confidence", "low")
    return {"scale_m_per_unit": s, "scale_method": scale["method"], "scale_basis": basis, "scale_source": scale.get("source", ""),
            "scale_confidence": conf, "scale_unverified": conf == "low",
            "up_world": up.tolist(), "up_cam": up_cam.tolist(), "up_orbit": up_orbit.tolist(),
            "up_angles_deg": {"cam_ground": ang(up_cam, up), "orbit_ground": ang(up_orbit, up), "cam_orbit": ang(up_cam, up_orbit)},
            "ground_inliers": n_inl, "ground_plane_world_scaled": {"normal": up.tolist(), "d": float(d * s)},
            "orbit_center_world": c_w.tolist(), "orbit_radius_m": r_orbit * s,
            "cam_height_m_p10_50_90": (np.percentile(hc, [10, 50, 90]) * s).tolist(),
            "center_hist_units": {"center_r": center_r, "edges": edges.round(3).tolist(), "counts": hist.tolist()},
            "nav_to_world": m.tolist(),
            "cams_nav": [{"img_name": c["img_name"], "pos_nav": (rm.T @ (np.asarray(c["position"]) * s - m[:3, 3])).round(4).tolist(),
                          "pos_world": (np.asarray(c["position"]) * s).round(4).tolist(), "rotation_c2w_world": c["rotation"]}
                         for c in cams]}


def step_align():
    al = align_scene(SRC_DIR, CFG["scale"])
    os.makedirs(SCENE_DIR, exist_ok=True)
    write_json(os.path.join(SCENE_DIR, "align.json"), al)
    log(f"align: up {np.round(al['up_world'], 4).tolist()}（地面内点 {al['ground_inliers']}，夹角 {al['up_angles_deg']}）；"
        f"{al['scale_basis']} -> {al['scale_m_per_unit']:.4f} m/单位（{al['scale_confidence']}）；相机离地 p10/50/90 "
        f"{np.round(al['cam_height_m_p10_50_90'], 2).tolist()} m；轨道半径 {al['orbit_radius_m']:.2f} m")
    if max(al["up_angles_deg"].values()) > UP_AGREE_MAX_DEG:
        raise SystemExit(f"三种 up 估计不一致 {al['up_angles_deg']}（> {UP_AGREE_MAX_DEG}°），重力不可信")


# ---------------------------------------------------------------- 2 米制 PLY
SCALE_TAG = "comment uavrgb_scale"


def ply_scale_of(path):
    """PLY 头注释里记的缩放倍数（scale_ply 写的）；没有返回 None。"""
    with open(path, "rb") as f:
        for _ in range(200):
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith(SCALE_TAG):
                return float(line.split()[2])
            if line == "end_header" or not line:
                return None
    return None


def scale_ply(src, dst, s):
    """Inria 3DGS PLY 各向同性缩放：xyz *= s、scale_0..2 += ln s，其余字节（球谐、不透明度、旋转）原样；头里加一行
    SCALE_TAG。只缩放不旋转：旋转要把 3 阶球谐 f_rest 按 Wigner-D 重投影，旋转留给 nav_to_world。先写 .tmp 再改名。"""
    if not (math.isfinite(s) and s > 0):
        raise ValueError(f"scale 必须是有限正数，得到 {s}")
    with open(src, "rb") as f:
        header, props, n = [], [], 0
        while True:
            line = f.readline()
            text = line.decode("ascii").strip()
            if text.startswith(SCALE_TAG):
                raise ValueError(f"{src} 已经缩放过（{text}），从原始件缩放")
            header.append(line)
            if text.startswith("format ") and text != "format binary_little_endian 1.0":
                raise ValueError(f"{src}: 只支持 binary_little_endian")
            if text.startswith("element vertex "):
                n = int(text.split()[2])
            elif text.startswith("property "):
                if text.split()[1] != "float":
                    raise ValueError(f"{src}: 只支持 float 属性，遇到 {text}")
                props.append(text.split()[2])
            elif text == "end_header":
                break
        table = np.fromfile(f, dtype=np.dtype([(p, "<f4") for p in props]), count=n)
    if len(table) != n:
        raise ValueError(f"{src}: 顶点数不符（头 {n}，实际 {len(table)}）")
    for k in ("x", "y", "z"):
        table[k] = (table[k].astype(np.float64) * s).astype(np.float32)
    for k in ("scale_0", "scale_1", "scale_2"):
        table[k] = (table[k].astype(np.float64) + math.log(s)).astype(np.float32)
    header.insert(2, f"{SCALE_TAG} {s:.6f} src {os.path.basename(src)} (tools/scene_import_mipnerf.py)\n".encode("ascii"))
    with open(dst + ".tmp", "wb") as f:
        f.write(b"".join(header))
        table.tofile(f)
    os.replace(dst + ".tmp", dst)
    return n


def step_ply():
    s = round(read_json(os.path.join(SCENE_DIR, "align.json"))["scale_m_per_unit"], 6)
    dst = os.path.join(SCENE_DIR, f"{SCENE}_30k_m.ply")
    if os.path.isfile(dst) and ply_scale_of(dst) == s:
        log(f"ply: {dst} 已是 ×{s}，跳过")
        return
    t0 = time.time()
    n = scale_ply(os.path.join(SRC_DIR, "point_cloud_30000.ply"), dst, s)
    log(f"ply: 写 {dst}（{n} 个高斯 ×{s}，{time.time() - t0:.0f}s）")


# ---------------------------------------------------------------- 3 scene.json + 路线校验
def build_scene(al, routes):
    s = al["scale_m_per_unit"]
    unverified = bool(al["scale_unverified"])
    notes = [
        f"尺度：{al['scale_basis']}；{s:.4f} m/COLMAP 单位，烘进 PLY（tools/scene_import_mipnerf.py scale_ply：只改 xyz 与 log-scale，"
        "球谐 f_rest 原样，所以 PLY 里不旋转）。",
        f"重力/地面：COLMAP 稀疏点 RANSAC 地面法线 {np.round(al['up_world'], 4).tolist()}，与相机 down 轴均值 / 轨道平面法线夹角 "
        f"{al['up_angles_deg']['cam_ground']}° / {al['up_angles_deg']['orbit_ground']}°；nav_to_world 第三列 = 该法线，平移 = 相机轨道中心"
        f"投到地面，nav z=0 即地面（ground_z=0）。相机离地 p10/50/90 = {np.round(al['cam_height_m_p10_50_90'], 2).tolist()} m。",
        "Collision is the v2 occupancy proxy (opacity-filtered ellipsoids, E1 MINI envelope), not ground truth.",
        "routes[]：routes[0] 与顶层 spawn/waypoint 相同；max_steps = planner.route_check 步数预算（cf_sitl 每 0.5 s 步 0.15 m / 17°，×1.25）。"
        "每条都用 tools/scene_probe_quality.py 沿 A* 路径渲染过（拼图 probe_quality/route_*.png）；Mip-NeRF 360 环绕中心物体采集，"
        "看向外围发糊或成雾，路线终点不朝外围。",
        "lapvar 只量锐度不量正确性：外围漂浮碎片的帧 lapvar 也高，路线定稿前必须看拼图，不能只看 low_frac。",
    ]
    if unverified:
        notes.append(f"尺度未核实（scale_unverified）：{al['scale_source']}。本场景只用于训练多样性与外观泛化，"
                     "不进任何米制结论（航程、净空、速度的米数只在场景内部自洽，和真实世界可能差 10–20%）。")
    return {
        "schema": "uavrgb.3dgs_scene.v1",
        "name": f"{SCENE}_mipnerf360_30k",
        "generator": "tools/scene_import_mipnerf.py",
        "source": (f"Mip-NeRF 360 '{SCENE}' (Barron et al. CVPR 2022), Inria 3DGS pretrained point_cloud/iteration_30000/"
                   "point_cloud.ply extracted from models.zip via HTTP Range (tmp/agent4-scene-mipnerf/remote_zip.py, 2026-11-07); "
                   f"scaled x{s:.4f} by tools/scene_import_mipnerf.py (xyz + log-scale only, SH kept)"),
        "license": LICENSE,
        "ply": f"{SCENE}_30k_m.ply",
        "nav_to_world": np.round(al["nav_to_world"], 6).tolist(),
        "world_up": "z",
        "ground_z": 0.0,
        "scale_m_per_unit": round(s, 6),
        "scale_source": f"{al['scale_source']}；{al['scale_basis']}",
        "scale_confidence": al["scale_confidence"],
        "scale_unverified": unverified,
        "spawn": dict(routes[0]["spawn"]) if routes else {"x": 0.0, "y": 0.0, "z": FLY_Z, "yaw_deg": 0.0},
        "waypoint": dict(routes[0]["waypoint"]) if routes else {"x": 1.0, "y": 0.0, "z": FLY_Z},
        "reach_radius_m": REACH_RADIUS_M,
        "camera": dict(CAMERA, fov_source="用户 2026-11-07 05:50 拍板：Gazebo 与 3DGS 统一 60° HFOV、320×240"),
        "collision": dict(COLLISION),
        "max_steps": (routes[0]["max_steps"] or 60) if routes else 60,
        "step_s": HOLD_S,
        "notes": notes,
        "routes": routes,
    }


def draw_map(occ, cams_xy, paths, path, px_per_m=20):
    """飞行层占据俯视图：黑 = 原始占据，灰 = 机体膨胀带，蓝点 = 训练相机，红线 = A* 路径（绿点出生、红圈航点），1 m 网格，图上 = +y。"""
    import cv2
    if occ.grid.size == 0:
        return
    k = max(1, int(round(px_per_m * occ.voxel)))
    img = np.full(occ.raw.shape[:2] + (3,), 255, np.uint8)
    img[occ.grid.any(axis=2)] = (190, 190, 190)
    img[occ.raw.any(axis=2)] = (0, 0, 0)
    img = np.repeat(np.repeat(np.ascontiguousarray(np.transpose(img, (1, 0, 2))[::-1]), k, 0), k, 1)
    h, w = img.shape[:2]

    def px(x, y):
        return int((x - occ.origin[0]) / occ.voxel * k), int(h - 1 - (y - occ.origin[1]) / occ.voxel * k)

    for g in range(int(math.floor(occ.origin[0])), int(math.ceil(occ.origin[0] + w / k * occ.voxel)) + 1):
        cv2.line(img, px(g, occ.origin[1]), px(g, occ.origin[1] + h / k * occ.voxel), (0, 140, 255) if g == 0 else (235, 215, 190), 1)
        if g % 2 == 0:
            cv2.putText(img, str(g), (px(g, 0)[0] + 2, h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 90, 0), 1)
    for g in range(int(math.floor(occ.origin[1])), int(math.ceil(occ.origin[1] + h / k * occ.voxel)) + 1):
        cv2.line(img, px(occ.origin[0], g), px(occ.origin[0] + w / k * occ.voxel, g), (0, 140, 255) if g == 0 else (235, 215, 190), 1)
        if g % 2 == 0:
            cv2.putText(img, str(g), (2, px(0, g)[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 90, 0), 1)
    for x, y in cams_xy:
        cv2.circle(img, px(x, y), 2, (255, 120, 0), -1)
    for i, pth in paths.items():
        if pth:
            cv2.polylines(img, [np.array([px(x, y) for x, y in pth], np.int32)], False, (0, 0, 255), 2)
            cv2.circle(img, px(*pth[0]), 5, (0, 170, 0), -1)
            cv2.circle(img, px(*pth[-1]), 7, (0, 0, 255), 2)
            cv2.putText(img, f"r{i}", (px(*pth[0])[0] + 6, px(*pth[0])[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
    cv2.imwrite(path, img)


def step_scene():
    """写 scene.json（尺度 / 置信度 / 路线），planner.route_check 校验每条路线、回填 max_steps，画占据 + 路径俯视图；
    路线为空或有不可行 / 净空不足的路线即停（先看 probe_quality/map_routes.png 改配置区 routes）。"""
    scene_mod = importlib.import_module("sim.3dgs.scene")
    collision = importlib.import_module("sim.3dgs.collision")
    planner_mod = importlib.import_module("sim.3dgs.planner")
    al = read_json(os.path.join(SCENE_DIR, "align.json"))
    ply = os.path.join(SCENE_DIR, f"{SCENE}_30k_m.ply")
    if not os.path.isfile(ply) or ply_scale_of(ply) != round(al["scale_m_per_unit"], 6):
        raise SystemExit(f"{ply} 缺失或缩放倍数与 align.json 不符：先跑 ply 步")
    routes = [{"name": r["name"], "spawn": {"x": float(r["spawn"]["x"]), "y": float(r["spawn"]["y"]), "z": FLY_Z,
                                            "yaw_deg": float(r["spawn"].get("yaw_deg", 0.0))},
               "waypoint": {"x": float(r["waypoint"]["x"]), "y": float(r["waypoint"]["y"]), "z": FLY_Z},
               "max_steps": None, "note": r.get("note", "")} for r in CFG["routes"]]
    if os.path.isfile(SCENE_JSON) and read_json(SCENE_JSON).get("generator") != "tools/scene_import_mipnerf.py":
        bak = SCENE_JSON + ".bak-scene-import"          # 手写 / SCENE-B 的旧文件只备份一次，不覆盖已有备份
        if not os.path.exists(bak):
            shutil.copy2(SCENE_JSON, bak)
            log(f"scene: 旧 scene.json 备份到 {bak}")
    scene = build_scene(al, routes)
    os.makedirs(SCENE_DIR, exist_ok=True)
    scene_mod.save_scene(scene, SCENE_JSON)
    t0 = time.time()
    occ = collision.from_scene(scene_mod.load_scene(SCENE_JSON))
    log(f"scene: 占据 {occ.summary()['n_occupied_cells']} 格（{occ.shape}，{time.time() - t0:.0f}s）")
    planner = planner_mod.OccPlanner(occ)
    rows, paths, bad = [], {}, []
    for i, r in enumerate(routes):
        chk = planner_mod.route_check(planner, r["spawn"], r["waypoint"])
        paths[i] = chk.pop("path", None)
        r["max_steps"] = int(chk["steps_estimate"]) if chk.get("steps_estimate") else None
        if chk.get("path_m") is not None:
            r["note"] = f"{r['note']}；航程 {chk['path_m']} m，转向 {chk['turn_deg']}°，最小净空 {chk['min_clearance_m']} m"
        rows.append({"route": i, "name": r["name"], **chk})
        ok = chk.get("feasible") and chk.get("clear_ok")
        log(f"scene: route {i} {r['name']}: {'OK ' if ok else 'BAD'} path {chk.get('path_m')} m turn {chk.get('turn_deg')}° "
            f"min_clear {chk.get('min_clearance_m')} steps {chk.get('steps_estimate')} {chk.get('reason', '')}")
        if not ok:
            bad.append(r["name"])
    scene_mod.save_scene(build_scene(al, routes), SCENE_JSON)
    os.makedirs(OUT, exist_ok=True)
    write_json(os.path.join(OUT, "routes_check.json"), rows)
    os.makedirs(os.path.join(SCENE_DIR, "probe_quality"), exist_ok=True)
    draw_map(occ, [c["pos_nav"][:2] for c in al["cams_nav"]], paths, os.path.join(SCENE_DIR, "probe_quality", "map_routes.png"))
    if not routes:
        raise SystemExit("配置区 routes 为空：看 probe_quality/map_routes.png（飞行层占据 + 训练相机）选路线后再跑 scene 步")
    if bad:
        raise SystemExit(f"路线不可行 / 净空不足：{bad}（看 probe_quality/map_routes.png 改配置区 routes）")
    log(f"scene: 写 {SCENE_JSON}（{len(routes)} 条路线 feasible / clear_ok）")


# ---------------------------------------------------------------- 4 沿路线清晰度探测 + 俯视高度复核
def step_probe():
    """清晰度全交给 tools/scene_probe_quality.run（采样、航向、lapvar 阈值、低质帧门槛、拼图只有那一份）；这里只加俯视复核：
    每条路线出生点 / 航点在 HEIGHT_CHECK_Z 各垂直向下渲一帧，中心 20×20 期望深度中位 = 3DGS 里的离地高，并进同一份
    routes_quality.json 的 height_check（nav z − 深度应 ≈ 0；Callinstrasse P0 那种整体偏 1.6 m 这里一眼可见）。"""
    pq = importlib.import_module("scene_probe_quality")
    scene_mod = importlib.import_module("sim.3dgs.scene")
    renderer_mod = importlib.import_module("sim.3dgs.renderer")
    out_dir = os.path.join(SCENE_DIR, "probe_quality")
    rows, ok = pq.run(SCENE_JSON, out_dir=out_dir)
    scene = scene_mod.load_scene(SCENE_JSON)
    rd = renderer_mod.make_renderer(scene)
    checks = []
    for r in scene.get("routes") or []:
        for key in ("spawn", "waypoint"):
            x, y = float(r[key]["x"]), float(r[key]["y"])
            for z in HEIGHT_CHECK_Z:
                _rgb, dep, _ms = rd.render_pose(x, y, z, 0.0, pitch_deg=-90.0)
                h, w = dep.shape
                c = dep[h // 2 - 10: h // 2 + 10, w // 2 - 10: w // 2 + 10]
                c = c[c > 0]
                med = round(float(np.median(c)), 3) if c.size else None
                checks.append({"route": r["name"], "at": key, "xy": [x, y], "z_nav": z, "nadir_depth_m": med,
                               "ground_from_nadir_m": round(z - med, 3) if med is not None else None})
    path = os.path.join(out_dir, "routes_quality.json")
    report = read_json(path)
    report["height_check"] = checks
    write_json(path, report)
    g = [c["ground_from_nadir_m"] for c in checks if c["ground_from_nadir_m"] is not None]
    log(f"probe: 俯视复核 {len(checks)} 帧，地面高 min/med/max = "
        f"{(min(g), float(np.median(g)), max(g)) if g else None} m")
    if not ok:
        raise SystemExit(f"有路线低质帧比例 > {pq.LOW_FRAC_MAX:.0%}（看 {out_dir}/route_*.png 改路线）")


# ---------------------------------------------------------------- 5 教师冷烟 / 6 三视角视频 / 7 汇总
def step_smoke():
    """collect_3dgs --policy astar --dynamics cf_sitl --hold 0.5 --stop-on-collision，每路线 SMOKE_EPISODES 集
    （与已有四场景冷烟同口径），最多 SMOKE_PARALLEL 个并行；有 run.json 的路线复用，半成品目录删掉重采。"""
    scene = read_json(SCENE_JSON)
    out_root = os.path.join(SMOKE_ROOT, SCENE)
    os.makedirs(out_root, exist_ok=True)
    jobs = []
    for i, r in enumerate(scene["routes"]):
        out = os.path.join(out_root, f"r{i}")
        if os.path.isfile(os.path.join(out, "run.json")):
            log(f"smoke: r{i} 已有 run.json，复用")
            continue
        if os.path.isdir(out):
            shutil.rmtree(out)                          # collect_3dgs 要求空目录
        jobs.append((f"r{i} {r['name']}", out, [PY, "-m", "sim.3dgs.collect_3dgs", "--scene", SCENE_JSON, "--out", out,
                                               "--route", str(i), "--policy", "astar", "--dynamics", "cf_sitl",
                                               "--episodes", str(SMOKE_EPISODES), "--seed", str(SEED0 + 10 * i),
                                               "--hold", str(HOLD_S), "--stop-on-collision"]))
    running, failed = [], []
    while jobs or running:
        while jobs and len(running) < SMOKE_PARALLEL:
            what, out, cmd = jobs.pop(0)
            logf = open(out + ".log", "w")
            log(f"smoke: 起 {what}: {' '.join(cmd)}（日志 {out}.log）")
            running.append((what, subprocess.Popen(cmd, cwd=REPO, env=ENV, stdout=logf, stderr=subprocess.STDOUT), logf, time.time()))
        time.sleep(5)
        for item in [x for x in running if x[1].poll() is not None]:
            what, p, logf, t0 = item
            logf.close()
            running.remove(item)
            log(f"smoke: {what} rc={p.returncode} wall={time.time() - t0:.0f}s")
            if p.returncode != 0:
                failed.append(what)
    if failed:
        raise SystemExit(f"collect_3dgs 失败：{failed}")


def step_video():
    eps = [os.path.join(SMOKE_ROOT, SCENE, f"r{i}", "ep_00000") for i in VIDEO_ROUTES]
    eps = [e for e in eps if os.path.isfile(os.path.join(e, "trace.jsonl"))]
    if not eps:
        raise SystemExit("video: VIDEO_ROUTES 没有可用的 ep_00000/trace.jsonl（先跑 smoke）")
    if all(os.path.isfile(os.path.join(e, "views.mp4")) for e in eps):
        log("video: views.mp4 都在，跳过")
        return
    run([PY, "-m", "sim.3dgs.video", "--scene", SCENE_JSON, "--ep", *eps, "--video-quality", VIDEO_QUALITY], "video 三视角")


def step_summary():
    """逐集摘要（demo_3dgs_routes.episode_row）+ 每路线过线判定（≥1 集到达、0 碰撞）+ 低质帧比例 -> smoke_summary.json/.md。
    到达 = 末位置距航点 ≤ reach_radius（SCENE-B 口径，比「轨迹任一点进圈」严）。"""
    demo = importlib.import_module("demo_3dgs_routes")
    scene = read_json(SCENE_JSON)
    reach = float(scene["reach_radius_m"])
    out_root = os.path.join(SMOKE_ROOT, SCENE)
    rows, verdict = [], []
    for i, r in enumerate(scene["routes"]):
        eps = sorted(glob.glob(os.path.join(out_root, f"r{i}", "ep_*")))
        mine = []
        for ep in eps:
            row = demo.episode_row(ep, reach)
            acts = demo.read_jsonl(os.path.join(ep, "actions.jsonl"))
            zs = [a["pose_gt"]["z"] for a in acts]
            row.update(route_index=i, route_name=r["name"], reached_final=row["final_dist_m"] is not None and row["final_dist_m"] <= reach,
                       z_range=[round(min(zs), 2), round(max(zs), 2)] if zs else None,
                       lapvar_min=round(min(a["frame_quality"]["lapvar"] for a in acts), 1) if acts else None,
                       wall_s=read_json(os.path.join(ep, "episode_manifest.json"))["performance"]["wall_s"],
                       video=os.path.isfile(os.path.join(ep, "views.mp4")))
            mine.append(row)
        rows += mine
        verdict.append({"route": i, "name": r["name"], "episodes": len(mine), "reached": sum(x["reached_final"] for x in mine),
                        "collisions": sum(x["collision"] for x in mine),
                        "pass": len(mine) == SMOKE_EPISODES and any(x["reached_final"] for x in mine) and not any(x["collision"] for x in mine)})
    n_frames = sum(x["steps"] for x in rows)
    n_low = sum(x["n_low_quality"] or 0 for x in rows)
    summary = {"scene": SCENE_JSON, "reach_radius_m": reach, "episodes": len(rows), "frames": n_frames, "low_frames": n_low,
               "low_frac": round(n_low / n_frames, 4) if n_frames else None, "routes": verdict, "rows": rows}
    write_json(os.path.join(out_root, "smoke_summary.json"), summary)
    lines = [f"# {SCENE} 冷烟（astar + cf_sitl，hold {HOLD_S}s，--stop-on-collision，到达圈 {reach} m）", "",
             "| 路线 | seed | 步数/预算 | 碰撞 | goal-stop | 终点距航点 m | 到达 | z 范围 m | 低质帧 | lapvar min | 最小净空 m | 动力学异常 | 墙钟 s | 视频 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        an = x["anomaly"]
        lines.append(f"| r{x['route_index']} {x['route_name']} | {x['seed']} | {x['steps']}/{x['requested_steps']} | {'是' if x['collision'] else '否'} | "
                     f"{'是' if x['stopped_on_goal'] else '否'} | {x['final_dist_m']} | {'是' if x['reached_final'] else '否'} | "
                     f"{x['z_range'][0]}–{x['z_range'][1]} | {x['n_low_quality']} | {x['lapvar_min']} | {x['min_clearance_m']} | "
                     f"{an['kind'] + '@' + str(an['step']) if an else '-'} | {x['wall_s']} | {'有' if x['video'] else '-'} |")
    lines += ["", f"{len(rows)} 集：到达 {sum(x['reached_final'] for x in rows)}，碰撞 {sum(x['collision'] for x in rows)}，"
              f"goal-stop {sum(bool(x['stopped_on_goal']) for x in rows)}，低质帧 {n_low}/{n_frames}，"
              f"动力学异常 {sum(x['anomaly'] is not None for x in rows)}；路线过线 {sum(v['pass'] for v in verdict)}/{len(verdict)}"]
    with open(os.path.join(out_root, "smoke_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log("summary: " + lines[-1])
    if not all(v["pass"] for v in verdict):
        raise SystemExit(f"有路线没过冷烟线：{[v['name'] for v in verdict if not v['pass']]}")


# ---------------------------------------------------------------- 主流程
def main():
    os.environ["TORCH_EXTENSIONS_DIR"] = torch_ext_dir()   # 进程内渲染（probe 步）与子进程同一个 JIT 缓存目录
    os.environ.pop("TORCH_CUDA_ARCH_LIST", None)          # 带着它，build.ninja 的 -gencode 变了就整目录重编
    os.environ.setdefault("TMPDIR", os.path.join(DATA_ROOT, "uav_rgb", "tmp"))   # 根分区只剩几 GB
    os.environ["CF_SITL_PROC"] = CF_SITL_PROC
    ENV.clear()
    ENV.update(os.environ, PYTHONDONTWRITEBYTECODE="1")
    os.makedirs(OUT, exist_ok=True)
    t0, rc = time.time(), 0
    log(f"scene={SCENE} steps={','.join(RUN_STEPS)} TORCH_EXTENSIONS_DIR={ENV['TORCH_EXTENSIONS_DIR']} CF_SITL_PROC={CF_SITL_PROC}")
    try:
        for step in STEPS:
            if step in RUN_STEPS:
                log(f"===== {step} =====")
                globals()[f"step_{step}"]()
    except SystemExit as e:
        rc = 1 if e.code not in (None, 0) else 0
        if rc:
            log(f"停止：{e}")
    except Exception as e:                       # noqa: BLE001 守护进程里要把异常写进日志再落 DONE
        rc = 2
        import traceback
        traceback.print_exc()
        log(f"异常：{e!r}")
    with open(os.path.join(OUT, "DONE"), "w") as f:
        f.write(f"rc={rc} {time.strftime('%Y-%m-%d %H:%M:%S')} wall={time.time() - t0:.0f}s steps={','.join(RUN_STEPS)}\n")
    log(f"DONE rc={rc} wall={time.time() - t0:.0f}s")
    sys.exit(rc)


if __name__ == "__main__":
    main()
