#!/usr/bin/env python3
"""SAGE-3D / InteriorGS 室内 3DGS 场景接入：下载 -> 转 PLY -> 地面与坐标系复核 -> scene.json + 路线过滤 -> 起点视角渲染。

一个场景一次：SCENE_ID=840865 python tools/scene_import_sage3d.py（或第 1 个参数给场景号）。每步落盘、可断点续跑（产物在就跳过）；
任一步失败即停。只跑部分步骤：IMPORT_STEPS=align,scene（顺序固定，见 STEPS）。风格照 tools/scene_import_wanderland.py。

数据来源（2026-11-10 bench 调研，tmp/phase12-20261110/reports/bench.md）：
  场景 = HF spatialverse/SAGE-3D_InteriorGS_usdz（InteriorGS 的 NuRec usdz 镜像，CC-BY-NC-4.0，gated=False）；
  路线与指令 = HF spatialverse/SAGE-3D_VLN_Data（CC-BY-4.0）。只能做研究，产物（PLY、scene.json）不再分发、不进仓库。
  usdz 转 PLY 由 tools/nurec_to_ply.py 做（要 /opt/conda/bin/python3 的 msgpack）；转完删本脚本下载的 usdz 并记名字与字节数。

跑哪个 python：download / convert 只要 numpy 以外的 curl 与 /opt/conda/bin/python3；align / scene 只要 numpy；render 要 3dgs-venv
（torch + gsplat，环境变量 TORCH_EXTENSIONS_DIR 与 CUDA_VISIBLE_DEVICES 由调用方给，见 render 步报错提示）。统一用 3dgs-venv 的 python 起即可：
  /root/rivermind-data/uav_rgb/3dgs-venv/bin/python tools/scene_import_sage3d.py 840865

SAGE VLN JSON 的结构（本脚本用到的字段）：
  scenes[0].samples[i] = {trajectory_id, instructions[{instruction_type, start, end, generated_instruction, scene_id}],
                          points[{position[x, y, z], rotation[四元数，相机约定，不用来算朝向]}]}
  points 间距 0.05 m、z 恒为 0.5（SAGE 的地面机器人高度；本项目按 FLY_H_M 另算飞行高度）。
  坐标系：轨迹坐标 = InteriorGS 场景的导航系；与 PLY（世界系）差绕 z 轴 180°、无平移（bench 3.2 统计检验；本脚本对每个新场景复核）。
"""
import gzip  # noqa: F401  (保持与 nurec_to_ply 同款依赖声明，下载/转换在子进程里做)
import importlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

# ---------------------------------------------------------------- 配置（改这里）
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # 仓库根（本文件在 tools/）
DATA_ROOT = os.environ.get("DATA_ROOT", "/root/rivermind-data")
PY = os.environ.get("UAV_RGB_3DGS_PYTHON", os.path.join(DATA_ROOT, "uav_rgb", "3dgs-venv", "bin", "python"))
CONDA_PY = "/opt/conda/bin/python3"                                    # nurec_to_ply 要 msgpack，只有它有
HF_MIRROR = "https://hf-mirror.com"
HF_USDZ_REPO = "spatialverse/SAGE-3D_InteriorGS_usdz"
HF_VLN_REPO = "spatialverse/SAGE-3D_VLN_Data"
VLN_TEST_SUBSETS = ("Instruction_Unseen", "Scene_Unseen", "Trajectory_Unseen")   # test/Trajectory_test/Trajectory_test_<子集>/<场景号>/
PUB_DIR = os.path.join(DATA_ROOT, "uav_rgb", "pubdata", "sage3d")      # 下载物：usdz/<id>.usdz（转完删）、vln/<id>/<划分>/*.json
SCENES_DIR = os.path.join(DATA_ROOT, "uav_rgb", "3dgs-scenes")
OUT_ROOT = os.environ.get("IMPORT_OUT", os.path.join(DATA_ROOT, "uav_rgb", "tmp", "phase12-20261110", "sage3d", "import"))
SCENE_ID = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SCENE_ID", ""))
STEPS = ("download", "convert", "align", "scene", "render")
RUN_STEPS = tuple(s for s in os.environ.get("IMPORT_STEPS", ",".join(STEPS)).split(",") if s)
DELETE_USDZ = True                  # 转 PLY 成功后删本脚本下载的 usdz（先记名字与字节数到 PUB_DIR/usdz_deleted.jsonl）
BACKOFF_S = 20                      # 下载遇 429 或其他非 200：第 n 次失败后等 n×20 s
DOWNLOAD_TRIES = 6

# 飞行高度。cf_sitl 的起飞高度取自 spawn.z − ground_z（sim/3dgs/dynamics.py CfSitlDynamics.takeoff_height，须在 (0, 2.5) m），
# dynamics_profile 只取决于 e1_mini.json 的值与库版本，所以高度可以按场景给，不改固件（报告 §1，本文件测试里锁定）。
FLY_H_M = 1.2                       # 室内门洞高约 2.0–2.1 m、桌面 0.7–0.8 m；1.2 m 在门框上沿以下、桌面以上（bench 试点同值）
Z_SLICE_HALF_M = 0.2                # 占据切片 = 飞行高度 ± 0.2 m（机体半高 0.028 m，另留定高误差与抖动）
VOXEL_M = 0.1                       # 室内净空只有 0.2–0.4 m：0.1 m 体素（Wanderland 室外用 0.2 m）
CAMERA = {"width": 320, "height": 240, "fov_deg": 60.0, "pitch_deg": 0.0}   # 与 Gazebo / 其他 3DGS 场景同口径
REACH_RADIUS_M = 1.0                # bench 试点同值；室内目标物体附近 1 m 内算到点
HOLD_S = 0.5                        # scene.step_s；route_check 的步数估计按这个口径
NAV_ROT_Z_DEG = 180.0               # SAGE 轨迹系 -> PLY 系：绕 z 轴 180°、零平移
# 地面：z 直方图。仅看不透明度 >= 0.5、尺度 <= 3 m 的高斯；地面带 = z 范围（0.5–99.5 分位）的最低 40%，取其中最高的桶
GROUND_BIN_M = 0.05
GROUND_OPACITY_MIN = 0.5
GROUND_MAX_SCALE_M = 3.0
GROUND_BAND_FRAC = 0.4
CEILING_MIN_ABOVE_M = 1.8           # 天花板峰只在地面以上 >= 1.8 m 处找（只记录，不参与飞行）
# 路线过滤（顶部配置区；三条是任务书定的，第四条是本脚本追加的，理由在注释里）
ROUTE_CLEAR_MARGIN_M = 0.1          # 沿 A* 路径最小净空 >= 机体半径 + 0.1 m（route_check 默认 0.3 m 对室内走廊过严）
ROUTE_ASTAR_OVER_SAGE_MAX = 1.3     # A* 长度 <= 1.3 × SAGE 轨迹长度（A* 绕远太多说明两者走的不是同一条通道）
# A* 有路径：route_check 的 feasible 与 A* 是否找到（见 judge_route）
ROUTE_MIN_GOAL_DIST_M = 2.0         # 追加：起终点直线距离 >= 2×REACH_RADIUS_M，否则出生即在到达半径内、SR 恒为 1
YAW_LOOKAHEAD_M = 0.6               # 出生朝向 = 朝 SAGE 路径上离起点 0.6 m 的点（bench 用第 12 点 = 0.6 m）
# 坐标系复核（bench frame_test6 的统计量）：墙层栅格 0.05 m，膨胀 0.15 m，轨迹点按 4 个旋转 × ±1.5 m 平移搜索
FRAME_RES_M = 0.05
FRAME_DILATE_M = 0.15
FRAME_WALL_OPACITY = 0.5
FRAME_SHIFT_M = 1.5
FRAME_SHIFT_STEP_M = 0.05
FRAME_RATIO_MAX = 0.7               # 180° 的零平移命中率 <= 0.7 × 第二名（bench 841150：0.11 对 0.24）
FRAME_SHIFT_OK_M = 0.5              # 180° 的最优平移 <= 0.5 m（零平移假设）
FRAME_MAX_POINTS = 20000            # 轨迹点最多取这么多（等间隔抽稀）
RENDER_VIEWS = 4                    # 每场景渲染几张路线起点视角
SPLIT_ORDER = ("test_Instruction_Unseen", "test_Scene_Unseen", "test_Trajectory_Unseen", "train")   # 路线排序：测试划分在前
SPLIT_SHORT = {"test_Instruction_Unseen": "ti", "test_Scene_Unseen": "ts", "test_Trajectory_Unseen": "tt", "train": "tr"}
LICENSE_TEXT = ("CC-BY-NC-4.0（SAGE-3D_InteriorGS_usdz；InteriorGS 使用条款：仅限非商业科研教学，不得再分发）；"
                "路线与指令来自 SAGE-3D_VLN_Data（CC-BY-4.0）。产物只用于研究，不再分发。")

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))
scene_mod = importlib.import_module("sim.3dgs.scene")
collision = importlib.import_module("sim.3dgs.collision")
planner_mod = importlib.import_module("sim.3dgs.planner")


# ---------------------------------------------------------------- 通用
def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def scene_name(sid):
    return f"sage3d_{sid}"


def scene_dir(sid):
    return os.path.join(SCENES_DIR, scene_name(sid))


def out_dir(sid):
    return os.path.join(OUT_ROOT, scene_name(sid))


def dist(values):
    """n / min / p10 / p50 / p90 / max（空列表只给 n=0）。"""
    v = np.asarray(list(values), dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    p = np.percentile(v, [10, 50, 90])
    return {"n": int(v.size), "min": round(float(v.min()), 3), "p10": round(float(p[0]), 3), "p50": round(float(p[1]), 3),
            "p90": round(float(p[2]), 3), "max": round(float(v.max()), 3)}


def spread_indices(n, k):
    """0..n-1 里取 k 个等间隔下标（含首尾）；n <= k 全取。用来按长度分位挑要渲染的路线。"""
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


def rz_matrix(deg):
    """绕 z 轴转 deg 的 4×4（nav_to_world 的旋转块）；舍入到 1e-12，180° 给精确的 -1 / 0。"""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    c, s = round(c, 12) + 0.0, round(s, 12) + 0.0
    return [[c, 0.0 - s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


# ---------------------------------------------------------------- 1 下载
def curl_download(url, dst, expect_size, run=subprocess.run, sleep=time.sleep):
    """curl -L 单线程；非 200 或字节数不符就删掉重来，第 n 次失败后等 n×BACKOFF_S s（429 退避），共 DOWNLOAD_TRIES 次。
    dst 已在且字节数相符直接返回（expect_size=None 时只要文件在）。"""
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    if os.path.isfile(dst) and (expect_size is None or os.path.getsize(dst) == expect_size):
        return dst
    for attempt in range(1, DOWNLOAD_TRIES + 1):
        res = run(["curl", "-sS", "-L", "-o", dst, "-w", "%{http_code}", url], capture_output=True, text=True)
        code = (res.stdout or "").strip()
        good = code == "200" and os.path.isfile(dst) and (expect_size is None or os.path.getsize(dst) == expect_size)
        if good:
            return dst
        got = os.path.getsize(dst) if os.path.isfile(dst) else None
        log(f"download: {url} 第 {attempt}/{DOWNLOAD_TRIES} 次 http={code} 字节={got} 期望={expect_size}")
        if os.path.isfile(dst):
            os.remove(dst)
        if attempt < DOWNLOAD_TRIES:
            sleep(BACKOFF_S * attempt)
    raise RuntimeError(f"下载失败（{DOWNLOAD_TRIES} 次）: {url}")


def hf_resolve(repo, path):
    return f"{HF_MIRROR}/datasets/{repo}/resolve/main/{path}"


def hf_head_size(url):
    """HEAD（跟随跳转）取最终 Content-Length；取不到返回 None。"""
    out = subprocess.run(["curl", "-sIL", url], capture_output=True, text=True).stdout
    sizes = [int(line.split(":", 1)[1]) for line in out.splitlines() if line.lower().startswith("content-length:")]
    return sizes[-1] if sizes else None


def hf_list_dir(repo, path):
    """HF tree API 列一个目录下的文件 [(path, size)]；目录不存在（404）返回 []。"""
    url = f"{HF_MIRROR}/api/datasets/{repo}/tree/main/{path}"
    res = subprocess.run(["curl", "-sS", "-L", "-w", "\n%{http_code}", url], capture_output=True, text=True)
    body, _, code = res.stdout.rpartition("\n")
    if code.strip() != "200":
        return []
    return [(x["path"], int(x.get("lfs", {}).get("size", x.get("size", 0)))) for x in json.loads(body) if x.get("type") == "file"]


def vln_paths(sid):
    """场景 sid 的 VLN 轨迹 JSON 在 HF 上的路径：[(划分标签, 仓库内路径, 字节数)]。训练划分只有 929 个训练场景有，测试划分只有 45 个场景有。"""
    found = []
    for sub in VLN_TEST_SUBSETS:
        for p, size in hf_list_dir(HF_VLN_REPO, f"test/Trajectory_test/Trajectory_test_{sub}/{sid}"):
            found.append((f"test_{sub}", p, size))
    for p, size in hf_list_dir(HF_VLN_REPO, f"train/Trajectory_train/{sid}"):
        found.append(("train", p, size))
    return found


def step_download(sid):
    os.makedirs(os.path.join(PUB_DIR, "vln"), exist_ok=True)
    for tag, p, size in vln_paths(sid):
        dst = os.path.join(PUB_DIR, "vln", sid, tag, os.path.basename(p))
        curl_download(hf_resolve(HF_VLN_REPO, p), dst, size)
        log(f"download: VLN {tag} {os.path.basename(p)} {size} B")
    if os.path.isfile(os.path.join(scene_dir(sid), "scene.ply")):
        log("download: scene.ply 已在，跳过 usdz")
        return
    rel = f"InteriorGS_usdz/{sid}.usdz"
    url = hf_resolve(HF_USDZ_REPO, rel)
    size = hf_head_size(url)
    if size is None:
        raise SystemExit(f"HEAD 取不到 {url} 的字节数")
    if size > 200 * 1024 * 1024:
        raise SystemExit(f"{rel} {size} B 超过单次下载 200 MB 上限，换场景")
    curl_download(url, os.path.join(PUB_DIR, "usdz", f"{sid}.usdz"), size)
    log(f"download: usdz {sid} {size} B")


# ---------------------------------------------------------------- 2 转 PLY
def ply_header_ok(path):
    """PLY 头里的顶点数 × 62 个 float32 + 头长 == 文件字节数。"""
    n = hl = 0
    with open(path, "rb") as f:
        for line in f:
            hl += len(line)
            t = line.decode("ascii", "replace").strip()
            if t.startswith("element vertex"):
                n = int(t.split()[2])
            if t == "end_header":
                break
    return os.path.getsize(path) == hl + n * 62 * 4, n


def step_convert(sid):
    ply = os.path.join(scene_dir(sid), "scene.ply")
    if os.path.isfile(ply):
        log(f"convert: {ply} 已在，跳过")
        return
    usdz = os.path.join(PUB_DIR, "usdz", f"{sid}.usdz")
    os.makedirs(scene_dir(sid), exist_ok=True)
    res = subprocess.run([CONDA_PY, os.path.join(REPO, "tools", "nurec_to_ply.py"), usdz, ply],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"nurec_to_ply 失败 rc={res.returncode}: {res.stderr[-800:]}")
    ok, n = ply_header_ok(ply)
    if not ok:
        raise SystemExit(f"{ply} 的体积与 PLY 头不符")
    log(f"convert: {n} 个高斯，{os.path.getsize(ply)} B")
    if DELETE_USDZ and os.path.dirname(usdz) == os.path.join(PUB_DIR, "usdz"):
        rec = {"file": usdz, "bytes": os.path.getsize(usdz), "deleted_at": time.strftime("%F %T"), "ply": ply,
               "ply_bytes": os.path.getsize(ply), "n_gaussians": n}
        with open(os.path.join(PUB_DIR, "usdz_deleted.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.remove(usdz)
        log(f"convert: 已删 usdz {rec['file']}（{rec['bytes']} B，已记入 usdz_deleted.jsonl）")


# ---------------------------------------------------------------- 3 地面与坐标系
def ground_from_z_hist(z, opacity=None, bin_m=GROUND_BIN_M, opacity_min=GROUND_OPACITY_MIN, band_frac=GROUND_BAND_FRAC,
                       ceiling_min_above_m=CEILING_MIN_ABOVE_M):
    """z 直方图找地面：只留不透明度 >= opacity_min 的高斯，z 的 0.5–99.5 分位范围里最低的 band_frac 内取最高的桶；
    ground_z = 该桶 ±1 桶内高斯 z 的中位数。天花板峰在地面 + ceiling_min_above_m 以上取最高的桶（只记录）。"""
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if opacity is not None:
        z = z[np.asarray(opacity, dtype=np.float64).reshape(-1) >= opacity_min]
    z = z[np.isfinite(z)]
    if z.size < 1000:
        raise ValueError(f"可用高斯只有 {z.size} 个，不足以做 z 直方图")
    lo, hi = np.percentile(z, [0.5, 99.5])
    n = max(2, int(math.ceil((hi - lo) / bin_m)))
    counts, edges = np.histogram(z, bins=n, range=(lo, lo + n * bin_m))
    centers = edges[:-1] + bin_m / 2
    band = centers <= lo + band_frac * (hi - lo)
    i = int(np.argmax(np.where(band, counts, -1)))
    near = np.abs(z - centers[i]) <= bin_m
    ground = float(np.median(z[near]))
    out = {"ground_z": round(ground, 3) + 0.0, "floor_frac": round(float(near.sum() / z.size), 4), "n_used": int(z.size),
           "bin_m": bin_m, "z_range_p05_p995": [round(float(lo), 3), round(float(hi), 3)],
           "global_peak_z": round(float(centers[int(np.argmax(counts))]), 3), "ceiling_z": None}
    up = centers >= ground + ceiling_min_above_m
    if up.any():
        j = int(np.argmax(np.where(up, counts, -1)))
        out["ceiling_z"] = round(float(np.median(z[np.abs(z - centers[j]) <= bin_m])), 3)
    return out


def _raster(xy, x0, y0, nx, ny, res, dil_cells):
    r = np.zeros((ny, nx), dtype=bool)
    if len(xy):
        ix = ((xy[:, 0] - x0) / res).astype(int)
        iy = ((xy[:, 1] - y0) / res).astype(int)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        r[iy[ok], ix[ok]] = True
    d = np.zeros_like(r)
    k = dil_cells
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            if dx * dx + dy * dy <= k * k:
                # 不能用 np.roll：它会把场景边界的墙环绕到另一侧，误拒绝合法路线。
                sy0, sy1 = max(0, -dy), ny - max(0, dy)
                sx0, sx1 = max(0, -dx), nx - max(0, dx)
                ty0, ty1 = max(0, dy), ny - max(0, -dy)
                tx0, tx1 = max(0, dx), nx - max(0, -dx)
                d[ty0:ty1, tx0:tx1] |= r[sy0:sy1, sx0:sx1]
    return d


def _hit_frac(pts, mask, x0, y0, res):
    ny, nx = mask.shape
    ix = ((pts[:, 0] - x0) / res).astype(int)
    iy = ((pts[:, 1] - y0) / res).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    hit = np.zeros(len(pts), dtype=bool)
    hit[ok] = mask[iy[ok], ix[ok]]
    return float(hit.mean())


def frame_check(means, opacity, scales_max, traj_xy, z_lo, z_hi, res=FRAME_RES_M, dilate_m=FRAME_DILATE_M,
                wall_opacity=FRAME_WALL_OPACITY, shift_m=FRAME_SHIFT_M, shift_step_m=FRAME_SHIFT_STEP_M):
    """轨迹（SAGE 导航系 xy）绕 z 转 0/90/180/270° 后，落进「飞行层（z_lo..z_hi）上不透明高斯栅格（膨胀 dilate_m）」的点占比；
    每个旋转再在 ±shift_m 平移里找最小占比。正确的坐标系下轨迹在走廊里，占比应明显低于其他三个旋转（bench 841150：11% 对 24–30%）。"""
    m = np.asarray(means, dtype=np.float64)
    keep = (np.asarray(opacity) > wall_opacity) & (np.asarray(scales_max) <= GROUND_MAX_SCALE_M) & (m[:, 2] > z_lo) & (m[:, 2] < z_hi)
    layer = m[keep][:, :2]
    if len(layer) < 100:
        raise ValueError(f"飞行层 z∈({z_lo:.2f}, {z_hi:.2f}) 里只有 {len(layer)} 个不透明高斯，没法做坐标系复核")
    pts = np.asarray(traj_xy, dtype=np.float64)
    if len(pts) > FRAME_MAX_POINTS:
        pts = pts[np.linspace(0, len(pts) - 1, FRAME_MAX_POINTS).astype(int)]
    k = int(round(dilate_m / res))
    pad = 1.0 + dilate_m
    x0, y0 = float(layer[:, 0].min() - pad), float(layer[:, 1].min() - pad)
    nx = int(math.ceil((layer[:, 0].max() + pad - x0) / res)) + 1
    ny = int(math.ceil((layer[:, 1].max() + pad - y0) / res)) + 1
    wall = _raster(layer, x0, y0, nx, ny, res, k)
    shifts = np.arange(-shift_m, shift_m + 1e-9, shift_step_m)
    rows = {}
    for deg in (0, 90, 180, 270):
        th = math.radians(deg)
        rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        q = pts @ rot.T
        zero = _hit_frac(q, wall, x0, y0, res)
        best = (zero, 0.0, 0.0)
        for dx in shifts:
            for dy in shifts:
                h = _hit_frac(q + (dx, dy), wall, x0, y0, res)
                if h < best[0] - 1e-12:
                    best = (h, float(dx), float(dy))
        rows[str(deg)] = {"hit_frac": round(zero, 4), "best_shift_hit_frac": round(best[0], 4),
                          "best_shift_dx": round(best[1], 3), "best_shift_dy": round(best[2], 3),
                          "best_shift_m": round(math.hypot(best[1], best[2]), 3)}
    best_deg = int(min(rows, key=lambda d: rows[d]["hit_frac"]))
    return {"rotations": rows, "best_deg": best_deg, "n_points": int(len(pts)), "n_layer_gaussians": int(len(layer))}


def frame_verdict(res):
    """(通过?, 说明)：180° 的零平移命中率最小、<= FRAME_RATIO_MAX × 第二名，且 180° 的最优平移 <= FRAME_SHIFT_OK_M。"""
    rot = res["rotations"]
    h = {int(k): v["hit_frac"] for k, v in rot.items()}
    desc = "命中率 " + " ".join(f"{d}°:{h[d]:.3f}" for d in sorted(h))
    if res["best_deg"] != 180:
        return False, f"最优旋转是 {res['best_deg']}°，不是 180°（{desc}）"
    second = min(v for d, v in h.items() if d != 180)
    if h[180] > FRAME_RATIO_MAX * second:
        return False, f"180° 命中率 {h[180]:.3f} 与第二名 {second:.3f} 之比 > {FRAME_RATIO_MAX}，区分度不够（{desc}）"
    if rot["180"]["best_shift_m"] > FRAME_SHIFT_OK_M:
        return False, f"180° 的最优平移 {rot['180']['best_shift_m']} m > {FRAME_SHIFT_OK_M} m，零平移假设不成立（{desc}）"
    return True, f"ok（{desc}；180° 最优平移 {rot['180']['best_shift_m']} m）"


def occupied_area_m2(xy, cell=0.1):
    if len(xy) == 0:
        return 0.0
    c = np.floor(np.asarray(xy) / cell).astype(np.int64)
    return float(len(np.unique(c, axis=0))) * cell * cell


def step_align(sid):
    ply = os.path.join(scene_dir(sid), "scene.ply")
    cands = load_candidates(sid)
    if not cands:
        raise SystemExit(f"{PUB_DIR}/vln/{sid} 下没有 VLN JSON，先跑 download")
    g = scene_mod.read_gaussian_ply(ply)
    m, op, sc = g["means"].astype(np.float64), np.asarray(g["opacities"]), np.asarray(g["scales"]).max(axis=1)
    ok = sc <= GROUND_MAX_SCALE_M
    gr = ground_from_z_hist(m[ok][:, 2], op[ok])
    ground = gr["ground_z"]
    traj = np.concatenate([np.array([c["start"], c["goal"]]) for c in cands] + [c["points_xy"][::5] for c in cands])
    z_lo, z_hi = ground + FLY_H_M - 0.1, ground + FLY_H_M + 0.1
    fc = frame_check(m, op, sc, traj, z_lo, z_hi)
    verdict, why = frame_verdict(fc)
    fl = ok & (op >= 0.3) & (m[:, 2] > ground - 0.1) & (m[:, 2] < ground + 0.15)
    wl = ok & (op >= FRAME_WALL_OPACITY) & (m[:, 2] > z_lo) & (m[:, 2] < z_hi)
    align = {"scene": sid, "n_gaussians": int(len(m)), "ground": gr, "fly_height_m": FLY_H_M, "frame_check": fc,
             "frame_ok": bool(verdict), "frame_why": why, "floor_area_m2": round(occupied_area_m2(m[fl][:, :2]), 1),
             "wall_layer_extent_xy": [[round(float(v), 2) for v in m[wl][:, :2].min(axis=0)],
                                      [round(float(v), 2) for v in m[wl][:, :2].max(axis=0)]],
             "n_trajectory_points_used": fc["n_points"]}
    write_json(os.path.join(scene_dir(sid), "align.json"), align)
    log(f"align: ground_z={ground} ceiling_z={gr['ceiling_z']} floor_area={align['floor_area_m2']} m² 坐标系：{why}")
    if not verdict:
        raise SystemExit(f"坐标系复核未过：{why}（该场景不入试点，原因记报告）")


# ---------------------------------------------------------------- 4 路线
def routes_from_vln(doc, split):
    """SAGE VLN JSON -> 候选路线列表。起点 = 第一个点、终点 = 最后一个点（xy，米，SAGE 导航系），朝向 = 朝路径上离起点
    YAW_LOOKAHEAD_M 的点（几何朝向；JSON 里的 rotation 是相机约定四元数，不用）；instructions 全部保留。"""
    out = []
    for sc in doc.get("scenes", []):
        for smp in sc.get("samples", []):
            pts = np.array([p["position"][:2] for p in smp.get("points", [])], dtype=np.float64)
            if len(pts) < 2:
                continue
            seg = np.hypot(*np.diff(pts, axis=0).T)
            length = float(seg.sum())
            if length <= 0.0:
                continue
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            k = min(max(int(np.searchsorted(cum, YAW_LOOKAHEAD_M)), 1), len(pts) - 1)
            yaw = math.degrees(math.atan2(pts[k, 1] - pts[0, 1], pts[k, 0] - pts[0, 0]))
            out.append({"trajectory_id": int(smp["trajectory_id"]), "split": split, "n_points": int(len(pts)),
                        "sage_path_m": round(length, 3), "start": (round(float(pts[0, 0]), 4) + 0.0, round(float(pts[0, 1]), 4) + 0.0),
                        "goal": (round(float(pts[-1, 0]), 4) + 0.0, round(float(pts[-1, 1]), 4) + 0.0), "yaw_deg": round(yaw, 2) + 0.0,
                        "points_xy": pts,
                        "instructions": [{"instruction_type": i.get("instruction_type"), "start": i.get("start"),
                                          "end": i.get("end"), "text": i.get("generated_instruction")}
                                         for i in smp.get("instructions", [])]})
    return out


def load_candidates(sid):
    """PUB_DIR/vln/<id>/<划分>/*.json 全部读进来，候选路线按 order_candidates 排序。"""
    root = os.path.join(PUB_DIR, "vln", sid)
    cands = []
    if os.path.isdir(root):
        for tag in sorted(os.listdir(root)):
            for fn in sorted(os.listdir(os.path.join(root, tag))):
                if fn.endswith(".json"):
                    with open(os.path.join(root, tag, fn), encoding="utf-8") as f:
                        cands += routes_from_vln(json.load(f), tag)
    return order_candidates(cands)


def order_candidates(cands):
    return sorted(cands, key=lambda c: (SPLIT_ORDER.index(c["split"]), c["trajectory_id"]))


def route_name(c):
    return f"{SPLIT_SHORT[c['split']]}_{c['trajectory_id']}"


REASON_LABELS = (("A* 无路径", "A*无路径"), ("起点被占据", "起点被占据"), ("终点被占据", "终点被占据"), ("净空", "净空不足"),
                 ("A* 长度", "A*长度比超限"), ("起终点直线距离", "起终点太近"))


def reason_label(why):
    """judge_route 的原因文字 -> 汇总用的固定标签（报告里的原因计数按它分）。"""
    for prefix, label in REASON_LABELS:
        if why.startswith(prefix):
            return label
    raise ValueError(f"未登记的过滤原因: {why!r}")


def judge_route(chk, cand):
    """route_check 结果 + 候选路线 -> (通过?, 未过原因列表)。规则见配置区 ROUTE_*：A* 有路径且起终点自由、净空、A*/SAGE 长度比、起终点直线距离。"""
    why = []
    if chk.get("reason") == "A* 无路径":
        why.append("A* 无路径")
    if not chk.get("spawn_free", True):
        why.append("起点被占据")
    if not chk.get("waypoint_free", True):
        why.append("终点被占据")
    if chk.get("path_m") is not None:
        if not chk.get("clear_ok"):
            why.append(f"净空 {chk.get('min_clearance_m')} m < 机体半径 + {ROUTE_CLEAR_MARGIN_M} m")
        ratio = chk["path_m"] / cand["sage_path_m"]
        if ratio > ROUTE_ASTAR_OVER_SAGE_MAX + 1e-9:
            why.append(f"A* 长度 {chk['path_m']} m 是 SAGE 长度 {cand['sage_path_m']} m 的 {ratio:.2f} 倍 > {ROUTE_ASTAR_OVER_SAGE_MAX}")
    straight = math.hypot(cand["goal"][0] - cand["start"][0], cand["goal"][1] - cand["start"][1])
    if straight < ROUTE_MIN_GOAL_DIST_M - 1e-9:
        why.append(f"起终点直线距离 {straight:.2f} m < {ROUTE_MIN_GOAL_DIST_M} m")
    return (not why), why


def evaluate_route(planner, cand, z, terrain_occ=None):
    spawn = {"x": cand["start"][0], "y": cand["start"][1], "z": z, "yaw_deg": cand["yaw_deg"]}
    waypoint = {"x": cand["goal"][0], "y": cand["goal"][1], "z": z}
    name = route_name(cand)
    chk = planner_mod.route_check(planner, spawn, waypoint, margin_m=ROUTE_CLEAR_MARGIN_M, terrain_occ=terrain_occ,
                                  route_name=name)
    ok, reasons = judge_route(chk, cand)
    return {"ok": ok, "reasons": reasons, "chk": chk, "spawn": spawn, "waypoint": waypoint, "name": name}


def build_scene(sid, ground_z, routes, extra_notes=None):
    """scene.json 的字典。routes[0] 同时作为顶层 spawn/waypoint/max_steps（apply_route 的约定）。"""
    z_c = float(ground_z) + FLY_H_M
    first = routes[0]
    notes = [
        f"飞行高度：nav z = ground_z + {FLY_H_M} m（spawn.z、占据切片中心、渲染高度三处同值，来自配置区 FLY_H_M）；"
        "cf_sitl 起飞高度 = spawn.z − ground_z，dynamics_profile 与高度无关。",
        f"nav_to_world = 绕 z 轴 {NAV_ROT_Z_DEG:g}°、零平移：SAGE 轨迹系与 PLY 世界系差这个旋转（每个场景做统计复核，结果在 align.json）。",
        "碰撞源 collision.source=gaussians：飞行层高斯在 nav 系按 0.1 m 体素占据（代理，非真值；同层 NuRec 高斯里的漂浮物会造成假障碍）。",
        "routes[]：起点 = SAGE 轨迹第一点、终点 = 最后一点（xy），朝向朝路径 0.6 m 处；max_steps = planner.route_check.steps_estimate（hold 0.5 口径）。"
        "routes[].sage 里有轨迹号、划分、SAGE 路径长、A* 长度、最小净空、全部指令（字段 instruction_type/start/end/text）。",
        "ToF 平地模型对飞行层以下的家具不成立（route_check 的 agl_model_unverified 只打标）；室内 1.2 m 飞越桌面时真机 ToF 会读短。",
    ] + list(extra_notes or [])
    return {
        "schema": "uavrgb.3dgs_scene.v1",
        "name": scene_name(sid),
        "source": f"SAGE-3D_InteriorGS_usdz 场景 {sid}（HF {HF_USDZ_REPO}，NuRec usdz 经 tools/nurec_to_ply.py 转 INRIA PLY）；"
                  f"路线与指令来自 HF {HF_VLN_REPO}",
        "license": LICENSE_TEXT,
        "ply": "scene.ply",
        "nav_to_world": rz_matrix(NAV_ROT_Z_DEG),
        "world_up": "z",
        "ground_z": float(ground_z),
        "spawn": dict(first["spawn"]), "waypoint": dict(first["waypoint"]),
        "reach_radius_m": REACH_RADIUS_M,
        "camera": dict(CAMERA, fov_source="与 Gazebo 及其他 3DGS 场景统一 60° HFOV / 320×240"),
        "collision": {"z_slice": [round(z_c - Z_SLICE_HALF_M, 6), round(z_c + Z_SLICE_HALF_M, 6)], "voxel_m": VOXEL_M,
                      "envelope": "e1_mini", "source": "gaussians"},
        "max_steps": first["max_steps"],
        "step_s": HOLD_S,
        "fly_height_m": FLY_H_M,
        "notes": notes,
        "routes": routes,
    }


def step_scene(sid):
    align = json.load(open(os.path.join(scene_dir(sid), "align.json"), encoding="utf-8"))
    if not align["frame_ok"]:
        raise SystemExit(f"align.json 的坐标系复核未过：{align['frame_why']}")
    ground = align["ground"]["ground_z"]
    z = ground + FLY_H_M
    cands = load_candidates(sid)
    first = cands[0]
    tmp_route = {"name": route_name(first), "spawn": {"x": first["start"][0], "y": first["start"][1], "z": z,
                                                      "yaw_deg": first["yaw_deg"]},
                 "waypoint": {"x": first["goal"][0], "y": first["goal"][1], "z": z}, "max_steps": 60, "note": "占位"}
    path = os.path.join(scene_dir(sid), "scene.json")
    scene_mod.save_scene(build_scene(sid, ground, [tmp_route]), path)
    loaded = scene_mod.load_scene(path)
    g = scene_mod.read_gaussian_ply(loaded["ply_path"])
    t0 = time.time()
    occ = collision.from_scene(loaded, gaussians=g)
    terrain = collision.terrain_from_scene(loaded, gaussians=g)
    log(f"scene: 占据（{occ.source}）{occ.summary()['n_occupied_cells']} 格，{time.time() - t0:.1f}s；候选路线 {len(cands)} 条")
    planner = planner_mod.OccPlanner(occ)
    rows, routes = [], []
    for c in cands:
        r = evaluate_route(planner, c, z, terrain_occ=terrain)
        chk = {k: v for k, v in r["chk"].items() if k != "path"}
        rows.append({"name": r["name"], "trajectory_id": c["trajectory_id"], "split": c["split"], "sage_path_m": c["sage_path_m"],
                     "n_points": c["n_points"], "n_instructions": len(c["instructions"]), "ok": r["ok"], "reasons": r["reasons"], **chk})
        if not r["ok"]:
            continue
        routes.append({"name": r["name"], "spawn": r["spawn"], "waypoint": r["waypoint"],
                       "max_steps": int(r["chk"]["steps_estimate"]),
                       "note": f"SAGE {c['split']} 轨迹 {c['trajectory_id']}，SAGE 路径 {c['sage_path_m']} m，A* {r['chk']['path_m']} m",
                       "sage": {"trajectory_id": c["trajectory_id"], "split": c["split"], "sage_path_m": c["sage_path_m"],
                                "n_points": c["n_points"], "astar_path_m": r["chk"]["path_m"], "turn_deg": r["chk"]["turn_deg"],
                                "min_clearance_m": r["chk"]["min_clearance_m"],
                                "agl_model_unverified": bool(r["chk"]["agl_model_unverified"]),
                                "tof_terrain_m": r["chk"]["tof_terrain_m"], "instructions": c["instructions"]}})
    reasons = {}
    for row in rows:
        for w in row["reasons"]:
            reasons[reason_label(w)] = reasons.get(reason_label(w), 0) + 1
    summary = {"scene": sid, "n_candidates": len(cands), "n_ok": len(routes),
               "n_by_split": {s: sum(c["split"] == s for c in cands) for s in SPLIT_ORDER},
               "n_ok_by_split": {s: sum(r["sage"]["split"] == s for r in routes) for s in SPLIT_ORDER},
               "reason_counts": reasons,
               "sage_path_m_all": dist(c["sage_path_m"] for c in cands),
               "min_clearance_m_all": dist(r["min_clearance_m"] for r in rows if r.get("min_clearance_m") is not None),
               "sage_path_m_ok": dist(r["sage"]["sage_path_m"] for r in routes),
               "astar_path_m_ok": dist(r["sage"]["astar_path_m"] for r in routes),
               "min_clearance_m_ok": dist(r["sage"]["min_clearance_m"] for r in routes),
               "steps_estimate_ok": dist(r["max_steps"] for r in routes),
               "instructions_ok": int(sum(len(r["sage"]["instructions"]) for r in routes)),
               "instructions_all": int(sum(len(c["instructions"]) for c in cands)),
               "n_agl_model_unverified_ok": int(sum(r["sage"]["agl_model_unverified"] for r in routes)),
               "n_gaussians": align["n_gaussians"], "ply_bytes": os.path.getsize(loaded["ply_path"]),
               "ground_z": ground, "ceiling_z": align["ground"]["ceiling_z"], "floor_area_m2": align["floor_area_m2"]}
    write_json(os.path.join(scene_dir(sid), "routes_check.json"), rows)
    write_json(os.path.join(scene_dir(sid), "import_summary.json"), summary)
    if not routes:
        raise SystemExit(f"{len(cands)} 条候选路线没有一条通过过滤（原因计数 {reasons}），看 routes_check.json")
    notes = [f"坐标系复核：{align['frame_why']}。地面：z 直方图主峰 {ground} m（天花板峰 {align['ground']['ceiling_z']} m）。",
             f"过滤：{len(cands)} 条候选 -> {len(routes)} 条（规则见 tools/scene_import_sage3d.py 配置区 ROUTE_*）。"]
    scene_mod.save_scene(build_scene(sid, ground, routes, extra_notes=notes), path)
    log(f"scene: 写 {path}：{len(routes)}/{len(cands)} 条路线通过；A* 长度 {summary['astar_path_m_ok']}")


# ---------------------------------------------------------------- 5 渲染起点视角
def step_render(sid):
    if "TORCH_EXTENSIONS_DIR" not in os.environ:
        raise SystemExit("render 步要 gsplat：先设 TORCH_EXTENSIONS_DIR（如 $D/torch-cache/extensions-sm75）与 CUDA_VISIBLE_DEVICES（只许用 uav_rgb 的那张卡）")
    from PIL import Image
    rend = importlib.import_module("sim.3dgs.renderer")
    scene = scene_mod.load_scene(os.path.join(scene_dir(sid), "scene.json"))
    routes = sorted(scene["routes"], key=lambda r: r["sage"]["astar_path_m"])
    pick = [routes[i] for i in spread_indices(len(routes), RENDER_VIEWS)]
    rd = rend.make_renderer(scene)
    out = os.path.join(out_dir(sid), "render")
    os.makedirs(out, exist_ok=True)
    tiles, stats = [], []
    for r in pick:
        sp = r["spawn"]
        rgb, dep, ms = rd.render_pose(sp["x"], sp["y"], sp["z"], sp["yaw_deg"])
        path = os.path.join(out, f"{scene_name(sid)}_{r['name']}.png")
        Image.fromarray(rgb).save(path)
        gray = rgb.astype(np.float64).mean(axis=2)
        lap = gray[1:-1, 1:-1] * 4 - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
        valid = dep[dep > 0]
        stats.append({"route": r["name"], "png": path, "mean_rgb": [round(float(v), 1) for v in rgb.reshape(-1, 3).mean(axis=0)],
                      "lapvar": round(float(lap.var()), 1), "depth_median_m": round(float(np.median(valid)), 2) if valid.size else None,
                      "astar_path_m": r["sage"]["astar_path_m"], "render_ms": round(float(ms), 1)})
        tiles.append(rgb)
        log(f"render: {r['name']} -> {path}")
    while len(tiles) < 4:
        tiles.append(np.zeros_like(tiles[0]))
    montage = np.concatenate([np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:4], axis=1)], axis=0)
    mpath = os.path.join(out, f"{scene_name(sid)}_starts.png")
    Image.fromarray(montage).save(mpath)
    write_json(os.path.join(out, "render_stats.json"), {"scene": sid, "montage": mpath, "views": stats})


def main():
    if not SCENE_ID:
        raise SystemExit("用法：python tools/scene_import_sage3d.py <场景号>（或环境变量 SCENE_ID）；IMPORT_STEPS 选步骤")
    os.makedirs(out_dir(SCENE_ID), exist_ok=True)
    fns = {"download": step_download, "convert": step_convert, "align": step_align, "scene": step_scene, "render": step_render}
    for name in STEPS:
        if name not in RUN_STEPS:
            continue
        log(f"=== {SCENE_ID}: {name}")
        t0 = time.time()
        fns[name](SCENE_ID)
        log(f"=== {SCENE_ID}: {name} 完成 {time.time() - t0:.1f}s")
    write_json(os.path.join(out_dir(SCENE_ID), "DONE.json"), {"scene": SCENE_ID, "steps": list(RUN_STEPS),
                                                              "finished": time.strftime("%F %T")})


if __name__ == "__main__":
    main()
