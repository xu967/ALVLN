#!/usr/bin/env python3
"""Wanderland（NYU ai4ce，Apache-2.0）单场景接入：下载 -> 转 PLY -> 尺度/重力/地面对齐 -> scene.json
-> 路线校验 -> 沿路线清晰度探测 -> A* 教师冒烟 -> 三视角视频。

按 docs/DATASET_3DGS_SELF_CAPTURE.md §3 的步 5–8 走；每步落盘、可断点续跑（产物在就跳过），
任一步失败即停，结束落 $OUT/DONE（rc + 时间）。

运行（守护进程，从仓库根；3dgs-venv 的 python）：
  OUT=/root/rivermind-data/uav_rgb/tmp/agent4-scene-wanderland/import; mkdir -p $OUT
  setsid nohup /root/rivermind-data/uav_rgb/3dgs-venv/bin/python tools/scene_import_wanderland.py \
      > $OUT/run.log 2>&1 < /dev/null & echo $! > $OUT/run.pid
只跑部分步骤：IMPORT_STEPS=download,convert,align（顺序固定，见 STEPS）。换场景只改下面配置区的 SCENE_ID。

场景来源（2026-11-07 调研，tmp/phase3-20261107/reports/SCENE_A_20261107.md）：
  HF ai4ce/wanderland，公开 335 场景、不设门槛；v1 场景每个带 3dgs.ply（gsplat 训练）、raw_pcd.ply
  （MetaCam LIV-SLAM 米制稠密 LiDAR 点云）、scene.usdz（由点云体素化 + marching cubes 的碰撞网格，Isaac Sim）、
  COLMAP sparse、transforms.json、episodes.json。几何与 3DGS 同一坐标系（米制）。
"""
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

# ---------------------------------------------------------------- 配置（改这里）
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根（本文件在 tools/）
PY = os.environ.get("UAV_RGB_3DGS_PYTHON", "/root/rivermind-data/uav_rgb/3dgs-venv/bin/python")   # torch + gsplat
DATA_ROOT = os.environ.get("DATA_ROOT", "/root/rivermind-data")
HF_REPO = "ai4ce/wanderland"                       # 数据集 HF 仓库（Apache-2.0，README 与 GitHub LICENSE 同文）
SCENE_ID = "1MlUucugRAnrx86cpUuQt8fUtxFMddxju"      # evaluation_ready、manifest 无质量标记；公园边步道：栅栏 + 树 + 灯柱，
                                                   # 相机轨迹 52×17 m、往返两趟、高度起伏 0.47 m（113 个 eval 场景里最平）
TIER_PATH = "data/v1/02_evaluation_ready"          # manifest data_path 的目录（wanderland_public_manifest.csv）
SCENE_NAME = "wanderland_1MlUucug"                 # 本仓库场景名：3dgs-scenes/<SCENE_NAME>/
FILES = ["3dgs.ply", "raw_pcd.ply", "transforms.json", "episodes.json", "scene.usdz",
         "sparse/0/cameras.bin", "sparse/0/images.bin", "sparse/0/points3D.bin"]   # 不下 images/fisheye（1.1 GB，渲染用不到）
DOWNLOAD_DIR = os.path.join(DATA_ROOT, "uav_rgb", "downloads", "wanderland", SCENE_ID)
SCENE_DIR = os.path.join(DATA_ROOT, "uav_rgb", "3dgs-scenes", SCENE_NAME)
OUT = os.environ.get("IMPORT_OUT", os.path.join(DATA_ROOT, "uav_rgb", "tmp", "agent4-scene-wanderland", "import"))
STEPS = ("download", "convert", "align", "scene", "probe", "smoke", "video")
RUN_STEPS = tuple(s for s in os.environ.get("IMPORT_STEPS", ",".join(STEPS)).split(",") if s)
DOWNLOAD_PARALLEL = 4             # 4 个大文件并行 curl；HF 单流约 10–30 MB/s
FLY_Z = 2.0                       # 飞行高度（nav 系，米）：E1 ToF 量程内、与 Callinstrasse/Gazebo 一致
CAMERA = {"width": 320, "height": 240, "fov_deg": 60.0, "pitch_deg": 0.0}   # 用户 11-07 拍板：与 Gazebo 同 60° HFOV
COLLISION = {"z_slice": [1.8, 2.2], "voxel_m": 0.2, "envelope": "e1_mini"}   # 与 Callinstrasse 同口径
GROUND_RADIUS_M = 1.5             # 相机脚下取地面：水平 1.5 m 内 LiDAR 点高度的 10 分位（3DGS 报告 P0 的同一方法）
GROUND_PCT = 10                   # 10 分位：避开草丛/长椅，又不被个别噪点拉低
LIDAR_SUBSAMPLE = 4               # raw_pcd 9 亿字节 ≈ 3,000 万点；每 4 点取 1 做对齐统计够用（0.1 m 体素）
HOLD_S = 0.5                      # 与 scene.step_s / Gazebo 训练数据窗口一致
SMOKE_EPISODES = 2                # 每路线 2 集冒烟（DATASET_3DGS_SELF_CAPTURE §3 步 8：到达 >=1/2、0 碰撞）
SEED0 = 0                         # 路线 r 第 k 集 seed = SEED0 + 100*r + k
PROBE_LAPVAR_LOW = 60.0           # collect_3dgs.frame_quality 同阈值；低质帧 <10% 过线
# 路线（nav 系，米；z 固定 FLY_Z）：先跑到 align 步看 OUT/map_*.png 与 align.json 再填；空列表时 scene 步只写
# 顶层 spawn/waypoint 占位并停下。max_steps 由 planner.route_check 算出后回填（None = 由本脚本回填）。
ROUTES = [
    {"name": "r0_path_east", "spawn": {"x": 4.0, "y": -1.5, "yaw_deg": -15.0}, "waypoint": {"x": 16.0, "y": -3.5},
     "note": "沿公园步道东行 12 m：左侧 2 m 树（x 10–12）、右侧 2–3 m 栅栏/树篱；开阔段，教师应接近全程 FORWARD"},
    {"name": "r1_tree_row_south", "spawn": {"x": 24.0, "y": -5.5, "yaw_deg": -10.0}, "waypoint": {"x": 38.0, "y": -8.5},
     "note": "穿第一片树丛（x 28–36, y −4..−6）与栅栏之间的夹道 14 m，需贴树干外侧绕行"},
    {"name": "r2_face_fence_left", "spawn": {"x": 14.0, "y": -3.5, "yaw_deg": -80.0}, "waypoint": {"x": 26.0, "y": -5.5},
     "note": "出生正对 1.5 m 外的栅栏/树篱（机载首帧是栅栏），必须左转约 90° 再沿步道东行 12 m。"
             "原候选「第二片树丛 (36,−7.5)->(48,−12) 穿树干」在 LiDAR 真值上 2 m 层被树冠封死，A* 绕到轨迹以北 4 m 的未拍摄区，弃用"},
    {"name": "r3_reverse_west", "spawn": {"x": 20.0, "y": -4.5, "yaw_deg": 165.0}, "waypoint": {"x": 6.0, "y": -2.0},
     "note": "r0 反向西行 14 m：出生背对步道方向需微调航向，树在右手侧"},
    {"name": "r4_face_tree_turn", "spawn": {"x": 26.0, "y": -4.5, "yaw_deg": 60.0}, "waypoint": {"x": 34.0, "y": -7.5},
     "note": "出生正对 3 m 外的树丛（机载首帧是树），必须右转约 90° 再沿夹道走 9 m"},
    {"name": "r5_bend_south", "spawn": {"x": 44.0, "y": -11.0, "yaw_deg": -30.0}, "waypoint": {"x": 51.5, "y": -15.5},
     "note": "步道末端向南的弯 9 m：栅栏角 + 树，转向约 45°"},
]
ENV = dict(os.environ, TORCH_EXTENSIONS_DIR=os.path.join(DATA_ROOT, "uav_rgb", "torch-cache", "extensions-sm89-20261106"),
           TMPDIR=os.path.join(DATA_ROOT, "uav_rgb", "tmp"))
ENV.pop("TORCH_CUDA_ARCH_LIST", None)     # gsplat JIT 按本机 sm_89 编；带着旧变量会重编/编错架构

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def run(cmd, what, capture=False):
    log(f"### {what}: {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    if capture:
        res = subprocess.run(cmd, cwd=REPO, env=ENV, text=True, capture_output=True)
        if res.stderr:
            print(res.stderr[-3000:], flush=True)
        rc, out = res.returncode, res.stdout
    else:
        rc, out = subprocess.call(cmd, cwd=REPO, env=ENV), None
    log(f"### {what} rc={rc} wall={time.time() - t0:.1f}s")
    if rc != 0:
        raise SystemExit(f"{what} 失败 rc={rc}")
    return out


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------- 1 下载
def hf_url(rel):
    return f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{TIER_PATH}/{SCENE_ID}/{rel}"


def hf_sizes():
    """HF tree API 的文件大小（LFS 用 lfs.size），用来校验下载完整；API 不通就返回 {}。"""
    sizes = {}
    for sub in ("", "sparse/0"):
        url = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main/{TIER_PATH}/{SCENE_ID}/{sub}".rstrip("/")
        try:
            raw = subprocess.run(["curl", "-sS", "--max-time", "60", url], capture_output=True, text=True,
                                 check=True).stdout
            for item in json.loads(raw):
                if item.get("type") == "file":
                    rel = item["path"].split(f"{SCENE_ID}/", 1)[1]
                    sizes[rel] = int((item.get("lfs") or {}).get("size") or item.get("size") or 0)
        except (subprocess.CalledProcessError, ValueError, KeyError) as e:
            log(f"HF tree API 失败（{sub or '.'}）：{e}")
    return sizes


def step_download():
    done = os.path.join(DOWNLOAD_DIR, "DONE")
    if os.path.isfile(done):
        log(f"download: 已有 {done}，跳过")
        return
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    sizes = hf_sizes()
    write_json(os.path.join(DOWNLOAD_DIR, "hf_sizes.json"), sizes)
    total = sum(sizes.get(f, 0) for f in FILES)
    log(f"download: {len(FILES)} 个文件，HF 报告合计 {total / 1e9:.2f} GB -> {DOWNLOAD_DIR}")
    if total > 15e9:
        raise SystemExit("下载量超过 15 GB 上限（BRIEF4 §1），改 FILES")
    pending = [f for f in FILES if not (f in sizes and os.path.isfile(os.path.join(DOWNLOAD_DIR, f))
                                       and os.path.getsize(os.path.join(DOWNLOAD_DIR, f)) == sizes[f])]
    procs = []
    for rel in pending:
        dst = os.path.join(DOWNLOAD_DIR, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cmd = ["curl", "-sSL", "--retry", "5", "--retry-delay", "5", "-C", "-", "-o", dst, hf_url(rel)]
        procs.append((rel, subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)))
        while len([p for _, p in procs if p.poll() is None]) >= DOWNLOAD_PARALLEL:
            time.sleep(2)
    failed = []
    for rel, p in procs:
        _, err = p.communicate()
        dst = os.path.join(DOWNLOAD_DIR, rel)
        got = os.path.getsize(dst) if os.path.isfile(dst) else -1
        want = sizes.get(rel)
        ok = p.returncode == 0 and got > 0 and (want is None or got == want)
        # curl -C - 对已完整的文件返回 rc=33（range 不可满足）；大小对得上就算成功
        if p.returncode == 33 and want is not None and got == want:
            ok = True
        log(f"  {rel}: rc={p.returncode} {got / 1e6:.1f} MB (HF {None if want is None else round(want / 1e6, 1)} MB)"
            f" {'OK' if ok else 'FAIL ' + err[-200:]}")
        if not ok:
            failed.append(rel)
    if failed:
        raise SystemExit(f"download 失败：{failed}")
    with open(done, "w") as f:
        f.write(f"rc=0 {time.strftime('%Y-%m-%d %H:%M:%S')} files={len(FILES)} bytes={total}\n")
    log(f"download: 完成，落 {done}")


# ---------------------------------------------------------------- 2 转换：3DGS PLY 校验 + USDZ 碰撞网格 -> PLY
def link_or_copy(src, dst):
    """同一文件系统硬链接（0 字节），否则复制；已存在且大小一致就跳过。"""
    if os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(src):
        return "exists"
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        import shutil
        shutil.copyfile(src, dst)
        return "copy"


def usda_mesh_to_arrays(usda_path):
    """Wanderland scene.usdz 里的 scene.usda（ASCII USD）-> (points float64 [N,3], faces int64 [M,3], xform)。

    只取第一个 `def Mesh`：Collider 三角网格（PhysicsCollisionAPI），点坐标在**局部系**，与同包
    3DGS / LiDAR 同一坐标系（extent 与 raw_pcd.ply 一致）；Collider Xform 的 rotateXYZ(-90,0,180)
    只是 Isaac Sim 的舞台旋转，这里原样返回给调用方记录，不作用到点上。
    不依赖 pxr：按 `point3f[] points = [...]` / `int[] faceVertexIndices = [...]` 文本切片解析。
    """
    import re
    with open(usda_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    xform = {}
    m = re.search(r'def Xform "Collider"(.*?)def Mesh', text, re.S)
    if m:
        for key in ("rotateXYZ", "scale", "translate"):
            mm = re.search(rf"xformOp:{key} = \(([^)]*)\)", m.group(1))
            if mm:
                xform[key] = [float(v) for v in mm.group(1).split(",")]

    def array_after(name):
        i = text.find(name)
        if i < 0:
            raise ValueError(f"{usda_path}: 缺 {name}")
        eq = text.index("=", i + len(name))         # 类型里的 "[]" 要跳过，取 "= [" 后的数组
        j = text.index("[", eq)
        k = text.index("]", j)
        return text[j + 1:k]

    counts = np.fromstring(array_after("int[] faceVertexCounts"), sep=",", dtype=np.int64)
    if len(counts) and (counts != 3).any():
        raise ValueError(f"{usda_path}: 网格含非三角面（faceVertexCounts != 3）")
    idx = np.fromstring(array_after("int[] faceVertexIndices"), sep=",", dtype=np.int64)
    pts_txt = array_after("point3f[] points").replace("(", "").replace(")", "")
    pts = np.fromstring(pts_txt, sep=",", dtype=np.float64)
    if len(pts) % 3 or len(idx) % 3:
        raise ValueError(f"{usda_path}: points/faceVertexIndices 长度不是 3 的倍数")
    pts = pts.reshape(-1, 3)
    faces = idx.reshape(-1, 3)
    if len(faces) != len(counts) or (len(faces) and faces.max() >= len(pts)):
        raise ValueError(f"{usda_path}: 面数与 faceVertexCounts 不符或索引越界")
    return pts, faces, xform


def write_mesh_ply(path, pts, faces):
    """binary_little_endian 三角网格 PLY（float x/y/z + uchar 计数 int 索引），collision.read_ply_vertices 可读。"""
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(pts)}",
              "property float x", "property float y", "property float z",
              f"element face {len(faces)}", "property list uchar int vertex_indices", "end_header"]
    v = np.asarray(pts, dtype="<f4")
    f_dt = np.dtype([("n", "u1"), ("idx", "<i4", (3,))])
    fa = np.zeros(len(faces), dtype=f_dt)
    fa["n"] = 3
    fa["idx"] = np.asarray(faces, dtype=np.int32)
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        v.tofile(f)
        fa.tofile(f)


def step_convert():
    """3dgs.ply 已是 INRIA 布局（opacity logit、scale log、45 个 f_rest），直接复用；LiDAR 点云硬链接；
    USDZ 里的碰撞网格抽成 PLY。都落在 SCENE_DIR。"""
    import importlib
    scene_mod = importlib.import_module("sim.3dgs.scene")
    os.makedirs(SCENE_DIR, exist_ok=True)
    src_ply = os.path.join(DOWNLOAD_DIR, "3dgs.ply")
    g = scene_mod.read_gaussian_ply(src_ply)
    op = g["opacities"]
    if not (0.0 <= float(op.min()) and float(op.max()) <= 1.0) or g["sh_degree"] != 3:
        raise SystemExit("3dgs.ply 不是预期的 INRIA 布局（sigmoid 后 opacity 应在 [0,1]，sh_degree=3）")
    info = {"n_gaussians": int(len(op)), "sh_degree": int(g["sh_degree"]),
            "opacity_ge_0.3_frac": round(float((op >= 0.3).mean()), 4),
            "smax_gt_3m_frac": round(float((g["scales"].max(axis=1) > 3).mean()), 5)}
    log(f"convert: 3dgs.ply {info}")
    log(f"convert: scene.ply <- 3dgs.ply ({link_or_copy(src_ply, os.path.join(SCENE_DIR, 'scene.ply'))})")
    log(f"convert: lidar.ply <- raw_pcd.ply ({link_or_copy(os.path.join(DOWNLOAD_DIR, 'raw_pcd.ply'), os.path.join(SCENE_DIR, 'lidar.ply'))})")
    mesh_ply = os.path.join(SCENE_DIR, "collision_mesh.ply")
    if not os.path.isfile(mesh_ply):
        import zipfile
        usd_dir = os.path.join(DOWNLOAD_DIR, "usd")
        usda = os.path.join(usd_dir, "scene.usda")
        if not os.path.isfile(usda):
            with zipfile.ZipFile(os.path.join(DOWNLOAD_DIR, "scene.usdz")) as z:
                z.extract("scene.usda", usd_dir)
        pts, faces, xform = usda_mesh_to_arrays(usda)
        write_mesh_ply(mesh_ply, pts, faces)
        info["mesh"] = {"n_vertices": int(len(pts)), "n_faces": int(len(faces)), "collider_xform": xform,
                        "extent_min": pts.min(axis=0).round(3).tolist(), "extent_max": pts.max(axis=0).round(3).tolist()}
        log(f"convert: collision_mesh.ply <- scene.usdz/scene.usda {info['mesh']}")
    write_json(os.path.join(OUT, "convert.json"), info)


# ---------------------------------------------------------------- 3 对齐：up 轴、地面、nav_to_world、验证
def read_colmap_images_bin(path):
    """COLMAP images.bin -> (names, centres [N,3] 世界系, image_up [N,3] 世界系单位向量)。跳过 2D 点。"""
    import struct
    names, centres, ups = [], [], []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(4)
            q = struct.unpack("<4d", f.read(32))
            t = np.array(struct.unpack("<3d", f.read(24)))
            f.read(4)
            name = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00" or not ch:
                    break
                name += ch
            npts, = struct.unpack("<Q", f.read(8))
            f.seek(24 * npts, 1)
            w, x, y, z = q
            r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                          [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                          [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
            names.append(name.decode(errors="replace"))
            centres.append(-r.T @ t)
            ups.append(r.T @ np.array([0.0, -1.0, 0.0]))       # 图像上方（相机 -y）在世界系
    return names, np.array(centres), np.array(ups)


def nav_to_world_from_up(up_world, centres):
    """用世界系的重力反方向（单位向量，已对齐到坐标轴）与相机轨迹定 nav 轴：
    nav z = up；nav x = 相机散布最大的水平轴（朝轨迹起点->终点为正）；nav y = z × x（右手）。
    返回 4x4 旋转块（平移留 0，由地面高度另填）。"""
    up = np.asarray(up_world, dtype=np.float64)
    up = up / np.linalg.norm(up)
    horiz = [i for i in range(3) if abs(up[i]) < 0.5]
    spread = centres[:, horiz].std(axis=0)
    ax = horiz[int(np.argmax(spread))]
    x = np.zeros(3)
    x[ax] = 1.0 if centres[-1, ax] >= centres[0, ax] else -1.0
    y = np.cross(up, x)
    m = np.eye(4)
    m[:3, :3] = np.stack([x, y, up], axis=1)          # 列 = nav 轴在世界系
    return m


def local_ground_height(h_pts, xy_pts, xy_query, radius, pct):
    """每个查询点水平 radius 内点的高度 pct 分位（地面）；没有点时 NaN。h/xy 都是世界系投影后的量。"""
    out = np.full(len(xy_query), np.nan)
    # 栅格桶加速：按 radius 分格，查相邻 9 格
    cell = np.floor(xy_pts / radius).astype(np.int64)
    key = cell[:, 0] * 1_000_003 + cell[:, 1]
    order = np.argsort(key)
    key_s = key[order]
    for i, q in enumerate(xy_query):
        cq = np.floor(q / radius).astype(np.int64)
        idx = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                k = (cq[0] + di) * 1_000_003 + (cq[1] + dj)
                a, b = np.searchsorted(key_s, k), np.searchsorted(key_s, k, side="right")
                if b > a:
                    idx.append(order[a:b])
        if not idx:
            continue
        idx = np.concatenate(idx)
        d = np.hypot(xy_pts[idx, 0] - q[0], xy_pts[idx, 1] - q[1])
        sel = idx[d <= radius]
        if len(sel) >= 20:
            out[i] = np.percentile(h_pts[sel], pct)
    return out


def step_align():
    """重力/尺度/地面：LiDAR 米制点云 + COLMAP 相机位姿 -> nav_to_world、ground_z；验证 nav z=FLY_Z 离地高度。"""
    import importlib
    col = importlib.import_module("sim.3dgs.collision")
    align_path = os.path.join(OUT, "align.json")
    names, C, ups = read_colmap_images_bin(os.path.join(DOWNLOAD_DIR, "sparse", "0", "images.bin"))
    order = np.argsort(names)                       # 文件名含时间戳，按采集顺序
    C, ups = C[order], ups[order]
    up_mean = ups.mean(axis=0)
    up_mean /= np.linalg.norm(up_mean)
    ax = int(np.argmax(np.abs(up_mean)))
    up_axis = np.zeros(3)
    up_axis[ax] = np.sign(up_mean[ax])
    tilt_deg = math.degrees(math.acos(min(1.0, abs(up_mean[ax]))))
    log(f"align: 相机图像上方均值 {up_mean.round(3).tolist()} -> up 轴 {up_axis.tolist()}（偏 {tilt_deg:.1f}°，手持抖动）")
    lidar = col.read_ply_vertices(os.path.join(SCENE_DIR, "lidar.ply"), stride=LIDAR_SUBSAMPLE)["xyz"]
    m = nav_to_world_from_up(up_axis, C)
    r = m[:3, :3]
    # 世界 -> nav（旋转部分）：nav = R^T world
    nav_pts = lidar @ r
    nav_c = C @ r
    # 地面平面拟合：相机脚下 radius 内、高度在 [cam-3, cam-0.5] 的点 -> SVD 平面法向与 up 的夹角（重力残差）
    g_cam = local_ground_height(nav_pts[:, 2], nav_pts[:, :2], nav_c[:, :2], GROUND_RADIUS_M, GROUND_PCT)
    ok = np.isfinite(g_cam)
    cam_h = nav_c[:, 2] - g_cam
    log(f"align: {ok.sum()}/{len(C)} 相机脚下有地面点；相机离地 p10/50/90 = "
        f"{np.nanpercentile(cam_h, 10):.2f}/{np.nanpercentile(cam_h, 50):.2f}/{np.nanpercentile(cam_h, 90):.2f} m；"
        f"地面高度(nav 旋转后未平移) p10/50/90 = {np.nanpercentile(g_cam, 10):.2f}/{np.nanpercentile(g_cam, 50):.2f}/{np.nanpercentile(g_cam, 90):.2f}")
    near = np.zeros(len(nav_pts), dtype=bool)
    step = max(1, len(nav_c) // 60)
    for c, g in zip(nav_c[::step], g_cam[::step]):
        if np.isfinite(g):
            d = np.hypot(nav_pts[:, 0] - c[0], nav_pts[:, 1] - c[1])
            near |= (d <= GROUND_RADIUS_M) & (np.abs(nav_pts[:, 2] - g) <= 0.15)
    gp = nav_pts[near]
    ctr = gp.mean(axis=0)
    _, _, vt = np.linalg.svd(gp - ctr, full_matrices=False)
    normal = vt[-1] * (1 if vt[-1, 2] > 0 else -1)
    plane_tilt = math.degrees(math.acos(min(1.0, normal[2])))
    log(f"align: 地面点 {len(gp)} 个，平面法向 {normal.round(4).tolist()}，与 up 轴夹角 {plane_tilt:.2f}°")
    ground_ref = float(np.nanmedian(g_cam))
    # 平移：nav z=0 <-> 相机轨迹下地面高度中位；水平原点保持世界原点（轨迹起点附近）。
    # world = R nav + t；nav z=0 的点 world 高度分量应为 ground_ref：t = R [0,0,ground_ref]
    m[:3, 3] = r @ np.array([0.0, 0.0, ground_ref])
    # 验证：沿相机轨迹，nav z=FLY_Z 的点换到世界再量到局部 LiDAR 地面的高度
    fly_h = FLY_Z + ground_ref - g_cam          # nav 高度 FLY_Z 相对局部地面
    q = lambda a, p: float(np.nanpercentile(a, p))
    log(f"align: nav z={FLY_Z} 沿相机轨迹离局部地面 min/p10/p50/p90/max = "
        f"{np.nanmin(fly_h):.2f}/{q(fly_h,10):.2f}/{q(fly_h,50):.2f}/{q(fly_h,90):.2f}/{np.nanmax(fly_h):.2f} m")
    # 网格的地面与 LiDAR 地面一致性（同一坐标系的证据）
    mesh = col.read_ply_vertices(os.path.join(SCENE_DIR, "collision_mesh.ply"))
    mesh_nav = col.sample_mesh_surface(mesh["xyz"], mesh["faces"], spacing=0.2) @ r   # 抽稀网格顶点太稀，面上撒点再量
    g_mesh = local_ground_height(mesh_nav[:, 2], mesh_nav[:, :2], nav_c[::step][:, :2], GROUND_RADIUS_M, GROUND_PCT)
    dg = g_mesh - g_cam[::step]
    mesh_top = float(np.nanpercentile(mesh_nav[:, 2] - ground_ref, 99.9))
    log(f"align: 网格地面 - LiDAR 地面（相机脚下）p10/50/90 = {q(dg,10):.2f}/{q(dg,50):.2f}/{q(dg,90):.2f} m；"
        f"网格最高 nav z(p99.9) = {mesh_top:.2f} m（Wanderland pcd2mesh --filter-vert 只保留相机上下 1.5 m）")
    info = {"scene_id": SCENE_ID, "up_axis_world": up_axis.tolist(), "image_up_mean_world": up_mean.round(4).tolist(),
            "image_up_tilt_deg": round(tilt_deg, 2), "ground_plane_normal_nav": normal.round(5).tolist(),
            "ground_plane_tilt_deg": round(plane_tilt, 3), "n_ground_points": int(len(gp)),
            "nav_to_world": m.round(6).tolist(), "ground_ref_nav_rot": round(ground_ref, 3),
            "camera_height_above_ground_m": {p: round(q(cam_h, p), 3) for p in (10, 50, 90)},
            "camera_count": int(len(C)), "cameras_with_ground": int(ok.sum()),
            "fly_z": FLY_Z, "fly_height_above_local_ground_m": {"min": round(float(np.nanmin(fly_h)), 3),
                                                                 "p10": round(q(fly_h, 10), 3), "p50": round(q(fly_h, 50), 3),
                                                                 "p90": round(q(fly_h, 90), 3), "max": round(float(np.nanmax(fly_h)), 3)},
            "mesh_minus_lidar_ground_m": {p: round(q(dg, p), 3) for p in (10, 50, 90)},
            "mesh_top_nav_z_p99.9": round(mesh_top, 3),
            "fly_height_outliers": int(np.sum(np.abs(fly_h[np.isfinite(fly_h)] - FLY_Z) > 0.3)),
            "lidar_points_used": int(len(lidar)), "lidar_stride": LIDAR_SUBSAMPLE,
            "cameras_nav": np.round(nav_c - [0, 0, ground_ref], 3).tolist(),
            "method": f"相机脚下 {GROUND_RADIUS_M} m 内 LiDAR 高度 {GROUND_PCT} 分位 = 局部地面；nav z=0 取其轨迹中位数"}
    write_json(align_path, info)
    # 顶视图（选路线用）：飞行层 LiDAR 占据（黑）+ 地面层（绿）+ 相机轨迹（蓝）+ 已填路线（红）；
    # 0.1 m/px、2 m 网格；3dgs-venv 没有 matplotlib，用 PIL 直接画栅格。
    draw_top_view(nav_pts - [0, 0, ground_ref], nav_c - [0, 0, ground_ref], os.path.join(OUT, "map_lidar_nav.png"))
    log(f"align: 写 {align_path}")


def draw_top_view(pts_nav, cams_nav, path, px_per_m=10, margin_m=8.0):
    from PIL import Image, ImageDraw
    x0, x1 = cams_nav[:, 0].min() - margin_m, cams_nav[:, 0].max() + margin_m
    y0, y1 = cams_nav[:, 1].min() - margin_m, cams_nav[:, 1].max() + margin_m
    w, h = int((x1 - x0) * px_per_m) + 1, int((y1 - y0) * px_per_m) + 1
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    def to_px(xy):
        return (np.clip(((xy[:, 0] - x0) * px_per_m).astype(int), 0, w - 1),
                np.clip(((y1 - xy[:, 1]) * px_per_m).astype(int), 0, h - 1))   # y 朝上

    z = pts_nav[:, 2]
    inside = (pts_nav[:, 0] >= x0) & (pts_nav[:, 0] <= x1) & (pts_nav[:, 1] >= y0) & (pts_nav[:, 1] <= y1)
    for mask, color in ((inside & (np.abs(z) <= 0.15), (200, 235, 200)),
                        (inside & (z >= COLLISION["z_slice"][0] - 0.3) & (z <= COLLISION["z_slice"][1] + 0.3), (40, 40, 40))):
        i, j = to_px(pts_nav[mask])
        img[j, i] = color
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for gx in np.arange(math.ceil(x0 / 2) * 2, x1, 2):
        px = int((gx - x0) * px_per_m)
        d.line([(px, 0), (px, h)], fill=(190, 190, 255) if gx % 10 else (120, 120, 255), width=1)
        d.text((px + 2, 2), f"{gx:.0f}", fill=(0, 0, 160))
    for gy in np.arange(math.ceil(y0 / 2) * 2, y1, 2):
        py = int((y1 - gy) * px_per_m)
        d.line([(0, py), (w, py)], fill=(190, 190, 255) if gy % 10 else (120, 120, 255), width=1)
        d.text((2, py + 2), f"{gy:.0f}", fill=(0, 0, 160))
    ci, cj = to_px(cams_nav)
    d.line(list(zip(ci.tolist(), cj.tolist())), fill=(0, 90, 255), width=2)
    for rt in ROUTES:
        s, wp = rt["spawn"], rt["waypoint"]
        (sx,), (sy,) = to_px(np.array([[s["x"], s["y"]]]))
        (wx,), (wy,) = to_px(np.array([[wp["x"], wp["y"]]]))
        d.line([(sx, sy), (wx, wy)], fill=(255, 0, 0), width=3)
        d.ellipse([wx - 5, wy - 5, wx + 5, wy + 5], outline=(255, 0, 0), width=2)
        d.text((sx + 4, sy + 4), rt["name"], fill=(200, 0, 0))
    im.save(path)


# ---------------------------------------------------------------- 4 scene.json + 路线校验
SCENE_JSON = os.path.join(SCENE_DIR, "scene.json")


def step_scene():
    """写 scene.json（nav_to_world 来自 align.json；碰撞源 = LiDAR 点云真值），用 planner.route_check 校验
    每条路线并回填 max_steps = steps_estimate；任一路线不可行/净空不足即停。"""
    import importlib
    scene_mod = importlib.import_module("sim.3dgs.scene")
    collision = importlib.import_module("sim.3dgs.collision")
    planner_mod = importlib.import_module("sim.3dgs.planner")
    align = json.load(open(os.path.join(OUT, "align.json"), encoding="utf-8"))
    if not ROUTES:
        raise SystemExit("ROUTES 为空：先看 OUT/map_lidar_nav.png 与 align.json 填路线（配置区）")
    routes = []
    for r in ROUTES:
        sp = dict(r["spawn"]); sp.setdefault("z", FLY_Z); sp.setdefault("yaw_deg", 0.0)
        wp = dict(r["waypoint"]); wp.setdefault("z", FLY_Z)
        routes.append({"name": r["name"], "spawn": sp, "waypoint": wp, "max_steps": r.get("max_steps"),
                       "note": r.get("note", "")})
    fh = align["fly_height_above_local_ground_m"]
    scene = {
        "schema": "uavrgb.3dgs_scene.v1",
        "name": SCENE_NAME,
        "source": f"Wanderland (NYU ai4ce, arXiv 2511.20620) scene {SCENE_ID}, HF {HF_REPO} {TIER_PATH}; "
                  "3dgs.ply (gsplat, INRIA layout) + raw_pcd.ply (MetaCam LIV-SLAM LiDAR) + scene.usdz collider mesh",
        "license": "Apache-2.0 (HF dataset card + github.com/ai4ce/wanderland LICENSE)",
        "ply": "scene.ply",
        "nav_to_world": align["nav_to_world"],
        "world_up": "-y (COLMAP y-down; nav z-up 由 nav_to_world 旋转块给出)",
        "ground_z": 0.0,
        "spawn": routes[0]["spawn"], "waypoint": routes[0]["waypoint"],
        "reach_radius_m": 2.0,
        "camera": dict(CAMERA, fov_source="用户 2026-11-07 拍板：Gazebo 与 3DGS 统一 60° HFOV / 320×240"),
        "collision": dict(COLLISION, source="points", geometry="lidar.ply", min_points=collision.DEFAULT_MIN_POINTS,
                          geometry_stride=1),
        "max_steps": routes[0]["max_steps"] or 60,
        "step_s": HOLD_S,
        "notes": [
            f"尺度：MetaCam LIV-SLAM 米制；验证：COLMAP {align['camera_count']} 个相机离 LiDAR 局部地面 "
            f"p10/50/90 = {align['camera_height_above_ground_m']['10']}/{align['camera_height_above_ground_m']['50']}/"
            f"{align['camera_height_above_ground_m']['90']} m（手持扫描仪约 1.3 m，沿 52 m 轨迹一致）。",
            f"重力/地面：相机图像上方均值指向世界 {align['up_axis_world']}（偏 {align['image_up_tilt_deg']}°），"
            f"地面平面法向与该轴夹角 {align['ground_plane_tilt_deg']}°（未校正，按轴对齐旋转）；nav z=0 = 相机脚下 "
            f"{GROUND_RADIUS_M} m 内 LiDAR 高度 {GROUND_PCT} 分位的轨迹中位数。nav z={FLY_Z} 沿相机轨迹离局部地面 "
            f"p10/50/90 = {fh['p10']}/{fh['p50']}/{fh['p90']} m（地面沿路起伏 ±0.15 m，未建模）。",
            "碰撞源 collision.source=points：LiDAR 稠密点云（lidar.ply = raw_pcd.ply 硬链接）在 nav 系按 0.2 m 体素计数占据，"
            f"min_points={collision.DEFAULT_MIN_POINTS}；高斯代理 v2 只作对照（tools/collision_truth_compare.py）。",
            f"collision_mesh.ply（scene.usdz 的 Collider）最高只到 nav z≈{align['mesh_top_nav_z_p99.9']} m"
            "（Wanderland pcd2mesh 只保留相机上下 1.5 m），覆盖不到 2 m 飞行层，不能当飞行层真值。",
            "routes[]：--route N 用 routes[N] 覆盖 spawn/waypoint/max_steps；routes[0] 与顶层相同。max_steps = "
            "planner.route_check.steps_estimate（1.25×(航程/0.15 m + 转向/17° + 8)）。",
        ],
        "routes": routes,
    }
    os.makedirs(SCENE_DIR, exist_ok=True)
    scene_mod.save_scene(scene, SCENE_JSON)
    loaded = scene_mod.load_scene(SCENE_JSON)
    t0 = time.time()
    occ = collision.from_scene(loaded)
    log(f"scene: 占据（{occ.source}）{occ.summary()['n_occupied_cells']} 格，{time.time() - t0:.1f}s")
    planner = planner_mod.OccPlanner(occ)
    rows, paths, bad = [], {}, []
    for i, r in enumerate(routes):
        chk = planner_mod.route_check(planner, r["spawn"], r["waypoint"])
        paths[i] = chk.pop("path", None)
        if r["max_steps"] is None and chk.get("steps_estimate"):
            r["max_steps"] = int(chk["steps_estimate"])
        row = {"route": i, "name": r["name"], "max_steps": r["max_steps"], **chk}
        rows.append(row)
        ok = chk.get("feasible") and chk.get("clear_ok")
        log(f"scene: route {i} {r['name']}: {'OK ' if ok else 'BAD'} path {chk.get('path_m')} m turn {chk.get('turn_deg')}° "
            f"min_clear {chk.get('min_clearance_m')} steps {chk.get('steps_estimate')} {chk.get('reason', '')}")
        if not ok:
            bad.append(r["name"])
    scene["routes"] = routes
    scene["max_steps"] = routes[0]["max_steps"]
    scene_mod.save_scene(scene, SCENE_JSON)
    write_json(os.path.join(OUT, "routes_check.json"), rows)
    write_json(os.path.join(OUT, "routes_paths.json"), {str(k): v for k, v in paths.items()})
    draw_occ_routes(occ, paths, os.path.join(OUT, "map_routes_truth.png"), align)
    if bad:
        raise SystemExit(f"路线不可行/净空不足：{bad}（看 OUT/map_routes_truth.png 调配置区 ROUTES）")
    log(f"scene: 写 {SCENE_JSON}（{len(routes)} 条路线全部 feasible/clear_ok）")


def draw_occ_routes(occ, paths, path, align, px_per_m=20, margin_m=6.0):
    """占据栅格（原始黑 / 膨胀灰）+ 相机轨迹（蓝）+ A* 路径（红）+ 起终点。"""
    from PIL import Image, ImageDraw
    cams = np.array(align["cameras_nav"])
    x0, x1 = cams[:, 0].min() - margin_m, cams[:, 0].max() + margin_m
    y0, y1 = cams[:, 1].min() - margin_m, cams[:, 1].max() + margin_m
    w, h = int((x1 - x0) * px_per_m) + 1, int((y1 - y0) * px_per_m) + 1
    im = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(im)

    def px(x, y):
        return (x - x0) * px_per_m, (y1 - y) * px_per_m

    if occ.grid.size:
        for arr, color in ((occ.grid.any(axis=2), (200, 200, 200)), (occ.raw.any(axis=2), (0, 0, 0))):
            ii, jj = np.nonzero(arr)
            for i, j in zip(ii, jj):
                a, b = px(occ.origin[0] + i * occ.voxel, occ.origin[1] + (j + 1) * occ.voxel)
                d.rectangle([a, b, a + occ.voxel * px_per_m, b + occ.voxel * px_per_m], fill=color)
    for gx in np.arange(math.ceil(x0 / 2) * 2, x1, 2):
        a, _ = px(gx, 0)
        d.line([(a, 0), (a, h)], fill=(190, 190, 255) if gx % 10 else (110, 110, 255))
        d.text((a + 2, 2), f"{gx:.0f}", fill=(0, 0, 160))
    for gy in np.arange(math.ceil(y0 / 2) * 2, y1, 2):
        _, b = px(0, gy)
        d.line([(0, b), (w, b)], fill=(190, 190, 255) if gy % 10 else (110, 110, 255))
        d.text((2, b + 2), f"{gy:.0f}", fill=(0, 0, 160))
    d.line([px(x, y) for x, y, _ in cams], fill=(0, 90, 255), width=1)
    for i, pth in paths.items():
        if not pth:
            continue
        d.line([px(x, y) for x, y in pth], fill=(255, 0, 0), width=3)
        sx, sy = px(*pth[0]); ex, ey = px(*pth[-1])
        d.ellipse([sx - 5, sy - 5, sx + 5, sy + 5], fill=(0, 160, 0))
        d.ellipse([ex - 5, ey - 5, ex + 5, ey + 5], outline=(255, 0, 0), width=2)
        d.text((sx + 6, sy - 12), f"r{i}", fill=(200, 0, 0))
    im.save(path)


# ---------------------------------------------------------------- 5 沿路线渲染清晰度探测
PROBE_SPACING_M = 1.0       # 每 1 m 一个机载位姿（航向 = 路径向前 1 m 的方向，首帧用出生 yaw）
PROBE_LOW_FRAC_MAX = 0.10   # DATASET_3DGS_SELF_CAPTURE §3 步 7：每条路线低质帧 <10% 过线


def step_probe():
    """沿每条路线的 A* 路径渲染机载帧，算 collect_3dgs.frame_quality 的 lapvar，写 routes_quality.json +
    每路线 contact sheet。临时实现（SCENE-B 在转正 tools/scene_probe_quality.py，这里不建同名文件）。"""
    import importlib
    scene_mod = importlib.import_module("sim.3dgs.scene")
    renderer_mod = importlib.import_module("sim.3dgs.renderer")
    collect = importlib.import_module("sim.3dgs.collect_3dgs")
    from PIL import Image, ImageDraw
    scene = scene_mod.load_scene(SCENE_JSON)
    paths = json.load(open(os.path.join(OUT, "routes_paths.json"), encoding="utf-8"))
    rd = renderer_mod.make_renderer(scene)
    log(f"probe: 渲染器 {rd.stats()}")
    summary, bad = {}, []
    for i, r in enumerate(scene["routes"]):
        path = paths.get(str(i))
        if not path:
            continue
        sp = r["spawn"]
        poses, acc = [(sp["x"], sp["y"], sp["yaw_deg"])], 0.0
        for k in range(1, len(path)):
            acc += math.hypot(path[k][0] - path[k - 1][0], path[k][1] - path[k - 1][1])
            if acc >= PROBE_SPACING_M:
                j = min(len(path) - 1, k + 5)
                yaw = (math.degrees(math.atan2(path[j][1] - path[k][1], path[j][0] - path[k][0]))
                       if j > k else poses[-1][2])
                poses.append((path[k][0], path[k][1], yaw))
                acc = 0.0
        frames, tiles = [], []
        for (x, y, yaw) in poses:
            rgb, depth, _ms = rd.render_pose(x, y, FLY_Z, yaw)
            q = collect.frame_quality(rgb, depth)
            frames.append({"pose": [round(x, 2), round(y, 2), round(yaw, 1)], **q,
                           "depth_med_m": round(float(np.median(depth[depth > 0])), 2) if (depth > 0).any() else None})
            im = Image.fromarray(rgb)
            ImageDraw.Draw(im).text((4, 4), f"({x:.1f},{y:.1f},{yaw:.0f}) lap={q['lapvar']:.0f}",
                                    fill=(255, 255, 0) if not q["low"] else (255, 60, 60))
            tiles.append(im)
        cols = 6
        rows_n = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * CAMERA["width"], rows_n * CAMERA["height"]))
        for k, t in enumerate(tiles):
            sheet.paste(t, ((k % cols) * CAMERA["width"], (k // cols) * CAMERA["height"]))
        sheet.save(os.path.join(OUT, f"probe_r{i}_{r['name']}.png"))
        # 高度复核（独立于 LiDAR）：沿路每隔一个位姿垂直向下渲染，中心 1/3 画面的期望深度中位数 = 3DGS 里的离地高度。
        down = []
        for (x, y, _yaw) in poses[::2]:
            _rgb, dep, _ms = rd.render_pose(x, y, FLY_Z, 0.0, pitch_deg=-90.0)
            h, w = dep.shape
            c = dep[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
            if (c > 0).any():
                down.append(float(np.median(c[c > 0])))
        laps = [f["lapvar"] for f in frames]
        low_frac = float(np.mean([f["low"] for f in frames]))
        summary[r["name"]] = {"route": i, "n_frames": len(frames), "lapvar_min": min(laps),
                              "lapvar_med": float(np.median(laps)), "low_frac": round(low_frac, 3),
                              "hole_frac_mean": round(float(np.mean([f["hole_frac"] for f in frames])), 4),
                              "ground_depth_3dgs_m": {"min": round(min(down), 2), "med": round(float(np.median(down)), 2),
                                                      "max": round(max(down), 2), "n": len(down)} if down else None,
                              "frames": frames}
        ok = low_frac < PROBE_LOW_FRAC_MAX
        log(f"probe: r{i} {r['name']}: {len(frames)} 帧 lapvar min/med {min(laps):.0f}/{np.median(laps):.0f} "
            f"low {low_frac:.0%} {'OK' if ok else 'BAD'}  俯视 3DGS 离地 {summary[r['name']]['ground_depth_3dgs_m']}"
            f"  per-frame {[int(v) for v in laps]}")
        if not ok:
            bad.append(r["name"])
    write_json(os.path.join(OUT, "routes_quality.json"), summary)
    if bad:
        raise SystemExit(f"路线低质帧比例 >= {PROBE_LOW_FRAC_MAX:.0%}：{bad}（看 OUT/probe_r*.png 改路线）")


# ---------------------------------------------------------------- 6 教师冒烟 + 7 三视角视频
def step_smoke():
    """collect_3dgs --policy astar --dynamics cf_sitl，每路线 SMOKE_EPISODES 集，代理碰撞即停；写 smoke_report.json。"""
    import importlib
    scene_mod = importlib.import_module("sim.3dgs.scene")
    scene = scene_mod.load_scene(SCENE_JSON)
    report = {}
    for i, r in enumerate(scene["routes"]):
        out = os.path.join(OUT, "smoke", f"route_{i}")
        if os.path.isfile(os.path.join(out, "run.json")):
            log(f"smoke: route {i} 已有 run.json，复用")
        else:
            if os.path.isdir(out):
                import shutil
                shutil.rmtree(out)
            run([PY, "-m", "sim.3dgs.collect_3dgs", "--scene", SCENE_JSON, "--out", out, "--route", str(i),
                 "--episodes", str(SMOKE_EPISODES), "--policy", "astar", "--dynamics", "cf_sitl",
                 "--hold", str(HOLD_S), "--seed", str(SEED0 + 100 * i), "--stop-on-collision"],
                f"smoke route {i} {r['name']}")
        demo = importlib.import_module("demo_3dgs_routes")          # 复用 Callinstrasse demo 的单集摘要
        rows = [demo.episode_row(os.path.join(out, ep), scene.get("reach_radius_m", 2.0))
                for ep in sorted(d for d in os.listdir(out) if d.startswith("ep_"))
                if os.path.isfile(os.path.join(out, ep, "episode_manifest.json"))]
        report[r["name"]] = rows
        log(f"smoke: route {i} {r['name']}: " + "; ".join(
            f"seed {x['seed']} steps={x['steps']} reached={x['reached']} collision={x['collision']} "
            f"min_clear={x['min_clearance_m']} low={x.get('n_low_quality')} anomaly={x['anomaly']}" for x in rows))
    write_json(os.path.join(OUT, "smoke_report.json"), report)


def step_video():
    """每路线第 1 集 -> sim.3dgs.video 三视角（main 上的 video.py，不含 VIZ 分支改动）。"""
    import importlib
    scene_mod = importlib.import_module("sim.3dgs.scene")
    scene = scene_mod.load_scene(SCENE_JSON)
    eps = []
    for i, _r in enumerate(scene["routes"]):
        ep = os.path.join(OUT, "smoke", f"route_{i}", "ep_00000")
        if os.path.isfile(os.path.join(ep, "trace.jsonl")):
            eps.append(ep)
    if not eps:
        raise SystemExit("video: 没有可用的 ep_00000/trace.jsonl（先跑 smoke）")
    run([PY, "-m", "sim.3dgs.video", "--scene", SCENE_JSON, "--ep", *eps], "video 三视角")


# ---------------------------------------------------------------- 主流程
def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    rc = 0
    try:
        for step in STEPS:
            if step not in RUN_STEPS:
                continue
            fn = globals().get(f"step_{step}")
            if fn is None:
                raise SystemExit(f"步骤 {step} 还没实现")
            log(f"===== {step} =====")
            fn()
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
