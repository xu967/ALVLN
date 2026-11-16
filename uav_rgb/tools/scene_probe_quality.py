#!/usr/bin/env python3
"""沿路线渲染机载帧，量 3DGS 清晰度：每帧 lapvar / hole_frac / low，路线级低质帧比例，关键帧拼图，高度阶梯。

为什么要有它：3DGS 只在采集视角附近清晰，离相机轨迹 1–2 m 就糊成雾墙（Callinstrasse 原 r4 出生帧 lapvar 1–2），
Mip-NeRF 360 这类环绕中心物体采集的场景外围更甚。路线写进 scene.json 之前先沿 A* 路径渲染一遍，低质帧比例
超过 LOW_FRAC_MAX 的路线不收（docs/DATASET_3DGS_SELF_CAPTURE.md §3 步 7）。阈值和判定函数直接用采集端的
sim/3dgs/collect_3dgs.FRAME_LAPVAR_MIN / frame_quality，探测说"低质"的帧，采集出来 actions.jsonl 里也标 low。
高度阶梯：同一水平位姿在 z=1/2/3 m 各渲一帧，肉眼核对 nav z 和离地高度是否一致（Callinstrasse 地面高度 P0：
nav_to_world 差 1.6 m，整批 pilot 是三楼视角，没人看图就放量了）。

用法（仓库根，3dgs-venv，GPU）：
  python tools/scene_probe_quality.py --scene $DATA_ROOT/uav_rgb/3dgs-scenes/garden/scene.json [--route 0 2] [--out DIR]
  python tools/scene_probe_quality.py --scene ... --candidate 'r_test:-4,0,0,5,2'   # 还没写进 scene.json 的候选：name:x,y,yaw,wx,wy[,z]
  python tools/scene_probe_quality.py --self-check                                 # mock 渲染器 + 合成走廊，不需要 GPU
产物（默认 <scene 目录>/probe_quality/）：routes_quality.json、route<i>_<name>.png（关键帧拼图）、ladder_<name>.png。
退出码：每条路线有路径、锐度与每帧深度有效率均过门为 0，否则 1。
默认深度阈值仅用于诊断，正式航线验收必须先在验证集冻结；本工具尚不证明训练相机覆盖。
"""
import argparse
import importlib
import json
import math
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
scene_mod = importlib.import_module("sim.3dgs.scene")
collision_mod = importlib.import_module("sim.3dgs.collision")
planner_mod = importlib.import_module("sim.3dgs.planner")
renderer_mod = importlib.import_module("sim.3dgs.renderer")
collect_mod = importlib.import_module("sim.3dgs.collect_3dgs")
frame_quality = collect_mod.frame_quality           # 单帧判定：lapvar / hole_frac / low（阈值 FRAME_LAPVAR_MIN）
FRAME_LAPVAR_MIN = collect_mod.FRAME_LAPVAR_MIN     # 与采集端同一个数（60），这里不另写一份

# ---------------------------------------------------------------- 配置（改这里）
STEP_M = 0.5                            # 采样间距：cf_sitl 每 0.5 s 步前进约 0.15 m，0.5 m ≈ 3 步一帧；雾区尺度 ≥ 1 m 不会漏
SWEEP_STEP_M = 1.0                      # 每 1.0 m 做一次 ±45°/±90° 转向扫视探针
SWEEP_OFFSETS_DEG = (-90.0, -45.0, 45.0, 90.0)
LOWER_ALPHA_THR = 0.50                  # 下半幅像素有效不透明度门槛 alpha >= 0.50
LOWER_ALPHA50_MIN = 0.85                # 下半幅有效不透明度比例下限
LOOKAHEAD_M = planner_mod.LOOKAHEAD_M   # 航向 = 朝前方 0.7 m 的路径点，与教师纯追踪一致，帧朝向和采集时一样
LOW_FRAC_MAX = 0.10                     # 路线门槛：低质帧 > 10% 不收（DATASET_3DGS_SELF_CAPTURE.md §3 步 7）
SHEET_COLS = 6                          # 拼图每行帧数；320×240 × 6 = 1920 宽，一屏能看
SHEET_MAX = 18                          # 拼图最多帧数，超过就等距抽（15 m 路线 0.5 m 一帧是 31 帧）
LADDER_Z = (1.0, 2.0, 3.0)              # 高度阶梯：1 m 桌面/长椅齐眼，2 m 俯视桌面，3 m 进树冠；差 1 m 肉眼可辨
LADDER_PITCH_DEG = -30.0                # 第四格：路线高度俯视 30°，看脚下地面有没有被当成"地面"
_DATA_TMP = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb", "tmp")
if "TMPDIR" not in os.environ and os.path.isdir(_DATA_TMP):
    tempfile.tempdir = _DATA_TMP        # 根分区只剩几 GB，自检的合成场景也写数据盘


def sample_poses(path, spawn_yaw_deg, step_m=STEP_M, lookahead_m=LOOKAHEAD_M):
    """A* 路径 [(x,y),...] -> [(x,y,yaw_deg),...]：首帧用出生朝向，之后每 step_m 一帧，航向朝前方 lookahead_m 的路径点，末点必含。"""
    if not path:
        return []
    poses = [(float(path[0][0]), float(path[0][1]), float(spawn_yaw_deg))]
    if len(path) == 1:
        return poses

    def heading_at(k):
        acc, j = 0.0, k
        while j + 1 < len(path) and acc < lookahead_m:
            acc += math.hypot(path[j + 1][0] - path[j][0], path[j + 1][1] - path[j][1])
            j += 1
        if j == k:                                   # 已是末点：沿用上一段方向
            j, k = k, k - 1
        return math.degrees(math.atan2(path[j][1] - path[k][1], path[j][0] - path[k][0]))

    acc = 0.0
    for k in range(1, len(path)):
        acc += math.hypot(path[k][0] - path[k - 1][0], path[k][1] - path[k - 1][1])
        if acc >= step_m or k == len(path) - 1:
            poses.append((float(path[k][0]), float(path[k][1]), heading_at(k)))
            acc = 0.0
    return poses


def sample_sweep_poses(path, spawn_yaw_deg=0.0, step_m=SWEEP_STEP_M,
                       offsets_deg=SWEEP_OFFSETS_DEG, lookahead_m=LOOKAHEAD_M):
    """沿路径每 step_m 取基点并附加 offsets_deg 转向扫视位姿 [(x, y, yaw_deg), ...]，检查转向时是否撞上未观测雾墙。"""
    base = sample_poses(path, spawn_yaw_deg, step_m=step_m, lookahead_m=lookahead_m)
    out = []
    for x, y, yaw in base:
        for off in offsets_deg:
            ang = (yaw + float(off) + 180.0) % 360.0 - 180.0
            out.append((x, y, ang))
    return out


def frustum_coverage_mask(cameras, grid_origin, grid_shape, voxel_m,
                          hfov_deg=60.0, max_range_m=15.0, min_cams=3):
    """计算 2D 导航栅格的训练相机视锥覆盖掩码：返回 bool[nx, ny]，True = 被 < min_cams 台相机观测到（不可通行盲区）。
    cameras: [(x, y, yaw_deg), ...]（nav 系米与逆时针度）。"""
    nx, ny = int(grid_shape[0]), int(grid_shape[1])
    ox, oy = float(grid_origin[0]), float(grid_origin[1])
    vx = float(voxel_m)
    if not cameras:
        return np.ones((nx, ny), dtype=bool)
    xs = ox + (np.arange(nx, dtype=np.float64) + 0.5) * vx
    ys = oy + (np.arange(ny, dtype=np.float64) + 0.5) * vx
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    counts = np.zeros((nx, ny), dtype=np.int32)
    cos_half = math.cos(math.radians(float(hfov_deg) * 0.5))
    r2_max = float(max_range_m) ** 2
    for cx, cy, cyaw in cameras:
        dx = gx - float(cx)
        dy = gy - float(cy)
        d2 = dx * dx + dy * dy
        dist = np.sqrt(np.maximum(d2, 1e-12))
        rad = math.radians(float(cyaw))
        dot = (dx * math.cos(rad) + dy * math.sin(rad)) / dist
        in_view = (d2 <= r2_max) & ((d2 <= 1e-6) | (dot >= cos_half))
        counts += in_view.astype(np.int32)
    return counts < int(min_cams)


def probe_poses(renderer, poses, z, pitch_deg=0.0):
    """逐位姿渲染并判质量；返回 (frames 列表, rgb 列表)。frames 每项含 lower_alpha50_frac。"""
    frames, rgbs = [], []
    for (x, y, yaw) in poses:
        rgb, depth, _ms = renderer.render_pose(x, y, z, yaw, pitch_deg=pitch_deg)
        q = frame_quality(rgb, depth)
        valid = np.isfinite(depth) & (depth > 0)
        q["depth_valid_frac"] = float(np.mean(valid)) if depth.size else 0.0
        q["hole_frac"] = 1.0 - q["depth_valid_frac"]
        alpha = getattr(renderer, "last_alpha", None)
        if alpha is not None and alpha.shape == depth.shape and alpha.size:
            lower = alpha[alpha.shape[0] // 2:, :]
            q["lower_alpha50_frac"] = float(np.mean(lower >= LOWER_ALPHA_THR))
        else:
            lower_d = valid[valid.shape[0] // 2:, :] if valid.size else valid
            q["lower_alpha50_frac"] = float(np.mean(lower_d)) if lower_d.size else 0.0
        dv = depth[valid]
        frames.append({"pose": [round(x, 3), round(y, 3), round(z, 3), round(yaw, 1)], **q,
                       "depth_med_m": round(float(np.median(dv)), 2) if dv.size else None})
        rgbs.append(rgb)
    return frames, rgbs


def summarize(name, frames, path_m, low_frac_max=LOW_FRAC_MAX, min_depth_valid_frac=0.9,
              min_lower_alpha50_frac=LOWER_ALPHA50_MIN, sweep_frames=None):
    """锐度、全幅有效深度、下半幅 alpha>=0.50 与转向扫视分别验收。"""
    if not math.isfinite(min_depth_valid_frac) or not 0 < min_depth_valid_frac <= 1:
        raise ValueError("min_depth_valid_frac 必须在 (0, 1] 内")
    depth_ok = bool(frames) and all(
        f.get("depth_valid_frac", 0.0) >= min_depth_valid_frac for f in frames)
    lower_alpha_vals = [f.get("lower_alpha50_frac", f.get("depth_valid_frac", 0.0)) for f in frames]
    lower_alpha_ok = bool(frames) and all(v >= min_lower_alpha50_frac for v in lower_alpha_vals)
    sweep_pass = True
    sweep_low_frac = None
    if sweep_frames is not None:
        sweep_low_frac = round(sum(int(f["low"]) for f in sweep_frames) / len(sweep_frames), 4) if sweep_frames else None
        sweep_pass = bool(sweep_frames) and (sweep_low_frac <= max(low_frac_max, 0.25)) and all(
            f.get("lower_alpha50_frac", 1.0) >= min_lower_alpha50_frac * 0.8 for f in sweep_frames
        )
    laps = [f["lapvar"] for f in frames]
    n_low = sum(int(f["low"]) for f in frames)
    low_frac = n_low / len(frames) if frames else None
    return {"name": name, "n_frames": len(frames), "path_m": round(path_m, 2),
            "n_low": n_low, "low_frac": round(low_frac, 4) if frames else None,
            "lapvar_min": min(laps) if laps else None,
            "lapvar_p10": round(float(np.percentile(laps, 10)), 1) if laps else None,
            "lapvar_med": round(float(np.median(laps)), 1) if laps else None,
            "hole_frac_max": max(f["hole_frac"] for f in frames) if frames else None,
            "lower_alpha50_min": round(float(min(lower_alpha_vals)), 4) if lower_alpha_vals else None,
            "lower_alpha_pass": lower_alpha_ok,
            "sweep_n_frames": len(sweep_frames) if sweep_frames is not None else 0,
            "sweep_low_frac": sweep_low_frac,
            "sweep_pass": sweep_pass,
            "lapvar_min_threshold": FRAME_LAPVAR_MIN, "low_frac_max": low_frac_max,
            "min_depth_valid_frac": min_depth_valid_frac, "depth_pass": depth_ok,
            "pass": bool(frames) and low_frac <= low_frac_max and depth_ok and lower_alpha_ok and sweep_pass}


def contact_sheet(rgbs, labels, cols=SHEET_COLS, max_frames=SHEET_MAX):
    """等距抽 <= max_frames 帧，左上角写标签，按 cols 列拼图；返回 BGR 图。"""
    import cv2
    idx = list(range(len(rgbs)))
    if len(idx) > max_frames:
        idx = [int(round(i)) for i in np.linspace(0, len(rgbs) - 1, max_frames)]
    tiles = []
    for i in idx:
        img = cv2.cvtColor(np.ascontiguousarray(rgbs[i]), cv2.COLOR_RGB2BGR)
        cv2.putText(img, labels[i], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        tiles.append(img)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + cols], axis=1) for i in range(0, len(tiles), cols)]
    return np.concatenate(rows, axis=0)


def probe_route(renderer, planner, name, spawn, waypoint, out_dir, pitch_deg=0.0, min_depth_valid_frac=0.9):
    """一条路线：A* 路径 -> 采样位姿 -> 渲染判质量 -> 拼图 + 高度阶梯；返回汇总 dict（含 frames）。"""
    import cv2
    sx, sy, syaw = float(spawn["x"]), float(spawn["y"]), float(spawn.get("yaw_deg", 0.0))
    z = float(spawn.get("z", 2.0))
    path = planner.plan((sx, sy), (float(waypoint["x"]), float(waypoint["y"])))
    tag = name.replace("/", "_")
    if path is None:
        return {"name": name, "n_frames": 0, "path_m": None, "pass": False, "reason": "A* 无路径", "frames": []}
    path[0] = (sx, sy)                                   # 首点用精确出生点（plan 返回格中心）
    path_m = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))
    poses = sample_poses(path, syaw)
    frames, rgbs = probe_poses(renderer, poses, z, pitch_deg)
    sweep_poses = sample_sweep_poses(path, syaw)
    sweep_frames, _ = probe_poses(renderer, sweep_poses, z, pitch_deg)
    labels = [f"{i:02d} ({f['pose'][0]:.1f},{f['pose'][1]:.1f},{f['pose'][3]:.0f}) lap={f['lapvar']:.0f}{' LOW' if f['low'] else ''}"
              for i, f in enumerate(frames)]
    cv2.imwrite(os.path.join(out_dir, f"route_{tag}.png"), contact_sheet(rgbs, labels))
    ladder_rgbs, ladder_labels = [], []
    for zz in LADDER_Z:
        rgb, depth, _ = renderer.render_pose(sx, sy, zz, syaw, pitch_deg=pitch_deg)
        ladder_rgbs.append(rgb)
        ladder_labels.append(f"z={zz:g} pitch={pitch_deg:g} lap={frame_quality(rgb, depth)['lapvar']:.0f}")
    rgb, depth, _ = renderer.render_pose(sx, sy, z, syaw, pitch_deg=LADDER_PITCH_DEG)
    ladder_rgbs.append(rgb)
    ladder_labels.append(f"z={z:g} pitch={LADDER_PITCH_DEG:g} lap={frame_quality(rgb, depth)['lapvar']:.0f}")
    cv2.imwrite(os.path.join(out_dir, f"ladder_{tag}.png"), contact_sheet(ladder_rgbs, ladder_labels, cols=2))
    out = summarize(name, frames, path_m, min_depth_valid_frac=min_depth_valid_frac, sweep_frames=sweep_frames)
    out.update({"spawn": spawn, "waypoint": waypoint, "frames": frames,
                "sheet": f"route_{tag}.png", "ladder": f"ladder_{tag}.png"})
    return out


def parse_candidate(text):
    """'name:x,y,yaw,wx,wy[,z]' -> (name, spawn, waypoint)；z 默认 2.0。"""
    name, _, nums = str(text).partition(":")
    vals = [float(v) for v in nums.split(",")]
    if not name or len(vals) not in (5, 6):
        raise ValueError(f"--candidate 需要 name:x,y,yaw,wx,wy[,z]，实际 {text!r}")
    z = vals[5] if len(vals) == 6 else 2.0
    return name, {"x": vals[0], "y": vals[1], "z": z, "yaw_deg": vals[2]}, {"x": vals[3], "y": vals[4], "z": z}


def run(scene_path, routes=None, candidates=(), out_dir=None, mock=False, device="cuda", min_depth_valid_frac=0.9):
    """主流程：读场景 -> 占据/规划器/渲染器各建一次 -> 逐路线探测 -> routes_quality.json；返回 (rows, all_pass)。"""
    scene = scene_mod.load_scene(scene_path)
    out_dir = out_dir or os.path.join(scene["dir"], "probe_quality")
    os.makedirs(out_dir, exist_ok=True)
    g = (scene_mod.read_gaussian_ply(scene["ply_path"])
         if scene.get("collision", {}).get("source", "gaussians") == "gaussians" else None)
    occ = collision_mod.from_scene(scene, gaussians=g)
    planner = planner_mod.OccPlanner(occ)
    renderer = renderer_mod.make_renderer(scene, mock=mock, device=device)
    pitch = float(scene.get("camera", {}).get("pitch_deg", 0.0))
    jobs = []
    n_routes = len(scene.get("routes") or [])
    for i in (routes if routes is not None else range(max(1, n_routes))):
        sc = scene_mod.apply_route(dict(scene), i)
        jobs.append(((sc.get("route") or {}).get("name") or f"route{i}", sc["spawn"], sc["waypoint"], i))
    for text in candidates:
        name, sp, wp = parse_candidate(text)
        jobs.append((name, sp, wp, None))
    rows = []
    for name, sp, wp, idx in jobs:
        row = probe_route(renderer, planner, name, sp, wp, out_dir, pitch, min_depth_valid_frac)
        row["route"] = idx
        rows.append(row)
        print(f"{name:28s} path {row.get('path_m')} m  frames {row['n_frames']:3d}  low {row.get('n_low', '-')} "
              f"({row.get('low_frac')})  lapvar min/p10/med {row.get('lapvar_min')}/{row.get('lapvar_p10')}/{row.get('lapvar_med')}  "
              f"hole_max {row.get('hole_frac_max')}  {'PASS' if row['pass'] else ('FAIL ' + row.get('reason', '')).strip()}", flush=True)
    report = {"schema": "uavrgb.3dgs_routes_quality.v2", "scene": scene["path"], "scene_name": scene.get("name"),
              "renderer": renderer.stats(), "step_m": STEP_M, "lookahead_m": LOOKAHEAD_M,
              "lapvar_min_threshold": FRAME_LAPVAR_MIN, "low_frac_max": LOW_FRAC_MAX, "camera_pitch_deg": pitch,
              "min_depth_valid_frac": min_depth_valid_frac,
              "threshold_status": "user_supplied_or_uncalibrated_default", "routes": rows}
    with open(os.path.join(out_dir, "routes_quality.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    return rows, all(r["pass"] for r in rows)


def self_check():
    """mock 渲染器 + 合成走廊：产物齐全、帧数与路径长度对得上、JSON 字段齐；不需要 GPU。"""
    with tempfile.TemporaryDirectory(prefix="probe-selfcheck-") as tmp:
        scene_mod.make_corridor(tmp, seed=0)
        rows, _ok = run(os.path.join(tmp, "scene.json"), mock=True, out_dir=os.path.join(tmp, "pq"),
                        candidates=["back:28,0,180,2,0"])
        assert len(rows) == 2 and rows[0]["name"] == "route0" and rows[1]["name"] == "back", rows
        for r in rows:
            assert r["path_m"] is not None and r["n_frames"] >= int(r["path_m"] / STEP_M), r
            assert set(r["frames"][0]) >= {"pose", "lapvar", "hole_frac", "low", "depth_med_m"}, r["frames"][0]
            assert r["n_low"] == sum(f["low"] for f in r["frames"]) and 0.0 <= r["low_frac"] <= 1.0, r
            assert os.path.isfile(os.path.join(tmp, "pq", r["sheet"])) and os.path.isfile(os.path.join(tmp, "pq", r["ladder"]))
        rep = json.load(open(os.path.join(tmp, "pq", "routes_quality.json")))
        assert rep["lapvar_min_threshold"] == FRAME_LAPVAR_MIN and len(rep["routes"]) == 2
        print(f"scene_probe_quality self-check PASS: {[(r['name'], r['n_frames'], r['low_frac']) for r in rows]}")


def main(argv=None):
    p = argparse.ArgumentParser(description="沿路线渲染机载帧并量 3DGS 清晰度（雾区/发糊探测）")
    p.add_argument("--scene", help="scene.json")
    p.add_argument("--route", type=int, nargs="*", default=None, help="routes 索引（默认全部；无 routes 时查顶层 spawn/waypoint）")
    p.add_argument("--candidate", action="append", default=[], help="候选路线 name:x,y,yaw,wx,wy[,z]，可多次")
    p.add_argument("--out", default=None, help="输出目录（默认 <scene 目录>/probe_quality）")
    p.add_argument("--mock", action="store_true", help="mock 渲染器（只验流程，lapvar 无意义）")
    p.add_argument("--device", default="cuda")
    p.add_argument("--min-depth-valid-frac", type=float, default=0.9,
                   help="每帧有效深度比例下限；默认 0.9 仅诊断，正式验收须在验证集冻结")
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args(argv)
    if args.self_check:
        self_check()
        return 0
    if not args.scene:
        p.error("--scene 必填（或 --self-check）")
    _rows, ok = run(args.scene, args.route, args.candidate, args.out, args.mock, args.device, args.min_depth_valid_frac)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
