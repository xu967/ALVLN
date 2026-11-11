#!/usr/bin/env python3
"""深度列距离的几何对拍：在真渲染器（gsplat）上用已知几何的合成场景，查清 depth 来源的三件事，再把三路列距离放在一起比。

  1. 渲染深度是 z 深度（相机系 z）还是沿射线的距离；
  2. 机体系与相机系的轴向 / 像素中心约定（杆子在左，图里是不是在左；杆轴落在哪一列），
     相机相对机体中心的前向偏移（camera_mount.xyz[0] = 0.045 m）是不是真的在渲染里生效；
  3. 同一位姿上三路列距离：解析表面（rule_policy.column_distances，Gazebo 同一个函数）、depth（rule_oracle.depth_columns）、
     occ（rule_oracle.occ_columns，膨胀栅格边界）的差，以及 depth 比 occ 远 0.3 m 以上的列占比（任务卡定义的「漏检」）。
场景：x = 4.0 的平面墙（y ∈ [-6, 6]，z ∈ [0, 3]）+ 两根竖直圆柱（(2.5, 0.0, 半径 0.15)、(3.0, 1.4, 半径 0.10)），高斯各向同性 σ=0.03 m、
间距 0.04 m、不透明度 0.99；渲染选项与实验同值（320×240、hfov 60°、supersample 2）。位姿用 cf_sitl 的四元数约定（ZYX，机体 FLU -> nav）。

用法（GPU 只许用 uav_rgb 的那张卡）：
  CUDA_VISIBLE_DEVICES=<UUID> TORCH_EXTENSIONS_DIR=<gsplat 扩展目录> <3dgs-venv python> depth_geometry_check.py --out <目录>
"""
import argparse
import importlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim", "tools"))
scene_mod = importlib.import_module("sim.3dgs.scene")
renderer_mod = importlib.import_module("sim.3dgs.renderer")
coll = importlib.import_module("sim.3dgs.collision")
dyn_mod = importlib.import_module("sim.3dgs.dynamics")
ro = importlib.import_module("sim.3dgs.rule_oracle")
import rule_policy as rp  # noqa: E402

MOUNT = {"xyz": [0.045, 0.0, 0.015], "pitch_deg": 0.0}      # e1_mini.json camera_mount（run 时再核对文件里的值）
Z_BODY = 1.2
POLES = [(2.5, 0.0, 0.15), (3.0, 1.4, 0.10)]                # (x, y, 半径)
WALL_X = 4.0
WIDEN_SWEEP_M = (0.0, 0.02, 0.04, 0.06, 0.08)                # 杆半径外扩扫描（米）
W, H, HFOV = 320, 240, 60.0


def quat_zyx(roll_deg, pitch_deg, yaw_deg):
    r, p, y = (math.radians(v) / 2 for v in (roll_deg, pitch_deg, yaw_deg))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def build_scene(out_dir, spacing=0.04, sigma=0.03):
    """写合成场景 PLY + scene.json（schema 同 make_corridor），返回 load_scene 的结果。"""
    os.makedirs(out_dir, exist_ok=True)
    pts = []
    ys, zs = np.meshgrid(np.arange(-6.0, 6.0 + 1e-9, spacing), np.arange(0.0, 3.0 + 1e-9, spacing))
    pts.append(np.stack([np.full(ys.size, WALL_X), ys.ravel(), zs.ravel()], 1))
    for px, py, rad in POLES:
        n_ang = max(8, int(round(2 * math.pi * rad / spacing)))
        ang, z = np.meshgrid(np.arange(n_ang) * 2 * math.pi / n_ang, np.arange(0.0, 3.0 + 1e-9, spacing))
        pts.append(np.stack([px + rad * np.cos(ang.ravel()), py + rad * np.sin(ang.ravel()), z.ravel()], 1))
    means = np.concatenate(pts).astype(np.float32)
    colors = np.tile(np.array([0.6, 0.55, 0.5], np.float32), (len(means), 1))
    ply = os.path.join(out_dir, "scene.ply")
    scene_mod.write_gaussian_ply(ply, means, colors, scales=np.full((len(means), 3), sigma, np.float32),
                                 opacities=np.full(len(means), 0.99, np.float32))
    scene = {"schema": "uavrgb.3dgs_scene.v1", "name": "depth_geometry_check", "source": "self-generated (depth_geometry_check.py)",
             "license": "self-authored", "ply": "scene.ply", "nav_to_world": np.eye(4).tolist(), "world_up": "z", "ground_z": 0.0,
             "spawn": {"x": 0.0, "y": 0.0, "z": Z_BODY, "yaw_deg": 0.0}, "waypoint": {"x": 3.5, "y": 0.0, "z": Z_BODY},
             "reach_radius_m": 1.0, "camera": {"width": W, "height": H, "fov_deg": HFOV, "pitch_deg": 0.0},
             "collision": {"z_slice": [1.0, 1.4], "voxel_m": 0.1, "envelope": "e1_mini"}, "max_steps": 20, "step_s": 0.5}
    scene_mod.save_scene(scene, os.path.join(out_dir, "scene.json"))
    return scene_mod.load_scene(os.path.join(out_dir, "scene.json")), len(means)


def analytic_depth(view, cam_override=None):
    """解析光线投射的 z 深度图与射线距离图（米，inf = 没打到）。cam_override：用别的相机中心（对照「没有前向偏移」的假设）。"""
    fx = 0.5 * W / math.tan(math.radians(HFOV) / 2)
    r_cn, t_c = view[:3, :3], view[:3, 3]
    cam = -r_cn.T @ t_c if cam_override is None else np.asarray(cam_override, float)
    v, u = np.mgrid[0:H, 0:W]
    dc = np.stack([(u + 0.5 - W / 2) / fx, (v + 0.5 - H / 2) / fx, np.ones((H, W))], -1).reshape(-1, 3)
    dn = dc @ r_cn
    best = np.full(len(dn), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (WALL_X - cam[0]) / dn[:, 0]
    p = cam[None, :] + t[:, None] * dn
    ok = (t > 0) & (np.abs(p[:, 1]) <= 6.0) & (p[:, 2] >= 0.0) & (p[:, 2] <= 3.0)
    best = np.where(ok, t, best)
    kind = np.where(ok, 1, 0)                                                # 1 = 墙，2.. = 杆
    for i, (px, py, rad) in enumerate(POLES):
        ox, oy = cam[0] - px, cam[1] - py
        a = dn[:, 0] ** 2 + dn[:, 1] ** 2
        b = 2 * (ox * dn[:, 0] + oy * dn[:, 1])
        disc = b * b - 4 * a * (ox * ox + oy * oy - rad * rad)
        tt = (-b - np.sqrt(np.maximum(disc, 0.0))) / (2 * a)
        z = cam[2] + tt * dn[:, 2]
        ok = (disc >= 0) & (tt > 0) & (z >= 0.0) & (z <= 3.0) & (tt < best)
        best = np.where(ok, tt, best)
        kind = np.where(ok, 2 + i, kind)
    zdep = best.reshape(H, W)
    ray = (best * np.sqrt(np.einsum("ij,ij->i", dc, dc))).reshape(H, W)       # 相机系点到相机中心的欧氏距离
    return zdep, ray, kind.reshape(H, W), cam


def interior(zdep):
    """远离深度不连续（杆的轮廓、墙边）的像素：与四邻差 < 0.05 m 且有限。"""
    z = np.where(np.isfinite(zdep), zdep, -1.0)
    m = np.isfinite(zdep)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2)):
        s = np.roll(np.roll(z, dy, 0), dx, 1)
        m &= np.abs(z - s) < 0.05
    m[:3], m[-3:], m[:, :3], m[:, -3:] = False, False, False, False
    return m


def stats(x):
    x = np.asarray(x, float)
    return {"n": int(x.size), "median": round(float(np.median(x)), 4), "p90": round(float(np.percentile(x, 90)), 4),
            "max": round(float(x.max()), 4)} if x.size else {"n": 0}


def pose_view(x, y, yaw, roll=0.0, pitch=0.0):
    quat = quat_zyx(roll, pitch, yaw)
    return quat, scene_mod.viewmat_from_quat((x, y, Z_BODY), quat, MOUNT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(ROOT, "sim", "dynamics", "e1_mini.json"), encoding="utf-8") as f:
        mount_file = json.load(f)["camera_mount"]["value"]
    assert mount_file["xyz"] == MOUNT["xyz"] and float(mount_file["pitch_deg"]) == MOUNT["pitch_deg"], mount_file
    scene, n_gauss = build_scene(os.path.join(args.out, "geom_scene"))
    renderer = renderer_mod.make_renderer(scene, False, None, None, None, "cuda")
    occ = coll.from_scene(scene)
    half_h = occ.body_half_height
    res = {"scene": {"n_gaussians": n_gauss, "wall_x": WALL_X, "poles": POLES, "body_z": Z_BODY, "camera_mount": mount_file,
                     "occ_voxel_m": occ.voxel, "occ_dilation_cells": list(occ.dilation_cells), "render": renderer.stats()},
           "poses": {}}
    # 欧拉角 <-> 四元数约定：dynamics 里的 _quat_to_euler_deg 必须还原出构造时的 roll/pitch/yaw
    for e in ((3.0, -5.0, 30.0), (0.0, 0.0, -140.0)):
        back = dyn_mod._quat_to_euler_deg(quat_zyx(*e))
        assert all(abs(a - b) < 1e-9 for a, b in zip(back, e)), (e, back)
    res["euler_roundtrip_ok"] = True

    poses = {"A 朝 +x、杆在正前": (0.0, 0.0, 0.0, 0.0, 0.0), "B 偏航 +30°（机头朝左）": (-0.5, -0.8, 30.0, 0.0, 0.0),
             "C 偏航 −20°（机头朝右，杆 1 在右、杆 2 在左）": (0.0, 1.8, -20.0, 0.0, 0.0),
             "D 横滚 3° 俯仰 4°": (-0.5, 0.0, 0.0, 3.0, 4.0)}
    agg = {"z": [], "ray": [], "z_nooff": []}
    for name, (x, y, yaw, roll, pitch) in poses.items():
        _quat, view = pose_view(x, y, yaw, roll, pitch)
        _rgb, depth, ms = renderer.render_view(view)
        zdep, ray, kind, cam = analytic_depth(view)
        zdep0, _r0, _k0, _c0 = analytic_depth(view, cam_override=np.array([x, y, Z_BODY]))     # 假设相机在机体中心
        m = interior(zdep) & (depth > 0)
        e_z, e_ray, e_z0 = (np.abs(depth[m] - zdep[m]), np.abs(depth[m] - ray[m]), np.abs(depth[m] - zdep0[m]))
        r = {"render_ms": round(ms, 2), "interior_pixels": int(m.sum()),
             "err_vs_z_depth_m": stats(e_z), "err_vs_ray_distance_m": stats(e_ray),
             "err_vs_z_depth_if_camera_at_body_centre_m": stats(e_z0[np.isfinite(e_z0)])}
        agg["z"].append(e_z)
        agg["ray"].append(e_ray)
        agg["z_nooff"].append(e_z0[np.isfinite(e_z0)])
        # 两根杆的轴落在图像哪一列：渲染里杆像素的列质心 vs 解析 u = fx·a + W/2 − 0.5（像素 j 的中心在 j+0.5）；
        # 杆在视野外（u 落在 [0, W) 之外）时不比，标 in_view=False。机体左（y>0 且朝 +x）必须落在图左（u < W/2）。
        fx = 0.5 * W / math.tan(math.radians(HFOV) / 2)
        r_cn, t_c = view[:3, :3], view[:3, 3]
        axes = []
        for i, (px, py, _rad) in enumerate(POLES):
            pc = r_cn @ np.array([px, py, Z_BODY]) + t_c
            u_exp = fx * pc[0] / pc[2] + W / 2 - 0.5 if pc[2] > 0 else float("nan")
            in_view = bool(pc[2] > 0 and 0.0 <= u_exp < W)
            pole_mask = (kind == 2 + i) & (depth > 0) & (np.abs(depth - zdep) < 0.1)
            u_obs = float(np.mean(np.nonzero(pole_mask)[1])) if (in_view and pole_mask.any()) else None
            axes.append({"pole": i + 1, "in_view": in_view, "expected_u_px": round(float(u_exp), 2),
                         "observed_u_centroid_px": None if u_obs is None else round(u_obs, 2),
                         "abs_error_px": None if u_obs is None else round(abs(u_obs - u_exp), 2),
                         "rendered_side": None if u_obs is None else ("图左" if u_obs < W / 2 else "图右"),
                         "analytic_side": "图左" if pc[0] < 0 else "图右"})
        r["pole_axis_columns"] = axes
        # 三路列距离
        d_depth = ro.depth_columns(depth, view, Z_BODY, yaw, half_h, HFOV)
        d_depth_ray = ro.depth_columns(depth, view, Z_BODY, yaw, half_h, HFOV, depth_is="ray")
        d_occ = ro.occ_columns(occ, x, y, Z_BODY, yaw, HFOV)
        pillars = [{"x": px_, "y": py_, "radius": rad_} for px_, py_, rad_ in POLES] + [{"x": WALL_X + 0.5, "y": 0.0, "size": (1.0, 12.0)}]
        d_an = np.array(rp.column_distances(x, y, yaw, pillars, hfov_deg=HFOV))
        r["columns"] = {"analytic_surface": [round(float(v), 3) for v in d_an], "depth": [round(float(v), 3) for v in d_depth],
                        "depth_read_as_ray": [round(float(v), 3) for v in d_depth_ray], "occ": [round(float(v), 3) for v in d_occ]}
        both = d_an < ro.MAX_RANGE_M - 1e-6
        r["depth_minus_analytic_m"] = stats(np.abs(d_depth - d_an))
        r["depth_read_as_ray_minus_analytic_m"] = stats(np.abs(d_depth_ray - d_an))
        r["occ_minus_analytic_signed_m(只含解析值 < 4 的列)"] = {"median": round(float(np.median((d_occ - d_an)[both])), 4),
                                                         "min": round(float((d_occ - d_an)[both].min()), 4),
                                                         "max": round(float((d_occ - d_an)[both].max()), 4)} if both.any() else None
        # 轮廓外扩：合成杆是 σ=0.03 m 的高斯点云，渲染出的轮廓比解析圆柱宽一点；把解析半径加 w 后再比，看 depth − 解析 的最大值掉到多少
        sweep = {}
        for w in WIDEN_SWEEP_M:
            wide = [{"x": px_, "y": py_, "radius": rad_ + w} for px_, py_, rad_ in POLES] + pillars[len(POLES):]
            d_w = np.array(rp.column_distances(x, y, yaw, wide, hfov_deg=HFOV))
            sweep[f"{w:.2f}"] = stats(np.abs(d_depth - d_w))
        r["depth_minus_analytic_when_pole_radius_widened_by_w_m"] = sweep
        r["occ_extra_columns(解析 ≥ 4 而 occ < 4，膨胀使障碍横向变宽)"] = int(np.sum((d_an >= ro.MAX_RANGE_M - 1e-6) & (d_occ < ro.MAX_RANGE_M - 1e-6)))
        r["occ_missing_columns(解析 < 4 而 occ ≥ 4)"] = int(np.sum((d_an < ro.MAX_RANGE_M - 1e-6) & (d_occ >= ro.MAX_RANGE_M - 1e-6)))
        r["depth_vs_occ_abs_m"] = stats(np.abs(d_depth - d_occ))
        r["depth_farther_than_occ_by_0.3m_frac"] = round(float(np.mean(d_depth - d_occ >= 0.3)), 3)
        res["poses"][name] = r
    res["pooled_interior_pixel_error_m"] = {"vs_z_depth": stats(np.concatenate(agg["z"])), "vs_ray_distance": stats(np.concatenate(agg["ray"])),
                                           "vs_z_depth_if_camera_at_body_centre": stats(np.concatenate(agg["z_nooff"]))}
    # 结论由数字机械给出：哪个假设的中位误差最小
    pooled = res["pooled_interior_pixel_error_m"]
    res["verdict"] = {"depth_semantics": "z" if pooled["vs_z_depth"]["median"] < pooled["vs_ray_distance"]["median"] else "ray",
                      "camera_forward_offset_applied": pooled["vs_z_depth"]["median"] < pooled["vs_z_depth_if_camera_at_body_centre"]["median"]}
    with open(os.path.join(args.out, "geometry_check.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
