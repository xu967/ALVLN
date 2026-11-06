#!/usr/bin/env python3
"""3DGS 回合帧 -> 视频：compose_rgb_video 只拼 onboard RGB；compose_views_video 给 cf_sitl
采集集（trace.jsonl）渲染机载 | 跟随 | 顶视三格画面。不依赖 Gazebo/record_views。

三视角 CLI（从仓库根，3dgs-venv）：
  python -m sim.3dgs.video --scene <scene.json> --ep <ep_dir> [<ep_dir> ...]   # 每集写 <ep_dir>/views.webm + views_key_*.png
"""
import glob
import importlib
import json
import math
import os
import sys

import numpy as np

# 既能 `python -m sim.3dgs.video` 也能 `python sim/3dgs/video.py` 跑自检（包名含 3dgs，只能 importlib）。
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
scene_mod = importlib.import_module("sim.3dgs.scene")

# ---------------------------------------------------------------- 三视角视频配置（改这里）
VIEW_FPS = 10.0        # 与 CfSitlDynamics.trace_dt=0.1 s 对应：每个 trace 样本一帧，播放与仿真时间 1:1
FOLLOW_BACK_M = 2.47   # 跟随机位：机体后方 2.47 m、上方 0.4 m，斜距 2.5 m（任务要求）；俯角 atan(0.4/2.47)=9°
FOLLOW_UP_M = 0.4      #   把机体放在画面中心。Callinstrasse 是行人高度拍的重建，机位高于约 2.6 m 就被
#                          天空壳/漂浮高斯糊住（tmp/agent-3dgs-sitl/views_probe：上方 1.2 m 已看不清），所以只抬 0.4 m
TOP_MARGIN_M = 3.0     # 顶视用占据代理 2D 地图（3DGS 从上往下渲染全是漂浮物，4–20 m 高都试过）：
#                          地图窗口 = spawn–waypoint 包围盒各向外扩 3 m，到达半径 2 m 圈和两侧建筑都在图里
PANEL_W, PANEL_H = 320, 240   # 与机载帧同尺寸；三格横拼 960×240
STATUS_H = 36          # 底部状态栏高度：一行 0.5 号 Hershey 字体 + 边距
KEYFRAMES = 4          # 关键帧 PNG：首、1/3、2/3、末，预览用
ACTION_NAMES = ("SAFE", "FORWARD", "YAW_LEFT", "YAW_RIGHT")
# 跟随格里的机体线框（3DGS 没有机体模型，以前只画一个固定 6 px 圆）：尺寸取 e1_mini.json
# collision_envelope.xy_half_extent / motor_positions / prop_diameter，按真实位姿投影，roll/pitch 可见。
BODY_GUARD_HALF_M = 0.084
BODY_MOTOR_M = 0.042
BODY_PROP_R_M = 0.0325
BODY_ROTOR_Z_M = 0.015


def compose_rgb_video(frames_dir, out_path, fps=10.0):
    """把 frames_dir/rgb_*.png 按文件名顺序合成视频；返回实际写入的帧数。

    out_path 以 .webm 结尾用 VP8（浏览器可直接播放），否则用 mp4v（浏览器放不了）。
    本机 .venv-tools / 3dgs-venv 的 cv2 都没有 H.264 编码器。
    读不出或尺寸与首帧不同的帧会跳过，所以返回值可能小于 PNG 数。
    """
    import cv2

    paths = sorted(glob.glob(os.path.join(frames_dir, "rgb_*.png")))
    if not paths:
        raise RuntimeError(f"{frames_dir} 没有 rgb_*.png")
    first = cv2.imread(paths[0])
    if first is None:
        raise RuntimeError(f"无法读取 {paths[0]}")
    h, w = first.shape[:2]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fourcc = "VP80" if str(out_path).lower().endswith(".webm") else "mp4v"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc),
                             float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建 {out_path}（fourcc={fourcc}）")
    written = 0
    try:
        for path in paths:
            img = cv2.imread(path)
            if img is None or img.shape[:2] != (h, w):
                continue
            writer.write(img)
            written += 1
    finally:
        writer.release()
    return written


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _label(img, text, org, color=(255, 255, 255), scale=0.5):
    """ASCII 文字 + 黑底框（亮背景也能读；粗描边会改变 Hershey 字距产生重影，所以不用）。"""
    import cv2
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 3, y - th - 3), (x + tw + 3, y + base + 1), (0, 0, 0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _draw_body(img, pos, quat, cam_view, fov_deg, n2w=None):
    """把 E1 机体线框投到跟随格：保护罩方框（黄）、4 个桨圆、机头线（红）。

    pos/quat 是 nav 系机体位姿，cam_view 是跟随相机的 world->camera 4x4（含 nav_to_world），
    与渲染用的同一矩阵，所以线框与 3DGS 画面严格同一投影。机体在相机后方/过近时不画。
    """
    import cv2
    k = scene_mod.intrinsics(PANEL_W, PANEL_H, fov_deg)
    r_nb = scene_mod._quat_rotmat(quat)
    m = np.eye(4) if n2w is None else np.asarray(n2w, np.float64)

    def proj(p_body):
        p_nav = np.asarray(pos, np.float64) + r_nb @ np.asarray(p_body, np.float64)
        p_cam = cam_view @ np.append(m[:3, :3] @ p_nav + m[:3, 3], 1.0)
        if p_cam[2] <= 0.05:
            return None, None
        return (int(round(k[0, 0] * p_cam[0] / p_cam[2] + k[0, 2])),
                int(round(k[1, 1] * p_cam[1] / p_cam[2] + k[1, 2]))), p_cam[2]

    h = BODY_GUARD_HALF_M
    corners = [proj((sx * h, sy * h, BODY_ROTOR_Z_M))[0] for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
    if all(c is not None for c in corners):
        cv2.polylines(img, [np.array(corners, np.int32)], True, (0, 255, 255), 1, cv2.LINE_AA)
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        c, depth = proj((sx * BODY_MOTOR_M, sy * BODY_MOTOR_M, BODY_ROTOR_Z_M))
        if c is not None:
            cv2.circle(img, c, max(1, int(round(k[0, 0] * BODY_PROP_R_M / depth))), (0, 255, 255), 1, cv2.LINE_AA)
    center, nose = proj((0.0, 0.0, BODY_ROTOR_Z_M))[0], proj((h, 0.0, BODY_ROTOR_Z_M))[0]
    if center is not None and nose is not None:
        cv2.line(img, center, nose, (0, 0, 255), 2, cv2.LINE_AA)


def _panel(renderer, view=None, pose=None):
    """渲染一格并转 BGR、缩放到 PANEL_W×PANEL_H。"""
    import cv2
    rgb = (renderer.render_view(view, PANEL_W, PANEL_H) if view is not None
           else renderer.render_pose(*pose, width=PANEL_W, height=PANEL_H))[0]
    img = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    if img.shape[:2] != (PANEL_H, PANEL_W):
        img = cv2.resize(img, (PANEL_W, PANEL_H))
    return img


class _TopMap:
    """固定顶视：占据代理（collision.Occupancy.raw 沿 z 投影）的 2D 地图，x 朝上、y 朝左（ENU 俯视）。"""

    def __init__(self, scene, occ, extra_xy=()):
        """extra_xy：也要落在图里的点（整条轨迹）；多路线的 A* 绕行会跑出 spawn–waypoint 包围盒。"""
        import cv2
        sp, wp = scene["spawn"], scene["waypoint"]
        pts = [(sp["x"], sp["y"]), (wp["x"], wp["y"])] + [(float(x), float(y)) for x, y in extra_xy]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        span_x, span_y = max(xs) - min(xs) + 2 * TOP_MARGIN_M, max(ys) - min(ys) + 2 * TOP_MARGIN_M
        self.scale = max(span_x / PANEL_H, span_y / PANEL_W)           # m/px，x 映射到图高
        self.cx, self.cy = 0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys))
        self.base = np.full((PANEL_H, PANEL_W, 3), (235, 235, 235), np.uint8)
        if occ.raw.size:
            ii, jj = np.nonzero(occ.raw.any(axis=2))
            half = max(1, int(round(0.5 * occ.voxel / self.scale)))
            for i, j in zip(ii, jj):
                px, py = self.px(occ.origin[0] + (i + 0.5) * occ.voxel, occ.origin[1] + (j + 0.5) * occ.voxel)
                if -half <= px < PANEL_W + half and -half <= py < PANEL_H + half:
                    cv2.rectangle(self.base, (px - half, py - half), (px + half, py + half), (90, 90, 90), -1)
        r = int(round(float(scene.get("reach_radius_m", 1.5)) / self.scale))
        cv2.circle(self.base, self.px(wp["x"], wp["y"]), r, (0, 0, 220), 1)          # 到达半径
        cv2.circle(self.base, self.px(wp["x"], wp["y"]), 4, (0, 0, 220), -1)
        cv2.circle(self.base, self.px(sp["x"], sp["y"]), 4, (0, 160, 0), -1)         # spawn
        # 比例尺 + 朝向：地图窗口随路线变（8–20 m），没有比例尺看不出 2 m 到达圈有多大。
        bar = int(round(1.0 / self.scale))
        x0, y0 = 8, PANEL_H - 10
        cv2.line(self.base, (x0, y0), (x0 + bar, y0), (0, 0, 0), 2)
        cv2.putText(self.base, "1 m", (x0 + bar + 4, y0 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.arrowedLine(self.base, (PANEL_W - 14, PANEL_H - 8), (PANEL_W - 14, PANEL_H - 30), (0, 0, 0), 1, tipLength=0.3)
        cv2.putText(self.base, "+x", (PANEL_W - 34, PANEL_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    def px(self, x, y):
        return (int(round(PANEL_W / 2 - (y - self.cy) / self.scale)),
                int(round(PANEL_H / 2 - (x - self.cx) / self.scale)))

    def draw(self, path_xy, yaw):
        import cv2
        img = self.base.copy()
        pts = np.array([self.px(x, y) for x, y in path_xy], np.int32)
        if len(pts) >= 2:
            cv2.polylines(img, [pts], False, (0, 200, 255), 2)
        x, y = path_xy[-1]
        c = np.array(self.px(x, y), np.float64)
        # 三角：尖端朝航向（yaw=0 朝 +x = 图上方）
        tri = [c + 10 * np.array([-math.sin(yaw), -math.cos(yaw)]),
               c + 7 * np.array([-math.sin(yaw + 2.5), -math.cos(yaw + 2.5)]),
               c + 7 * np.array([-math.sin(yaw - 2.5), -math.cos(yaw - 2.5)])]
        cv2.fillPoly(img, [np.array(tri, np.int32)], (0, 170, 255))
        return img


def compose_views_video(ep_dir, scene, renderer, mount, occ, out_path=None, png_dir=None):
    """cf_sitl 采集集（trace.jsonl + actions.jsonl）-> 机载 | 跟随 | 顶视三格 .webm + 关键帧 PNG。

    机载 = 机体四元数 ⊗ camera_mount（roll/pitch 进画面）；跟随 = 同一 3DGS，只跟位置与偏航，机体
    后方 FOLLOW_BACK_M、上方 FOLLOW_UP_M；顶视 = 固定的占据代理 2D 地图 + 轨迹叠加（occ 是
    collision.from_scene 的结果，与碰撞判定同一份数据）。3DGS 里没有机体模型，跟随格在机体投影处
    画圈。每个 trace 样本（10 Hz）一帧；决策仍只在窗口起点。返回 {"video", "frames", "fps", "keyframes"}。
    """
    import cv2
    trace = _read_jsonl(os.path.join(ep_dir, "trace.jsonl"))
    if not trace:
        raise RuntimeError(f"{ep_dir} 没有 trace.jsonl（只有 --dynamics cf_sitl 的采集集才有）")
    steps = {a["step"]: a for a in _read_jsonl(os.path.join(ep_dir, "actions.jsonl"))}
    n2w = scene.get("_nav_to_world")
    out_path = out_path or os.path.join(ep_dir, "views.webm")
    png_dir = png_dir or ep_dir
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    w, h = 3 * PANEL_W, PANEL_H + STATUS_H
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"VP80"), float(VIEW_FPS), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建 {out_path}（fourcc=VP80）")
    key_idx = sorted({int(round(k * (len(trace) - 1) / max(1, KEYFRAMES - 1))) for k in range(KEYFRAMES)})
    keyframes = []
    pitch_follow = -math.degrees(math.atan2(FOLLOW_UP_M, FOLLOW_BACK_M))
    topmap, path_xy = _TopMap(scene, occ, extra_xy=[row["pos"][:2] for row in trace]), []
    try:
        for i, row in enumerate(trace):
            x, y, z = (float(v) for v in row["pos"])
            w_, qx, qy, qz = (float(v) for v in row["quat"])
            yaw = math.atan2(2 * (w_ * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            roll = math.degrees(math.atan2(2 * (w_ * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)))
            pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w_ * qy - qz * qx)))))
            path_xy.append((x, y))
            onboard = _panel(renderer, view=scene_mod.viewmat_from_quat((x, y, z), row["quat"], mount, n2w))
            follow_pose = (x - FOLLOW_BACK_M * math.cos(yaw), y - FOLLOW_BACK_M * math.sin(yaw),
                           z + FOLLOW_UP_M, math.degrees(yaw), pitch_follow)
            follow = _panel(renderer, pose=follow_pose)
            top = topmap.draw(path_xy, yaw)
            # 机体线框用跟随相机同一 view 矩阵投影（以前是固定画在格子中心的 6 px 圆，看不出尺寸和姿态）
            _draw_body(follow, (x, y, z), row["quat"], scene_mod.viewmat_from_pose(*follow_pose, n2w),
                       renderer.fov_deg, n2w)
            for img, name in ((onboard, "ONBOARD %dx%d HFOV %gdeg" % (PANEL_W, PANEL_H, renderer.fov_deg)),
                              (follow, "FOLLOW  back %.1fm up %.1fm  (E1 wireframe 0.17m)" % (
                                  FOLLOW_BACK_M, FOLLOW_UP_M)),
                              (top, "TOP MAP  occupancy proxy + path")):
                _label(img, name, (6, 16), scale=0.45)
            frame = np.zeros((h, w, 3), np.uint8)
            frame[:PANEL_H] = np.concatenate([onboard, follow, top], axis=1)
            a = steps.get(int(row["step"]), {})
            act = a.get("action_executed", row.get("action", -1))
            speed = math.sqrt(sum(float(v) ** 2 for v in row["vel"]))
            text = ("t=%5.1fs step %2d  act=%-9s  z=%.2fm  v=%.2fm/s  yaw=%6.1f roll=%5.1f pitch=%5.1f deg"
                    % (row["t"], row["step"], ACTION_NAMES[act] if 0 <= int(act) < 4 else "HOVER",
                       z, speed, math.degrees(yaw), roll, pitch))
            _label(frame, text, (6, PANEL_H + 24), scale=0.5)
            if a.get("collision"):
                _label(frame, "COLLISION", (w - 130, PANEL_H + 24), color=(0, 0, 255), scale=0.6)
            writer.write(frame)
            if i in key_idx:
                path = os.path.join(png_dir, "views_key_%02d.png" % len(keyframes))
                cv2.imwrite(path, frame)
                keyframes.append(path)
    finally:
        writer.release()
    return {"video": out_path, "frames": len(trace), "fps": VIEW_FPS, "keyframes": keyframes}


def _selftest():
    import tempfile

    import cv2

    with tempfile.TemporaryDirectory() as d:
        for i in range(3):
            img = np.zeros((24, 32, 3), np.uint8)
            img[:, :, 0] = i * 80
            cv2.imwrite(os.path.join(d, f"rgb_{i:05d}.png"), img)
        # 尺寸不同的帧被跳过：返回值必须是实际写入数，不是文件数
        cv2.imwrite(os.path.join(d, "rgb_00003.png"), np.zeros((12, 16, 3), np.uint8))
        out = os.path.join(d, "v.webm")
        n = compose_rgb_video(d, out, fps=2.0)
        cap = cv2.VideoCapture(out)
        frames = 0
        while cap.read()[0]:
            frames += 1
        cap.release()
        assert n == 3 and frames == 3, (n, frames)
        # 三视角：mock 渲染器 + 2 步 × 5 个 trace 样本 -> 10 帧、4 张关键帧
        MockRenderer = importlib.import_module("sim.3dgs.renderer").MockRenderer
        ep = os.path.join(d, "ep")
        os.makedirs(ep)
        with open(os.path.join(ep, "trace.jsonl"), "w") as f:
            for k in range(10):
                f.write(json.dumps({"step": k // 5, "t": 0.1 * (k + 1), "pos": [0.1 * k, 0, 2.0],
                                    "quat": [1, 0, 0, 0], "vel": [0.3, 0, 0], "action": 1,
                                    "exec_state": 3}) + "\n")
        with open(os.path.join(ep, "actions.jsonl"), "w") as f:
            f.write(json.dumps({"step": 0, "action_executed": 1, "collision": False}) + "\n")
            f.write(json.dumps({"step": 1, "action_executed": 2, "collision": True}) + "\n")
        scene = {"_nav_to_world": np.eye(4), "camera": {"width": 320, "height": 240, "fov_deg": 60},
                 "spawn": {"x": 0.0, "y": 0.0, "z": 2.0}, "waypoint": {"x": 8.0, "y": 0.0, "z": 2.0},
                 "reach_radius_m": 1.5}
        Occupancy = importlib.import_module("sim.3dgs.collision").Occupancy
        occ = Occupancy(np.array([[4.0, 1.0, 2.0], [4.0, -1.0, 2.0]]), np.tile([0.1, 0.3, 0.3], (2, 1)),
                        None, np.array([0.9, 0.9]), z_range=(1.65, 2.35), voxel_m=0.1)
        r = compose_views_video(ep, scene, MockRenderer(scene), {"xyz": [0.045, 0, 0.015], "pitch_deg": 0}, occ)
        cap = cv2.VideoCapture(r["video"])
        back = 0
        while cap.read()[0]:
            back += 1
        cap.release()
        assert (r["frames"], back, len(r["keyframes"])) == (10, 10, 4), (r, back)
        key0 = cv2.imread(r["keyframes"][0])
        assert key0.shape == (PANEL_H + STATUS_H, 3 * PANEL_W, 3)
        # 跟随格机体线框：水平机体、相机正后方 2.5 m 时，保护罩方框 0.168 m -> 约 19 px，落在格子中心附近
        follow = key0[:PANEL_H, PANEL_W:2 * PANEL_W]
        ys, xs = np.nonzero((follow[:, :, 0] < 40) & (follow[:, :, 1] > 215) & (follow[:, :, 2] > 215))
        assert len(xs) >= 20, ("跟随格没画出机体线框", len(xs))
        assert (abs(xs - PANEL_W / 2).max() <= 24 and abs(ys - PANEL_H / 2).max() <= 24), (
            "线框应在格子中心、尺寸约 19 px", xs.min(), xs.max(), ys.min(), ys.max())
        # 机体在相机后方时不画（proj 返回 None 不崩）
        blank = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
        _draw_body(blank, (-5.0, 0.0, 2.0), [1, 0, 0, 0],
                   scene_mod.viewmat_from_pose(0.0, 0.0, 2.0, 0.0, 0.0), 60.0)
        assert not blank.any(), "相机后方的机体不该画"
        print(json.dumps({"3dgs_video": "PASS", "frames": frames, "files": 4, "views_frames": back,
                          "body_px": int(len(xs))}))


def main():
    import argparse
    renderer_mod = importlib.import_module("sim.3dgs.renderer")
    p = argparse.ArgumentParser(description="cf_sitl 采集集 -> 三视角 .webm + 关键帧 PNG")
    p.add_argument("--scene", required=True)
    p.add_argument("--ep", nargs="+", required=True, help="episode 目录（含 trace.jsonl）")
    p.add_argument("--mock-renderer", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    scene = scene_mod.load_scene(args.scene)
    renderer = renderer_mod.make_renderer(scene, args.mock_renderer, device=args.device)
    occ = importlib.import_module("sim.3dgs.collision").from_scene(scene)   # 顶视地图 = 碰撞用的同一份占据
    for ep in args.ep:
        with open(os.path.join(ep, "episode_manifest.json"), encoding="utf-8") as f:
            man = json.load(f)
        mount = man["dynamics_params"].get("camera_mount")
        # 顶视的 spawn/waypoint 取本集 manifest 里的（--route 选的路线），不是 scene.json 顶层那条。
        ep_scene = dict(scene, **{k: man["scene"][k] for k in ("spawn", "waypoint") if man.get("scene", {}).get(k)})
        r = compose_views_video(ep, ep_scene, renderer, mount, occ)
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        _selftest()
