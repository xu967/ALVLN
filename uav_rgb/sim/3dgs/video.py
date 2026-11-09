#!/usr/bin/env python3
"""3DGS 回合帧 -> 视频：compose_rgb_video 只拼 onboard RGB；compose_views_video 给 cf_sitl
采集集（trace.jsonl）渲染 跟随 | 顶视 / 机载 | 状态面板 的 1280×720 四格画面，版式与 Gazebo
录像（sim/tools/record_views.py）同一份：sim/tools/view_layout.py。不依赖 Gazebo。

三视角 CLI（从仓库根，3dgs-venv）：
  python -m sim.3dgs.video --scene <scene.json> --ep <ep_dir> [<ep_dir> ...]   # 每集写 <ep_dir>/views.mp4 + views_key_*.png
  python -m sim.3dgs.video --scene ... --ep ... --out-name views_v2.mp4 --video-quality high   # 重合成旧集：不覆盖采集时的产物
画质档（--video-quality / 环境变量 UAVRGB_VIDEO_QUALITY）见 sim/tools/view_layout.VIDEO_QUALITY；.mp4 由 ffmpeg libx264
编码（imageio-ffmpeg 自带，见 tools/requirements.txt），没有 ffmpeg 回退 cv2 mp4v 并打日志；.webm 仍走 cv2 VP8。
"""
import glob
import hashlib
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
_TOOLS = os.path.join(_ROOT, "sim", "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
scene_mod = importlib.import_module("sim.3dgs.scene")
import e1_body  # noqa: E402 - E1 机体部件表 + 着色投影 / 顶视图标，与 Gazebo 顶视同一份
import view_layout as vl  # noqa: E402 - 与 record_views 共用的 1280×720 版式

# ---------------------------------------------------------------- 三视角视频配置（改这里）
VIEW_FPS = 10.0        # 与 CfSitlDynamics.trace_dt=0.1 s 对应：每个 trace 样本一帧，播放与仿真时间 1:1
FOLLOW_BACK_M = 1.1    # 跟随机位：机体后方 1.1 m、上方 0.3 m，俯角 atan(0.3/1.1)=15°，机体在画面中心偏下。
FOLLOW_UP_M = 0.3      #   §6.1 要求原生 640 宽跟随格里机体 ≥ 80 px：60° HFOV 下 f = 320/tan30° ≈ 554 px，
#                          0.172 m 宽的机体在 1.5 m 处只有 63 px（brief 建议值不够），1.1 m 处 ≈ 90 px（后保护罩斜距 1.06 m）。
#                          Callinstrasse 是行人高度拍的重建，机位高于约 2.6 m 就被天空壳/漂浮高斯糊住，所以只抬 0.3 m
TOP_MARGIN_M = 3.0     # 顶视用占据代理 2D 地图（3DGS 从上往下渲染全是漂浮物，4–20 m 高都试过）：
#                          地图窗口 = spawn–waypoint 包围盒各向外扩 3 m，到达半径 2 m 圈和两侧建筑都在图里
HOLD_END_S = 1.5       # 结尾定格秒数（写 SUCCESS/COLLISION 大字），与 record_views.compose 默认一致，短片里结果停得住
KEYFRAMES = 4          # 关键帧 PNG：首、1/3、2/3、末（末 = 定格帧，带结果字），预览用
# 跟随格里的机体：e1_body.PARTS（与 sim/worlds/uav_rgb_e1.sdf 的 19 件 visual 一致）按真实位姿着色投影，
# roll/pitch 可见；以前是黄色线框（更早是固定 6 px 圆）。顶视地图上的机体是 e1_body.draw_top_icon，与 Gazebo 顶视同一图标。
# 顶视地图配色：中灰底 + 深灰占据，白字黑边标签在上面读得出；轨迹黄与 Gazebo 顶视 trail 同色。
MAP_BG, MAP_GRID, MAP_OCC = (150, 150, 150), (168, 168, 168), (55, 55, 55)
MAP_TRAIL, MAP_GOAL, MAP_SPAWN = (0, 255, 255), (0, 0, 220), (0, 160, 0)


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


def _draw_body(img, pos, quat, cam_view, fov_deg, n2w=None):
    """把 E1 机体（e1_body.PARTS 着色投影 + 红机头线）画到跟随格；返回像素包围盒或 None。

    pos/quat 是 nav 系机体位姿，cam_view 是跟随相机的 world->camera 4x4（含 nav_to_world），
    与渲染用的同一矩阵，所以机体与 3DGS 画面严格同一投影（内参按 img 实际尺寸算）。
    机体在相机后方/过近时不画（e1_body.draw_shaded 跳过 z<=0.05 的面）。
    """
    k = scene_mod.intrinsics(img.shape[1], img.shape[0], fov_deg)
    r_nb = scene_mod._quat_rotmat(quat)
    m = np.eye(4) if n2w is None else np.asarray(n2w, np.float64)
    body_to_nav = np.eye(4)
    body_to_nav[:3, :3], body_to_nav[:3, 3] = r_nb, np.asarray(pos, np.float64)
    import cv2
    return e1_body.draw_shaded(cv2, img, np.asarray(cam_view, np.float64) @ m @ body_to_nav, k)


def _render(renderer, width, height, view=None, pose=None, ssaa=None):
    """渲染一格（任意分辨率，renderer 按宽度算内参、HFOV 不变）并转 BGR。
    ssaa 临时覆盖 renderer.supersample（画质档的 3DGS 超采样）；MockRenderer 没有这个属性就忽略。"""
    import cv2
    old = getattr(renderer, "supersample", None)
    if ssaa is not None and old is not None:
        renderer.supersample = int(ssaa)
    try:
        rgb = (renderer.render_view(view, width, height) if view is not None
               else renderer.render_pose(*pose, width=width, height=height))[0]
    finally:
        if ssaa is not None and old is not None:
            renderer.supersample = old
    return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)


class _TopMap:
    """固定顶视：占据代理（collision.Occupancy.raw 沿 z 投影）的 2D 地图，640×360 一格。

    朝向与 Gazebo 顶视相机（record_views.project_top）一致：图右 = +x、图上 = +y，两域短片
    拼在一起时轨迹读法相同；路线大多沿 +x，也正好用满 16:9 的宽。
    """

    def __init__(self, scene, occ, extra_xy=(), s=1):
        """extra_xy：也要落在图里的点（整条轨迹）；多路线的 A* 绕行会跑出 spawn–waypoint 包围盒。
        s：画布倍数（high 档 2）：地图按 CW·s × CH·s 原生画，字号/线宽随之放大，不靠 paste 放大。"""
        import cv2
        font = cv2.FONT_HERSHEY_SIMPLEX
        self.s, self.w, self.h = s, vl.CW * s, vl.CH * s
        sp, wp = scene["spawn"], scene["waypoint"]
        pts = [(sp["x"], sp["y"]), (wp["x"], wp["y"])] + [(float(x), float(y)) for x, y in extra_xy]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        span_x, span_y = max(xs) - min(xs) + 2 * TOP_MARGIN_M, max(ys) - min(ys) + 2 * TOP_MARGIN_M
        self.scale = max(span_x / self.w, span_y / self.h)           # m/px，x 映射到图宽
        self.cx, self.cy = 0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys))
        self.base = np.full((self.h, self.w, 3), MAP_BG, np.uint8)
        # 1 m 格线：地图窗口随路线变（8–20 m），格线让图上任何位置都读得出距离，比例尺只标一处。
        half_w, half_h = 0.5 * self.w * self.scale, 0.5 * self.h * self.scale
        for gx in range(math.ceil(self.cx - half_w), math.floor(self.cx + half_w) + 1):
            u = self.px(gx, self.cy)[0]
            cv2.line(self.base, (u, 0), (u, self.h - 1), MAP_GRID, s)
        for gy in range(math.ceil(self.cy - half_h), math.floor(self.cy + half_h) + 1):
            v = self.px(self.cx, gy)[1]
            cv2.line(self.base, (0, v), (self.w - 1, v), MAP_GRID, s)
        if occ.raw.size:
            ii, jj = np.nonzero(occ.raw.any(axis=2))
            half = max(1, int(round(0.5 * occ.voxel / self.scale)))
            for i, j in zip(ii, jj):
                px, py = self.px(occ.origin[0] + (i + 0.5) * occ.voxel, occ.origin[1] + (j + 0.5) * occ.voxel)
                if -half <= px < self.w + half and -half <= py < self.h + half:
                    cv2.rectangle(self.base, (px - half, py - half), (px + half, py + half), MAP_OCC, -1)
        reach = float(scene.get("reach_radius_m", 1.5))
        cv2.circle(self.base, self.px(wp["x"], wp["y"]), int(round(reach / self.scale)), MAP_GOAL, 2 * s)   # 到达半径
        cv2.circle(self.base, self.px(wp["x"], wp["y"]), 5 * s, MAP_GOAL, -1)
        cv2.circle(self.base, self.px(sp["x"], sp["y"]), 5 * s, MAP_SPAWN, -1)                              # spawn
        # 图例放左上、TOP 标签（paste 写在 y=22）之下，与 Gazebo 顶视图例同位置（record_views.LEGEND_TOP）。
        legend = (("trail", MAP_TRAIL), ("waypoint r=%gm" % reach, MAP_GOAL), ("spawn", MAP_SPAWN),
                  ("occupancy proxy", MAP_OCC))
        for i, (label, color) in enumerate(legend):
            cv2.putText(self.base, label, (8 * s, (44 + 15 * i) * s), font, 0.4 * s, color, s)
        # 比例尺 + 朝向：没有比例尺看不出 2 m 到达圈有多大；箭头朝右 = +x。放在 paste 的时间角标（y=CH-8）之上。
        bar = int(round(1.0 / self.scale))
        x0, y0 = 8 * s, self.h - 36 * s
        cv2.line(self.base, (x0, y0), (x0 + bar, y0), (0, 0, 0), 2 * s)
        cv2.putText(self.base, "1 m", (x0 + bar + 4 * s, y0 + 4 * s), font, 0.4 * s, (0, 0, 0), s, cv2.LINE_AA)
        cv2.arrowedLine(self.base, (self.w - 40 * s, self.h - 14 * s), (self.w - 12 * s, self.h - 14 * s),
                        (0, 0, 0), s, tipLength=0.3)
        cv2.putText(self.base, "+x", (self.w - 62 * s, self.h - 10 * s), font, 0.4 * s, (0, 0, 0), s, cv2.LINE_AA)

    def px(self, x, y):
        return (int(round(self.w / 2 + (x - self.cx) / self.scale)),
                int(round(self.h / 2 - (y - self.cy) / self.scale)))

    def draw(self, path_xy, yaw):
        """轨迹 + 当前位置的四旋翼图标（e1_body.draw_top_icon，按比例、最小 18·s px；图右 = +x、图上 = +y）。"""
        import cv2
        img = self.base.copy()
        pts = np.array([self.px(x, y) for x, y in path_xy], np.int32)
        if len(pts) >= 2:
            cv2.polylines(img, [pts], False, MAP_TRAIL, 2 * self.s)
        e1_body.draw_top_icon(cv2, img, self.px(*path_xy[-1]), yaw, 1.0 / self.scale, min_px=18 * self.s)
        return img


def compose_views_video(ep_dir, scene, renderer, mount, occ, out_path=None, png_dir=None,
                        title="", result="", quality=None, log=print):
    """cf_sitl 采集集（trace.jsonl + actions.jsonl）-> 四格视频（.mp4 / .webm）+ 关键帧 PNG。

    左上 FOLLOW：同一 3DGS，只跟位置与偏航，机体后方 FOLLOW_BACK_M、上方 FOLLOW_UP_M，格子原生分辨率
    渲染，叠 E1 机体着色投影；右上 TOP：占据代理 2D 地图 + 轨迹 + 机体图标（occ 是 collision.from_scene 的
    结果，与碰撞判定同一份数据）；左下 ONBOARD：机体四元数 ⊗ camera_mount（roll/pitch 进画面），按 scene
    相机分辨率（320×240）渲染后由 paste 放大，模型输入不变；右下：状态面板（title、路线/步数、时间、位置、
    速度/姿态、当前动作、动作时间线，结尾定格 HOLD_END_S 写 result）。
    quality：view_layout.VIDEO_QUALITY 档名（None = 默认 / 环境变量）：high 档画布 2560×1440，FOLLOW / TOP 按
    2× 原生渲染，ONBOARD 为显示另渲一张 2× 帧（标题注明 display 2x），3DGS 超采样取档位的 ssaa，.mp4 走
    ffmpeg libx264 CRF。
    每个 trace 样本（10 Hz）一帧；决策仍只在窗口起点。关键帧 PNG 名字跟 out_path 的主干走
    （views.mp4 -> views_key_00.png，views_v2.mp4 -> views_v2_key_00.png），重合成不盖旧图。
    返回 {"video", "frames", "fps", "keyframes", "quality", "backend", "body_px"}（body_px：FOLLOW 格里机体
    像素宽的中位数，§6.1 要求原生 640 宽下 ≥ 80）。
    """
    import cv2
    q = vl.quality(quality)
    s, ssaa = q["scale"], q["ssaa"]
    W, H, CW, CH, tl, tr, bl, br = vl.grid(s)
    trace = _read_jsonl(os.path.join(ep_dir, "trace.jsonl"))
    if not trace:
        raise RuntimeError(f"{ep_dir} 没有 trace.jsonl（只有 --dynamics cf_sitl 的采集集才有）")
    actions = _read_jsonl(os.path.join(ep_dir, "actions.jsonl"))
    steps = {a["step"]: a for a in actions}
    n2w = scene.get("_nav_to_world")
    out_path = out_path or os.path.join(ep_dir, "views.mp4")
    png_dir = png_dir or ep_dir
    stem = os.path.splitext(os.path.basename(out_path))[0]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    writer = vl.VideoSink(cv2, out_path, VIEW_FPS, (W, H), q["crf"], log=log)
    if not writer.isOpened():
        raise RuntimeError(f"无法创建 {out_path}（{writer.backend}）")
    seq = trace + [trace[-1]] * int(HOLD_END_S * VIEW_FPS)        # 末尾定格：画面不变，只多 result 字
    key_idx = sorted({int(round(k * (len(seq) - 1) / max(1, KEYFRAMES - 1))) for k in range(KEYFRAMES)})
    keyframes, body_px = [], []
    pitch_follow = -math.degrees(math.atan2(FOLLOW_UP_M, FOLLOW_BACK_M))
    topmap, path_xy = _TopMap(scene, occ, extra_xy=[row["pos"][:2] for row in trace], s=s), []
    route = scene.get("route") or {}
    route_tag = f"route {route['index']} {route['name']}  " if route else ""
    t0, t_end = float(trace[0]["t"]), float(trace[-1]["t"])
    span = max(t_end - t0, 1e-3)
    # 动作时间条：每步一段，起点 = 该步第一个 trace 样本的 t，终点 = 下一步起点 / t_end。
    starts = {}
    for row in trace:
        starts.setdefault(int(row["step"]), float(row["t"]))
    order = sorted(starts)
    segments = [(starts[s] - t0, (starts[order[i + 1]] if i + 1 < len(order) else t_end) - t0,
                 int(steps.get(s, {}).get("action_executed", 0)))
                for i, s in enumerate(order)]
    # 3DGS 的 collision 是占据代理沿位移扫掠的判定，没有 Gazebo 那种 contact / stuck 之分，只有一种标记。
    hit = next((a for a in actions if a.get("collision")), None)
    follow_label = "FOLLOW back %.1fm up %.1fm (E1 body %.2fm)" % (FOLLOW_BACK_M, FOLLOW_UP_M, e1_body.WIDTH_M)
    cells = None
    try:
        for i, row in enumerate(seq):
            x, y, z = (float(v) for v in row["pos"])
            w_, qx, qy, qz = (float(v) for v in row["quat"])
            yaw = math.atan2(2 * (w_ * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            roll = math.degrees(math.atan2(2 * (w_ * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)))
            pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w_ * qy - qz * qx)))))
            if i < len(trace):
                path_xy.append((x, y))
                # 机载：模型输入是 renderer.width×height（320×240）；high 档为显示另渲 s 倍帧，标题注明 display 2x
                onboard = _render(renderer, renderer.width * s, renderer.height * s, ssaa=ssaa,
                                  view=scene_mod.viewmat_from_quat((x, y, z), row["quat"], mount, n2w))
                follow_pose = (x - FOLLOW_BACK_M * math.cos(yaw), y - FOLLOW_BACK_M * math.sin(yaw),
                               z + FOLLOW_UP_M, math.degrees(yaw), pitch_follow)
                follow = _render(renderer, CW, CH, pose=follow_pose, ssaa=ssaa)
                # 机体用跟随相机同一 view 矩阵投影（以前是黄线框，更早是固定画在格子中心的 6 px 圆）
                box = _draw_body(follow, (x, y, z), row["quat"], scene_mod.viewmat_from_pose(*follow_pose, n2w),
                                 renderer.fov_deg, n2w)
                if box is not None:
                    body_px.append((box[2] - box[0]) / s)      # 折算到原生 640 宽
                cells = (onboard, follow, topmap.draw(path_xy, yaw))
            onboard, follow, top = cells
            t = float(row["t"])
            note = f"sim {t:.2f}s"                 # 每帧按 trace 时刻现渲，没有 Gazebo 那种帧龄
            canvas = np.zeros((H, W, 3), np.uint8)
            vl.paste(cv2, canvas, follow, *tl, CW, CH, follow_label, note, s=s)
            vl.paste(cv2, canvas, top, *tr, CW, CH, "TOP MAP occupancy proxy + path", note, s=s)
            vl.paste(cv2, canvas, onboard, *bl, CW, CH,
                     vl.onboard_label(onboard, renderer.fov_deg, display=s, rendered_at=s), note, s=s)
            step = int(row["step"])
            act = steps.get(step, {}).get("action_executed", row.get("action"))
            speed = math.sqrt(sum(float(v) ** 2 for v in row["vel"]))
            lines = [title or str(scene.get("name", "")),
                     f"{route_tag}step {step + 1}/{len(actions)}",
                     f"sim t = {t:.2f} s  (video +{t - t0:.1f} s)",
                     f"pos x={x:5.2f} y={y:5.2f} z={z:4.2f} m",
                     f"speed {speed:4.2f} m/s  yaw {math.degrees(yaw):6.1f} roll {roll:5.1f} pitch {pitch:5.1f} deg"]
            flags = [(f"COLLISION @ step {hit['step'] + 1}", (0, 0, 255))] if hit and step >= hit["step"] else []
            vl.panel(cv2, canvas, *br, lines,
                     int(act) if act is not None and 0 <= int(act) < 4 else None,
                     flags, segments, t - t0, span, result if i >= len(trace) else "", s=s)
            writer.write(canvas)
            if i in key_idx:
                path = os.path.join(png_dir, "%s_key_%02d.png" % (stem, len(keyframes)))
                cv2.imwrite(path, canvas)
                keyframes.append(path)
    finally:
        backend = writer.release()
    return {"video": out_path, "frames": len(seq), "fps": VIEW_FPS, "keyframes": keyframes,
            "quality": q["name"], "backend": backend,
            "body_px": (round(float(np.median(body_px)), 1) if body_px else None)}


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
        # 三视角：mock 渲染器 + 2 步 × 5 个 trace 样本 -> 10 帧 + 定格 15 帧、4 张关键帧
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
                 "reach_radius_m": 1.5, "route": {"index": 0, "name": "r0_test"}}
        Occupancy = importlib.import_module("sim.3dgs.collision").Occupancy
        occ = Occupancy(np.array([[4.0, 1.0, 2.0], [4.0, -1.0, 2.0]]), np.tile([0.1, 0.3, 0.3], (2, 1)),
                        None, np.array([0.9, 0.9]), z_range=(1.65, 2.35), voxel_m=0.1)
        logs = []
        r = compose_views_video(ep, scene, MockRenderer(scene), {"xyz": [0.045, 0, 0.015], "pitch_deg": 0}, occ,
                                out_path=os.path.join(ep, "views_1280.webm"), title="selftest", result="COLLISION",
                                quality="medium", log=logs.append)
        cap = cv2.VideoCapture(r["video"])
        back = 0
        while cap.read()[0]:
            back += 1
        cap.release()
        n_hold = int(HOLD_END_S * VIEW_FPS)
        assert (r["frames"], back, len(r["keyframes"])) == (10 + n_hold, 10 + n_hold, 4), (r, back)
        assert [os.path.basename(k) for k in r["keyframes"]] == ["views_1280_key_%02d.png" % k for k in range(4)], r
        assert r["quality"] == "medium" and r["backend"].startswith("cv2 VP80") and not logs, r
        key0 = cv2.imread(r["keyframes"][0])
        assert key0.shape == (vl.H, vl.W, 3), key0.shape
        # 与 Gazebo 同一版式：机载 320x240 放大到 480x360 居中在左下格（两侧 80 px 黑边），右下是面板（黑底白字）
        onboard = key0[vl.CH:, :vl.CW]
        mid = onboard[40:-24]            # 去掉标签行（顶部）和 sim 时间角标（底部）
        assert (mid[:, 80:560].any(axis=2).all() and not mid[:, :80].any()
                and not mid[:, 560:].any()), "机载格应居中 480x360"
        panel_bgr = key0[vl.CH:, vl.CW:]
        assert (panel_bgr == 235).all(axis=2).any(), "右下应是状态面板文字（白 235）"
        # 顶视：图右 = +x。spawn (0,0) 在 waypoint (8,0) 左侧；绿点列号 < 红点列号；机体图标（白桨盘 200 灰）在轨迹末端
        topmap = _TopMap(scene, occ)
        assert topmap.px(0.0, 0.0)[0] < topmap.px(8.0, 0.0)[0] and topmap.px(0.0, 1.0)[1] < topmap.px(0.0, 0.0)[1]
        top = key0[:vl.CH, vl.CW:]
        ys, xs = np.nonzero((top == 200).all(axis=2))
        assert len(xs) >= 30 and abs(xs.mean() - topmap.px(0.0, 0.0)[0]) < 12, ("顶视图标应画在当前位置", len(xs))
        # 跟随格（左上 640x360）机体着色投影：水平机体、相机后方 1.1 m / 上 0.3 m，§6.1 要求机体 ≥ 80 px、
        # 画面水平居中；红机头线在机体上半（远端）
        assert r["body_px"] is not None and r["body_px"] >= 80, ("跟随格机体宽应 ≥ 80 px", r["body_px"])
        follow = key0[:vl.CH, :vl.CW]
        red_mask = (follow[:, :, 2] > 200) & (follow[:, :, 1] < 60) & (follow[:, :, 0] < 60)
        ys, xs = np.nonzero(red_mask)
        assert len(xs) >= 20 and abs(xs.mean() - vl.CW / 2) < 8, ("机头红线应在格子水平中央", len(xs))
        grey = (abs(follow[:, :, 0].astype(int) - follow[:, :, 2].astype(int)) < 12) & (follow[:, :, 1] > 120)
        gys, gxs = np.nonzero(grey[40:-24])          # 保护罩（浅灰着色）像素
        assert gxs.max() - gxs.min() >= 80, ("保护罩像素跨度应 ≥ 80", gxs.min(), gxs.max())
        # 末帧是定格帧：结果大字 COLLISION（红）出现在面板里；首帧没有
        last = cv2.imread(r["keyframes"][-1])[vl.CH:, vl.CW:]
        red = lambda im: int(((im[:, :, 2] == 255) & (im[:, :, 1] == 0) & (im[:, :, 0] == 0)).sum())   # noqa: E731
        assert red(last) > red(panel_bgr) + 200, ("定格帧应多出红色结果大字", red(last), red(panel_bgr))
        # 机体在相机后方时不画（draw_shaded 返回 None 不崩）
        blank = np.zeros((vl.CH, vl.CW, 3), np.uint8)
        assert _draw_body(blank, (-5.0, 0.0, 2.0), [1, 0, 0, 0],
                          scene_mod.viewmat_from_pose(0.0, 0.0, 2.0, 0.0, 0.0), 60.0) is None
        assert not blank.any(), "相机后方的机体不该画"
        # high 档：2560x1440 画布、机载标题注明 display 2x、.mp4 走 ffmpeg（有）或回退 cv2（无，打日志）
        logs = []
        r2 = compose_views_video(ep, scene, MockRenderer(scene), {"xyz": [0.045, 0, 0.015], "pitch_deg": 0}, occ,
                                 out_path=os.path.join(ep, "views_v2.mp4"), title="selftest", result="COLLISION",
                                 quality="high", log=logs.append)
        k2 = cv2.imread(r2["keyframes"][0])
        assert k2.shape == (2 * vl.H, 2 * vl.W, 3) and r2["quality"] == "high", (k2.shape, r2)
        assert r2["body_px"] is not None and abs(r2["body_px"] - r["body_px"]) <= 2, (r2["body_px"], r["body_px"])
        if vl.find_ffmpeg():
            assert r2["backend"].startswith("ffmpeg libx264 crf 18") and not logs, (r2, logs)
        else:
            assert "回退" in logs[0], logs
        cap = cv2.VideoCapture(r2["video"])
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 2 * vl.W and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10 + n_hold
        cap.release()
        print(json.dumps({"3dgs_video": "PASS", "frames": frames, "files": 4, "views_frames": back,
                          "body_px": r["body_px"], "high_backend": r2["backend"]}))


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def video_policy(man):
    """标题里的策略名 = 谁执行了动作（manifest executed_by：teacher / student / forward）。
    旧 manifest 没有 executed_by 时按 teacher 字段（= 策略名）推断；以前只要有 teacher 字段就写 teacher，
    学生 eval 集（teacher=student）也被标成 teacher（FIXLIST D7）。"""
    if man.get("executed_by"):
        return str(man["executed_by"])
    name = man.get("teacher") or man.get("policy") or "student"
    return name if name in ("student", "forward") else "teacher"


def episode_result(man, oracle=None, reach_radius_m=None, collided=False):
    """结尾大字：与 Gazebo eval_closed_loop._write_video 同一套词（SUCCESS / COLLISION / …）。

    没有 goal-stop 的集（深度教师 / forward / 学生 eval）给 oracle 与 reach_radius_m 时按 eval_3dgs 同口径：
    轨迹任一点到 waypoint 的欧氏距离 <= reach_radius_m 就是 SUCCESS。碰撞（即停 / 截断 / 任一行 collision）优先于到达：
    撞后穿模飞到目标不算到。
    """
    if man.get("stopped_on_goal"):
        return "SUCCESS reached"
    if man.get("stopped_on_collision") or man.get("truncated_on_collision") or collided:
        return "COLLISION"
    if man.get("dynamics_anomaly"):
        return "ANOMALY %s" % (man["dynamics_anomaly"].get("kind") if isinstance(man["dynamics_anomaly"], dict)
                               else man["dynamics_anomaly"])
    wp = man.get("waypoint") or {}
    if oracle and reach_radius_m is not None and "x" in wp and "y" in wp:
        dmin = min(math.hypot(float(o["pose_gt"]["x"]) - float(wp["x"]), float(o["pose_gt"]["y"]) - float(wp["y"]))
                   for o in oracle if isinstance(o.get("pose_gt"), dict))
        if dmin <= float(reach_radius_m):
            return "SUCCESS reached (min %.2f m)" % dmin
    return "NOT REACHED (%s steps)" % man.get("num_steps")


def _read_jsonl(path):
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    import argparse
    renderer_mod = importlib.import_module("sim.3dgs.renderer")
    p = argparse.ArgumentParser(description="cf_sitl 采集集 -> 三视角四格视频（.mp4 libx264 / .webm VP8）+ 关键帧 PNG")
    p.add_argument("--scene", required=True)
    p.add_argument("--ep", nargs="+", required=True, help="episode 目录（含 trace.jsonl）")
    p.add_argument("--out-name", default="views.mp4",
                   help="每集输出文件名（写在 ep 目录；关键帧 <主干>_key_NN.png）。重合成旧集用别的名字（如 views_v2.mp4），不覆盖采集时的产物")
    p.add_argument("--video-quality", default=None, choices=sorted(vl.VIDEO_QUALITY),
                   help=f"画质档（sim/tools/view_layout.VIDEO_QUALITY；默认 {vl.DEFAULT_QUALITY}，环境变量 {vl.QUALITY_ENV} 覆盖）")
    p.add_argument("--allow-scene-mismatch", action="store_true",
                   help="manifest 的 scene_json_sha256 与 --scene 不一致时仍渲染（scene 的 nav_to_world/FOV 变过，画面会对不上采集时的轨迹）")
    p.add_argument("--mock-renderer", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    scene = scene_mod.load_scene(args.scene)
    scene_sha = _sha256(args.scene)
    renderer = renderer_mod.make_renderer(scene, args.mock_renderer, device=args.device)
    occ = importlib.import_module("sim.3dgs.collision").from_scene(scene)   # 顶视地图 = 碰撞用的同一份占据
    scene_short = os.path.basename(scene["dir"])
    for ep in args.ep:
        with open(os.path.join(ep, "episode_manifest.json"), encoding="utf-8") as f:
            man = json.load(f)
        want = (man.get("scene") or {}).get("scene_json_sha256")
        if want and want != scene_sha and not args.allow_scene_mismatch:
            sys.exit(f"{ep}: 采集时 scene.json sha256 {want[:12]} != 当前 {scene_sha[:12]}；"
                     f"换回采集时的 scene.json，或确认过差异无关后加 --allow-scene-mismatch")
        mount = man["dynamics_params"].get("camera_mount")
        # 顶视的 spawn/waypoint/路线名取本集 manifest 里的（--route 选的路线），不是 scene.json 顶层那条。
        ep_scene = dict(scene, **{k: man["scene"][k] for k in ("spawn", "waypoint") if man.get("scene", {}).get(k)})
        if isinstance(man.get("route"), dict):
            ep_scene["route"] = man["route"]
        dyn = man.get("dynamics_backend") or man["dynamics_params"].get("dynamics", "?")
        oracle = _read_jsonl(os.path.join(ep, "oracle.jsonl"))
        collided = any(a.get("collision") for a in _read_jsonl(os.path.join(ep, "actions.jsonl")))
        result = episode_result(man, oracle, float(ep_scene.get("reach_radius_m", 1.5)), collided)
        r = compose_views_video(ep, ep_scene, renderer, mount, occ, out_path=os.path.join(ep, args.out_name),
                                title=f"{video_policy(man)} | {dyn} | {scene_short} | seed {man.get('seed')}",   # 与 Gazebo 标题同格式
                                result=result, quality=args.video_quality,
                                log=lambda m: print(m, file=sys.stderr, flush=True))
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        _selftest()
