#!/usr/bin/env python3
"""三视图任务视频：录 机载/追尾/顶视 三路相机，再按仿真时间合成一段视频。

  record   在 sim 运行期间常驻订阅三路相机，按仿真时间 ts//(1/FPS) 分桶、每桶留一帧，原始 RGB 写盘。
           SIGTERM 结束（eval_closed_loop 用 stop_group 收）。纯 stdlib，任意解释器可跑。
  compose  读帧 + actions/oracle.jsonl，按同一分桶网格合成 1280x720 视频：
           追尾 | 顶视（叠加轨迹、碰撞点、终点线） / 机载 | 状态面板（动作、速度、碰撞）。
           网格、贴图、面板的版式在 view_layout.py，与 3DGS 的 sim/3dgs/video.py 共用同一份。
           顶视当前位置画按比例的 E1 四旋翼图标（e1_body.draw_top_icon，机头朝向取 actions 的 yaw_before）。
           --out 以 .mp4 结尾走 view_layout.VideoSink：ffmpeg libx264、CRF 按 --video-quality 档（没 ffmpeg 回退
           cv2 mp4v 并打日志）；.webm 用 cv2 VP8（浏览器可直接播放）。需要 cv2+numpy（.venv-tools / 3dgs-venv）。
  patch_chase_cam  eval_closed_loop --record-video 时对本集生成的 world 打追尾相机补丁（CHASE_PATCH），模板不改。

用法：
  python3 record_views.py record --out FRAMES_DIR            # 需在 sim 的 GZ_PARTITION 环境下
  python3 record_views.py compose --frames FRAMES_DIR --episode EP_DIR \\
      --world scene_42.sdf --out seed_42.webm [--title "..."] [--result SUCCESS]
"""
import argparse
import json
import math
import os
import re
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_body
import gz_io
import save_frame
import view_layout as vl
from view_layout import onboard_label  # noqa: F401 - 机载标题文案归版式模块，这里保留名字给 compose 与测试用

CAMS = {"uav": "/uav/camera", "chase": "/uav/chase_camera", "top": "/uav/top_camera"}
FPS = 10                      # 视频帧率 = 每仿真秒 10 帧
STALE_AFTER_S = 1.0           # 帧龄超过此值标 STALE（相机 10–30 Hz，1 s 已是停更）
LEGEND_TOP = 44               # 顶视图例首行 y：TOP 标签（y=22）之下、criterion 文字（右上）之外


def _src(step):
    return step.get("src") or ""


def contact_step(actions):
    """第一个真实接触步；stuck 不是碰撞，不能拿来标 COLLISION。"""
    return next((r for r in actions if r.get("collision") and "contact" in _src(r)), None)


def stuck_step(actions):
    """第一个卡住步；contact+stuck 同时存在时两类都会显示。"""
    return next((r for r in actions if r.get("collision") and "stuck" in _src(r)), None)


def _frame_note(ts, idx, old_idx, t):
    """面板角标：该帧 sim time / 帧龄；重复帧 REPEAT，停更帧 STALE。只用 ASCII（Hershey 字体画不了「·」和中文）。"""
    if ts is None:
        return "", None
    age = max(0.0, (t - ts) / 1e9)
    if age > STALE_AFTER_S:
        tag, color = " | STALE", (0, 0, 255)
    elif idx == old_idx:
        tag, color = " | REPEAT", (0, 165, 255)
    else:
        tag, color = "", (235, 235, 235)
    return f"sim {ts / 1e9:.2f}s | age {age:.2f}s{tag}", color


fourcc_for = vl.fourcc_for   # cv2 回退编码的 fourcc 选择归版式模块；名字保留给旧调用与测试

# 录像时对生成场景 SDF 打的追尾相机补丁（只动 chase_cam，模板文件本身不改）。
# 机位：机体后 1.0 m、上 0.45 m（模板是后 1.0 上 0.5），HFOV 80° -> 60°：
# 60° 下 f = 320/tan30° ≈ 554 px，0.172 m 宽的机体在 1.0 m 处约 95 px（80° 时 ≈ 55 px，看不清机体）。
# chase_cam 的位姿由 CfSitlSystem 插件按 <chase_offset>/<chase_pitch> 每步重设，所以改的是插件参数，
# 不是 chase_cam 模型的 <pose>；分辨率按画质档的显示倍率放大，让 2× 画布上追尾视角也是原生渲染。
CHASE_PATCH = {"offset": "-1.0 0 0.45", "pitch": "0.45", "hfov": "1.0471975512", "size": (640, 360)}


def patch_chase_cam(world_path, quality=None):
    """就地改写 world_path 里的追尾相机参数，返回补丁描述（写进 eval.json 的 video 字段）。

    只替换 <plugin ... CfSitlSystem> 的 chase_offset/chase_pitch 和 <sensor name="chase_camera"> 的
    horizontal_fov/width/height 这 5 个值；文件其余字节不动。模板里没有这些节点（旧 x3 模板）则不改、返回 None。
    """
    import hashlib
    q = vl.quality(quality)
    s = q["scale"]
    with open(world_path, encoding="utf-8") as f:
        text = f.read()
    m = re.search(r'<sensor name="chase_camera" type="camera">.*?</sensor>', text, re.S)
    if not m or "<chase_offset>" not in text:
        return None
    cam = m.group(0)
    cam = re.sub(r"<horizontal_fov>[^<]*</horizontal_fov>", f"<horizontal_fov>{CHASE_PATCH['hfov']}</horizontal_fov>", cam, 1)
    cam = re.sub(r"<width>\d+</width>", f"<width>{CHASE_PATCH['size'][0] * s}</width>", cam, 1)
    cam = re.sub(r"<height>\d+</height>", f"<height>{CHASE_PATCH['size'][1] * s}</height>", cam, 1)
    text = text[:m.start()] + cam + text[m.end():]
    text = re.sub(r"<chase_offset>[^<]*</chase_offset>", f"<chase_offset>{CHASE_PATCH['offset']}</chase_offset>", text, 1)
    text = re.sub(r"<chase_pitch>[^<]*</chase_pitch>", f"<chase_pitch>{CHASE_PATCH['pitch']}</chase_pitch>", text, 1)
    with open(world_path, "w", encoding="utf-8") as f:
        f.write(text)
    return {"chase_offset": CHASE_PATCH["offset"], "chase_pitch": float(CHASE_PATCH["pitch"]),
            "chase_hfov_deg": round(math.degrees(float(CHASE_PATCH["hfov"])), 1),
            "chase_size": [CHASE_PATCH["size"][0] * s, CHASE_PATCH["size"][1] * s],
            "quality": q["name"], "world_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def record(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    subs = {k: gz_io.TopicSub(t, buf=8) for k, t in CAMS.items()}
    # "wb"/"w"：目录里残留上一次的帧时不能追加，否则两次录像的帧混在一起。
    files = {k: (open(os.path.join(out_dir, k + ".rgb"), "wb"),
                 open(os.path.join(out_dir, k + ".idx.jsonl"), "w")) for k in CAMS}
    last_seq = dict.fromkeys(CAMS, 0)
    last_bucket = dict.fromkeys(CAMS, None)
    period = int(1e9 / FPS)
    try:
        while True:
            for k, sub in subs.items():
                for seq, msg in sub.frames():
                    if seq <= last_seq[k]:
                        continue
                    last_seq[k] = seq
                    ts = gz_io.header_stamp_ns(msg)
                    # 按 ts // period 分桶、每桶留一帧，与 compose 的取帧网格对齐。以前用
                    # “距上一帧 >= period”，15 Hz 的 chase 实际只留 7.5 Hz，合成时约每 4 帧重复 1 帧。
                    bucket = ts // period
                    if bucket == last_bucket[k]:
                        continue
                    try:
                        w, h, nch, rows = save_frame.decode(msg)
                    except ValueError:
                        continue
                    data, idx = files[k]
                    idx.write(json.dumps({"ts": ts, "w": w, "h": h, "c": nch,
                                          "off": data.tell()}) + "\n")
                    data.write(rows)
                    data.flush()
                    idx.flush()
                    last_bucket[k] = bucket
            time.sleep(0.01)
    finally:
        for data, idx in files.values():
            data.close()
            idx.close()
        for sub in subs.values():
            sub.close()


def load_frames(frames_dir, cam):
    """-> [(ts, w, h, c, off)]，按 ts 升序；文件缺失返回 []。"""
    path = os.path.join(frames_dir, cam + ".idx.jsonl")
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                rows.append((r["ts"], r["w"], r["h"], r["c"], r["off"]))
            except (ValueError, KeyError):
                continue          # SIGTERM 可能截断最后一行
    return sorted(rows)


def top_camera_pose(world_text):
    """从 world SDF 取 studio_top_camera 的 pose/HFOV/分辨率；找不到返回 None。"""
    m = re.search(r'<model name="studio_top_camera">.*?<pose>([^<]+)</pose>.*?'
                  r'<horizontal_fov>([^<]+)</horizontal_fov>.*?<width>(\d+)</width>'
                  r'.*?<height>(\d+)</height>', world_text, re.S)
    if not m:
        return None
    x, y, z, roll, pitch, yaw = map(float, m.group(1).split())
    # 投影只支持模板里的「正下视、机头朝 +Y」姿态；改了相机就不画轨迹，免得画错位置。
    if abs(pitch - math.pi / 2) > 1e-3 or abs(yaw - math.pi / 2) > 1e-3 or abs(roll) > 1e-3:
        return None
    return x, y, z, float(m.group(2)), int(m.group(3)), int(m.group(4))


def student_camera_fov(world_text):
    """机载学生相机 <sensor name="camera"> 的水平 FOV（度）；模板里没有返回 None。"""
    m = re.search(r'<sensor name="camera" type="camera">.*?<horizontal_fov>([^<]+)</horizontal_fov>',
                  world_text, re.S)
    return math.degrees(float(m.group(1))) if m else None


def project_top(cam, px, py, pz):
    """世界点 -> 顶视图像素。下视且 yaw=90°：图像右 = 世界 +X，图像上 = 世界 +Y。"""
    cx, cy, cz, hfov, w, h = cam
    d = cz - pz
    if d <= 0:
        return None
    f = (w / 2) / math.tan(hfov / 2)
    return int(round(w / 2 + f * (px - cx) / d)), int(round(h / 2 - f * (py - cy) / d))


def in_image(cam, p):
    """投影点是否落在顶视图像内。"""
    return p is not None and 0 <= p[0] < cam[4] and 0 <= p[1] < cam[5]


def _jsonl(path):
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class _Reader:
    """按仿真时间取某路相机「不晚于 t 的最新一帧」，只解码换帧时的那一张。"""

    def __init__(self, frames_dir, cam, np):
        self.idx = load_frames(frames_dir, cam)
        self.f = (open(os.path.join(frames_dir, cam + ".rgb"), "rb") if self.idx else None)
        self.np, self.i, self.img = np, -1, None
        self.ts = None

    def at(self, t):
        j = self.i
        while j + 1 < len(self.idx) and self.idx[j + 1][0] <= t:
            j += 1
        if j != self.i and j >= 0:
            ts, w, h, c, off = self.idx[j]
            self.f.seek(off)
            buf = self.f.read(w * h * c)
            if len(buf) == w * h * c:          # 截断的最后一帧不用
                rgb = self.np.frombuffer(buf, self.np.uint8).reshape(h, w, c)[:, :, :3]
                self.img = rgb[:, :, ::-1].copy()   # RGB -> BGR（cv2）
                self.ts = ts
            self.i = j
        return self.img

    def close(self):
        if self.f:
            self.f.close()


def _track(actions, oracle):
    """位姿真值轨迹 [(ts, x, y, z)]：观测位姿 + 动作后位姿，按时间排序去重。"""
    pts = {}
    for r in actions:
        p = r.get("pose_before")
        if isinstance(p, dict) and r.get("pose_ts"):
            pts[r["pose_ts"]] = (p["x"], p["y"], p.get("z", 0.0))
    for o in oracle:
        p = o.get("pose_gt")
        if isinstance(p, dict) and o.get("timestamp_ns"):
            pts[o["timestamp_ns"]] = (p["x"], p["y"], p.get("z", 0.0))
    return [(t, *pts[t]) for t in sorted(pts)]


def _pos_at(track, t):
    """t 时刻的位姿估计：在相邻两个轨迹点之间线性插值。

    track 只有动作前位姿（动作起点）与动作后真值（动作终点），10 Hz 采样、窗口 1 s；
    直接取 ts<=t 的最后一点最多落后一步（实测中位 0.39 s、学生最大 2.08 s），
    图标/pos 行会和 CHASE 里渲染的真机体错开。两端之外取端点。
    """
    if not track:
        return None
    if t <= track[0][0]:
        return track[0][1:]
    if t >= track[-1][0]:
        return track[-1][1:]
    for a, b in zip(track, track[1:]):
        if a[0] <= t <= b[0]:
            span = b[0] - a[0]
            f = 0.0 if span <= 0 else (t - a[0]) / span
            return tuple(a[i] + f * (b[i] - a[i]) for i in (1, 2, 3))
    return track[-1][1:]


def _speed_at(track, t):
    """t 所在轨迹段的水平速度（m/s）：取包住 t 的那两个点做差分；越界时用最近的一段。

    不再按「t-0.5 s 的最后一点」取点：track 点间隔 0.1–0.9 s，那样常取到与当前点相同的点，
    分子为 0。cf_sitl 有 oracle speed_mps 时优先用 _oracle_speed。
    """
    if len(track) < 2:
        return None
    for a, b in zip(track, track[1:]):
        if a[0] <= t <= b[0] or (t < track[0][0] and a is track[0]):
            dt = (b[0] - a[0]) / 1e9
            return math.dist(a[1:3], b[1:3]) / dt if dt > 0 else None
    a, b = track[-2], track[-1]
    dt = (b[0] - a[0]) / 1e9
    return math.dist(a[1:3], b[1:3]) / dt if dt > 0 else None


def _yaws(actions):
    """机头朝向 [(ts, yaw_rad)]：只有 actions 的 yaw_before 带航向（oracle 没有），按时间排序。

    yaw_before 是度（collect._pose_parts -> yaw_deg，原样落盘）；draw_top_icon 的 cos/sin 吃弧度。
    2026-11-10 前这里直接存度再当弧度传给图标，实测朝向偏 7°–139°（TOP 格左右转看反）。
    """
    pts = {r["pose_ts"]: math.radians(r["yaw_before"]) for r in actions
           if r.get("pose_ts") and isinstance(r.get("yaw_before"), (int, float))}
    return [(t, pts[t]) for t in sorted(pts)]


def _yaw_at(yaws, t):
    """t 时刻的最新已知航向；没有返回 None（顶视图标就不画机头线）。"""
    cur = None
    for ts, yaw in yaws:
        if ts <= t:
            cur = yaw
        else:
            break
    return cur


def sample_grid(t_first, t_end):
    """(period, 各帧区间起点)。第 k 帧代表 [t0+k*period, t0+(k+1)*period)，t0 对齐到
    record 的 ts//period 分桶；每帧取区间里最新的那张，15 Hz 的 chase 每格都有新帧。"""
    period = int(1e9 / FPS)
    t0 = t_first // period * period
    return period, list(range(t0, t_end + 1, period))


def compose(frames_dir, ep_dir, world_path, out_path, title="", result="",
            goal_x=None, waypoint=None, hold_end_s=1.5,
            success_criterion=None, goal_radius=None, quality=None, log=print):
    """合成视频，返回写出的帧数。没有任何机载帧时抛 RuntimeError。

    success_criterion=None 保持旧行为（有 goal_x 就画终点线）；显式传
    goal_reach_ever/visible_goal_reach_v1 时只画 waypoint 及其 goal_radius。
    quality 是 view_layout.VIDEO_QUALITY 的档名（None = 默认档 / 环境变量）：决定画布尺寸、
    显示视角的放大倍率（机载 320x240 模型输入不变，只是显示放大）和 .mp4 的 CRF。
    """
    import cv2
    import numpy as np
    q = vl.quality(quality)
    W, H, CW, CH, tl, tr, bl, br = vl.grid(q["scale"])
    s = q["scale"]
    actions = sorted(_jsonl(os.path.join(ep_dir, "actions.jsonl")),
                     key=lambda r: r.get("action_start_ns") or 0)
    oracle = _jsonl(os.path.join(ep_dir, "oracle.jsonl"))
    track = _track(actions, oracle)
    yaws = _yaws(actions)
    with open(world_path, encoding="utf-8") as f:
        world_text = f.read()
    top_cam, hfov_deg = top_camera_pose(world_text), student_camera_fov(world_text)
    readers = {k: _Reader(frames_dir, k, np) for k in CAMS}
    if not readers["uav"].idx:
        raise RuntimeError(f"{frames_dir} 没有机载相机帧")
    t_end = max([readers["uav"].idx[-1][0]] + [o.get("timestamp_ns", 0) for o in oracle])
    period, times = sample_grid(readers["uav"].idx[0][0], t_end)
    t0 = times[0]
    hit = contact_step(actions)
    stuck = stuck_step(actions)
    hit_mark = _step_mark(hit, oracle, t_end)
    stuck_mark = _step_mark(stuck, oracle, t_end)
    writer = vl.VideoSink(cv2, out_path, FPS, (W, H), q["crf"], log=log)
    if not writer.isOpened():
        raise RuntimeError(f"视频写不开 {out_path}（{writer.backend}）")
    n_end = int(hold_end_s * FPS)
    frames = 0
    try:
        for k, t in enumerate(times + [t_end] * n_end):
            canvas = np.zeros((H, W, 3), np.uint8)
            old = {key: readers[key].i for key in CAMS}
            images = {key: readers[key].at(t + period - 1) for key in CAMS}
            notes = {key: _frame_note(readers[key].ts, readers[key].i, old[key], t)
                     for key in CAMS}
            vl.paste(cv2, canvas, images["chase"], *tl, CW, CH, "CHASE", *notes["chase"], s=s)
            top = images["top"]
            if top is not None:
                top = top.copy()
                _draw_top(cv2, top, top_cam, track, t, hit_mark, stuck_mark, goal_x,
                          waypoint, success_criterion, goal_radius, yaw=_yaw_at(yaws, t))
            vl.paste(cv2, canvas, top, *tr, CW, CH, "TOP", *notes["top"], s=s)
            vl.paste(cv2, canvas, images["uav"], *bl, CW, CH,
                     onboard_label(images["uav"], hfov_deg, display=s), *notes["uav"], s=s)
            _panel(cv2, canvas, *br, t, t0, t_end, actions, track, hit, stuck, title,
                   result if k >= len(times) else "", oracle=oracle, s=s)
            writer.write(canvas)
            frames += 1
    finally:
        writer.release()
        for r in readers.values():
            r.close()
    return frames


def _step_mark(step, oracle, t_end):
    """(碰撞步位置, 时间) 用于顶视图标记；拿不到 oracle 就不画。"""
    if step is None:
        return None
    o = next((o for o in oracle if o.get("step") == step.get("step")), None)
    if o and isinstance(o.get("pose_gt"), dict):
        g = o["pose_gt"]
        return ((g["x"], g["y"], g.get("z", 0.0)), o.get("timestamp_ns") or t_end)
    return None


def _draw_top(cv2, img, cam, track, t, hit_mark, stuck_mark, goal_x, waypoint,
              success_criterion=None, goal_radius=None, yaw=None):
    """顶视图叠加：轨迹（黄）、当前位置（按比例的四旋翼图标，yaw 给机头方向）、contact（红 X）、stuck（橙 X）。

    goal 线只在 success_criterion 为 route_choice（或旧调用未传判据）时画；
    L1/campus 的 goal_reach 判据改画 waypoint 及其半径。
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    if cam is None:
        cv2.putText(img, "top camera pose unsupported: no overlay", (8, img.shape[0] - 10),
                    font, 0.45, (0, 0, 255), 1)
        return
    # 判据线和 waypoint 投到无人机当前高度，不投地面：顶视相机只有 6 m 高，是透视投影，
    # 同一 (x,y) 在地面和 2.3 m 机体高度差 25–30 px、半径圈小 38%（基线 demo 实测），
    # 机体到了圈里画面上却在圈外。按机体高度投，高度变 0.2 m 圈心只移约 3 px。没有轨迹时按地面。
    # 视口内取「离 t 最近」的点：track 只有动作前位姿与动作后真值，按 ts<=t 取会最多落后一步
    # （实测中位 0.39 s、学生最大 2.08 s），图标、pos 行与真机体错开。
    past = [p for p in track if p[0] <= t]
    # 投影高度沿用「t 前最后一点」的 z（高度变化慢、且判据圈半径按它算，保持旧行为）；
    # x/y 用插值（见 _pos_at）。
    z = past[-1][3] if past else 0.0
    draw_goal = (goal_x is not None and
                 success_criterion in (None, "", "route_choice"))
    if draw_goal:
        a, b = project_top(cam, goal_x, -10, z), project_top(cam, goal_x, 10, z)
        if a and b:
            cv2.line(img, a, b, (255, 255, 0), 1)
    radius_px = 8
    if isinstance(waypoint, dict):
        p = project_top(cam, waypoint["x"], waypoint["y"], z)
        if p:
            if goal_radius is not None:
                cx, cy, cz, hfov, w, _h = cam
                radius_px = max(3, int(round((w / 2) / math.tan(hfov / 2) *
                                             goal_radius / max(cz - z, 1e-6))))
            cv2.circle(img, p, radius_px, (255, 0, 255), 2)
    pts = [project_top(cam, x, y, pz) for ts, x, y, pz in past]
    pts = [p for p in pts if p]
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, a, b, (0, 255, 255), 2)
    if pts:
        # 机体图标按真实比例（6 m 高顶视相机下 0.172 m 只有约 12 px，不够看，图标最小 18 px）；
        # 顶视相机 yaw=90°，图像右 = 世界 +X，与 e1_body.draw_top_icon 的约定一致（yaw 弧度）。
        # 位置按 t 在轨迹点间插值（见 _pos_at）：past[-1] 最多落后一步（实测中位 0.39 s、
        # 学生最大 2.08 s），画在那里图标会和 CHASE 里的真机体错开。
        cx, cy, cz, hfov, w, _h = cam
        px_per_m = (w / 2) / math.tan(hfov / 2) / max(cz - z, 1e-6)
        px_, py_, _ = _pos_at(track, t)
        e1_body.draw_top_icon(cv2, img, project_top(cam, px_, py_, z), yaw, px_per_m)
    if pts and not in_image(cam, pts[-1]):
        # 机体飞出顶视相机视野（旧模板 6 m 高 80° 只盖到 x≈5.9 m，森林任务终点在 6–7 m）：
        # 轨迹被裁在图边、机体点画不出来，不说明的话像是停在边上。把点夹到图边并标注。
        _h, _w = img.shape[:2]
        edge = (min(max(pts[-1][0], 4), _w - 5), min(max(pts[-1][1], 4), _h - 5))
        cv2.circle(img, edge, 7, (0, 0, 255), 2)
        warn = "UAV OUT OF TOP VIEW"
        (tw, _), _ = cv2.getTextSize(warn, font, 0.5, 1)
        cv2.putText(img, warn, (max(8, _w - tw - 8), _h - 28), font, 0.5, (0, 0, 255), 1)
    if hit_mark is not None and t >= hit_mark[1]:
        p = project_top(cam, *hit_mark[0])
        if p:
            cv2.drawMarker(img, p, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 3)
    if stuck_mark is not None and t >= stuck_mark[1]:
        p = project_top(cam, *stuck_mark[0])
        if p:
            cv2.drawMarker(img, p, (0, 165, 255), cv2.MARKER_TILTED_CROSS, 18, 3)
    # 判据/半径写右上角，避开左上角的 TOP 标签。只用 ASCII（Hershey 字体）。
    text = f"criterion={success_criterion or 'unlabeled'}"
    if goal_radius is not None:
        text += f" | r={goal_radius:g} m"
    (tw, _), _ = cv2.getTextSize(text, font, 0.45, 1)
    cv2.putText(img, text, (max(8, img.shape[1] - tw - 8), 20), font, 0.45,
                (235, 235, 235), 1)
    legend = []
    if track:
        legend.append(("trail", (0, 255, 255)))
    if draw_goal:
        legend.append(("goal", (255, 255, 0)))
    if isinstance(waypoint, dict):
        label = "waypoint" if goal_radius is None else f"waypoint r={goal_radius:g}m"
        legend.append((label, (255, 0, 255)))
    if hit_mark is not None:
        legend.append(("contact", (0, 0, 255)))
    if stuck_mark is not None:
        legend.append(("stuck", (0, 165, 255)))
    # 图例放左上、TOP 标签下方：左下角是 _paste 写的 sim 时间/帧龄，以前两者叠在一起。
    for i, (label, color) in enumerate(legend):
        cv2.putText(img, label, (8, LEGEND_TOP + 15 * i), font, 0.4, color, 1)


def _active(actions, t):
    """t 时刻正在执行的那一步（action_start_ns <= t 的最后一步）。"""
    cur = None
    for r in actions:
        if (r.get("action_start_ns") or 0) <= t:
            cur = r
        else:
            break
    return cur


def _oracle_speed(oracle, t):
    """t 时刻之前最近一条带 speed_mps 的 oracle 行（cf_sitl 才有）；没有返回 None。"""
    cur = None
    for o in oracle:
        ts = o.get("timestamp_ns")
        if ts is None or ts > t:
            continue
        v = o.get("speed_mps")
        if isinstance(v, (int, float)):
            cur = float(v)
    return cur


def _panel(cv2, canvas, x, y, t, t0, t_end, actions, track, hit, stuck, title, result,
           oracle=(), s=1):
    """把本集数据整理成公共面板的几行字 / 动作段；版式本身在 view_layout.panel（s 是画布放大倍率）。"""
    # 与各视角角标同一口径：绝对仿真时间；括号里才是从视频第一帧起算的时长。
    # 以前面板只写相对时长，角标写绝对时间，同一画面两个数差 t0（基线 demo 差 5.6 s）。
    lines = [title, f"sim t = {t / 1e9:.2f} s  (video +{(t - t0) / 1e9:.1f} s)"]
    cur = _active(actions, t)
    if cur is not None:
        lines.append(f"step {cur['step'] + 1}/{len(actions)}")
    pos_now = _pos_at(track, t)
    if pos_now is not None:
        px, py, pz = pos_now
        lines.append(f"pos x={px:5.2f} y={py:5.2f} z={pz:4.2f} m")
        # 速度优先用 oracle 的 speed_mps（cf_sitl 每步都写，真值）；没有才用所在轨迹段差分。
        # 2026-11-10 前按「t-0.5 s 的最后一点」取点，常取到与当前点相同的点，分子为 0，
        # 28–30% 的帧显示 speed 0.00（同一帧还是 FORWARD_SLOW）。
        spd = _oracle_speed(oracle, t)
        if spd is None:
            spd = _speed_at(track, t)
        if spd is not None:
            lines.append(f"speed {spd:4.2f} m/s")
    flags = []
    if hit is not None and (hit.get("action_start_ns") or 0) <= t:
        flags.append((f"COLLISION contact @ step {hit['step'] + 1}", (0, 0, 255)))
    if stuck is not None and (stuck.get("action_start_ns") or 0) <= t:
        flags.append((f"STUCK @ step {stuck['step'] + 1}", (0, 165, 255)))
    # 动作时间条：每步一段，起止 = 本步/下步的 action_start_ns，最后一步到 t_end。
    segments = []
    for r, nxt in zip(actions, actions[1:] + [None]):
        st = (r.get("action_start_ns") or t0) - t0
        en = ((nxt.get("action_start_ns") if nxt else t_end) or t_end) - t0
        segments.append((st, en, r.get("action_executed", 0)))
    vl.panel(cv2, canvas, x, y, lines, None if cur is None else cur.get("action_executed", 0),
             flags, segments, t - t0, max(t_end - t0, 1), result, s=s)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--out", required=True)
    c = sub.add_parser("compose")
    c.add_argument("--frames", required=True)
    c.add_argument("--episode", required=True, help="collect.py 的 episode 目录")
    c.add_argument("--world", required=True, help="本集用的 world SDF（取顶视相机位姿）")
    c.add_argument("--out", required=True)
    c.add_argument("--title", default="")
    c.add_argument("--result", default="")
    c.add_argument("--goal-x", type=float, default=None)
    c.add_argument("--success-criterion", default=None,
                   help="eval.json 的 success_criterion；route_choice 才画 goal 线")
    c.add_argument("--goal-radius", type=float, default=None,
                   help="goal_reach 判据的 waypoint 半径（米），用于顶视图圆圈")
    c.add_argument("--video-quality", default=None, choices=sorted(vl.VIDEO_QUALITY),
                   help=f"画质档（view_layout.VIDEO_QUALITY；默认 {vl.DEFAULT_QUALITY}，环境变量 {vl.QUALITY_ENV} 覆盖）")
    args = p.parse_args()
    if args.cmd == "record":
        record(args.out)
    else:
        n = compose(args.frames, args.episode, args.world, args.out, args.title,
                    args.result, args.goal_x,
                    success_criterion=args.success_criterion,
                    goal_radius=args.goal_radius, quality=args.video_quality)
        print(f"wrote {args.out} ({n} frames @ {FPS} fps)")


if __name__ == "__main__":
    main()
