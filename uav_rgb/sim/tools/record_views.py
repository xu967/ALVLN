#!/usr/bin/env python3
"""三视图任务视频：录 机载/追尾/顶视 三路相机，再按仿真时间合成一段 mp4。

  record   在 sim 运行期间常驻订阅三路相机，按仿真时间降到 ≤FPS 帧/秒，原始 RGB 追加写盘。
           SIGTERM 结束（eval_closed_loop 用 stop_group 收）。纯 stdlib，任意解释器可跑。
  compose  读帧 + actions/oracle.jsonl，按仿真时间轴合成 1280x720 mp4：
           追尾 | 顶视（叠加轨迹、碰撞点、终点线） / 机载 | 状态面板（动作、速度、碰撞）。
           需要 cv2+numpy（本机 conda python3 有；/usr/bin/python3 没有）。

用法：
  python3 record_views.py record --out FRAMES_DIR            # 需在 sim 的 GZ_PARTITION 环境下
  python3 record_views.py compose --frames FRAMES_DIR --episode EP_DIR \\
      --world scene_42.sdf --out seed_42.mp4 [--title "..."] [--result SUCCESS]
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
import gz_io
import save_frame

CAMS = {"uav": "/uav/camera", "chase": "/uav/chase_camera", "top": "/uav/top_camera"}
FPS = 10                      # 视频帧率 = 每仿真秒 10 帧
STALE_AFTER_S = 1.0           # 帧龄超过此值标 STALE（相机 10–30 Hz，1 s 已是停更）
ACTIONS = ("SAFE_NEUTRAL", "FORWARD_SLOW", "YAW_LEFT", "YAW_RIGHT")
ACTION_BGR = ((200, 200, 200), (80, 200, 80), (230, 160, 40), (40, 160, 230))


def _src(step):
    return step.get("src") or ""


def contact_step(actions):
    """第一个真实接触步；stuck 不是碰撞，不能拿来标 COLLISION。"""
    return next((r for r in actions if r.get("collision") and "contact" in _src(r)), None)


def stuck_step(actions):
    """第一个卡住步；contact+stuck 同时存在时两类都会显示。"""
    return next((r for r in actions if r.get("collision") and "stuck" in _src(r)), None)


def _frame_note(ts, idx, old_idx, t):
    """面板角标：该帧 sim time / 帧龄；重复帧 REPEAT，停更帧 STALE。"""
    if ts is None:
        return "", None
    age = max(0.0, (t - ts) / 1e9)
    if age > STALE_AFTER_S:
        tag, color = " · STALE", (0, 0, 255)
    elif idx == old_idx:
        tag, color = " · REPEAT", (0, 165, 255)
    else:
        tag, color = "", (235, 235, 235)
    return f"sim {ts / 1e9:.2f}s · age {age:.2f}s{tag}", color


def record(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    subs = {k: gz_io.TopicSub(t, buf=8) for k, t in CAMS.items()}
    files = {k: (open(os.path.join(out_dir, k + ".rgb"), "ab"),
                 open(os.path.join(out_dir, k + ".idx.jsonl"), "a")) for k in CAMS}
    last_seq = dict.fromkeys(CAMS, 0)
    last_ts = dict.fromkeys(CAMS, -10**18)
    period = int(1e9 / FPS)
    try:
        while True:
            for k, sub in subs.items():
                for seq, msg in sub.frames():
                    if seq <= last_seq[k]:
                        continue
                    last_seq[k] = seq
                    ts = gz_io.header_stamp_ns(msg)
                    if ts < last_ts[k] + period:
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
                    last_ts[k] = ts
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


def project_top(cam, px, py, pz):
    """世界点 -> 顶视图像素。下视且 yaw=90°：图像右 = 世界 +X，图像上 = 世界 +Y。"""
    cx, cy, cz, hfov, w, h = cam
    d = cz - pz
    if d <= 0:
        return None
    f = (w / 2) / math.tan(hfov / 2)
    return int(round(w / 2 + f * (px - cx) / d)), int(round(h / 2 - f * (py - cy) / d))


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


def compose(frames_dir, ep_dir, world_path, out_path, title="", result="",
            goal_x=None, waypoint=None, hold_end_s=1.5,
            success_criterion=None, goal_radius=None):
    """合成视频，返回写出的帧数。没有任何机载帧时抛 RuntimeError。

    success_criterion=None 保持旧行为（有 goal_x 就画终点线）；显式传
    goal_reach_ever/visible_goal_reach_v1 时只画 waypoint 及其 goal_radius。
    """
    import cv2
    import numpy as np
    actions = sorted(_jsonl(os.path.join(ep_dir, "actions.jsonl")),
                     key=lambda r: r.get("action_start_ns") or 0)
    oracle = _jsonl(os.path.join(ep_dir, "oracle.jsonl"))
    track = _track(actions, oracle)
    with open(world_path, encoding="utf-8") as f:
        top_cam = top_camera_pose(f.read())
    readers = {k: _Reader(frames_dir, k, np) for k in CAMS}
    if not readers["uav"].idx:
        raise RuntimeError(f"{frames_dir} 没有机载相机帧")
    t0 = readers["uav"].idx[0][0]
    t_end = max([readers["uav"].idx[-1][0]] + [o.get("timestamp_ns", 0) for o in oracle])
    hit = contact_step(actions)
    stuck = stuck_step(actions)
    hit_mark = _step_mark(hit, oracle, t_end)
    stuck_mark = _step_mark(stuck, oracle, t_end)
    W, H = 1280, 720
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter 打不开 {out_path}")
    period = int(1e9 / FPS)
    n_end = int(hold_end_s * FPS)
    times = list(range(t0, t_end + 1, period))
    frames = 0
    try:
        for k, t in enumerate(times + [t_end] * n_end):
            canvas = np.zeros((H, W, 3), np.uint8)
            old = {key: readers[key].i for key in CAMS}
            images = {key: readers[key].at(t) for key in CAMS}
            notes = {key: _frame_note(readers[key].ts, readers[key].i, old[key], t)
                     for key in CAMS}
            _paste(cv2, canvas, images["chase"], 0, 0, 640, 360, "CHASE", *notes["chase"])
            top = images["top"]
            if top is not None:
                top = top.copy()
                _draw_top(cv2, top, top_cam, track, t, hit_mark, stuck_mark, goal_x,
                          waypoint, success_criterion, goal_radius)
            _paste(cv2, canvas, top, 640, 0, 640, 360, "TOP", *notes["top"])
            _paste(cv2, canvas, images["uav"], 0, 360, 640, 360, "ONBOARD (model input)",
                   *notes["uav"])
            _panel(cv2, canvas, 640, 360, t, t0, t_end, actions, track, hit, stuck, title,
                   result if k >= len(times) else "")
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


def _paste(cv2, canvas, img, x, y, w, h, label, note="", note_color=(235, 235, 235)):
    """按比例缩放居中贴到 (x,y,w,h)；没有帧时写 NO FRAME，并在右下角标帧龄。"""
    if img is None:
        cv2.putText(canvas, label + ": NO FRAME", (x + 20, y + h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = int(iw * s), int(ih * s)
    ox, oy = x + (w - nw) // 2, y + (h - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    cv2.putText(canvas, label, (x + 8, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)
    if note:
        cv2.putText(canvas, note, (x + 8, y + h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    note_color or (235, 235, 235), 1)


def _draw_top(cv2, img, cam, track, t, hit_mark, stuck_mark, goal_x, waypoint,
              success_criterion=None, goal_radius=None):
    """顶视图叠加：轨迹（黄）、当前位置、contact（红 X）、stuck（橙 X）。

    goal 线只在 success_criterion 为 route_choice（或旧调用未传判据）时画；
    L1/campus 的 goal_reach 判据改画 waypoint 及其半径。
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    if cam is None:
        cv2.putText(img, "top camera pose unsupported: no overlay", (8, img.shape[0] - 10),
                    font, 0.45, (0, 0, 255), 1)
        return
    # 判据线和 waypoint 都投到地面 z=0，不能用无人机当前高度，否则同一位置
    # 会随飞行高度在画面里漂移。
    z = 0.0
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
    pts = [project_top(cam, x, y, pz) for ts, x, y, pz in track if ts <= t]
    pts = [p for p in pts if p]
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, a, b, (0, 255, 255), 2)
    if pts:
        cv2.circle(img, pts[-1], 5, (0, 255, 255), -1)
    if hit_mark is not None and t >= hit_mark[1]:
        p = project_top(cam, *hit_mark[0])
        if p:
            cv2.drawMarker(img, p, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 3)
    if stuck_mark is not None and t >= stuck_mark[1]:
        p = project_top(cam, *stuck_mark[0])
        if p:
            cv2.drawMarker(img, p, (0, 165, 255), cv2.MARKER_TILTED_CROSS, 18, 3)
    # 判据/半径和简单图例；放在底部，避开左上角的 TOP 标签。
    text = f"criterion={success_criterion or '未标注'}"
    if goal_radius is not None:
        text += f" · r={goal_radius:g} m"
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
    for i, (label, color) in enumerate(legend):
        cv2.putText(img, label, (8, img.shape[0] - 10 - 15 * (len(legend) - 1 - i)),
                    font, 0.4, color, 1)


def _active(actions, t):
    """t 时刻正在执行的那一步（action_start_ns <= t 的最后一步）。"""
    cur = None
    for r in actions:
        if (r.get("action_start_ns") or 0) <= t:
            cur = r
        else:
            break
    return cur


def _panel(cv2, canvas, x, y, t, t0, t_end, actions, track, hit, stuck, title, result):
    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (235, 235, 235)
    lines = [title, f"sim t = {(t - t0) / 1e9:6.1f} s"]
    cur = _active(actions, t)
    if cur is not None:
        lines.append(f"step {cur['step'] + 1}/{len(actions)}")
    past = [p for p in track if p[0] <= t]
    if past:
        _, px, py, pz = past[-1]
        lines.append(f"pos x={px:5.2f} y={py:5.2f} z={pz:4.2f} m")
        prev = [p for p in track if p[0] <= t - int(0.5e9)]
        if prev:
            d = math.dist(past[-1][1:3], prev[-1][1:3])
            lines.append(f"speed {d / max((past[-1][0] - prev[-1][0]) / 1e9, 1e-3):4.2f} m/s")
    for i, s in enumerate(lines):
        cv2.putText(canvas, s, (x + 16, y + 34 + 30 * i), font, 0.65, white, 2)
    ay = y + 34 + 30 * len(lines) + 10
    if cur is not None:
        a = cur.get("action_executed", 0)
        cv2.rectangle(canvas, (x + 16, ay - 24), (x + 330, ay + 10), ACTION_BGR[a], -1)
        cv2.putText(canvas, ACTIONS[a], (x + 24, ay), font, 0.8, (0, 0, 0), 2)
    if hit is not None and (hit.get("action_start_ns") or 0) <= t:
        cv2.putText(canvas, f"COLLISION contact @ step {hit['step'] + 1}",
                    (x + 16, ay + 50), font, 0.7, (0, 0, 255), 2)
    if stuck is not None and (stuck.get("action_start_ns") or 0) <= t:
        cv2.putText(canvas, f"STUCK @ step {stuck['step'] + 1}",
                    (x + 16, ay + 84), font, 0.7, (0, 165, 255), 2)
    # 底部动作时间条：每步一段颜色，竖线是当前时刻。
    bx0, bx1, by = x + 16, x + 624, y + 330
    span = max(t_end - t0, 1)
    for r, nxt in zip(actions, actions[1:] + [None]):
        s = (r.get("action_start_ns") or t0) - t0
        e = ((nxt.get("action_start_ns") if nxt else t_end) or t_end) - t0
        u0, u1 = bx0 + int((bx1 - bx0) * s / span), bx0 + int((bx1 - bx0) * e / span)
        cv2.rectangle(canvas, (u0, by - 14), (max(u1, u0 + 1), by),
                      ACTION_BGR[r.get("action_executed", 0)], -1)
    u = bx0 + int((bx1 - bx0) * (t - t0) / span)
    cv2.line(canvas, (u, by - 20), (u, by + 4), (255, 255, 255), 2)
    if result:
        ok = result.upper().startswith("SUCCESS")
        cv2.putText(canvas, result, (x + 16, y + 290), font, 1.1,
                    (80, 220, 80) if ok else (0, 0, 255), 3)


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
    args = p.parse_args()
    if args.cmd == "record":
        record(args.out)
    else:
        n = compose(args.frames, args.episode, args.world, args.out, args.title,
                    args.result, args.goal_x,
                    success_criterion=args.success_criterion,
                    goal_radius=args.goal_radius)
        print(f"wrote {args.out} ({n} frames @ {FPS} fps)")


if __name__ == "__main__":
    main()
