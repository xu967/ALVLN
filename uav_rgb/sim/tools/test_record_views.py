#!/usr/bin/env python3
"""record_views 自检：顶视投影方向、按仿真时间取帧、合成视频帧数。不需要 Gazebo。

compose 需要 cv2+numpy（conda python3）；缺了只跳过合成那一项。
用法：python3 sim/tools/test_record_views.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import record_views as rv

WORLD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "worlds", "uav_rgb.sdf")
S = 1_000_000_000


class _FakeImage:
    shape = (360, 480, 3)


class _FakeCv2:
    """只记录绘制调用，用来验证判据/投影逻辑，不需要真实渲染。"""
    FONT_HERSHEY_SIMPLEX = 0
    MARKER_TILTED_CROSS = 0

    def __init__(self):
        self.lines, self.circles, self.texts, self.markers = [], [], [], []

    def line(self, _img, a, b, color, _width):
        self.lines.append((a, b, color))

    def circle(self, _img, p, r, color, _width):
        self.circles.append((p, r, color))

    def putText(self, _img, text, *_args):
        self.texts.append(text)

    def drawMarker(self, _img, p, color, *_args):
        self.markers.append((p, color))

    def getTextSize(self, text, _font, _scale, _thickness):
        return (len(text) * 6, 10), 2


def check_labels_and_notes():
    actions = [
        {"step": 0, "collision": True, "src": "stuck"},
        {"step": 1, "collision": True, "src": "contact"},
        {"step": 2, "collision": True, "src": "contact+stuck"},
        {"step": 3, "collision": False, "src": None},
    ]
    assert rv.contact_step(actions)["step"] == 1, "碰撞只认 src 含 contact"
    assert rv.stuck_step(actions)["step"] == 0, "stuck 必须单独识别"
    assert rv.contact_step([{"collision": True}]) is None, "旧数据无 src 不能当 contact"
    note, _ = rv._frame_note(10 * S, 2, 2, 10 * S + S // 2)
    assert "REPEAT" in note, note
    note, _ = rv._frame_note(10 * S, 2, 1, 12 * S)
    assert "STALE" in note, note
    assert rv._frame_note(None, -1, -1, 0) == ("", None)
    print("  ok  contact/stuck 标签与 REPEAT/STALE 帧龄")


def check_goal_criterion(cam):
    calls = []
    real_project = rv.project_top

    def record(cam_arg, x, y, z):
        calls.append((x, y, z))
        return 10, 10

    rv.project_top = record
    try:
        cv2, img = _FakeCv2(), _FakeImage()
        rv._draw_top(cv2, img, cam, [], 0, None, None, 5.0, None,
                     success_criterion="goal_reach_ever")
        assert calls == [], ("goal_reach 不画终点线", calls)
        rv._draw_top(cv2, img, cam, [], 0, None, None, 5.0, None,
                     success_criterion="route_choice")
        assert sorted(calls) == [(5.0, -10, 0.0), (5.0, 10, 0.0)], calls
        calls.clear()
        rv._draw_top(cv2, img, cam, [], 0, None, None, 5.0, None,
                     success_criterion=None)
        assert sorted(calls) == [(5.0, -10, 0.0), (5.0, 10, 0.0)], ("默认保持旧行为", calls)
        calls.clear(); cv2.circles.clear()
        rv._draw_top(cv2, img, cam, [], 0, None, None, None, {"x": 2.0, "y": 1.0},
                     success_criterion="goal_reach_ever", goal_radius=1.5)
        assert calls == [(2.0, 1.0, 0.0)], ("waypoint 必须投到地面 z=0", calls)
        assert cv2.circles and cv2.circles[0][1] >= 70, cv2.circles
    finally:
        rv.project_top = real_project
    print("  ok  goal 线仅 route_choice；waypoint 半径/地面投影")


def write_cam(d, cam, frames, w=8, h=6):
    with open(os.path.join(d, cam + ".rgb"), "wb") as data, \
            open(os.path.join(d, cam + ".idx.jsonl"), "w") as idx:
        for ts, v in frames:
            idx.write(json.dumps({"ts": ts, "w": w, "h": h, "c": 3, "off": data.tell()}) + "\n")
            data.write(bytes([v]) * (w * h * 3))
        idx.write('{"ts": 9')            # SIGTERM 截断的半行必须被忽略


def main():
    with open(WORLD, encoding="utf-8") as f:
        cam = rv.top_camera_pose(f.read())
    assert cam is not None, "模板里的 studio_top_camera 没解析出来"
    cx, cy, cz, _, w, h = cam
    c = rv.project_top(cam, cx, cy, 2.0)
    r = rv.project_top(cam, cx + 1, cy, 2.0)
    u = rv.project_top(cam, cx, cy + 1, 2.0)
    assert c == (w // 2, h // 2), c
    assert r[0] > c[0] and r[1] == c[1], ("世界 +X 应在图像右侧", c, r)
    assert u[1] < c[1] and u[0] == c[0], ("世界 +Y 应在图像上方", c, u)
    assert rv.project_top(cam, cx, cy, cz + 1) is None
    assert rv.top_camera_pose(open(WORLD).read().replace(
        "2.5 0 6 0 1.57079632679 1.57079632679", "2.5 0 6 0 1.2 0")) is None, "非下视姿态应拒绝"
    print("  ok  顶视投影方向/姿态门禁")
    check_labels_and_notes()
    check_goal_criterion(cam)

    with tempfile.TemporaryDirectory() as d:
        write_cam(d, "uav", [(10 * S, 10), (10 * S + S // 2, 20), (11 * S, 30)])
        write_cam(d, "chase", [(10 * S, 40)])
        assert len(rv.load_frames(d, "uav")) == 3
        try:
            import numpy as np
        except ImportError:
            print("  skip 无 numpy：跳过取帧/合成")
            return 0
        rd = rv._Reader(d, "uav", np)
        assert rd.at(10 * S - 1) is None
        assert rd.at(10 * S + S // 4)[0, 0, 0] == 10
        assert rd.ts == 10 * S
        assert rd.at(11 * S)[0, 0, 0] == 30
        assert rd.ts == 11 * S
        rd.close()
        print("  ok  按仿真时间取不晚于 t 的最新帧，截断行忽略")
        try:
            import cv2
        except ImportError:
            print("  skip 无 cv2：跳过合成")
            return 0
        ep = os.path.join(d, "ep")
        os.makedirs(ep)
        with open(os.path.join(ep, "actions.jsonl"), "w") as f:
            for i in range(2):
                f.write(json.dumps({"step": i, "action_executed": 1, "action_start_ns":
                                    10 * S + i * S // 2, "pose_ts": 10 * S + i * S // 2,
                                    "pose_before": {"x": i * 0.1, "y": 0, "z": 2.1},
                                    "collision": i == 1, "src": "contact"}) + "\n")
        with open(os.path.join(ep, "oracle.jsonl"), "w") as f:
            for i in range(2):
                f.write(json.dumps({"step": i, "timestamp_ns": 10 * S + (i + 1) * S // 2,
                                    "pose_gt": {"x": (i + 1) * 0.1, "y": 0, "z": 2.1}}) + "\n")
        out = os.path.join(d, "v.mp4")
        n = rv.compose(d, ep, WORLD, out, title="t", result="COLLISION", goal_x=5.0,
                       hold_end_s=0.5)
        # 10.0~11.0 s @10 fps = 11 帧 + 结尾停留 5 帧；顶视没帧要画 NO FRAME 而不是崩
        assert n == 16, n
        cap = cv2.VideoCapture(out)
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 16
        ok, frame = cap.read()
        assert ok and frame.shape == (720, 1280, 3)
        cap.release()
        print("  ok  合成 16 帧 1280x720（缺顶视帧不崩）")
    print("test_record_views: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
