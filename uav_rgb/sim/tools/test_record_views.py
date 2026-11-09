#!/usr/bin/env python3
"""record_views 自检：顶视投影方向、按仿真时间取帧、合成视频帧数。不需要 Gazebo。

compose 需要 cv2+numpy（conda python3）；缺了只跳过合成那一项。
用法：python3 sim/tools/test_record_views.py
"""
import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import record_views as rv

WORLD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "worlds", "uav_rgb.sdf")
WORLD_E1 = os.path.join(os.path.dirname(WORLD), "uav_rgb_e1.sdf")
S = 1_000_000_000


class _FakeImage:
    shape = (360, 480, 3)


class _FakeCv2:
    """只记录绘制调用，用来验证判据/投影逻辑，不需要真实渲染。"""
    FONT_HERSHEY_SIMPLEX = 0
    MARKER_TILTED_CROSS = 0
    LINE_AA = 16

    def __init__(self):
        self.lines, self.circles, self.texts, self.markers, self.polys = [], [], [], [], []

    def line(self, _img, a, b, color, _width, *_rest):
        self.lines.append((a, b, color))

    def circle(self, _img, p, r, color, _width, *_rest):
        self.circles.append((p, r, color))

    def polylines(self, _img, pts, _closed, color, *_rest):
        self.polys.append((pts, color))

    def putText(self, _img, text, *_args):
        self.texts.append(text)

    def drawMarker(self, _img, p, color, *_args):
        self.markers.append((p, color))

    def rectangle(self, *_args):
        pass

    def getTextSize(self, text, _font, _scale, _thickness):
        return (len(text) * 6, 10), 2


def check_ascii_overlay(cam):
    """Hershey 字体画不了「·」和中文，叠字一律 ASCII。"""
    cv2 = _FakeCv2()
    note, _ = rv._frame_note(10 * S, 2, 2, 10 * S + S // 2)
    note2, _ = rv._frame_note(10 * S, 2, 1, 12 * S)
    cv2.texts += [note, note2]
    rv._draw_top(cv2, _FakeImage(), cam, [(10 * S, 1.0, 0.0, 2.0)], 10 * S, None, None, None,
                 {"x": 2.0, "y": 1.0}, success_criterion=None, goal_radius=1.5)
    actions = [{"step": 0, "action_executed": 1, "action_start_ns": 10 * S,
                "collision": True, "src": "contact+stuck"}]
    rv._panel(cv2, None, 640, 360, 10 * S + 1, 10 * S, 11 * S, actions,
              [(10 * S, 1.0, 0.0, 2.0)], actions[0], actions[0], "student | x3 | seed 1",
              "SUCCESS")
    bad = [t for t in cv2.texts if any(ord(ch) > 127 for ch in t)]
    assert not bad, ("叠字含非 ASCII，会被画成 ?", bad)
    print("  ok  录像叠字全是 ASCII")


class _FakeSub:
    """record() 用的 TopicSub 替身：第一次给出全部帧，第二次模拟 SIGTERM 退出。"""
    frames_by_topic = {}

    def __init__(self, topic, buf=8):
        self.msgs = self.frames_by_topic.get(topic, [])
        self.calls = 0

    def frames(self):
        self.calls += 1
        if self.calls > 1:
            raise SystemExit(0)
        return [(i + 1, m) for i, m in enumerate(self.msgs)]

    def close(self):
        pass


def _msg(ts, v=7, w=8, h=6):
    import base64
    return {"width": w, "height": h, "step": w * 3, "pixelFormatType": "RGB_INT8",
            "data": base64.b64encode(bytes([v]) * (w * h * 3)).decode(),
            "header": {"stamp": {"sec": ts // S, "nsec": ts % S}}}


def run_record(out, stamps):
    """真跑 record()：TopicSub 换成替身，SIGTERM handler 跑完恢复。"""
    import signal
    _FakeSub.frames_by_topic = {rv.CAMS[k]: [_msg(t) for t in ts] for k, ts in stamps.items()}
    real_sub, real_term = rv.gz_io.TopicSub, signal.getsignal(signal.SIGTERM)
    rv.gz_io.TopicSub = _FakeSub
    try:
        rv.record(out)
    except SystemExit:
        pass
    finally:
        rv.gz_io.TopicSub = real_sub
        signal.signal(signal.SIGTERM, real_term)


def check_record_buckets_and_overwrite():
    period = S // rv.FPS
    chase = [10 * S + S // 30 + i * S // 15 for i in range(30)]   # 15 Hz，相位不对齐 100 ms
    uav = [10 * S + i * S // 30 for i in range(60)]                # 30 Hz
    with tempfile.TemporaryDirectory() as d:
        # 残留目录：上一次录像的帧不能混进这一次（以前用 "ab"/"a" 追加）
        with open(os.path.join(d, "chase.idx.jsonl"), "w") as f:
            f.write(json.dumps({"ts": 1, "w": 8, "h": 6, "c": 3, "off": 0}) + "\n")
        with open(os.path.join(d, "chase.rgb"), "wb") as f:
            f.write(b"OLD!" * 36)
        run_record(d, {"uav": uav, "chase": chase, "top": []})
        rows = rv.load_frames(d, "chase")
        assert rows[0][0] != 1 and rows[0][4] == 0, ("残留帧混进新录像", rows[:2])
        assert os.path.getsize(os.path.join(d, "chase.rgb")) == len(rows) * 8 * 6 * 3
        buckets = [r[0] // period for r in rows]
        want = sorted({t // period for t in chase})
        assert buckets == want, ("每个 100 ms 桶必须恰好一帧", buckets, want)
        print(f"  ok  record 覆盖残留帧；15 Hz chase 每桶一帧（{len(rows)} 帧 / {len(want)} 桶）")
        try:
            import numpy as np
        except ImportError:
            print("  skip 无 numpy：跳过重复帧检查")
            return
        _, times = rv.sample_grid(uav[0], uav[-1])

        def repeats(reader):
            old, n = reader.i, 0
            for t in times:
                reader.at(t + period - 1)
                n += reader.i == old
                old = reader.i
            return n
        rd = rv._Reader(d, "chase", np)
        got = repeats(rd)
        rd.close()
        assert got == 0, ("15 Hz chase 在 10 fps 网格上不该有重复帧", got)
        # 对照：改前的取帧规则（距上一帧 >= period 才留）在同一组帧上会产生重复帧。
        kept, last = [], -10**18
        for t in chase:
            if t >= last + period:
                kept.append(t)
                last = t
        old_dir = os.path.join(d, "old")
        os.makedirs(old_dir)
        write_cam(old_dir, "chase", [(t, 1) for t in kept])
        rd = rv._Reader(old_dir, "chase", np)
        old_repeats = repeats(rd)
        rd.close()
        assert old_repeats > 0, old_repeats
        print(f"  ok  10 fps 网格无重复帧（改前规则同一组帧重复 {old_repeats} 次）")


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
        assert calls == [(2.0, 1.0, 0.0)], ("没有轨迹时 waypoint 投到地面 z=0", calls)
        assert cv2.circles and cv2.circles[0][1] >= 70, cv2.circles
        # 有轨迹时 waypoint/goal 线投到机体当前高度：透视相机下地面投影和 2.3 m 高的机体差 25–30 px
        calls.clear(); cv2.circles.clear()
        track = [(0, 0.5, 0.0, 2.3), (10 * S, 1.0, 0.0, 2.1)]
        rv._draw_top(cv2, img, cam, track, 5 * S, None, None, 5.0, {"x": 2.0, "y": 1.0},
                     success_criterion="route_choice", goal_radius=1.5)
        assert (2.0, 1.0, 2.3) in calls and (5.0, 10, 2.3) in calls, ("按当前高度投影", calls)
        assert (2.0, 1.0, 0.0) not in calls, calls
        r_alt = cv2.circles[0][1]
        assert r_alt > 70 * 1.3, ("半径圈也按机体高度算，比地面投影大", r_alt)
    finally:
        rv.project_top = real_project
    print("  ok  goal 线仅 route_choice；waypoint 半径按机体高度投影")


def check_overlay_layout(cam):
    """图例不能压住左下角的时间角标；面板时间与角标同口径；机载标题写实际分辨率。"""
    cv2, img = _FakeCv2(), _FakeImage()
    texts = []
    cv2.putText = lambda _img, text, org, *_a: texts.append((text, org))
    track = [(10 * S, 1.0, 0.0, 2.2)]
    rv._draw_top(cv2, img, cam, track, 10 * S, ((1.0, 0.0, 2.2), 10 * S), None, None,
                 {"x": 2.0, "y": 1.0}, success_criterion="goal_reach_ever", goal_radius=1.5)
    legend = [(t, o) for t, o in texts if t in ("trail", "contact") or t.startswith("waypoint")]
    assert len(legend) == 3, texts
    assert all(o[1] < img.shape[0] // 2 for _, o in legend), ("图例应在上半部", legend)
    assert all(o[1] > 22 for _, o in legend), ("图例在 TOP 标签之下", legend)
    texts.clear()
    cv2.rectangle = lambda *_a: None
    cv2.line = lambda *_a: None
    actions = [{"step": 0, "action_executed": 1, "action_start_ns": 10 * S}]
    rv._panel(cv2, None, 640, 360, 16 * S + S // 5, 10 * S, 20 * S, actions, track,
              None, None, "t", "")
    sim_line = next(t for t, _ in texts if t.startswith("sim t"))
    note, _ = rv._frame_note(16 * S + S // 5, 0, -1, 16 * S + S // 5)
    assert sim_line.startswith("sim t = 16.20 s") and note.startswith("sim 16.20s"), (sim_line, note)
    assert "+6.2 s" in sim_line, sim_line
    assert rv.onboard_label(_FakeImage()) == "ONBOARD 480x360 camera (model input is resized)"
    assert rv.onboard_label(_FakeImage(), 60.0) == \
        "ONBOARD 480x360 HFOV 60deg camera (model input is resized)"
    assert rv.onboard_label(None) == "ONBOARD"
    print("  ok  图例在上半部不压角标；面板与角标同为绝对仿真时间；机载标题写实际分辨率与 FOV")


def check_out_of_view_marker(cam):
    """机体投影落在顶视图外时要标注，不能让裁在图边的轨迹冒充机体位置。"""
    cv2, img = _FakeCv2(), _FakeImage()
    cx, cy, cz, hfov, w, h = cam
    far_x = cx + (w / 2 + 50) * (cz - 2.0) / ((w / 2) / math.tan(hfov / 2))   # 图右边外 50 px
    rv._draw_top(cv2, img, cam, [(10 * S, far_x, cy, 2.0)], 10 * S, None, None, None, None)
    assert any("OUT OF TOP VIEW" in t for t in cv2.texts), cv2.texts
    edge = [c for c in cv2.circles if c[2] == (0, 0, 255)]
    assert edge and 0 <= edge[0][0][0] < w and 0 <= edge[0][0][1] < h, ("标记应夹在图内", edge)
    cv2 = _FakeCv2()
    rv._draw_top(cv2, img, cam, [(10 * S, cx, cy, 2.0)], 10 * S, None, None, None, None)
    assert not any("OUT OF TOP VIEW" in t for t in cv2.texts), cv2.texts
    print("  ok  机体出顶视视野时画边缘标记 + OUT OF TOP VIEW")


def check_e1_top_coverage():
    """E1 模板顶视相机在 2 m 飞行高度要盖住 forest_e1 整条走廊（spawn x=-0.5 到 goal x=7）和
    柱阵任务区（x 1.3–5.2, y ±1.85）；旧位姿 (2.5,0,6) 480x360 只到 x≈5.9，机体飞到终点就出画。"""
    with open(WORLD_E1, encoding="utf-8") as f:
        text = f.read()
    cam = rv.top_camera_pose(text)
    assert cam is not None, "E1 模板的 studio_top_camera 没解析出来"
    assert abs(rv.student_camera_fov(text) - 60.0) < 1e-6, rv.student_camera_fov(text)
    corners = [(-0.5, 0.0), (7.0, 1.65), (7.0, -1.65), (1.3, 1.85), (5.2, -1.85)]
    for x, y in corners:
        assert rv.in_image(cam, rv.project_top(cam, x, y, 2.0)), ("顶视盖不住", (x, y), cam)
    old = (2.5, 0.0, 6.0, cam[3], 480, 360)
    assert not rv.in_image(old, rv.project_top(old, 7.0, 0.0, 2.0)), "旧位姿本应盖不住终点"
    print(f"  ok  E1 顶视相机 {cam[:3]} {cam[4]}x{cam[5]} 盖住森林走廊与柱阵区")


def check_top_icon_and_yaw(cam):
    """当前位置画按比例四旋翼图标（保护罩方框 + 机头线），不再是圆点；机头线只在有 yaw 时画。"""
    cv2, img = _FakeCv2(), _FakeImage()
    track = [(10 * S, 1.0, 0.0, 2.2)]
    rv._draw_top(cv2, img, cam, track, 10 * S, None, None, None, None, yaw=0.3)
    assert len(cv2.polys) == 2 and all(len(p[0]) == 4 for p, _c in cv2.polys), "保护罩方框（黑描边 + 白线）"
    assert len([c for c in cv2.circles if c[2] == (200, 200, 200)]) == 4, "4 个桨盘"
    nose = [l for l in cv2.lines if l[2] == (0, 0, 255)]
    assert len(nose) == 1 and nose[0][0] == rv.project_top(cam, 1.0, 0.0, 2.2), nose
    assert not [c for c in cv2.circles if c[2] == (0, 255, 255)], "不再画黄圆点"
    cv2 = _FakeCv2()
    rv._draw_top(cv2, img, cam, track, 10 * S, None, None, None, None)
    assert not [l for l in cv2.lines if l[2] == (0, 0, 255)], "yaw=None 不画机头线"
    acts = [{"pose_ts": 10 * S, "yaw_before": 0.1}, {"pose_ts": 12 * S, "yaw_before": 0.2},
            {"pose_ts": 11 * S}, {"pose_before": {}}]
    yaws = rv._yaws(acts)
    assert yaws == [(10 * S, 0.1), (12 * S, 0.2)], yaws
    assert rv._yaw_at(yaws, 9 * S) is None and rv._yaw_at(yaws, 11 * S) == 0.1 and rv._yaw_at(yaws, 13 * S) == 0.2
    print("  ok  顶视当前位置画四旋翼图标；航向取 actions 的 yaw_before")


def check_chase_patch():
    """录像时的追尾相机补丁只动 chase_cam 的 5 个值，模板文件本身不改；x3 模板没有 chase_offset 就不动。"""
    import hashlib
    import shutil
    with open(WORLD_E1, "rb") as f:
        tpl_sha = hashlib.sha256(f.read()).hexdigest()
    with tempfile.TemporaryDirectory() as d:
        w = os.path.join(d, "scene_1.sdf")
        shutil.copy(WORLD_E1, w)
        patch = rv.patch_chase_cam(w, "high")
        assert patch["chase_size"] == [1280, 720] and patch["chase_hfov_deg"] == 60.0, patch
        assert patch["chase_offset"] == "-1.0 0 0.45" and patch["quality"] == "high"
        with open(WORLD_E1, encoding="utf-8") as f:
            before = f.read().splitlines()
        with open(w, encoding="utf-8") as f:
            after = f.read().splitlines()
        assert len(before) == len(after)
        changed = [(a, b) for a, b in zip(before, after) if a != b]
        tags = sorted(b.strip().split(">")[0] + ">" for _a, b in changed)
        assert tags == ["<chase_offset>", "<height>", "<horizontal_fov>", "<width>"], changed
        assert "<chase_offset>-1.0 0 0.5</chase_offset>" in "".join(a for a, _b in changed)
        assert rv.top_camera_pose("\n".join(after)) == rv.top_camera_pose("\n".join(before)), "顶视相机不动"
        assert abs(rv.student_camera_fov("\n".join(after)) - 60.0) < 1e-6, "机载相机不动"
        with open(w, "rb") as f:
            assert patch["world_sha256"] == hashlib.sha256(f.read()).hexdigest()
        # medium 档：分辨率保持 640x360
        shutil.copy(WORLD_E1, w)
        assert rv.patch_chase_cam(w, "medium")["chase_size"] == [640, 360]
        # x3 模板没有 chase_offset：不改、返回 None
        shutil.copy(WORLD, w)
        with open(w, "rb") as f:
            x3 = f.read()
        assert rv.patch_chase_cam(w, "high") is None
        with open(w, "rb") as f:
            assert f.read() == x3
    with open(WORLD_E1, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == tpl_sha, "模板 sha 不能变"
    print("  ok  追尾相机补丁只改 chase_offset / HFOV / 分辨率 4 行，模板不动，x3 模板跳过")


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
    check_overlay_layout(cam)
    check_out_of_view_marker(cam)
    check_e1_top_coverage()
    check_ascii_overlay(cam)
    check_top_icon_and_yaw(cam)
    check_chase_patch()
    check_record_buckets_and_overwrite()

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
        # .webm 走 VP8（浏览器可播）；本机 cv2 没有 H.264，mp4 只能是 mp4v。
        assert rv.fourcc_for("a/seed_1.webm") == "VP80" and rv.fourcc_for("a/b.mp4") == "mp4v"
        web = os.path.join(d, "v.webm")
        n = rv.compose(d, ep, WORLD, web, title="t", result="COLLISION", goal_x=5.0,
                       hold_end_s=0.5)
        with open(web, "rb") as f:
            head = f.read(4096)
        assert head[:4] == b"\x1a\x45\xdf\xa3" and b"webm" in head and b"V_VP8" in head, head[:40]
        cap = cv2.VideoCapture(web)
        back = 0
        while cap.read()[0]:
            back += 1
        cap.release()
        assert n == 16 and back == 16, (n, back)
        print("  ok  .webm 合成为 VP8/WebM，回读 16 帧")
        # high 档：2560x1440 画布；有 ffmpeg 时 .mp4 是 libx264（avc1），否则回退 cv2 并打日志
        logs = []
        hi = os.path.join(d, "hi.mp4")
        n = rv.compose(d, ep, WORLD, hi, title="t", result="COLLISION", goal_x=5.0,
                       hold_end_s=0.5, quality="high", log=logs.append)
        cap = cv2.VideoCapture(hi)
        ok, frame = cap.read()
        assert ok and frame.shape == (1440, 2560, 3) and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 16
        cap.release()
        with open(hi, "rb") as f:
            head = f.read(4096)
        if rv.vl.find_ffmpeg():
            assert b"avc1" in head and not logs, (head[:40], logs)
        else:
            assert logs and "回退" in logs[0], logs
        print(f"  ok  high 档合成 2560x1440（{'libx264' if rv.vl.find_ffmpeg() else 'cv2 回退'}）")
    print("test_record_views: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
