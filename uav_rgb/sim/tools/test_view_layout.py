#!/usr/bin/env python3
"""view_layout 自检：网格常量、paste 的缩放/居中、panel 的行距与时间条、叠字 ASCII。不需要 Gazebo。

paste 的像素检查需要 cv2+numpy（.venv-tools / 3dgs-venv）；缺了只跳过那一项。
用法：python3 sim/tools/test_view_layout.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import view_layout as vl


class _FakeCv2:
    """只记录绘制调用：文字 (text, org, scale, color, thickness)、矩形、线段。"""
    FONT_HERSHEY_SIMPLEX = 0

    def __init__(self):
        self.texts, self.rects, self.lines = [], [], []

    def putText(self, _img, text, org, _font, scale, color, thickness, *_a):
        self.texts.append((text, org, scale, color, thickness))

    def rectangle(self, _img, p0, p1, color, _t):
        self.rects.append((p0, p1, color))

    def line(self, _img, a, b, color, _t):
        self.lines.append((a, b, color))


def check_grid():
    cells = (vl.CELL_TL, vl.CELL_TR, vl.CELL_BL, vl.CELL_BR)
    assert (vl.W, vl.H) == (1280, 720) and (vl.CW, vl.CH) == (640, 360)
    covered = set()
    for x, y in cells:
        for u in range(x, x + vl.CW, 40):
            for v in range(y, y + vl.CH, 40):
                assert (u, v) not in covered, ("格子重叠", (u, v))
                covered.add((u, v))
    assert len(covered) == (vl.W // 40) * (vl.H // 40), "四格应正好铺满 1280x720"
    assert len(vl.ACTIONS) == len(vl.ACTION_BGR) == 4
    print("  ok  四格 640x360 铺满 1280x720，不重叠")


def check_panel_layout():
    """行距 30 px、动作色块紧跟文字行、告警行 34 px、时间条按 start/span 比例、结果大字颜色。"""
    cv2 = _FakeCv2()
    x, y = vl.CELL_BR
    lines = ["title", "sim t = 1.00 s", "step 2/2", "pos", "speed"]
    segs = [(0.0, 0.5, 1), (0.5, 1.0, 2)]
    vl.panel(cv2, None, x, y, lines, 2, [("COLLISION @ step 2", (0, 0, 255))], segs, 0.25, 1.0, "SUCCESS")
    ys = [org[1] for text, org, *_ in cv2.texts if text in lines]
    assert ys == [y + 34 + 30 * i for i in range(5)], ys
    ay = y + 34 + 30 * 5 + 10
    box = [r for r in cv2.rects if r[2] == vl.ACTION_BGR[2] and r[0][1] == ay - 24]
    assert box and box[0][0] == (x + 16, ay - 24), ("动作色块应在最后一行文字下方", cv2.rects)
    assert ("YAW_LEFT", (x + 24, ay)) == [(t, o) for t, o, *_ in cv2.texts if t == "YAW_LEFT"][0]
    flag = [o for t, o, *_ in cv2.texts if t.startswith("COLLISION")][0]
    assert flag == (x + 16, ay + 50), flag
    bx0, bx1, by = x + 16, x + vl.CW - 16, y + 330
    bar = [r for r in cv2.rects if r[1][1] == by]
    assert [r[0][0] for r in bar] == [bx0, bx0 + (bx1 - bx0) // 2] and bar[-1][1][0] == bx1, bar
    assert [r[2] for r in bar] == [vl.ACTION_BGR[1], vl.ACTION_BGR[2]]
    cursor = [a for a, b, c in cv2.lines if c == (255, 255, 255)][0]
    assert cursor == (bx0 + (bx1 - bx0) // 4, by - 20), cursor
    res = [(o, c, th) for t, o, _s, c, th in cv2.texts if t == "SUCCESS"][0]
    assert res == ((x + 16, y + 290), (80, 220, 80), 3), res
    # 失败结果红色；action=None 不画色块；result 空不画
    cv2 = _FakeCv2()
    vl.panel(cv2, None, x, y, ["t"], None, [], [], 0.0, 1.0, "COLLISION")
    assert [c for t, _o, _s, c, _th in cv2.texts if t == "COLLISION"] == [(0, 0, 255)]
    assert not any(r[0][1] == y + 34 + 30 + 10 - 24 for r in cv2.rects), "action=None 不该画色块"
    cv2 = _FakeCv2()
    vl.panel(cv2, None, x, y, ["t"], 1, [], [], 0.0, 1.0, "")
    assert all(t in ("t", "FORWARD_SLOW") for t, *_ in cv2.texts), cv2.texts
    bad = [t for t, *_ in cv2.texts if any(ord(ch) > 127 for ch in t)]
    assert not bad, ("叠字含非 ASCII", bad)
    print("  ok  面板行距/色块/告警/时间条/结果字位置与颜色")


def check_onboard_label():
    class _Img:
        shape = (240, 320, 3)

    class _Img2:
        shape = (480, 640, 3)
    assert vl.onboard_label(None) == "ONBOARD"
    assert vl.onboard_label(_Img()) == "ONBOARD 320x240 camera (model input is resized)"
    assert vl.onboard_label(_Img(), 59.8) == "ONBOARD 320x240 HFOV 60deg camera (model input is resized)"
    # high 档：显示帧是 2× 渲染，标题仍写模型输入 320x240
    assert vl.onboard_label(_Img2(), 60.0, display=2, rendered_at=2) == "ONBOARD 320x240 HFOV 60deg (model input; display 2x)"
    # Gazebo 的机载帧是原生 320x240、由 paste 放大：rendered_at 默认 1
    assert vl.onboard_label(_Img(), 60.0, display=2) == "ONBOARD 320x240 HFOV 60deg (model input; display 2x)"
    print("  ok  机载标题")


def check_quality_and_scale():
    """画质表三档；环境变量覆盖；grid(2) 是 2560x1440；paste/panel 的 s=2 坐标、字号、线宽全部翻倍。"""
    assert set(vl.VIDEO_QUALITY) == {"low", "medium", "high"}
    for name, q in vl.VIDEO_QUALITY.items():
        assert q["canvas"] == (vl.W * q["scale"], vl.H * q["scale"]) and q["ssaa"] >= 1 and 0 < q["crf"] < 52, (name, q)
    assert vl.VIDEO_QUALITY["high"]["crf"] < vl.VIDEO_QUALITY["medium"]["crf"] < vl.VIDEO_QUALITY["low"]["crf"]
    old = os.environ.pop(vl.QUALITY_ENV, None)
    try:
        assert vl.quality()["name"] == vl.DEFAULT_QUALITY and vl.quality("high")["canvas"] == (2560, 1440)
        os.environ[vl.QUALITY_ENV] = "low"
        assert vl.quality("high")["name"] == "low", "环境变量应覆盖传入的档位"
        try:
            del os.environ[vl.QUALITY_ENV]
            vl.quality("ultra")
            raise AssertionError("未知档位应报错")
        except ValueError:
            pass
    finally:
        if old is not None:
            os.environ[vl.QUALITY_ENV] = old
    w, h, cw, ch, tl, tr, bl, br = vl.grid(2)
    assert (w, h, cw, ch) == (2560, 1440, 1280, 720) and (tl, tr, bl, br) == ((0, 0), (1280, 0), (0, 720), (1280, 720))
    assert vl.grid(1) == (vl.W, vl.H, vl.CW, vl.CH, vl.CELL_TL, vl.CELL_TR, vl.CELL_BL, vl.CELL_BR)
    # panel s=2：每个坐标 = s=1 的两倍（面板原点同为 0,0），字号、线宽也翻倍
    one, two = _FakeCv2(), _FakeCv2()
    args = (["t", "u"], 1, [("COLLISION", (0, 0, 255))], [(0.0, 1.0, 1)], 0.5, 1.0, "SUCCESS")
    vl.panel(one, None, 0, 0, *args)
    vl.panel(two, None, 0, 0, *args, s=2)
    assert len(one.texts) == len(two.texts) and len(one.rects) == len(two.rects)
    for (t1, o1, s1, c1, th1), (t2, o2, s2, c2, th2) in zip(one.texts, two.texts):
        assert t1 == t2 and o2 == (o1[0] * 2, o1[1] * 2) and abs(s2 - 2 * s1) < 1e-9 and th2 == 2 * th1 and c1 == c2, (t1, o1, o2)
    for (p0, p1, c), (q0, q1, d) in zip(one.rects, two.rects):
        assert q0 == (p0[0] * 2, p0[1] * 2) and q1 == (p1[0] * 2, p1[1] * 2) and c == d, (p0, q0)
    # paste s=2：标签/角标位置与字号翻倍（NO FRAME 路径不需要图像）
    one, two = _FakeCv2(), _FakeCv2()
    vl.paste(one, None, None, 0, 0, vl.CW, vl.CH, "CHASE")
    vl.paste(two, None, None, 0, 0, vl.CW * 2, vl.CH * 2, "CHASE", s=2)
    (t1, o1, s1, _c, th1), (t2, o2, s2, _d, th2) = one.texts[0], two.texts[0]
    assert t1 == t2 and o2 == (o1[0] * 2, o1[1] * 2) and abs(s2 - 2 * s1) < 1e-9 and th2 == 2 * th1
    print("  ok  画质三档 / 环境变量覆盖 / grid(2) / panel、paste 的 s=2 全部翻倍")


def check_video_sink():
    """VideoSink：.mp4 有 ffmpeg 走 libx264（回读帧数、尺寸、avc1）；强制无 ffmpeg 时回退 cv2 并写日志；.webm 走 VP8。"""
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("  skip 无 cv2/numpy：跳过 VideoSink")
        return
    import tempfile
    tmp = tempfile.mkdtemp(prefix="vl-sink-", dir=os.environ.get("TMPDIR"))
    frames = [np.full((72, 128, 3), (10 * i, 50, 200), np.uint8) for i in range(6)]

    def run(path, **kw):
        logs = []
        sink = vl.VideoSink(cv2, path, 10, (128, 72), 22, log=logs.append, **kw)
        assert sink.isOpened()
        for f in frames:
            sink.write(f)
        backend = sink.release()
        cap = cv2.VideoCapture(path)
        n = 0
        while cap.read()[0]:
            n += 1
        cap.release()
        return backend, n, logs

    backend, n, logs = run(os.path.join(tmp, "fallback.mp4"), ffmpeg="")
    assert backend.startswith("cv2 mp4v") and n == 6 and logs and "回退" in logs[0], (backend, n, logs)
    backend, n, logs = run(os.path.join(tmp, "vp8.webm"))
    assert backend.startswith("cv2 VP80") and n == 6 and not logs, (backend, n, logs)
    exe = vl.find_ffmpeg()
    if exe:
        backend, n, logs = run(os.path.join(tmp, "x264.mp4"))
        with open(os.path.join(tmp, "x264.mp4"), "rb") as f:
            head = f.read(4096)
        assert backend == "ffmpeg libx264 crf 22" and n == 6 and not logs and b"avc1" in head, (backend, n, head[:64])
        print(f"  ok  VideoSink：libx264（{os.path.basename(exe)}）/ 回退 cv2 mp4v 有日志 / .webm VP8")
    else:
        print("  ok  VideoSink：回退 cv2 mp4v 有日志 / .webm VP8（本机没有 ffmpeg，libx264 路径未测）")
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)


def check_paste_pixels():
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("  skip 无 cv2/numpy：跳过 paste 像素检查")
        return
    canvas = np.zeros((vl.H, vl.W, 3), np.uint8)
    # 320x240 -> 等比放大 480x360，左下格居中：两侧各 80 px 黑边
    img = np.full((240, 320, 3), (10, 20, 200), np.uint8)
    vl.paste(cv2, canvas, img, *vl.CELL_BL, vl.CW, vl.CH, "", "")
    cell = canvas[vl.CH:, :vl.CW]
    assert (cell[:, 80:560] == (10, 20, 200)).all(axis=2).all(), "机载应放大到 480x360 居中"
    assert not cell[:, :80].any() and not cell[:, 560:].any(), "两侧应是黑边"
    assert not canvas[:vl.CH].any() and not canvas[vl.CH:, vl.CW:].any(), "不能画出格子"
    # 640x360 原生：逐像素原样（不重采样）
    native = np.random.default_rng(0).integers(0, 255, (vl.CH, vl.CW, 3), dtype=np.uint8)
    canvas[:] = 0
    vl.paste(cv2, canvas, native, *vl.CELL_TR, vl.CW, vl.CH, "", "")
    assert (canvas[:vl.CH, vl.CW:] == native).all(), "原生尺寸应逐像素相同"
    # 标签 + 角标：白字黑边，落在格子左上 / 左下
    canvas[:] = 0
    vl.paste(cv2, canvas, np.full((vl.CH, vl.CW, 3), 128, np.uint8), *vl.CELL_TL, vl.CW, vl.CH, "TOP", "sim 1.00s")
    top_left, bottom_left = canvas[:30, :60], canvas[vl.CH - 20:vl.CH, :90]
    assert (top_left == 255).all(axis=2).any() and (top_left == 0).all(axis=2).any(), "标签应有白字 + 黑描边"
    assert (bottom_left == 235).all(axis=2).any() and (bottom_left == 0).all(axis=2).any(), "角标应有白字 + 黑描边"
    # None -> NO FRAME（红字），格子其余保持黑
    canvas[:] = 0
    vl.paste(cv2, canvas, None, *vl.CELL_TL, vl.CW, vl.CH, "CHASE", "")
    assert (canvas == (0, 0, 255)).all(axis=2).any() and not canvas[:vl.CH, :vl.CW][:120].any()
    print("  ok  paste：320x240 放大居中、原生 640x360 逐像素、标签/角标描边、NO FRAME")


def main():
    check_grid()
    check_panel_layout()
    check_onboard_label()
    check_quality_and_scale()
    check_paste_pixels()
    check_video_sink()
    print("test_view_layout: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
