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
    assert vl.onboard_label(None) == "ONBOARD"
    assert vl.onboard_label(_Img()) == "ONBOARD 320x240 camera (model input is resized)"
    assert vl.onboard_label(_Img(), 59.8) == "ONBOARD 320x240 HFOV 60deg camera (model input is resized)"
    print("  ok  机载标题")


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
    check_paste_pixels()
    print("test_view_layout: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
