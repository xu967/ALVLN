#!/usr/bin/env python3
"""三视角录像的公共版式：Gazebo（record_views.py）与 3DGS（sim/3dgs/video.py）只用这一份。

1280×720 四格：左上 CHASE/FOLLOW 640×360、右上 TOP 640×360、左下 ONBOARD 640×360（机载
320×240 等比放大到 480×360 居中，模型输入分辨率不变）、右下状态面板 640×360。
两域的 demo 短片要能直接拼接，所以网格、标签位置、面板行距、动作时间条都定义在这里，
两边只负责把各自的数据整理成 panel() 需要的几行文字 / 动作段。

写法沿用 record_views：cv2 作参数传入，测试用假 cv2 记录绘制调用即可，不需要真渲染；
叠字只用 ASCII（Hershey 字体画不了中文和「·」，会画成 ?）。
"""
W, H = 1280, 720
CW, CH = 640, 360                                      # 每格尺寸
CELL_TL, CELL_TR, CELL_BL, CELL_BR = (0, 0), (CW, 0), (0, CH), (CW, CH)
ACTIONS = ("SAFE_NEUTRAL", "FORWARD_SLOW", "YAW_LEFT", "YAW_RIGHT")
ACTION_BGR = ((200, 200, 200), (80, 200, 80), (230, 160, 40), (40, 160, 230))
WHITE = (235, 235, 235)


def onboard_label(img, hfov_deg=None):
    """机载面板标题写相机实际分辨率和水平 FOV（取自本集 world SDF / scene.json，不是硬件文档值：
    e1_mini.json camera_fov 是 64.9°，模板仍是 X3 的 60°，写出来才看得见这个差异）。
    以前写 "(model input)"：画面是 320x240 的相机帧，模型吃的是 student.py 再缩放后的
    输入（本仓库 160x120），录像里拿不到那个尺寸，不冒充。"""
    if img is None:
        return "ONBOARD"
    fov = f" HFOV {hfov_deg:.0f}deg" if hfov_deg else ""
    return f"ONBOARD {img.shape[1]}x{img.shape[0]}{fov} camera (model input is resized)"


def _text(cv2, canvas, s, org, scale, color, thickness):
    """白字先垫一圈 1 px 黑描边：3DGS 的天空/白墙和 Gazebo 顶视的浅灰地面上，裸白字读不出来。
    描边用 8 个方向各偏 1 px 画黑字，不用加粗 thickness：cv2 5.0 的 putText 走内置 TTF 字体，
    thickness 是字重，+2 几乎不变宽，盖不出边。"""
    x, y = org
    for dx, dy in ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)):
        cv2.putText(canvas, s, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness)
    cv2.putText(canvas, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def paste(cv2, canvas, img, x, y, w, h, label, note="", note_color=WHITE):
    """按比例缩放居中贴到 (x,y,w,h)，左上写标签；没有帧时写 NO FRAME；note 写右下角（帧龄/时间）。"""
    if img is None:
        cv2.putText(canvas, label + ": NO FRAME", (x + 20, y + h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = int(iw * s), int(ih * s)
    ox, oy = x + (w - nw) // 2, y + (h - nh) // 2
    if (nw, nh) == (iw, ih):
        canvas[oy:oy + nh, ox:ox + nw] = img          # 原生尺寸的格子不重采样
    else:
        canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    _text(cv2, canvas, label, (x + 8, y + 22), 0.6, (255, 255, 255), 2)
    if note:
        _text(cv2, canvas, note, (x + 8, y + h - 8), 0.45, note_color or WHITE, 1)


def panel(cv2, canvas, x, y, lines, action, flags, segments, t_rel, span, result):
    """右下状态面板。

    lines     文本行（标题、时间、步数、位置、速度…），每行 30 px；两域都是 5 行，
              再多会和 COLLISION / 结果大字叠在一起
    action    当前动作 0..3（画色块 + 名字），None 不画
    flags     [(text, bgr)]：色块下方的告警行（COLLISION / STUCK），每行 34 px
    segments  [(start, end, action)]：底部动作时间条，每步一段颜色；start/end 从视频起点算，
              单位随调用方（Gazebo 用 ns，3DGS 用 s），与 t_rel / span 一致即可
    t_rel     当前时刻（从视频起点算），画白色竖线
    span      时间条总长（> 0）
    result    结尾大字；SUCCESS 开头绿色，其余红色；空串不画
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, s in enumerate(lines):
        cv2.putText(canvas, s, (x + 16, y + 34 + 30 * i), font, 0.65, WHITE, 2)
    ay = y + 34 + 30 * len(lines) + 10
    if action is not None:
        cv2.rectangle(canvas, (x + 16, ay - 24), (x + 330, ay + 10), ACTION_BGR[action], -1)
        cv2.putText(canvas, ACTIONS[action], (x + 24, ay), font, 0.8, (0, 0, 0), 2)
    for i, (text, color) in enumerate(flags):
        cv2.putText(canvas, text, (x + 16, ay + 50 + 34 * i), font, 0.7, color, 2)
    bx0, bx1, by = x + 16, x + CW - 16, y + 330
    for s, e, a in segments:
        u0, u1 = bx0 + int((bx1 - bx0) * s / span), bx0 + int((bx1 - bx0) * e / span)
        cv2.rectangle(canvas, (u0, by - 14), (max(u1, u0 + 1), by), ACTION_BGR[a], -1)
    u = bx0 + int((bx1 - bx0) * t_rel / span)
    cv2.line(canvas, (u, by - 20), (u, by + 4), (255, 255, 255), 2)
    if result:
        ok = result.upper().startswith("SUCCESS")
        cv2.putText(canvas, result, (x + 16, y + 290), font, 1.1,
                    (80, 220, 80) if ok else (0, 0, 255), 3)
