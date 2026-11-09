#!/usr/bin/env python3
"""三视角录像的公共版式：Gazebo（record_views.py）与 3DGS（sim/3dgs/video.py）只用这一份。

1280×720 四格：左上 CHASE/FOLLOW 640×360、右上 TOP 640×360、左下 ONBOARD 640×360（机载
320×240 等比放大到 480×360 居中，模型输入分辨率不变）、右下状态面板 640×360。
两域的 demo 短片要能直接拼接，所以网格、标签位置、面板行距、动作时间条都定义在这里，
两边只负责把各自的数据整理成 panel() 需要的几行文字 / 动作段。

画质档位（VIDEO_QUALITY）也在这里：high 档画布 2560×1440，版式按 s=2 整体放大（grid(s) /
paste(..., s) / panel(..., s)），显示视角按 s 倍渲染，机载 320×240 仍是模型输入；编码走 VideoSink
（ffmpeg libx264 -crf，没有 ffmpeg 回退 cv2）。入口脚本顶部各有一行 VIDEO_QUALITY，这里只认档位名。

写法沿用 record_views：cv2 作参数传入，测试用假 cv2 记录绘制调用即可，不需要真渲染；
叠字只用 ASCII（Hershey 字体画不了中文和「·」，会画成 ?）。
"""
import os
import shutil
import subprocess

W, H = 1280, 720
CW, CH = 640, 360                                      # 每格尺寸
CELL_TL, CELL_TR, CELL_BL, CELL_BR = (0, 0), (CW, 0), (0, CH), (CW, CH)
ACTIONS = ("SAFE_NEUTRAL", "FORWARD_SLOW", "YAW_LEFT", "YAW_RIGHT")
ACTION_BGR = ((200, 200, 200), (80, 200, 80), (230, 160, 40), (40, 160, 230))
WHITE = (235, 235, 235)

# 画质档位（BRIEF_VIZ_LAYOUT §6.2）：(画布, 显示视角渲染倍数 s, 3DGS 超采样, x264 CRF)。
# record_views.compose / compose_views_video / make_demo_reel.build 只认这张表；入口脚本顶部一行选档位名。
VIDEO_QUALITY = {
    "low": dict(canvas=(1280, 720), scale=1, ssaa=1, crf=28),      # 预览：最快、文件最小
    "medium": dict(canvas=(1280, 720), scale=1, ssaa=2, crf=22),   # 日常评测录像
    "high": dict(canvas=(2560, 1440), scale=2, ssaa=2, crf=18),    # demo：显示视角 2× 渲染，编码约 3× 慢
}
DEFAULT_QUALITY = "medium"
QUALITY_ENV = "UAVRGB_VIDEO_QUALITY"


def quality(name=None):
    """档位名 -> 参数表（含 name）。环境变量 UAVRGB_VIDEO_QUALITY 覆盖一切（入口脚本透传的也算），
    便于不改文件临时切档；没给就用 DEFAULT_QUALITY。"""
    name = os.environ.get(QUALITY_ENV) or name or DEFAULT_QUALITY
    if name not in VIDEO_QUALITY:
        raise ValueError(f"未知画质档 {name!r}，可选 {sorted(VIDEO_QUALITY)}")
    return dict(VIDEO_QUALITY[name], name=name)


def grid(s=1):
    """s 倍画布的网格：(W, H, CW, CH, TL, TR, BL, BR)。s=1 就是模块常量。"""
    cw, ch = CW * s, CH * s
    return W * s, H * s, cw, ch, (0, 0), (cw, 0), (0, ch), (cw, ch)


def onboard_label(img, hfov_deg=None, display=1, rendered_at=None):
    """机载面板标题写相机实际分辨率和水平 FOV（取自本集 world SDF / scene.json，不是硬件文档值：
    e1_mini.json camera_fov 是 64.9°，模板仍是 X3 的 60°，写出来才看得见这个差异）。
    以前写 "(model input)"：画面是 320x240 的相机帧，模型吃的是 student.py 再缩放后的
    输入（本仓库 160x120），录像里拿不到那个尺寸，不冒充。
    display>1：high 档，机载格按 display 倍显示，标题仍写模型输入的 320x240。rendered_at 是 img 相对
    模型输入的渲染倍数：3DGS 为显示现渲 2× 帧时等于 display；Gazebo 的 img 是原生帧（1，默认），由 paste 放大。"""
    if img is None:
        return "ONBOARD"
    fov = f" HFOV {hfov_deg:.0f}deg" if hfov_deg else ""
    r = rendered_at or 1
    w, h = img.shape[1] // r, img.shape[0] // r
    if display > 1:
        return f"ONBOARD {w}x{h}{fov} (model input; display {display}x)"
    return f"ONBOARD {w}x{h}{fov} camera (model input is resized)"


def _text(cv2, canvas, s, org, scale, color, thickness, k=1):
    """白字先垫一圈 k px 黑描边：3DGS 的天空/白墙和 Gazebo 顶视的浅灰地面上，裸白字读不出来。
    描边用 8 个方向各偏 k px 画黑字，不用加粗 thickness：cv2 5.0 的 putText 走内置 TTF 字体，
    thickness 是字重，+2 几乎不变宽，盖不出边。"""
    x, y = org
    for dx, dy in ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)):
        cv2.putText(canvas, s, (x + dx * k, y + dy * k), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness)
    cv2.putText(canvas, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def paste(cv2, canvas, img, x, y, w, h, label, note="", note_color=WHITE, s=1):
    """按比例缩放居中贴到 (x,y,w,h)，左上写标签；没有帧时写 NO FRAME；note 写左下角（帧龄/时间）。
    s 是画布倍数（grid(s)），字号、描边、边距随之放大。放大用 INTER_CUBIC（机载 320×240 放到格子里
    是放大，LINEAR 糊），缩小用 INTER_AREA。"""
    if img is None:
        cv2.putText(canvas, label + ": NO FRAME", (x + 20 * s, y + h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (0, 0, 255), 2 * s)
        return
    ih, iw = img.shape[:2]
    k = min(w / iw, h / ih)
    nw, nh = int(iw * k), int(ih * k)
    ox, oy = x + (w - nw) // 2, y + (h - nh) // 2
    if (nw, nh) == (iw, ih):
        canvas[oy:oy + nh, ox:ox + nw] = img          # 原生尺寸的格子不重采样
    else:
        canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(
            img, (nw, nh), interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
    _text(cv2, canvas, label, (x + 8 * s, y + 22 * s), 0.6 * s, (255, 255, 255), 2 * s, s)
    if note:
        _text(cv2, canvas, note, (x + 8 * s, y + h - 8 * s), 0.45 * s, note_color or WHITE, 1 * s, s)


def panel(cv2, canvas, x, y, lines, action, flags, segments, t_rel, span, result, s=1):
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
    s         画布倍数：所有像素偏移、字号、线宽乘 s
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, text in enumerate(lines):
        cv2.putText(canvas, text, (x + 16 * s, y + (34 + 30 * i) * s), font, 0.65 * s, WHITE, 2 * s)
    ay = y + (34 + 30 * len(lines) + 10) * s
    if action is not None:
        cv2.rectangle(canvas, (x + 16 * s, ay - 24 * s), (x + 330 * s, ay + 10 * s), ACTION_BGR[action], -1)
        cv2.putText(canvas, ACTIONS[action], (x + 24 * s, ay), font, 0.8 * s, (0, 0, 0), 2 * s)
    for i, (text, color) in enumerate(flags):
        cv2.putText(canvas, text, (x + 16 * s, ay + (50 + 34 * i) * s), font, 0.7 * s, color, 2 * s)
    bx0, bx1, by = x + 16 * s, x + (CW - 16) * s, y + 330 * s
    for st, en, a in segments:
        u0, u1 = bx0 + int((bx1 - bx0) * st / span), bx0 + int((bx1 - bx0) * en / span)
        cv2.rectangle(canvas, (u0, by - 14 * s), (max(u1, u0 + 1), by), ACTION_BGR[a], -1)
    u = bx0 + int((bx1 - bx0) * t_rel / span)
    cv2.line(canvas, (u, by - 20 * s), (u, by + 4 * s), (255, 255, 255), 2 * s)
    if result:
        ok = result.upper().startswith("SUCCESS")
        cv2.putText(canvas, result, (x + 16 * s, y + 290 * s), font, 1.1 * s,
                    (80, 220, 80) if ok else (0, 0, 255), 3 * s)


# ---------------------------------------------------------------- 编码
def fourcc_for(path):
    """cv2 回退编码：.webm 用 VP8（浏览器直接能放）；其他用 mp4v（浏览器放不了）。
    本机 cv2 5.0.0 没有 H.264 编码器；VP90 在 1280x720 下约 190 ms/帧，太慢。"""
    return "VP80" if str(path).lower().endswith(".webm") else "mp4v"


def find_ffmpeg():
    """ffmpeg 可执行文件：环境变量 UAVRGB_FFMPEG > PATH 上的 ffmpeg > 本解释器里 imageio-ffmpeg 自带的
    静态 ffmpeg（tools/requirements.txt；3dgs-venv 也装了）；都没有返回 None。"""
    exe = os.environ.get("UAVRGB_FFMPEG") or shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - 没装或二进制损坏都视为没有 ffmpeg
        return None


class VideoSink:
    """帧 -> 视频文件，两域与短片共用。

    .mp4：ffmpeg libx264 -crf <档位> -preset slow -pix_fmt yuv420p（通用播放，码率随画质档）；帧经 stdin
    rawvideo 管道送入，不落 PNG：2560×1440 一帧 PNG 约 4 MB，一集 1100 帧要 4 GB 临时盘。没有 ffmpeg 时
    回退 cv2 mp4v，并在 log 写明「码率不可控」。.webm 维持 cv2 VP8（旧调用方、浏览器直播放）。
    用法与 cv2.VideoWriter 相同：isOpened / write / release；backend 字符串写进日志或 manifest。
    """

    def __init__(self, cv2, out_path, fps, size, crf, log=print, ffmpeg=None):
        self.out_path, self.proc, self.writer = str(out_path), None, None
        w, h = int(size[0]), int(size[1])
        exe = ffmpeg if ffmpeg is not None else find_ffmpeg()
        if self.out_path.lower().endswith(".mp4") and exe:
            cmd = [exe, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-preset", "slow",
                   "-crf", str(int(crf)), "-pix_fmt", "yuv420p", "-movflags", "+faststart", self.out_path]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.PIPE)
            self.backend = f"ffmpeg libx264 crf {int(crf)}"
        else:
            fourcc = fourcc_for(self.out_path)
            self.writer = cv2.VideoWriter(self.out_path, cv2.VideoWriter_fourcc(*fourcc), float(fps), (w, h))
            if not self.writer.isOpened():
                raise RuntimeError(f"cv2.VideoWriter 打不开 {self.out_path}（fourcc={fourcc}）")
            if self.out_path.lower().endswith(".mp4"):
                self.backend = "cv2 mp4v (fallback: no ffmpeg, bitrate uncontrolled)"
                if log:
                    log(f"[VideoSink] 没找到 ffmpeg，{os.path.basename(self.out_path)} 回退 cv2 mp4v，码率不可控；"
                        f"装法见 tools/requirements.txt（imageio-ffmpeg）")
            else:
                self.backend = f"cv2 {fourcc} (crf not applicable)"

    def isOpened(self):  # noqa: N802 - 与 cv2.VideoWriter 同名
        return (self.proc is not None and self.proc.poll() is None) or (self.writer is not None and self.writer.isOpened())

    def write(self, frame):
        if self.proc is not None:
            self.proc.stdin.write(frame.tobytes())
        else:
            self.writer.write(frame)

    def release(self):
        if self.proc is not None:
            self.proc.stdin.close()
            err = self.proc.stderr.read().decode("utf-8", "replace")
            rc = self.proc.wait()
            self.proc = None
            if rc != 0:
                raise RuntimeError(f"ffmpeg 退出 rc={rc} 写 {self.out_path}：{err.strip()[-500:]}")
        elif self.writer is not None:
            self.writer.release()
            self.writer = None
        return self.backend
