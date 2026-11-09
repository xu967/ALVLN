#!/usr/bin/env python3
"""把多段闭环视频拼成一条带标题卡、章节卡、字幕和结尾对照表的 demo 短片（给老师看的那条）。

运行（仓库根）：
  .venv-tools/bin/python tools/make_demo_reel.py                  # 用下面 SECTIONS 配置，输出 OUT（完整版）+ SHORT_OUT（60 s 精简版）+ 代表帧 STILLS_DIR
  .venv-tools/bin/python tools/make_demo_reel.py --self-check     # 合成小视频跑一遍：截取/字幕/恒等/编码，三档画质各出 2 s 小片并打印大小/分辨率
改素材、字幕、起止秒只改配置区；不改函数。输出旁边写 <OUT>_manifest.json（每段来源、起止秒、帧数、画质档、编码器），
视频里的任何一段都能对回原始集目录。

画质档 VIDEO_QUALITY（sim/tools/view_layout.VIDEO_QUALITY；--video-quality / 环境变量 UAVRGB_VIDEO_QUALITY 覆盖）决定画布
（high 2560×1440）与 CRF；标题卡 / 字幕按倍率 S 原生画，不是放大。素材约定：Gazebo（record_views.py）与 3DGS（sim/3dgs/video.py）
都是 view_layout 的四格版式：同画布尺寸的逐像素进片（fit_frame 恒等），1280×720 素材上 2560×1440 画布用 INTER_CUBIC 放大。
字幕叠在左下 ONBOARD 格的底部 64·S px（只盖住机载画面下沿和 sim 时间角标），不遮右下面板的动作时间条。
编码走 view_layout.VideoSink：ffmpeg libx264 -crf（imageio-ffmpeg 自带的静态 ffmpeg，tools/requirements.txt），没有就回退 cv2 mp4v。
cv2.putText 不支持中文，字幕用 PIL + Noto CJK 画。
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "sim", "tools"))
import view_layout as vl  # noqa: E402

# ───────────────────────────── 配置区（改这里） ─────────────────────────────
VIDEO_QUALITY = "high"   # low=1280×720/CRF 28（预览）；medium=1280×720/CRF 22；high=2560×1440 画布、显示视角 2× 渲染、CRF 18（demo 用；编码约 3× 慢）
DATA = os.environ.get("DATA_ROOT", "/root/rivermind-data") + "/uav_rgb"
OUT = os.environ.get("REEL_OUT", f"{DATA}/demo-e1-20261107/reel/demo_reel_v2.mp4")
SHORT_OUT = os.path.splitext(OUT)[0] + "_short.mp4"                         # 60 s 精简版（SHORT_SECTIONS）
STILLS_DIR = os.path.splitext(OUT)[0].replace("demo_reel_", "stills_")   # 代表帧 PNG：reel/stills_v2/
FPS = 10                           # 素材都是 10 fps（record_views.FPS / 3DGS VIEW_FPS）；不插帧
CARD_S = 3.0                       # 标题卡停留秒数
SECTION_S = 2.0                    # 章节卡停留秒数（代替 v0 每段前 1.5 s 的说明卡：22 段会吃掉 33 s）
END_S = 6.0                        # 结尾对照表停留秒数（5 行表 + 口径 + 脚注，要读得完）
CAPTION_H = 64                     # 字幕条高度（×S）：两行（中 22 px / 英 17 px）；叠在左下格底部
CAPTION_W = 640                    # = view_layout.CW（×S）；只盖 ONBOARD 格，右下面板的时间条不被遮
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"   # fc-list :lang=zh 查到的；没有就退回 PIL 默认字体（中文会是方框）
TITLE = ["TinyAerialVLN · E1 MINI 固件在环仿真", "真实场景 3DGS × Gazebo 柱阵/森林 · 教师与学生闭环三视角",
         "v2 · 2026-11-07 · SITL 库 v3 · 所有参数 calibrated=false（E1 参数为硬件草案值）"]
# 结尾卡：n=20 五组对照，数字来自 $DATA/e1-pipeline-20261107-0417-eval3/summary.md（SITL 库 v3、新 test seed 集，15:40 出）。
# 片中柱阵片段仍是 eval2（旧库）的录像：eval3 的录像没有新机体 / 追尾机位。
END_TITLE = "Gazebo 柱阵 · 20 seed 五组对照（E1 机体 cf_sitl · SITL 库 v3 · hold 1.0 s · observation 语义）"
END_HEADER = ("组", "成功 / 20", "碰撞 / 20")
END_ROWS = (("教师 A*（柱子真值）", "20", "0"), ("新 E1 学生", "10", "2"), ("新 E1 学生 · 成功线截断重训", "13", "5"),
            ("旧 X3 学生（载具迁移）", "4", "11"), ("盲前进", "2", "18"))
END_NOTES = ["数字：e1-pipeline-20261107-0417-eval3（SITL 库 v3、新 test seed）；片中柱阵片段为 eval2 录像（旧库，seed 不同）",
             "口径：SR 分母 = 请求集 20（失败集计入分母）；五组同一批 test seed；详见 docs/EVAL_STANDARD.md",
             "所有数字 calibrated=false：E1 参数为硬件草案值，尚未用真机数据标定"]
E2 = f"{DATA}/e1-pipeline-20261107-0417-eval2"
FOREST = f"{DATA}/demo-e1-20261107/gz-forest-4103-v3"                # 16:03 用当前模板 + 追尾补丁重录（tmp/agent6-viz-layout/gz-forest-4103）
C3 = f"{DATA}/demo-e1-20261107/3dgs-callinstrasse-routes-v3-fov60/collect"
RC = f"{DATA}/demo-e1-20261107/3dgs-recollect-v3"                    # SITL 库 v3 重采的 3 集（原 ep_00000 末步 dynamics_anomaly）
WL = f"{DATA}/tmp/agent4-scene-wanderland/import/smoke"
M360 = f"{DATA}/3dgs-smoke-20261107"
V = "views_v2.mp4"                                                   # 3DGS 高档重合成（着色机体、1.1 m 跟随机位）
# 每段：(视频路径, (start_s, end_s), 中文字幕, 英文字幕, 代表帧 (段内秒数, 文件名) 或 None)
#   start_s 为负表示从片尾倒数（-8 = 最后 8 s，能看到到达 / 碰撞的结果定格）；end_s None = 到片尾。每段 ≤ 20 s。
#   Gazebo 起止秒按 tmp/agent6-viz-layout/pick/*_top_sheet.png 挑：柱阵穿越段；3DGS 取结尾 8 s（含 1.5 s 到达定格）。
SECTIONS = [   # (章节卡两行, [段...])
    (["Gazebo 柱阵 · 教师闭环（柱子真值 A* + E1 固件在环）", "Gazebo pillar field · teacher A* + E1 SITL"], [
        # 柱阵 base 任务的成功判据是 route_choice：越过目标线 x = 6 m 且 0 碰撞（eval.json success / max_x）
        (f"{E2}/eval-teacher/videos/seed_127.mp4", (16, 31), "教师 · seed 127 · 近直线穿过柱阵 · 越过目标线 x=6 m · 0 碰撞", "teacher · seed 127 · near-straight pass · crossed the goal line, no collision", (8, "gz_teacher_seed127")),
        (f"{E2}/eval-teacher/videos/seed_312.mp4", (18, 33), "教师 · seed 312 · 先左绕再穿柱缝 · 越过目标线 · 0 碰撞", "teacher · seed 312 · detour left then through the gap · crossed the goal line", None),
    ]),
    (["Gazebo 柱阵 · 新 E1 学生（单目 RGB → 四动作 CNN）", "Gazebo pillar field · new E1 student (monocular RGB CNN)"], [
        (f"{E2}/eval-student-e1-new/videos/seed_1.mp4", (22, 37), "学生 · seed 1 · 成功 · 沿柱缝直穿越过目标线 · 0 碰撞", "student · seed 1 · success · straight through the gap, no collision", (8, "gz_student_seed1_success")),
        (f"{E2}/eval-student-e1-new/videos/seed_190.mp4", (22, 37), "学生 · seed 190 · 成功 · 柱间蛇形通过 · 0 碰撞", "student · seed 190 · success · weaves between pillars, no collision", None),
        (f"{E2}/eval-student-e1-new/videos/seed_113.mp4", (-15, None), "学生 · seed 113 · 失败 · 第 24 步撞柱（COLLISION）", "student · seed 113 · failure · hits a pillar at step 24", (13, "gz_student_seed113_collision")),
    ]),
    (["Gazebo 随机圆柱森林 forest_e1（sparse）· 教师 · SITL 库 v3 重录", "Gazebo random cylinder forest · teacher · re-recorded with SITL lib v3"], [
        # 05:04 旧库的同一 seed 是 80 步绕林 0 碰撞；库 v3 下教师第 27 步撞树（x≈4.2 m）。追尾机位：后 1.0 m / 上 0.45 m / HFOV 60°，机体约 95 px（原生 640 宽）
        (f"{FOREST}/videos/seed_4103.mp4", (-15, None), "森林 · seed 4103 · 教师 · 库 v3 下第 27 步撞树（COLLISION）· 追尾机位 1.0 m 可见 E1 机体", "forest · seed 4103 · teacher · hits a tree at step 27 under SITL lib v3 · E1 body visible from the 1.0 m chase cam", (8, "gz_forest_seed4103")),
    ]),
    # 3DGS 段落：步数 / 到达 来自各集 episode_manifest.json（num_steps / stopped_on_goal）。
    # calli route_5、wanderland route_1 / route_5 用 SITL 库 v3 重采的 ep_00000（RC），三集都到达、无 dynamics_anomaly。
    (["3DGS 真实校园 Callinstrasse（LUH，MIT）· 六路线 · 教师 A* + E1 固件在环", "3DGS real campus Callinstrasse · 6 routes · teacher A* + E1 SITL"], [
        (f"{C3}/route_0/ep_00000/{V}", (-8, None), "Callinstrasse 路线 0 · 走廊直线 8 m · 57 步到达", "Callinstrasse route 0 · straight corridor 8 m · reached in 57 steps", (4, "3dgs_callinstrasse_route0")),
        (f"{C3}/route_1/ep_00000/{V}", (-8, None), "路线 1 · 出生正对树篱，左转 60–70° 再沿走廊 · 41 步到达", "route 1 · spawns facing the hedge, turns left · reached in 41 steps", None),
        (f"{C3}/route_2/ep_00000/{V}", (-8, None), "路线 2 · 贴树丛右偏，穿走廊进广场 12.2 m · 82 步到达", "route 2 · dogleg past the trees into the plaza · reached in 82 steps", None),
        (f"{C3}/route_3/ep_00000/{V}", (-8, None), "路线 3 · 沿街东行到路口左转向北 11.9 m · 82 步到达", "route 3 · east to the crossing, turn left north · reached in 82 steps", None),
        (f"{C3}/route_4/ep_00000/{V}", (-8, None), "路线 4 · 到路口右转向南 11.9 m · 84 步到达", "route 4 · to the crossing, turn right south · reached in 84 steps", None),
        (f"{RC}/callinstrasse_route_5/ep_00000/{V}", (-8, None), "路线 5 · 西侧开阔地 S 弯汇入走廊 13.2 m · 101 步到达（库 v3 重采）", "route 5 · S-bend from the west field into the corridor · reached in 101 steps (SITL lib v3)", (4, "3dgs_callinstrasse_route5_v3")),
    ]),
    (["3DGS 真实公园 Wanderland（NYU ai4ce，Apache-2.0）· 六路线", "3DGS real park Wanderland · 6 routes"], [
        (f"{WL}/route_0/ep_00000/{V}", (-8, None), "Wanderland 路线 0 · 沿步道东行 12 m · 88 步到达", "Wanderland route 0 · along the path east 12 m · reached in 88 steps", (4, "3dgs_wanderland_route0")),
        (f"{RC}/wanderland_route_1/ep_00000/{V}", (-8, None), "路线 1 · 树丛与栅栏之间的夹道 14 m · 91 步到达（库 v3 重采）", "route 1 · lane between the trees and the fence · reached in 91 steps (SITL lib v3)", None),
        (f"{WL}/route_2/ep_00000/{V}", (-8, None), "路线 2 · 出生正对栅栏，左转 90° 再东行 · 93 步到达", "route 2 · spawns facing the fence, turns left 90° · reached in 93 steps", None),
        (f"{WL}/route_3/ep_00000/{V}", (-8, None), "路线 3 · 反向西行 14 m · 108 步到达", "route 3 · reverse, heading west 14 m · reached in 108 steps", None),
        (f"{WL}/route_4/ep_00000/{V}", (-8, None), "路线 4 · 出生正对树丛，右转 90° 走夹道 · 69 步到达", "route 4 · spawns facing a tree, turns right 90° · reached in 69 steps", None),
        (f"{RC}/wanderland_route_5/ep_00000/{V}", (-8, None), "路线 5 · 步道末端向南的弯 · 73 步到达（库 v3 重采）", "route 5 · bend to the south at the path's end · reached in 73 steps (SITL lib v3)", None),
    ]),
    (["3DGS Mip-NeRF 360 garden / bicycle（CC-BY 4.0 照片 · Inria 预训练 3DGS）", "3DGS Mip-NeRF 360 garden / bicycle"], [
        (f"{M360}/garden/r0/ep_00000/{V}", (-8, None), "garden 路线 0 · 南侧直线，桌与树在左手 · 50 步到达", "garden route 0 · south lane, table and tree on the left · reached in 50 steps", (4, "3dgs_garden_route0")),
        (f"{M360}/garden/r1/ep_00000/{V}", (-8, None), "garden 路线 1 · 西南→东北，绕树南侧 · 85 步到达", "garden route 1 · SW to NE around the tree · reached in 85 steps", None),
        (f"{M360}/bicycle/r0/ep_00000/{V}", (-8, None), "bicycle 路线 0 · 西→东沿小径，长椅在左前 · 57 步到达", "bicycle route 0 · west to east along the path · reached in 57 steps", (4, "3dgs_bicycle_route0")),
        (f"{M360}/bicycle/r1/ep_00000/{V}", (-8, None), "bicycle 路线 1 · 从长椅南侧经过 · 77 步到达", "bicycle route 1 · past the bench on its south side · reached in 77 steps", None),
    ]),
]
# 60 s 精简版：每域 1–2 段，各 8 s，不放章节卡（章节行为空就不画），标题 3 s + 6×8 s + 结尾表 6 s = 57 s
SHORT_SECTIONS = [
    ([], [
        (f"{E2}/eval-teacher/videos/seed_127.mp4", (20, 28), "Gazebo 柱阵 · 教师 A* · seed 127 · 越过目标线 · 0 碰撞", "Gazebo pillars · teacher A* · seed 127 · crossed the goal line", None),
        (f"{E2}/eval-student-e1-new/videos/seed_1.mp4", (26, 34), "Gazebo 柱阵 · 新 E1 学生（单目 RGB CNN）· seed 1 · 成功", "Gazebo pillars · new E1 student (RGB CNN) · seed 1 · success", None),
        (f"{FOREST}/videos/seed_4103.mp4", (-8, None), "Gazebo 森林 · 教师 · seed 4103 · 库 v3 下第 27 步撞树", "Gazebo forest · teacher · seed 4103 · hits a tree at step 27 (SITL lib v3)", None),
        (f"{RC}/callinstrasse_route_5/ep_00000/{V}", (-8, None), "3DGS Callinstrasse 路线 5 · S 弯 13.2 m · 101 步到达", "3DGS Callinstrasse route 5 · S-bend 13.2 m · reached in 101 steps", None),
        (f"{RC}/wanderland_route_1/ep_00000/{V}", (-8, None), "3DGS Wanderland 路线 1 · 夹道 14 m · 91 步到达", "3DGS Wanderland route 1 · lane 14 m · reached in 91 steps", None),
        (f"{M360}/garden/r0/ep_00000/{V}", (-8, None), "3DGS Mip-NeRF 360 garden 路线 0 · 50 步到达", "3DGS Mip-NeRF 360 garden route 0 · reached in 50 steps", None),
    ]),
]
# ──────────────────────────────────────────────────────────────────────────

Q = S = CANVAS_W = CANVAS_H = None   # set_quality() 填：画质档、倍率、画布尺寸（卡片 / 字幕 / fit_frame 都按它们算）


def set_quality(name=None):
    """按画质档设画布与倍率；返回档位表。环境变量 UAVRGB_VIDEO_QUALITY 覆盖 name（view_layout.quality 的约定）。"""
    global Q, S, CANVAS_W, CANVAS_H
    Q = vl.quality(name)
    S = Q["scale"]
    CANVAS_W, CANVAS_H = Q["canvas"]
    return Q


set_quality(VIDEO_QUALITY)


def font(size):
    try:
        return ImageFont.truetype(FONT, size * S)
    except OSError:
        return ImageFont.load_default()


def fit_font(d, lines, size, max_w=None):
    """最长一行放不下就逐级缩小字号（下限 18）：长标题溢出画布两侧会被裁掉，缩字比裁字好。"""
    max_w = CANVAS_W - 40 * S if max_w is None else max_w
    f = font(size)
    while size > 18 and max(d.textbbox((0, 0), s, font=f)[2] for s in lines) > max_w:
        size -= 2
        f = font(size)
    return f


def text_card(lines, size=44):
    """整屏居中的多行文字卡（BGR）。"""
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), (18, 18, 22))
    d = ImageDraw.Draw(img)
    f = fit_font(d, lines, size)
    heights = [d.textbbox((0, 0), s, font=f)[3] for s in lines]
    y = (CANVAS_H - sum(heights) - 18 * S * (len(lines) - 1)) // 2
    for s, h in zip(lines, heights):
        w = d.textbbox((0, 0), s, font=f)[2]
        d.text(((CANVAS_W - w) // 2, y), s, font=f, fill=(235, 235, 235))
        y += h + 18 * S
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def table_card(title, header, rows, notes):
    """结尾对照表：标题 + 表头 + 若干行 + 脚注（BGR）。列宽按最长单元格算，表居中。"""
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), (18, 18, 22))
    d = ImageDraw.Draw(img)
    f_title, f_cell, f_note = fit_font(d, [title], 34), font(30), fit_font(d, notes, 22)
    d.text(((CANVAS_W - d.textbbox((0, 0), title, font=f_title)[2]) // 2, 70 * S), title, font=f_title, fill=(235, 235, 235))
    cols = [max(d.textbbox((0, 0), r[i], font=f_cell)[2] for r in (header,) + tuple(rows)) + 60 * S for i in range(len(header))]
    x0, y = (CANVAS_W - sum(cols)) // 2, 160 * S
    for k, row in enumerate((header,) + tuple(rows)):
        x = x0
        color = (160, 200, 255) if k == 0 else (235, 235, 235)
        for i, cell in enumerate(row):
            w = d.textbbox((0, 0), cell, font=f_cell)[2]
            d.text((x + (cols[i] - w) // 2 if i else x + 20 * S, y), cell, font=f_cell, fill=color)
            x += cols[i]
        y += 52 * S
        if k == 0:
            d.line([(x0, y - 8 * S), (x0 + sum(cols), y - 8 * S)], fill=(120, 120, 130), width=2 * S)
    y += 20 * S
    for s in notes:
        d.text(((CANVAS_W - d.textbbox((0, 0), s, font=f_note)[2]) // 2, y), s, font=f_note, fill=(200, 200, 205))
        y += 36 * S
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def fit_frame(frame):
    """素材放到画布上：同尺寸原样返回（统一版式素材逐像素进片）；其他尺寸等比缩放居中留黑边
    （放大 INTER_CUBIC：1280×720 的 Gazebo 评测录像上 2560×1440 画布；缩小 INTER_AREA）。"""
    h, w = frame.shape[:2]
    if (w, h) == (CANVAS_W, CANVAS_H):
        return frame
    k = min(CANVAS_W / w, CANVAS_H / h)
    nw, nh = int(w * k), int(h * k)
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
    x0, y0 = (CANVAS_W - nw) // 2, (CANVAS_H - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
    return canvas


def caption(frame, zh, en=""):
    """字幕条：半透明深底叠在左下格底部（0..CAPTION_W·S, CANVAS_H-CAPTION_H·S..CANVAS_H），中文一行、英文一行（可空）。"""
    if not zh and not en:
        return frame
    y0, cw = CANVAS_H - CAPTION_H * S, CAPTION_W * S
    bar = frame[y0:, :cw]
    bar[:] = (bar.astype(np.uint16) * 3 // 10 + np.array((22, 22, 26), np.uint16) * 7 // 10).astype(np.uint8)
    img = Image.fromarray(cv2.cvtColor(bar, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    if zh:
        d.text((12 * S, 6 * S), zh, font=font(22), fill=(240, 240, 240))
    if en:
        d.text((12 * S, (36 if zh else 18) * S), en, font=font(17), fill=(200, 200, 205))
    frame[y0:, :cw] = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    return frame


def clip_range(path, window):
    """(start_s, end_s) -> (首帧号, 末帧号+1)；start 为负从片尾倒数，end None 到片尾。超界截到 [0, n]。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"读不了 {path}")
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or FPS
    cap.release()
    start_s, end_s = window or (0, None)
    end = n if end_s is None else min(n, int(round(end_s * fps)))
    if start_s < 0:                       # 负数：从截取终点倒数
        start_s = end / fps + start_s
    start = max(0, min(int(round(start_s * fps)), end))
    if end - start > 20 * fps + 1:
        raise ValueError(f"{os.path.basename(path)} 截取 {(end - start) / fps:.1f} s > 20 s 上限（BRIEF：每段 ≤ 20 s）")
    return start, end


def seek(cap, path, start):
    """定位到第 start 帧：.mp4（H.264/mp4v）用 set(POS_FRAMES) 并核对落点；VP8 webm 的 set 落点不可靠，顺序丢帧
    （素材最长 1100 帧，顺序读只多花几秒）。返回当前帧号。"""
    if not path.lower().endswith(".webm") and start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) == start:
            return start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    pos = 0
    while pos < start and cap.grab():
        pos += 1
    return pos


def build(out_path, sections, title, end_card, stills_dir=None, card_s=None, section_s=None, end_s=None):
    """拼片：标题卡 + 每节（章节卡可空）+ 各段字幕 + 结尾卡；返回 manifest 字典（也写到 <out>_manifest.json）。
    card_s / section_s / end_s 为 None 用配置区常量（自检用 0 只出素材段）。"""
    card_s = CARD_S if card_s is None else card_s
    section_s = SECTION_S if section_s is None else section_s
    end_s = END_S if end_s is None else end_s
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if stills_dir:
        os.makedirs(stills_dir, exist_ok=True)
    writer = vl.VideoSink(cv2, out_path, FPS, (CANVAS_W, CANVAS_H), Q["crf"], log=print)
    if not writer.isOpened():
        raise RuntimeError(f"视频写不开 {out_path}（{writer.backend}）")
    manifest, stills, n_out = [], [], 0

    def emit(frame, n=1):
        nonlocal n_out
        for _ in range(n):
            writer.write(frame)
        n_out += n

    def still(frame, name):
        if stills_dir:
            path = os.path.join(stills_dir, f"{len(stills):02d}_{name}.png")
            cv2.imwrite(path, frame)
            stills.append(path)

    try:
        card = text_card(title)
        still(card, "title")
        emit(card, int(card_s * FPS))
        for section_lines, clips in sections:
            if section_lines:
                emit(text_card(section_lines, size=36), int(section_s * FPS))
            for path, window, zh, en, still_at in clips:
                start, end = clip_range(path, window)
                cap = cv2.VideoCapture(path)
                pos = seek(cap, path, start)
                reel_start, n_src = n_out, 0
                while pos < end:
                    ok, fr = cap.read()
                    if not ok:
                        break
                    fr = caption(fit_frame(fr), zh, en)
                    if still_at is not None and n_src == int(still_at[0] * FPS):
                        still(fr, still_at[1])
                    emit(fr)
                    n_src += 1
                    pos += 1
                cap.release()
                if n_src == 0:
                    raise RuntimeError(f"{path} 一帧都没读到（编码不支持或截取区间为空）")
                manifest.append({"src": path, "src_start_s": round(start / FPS, 1), "src_end_s": round((start + n_src) / FPS, 1),
                                 "frames": n_src, "reel_start_s": round(reel_start / FPS, 1),
                                 "reel_end_s": round(n_out / FPS, 1), "caption": zh, "caption_en": en})
        still(end_card, "end_table")
        emit(end_card, int(end_s * FPS))
    finally:
        backend = writer.release()
    side = {"out": out_path, "fps": FPS, "canvas": [CANVAS_W, CANVAS_H], "quality": Q["name"], "backend": backend,
            "frames": n_out, "duration_s": round(n_out / FPS, 1), "bytes": os.path.getsize(out_path),
            "clips": manifest, "stills": stills, "built": time.strftime("%F %T")}
    with open(os.path.splitext(out_path)[0] + "_manifest.json", "w", encoding="utf-8") as f:
        json.dump(side, f, ensure_ascii=False, indent=2)
    return side


def self_check():
    """两段合成小视频（960×276 旧尺寸、1280×720 统一版式）→ 截取拼接 → 帧数、尺寸、恒等、字幕区、代表帧都对；
    再按三档画质各出一条 2 s 小片（只有素材段），打印文件大小 / 分辨率 / 编码器。"""
    import tempfile
    tmp = tempfile.mkdtemp(prefix="reel-selfcheck-", dir=os.environ.get("TMPDIR"))
    srcs = []
    for i, (w, h, n_src) in enumerate(((960, 276, 40), (1280, 720, 220))):   # 第二段 22 s：用来验证 > 20 s 的截取上限
        p = os.path.join(tmp, f"clip{i}.mp4")
        wr = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        for k in range(n_src):
            fr = np.full((h, w, 3), 40 + (5 * k) % 200, np.uint8)
            cv2.circle(fr, (w // 2, h // 2), 30, (0, 0, 255), -1)
            wr.write(fr)
        wr.release()
        srcs.append(p)
    set_quality("medium")                    # 版式逻辑先在 1× 下验
    # 同尺寸恒等：fit_frame 不复制、不缩放
    same = np.random.default_rng(0).integers(0, 255, (CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    assert fit_frame(same) is same, "1280x720 输入应原样返回"
    small = fit_frame(np.full((276, 960, 3), 200, np.uint8))
    assert small.shape == (CANVAS_H, CANVAS_W, 3) and not small[:150].any() and small[CANVAS_H // 2].all()
    # 长标题缩字不裁字：50 个汉字的一行（44 px 放不下）与配置里的结尾表，左右各 20 px 仍是底色
    for card in (text_card(["标题" * 25]), table_card(END_TITLE, END_HEADER, END_ROWS, END_NOTES)):
        bg = card[0, 0]
        assert (card[:, :20] == bg).all() and (card[:, -20:] == bg).all(), "文字溢出画布"
    # 字幕只盖左下 640x64，其余像素不动
    base = np.full((CANVAS_H, CANVAS_W, 3), 100, np.uint8)
    cap_img = caption(base.copy(), "中文字幕", "english")
    assert (cap_img[:CANVAS_H - CAPTION_H] == 100).all() and (cap_img[:, CAPTION_W:] == 100).all()
    assert not (cap_img[CANVAS_H - CAPTION_H:, :CAPTION_W] == 100).all(), "字幕区应被改写"
    # 截取：(1.0, 2.5) 15 帧；(-1.0, None) 片尾 10 帧；整段 22 s / (0, 21) 都超过 20 s 上限要报错；超界截到片长
    assert clip_range(srcs[1], (1.0, 2.5)) == (10, 25) and clip_range(srcs[1], (-1.0, None)) == (210, 220)
    assert clip_range(srcs[0], None) == (0, 40) and clip_range(srcs[0], (-8, None)) == (0, 40)
    for bad in (None, (0, 21), (-21, None)):
        try:
            clip_range(srcs[1], bad)
            raise AssertionError(f"{bad} 超过 20 s 应报错")
        except ValueError:
            pass
    sections = [(["自检章节"], [(srcs[0], (1.0, 2.5), "字幕 0", "cap 0", (0.5, "clip0")), (srcs[1], (-1.0, None), "字幕 1", "", None)])]
    out = os.path.join(tmp, "reel.mp4")
    side = build(out, sections, ["自检标题"], table_card("t", ("a", "b"), (("1", "2"),), ["n"]), stills_dir=os.path.join(tmp, "stills"))
    cap = cv2.VideoCapture(out)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, fr = cap.read()
    cap.release()
    expect = int(CARD_S * FPS) + int(SECTION_S * FPS) + 15 + 10 + int(END_S * FPS)
    assert n == expect == side["frames"], (n, expect, side["frames"])
    assert ok and fr.shape == (CANVAS_H, CANVAS_W, 3), fr.shape
    assert [c["frames"] for c in side["clips"]] == [15, 10] and side["clips"][0]["src_start_s"] == 1.0
    assert side["clips"][1]["reel_start_s"] == side["clips"][0]["reel_end_s"]
    assert len(side["stills"]) == 3 and all(os.path.isfile(p) for p in side["stills"]), side["stills"]
    # mp4 定位：seek 到第 210 帧后读到的帧 = 顺序丢 210 帧后读到的帧（mp4v 有量化误差，不与生成值比）
    cap = cv2.VideoCapture(srcs[1])
    for _ in range(210):
        cap.grab()
    want = cap.read()[1]
    cap.release()
    cap = cv2.VideoCapture(srcs[1])
    assert seek(cap, srcs[1], 210) == 210 and np.array_equal(cap.read()[1], want), "mp4 seek 落点不对"
    cap.release()
    # 三档画质各出 2 s（20 帧，只有素材段）：分辨率 = 档位画布；.mp4 有 ffmpeg 就是 libx264 + 档位 CRF
    for tier in sorted(vl.VIDEO_QUALITY):
        q = set_quality(tier)
        if q["name"] != tier:            # 环境变量 UAVRGB_VIDEO_QUALITY 把三档都压成一档了，只验这一档
            print(f"  note {vl.QUALITY_ENV}={q['name']} 覆盖了 {tier}")
        p = os.path.join(tmp, f"tier_{tier}.mp4")
        side = build(p, [([], [(srcs[1], (0, 2.0), "字幕", "", None)])], ["t"], text_card(["e"]),
                     card_s=0, section_s=0, end_s=0)
        cap = cv2.VideoCapture(p)
        wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert wh == tuple(q["canvas"]) and nf == 20 == side["frames"], (tier, wh, nf, side["frames"])
        if vl.find_ffmpeg():
            assert side["backend"] == f"ffmpeg libx264 crf {q['crf']}", side["backend"]
        print(f"  {tier:6s} {wh[0]}x{wh[1]}  {side['bytes'] / 1024:7.1f} KiB  2.0 s  {side['backend']}")
    set_quality(VIDEO_QUALITY)
    print(f"make_demo_reel self-check PASS（{n} 帧，{tmp}）")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--out", default=OUT)
    p.add_argument("--video-quality", default=VIDEO_QUALITY, choices=sorted(vl.VIDEO_QUALITY),
                   help=f"画质档（默认配置区 {VIDEO_QUALITY}；环境变量 {vl.QUALITY_ENV} 覆盖）")
    p.add_argument("--no-short", action="store_true", help="不出 60 s 精简版")
    args = p.parse_args()
    if args.self_check:
        self_check()
        return
    q = set_quality(args.video_quality)
    missing = [c[0] for sections in (SECTIONS, SHORT_SECTIONS) for _s, clips in sections for c in clips if not os.path.isfile(c[0])]
    if missing:
        sys.exit("缺素材：\n  " + "\n  ".join(sorted(set(missing))))
    end_card = table_card(END_TITLE, END_HEADER, END_ROWS, END_NOTES)
    stills_dir = os.path.splitext(args.out)[0].replace("demo_reel_", "stills_") if args.out != OUT else STILLS_DIR
    jobs = [(args.out, SECTIONS, stills_dir)]
    if not args.no_short:
        jobs.append((os.path.splitext(args.out)[0] + "_short.mp4", SHORT_SECTIONS, None))
    for out, sections, stills in jobs:
        side = build(out, sections, TITLE, end_card, stills)
        print(json.dumps({k: side[k] for k in ("out", "quality", "backend", "frames", "duration_s", "bytes")}, ensure_ascii=False))
        for c in side["clips"]:
            print(f"  {c['reel_start_s']:6.1f}–{c['reel_end_s']:6.1f} s  {c['frames']:4d} 帧  "
                  f"src {c['src_start_s']:5.1f}–{c['src_end_s']:5.1f} s  {os.path.basename(os.path.dirname(c['src']))}/{os.path.basename(c['src'])}")
        if side["stills"]:
            print("stills:", *side["stills"], sep="\n  ")
    print(f"画质档 {q['name']}：画布 {CANVAS_W}x{CANVAS_H}，CRF {q['crf']}")


if __name__ == "__main__":
    main()
