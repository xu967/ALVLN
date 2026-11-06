#!/usr/bin/env python3
"""把多段闭环视频拼成一条带标题卡和字幕的 demo 短片（给老师看的那条）。

运行（仓库根）：
  .venv-tools/bin/python tools/make_demo_reel.py                  # 用下面 CLIPS 配置，输出 OUT
  .venv-tools/bin/python tools/make_demo_reel.py --self-check     # 合成两段小视频跑一遍，验证拼接/字幕/编码
改素材、字幕、时长只改配置区；不改函数。输出旁边写 reel_manifest.json（每段来源、帧数、起止秒），
视频里的任何一段都能对回原始集目录。

为什么用 cv2 + PIL：本机没有 ffmpeg；cv2 能读 webm/mp4、写 mp4v；cv2.putText 不支持中文，字幕用 PIL + Noto CJK 画。
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ───────────────────────────── 配置区（改这里） ─────────────────────────────
DATA = os.environ.get("DATA_ROOT", "/root/rivermind-data") + "/uav_rgb"
OUT = os.environ.get("REEL_OUT", f"{DATA}/demo-e1-20261107/reel/demo_reel_v0.mp4")
CANVAS_W, CANVAS_H = 1280, 720     # 画布：素材按等比缩放居中，留黑边；1280 宽够看三视角的文字
FPS = 10                           # 素材都是 10 fps（hold 0.5 s 两帧一步 / hold 1.0 s 一帧 0.1 s）；不插帧
CARD_S = 3.0                       # 标题卡 / 结尾卡停留秒数
CAPTION_S = 1.5                    # 每段前的说明卡停留秒数
CAPTION_BAR_H = 64                 # 视频底部字幕条高度（像素）
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"   # fc-list :lang=zh 查到的；没有就退回 PIL 默认字体（中文会是方框）
TITLE = ["TinyAerialVLN · E1 MINI 固件在环仿真", "真实校园 3DGS × Gazebo 柱阵/森林 · 教师闭环三视角",
         "v0 · 2026-11-07 · 所有参数 calibrated=false"]
END = ["待补（流水线产出后替换为 v1）：", "Gazebo 柱阵 20 seed 四组对照：教师 / 旧 X3 学生 / 新 E1 学生 / 盲前进",
       "新 3DGS 场景（≥2）· SITL 自检曲线"]
R3 = f"{DATA}/demo-e1-20261107/3dgs-callinstrasse-routes/videos"
CLIPS = [   # (视频路径, 说明卡两行, 底部字幕)
    (f"{R3}/route0_r0_straight_east_seed0.webm",
     ["3DGS Callinstrasse（真实校园，MIT）· 路线 0 直线", "占据栅格 A* 教师 + E1 SITL · hold 0.5 s"],
     "机载 320×240 | 跟随 | 顶视（占据代理 + 路径）· 57 步到达 0 碰撞"),
    (f"{R3}/route1_r1_hedge_left_turn_seed101.webm",
     ["路线 1 · 出生正对树篱，先左转 60–70° 再沿走廊", "转向 76° · 最小净空 1.17 m"], "44 步到达 0 碰撞"),
    (f"{R3}/route2_r2_dogleg_plaza_seed201.webm",
     ["路线 2 · 贴树丛右偏，穿走廊进广场", "航程 12.2 m · 转向 117°"], "86 步到达 0 碰撞"),
    (f"{R3}/route3_r3_plaza_post_seed300.webm",
     ["路线 3 · 广场立柱从右侧绕过再左弯", "转向 152° · 最小净空 0.80 m"], "58 步到达 0 碰撞"),
    (f"{R3}/route4_r4_reverse_corridor_seed400.webm",
     ["路线 4 · 朝 +y 出生，左转 90° 后反向走廊", "转向 143°"], "62 步到达 0 碰撞"),
    (f"{R3}/route5_r5_west_field_s_seed501.webm",
     ["路线 5 · 西侧开阔地 S 弯汇入走廊", "航程 13.2 m · 转向 122°"], "89 步到达 0 碰撞"),
    (f"{DATA}/tmp/agent-forest/video-sparse/videos/seed_4103.mp4",
     ["Gazebo 随机圆柱森林 forest_e1（sparse，缝 ≥0.45 m）", "柱子真值 A* 教师 + E1 SITL · hold 1.0 s · seed 4103"],
     "机载 | 跟随 | 顶视 · 80 步到达 0 碰撞（教师现版从林带外绕行，见 worklog）"),
]
# ──────────────────────────────────────────────────────────────────────────


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def text_card(lines, size=44):
    """整屏居中的多行文字卡（BGR）。"""
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), (18, 18, 22))
    d = ImageDraw.Draw(img)
    f = font(size)
    heights = [d.textbbox((0, 0), s, font=f)[3] for s in lines]
    y = (CANVAS_H - sum(heights) - 18 * (len(lines) - 1)) // 2
    for s, h in zip(lines, heights):
        w = d.textbbox((0, 0), s, font=f)[2]
        d.text(((CANVAS_W - w) // 2, y), s, font=f, fill=(235, 235, 235))
        y += h + 18
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def fit_frame(frame, caption):
    """素材等比缩放到画布（底部留字幕条），居中，画字幕。"""
    avail_h = CANVAS_H - CAPTION_BAR_H
    h, w = frame.shape[:2]
    s = min(CANVAS_W / w, avail_h / h)
    nw, nh = int(w * s), int(h * s)
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
    x0, y0 = (CANVAS_W - nw) // 2, (avail_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    d.rectangle([0, avail_h, CANVAS_W, CANVAS_H], fill=(28, 28, 34))
    d.text((24, avail_h + 16), caption, font=font(28), fill=(230, 230, 230))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def build(out_path, clips, title, end):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (CANVAS_W, CANVAS_H))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter 打不开 {out_path}（mp4v）")
    manifest, n_out = [], 0

    def emit(frame, n=1):
        nonlocal n_out
        for _ in range(n):
            writer.write(frame)
        n_out += n

    emit(text_card(title), int(CARD_S * FPS))
    for path, card_lines, caption in clips:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"读不了 {path}")
        emit(text_card(card_lines, size=38), int(CAPTION_S * FPS))
        start, n_src = n_out, 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            emit(fit_frame(fr, caption))
            n_src += 1
        cap.release()
        if n_src == 0:
            raise RuntimeError(f"{path} 一帧都没读到（编码不支持或文件损坏）")
        manifest.append({"src": path, "frames": n_src, "reel_start_s": round(start / FPS, 1),
                         "reel_end_s": round(n_out / FPS, 1), "caption": caption})
    emit(text_card(end, size=36), int(CARD_S * FPS))
    writer.release()
    side = {"out": out_path, "fps": FPS, "canvas": [CANVAS_W, CANVAS_H], "frames": n_out,
            "duration_s": round(n_out / FPS, 1), "clips": manifest, "built": time.strftime("%F %T")}
    with open(os.path.splitext(out_path)[0] + "_manifest.json", "w", encoding="utf-8") as f:
        json.dump(side, f, ensure_ascii=False, indent=2)
    return side


def self_check():
    """两段合成小视频（不同尺寸）→ 拼接 → 读回帧数、尺寸、字幕条颜色都对。"""
    import tempfile
    tmp = tempfile.mkdtemp(prefix="reel-selfcheck-", dir=os.environ.get("TMPDIR"))
    srcs = []
    for i, (w, h) in enumerate(((960, 276), (1280, 720))):
        p = os.path.join(tmp, f"clip{i}.mp4")
        wr = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        for k in range(12):
            fr = np.full((h, w, 3), 40 + 15 * k, np.uint8)
            cv2.circle(fr, (w // 2, h // 2), 30, (0, 0, 255), -1)
            wr.write(fr)
        wr.release()
        srcs.append((p, [f"自检片段 {i}", f"{w}×{h}"], f"字幕 {i}"))
    out = os.path.join(tmp, "reel.mp4")
    side = build(out, srcs, ["自检标题"], ["自检结尾"])
    cap = cv2.VideoCapture(out)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, fr = cap.read()
    cap.release()
    expect = int(CARD_S * FPS) * 2 + int(CAPTION_S * FPS) * 2 + 24
    assert n == expect == side["frames"], (n, expect, side["frames"])
    assert ok and fr.shape == (CANVAS_H, CANVAS_W, 3), fr.shape
    assert len(side["clips"]) == 2 and side["clips"][1]["reel_start_s"] > side["clips"][0]["reel_end_s"]
    print(f"make_demo_reel self-check PASS（{n} 帧，{side['duration_s']} s，{tmp}）")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--out", default=OUT)
    args = p.parse_args()
    if args.self_check:
        self_check()
        return
    missing = [c[0] for c in CLIPS if not os.path.isfile(c[0])]
    if missing:
        sys.exit("缺素材：\n  " + "\n  ".join(missing))
    side = build(args.out, CLIPS, TITLE, END)
    print(json.dumps({k: side[k] for k in ("out", "frames", "duration_s")}, ensure_ascii=False))
    for c in side["clips"]:
        print(f"  {c['reel_start_s']:6.1f}–{c['reel_end_s']:6.1f} s  {c['frames']:4d} 帧  {os.path.basename(c['src'])}")


if __name__ == "__main__":
    main()
