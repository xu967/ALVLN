#!/usr/bin/env python3
"""3DGS 回合帧 -> 视频；只做 onboard RGB，不依赖 Gazebo/record_views。"""
import glob
import os


def compose_rgb_video(frames_dir, out_path, fps=10.0):
    """把 frames_dir/rgb_*.png 按文件名顺序合成视频；返回实际写入的帧数。

    out_path 以 .webm 结尾用 VP8（浏览器可直接播放），否则用 mp4v（浏览器放不了）。
    本机 .venv-tools / 3dgs-venv 的 cv2 都没有 H.264 编码器。
    读不出或尺寸与首帧不同的帧会跳过，所以返回值可能小于 PNG 数。
    """
    import cv2

    paths = sorted(glob.glob(os.path.join(frames_dir, "rgb_*.png")))
    if not paths:
        raise RuntimeError(f"{frames_dir} 没有 rgb_*.png")
    first = cv2.imread(paths[0])
    if first is None:
        raise RuntimeError(f"无法读取 {paths[0]}")
    h, w = first.shape[:2]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fourcc = "VP80" if str(out_path).lower().endswith(".webm") else "mp4v"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc),
                             float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建 {out_path}（fourcc={fourcc}）")
    written = 0
    try:
        for path in paths:
            img = cv2.imread(path)
            if img is None or img.shape[:2] != (h, w):
                continue
            writer.write(img)
            written += 1
    finally:
        writer.release()
    return written


def _selftest():
    import json
    import tempfile

    import cv2
    import numpy as np

    with tempfile.TemporaryDirectory() as d:
        for i in range(3):
            img = np.zeros((24, 32, 3), np.uint8)
            img[:, :, 0] = i * 80
            cv2.imwrite(os.path.join(d, f"rgb_{i:05d}.png"), img)
        # 尺寸不同的帧被跳过：返回值必须是实际写入数，不是文件数
        cv2.imwrite(os.path.join(d, "rgb_00003.png"), np.zeros((12, 16, 3), np.uint8))
        out = os.path.join(d, "v.webm")
        n = compose_rgb_video(d, out, fps=2.0)
        cap = cv2.VideoCapture(out)
        frames = 0
        while cap.read()[0]:
            frames += 1
        cap.release()
        assert n == 3 and frames == 3, (n, frames)
        print(json.dumps({"3dgs_video": "PASS", "frames": frames, "files": 4}))


if __name__ == "__main__":
    _selftest()
