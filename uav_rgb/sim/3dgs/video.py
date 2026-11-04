#!/usr/bin/env python3
"""3DGS 回合帧 -> mp4；只做 onboard RGB，不依赖 Gazebo/record_views。"""
import glob
import os


def compose_rgb_video(frames_dir, out_path, fps=10.0):
    """把 frames_dir/rgb_*.png 按文件名顺序合成 mp4；返回帧数。"""
    import cv2

    paths = sorted(glob.glob(os.path.join(frames_dir, "rgb_*.png")))
    if not paths:
        raise RuntimeError(f"{frames_dir} 没有 rgb_*.png")
    first = cv2.imread(paths[0])
    if first is None:
        raise RuntimeError(f"无法读取 {paths[0]}")
    h, w = first.shape[:2]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建 {out_path}")
    try:
        for path in paths:
            img = cv2.imread(path)
            if img is None or img.shape[:2] != (h, w):
                continue
            writer.write(img)
    finally:
        writer.release()
    return len(paths)


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
        out = os.path.join(d, "v.mp4")
        n = compose_rgb_video(d, out, fps=10)
        cap = cv2.VideoCapture(out)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert n == 3 and frames == 3, (n, frames)
        print(json.dumps({"3dgs_video": "PASS", "frames": frames}))


if __name__ == "__main__":
    _selftest()
