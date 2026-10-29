#!/usr/bin/env python3
"""把 collect.py 产出的 episode 目录预处理成 C++ 训练器可直接读的 UAVR 二进制数据集。

输入（每个 episode 目录）：
  episode_manifest.json   episode_id/scene_id/seed/split/version/camera
  actions.jsonl           每步 step/action_expert(0..3)/...
  rgb_%05d.png            320x240 RGB 原始帧（step 对应文件名）
输出：
  <out>/dataset.uavr      格式见 tools/DATASET_FORMAT.md（meta JSON 在文件尾部）

分辨率由 --size WxH 显式指定，默认 160x120（320x240 的一半，保持 4:3 不失真）；
--size 128x128 会把 4:3 画面压扁成正方形，仅作对照。实际 H/W 写进文件头与 meta。

split 由每个 episode manifest 的 split 决定，并按 scene_id 强制隔离：
同一 scene_id 不允许同时出现在 train/val/test（帧级随机拆分会造成泄漏）。

碰撞步（actions.jsonl 的 collision=true）：默认保留，但把标记写进 meta 的
collision_flags（N 个 '0'/'1'，C++ 读端用 LoadOptions.drop_collision 过滤）。
--drop-collision 在打包时直接跳过这些步，并在 report 里打印丢弃步数与占比。

用法：
  .venv-tools/bin/python tools/preprocess.py --out /tmp/ds /tmp/ep0 /tmp/ep1
  python3 tools/preprocess.py --drop-collision --out /tmp/ds /tmp/ep0  # 丢碰撞步
  python3 tools/preprocess.py --size 128x128 --out /tmp/ds128 /tmp/ep0   # 压扁对照
  python3 tools/preprocess.py --help
"""
import argparse
import json
import os
import struct
import sys
from datetime import datetime, timezone

try:
    import numpy as np
    from PIL import Image
except ImportError as exc:  # --help 之外的真跑才依赖这两个包
    sys.exit("需要 numpy/Pillow：%s（用 .venv-tools/bin/python 运行）" % exc)

MAGIC = b"UAVR"
VERSION = 1
# <4s HH I HH BBBB QQQ I -> 48 字节，小端、无对齐填充
HEADER = struct.Struct("<4sHHIHHBBBBQQQI")
HEADER_SIZE = HEADER.size
DTYPE_UINT8 = 1
OBS_C = 3
# 默认 160x120：正好是 320x240 的一半，保持 4:3 不失真。
DEFAULT_SIZE = (160, 120)
ACTION_NAMES = ["SAFE_NEUTRAL", "FORWARD_SLOW", "YAW_LEFT", "YAW_RIGHT"]
SPLITS = ("train", "val", "test")
RESAMPLE = (Image.Resampling.BILINEAR if hasattr(Image, "Resampling")
            else Image.BILINEAR)


def die(msg):
    sys.exit("preprocess: %s" % msg)


def load_episode(ep_dir, drop_collision=False):
    """读 manifest + actions.jsonl，只建立样本索引，不碰像素（内存友好）。"""
    man_path = os.path.join(ep_dir, "episode_manifest.json")
    try:
        with open(man_path, encoding="utf-8") as f:
            man = json.load(f)
    except (OSError, ValueError) as exc:
        die("读取 %s 失败：%s（输入应为 collect.py 产物）" % (man_path, exc))

    split = man.get("split")
    if split not in SPLITS:
        die("%s: split=%r 非法（应为 %s）" % (man_path, split, "/".join(SPLITS)))
    scene_id = man.get("scene_id")
    if scene_id is None:
        die("%s: 缺 scene_id（scene 隔离 split 必需）" % man_path)
    camera = man.get("camera") or {}
    src_w = int(camera.get("width") or 320)
    src_h = int(camera.get("height") or 240)

    actions_path = os.path.join(ep_dir, man.get("actions_ref") or "actions.jsonl")
    try:
        with open(actions_path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    except (OSError, ValueError) as exc:
        die("读取 %s 失败：%s" % (actions_path, exc))
    if not rows:
        die("%s: actions.jsonl 为空" % ep_dir)

    samples, seen = [], set()
    dropped = 0
    for row in sorted(rows, key=lambda r: int(r["step"])):
        try:
            step = int(row["step"])
            action = int(row["action_expert"])
        except (KeyError, TypeError, ValueError):
            die("%s: action 行缺 step/action_expert：%r" % (actions_path, row))
        if action not in range(len(ACTION_NAMES)):
            die("%s step %d: action_expert=%d 超出 0..3" % (actions_path, step, action))
        if step in seen:
            die("%s: step %d 重复" % (actions_path, step))
        seen.add(step)
        # 碰撞步的 action_expert 是专家在撞墙后的错误标签，过滤是标签清洗而不是丢样本。
        collision = bool(row.get("collision", False))
        if drop_collision and collision:
            dropped += 1
            continue
        png = os.path.join(ep_dir, "rgb_%05d.png" % step)
        if not os.path.isfile(png):
            die("缺帧：%s（actions.jsonl 与 rgb_*.png 不同步）" % png)
        samples.append({"png": png, "label": action, "step": step,
                        "collision": collision})

    return {
        "dir": os.path.abspath(ep_dir),
        "episode_id": man.get("episode_id"),
        "scene_id": scene_id,
        "seed": man.get("seed", 0),
        "version": str(man.get("version", "")),
        "split": split,
        "src_w": src_w,
        "src_h": src_h,
        "samples": samples,
        "dropped": dropped,
    }


def scan(ep_dirs, drop_collision=False):
    """加载全部 episode 索引，并检查 scene_id 的 split 唯一性。"""
    episodes, scene_split, total, dropped = [], {}, 0, 0
    for d in ep_dirs:
        ep = load_episode(d, drop_collision)
        sid = ep["scene_id"]
        if sid in scene_split and scene_split[sid] != ep["split"]:
            die("scene_id=%r 同时出现在 split=%s 与 %s（必须按场景隔离）"
                % (sid, scene_split[sid], ep["split"]))
        scene_split[sid] = ep["split"]
        ep["sample_start"] = total
        ep["num_samples"] = len(ep["samples"])
        if episodes and (ep["src_w"], ep["src_h"]) != (episodes[0]["src_w"], episodes[0]["src_h"]):
            die("输入 episode 的相机尺寸不一致：%s"
                % sorted({(e["src_w"], e["src_h"]) for e in episodes + [ep]}))
        total += ep["num_samples"]
        dropped += ep["dropped"]
        episodes.append(ep)
    if total == 0:
        die("没有可写样本")
    return episodes, total, dropped


def build_meta(episodes, total, size, dropped=0):
    src_w, src_h = episodes[0]["src_w"], episodes[0]["src_h"]
    out_w, out_h = size
    preserve = out_w * src_h == out_h * src_w  # 交叉相乘判断目标是否与原图同纵横比
    counts = {s: 0 for s in SPLITS}
    for ep in episodes:
        counts[ep["split"]] += ep["num_samples"]
    scenes = {}
    for ep in episodes:
        scenes[ep["scene_id"]] = ep["split"]
    meta = {
        "schema": "uavrgb.dataset.v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "images": {"layout": "NCHW", "dtype": "uint8",
                   "height": out_h, "width": out_w, "channels": OBS_C},
        "labels": {"dtype": "uint8", "num_classes": len(ACTION_NAMES),
                   "names": ACTION_NAMES, "source": "actions.jsonl:action_expert"},
        "resize": {"method": "bilinear", "from": [src_w, src_h],
                   "to": [out_w, out_h],
                   "aspect": "preserve" if preserve else "squash"},
        "split_policy": "scene_id",
        # 逐样本碰撞标记（'1' = collision=true），供 C++ 读端 LoadOptions.drop_collision
        # 在不重新打包的情况下过滤；--drop-collision 打包时已丢弃的步不在这里。
        "collision_flags": "".join("1" if s["collision"] else "0"
                                   for ep in episodes for s in ep["samples"]),
        "collision_dropped": dropped,
        "splits": counts,
        "num_samples": total,
        "scenes": [{"scene_id": sid, "split": sp}
                   for sid, sp in sorted(scenes.items())],
        "episodes": [{
            "episode_id": ep["episode_id"],
            "scene_id": ep["scene_id"],
            "seed": ep["seed"],
            "version": ep["version"],
            "split": ep["split"],
            "num_steps": ep["num_samples"],
            "sample_start": ep["sample_start"],
            "sample_count": ep["num_samples"],
            "source_dir": ep["dir"],
        } for ep in episodes],
    }
    return meta


def load_frame(path, src_w, src_h, size):
    """PNG -> 目标 size RGB uint8，HWC -> CHW（NCHW 数据集布局）。"""
    out_w, out_h = size
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            if im.size != (src_w, src_h):
                die("%s: 帧尺寸 %dx%d 与 manifest 相机 %dx%d 不一致"
                    % (path, im.size[0], im.size[1], src_w, src_h))
            im = im.resize((out_w, out_h), RESAMPLE)
            arr = np.asarray(im, dtype=np.uint8)
    except OSError as exc:
        die("解码 %s 失败：%s" % (path, exc))
    if arr.shape != (out_h, out_w, OBS_C):
        die("%s: 缩放后形状 %r 非 (%d,%d,%d)" % (path, arr.shape, out_h, out_w, OBS_C))
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def write_dataset(out_dir, episodes, total, size, dropped=0):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dataset.uavr")
    tmp_path = out_path + ".tmp"
    out_w, out_h = size

    meta_bytes = json.dumps(build_meta(episodes, total, size, dropped),
                            ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    images_off = HEADER_SIZE
    images_len = total * OBS_C * out_h * out_w
    labels_off = images_off + images_len
    meta_off = labels_off + total

    try:
        with open(tmp_path, "wb") as f:
            f.write(HEADER.pack(MAGIC, VERSION, HEADER_SIZE, total,
                                out_h, out_w, OBS_C,
                                DTYPE_UINT8, DTYPE_UINT8, 0,
                                images_off, labels_off, meta_off, len(meta_bytes)))
            for ep in episodes:
                for s in ep["samples"]:
                    frame = load_frame(s["png"], ep["src_w"], ep["src_h"], size)
                    f.write(frame.tobytes(order="C"))
            labels = bytearray()
            for ep in episodes:
                labels.extend(s["label"] for s in ep["samples"])
            f.write(bytes(labels))
            f.write(meta_bytes)
        os.replace(tmp_path, out_path)  # 原子落盘，失败不留半截数据集
    except SystemExit:  # load_frame/die 在写盘中途失败
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    except OSError as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        die("写 %s 失败：%s" % (out_path, exc))

    return out_path, meta_bytes


def parse_size(text):
    try:
        w, h = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("尺寸格式应为 WxH，如 160x120")
    if not (1 <= w <= 4096 and 1 <= h <= 4096):
        raise argparse.ArgumentTypeError("W/H 需在 1..4096")
    return w, h


def main():
    p = argparse.ArgumentParser(
        description="episode 目录 -> UAVR 二进制数据集（尺寸/纵横比显式，按 scene_id 隔离 split）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out", required=True, help="输出目录（写 dataset.uavr）")
    p.add_argument("--size", type=parse_size, default=DEFAULT_SIZE, metavar="WxH",
                   help="输出分辨率；160x120 保持 4:3，128x128 会压扁画面（对照实验）")
    p.add_argument("--drop-collision", action="store_true",
                   help="跳过 collision=true 的步（专家撞墙后的标签本身是错的）")
    p.add_argument("episodes", nargs="+", help="一个或多个 collect.py episode 目录")
    args = p.parse_args()

    episodes, total, dropped = scan(args.episodes, args.drop_collision)
    out_path, meta_bytes = write_dataset(args.out, episodes, total, args.size, dropped)
    meta = json.loads(meta_bytes.decode("utf-8"))
    out_w, out_h = args.size
    aspect = meta["resize"]["aspect"]

    print("episodes: %d  samples: %d  scenes: %d"
          % (len(episodes), total, len(meta["scenes"])))
    print("splits: train=%d val=%d test=%d"
          % (meta["splits"]["train"], meta["splits"]["val"], meta["splits"]["test"]))
    seen = total + dropped
    print("collision: dropped %d of %d steps (%.1f%%), kept %d"
          % (dropped, seen, 100.0 * dropped / seen if seen else 0.0, total))
    for ep in episodes:
        print("  episode_id=%s scene_id=%s split=%s steps=%d sample_start=%d"
              % (ep["episode_id"], ep["scene_id"], ep["split"],
                 ep["num_samples"], ep["sample_start"]))
    print("resize: bilinear %dx%d RGB -> %dx%d NCHW uint8 (aspect=%s)"
          % (episodes[0]["src_w"], episodes[0]["src_h"], out_w, out_h, aspect))
    print("wrote %s (%d bytes, meta %d bytes)"
          % (out_path, os.path.getsize(out_path), len(meta_bytes)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
