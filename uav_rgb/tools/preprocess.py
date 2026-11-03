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
import tempfile
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "sim", "tools"))
import split_manifest as sm  # noqa: E402
import instruction as instr  # noqa: E402

# 必须与 src/include/uavrgb/text.hpp 的 kTextLen 一致：不足补 <pad>，超长由两端报错。
TEXT_LEN = 9

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


def load_episode(ep_dir, drop_collision=False, exclude_collision_episodes=False):
    """读 manifest + actions.jsonl，只建立样本索引，不碰像素（内存友好）。

    exclude_collision_episodes=True 时，只要该集有 collision=true 就返回 None
    （整集丢弃）。碰撞后的专家标签不可信，T2 的 SAFE 硬约束也不允许 SAFE+碰撞。
    """
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
    gid = man.get("geometry_gid")
    if gid is not None and (not isinstance(gid, str) or not gid):
        die("%s: geometry_gid 必须是非空字符串（实际 %r）" % (man_path, gid))
    # collect.py 采集中断时会写 error 并把半截 episode 留在磁盘上；
    # 不能让没采完的 episode 静默进数据集（它会伪装成一条正常的短 episode）。
    if man.get("error"):
        die("%s: 采集失败（%s），不能打包" % (man_path, man["error"]))
    # 评测集会在碰撞处提前结束；它看起来像一条正常短 episode，但标签只到碰撞点，
    # 直接进训练会污染数据。训练数据必须逐集跑满。
    if man.get("stopped_on_collision"):
        die("%s: stopped_on_collision=true（闭环评测产物），不能当训练数据打包"
            % man_path)
    validity = man.get("validity")
    if not isinstance(validity, dict) or not all(
            validity.get(k) for k in ("camera_verified", "physics_verified",
                                      "episode_executed")):
        die("%s: validity 缺失或未全 true（%r），不能打包" % (man_path, validity))
    # v2/v3 的 geometry_gid 天然不同，只查 gid 交集查不出两套几何混包。
    gver = man.get("geometry_version")
    if gver not in (sm.SPLIT_SALT, "campus.v1"):
        die("%s: geometry_version=%r 与当前 %s 不一致（旧几何数据不能混进新数据集）"
            % (man_path, gver, sm.SPLIT_SALT))
    # 最后一道防泄漏门：split 必须等于几何 gid 应属的 split。只查「同 gid 跨 split」
    # 挡不住单集包内被改过的 manifest（没有对照集，一致性检查永远通过）。
    if gid is not None:
        want = sm.split_for_scene(man)
        if split != want:
            die("%s: split=%s 与 geometry_gid=%s 应属的 split=%s 不一致（manifest 被改动？）"
                % (man_path, split, gid, want))
    # 标签只能来自教师。forward=盲策略对照、student=学生自采，它们的标签是
    # 「被测对象」而不是监督信号；这类集 validity 也全 true，不拦就会静默进训练集，
    # 而 audit 的 label_recompute 只复算 plan_action，不会发现。
    teacher = man.get("teacher")
    if teacher not in ("plan_action", "expert_action", "synthetic"):
        die("%s: teacher=%r 不是教师标签（forward/student 产物不能当训练数据打包）"
            % (man_path, teacher))
    instruction = man.get("instruction")
    if instruction is not None and not isinstance(instruction, str):
        die("%s: instruction 必须是字符串（实际 %r）" % (man_path, instruction))
    task_version = man.get("task_version")
    target_color = man.get("target_color")
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
    if exclude_collision_episodes and any(r.get("collision") is True for r in rows):
        return None
    expected = man.get("num_steps")
    if isinstance(expected, int) and len(rows) != expected:
        die("%s: actions.jsonl %d 行 != manifest num_steps %d（文件被截断？）"
            % (actions_path, len(rows), expected))

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
        # 必须是真正的 bool：bool("false") 是 True，字符串会让过滤彻底失效。
        collision = row.get("collision", False)
        if not isinstance(collision, bool):
            die("%s step %d: collision 必须是 true/false（实际 %r）"
                % (actions_path, step, collision))
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
        "geometry_gid": gid,
        "geometry_version": gver,
        **{key: man.get(key) for key in sm.PROVENANCE_KEYS},
        "teacher": man.get("teacher"),
        "planner": man.get("planner"),
        "instruction": instruction,
        "task_version": task_version,
        "target_color": target_color,
        "waypoint": man.get("waypoint"),
        "stop_radius": man.get("stop_radius"),
        "window_from": man.get("window_from"),
        "hold_s": man.get("hold_s"),
        "src_w": src_w,
        "src_h": src_h,
        "samples": samples,
        "dropped": dropped,
    }


def scan(ep_dirs, drop_collision=False, exclude_collision_episodes=False):
    """加载全部 episode 索引，并检查 scene_id 的 split 唯一性。"""
    episodes, scene_split, gid_split, total, dropped = [], {}, {}, 0, 0
    skipped = 0
    seen_dirs = set()
    for d in ep_dirs:
        # 同一目录经两个 glob/软链传两次会静默双打包，同一批样本被算两遍。
        real = os.path.realpath(d)
        if real in seen_dirs:
            die("重复的 episode 目录：%s（同一集只能打包一次）" % real)
        seen_dirs.add(real)
        ep = load_episode(d, drop_collision, exclude_collision_episodes)
        if ep is None:
            skipped += 1
            continue
        sid = ep["scene_id"]
        if sid in scene_split and scene_split[sid] != ep["split"]:
            die("scene_id=%r 同时出现在 split=%s 与 %s（必须按场景隔离）"
                % (sid, scene_split[sid], ep["split"]))
        scene_split[sid] = ep["split"]
        gid = ep["geometry_gid"]
        if gid is not None:
            if gid in gid_split and gid_split[gid] != ep["split"]:
                die("几何组 %s 同时出现在 split=%s 与 %s（同几何必须同 split）"
                    % (gid, gid_split[gid], ep["split"]))
            gid_split[gid] = ep["split"]
        ep["sample_start"] = total
        ep["num_samples"] = len(ep["samples"])
        if episodes and (ep["src_w"], ep["src_h"]) != (episodes[0]["src_w"], episodes[0]["src_h"]):
            die("输入 episode 的相机尺寸不一致：%s"
                % sorted({(e["src_w"], e["src_h"]) for e in episodes + [ep]}))
        total += ep["num_samples"]
        dropped += ep["dropped"]
        episodes.append(ep)
    if skipped:
        print("preprocess: 跳过整集 %d 个（--exclude-collision-episodes）" % skipped)
    if total == 0:
        die("没有可写样本")
    # 同一数据集不能混 observation/dispatch、不同 hold 或不同任务版本，否则
    # eval.json / 训练日志里的数字无法归因。缺字段的旧数据与有字段的数据也不能混。
    for key in ("window_from", "hold_s", "task_version", "stop_radius", "geometry_version",
                "appearance_version", "split_policy", "camera_profile", "dynamics_profile",
                "success_criterion", "goal_radius"):
        vals = {ep.get(key) for ep in episodes if ep.get(key) is not None}
        if len(vals) > 1:
            die("episode 的 %s 不一致: %s（不同口径不能混包）"
                % (key, sorted(str(v) for v in vals)))
        if vals and any(ep.get(key) is None for ep in episodes):
            die("部分 episode 缺 %s，不能与有该字段的数据混包" % key)
    # L0：指令要么全有、要么全无。部分集有指令会让「文本是否可用」变成静默偏置。
    have = [bool(ep.get("instruction") and ep["instruction"].strip()) for ep in episodes]
    if any(have) and not all(have):
        die("instruction 只覆盖部分 episode（%d/%d 集缺）"
            % (have.count(False), len(have)))
    return episodes, total, dropped


def _uniform(episodes, key):
    """所有 episode 同值才返回该值；调用方已在 scan 里拒绝不一致。"""
    vals = {ep.get(key) for ep in episodes if ep.get(key) is not None}
    return next(iter(vals)) if len(vals) == 1 else None


def build_meta(episodes, total, size, dropped=0):
    src_w, src_h = episodes[0]["src_w"], episodes[0]["src_h"]
    out_w, out_h = size
    preserve = out_w * src_h == out_h * src_w  # 交叉相乘判断目标是否与原图同纵横比
    counts = {s: 0 for s in SPLITS}
    class_counts = {s: [0] * len(ACTION_NAMES) for s in SPLITS}
    for ep in episodes:
        counts[ep["split"]] += ep["num_samples"]
        for s in ep["samples"]:
            class_counts[ep["split"]][s["label"]] += 1
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
        # 口径字段：eval/训练/audit 从数据集本身就能看出 observation/dispatch 与任务版本。
        "geometry_version": _uniform(episodes, "geometry_version") or sm.SPLIT_SALT,
        "appearance_version": _uniform(episodes, "appearance_version"),
        "success_criterion": _uniform(episodes, "success_criterion"),
        "goal_radius": _uniform(episodes, "goal_radius"),
        "window_from": _uniform(episodes, "window_from"),
        "hold_s": _uniform(episodes, "hold_s"),
        "task_version": _uniform(episodes, "task_version"),
        "stop_radius": _uniform(episodes, "stop_radius"),
        # 逐样本碰撞标记（'1' = collision=true），供 C++ 读端 LoadOptions.drop_collision
        # 在不重新打包的情况下过滤；--drop-collision 打包时已丢弃的步不在这里。
        "collision_flags": "".join("1" if s["collision"] else "0"
                                   for ep in episodes for s in ep["samples"]),
        "collision_dropped": dropped,
        "splits": counts,
        # 每 split 的每类样本数：T3 的「四类 recall >0」需要先确认 val/test 有 support。
        "class_counts": class_counts,
        "num_samples": total,
        "scenes": [{"scene_id": sid, "split": sp}
                   for sid, sp in sorted(scenes.items())],
        "episodes": [{
            "episode_id": ep["episode_id"],
            "scene_id": ep["scene_id"],
            "seed": ep["seed"],
            "version": ep["version"],
            "split": ep["split"],
            "geometry_gid": ep["geometry_gid"],
            # airsim_to_uavr 直接构造 episode dict，没有这两键；必须用 get。
            "geometry_version": ep.get("geometry_version"),
            **{key: ep.get(key) for key in sm.PROVENANCE_KEYS},
            "teacher": ep.get("teacher"),
            "task_version": ep.get("task_version"),
            "instruction": ep.get("instruction"),
            "target_color": ep.get("target_color"),
            # 扁平字段：C++ meta 扫描器假定 episode 项内无嵌套对象。
            "waypoint_x": (ep.get("waypoint") or {}).get("x"),
            "waypoint_y": (ep.get("waypoint") or {}).get("y"),
            "stop_radius": ep.get("stop_radius"),
            "window_from": ep.get("window_from"),
            "hold_s": ep.get("hold_s"),
            "num_steps": ep["num_samples"],
            "sample_start": ep["sample_start"],
            "sample_count": ep["num_samples"],
            "source_dir": ep["dir"],
        } for ep in episodes],
    }
    return meta


def load_frame(path, src_w, src_h, size, crop=None, src_expected=None):
    """PNG -> 目标 size RGB uint8，HWC -> CHW（NCHW 数据集布局）。

    crop=(left, top, right, bottom) 裁剪；src_w/src_h 指裁剪后的尺寸。
    src_expected=(w,h) 是**裁剪前**的源图尺寸，给了就强制校验。
    没有它时裁剪会让尺寸校验变成恒真式（裁完永远是 crop 盒子的尺寸），
    错尺寸的源图会被静默打包成黑边样本，所以用 crop 的调用方必须给。
    """
    out_w, out_h = size
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            if src_expected is not None and im.size != src_expected:
                die("%s: 源图尺寸 %dx%d 与预期 %dx%d 不一致"
                    % (path, im.size[0], im.size[1], src_expected[0], src_expected[1]))
            if crop is not None:
                im = im.crop(crop)
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


def write_dataset(out_dir, episodes, total, size, dropped=0, crop=None,
                  src_expected=None):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dataset.uavr")
    out_w, out_h = size

    meta_bytes = json.dumps(build_meta(episodes, total, size, dropped),
                            ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    images_off = HEADER_SIZE
    images_len = total * OBS_C * out_h * out_w
    labels_off = images_off + images_len
    meta_off = labels_off + total

    # 唯一临时名：两个进程写同一个 --out 时不会共写一个文件，finally 保证
    # Ctrl-C / 异常都会清掉（旧版只捕 SystemExit/OSError，Ctrl-C 会留下整份 .tmp）。
    tmp_path = None
    try:
        # 放在 try 里：分区满/无权限时也要走下面的 die，而不是甩一段 traceback。
        fd, tmp_path = tempfile.mkstemp(dir=out_dir, prefix=".dataset.", suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            f.write(HEADER.pack(MAGIC, VERSION, HEADER_SIZE, total,
                                out_h, out_w, OBS_C,
                                DTYPE_UINT8, DTYPE_UINT8, 0,
                                images_off, labels_off, meta_off, len(meta_bytes)))
            for ep in episodes:
                for s in ep["samples"]:
                    frame = load_frame(s["png"], ep["src_w"], ep["src_h"], size,
                                       ep.get("crop", crop),
                                       ep.get("src_expected", src_expected))
                    f.write(frame.tobytes(order="C"))
            labels = bytearray()
            for ep in episodes:
                labels.extend(s["label"] for s in ep["samples"])
            f.write(bytes(labels))
            f.write(meta_bytes)
        os.replace(tmp_path, out_path)  # 原子落盘，失败不留半截数据集
        tmp_path = None
    except OSError as exc:
        die("写 %s 失败：%s" % (out_path, exc))
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return out_path, meta_bytes


def instruction_rows(episodes):
    """先算好并校验全部指令行，避免 dataset.uavr 已替换后才发现侧车写不了。"""
    rows = []
    for ep in episodes:
        text = ep["instruction"].strip()
        ids = instr.tokenize(text)
        if len(ids) > TEXT_LEN:
            die("%s: 指令 %r 有 %d 个 token，超过 TEXT_LEN=%d"
                % (ep["dir"], text, len(ids), TEXT_LEN))
        rows.append({
            "episode_id": ep["episode_id"],
            "instruction": text,
            "sample_start": ep["sample_start"],
            "sample_count": ep["num_samples"],
            "text_ids": ids,
        })
    return rows


def write_instructions(out_dir, rows):
    """旁路写 instructions.jsonl；tmp + rename，失败不留半截/不截断旧文件。"""
    path = os.path.join(out_dir, "instructions.jsonl")
    # mkstemp：固定 ".tmp" 名在并发打包同一 --out 时会互相截断。
    fd, tmp = tempfile.mkstemp(prefix=".instructions.", suffix=".tmp", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    except OSError as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        die("写 %s 失败：%s" % (path, exc))
    return path


def parse_size(text):
    try:
        w, h = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("尺寸格式应为 WxH，如 160x120")
    if not (1 <= w <= 4096 and 1 <= h <= 4096):
        raise argparse.ArgumentTypeError("W/H 需在 1..4096")
    # make_cnn 要求 H/W 是 8 的倍数；在这里挡住，别让几十分钟的预处理白跑。
    if w % 8 != 0 or h % 8 != 0:
        raise argparse.ArgumentTypeError("W/H 必须是 8 的倍数（模型下采样要求），如 160x120")
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
    p.add_argument("--exclude-collision-episodes", action="store_true",
                   help="整集丢弃含 collision 步的 episode（T2 硬约束）")
    p.add_argument("episodes", nargs="+", help="一个或多个 collect.py episode 目录")
    args = p.parse_args()

    episodes, total, dropped = scan(args.episodes, args.drop_collision,
                                    args.exclude_collision_episodes)
    # 先校验侧车内容，再替换 dataset.uavr；有指令时先删旧侧车。
    # 这样侧车写失败时结果是「新数据集、没有侧车」（文本模型会显式报错），
    # 而不是「新数据集 + 旧侧车」被 C++ 按同集区间静默绑定。
    rows = (instruction_rows(episodes)
            if any(ep.get("instruction") and ep["instruction"].strip() for ep in episodes)
            else [])
    stale = os.path.join(args.out, "instructions.jsonl")
    if rows and os.path.exists(stale):
        os.remove(stale)
    out_path, meta_bytes = write_dataset(args.out, episodes, total, args.size, dropped)
    if rows:
        ipath = write_instructions(args.out, rows)
        print("wrote %s (%d instructions)" % (ipath, len(rows)))
    else:
        if os.path.exists(stale):
            os.remove(stale)
            print("removed stale %s（新数据集没有指令）" % stale)
    meta = json.loads(meta_bytes.decode("utf-8"))
    out_w, out_h = args.size
    aspect = meta["resize"]["aspect"]

    print("episodes: %d  samples: %d  scenes: %d"
          % (len(episodes), total, len(meta["scenes"])))
    print("splits: train=%d val=%d test=%d"
          % (meta["splits"]["train"], meta["splits"]["val"], meta["splits"]["test"]))
    for split in SPLITS:
        cc = meta["class_counts"][split]
        print("  %-5s %s" % (split, dict(zip(ACTION_NAMES, cc))))
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
