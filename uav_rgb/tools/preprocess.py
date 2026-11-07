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
打包时按 manifest 记录的 split_policy/geometry_version 复算并校验（冻结数据不回落
当前 salt），实际策略/方案/salt/每 split 样本数/geometry gid 全部写进 meta 供审计。
map-hash:* / diag* 诊断性划分默认拒绝，必须显式 --allow-diagnostic-split。

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
import hashlib
import json
import os
import struct
import sys
import tempfile
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "sim", "tools"))
import split_manifest as sm  # noqa: E402
import campus  # noqa: E402
import asset_campus  # noqa: E402
import instruction as instr  # noqa: E402
import scene_forest  # noqa: E402

# 当前允许打包的几何族版本（不加 --allow-frozen-geometry 时）。3dgs.* 按前缀放行（collect_3dgs 写）。
# 旧柱阵版本（v3/v4）、campus.v1 要显式 --allow-frozen-geometry。
KNOWN_GEOMETRY_VERSIONS = (sm.SPLIT_SALT, campus.VERSION, asset_campus.VERSION,
                           scene_forest.GEOMETRY_VERSION_FOREST)
# 口径键：任何模式下一个包只能有一种值（含「部分缺」也不行）。hold/窗口语义/动力学不同意味着
# 同一个动作标签对应的位移不同（Gazebo hold 1.0 前飞 0.25 m/步，3DGS hold 0.5 约 0.12 m），混了无法归因。
STRICT_KEYS = ("window_from", "hold_s", "dynamics_profile")
# 族身份键：默认同样必须一致；--allow-mixed-geometry（跨族合并实验，如柱阵+森林、Gazebo+3DGS）才放开，
# meta 按族记录计数/salt/口径，各族仍按各自的 split_salt 复算 split（同几何同 split 的泄漏门照常）。
FAMILY_KEYS = ("geometry_version", "appearance_version", "split_policy", "split_salt",
               "task_version", "stop_radius", "success_criterion", "goal_radius")
# 相机口径：混包时允许「部分集缺字段」（3DGS pilot 没写），但两种不同的非空值仍不能混。
CAMERA_KEY = "camera_profile"

# 必须与 src/include/uavrgb/text.hpp 的 kTextLen 一致：不足补 <pad>，超长由两端报错。
TEXT_LEN = 9

# UAVRAUX1 侧车（dataset.aux，DATASET_FORMAT.md §6）：双帧的上一帧下标、prev-action、
# 深度分带目标。dataset.uavr 本身字节不变，旧读端不受影响。
AUX_MAGIC = b"UAVRAUX1"
AUX_HEADER = struct.Struct("<8sQIHHBBH")  # magic N flags depth_h depth_w K unknown reserved = 28 B
AUX_HEADER_SIZE = 128
AUX_UNKNOWN = 255
# 分带上界（m）：0.3/0.5/0.8/1.2/1.8/2.5/4，第 8 带 = ≥4 m（ALGO 报告 §2 P2）。
DEPTH_BAND_EDGES_M = (0.3, 0.5, 0.8, 1.2, 1.8, 2.5, 4.0)
DEPTH_CLASSES = len(DEPTH_BAND_EDGES_M) + 1
# 机体包络半径：sim/dynamics/e1_mini.json collision_envelope.radius_from_center（E1 实机）。
DEFAULT_ENVELOPE_RADIUS_M = 0.102
DEFAULT_HFOV_DEG = 60.0  # manifest 没写 fx 时按 Gazebo 相机 60° 水平视场推焦距
# 网格 = 模型第三层特征图（H/8 × W/8），与 1x1 辅助头逐格对齐。
DEPTH_GRID_DOWN = 8

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


def diag_split(gid, salt):
    """同族留出诊断用的 split：按 geometry_gid 的哈希切成 70/15/15。

    只用于回答「模型在同族留出地图上能不能学」，**不是常规划分**：
    它无视拓扑族归属，会把同一族的图分到 train/val/test 三边。常规训练仍用
    split_for_scene，混用会让「跨族」结论失效。
    """
    h = int(hashlib.sha256(("%s:%s" % (salt, gid)).encode()).hexdigest()[:8], 16)
    bucket = h % 100
    return "train" if bucket < 70 else ("val" if bucket < 85 else "test")


def load_episode(ep_dir, drop_collision=False, exclude_collision_episodes=False,
                 split_salt=None, allow_frozen=False, allow_diagnostic=False):
    """读 manifest + actions.jsonl，只建立样本索引，不碰像素（内存友好）。

    exclude_collision_episodes=True 时，只要该集有 collision=true 就返回 None
    （整集丢弃）。碰撞后的专家标签不可信，T2 的 SAFE 硬约束也不允许 SAFE+碰撞。
    split_salt 非空 = --split-by-map 诊断构建；allow_diagnostic 才允许 map-hash/diag*。
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
    policy = man.get("split_policy")
    if policy is not None and policy not in sm.SPLIT_POLICIES:
        if not sm.is_diagnostic_scheme(policy):
            die("%s: split_policy=%r 未知（支持 %s）"
                % (man_path, policy, "/".join(sm.SPLIT_POLICIES)))
        if not allow_diagnostic:
            die("%s: split_policy=%r 是诊断性划分，需显式 --allow-diagnostic-split"
                % (man_path, policy))
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
    # 只认当前 campus 版本或资产校园版本：v1 的 480 集已冻结并打包，改布局后同
    # seed 是另一张图，允许 v1 通过等于允许两套几何混包。（2026-11-04 升 v2）
    # allow_frozen=True 用于对冻结数据做诊断（同族留出/重打包）：此时不再要求等于
    # 当前版本，改由 build_meta 断言全包版本一致——真正要守的是「不混包」，不是「必须最新」。
    if not allow_frozen and gver not in KNOWN_GEOMETRY_VERSIONS and not (
            isinstance(gver, str) and gver.startswith("3dgs.")):
        die("%s: geometry_version=%r 不在当前族版本 %s 内（旧几何数据不能混进新数据集）"
            % (man_path, gver, list(KNOWN_GEOMETRY_VERSIONS) + ["3dgs.*"]))
    # 最后一道防泄漏门：split 必须等于几何 gid/地图按记录策略应属的 split。只查
    # 「同 gid 跨 split」挡不住单集包内被改过的 manifest（没有对照集，一致性检查
    # 永远通过）。同族留出诊断会按 gid 哈希重新划分，这条门按定义不成立，显式跳过；
    # 下方 scan() 的 gid->split 唯一性检查仍然生效，不会让同图跨 split。
    if gid is not None:
        if split_salt:
            split = diag_split(gid, split_salt)
        elif sm.is_diagnostic_scheme(policy):
            pass  # 诊断 manifest 自带 split，没有可复算的标准方案
        else:
            try:
                want = sm.split_for_scene(man)
            except ValueError as exc:
                die("%s: 无法按记录的策略复算 split：%s" % (man_path, exc))
            if split != want:
                die("%s: split=%s 与 geometry_gid=%s 按记录策略应属的 split=%s 不一致（manifest 被改动？）"
                    % (man_path, split, gid, want))
    elif split_salt:
        die("%s: --split-by-map 需要每集都有 geometry_gid（诊断划分按 gid 哈希）"
            % man_path)
    # 标签只能来自教师。forward=盲策略对照、student=学生自采，它们的标签是
    # 「被测对象」而不是监督信号；这类集 validity 也全 true，不拦就会静默进训练集，
    # 而 audit 的 label_recompute 只复算 plan_action，不会发现。
    teacher = man.get("teacher")
    if teacher not in ("plan_action", "expert_action", "synthetic"):
        die("%s: teacher=%r 不是教师标签（forward/student 产物不能当训练数据打包）"
            % (man_path, teacher))
    # DAgger 自采集（collect_3dgs --policy student / collect_parallel --policy student）：动作由学生执行，
    # 但 action_expert 仍是教师标签，允许进入；executed_by/dagger_round/beta 写进 episodes 表供溯源。
    # 老 manifest 没有这几项 = 教师自采（round 0、β=1）。
    executed_by = man.get("executed_by", "teacher")
    if executed_by not in ("teacher", "student"):
        die("%s: executed_by=%r 既不是 teacher 也不是 student" % (man_path, executed_by))
    dagger_round = man.get("dagger_round", 0)
    beta = man.get("beta", 1.0)
    if not isinstance(dagger_round, int) or dagger_round < 0:
        die("%s: dagger_round 必须是非负整数（实际 %r）" % (man_path, dagger_round))
    if not isinstance(beta, (int, float)) or not (0.0 <= float(beta) <= 1.0):
        die("%s: beta 必须在 [0,1]（实际 %r）" % (man_path, beta))
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
    # prev-action 取上一步**执行**的动作（DAgger 数据里学生执行的 ≠ 教师标签；它要补偿的是
    # 相机真实的自运动）；旧数据没有 action_executed 时退回 action_expert。
    by_step = {}
    for row in rows:
        try:
            by_step[int(row["step"])] = row
        except (KeyError, TypeError, ValueError):
            die("%s: action 行缺 step：%r" % (actions_path, row))
    depth_format = man.get("depth_format")
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
        prev_row = by_step.get(step - 1)
        prev_action = 0  # 首帧：没有上一步，按 SAFE_NEUTRAL
        if prev_row is not None:
            try:
                prev_action = int(prev_row.get("action_executed", prev_row["action_expert"]))
            except (KeyError, TypeError, ValueError):
                die("%s step %d: 上一步缺 action_executed/action_expert" % (actions_path, step))
            if prev_action not in range(len(ACTION_NAMES)):
                die("%s step %d: 上一步动作 %d 超出 0..3" % (actions_path, step, prev_action))
        depth = row.get("depth_file")
        if depth is not None:
            if depth_format not in DEPTH_LOADERS:
                die("%s: 有 depth_file 但 manifest.depth_format=%r 不认识（支持 %s）"
                    % (man_path, depth_format, "/".join(DEPTH_LOADERS)))
            depth = os.path.join(ep_dir, depth)
            if not os.path.isfile(depth):
                die("缺深度帧：%s（actions.jsonl 的 depth_file 指向不存在的文件）" % depth)
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
                        "collision": collision, "prev_action": prev_action,
                        "depth": depth, "depth_format": depth_format})

    return {
        "dir": os.path.abspath(ep_dir),
        "camera_fx": camera.get("fx"),
        "episode_id": man.get("episode_id"),
        "scene_id": scene_id,
        "seed": man.get("seed", 0),
        "version": str(man.get("version", "")),
        "split": split,
        "geometry_gid": gid,
        "geometry_version": gver,
        **{key: man.get(key) for key in sm.PROVENANCE_KEYS},
        # 记录型 salt：冻结数据重打包必须按它复算，不能回落当前 SPLIT_SALT。
        "split_salt": man.get("split_salt"),
        "teacher": man.get("teacher"),
        "planner": man.get("planner"),
        "executed_by": executed_by,
        "dagger_round": dagger_round,
        "beta": float(beta),
        "renderer_backend": (man.get("renderer") or {}).get("backend"),
        "scene_json": ("scene.json"
                       if os.path.isfile(os.path.join(ep_dir, "scene.json")) else None),
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


def _check_uniform_keys(episodes, allow_mixed=False):
    """同一数据集不能混 observation/dispatch、不同 hold 或不同任务版本，否则
    eval.json / 训练日志里的数字无法归因。缺字段的旧数据与有字段的数据也不能混。

    allow_mixed=True 只放开族身份键（FAMILY_KEYS）；口径键（STRICT_KEYS）仍必须一致，
    camera_profile 允许部分缺、不允许两种值。
    """
    for key in STRICT_KEYS + (CAMERA_KEY,) + FAMILY_KEYS:
        vals = {ep.get(key) for ep in episodes if ep.get(key) is not None}
        missing = any(ep.get(key) is None for ep in episodes)
        if allow_mixed and key in FAMILY_KEYS:
            continue
        if len(vals) > 1:
            die("episode 的 %s 不一致: %s（不同口径不能混包%s）"
                % (key, sorted(str(v) for v in vals),
                   "" if allow_mixed or key not in FAMILY_KEYS
                   else "；跨族合并实验请显式 --allow-mixed-geometry"))
        if vals and missing:
            if allow_mixed and key == CAMERA_KEY:
                print("preprocess: 警告：部分 episode 缺 %s（混包模式放行；缺字段的集相机口径未知）"
                      % key, file=sys.stderr)
                continue
            die("部分 episode 缺 %s，不能与有该字段的数据混包%s"
                % (key, "" if allow_mixed or key not in FAMILY_KEYS
                   else "；跨族合并实验请显式 --allow-mixed-geometry"))


def scan(ep_dirs, drop_collision=False, exclude_collision_episodes=False,
         split_salt=None, allow_frozen=False, allow_diagnostic=False, allow_mixed=False):
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
        ep = load_episode(d, drop_collision, exclude_collision_episodes, split_salt,
                          allow_frozen, allow_diagnostic)
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
    _check_uniform_keys(episodes, allow_mixed)
    # L0：指令要么全有、要么全无。部分集有指令会让「文本是否可用」变成静默偏置。
    have = [bool(ep.get("instruction") and ep["instruction"].strip()) for ep in episodes]
    if any(have) and not all(have):
        die("instruction 只覆盖部分 episode（%d/%d 集缺）"
            % (have.count(False), len(have)))
    return episodes, total, dropped, skipped


def _uniform(episodes, key):
    """所有 episode 同值才返回该值；调用方已在 scan 里拒绝不一致。"""
    vals = {ep.get(key) for ep in episodes if ep.get(key) is not None}
    return next(iter(vals)) if len(vals) == 1 else None


def _uniform_all(episodes, key):
    """同 _uniform，但「部分缺」也算不一致：混包时族身份键只有一族有值不能写成全包的值。"""
    vals = {ep.get(key) for ep in episodes}
    return next(iter(vals)) if len(vals) == 1 else None


def _require_uniform(episodes, key):
    """同 _uniform，但发现不一致直接 die。

    allow_frozen 放开了「必须等于当前版本」这道门，防止混包就落在这里：
    一个包里只能有一种几何版本，不管是哪一版。
    """
    vals = {ep.get(key) for ep in episodes if ep.get(key) is not None}
    if len(vals) > 1:
        die("%s 不一致：%s（一个数据集只能含一种几何，冻结版本也不行）"
            % (key, sorted(vals)))
    return next(iter(vals)) if vals else None


def family_facts(episodes):
    """按 geometry_version 分族：每族的集数/样本数/salt/口径。单族包也记（一项），混包时逐族看。"""
    # 键名不能与 C++ 读端按子串查找的顶层/逐集键重名（"episodes"/"split_salt"/"split"…）：
    # dataset.cpp 用 meta.find("\"episodes\"") 定位数组，嵌套对象里出现同名键会让它读错位置。
    fams = {}
    for ep in episodes:
        fam = fams.setdefault(str(ep.get("geometry_version")), {"n_episodes": 0, "n_samples": 0})
        fam["n_episodes"] += 1
        fam["n_samples"] += ep["num_samples"]
    for ver, fam in fams.items():
        eps = [ep for ep in episodes if str(ep.get("geometry_version")) == ver]
        try:
            _policy, scheme, salt = _split_facts_one_family(eps)
        except ValueError:   # 诊断构建（--split-by-map）下未知版本无 salt 可复算：只记计数
            scheme, salt = None, None
        fam.update({"family_split_scheme": scheme, "family_split_salt": salt,
                    **{key: _uniform(eps, key) for key in
                       ("appearance_version", "task_version", "hold_s", "camera_profile")}})
    return fams


def split_facts(episodes, split_scheme=None, allow_mixed=False):
    """返回 (split_policy, split_scheme, split_salt)：记录实际用的划分。

    split_scheme 非空 = --split-by-map 诊断构建；否则按 manifest 记录的策略/版本
    推导。salt 取记录值，冻结数据不能回落当前 SPLIT_SALT（会改 split）。
    allow_mixed：逐族推导，要求每族都是 geometry_gid 方案（C++ 读端只认一个 split_scheme，
    且逐 gid 校验同几何同 split）；各族 salt 不同时 meta.split_salt 为空、逐族 salt 在 geometry_versions 里。
    """
    if split_scheme:
        return split_scheme, split_scheme, split_scheme.partition(":")[2] or None
    if not allow_mixed:
        return _split_facts_one_family(episodes)
    versions = sorted({str(ep.get("geometry_version")) for ep in episodes})
    facts = {ver: _split_facts_one_family([ep for ep in episodes
                                           if str(ep.get("geometry_version")) == ver])
             for ver in versions}
    bad = {ver: scheme for ver, (_p, scheme, _s) in facts.items() if scheme != "geometry_gid"}
    if bad:
        die("跨族混包要求每族都按 geometry_gid 划分，实际：%s" % bad)
    salts = {salt for _p, _s, salt in facts.values()}
    return "geometry_gid", "geometry_gid", (next(iter(salts)) if len(salts) == 1 else None)


def _split_facts_one_family(episodes):
    policy = _uniform(episodes, "split_policy")
    gver = _require_uniform(episodes, "geometry_version")
    recorded_salt = _uniform(episodes, "split_salt")
    if sm.is_diagnostic_scheme(policy):
        return policy, policy, recorded_salt
    all_gids = all(ep.get("geometry_gid") for ep in episodes)
    if policy == "campus_map_v1":
        scheme = "campus_map_v1"
    elif policy == "campus_assets_map_v1":
        scheme = "campus_assets_map_v1"
    elif policy == "geometry_gid" or (policy is None and gver and all_gids):
        scheme = "geometry_gid"
    else:
        scheme = "scene_id"
    salt = None
    if scheme != "scene_id":
        salt = recorded_salt or sm.split_salt_for(
            {"split_policy": policy, "geometry_version": gver})
    return policy or scheme, scheme, salt


def build_meta(episodes, total, size, dropped=0, split_scheme=None,
               collision_filter=None, allow_mixed=False, aux_info=None):
    src_w, src_h = episodes[0]["src_w"], episodes[0]["src_h"]
    families = family_facts(episodes)
    if allow_mixed and len(families) > 1:
        # 不能回落 SPLIT_SALT，也不能写成 uavrgb.geom.*（train_student_with_video 等按前缀认柱阵）。
        geometry_version = "mixed:" + "+".join(sorted(families))
        pick = _uniform_all   # 族身份/任务口径键：只有一族有值时顶层不能冒充全包
    else:
        geometry_version = _require_uniform(episodes, "geometry_version") or sm.SPLIT_SALT
        pick = _uniform
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
    groups = {}
    for ep in episodes:
        gid = ep.get("geometry_gid")
        if not gid:
            continue
        if groups.setdefault(gid, ep["split"]) != ep["split"]:
            die("几何组 %s 跨 split（%s 与 %s）"
                % (gid, groups[gid], ep["split"]))
    split_policy, split_scheme, split_salt = split_facts(episodes, split_scheme, allow_mixed)
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
        "split_policy": split_policy,
        # 实际使用的分配函数；map-hash:* / diag* 是诊断构建，读端默认拒绝。
        "split_scheme": split_scheme,
        # 复算 split 用的 salt（campus 按固定地图族归属，没有 salt）。
        "split_salt": split_salt,
        # 每 split 样本数（splits）之外，再落一份几何组归属，审计不必重跑哈希。
        "split_groups": [{"geometry_gid": gid, "split": sp}
                         for gid, sp in sorted(groups.items())],
        # 碰撞过滤会改变各 split 的样本集（尤其 test）：记录策略，跨 run 才可解释。
        "collision_filter": collision_filter,
        # 口径字段：eval/训练/audit 从数据集本身就能看出 observation/dispatch 与任务版本。
        # 混包（--allow-mixed-geometry）时 geometry_version 写 mixed:<族+族>，逐族计数/salt/口径见 geometry_versions；
        # 族间不一致的口径字段（appearance/task_version/stop_radius…）在顶层为 null。
        "geometry_version": geometry_version,
        "mixed_geometry": bool(allow_mixed and len(families) > 1),
        "geometry_versions": families,
        "appearance_version": pick(episodes, "appearance_version"),
        "success_criterion": pick(episodes, "success_criterion"),
        "goal_radius": pick(episodes, "goal_radius"),
        "window_from": _uniform(episodes, "window_from"),
        "hold_s": _uniform(episodes, "hold_s"),
        # 3DGS 数据：记录渲染后端与每集 scene.json，自动评测据此走 eval_3dgs。
        "renderer_backend": _uniform(episodes, "renderer_backend"),
        "scene_json": _uniform(episodes, "scene_json"),
        "task_version": pick(episodes, "task_version"),
        "stop_radius": pick(episodes, "stop_radius"),
        # 逐样本碰撞标记（'1' = collision=true），供 C++ 读端 LoadOptions.drop_collision
        # 在不重新打包的情况下过滤；--drop-collision 打包时已丢弃的步不在这里。
        "collision_flags": "".join("1" if s["collision"] else "0"
                                   for ep in episodes for s in ep["samples"]),
        "collision_dropped": dropped,
        # UAVRAUX1 侧车摘要（新增 meta 字段不升版本，DATASET_FORMAT.md §5）；权威信息在侧车头里。
        # 键名不能含 "episodes"/split_* 子串（C++ 读端按子串定位）。
        "aux_sidecar": aux_info,
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
            # DAgger 溯源（airsim_to_uavr 等直接构造的 episode dict 没有：按教师自采补默认）
            "executed_by": ep.get("executed_by", "teacher"),
            "dagger_round": ep.get("dagger_round", 0),
            "beta": ep.get("beta", 1.0),
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
                  src_expected=None, split_scheme=None, collision_filter=None,
                  allow_mixed=False, aux_info=None):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dataset.uavr")
    out_w, out_h = size

    meta_bytes = json.dumps(build_meta(episodes, total, size, dropped, split_scheme,
                                       collision_filter, allow_mixed, aux_info),
                            ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    # C++ 读端（dataset.cpp）按子串定位 "episodes" 数组与 split_* 标量，不是完整 JSON 解析：
    # "episodes" 只能出现一次，split_*/collision_flags 的第一次出现必须是顶层（在 episodes 数组之前），
    # 否则读端静默读错（2026-11-07 逐族计数表里的 "episodes" 键实测让读端报「episodes[] 为空」）。
    ep_at = meta_bytes.find(b'"episodes":')
    if meta_bytes.count(b'"episodes":') != 1:
        die("meta 里 \"episodes\" 出现 %d 次（C++ 读端按子串查找，必须恰好 1 次）"
            % meta_bytes.count(b'"episodes":'))
    for key in ("split_policy", "split_scheme", "split_salt", "collision_flags"):
        at = meta_bytes.find(('"%s":' % key).encode("utf-8"))
        if at < 0 or at > ep_at:
            die("meta 顶层 \"%s\" 必须出现在 episodes 数组之前（C++ 读端取第一次出现）" % key)
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


# ---- UAVRAUX1 侧车：上一帧链 / prev-action / 深度分带目标 ----
# 加载器统一输出 float32 米：+inf = 「前方没东西」（有效，落最远带），nan = 未知（255，损失屏蔽）。
# 0 的语义按 manifest.depth_format 区分（它同时决定解码方式，不另按扩展名猜）。

def _load_depth_png16_mm(path):
    """collect.py --save-depth：16 位灰度 PNG，毫米。写端把 inf（20 m 远裁剪内没打到）和负值编成 0、
    >65.535 m 截到 65535（sim/tools/test_save_depth.py）——所以 0 = 天空/超远 → +inf，是最远带的监督。"""
    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.dtype != np.uint16:
        die("%s: 深度 PNG 不是 16 位（dtype=%s）" % (path, arr.dtype))
    depth = arr.astype(np.float32) / 1000.0
    depth[arr == 0] = np.inf
    return depth


def _load_depth_npy_m(path):
    """collect_3dgs.py --save-depth：float32 米；0（alpha 空洞，渲染没覆盖）/负/非有限 → 未知 nan。"""
    arr = np.load(path)
    if arr.ndim != 2:
        die("%s: 深度 npy 应为二维（实际 shape=%r）" % (path, arr.shape))
    depth = arr.astype(np.float32)
    depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
    return depth


DEPTH_LOADERS = {"png16_mm": _load_depth_png16_mm, "npy_float32_m": _load_depth_npy_m}


def cell_min_depth(depth, grid):
    """(src_h, src_w) 深度 → (dh, dw) 每格最小已知深度。

    nan（未知）不参与 min；+inf（没东西）参与：整格只有 inf → inf（最远带），整格全 nan → nan（255）。
    """
    dh, dw = grid
    src_h, src_w = depth.shape
    known = ~np.isnan(depth)
    filled = np.where(known, depth, np.inf).astype(np.float32)
    if src_h % dh == 0 and src_w % dw == 0:
        blocks = (dh, src_h // dh, dw, src_w // dw)
        out = filled.reshape(blocks).min(axis=(1, 3))
        any_known = known.reshape(blocks).any(axis=(1, 3))
    else:
        out = np.empty((dh, dw), dtype=np.float32)
        any_known = np.empty((dh, dw), dtype=bool)
        ys = (np.arange(dh + 1) * src_h) // dh
        xs = (np.arange(dw + 1) * src_w) // dw
        for r in range(dh):
            for c in range(dw):
                out[r, c] = filled[ys[r]:ys[r + 1], xs[c]:xs[c + 1]].min()
                any_known[r, c] = known[ys[r]:ys[r + 1], xs[c]:xs[c + 1]].any()
    out[~any_known] = np.nan
    return out


def dilate_by_envelope(cells, focal_px, cell_px, radius_m):
    """按机体包络膨胀：深度 z 的格把 z 扩散到像素半径 f·r/z（换算成格）内的邻格，取 min。

    这就是「E1 包络膨胀」：近处障碍在图上要按机身半径加宽，模型学的是「这格飞过去会碰」，
    而不是「这格有像素」。cells 为 (N, dh, dw)：nan = 未知、+inf = 没东西，两者都不向外膨胀
    （半径 0 / 无定义），但都会被邻近的近处格覆盖。
    """
    if radius_m <= 0 or focal_px <= 0:
        return cells
    out = cells.copy()
    with np.errstate(all="ignore"):
        rad = focal_px * radius_m / cells / cell_px  # 每格的膨胀半径（格）
    finite = np.isfinite(rad)
    if not finite.any():
        return out
    n, dh, dw = cells.shape
    # 偏移上界按网格截断：|dy|≥dh 或 |dx|≥dw 的偏移落不到任何格，结果不变；不截断时一个
    # 极近的格（3DGS 深度 1e-6 m）会让循环走 (2·1.8e6+1)² 次，preprocess 卡死不报错。
    rmax = min(int(np.ceil(np.nanmax(rad[finite]))), max(dh, dw) - 1)
    for dy in range(-rmax, rmax + 1):
        for dx in range(-rmax, rmax + 1):
            if dy == 0 and dx == 0:
                continue
            dist = (dy * dy + dx * dx) ** 0.5
            src = np.where(finite & (rad >= dist), cells, np.nan)
            # src[:, y, x] 影响 out[:, y+dy, x+dx]
            ys, ye = max(0, -dy), min(dh, dh - dy)
            xs, xe = max(0, -dx), min(dw, dw - dx)
            if ys >= ye or xs >= xe:
                continue
            tgt = out[:, ys + dy:ye + dy, xs + dx:xe + dx]
            out[:, ys + dy:ye + dy, xs + dx:xe + dx] = np.fmin(tgt, src[:, ys:ye, xs:xe])
    return out


def depth_bands(cells):
    """每格深度 → 分带 0..K-1（≥最后一个上界含 +inf = K-1）；nan → AUX_UNKNOWN。"""
    bands = np.searchsorted(np.asarray(DEPTH_BAND_EDGES_M, dtype=np.float32),
                            np.nan_to_num(cells, nan=np.inf), side="right").astype(np.uint8)
    bands[np.isnan(cells)] = AUX_UNKNOWN
    return bands


def aux_payload(episodes, size, envelope_radius_m):
    """算 prev_index / prev_action / 深度分带；返回 (prev_index, prev_action, depth 或 None, grid)。"""
    out_w, out_h = size
    grid = (out_h // DEPTH_GRID_DOWN, out_w // DEPTH_GRID_DOWN)
    flat = [(ep, s) for ep in episodes for s in ep["samples"]]
    n = len(flat)
    prev_index = np.arange(n, dtype=np.uint32)
    prev_action = np.zeros(n, dtype=np.uint8)
    for ep in episodes:
        by_step = {s["step"]: ep["sample_start"] + i for i, s in enumerate(ep["samples"])}
        for i, s in enumerate(ep["samples"]):
            g = ep["sample_start"] + i
            # 上一步被 --drop-collision 丢掉时 step-1 不在 by_step：同首帧处理（指向自身）。
            prev_index[g] = by_step.get(s["step"] - 1, g)
            prev_action[g] = s["prev_action"]
    have = [s["depth"] is not None for _, s in flat]
    if not any(have):
        return prev_index, prev_action, None, grid
    if not all(have):
        die("深度文件部分缺失（%d/%d 个样本有 depth_file）：一个包要么全有深度要么全无"
            % (sum(have), n))
    depth = np.empty((n, grid[0], grid[1]), dtype=np.uint8)
    chunk = 1024  # 分块做包络膨胀，内存按块算
    for start in range(0, n, chunk):
        part = flat[start:start + chunk]
        cells = np.empty((len(part), grid[0], grid[1]), dtype=np.float32)
        for k, (ep, s) in enumerate(part):
            d = DEPTH_LOADERS[s["depth_format"]](s["depth"])
            if d.shape != (ep["src_h"], ep["src_w"]):
                die("%s: 深度尺寸 %r 与相机 %dx%d 不一致"
                    % (s["depth"], d.shape, ep["src_w"], ep["src_h"]))
            cells[k] = cell_min_depth(d, grid)
        ep0 = part[0][0]
        fx = ep0.get("camera_fx")
        focal = float(fx) if fx else (ep0["src_w"] / 2.0) / np.tan(np.radians(DEFAULT_HFOV_DEG / 2))
        cells = dilate_by_envelope(cells, focal, ep0["src_w"] / grid[1], envelope_radius_m)
        depth[start:start + len(part)] = depth_bands(cells)
    return prev_index, prev_action, depth, grid


def write_aux_sidecar(out_dir, uavr_path, prev_index, prev_action, depth, grid,
                      envelope_radius_m):
    """写 <out>/dataset.aux（UAVRAUX1，tmp + rename）；头里绑定 dataset.uavr 的 SHA256。"""
    n = len(prev_index)
    with open(uavr_path, "rb") as f:
        uavr_sha = hashlib.sha256(f.read()).digest()
    has_depth = depth is not None
    payload = prev_index.astype("<u4").tobytes() + prev_action.tobytes()
    if has_depth:
        payload += depth.tobytes(order="C")
    header = AUX_HEADER.pack(AUX_MAGIC, n, 1 if has_depth else 0,
                             grid[0] if has_depth else 0, grid[1] if has_depth else 0,
                             DEPTH_CLASSES, AUX_UNKNOWN, 0)
    header += uavr_sha + hashlib.sha256(payload).digest()
    header += struct.pack("<7f", *DEPTH_BAND_EDGES_M)
    header += struct.pack("<fI", envelope_radius_m, 0)
    assert len(header) == AUX_HEADER_SIZE
    path = os.path.join(out_dir, "dataset.aux")
    fd, tmp = tempfile.mkstemp(prefix=".dataset.aux.", suffix=".tmp", dir=out_dir)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(header)
            f.write(payload)
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
    p.add_argument("--allow-frozen-geometry", action="store_true",
                   help="【诊断专用】允许冻结版本（如 campus.v1）打包；仍强制全包版本一致")
    p.add_argument("--split-by-map", metavar="SALT", default=None,
                   help="【诊断专用】按 geometry_gid 哈希 70/15/15 重划 split，"
                        "无视拓扑族归属。只用于回答「同族留出能不能学」，"
                        "混进常规实验会让跨族结论失效；必须配 --allow-diagnostic-split")
    p.add_argument("--allow-diagnostic-split", action="store_true",
                   help="显式允许 map-hash:*/diag* 诊断性划分（读端默认拒绝）")
    p.add_argument("--allow-mixed-geometry", action="store_true",
                   help="【跨族合并实验】允许不同几何族（柱阵/森林/3DGS）混包：只放开族身份字段"
                        "（geometry/appearance/split_salt/task_version…），hold_s/window_from/"
                        "dynamics_profile 仍必须一致；meta.geometry_versions 记每族计数与 salt")
    p.add_argument("--envelope-radius", type=float, default=DEFAULT_ENVELOPE_RADIUS_M,
                   metavar="M", help="深度分带目标的包络膨胀半径（m），E1 机体 0.102；0 = 不膨胀")
    p.add_argument("episodes", nargs="+", help="一个或多个 collect.py episode 目录")
    args = p.parse_args()
    if args.split_by_map and not args.allow_diagnostic_split:
        p.error("--split-by-map 生成 map-hash:* 诊断划分，需显式 --allow-diagnostic-split")
    if not (0 <= args.envelope_radius < 10):
        p.error("--envelope-radius 须在 [0,10) m")

    episodes, total, dropped, skipped = scan(
        args.episodes, args.drop_collision, args.exclude_collision_episodes,
        args.split_by_map, args.allow_frozen_geometry, args.allow_diagnostic_split,
        args.allow_mixed_geometry)
    split_scheme = "map-hash:%s" % args.split_by_map if args.split_by_map else None
    collision_filter = {
        "drop_collision_steps": bool(args.drop_collision),
        "exclude_collision_episodes": bool(args.exclude_collision_episodes),
        "excluded_episodes": skipped,
    }
    # 先校验侧车内容，再替换 dataset.uavr；有指令时先删旧侧车。
    # 这样侧车写失败时结果是「新数据集、没有侧车」（文本模型会显式报错），
    # 而不是「新数据集 + 旧侧车」被 C++ 按同集区间静默绑定。
    rows = (instruction_rows(episodes)
            if any(ep.get("instruction") and ep["instruction"].strip() for ep in episodes)
            else [])
    stale = os.path.join(args.out, "instructions.jsonl")
    if rows and os.path.exists(stale):
        os.remove(stale)
    # 侧车先算后写：深度文件坏了要在替换 dataset.uavr 之前报错。旧 dataset.aux 先删——
    # 它绑定的是旧 uavr 的 SHA，留着只会让读端报「SHA 不匹配」。
    prev_index, prev_action, depth, grid = aux_payload(episodes, args.size, args.envelope_radius)
    stale_aux = os.path.join(args.out, "dataset.aux")
    if os.path.exists(stale_aux):
        os.remove(stale_aux)
    aux_info = {"file": "dataset.aux", "format": AUX_MAGIC.decode(), "depth": depth is not None,
                "depth_grid": list(grid) if depth is not None else None,
                "band_edges_m": list(DEPTH_BAND_EDGES_M),
                "envelope_radius_m": args.envelope_radius}
    out_path, meta_bytes = write_dataset(args.out, episodes, total, args.size, dropped,
                                         split_scheme=split_scheme,
                                         collision_filter=collision_filter,
                                         allow_mixed=args.allow_mixed_geometry,
                                         aux_info=aux_info)
    apath = write_aux_sidecar(args.out, out_path, prev_index, prev_action, depth, grid,
                              args.envelope_radius)
    first = int((prev_index == np.arange(len(prev_index), dtype=np.uint32)).sum())
    print("wrote %s (prev chain: %d first/self frames; depth bands: %s)"
          % (apath, first, ("%dx%d grid, unknown %.1f%%" % (
              grid[0], grid[1], 100.0 * float((depth == AUX_UNKNOWN).mean())))
              if depth is not None else "none"))
    # 3DGS 数据集把 scene.json 放到 <out>/scene.json，自动评测/换机复现不必回找 episode。
    scene_paths = [os.path.join(ep["dir"], "scene.json") for ep in episodes
                   if ep.get("scene_json")]
    if scene_paths and len(scene_paths) == len(episodes) and all(
            os.path.isfile(p) for p in scene_paths):
        shas = set()
        for p in scene_paths:
            with open(p, "rb") as f:
                shas.add(hashlib.sha256(f.read()).hexdigest())
        if len(shas) == 1:
            with open(scene_paths[0], encoding="utf-8") as f:
                scene = json.load(f)
            for key in ("ply", "ply_path"):
                ply = scene.get(key)
                if isinstance(ply, str) and not os.path.isabs(ply):
                    scene[key] = os.path.abspath(
                        os.path.join(os.path.dirname(scene_paths[0]), ply))
            dst = os.path.join(args.out, "scene.json")
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(scene, f, ensure_ascii=False, indent=2)
            print("wrote %s (3DGS scene)" % dst)
        else:
            print("警告: episodes 的 scene.json 不一致，未复制到数据集根目录",
                  file=sys.stderr)
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
    print("split: policy=%s scheme=%s salt=%s"
          % (meta["split_policy"], meta["split_scheme"], meta["split_salt"]))
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
