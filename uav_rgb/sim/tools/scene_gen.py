#!/usr/bin/env python3
"""按 seed 确定性生成随机障碍世界（纯 stdlib + 文本替换）。

用法：
  python3 sim/tools/scene_gen.py --seed 42 --out /tmp/worlds/scene_42.sdf
  python3 sim/tools/scene_gen.py --self-check          # 100 个 seed 的自检

以 sim/worlds/uav_rgb.sdf 为模板，只替换「tall pillars」注释到 X3 模型之间的
障碍段落；四旋翼 / RGB 相机 / 深度相机 / contact 插件原样保留。同 seed 两次
生成字节级一致（sha256 相同）。

外观默认 v1（删除模板里 3 个固定纯色地标 + 随机光照/天空）；`--appearance legacy`
显式复现旧世界。外观不改变 pillars 几何/gid/split。

随机化契约：
  - 障碍 8–9 根，0.4×0.4×3 m，z∈[0,3]，挡住 z≈2.3 m 巡航高度；
    x∈[1.5, 5.0] 均匀递增（实测每步仅前进 0.105 m，穿完这段要 68–76 步；默认
    --steps 因此是 80，见 collect_parallel.py 的参数注释），
    y∈[-1.7, 1.7] 走三条车道：中心车道 ±0.1 m、左右车道 ±1.6 m。
  - 车道间距 1.6 m（柱宽 0.4 + 缝 1.2 m）；按「中-左-右」蛇形循环，
    柱 i 与 i+3 同车道、x 间距 3×0.4375=1.3125 m（纵向缝 0.9125 m ≥ MIN_GAP）。
    走到下一根同车道障碍约 1.3 m，对应「每前进 1–1.5 m 做一次左右选择」。
    v3 把缝从 0.902 放宽到 1.2 m：机体外接直径 0.711 m，单侧余量从 9.5 cm
    提到 24.5 cm（T1，2026-10-30）。
  - 第一根在中心车道（|y|≤0.25）作「门柱」：teacher 只在中间 ±10.9°、
    1.5 m 内转向；出生点 x=-0.5 时相机（x≈-0.3）到门柱约 1.6 m，Step 0 附近
    就触发避障。
  - 出生区 x<1.0 无遮挡（柱体最近边 x≥1.3）。
  - 任意两柱在 x 或 y 方向至少有一条 ≥0.9 m 的缝（同车道纵向缝仍由 x 间距
    保证，最小 0.9125 m）。机体真实外接直径是 0.711 m
    （旋翼中心 hypot(0.13,0.22)=0.256 m + 桨盘半径 0.1 m，再乘 2），
    横向车道缝 1.2 m 留 0.2445 m 单侧余量；每根柱到 y=±3 走廊边界至少留
    1.1 m。不满足则整组重采样，最多 100 次。
  - 同次生成在 <out>.json 写 scene_id / seed / 障碍清单 / 净空检查，供 manifest 用。

注：任务要求 8–12 根，但 3.5×3.0 m 的密布区 + 全对 ≥0.9 m 缝（柱心间距
Chebyshev ≥1.3 m）最多只能放 3×3=9 根；10–12 根必然出现一对缝隙 <0.9 m，
因此采样上限取 9（详见 sim/README.md §4.6.1）。三条车道占 y∈[-1.7, 1.7]，
边界净空 3−1.7−0.2=1.1 m。
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import sys

import landmarks

# 几何语义版本：改柱子/车道/spawn 规则必须 bump。split_manifest 直接用它当
# SPLIT_SALT，collect 写进 episode manifest，preprocess 拒绝跨版本混包。
# v4：删除 safe 出生模式（相机贴柱面导致的纯色废帧）。生成规则变了，
#     旧 v3 数据不能再和新数据混包，gid 也随之全部重排。
# v5：landmark 改成「两根同色柱分居左右 + 关系词指令」（任务语义变了，
#     和 v4 的「异色柱 + 目标色指令」训练目标不同，不能混包）。
GEOMETRY_VERSION = "uavrgb.geom.v5"

PILLAR_SIZE = 0.4
PILLAR_HEIGHT = 3.0
PILLAR_Z = PILLAR_HEIGHT / 2.0
PILLAR_X_RANGE = (1.5, 5.0)
PILLAR_Y_RANGE = (-1.7, 1.7)
Y_RANGE = (-3.0, 3.0)      # 走廊边界（比密布区宽），边界净空按此校验
LANE_SEP = 1.6             # 车道中心距 = 柱宽 0.4 + 缝隙 1.2（T1 放宽几何）
LANE_CENTER_ABS_MAX = 0.1  # 中心车道摆动上限：0.1+1.6 ≤ 1.7
PILLAR_COUNT_RANGE = (8, 9)  # 10–12 根违反全对缝隙契约（3×3 密堆上限）
GATE_Y_ABS_MAX = 0.25   # 门柱 |y| 上限：teacher 只对正前方 1.5 m 内障碍转向
MIN_GAP = 0.9
SPAWN_CLEAR_X = 1.0
# 出生点后移 0.5 m：门柱在 x=1.5，原来 1.5 m 内要横移 ~0.8 m 才能避开，
# 实测 15/24 集在 x≈1.05 擦柱。后移给转向多 0.5 m 的纵向空间。
SPAWN_X = -0.5
SPAWN_Z = 0.15
SPAWN_BODY_HALF = 0.235 + 0.01  # 0.47 m 机身方框 + 1 cm 余量
# SDF 的四个旋翼中心与桨盘半径；出生净空必须按真实几何查，不能只查机身方框。
ROTOR_OFFSETS = ((0.13, -0.22), (0.13, 0.22), (-0.13, 0.2), (-0.13, -0.2))
ROTOR_RADIUS = 0.1
CRUISE_HEIGHT = 2.3
MAX_ATTEMPTS = 100

SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SIM_DIR)
DEFAULT_TEMPLATE = os.path.join(SIM_DIR, "worlds", "uav_rgb.sdf")
# 载具 -> 默认模板。模板里的载具模型名必须与键一致（_apply_spawn 按它找 pose）。
# x3 是默认载具，旧命令行为不变；e1_mini 是 SITL 固件在环的 E1 MINI（CfSitlSystem 插件）。
VEHICLES = {
    "x3": {"template": DEFAULT_TEMPLATE, "dynamics_profile": "x3_v1"},
    "e1_mini": {"template": os.path.join(SIM_DIR, "worlds", "uav_rgb_e1.sdf"),
                "dynamics_profile": "cf_sitl_v1"},
}
DEFAULT_VEHICLE = "x3"
START_MARK = "    <!-- tall pillars"
END_MARK = "    <!-- X3-equivalent quadrotor"
# E1 模板里可选相机（chase/top）的包裹标记；--no-aux-cameras 整块删掉（采集提速，录 demo 时才开）。
AUX_CAMERA_RE = re.compile(r"\n    <!-- aux-camera:.*?<!-- /aux-camera -->", re.S)
# 学生相机与配对深度相机的 update_rate（只匹配 name="camera"/"depth" 的两个传感器）。
CAMERA_RATE_RE = re.compile(
    r'(<sensor name="(?:camera|depth)" type="(?:camera|depth_camera)">\s*<pose>[^<]*</pose>'
    r'\s*<always_on>1</always_on>\s*<update_rate>)([0-9.]+)(</update_rate>)')
# 世界物理里的目标实时因子（两种载具的模板各 1 处）。--rtf 只改这一个数。
RTF_RE = re.compile(r"(<real_time_factor>)[^<]*(</real_time_factor>)")
# 外观版本：只改材质/光照/固定地标，不动 pillars 几何 → geometry_gid 与旧数据都保留。
# legacy 显式复现改动前逐字节输出（含 3 个固定纯色方块）；v1 删掉它们并随机化光照/天空。
# 2026-11-04 起新场景默认 v1：固定地标是每集重复的捷径，且不在教师真值里。
# 背景见 docs/worklog/2026-11-03-sim-scene-realism-proposal.md §1.1 G8/G9。
APPEARANCE_LEGACY = "legacy"
APPEARANCE_V1 = "v1"
APPEARANCE_DEFAULT = APPEARANCE_V1
# v1 世界内容版本：本次把 base/l1 新默认 v1 的太阳方位从单侧 [-160,-20]
# 改为全圆 [-180,180]，旧 v1 世界不能按 seed 逐字节复现。appearance_version
# 仍写开关值（保持与 test_spawn/preprocess 的既有口径兼容），另写 schema 字段
# 标识内容版本。校园任务有已冻结的数据，仍用单侧太阳（FROZEN 版本）。
APPEARANCE_SCHEMA_VERSION = "uavrgb.appearance.v2"
APPEARANCE_SCHEMA_VERSION_FROZEN = "uavrgb.appearance.v1"
LANDMARK_MARK = "    <!-- colored landmarks in front of the spawn (forward = +X) -->"
LIGHT_MARK = '    <light type="directional" name="sun">'
LIGHT_END_MARK = "    </light>"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _rel(path):
    try:
        return os.path.relpath(path, REPO_ROOT)
    except ValueError:
        return path


def _pillar_gap(a, b):
    """两柱之间可通行缝宽：x / y 两个方向取较大者（柱为 0.4×0.4 方柱）。"""
    gx = max(0.0, abs(a["x"] - b["x"]) - PILLAR_SIZE)
    gy = max(0.0, abs(a["y"] - b["y"]) - PILLAR_SIZE)
    return max(gx, gy)


def validate_pillars(pillars):
    """校验数量、出生区、巡航高度遮挡与缝隙；不合格抛 ValueError。"""
    n = len(pillars)
    if not PILLAR_COUNT_RANGE[0] <= n <= PILLAR_COUNT_RANGE[1]:
        raise ValueError(f"pillar count {n} not in {PILLAR_COUNT_RANGE}")
    min_pair = None
    for i, p in enumerate(pillars):
        if not PILLAR_X_RANGE[0] <= p["x"] <= PILLAR_X_RANGE[1]:
            raise ValueError(f"{p['name']} x={p['x']} outside {PILLAR_X_RANGE}")
        if not PILLAR_Y_RANGE[0] <= p["y"] <= PILLAR_Y_RANGE[1]:
            raise ValueError(f"{p['name']} y={p['y']} outside {PILLAR_Y_RANGE}")
        if i == 0 and abs(p["y"]) > GATE_Y_ABS_MAX:
            raise ValueError(f"gate {p['name']} y={p['y']} exceeds "
                             f"±{GATE_Y_ABS_MAX}")
        if p["x"] - PILLAR_SIZE / 2 < SPAWN_CLEAR_X:
            raise ValueError(f"{p['name']} blocks spawn: x edge "
                             f"{p['x'] - PILLAR_SIZE / 2:.3f} < {SPAWN_CLEAR_X}")
        if p["z"] + PILLAR_HEIGHT / 2 <= CRUISE_HEIGHT:
            raise ValueError(f"{p['name']} does not block cruise height {CRUISE_HEIGHT}")
        side = min(p["y"] - Y_RANGE[0] - PILLAR_SIZE / 2,
                   Y_RANGE[1] - p["y"] - PILLAR_SIZE / 2)
        if side < MIN_GAP:
            raise ValueError(f"{p['name']} y={p['y']} leaves only {side:.3f} m "
                             f"to corridor boundary (need {MIN_GAP})")
        for q in pillars[:i]:
            gap = _pillar_gap(p, q)
            if gap < MIN_GAP:
                raise ValueError(f"{p['name']} vs {q['name']}: passable gap "
                                 f"{gap:.3f} m < {MIN_GAP}")
            min_pair = gap if min_pair is None else min(min_pair, gap)
    min_boundary = min(
        min(p["y"] - Y_RANGE[0] - PILLAR_SIZE / 2,
            Y_RANGE[1] - p["y"] - PILLAR_SIZE / 2) for p in pillars)
    return {"min_pair_gap_m": round(min_pair, 4),
            "min_boundary_gap_m": round(min_boundary, 4)}


def sample_pillars(seed):
    """random.Random(seed) 采样；整组不满足契约则重采样，最多 MAX_ATTEMPTS 次。

    8–9 根沿 x 均匀递增；y 走「中心-侧-反侧」三车道蛇形，第二根随机选左/右。
    三条车道间距 1.6 m，柱 i 与 i+3 同车道且 x 间距 ≥1.3 m，构造上满足缝隙
    契约；validate_pillars 只做兜底。rng 调用顺序固定（n、center、side），
    同 seed 结果字节级一致。
    """
    rng = random.Random(seed)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        n = rng.randint(*PILLAR_COUNT_RANGE)
        center = round(rng.uniform(-LANE_CENTER_ABS_MAX, LANE_CENTER_ABS_MAX), 3)
        side = rng.choice((-1, 1))
        lanes = (center, center + side * LANE_SEP, center - side * LANE_SEP)
        step = (PILLAR_X_RANGE[1] - PILLAR_X_RANGE[0]) / (n - 1)
        pillars = []
        for idx in range(n):
            pillars.append({
                "name": f"pillar_{idx}",
                "x": round(PILLAR_X_RANGE[0] + idx * step, 3),
                "y": round(lanes[idx % 3], 3),
                "z": PILLAR_Z,
                "size": [PILLAR_SIZE, PILLAR_SIZE, PILLAR_HEIGHT],
            })
        try:
            clearance = validate_pillars(pillars)
        except ValueError:
            continue
        return pillars, clearance, attempt
    raise RuntimeError(f"seed={seed}: {MAX_ATTEMPTS} 次重采样仍未满足缝隙契约")


def _spawn_collides(x, y, pillars):
    """出生点是否与柱子重叠：0.47 m 机身方框 + 四个桨盘（半径 0.1）。

    只查机身方框会放进「机身没碰、桨盘已进柱」的 safe 出生点（review 实测
    0.30 m 面距时桨盘只剩 7 cm 余量）；按真实旋翼中心查更准。
    """
    for p in pillars:
        if (abs(x - p["x"]) < PILLAR_SIZE / 2 + SPAWN_BODY_HALF and
                abs(y - p["y"]) < PILLAR_SIZE / 2 + SPAWN_BODY_HALF):
            return True
        for rx, ry in ROTOR_OFFSETS:
            dx = max(0.0, abs(x + rx - p["x"]) - PILLAR_SIZE / 2)
            dy = max(0.0, abs(y + ry - p["y"]) - PILLAR_SIZE / 2)
            if dx * dx + dy * dy < ROTOR_RADIUS * ROTOR_RADIUS:
                return True
    return False


def make_spawn(pillars, seed, mode="default"):
    """返回出生位姿。mode=default/random。

    2026-11-01 删除 safe 模式：plan_action 只在「±85° 内每个方向都被柱子挡住」时
    才返回 SAFE，而那种位姿的相机必然贴着柱面——实测 100/100 个 seed 的距离都是
    0.100 m，60° 视场被填满，产出 4000/10442 步纯色 SAFE 帧。全格搜索也找不到
    「plan_action=SAFE 且视野 ≥0.8 m」的位姿（0/100）。SAFE 样本改由 astar 集提供
    （到达 waypoint + 被围住，占 18.5%）。
    """
    if mode == "default":
        return {"x": SPAWN_X, "y": 0.0, "z": SPAWN_Z, "yaw_deg": 0.0, "mode": mode}
    if mode != "random":
        raise ValueError(f"unknown spawn mode {mode!r}")
    rng = random.Random(seed ^ 0x5AFE)   # 独立 RNG，不能扰动 sample_pillars 的序列
    # 出生在门柱与侧柱之间的缝中心附近（|y|≈0.8）：直线就能穿过第一排，
    # 不再需要 1.5 m 内横移 0.8 m。y=0 的起始留给 default 模式。
    y = rng.choice((-1.0, 1.0)) * round(rng.uniform(0.72, 0.88), 3)
    return {"x": SPAWN_X, "y": y, "z": SPAWN_Z,
            "yaw_deg": round(rng.uniform(-5.0, 5.0), 1), "mode": mode}


VEHICLE_POSE_RE = re.compile(
    r'(<model name="(' + "|".join(map(re.escape, VEHICLES)) + r')">\s*<pose>)([^<]*)(</pose>)')
X3_POSE_RE = VEHICLE_POSE_RE   # 旧名字保留给外部引用


def vehicle_of_template(template):
    """模板里是哪种载具：按 VEHICLES 的模型名找；找不到或多个都报错。"""
    found = [m.group(2) for m in VEHICLE_POSE_RE.finditer(template)]
    if len(found) != 1:
        raise ValueError(f"template 里载具模型匹配到 {found}，应恰好 1 个（{sorted(VEHICLES)}）")
    return found[0]


def _apply_spawn(world, spawn):
    """只替换载具模型（x3 / e1_mini）的第一条 pose；控制 spawn 不影响其它模板内容。"""
    def repl(m):
        return (m.group(1) + f"{spawn['x']:g} {spawn['y']:g} {spawn['z']:g} "
                f"0 0 {spawn['yaw_deg'] * 3.141592653589793 / 180.0:.6f}" + m.group(4))
    world, n = VEHICLE_POSE_RE.subn(repl, world, count=1)
    if n != 1:
        raise ValueError(f"template 载具 pose 匹配到 {n} 处，应为 1")
    return world


def set_camera_rate(world, hz):
    """改学生 RGB 与配对深度相机的 update_rate（两者必须同频，配对误差才接近 0）。"""
    if not (hz > 0 and math.isfinite(hz)):
        raise ValueError(f"camera rate 无效: {hz}")
    world, n = CAMERA_RATE_RE.subn(lambda m: f"{m.group(1)}{hz:g}{m.group(3)}", world)
    if n != 2:
        raise ValueError(f"template 里学生相机/深度相机 update_rate 匹配到 {n} 处，应为 2")
    return world


def set_real_time_factor(world, rtf):
    """改世界物理里的目标实时因子 <real_time_factor>（--rtf，W1-0）。

    目标 RTF 只能压低、不能抬高：仿真本来就跑不到它（软件渲染实测约 0.8）时无效。用途是让慢策略（A* 有缓存的 K=3、
    大模型）在仿真时间里少承担一段延迟：墙钟耗时 × RTF = 仿真延迟。几何与 gid 不受影响。
    """
    if not (rtf > 0 and math.isfinite(rtf)):
        raise ValueError(f"rtf 无效: {rtf}")
    world, n = RTF_RE.subn(lambda m: f"{m.group(1)}{rtf:g}{m.group(2)}", world)
    if n != 1:
        raise ValueError(f"template 里 real_time_factor 匹配到 {n} 处，应为 1")
    return world


def camera_rate_of(world):
    """生成结果里学生相机的 update_rate（Hz）；没匹配到返回 None。"""
    m = CAMERA_RATE_RE.search(world)
    return float(m.group(2)) if m else None


def strip_aux_cameras(world):
    """删掉 aux-camera 标记包裹的 chase/top 相机模型（只有 E1 模板有标记）。"""
    world, n = AUX_CAMERA_RE.subn("", world)
    if n < 1:
        raise ValueError("template 没有 aux-camera 标记（X3 模板不支持 --no-aux-cameras）")
    return world


def _pillar_xml(p):
    # L1 彩色 landmark：p["color"] 存在时按 landmarks.COLOR_RGB 上色，否则灰柱。
    rgb = landmarks.COLOR_RGB.get(p.get("color"), (0.6, 0.6, 0.6))
    mat = f"{rgb[0]:g} {rgb[1]:g} {rgb[2]:g} 1"
    return f"""    <model name="{p['name']}">
      <static>true</static>
      <pose>{p['x']:.3f} {p['y']:.3f} {PILLAR_Z:.1f} 0 0 0</pose>
      <link name="link">
        <collision name="collision"><geometry><box><size>{PILLAR_SIZE:g} {PILLAR_SIZE:g} {PILLAR_HEIGHT:g}</size></box></geometry></collision>
        <visual name="visual">
          <geometry><box><size>{PILLAR_SIZE:g} {PILLAR_SIZE:g} {PILLAR_HEIGHT:g}</size></box></geometry>
          <material><ambient>{mat}</ambient><diffuse>{mat}</diffuse><specular>{mat}</specular></material>
        </visual>
      </link>
    </model>"""


def appearance_params(seed, full_sun=True):
    """v1 的光照/天空参数；独立 RNG 流，不扰动 sample_pillars/make_spawn。

    full_sun=False 冻结 2026-11-03 的单侧太阳，供已冻结的校园/资产校园数据复现。
    """
    rng = random.Random(seed ^ 0x4F0EA7)
    if full_sun:
        # 全圆方位：-180..180 覆盖左右两侧，避免所有场景太阳都从同一侧来。
        az = math.radians(rng.uniform(-180.0, 180.0))   # 太阳方位（+X 前方为 0°）
    else:
        az = math.radians(rng.uniform(-160.0, -20.0))   # 冻结版本：只有右侧受光
    el = math.radians(rng.uniform(25.0, 70.0))      # 太阳高度角
    intensity = rng.uniform(0.60, 0.95)
    warm = rng.uniform(-0.04, 0.04)                 # 正=偏暖（红多蓝少）
    return {
        "light_direction": [round(-math.cos(el) * math.cos(az), 4),
                            round(-math.cos(el) * math.sin(az), 4),
                            round(-math.sin(el), 4)],
        "light_diffuse": [round(min(1.0, intensity * (1.0 + warm)), 4),
                          round(intensity, 4),
                          round(max(0.0, intensity * (1.0 - warm)), 4)],
        "light_specular": round(rng.uniform(0.10, 0.30), 4),
        "ambient": round(rng.uniform(0.25, 0.45), 4),
        "background": [round(rng.uniform(0.45, 0.72), 4),
                       round(rng.uniform(0.55, 0.78), 4),
                       round(rng.uniform(0.70, 0.92), 4)],
        # 落盘可复核的太阳参数；方向向量的 y 分量符号即受光侧。
        "sun_azimuth_deg": round(math.degrees(az), 2),
        "sun_elevation_deg": round(math.degrees(el), 2),
        "shadows": True,
    }


def _apply_appearance(prefix, seed, appearance, full_sun=True):
    """对柱子之前的世界头部做外观替换；legacy 原样返回（必须逐字节不变）。"""
    if appearance == APPEARANCE_LEGACY:
        return prefix, None
    if appearance != APPEARANCE_V1:
        raise ValueError(f"unknown appearance {appearance!r}")
    for mark in (LANDMARK_MARK, LIGHT_MARK, LIGHT_END_MARK):
        if prefix.count(mark) != 1:
            raise ValueError(f"template marker {mark!r} count != 1")
    p = appearance_params(seed, full_sun)
    # 1) 删掉 3 个固定纯色方块：它们不在 pillars/gid/教师真值里，学生却看得见（G9）。
    prefix = prefix[:prefix.index(LANDMARK_MARK)]
    # 2) 换掉固定方向光，并补 <scene>（ambient/background/shadows）。
    i = prefix.index(LIGHT_MARK)
    j = prefix.index(LIGHT_END_MARK, i) + len(LIGHT_END_MARK)
    d, dif, bg = p["light_direction"], p["light_diffuse"], p["background"]
    block = (
        "    <scene>\n"
        f"      <ambient>{p['ambient']:g} {p['ambient']:g} {p['ambient']:g} 1</ambient>\n"
        f"      <background>{bg[0]:g} {bg[1]:g} {bg[2]:g} 1</background>\n"
        "      <shadows>true</shadows>\n"
        "    </scene>\n"
        '    <light type="directional" name="sun">\n'
        "      <cast_shadows>true</cast_shadows>\n"
        "      <pose>0 0 10 0 0 0</pose>\n"
        f"      <diffuse>{dif[0]:g} {dif[1]:g} {dif[2]:g} 1</diffuse>\n"
        f"      <specular>{p['light_specular']:g} {p['light_specular']:g} "
        f"{p['light_specular']:g} 1</specular>\n"
        "      <attenuation>\n"
        "        <range>1000</range>\n"
        "        <constant>0.9</constant>\n"
        "        <linear>0.01</linear>\n"
        "        <quadratic>0.001</quadratic>\n"
        "      </attenuation>\n"
        f"      <direction>{d[0]:g} {d[1]:g} {d[2]:g}</direction>\n"
        "    </light>"
    )
    return prefix[:i] + block + prefix[j:], p


def strip_velocity_control(world):
    """cf_sil 用：删掉 MulticopterVelocityControl，避免它的 Actuators component
    覆盖 transport 电机指令。只删这一个 plugin，保留 odometry/motor model。"""
    out, n = re.subn(
        r'\n\s*<plugin filename="gz-sim-multicopter-control-system".*?</plugin>',
        "", world, flags=re.S)
    if n != 1:
        raise ValueError(f"template 里 MulticopterVelocityControl 匹配到 {n} 处，应为 1")
    return out


def build_scene(template, seed, scene_id, spawn_mode="default", task="base",
                appearance=APPEARANCE_LEGACY, density=None, blocks=None):
    """返回 (world_text, meta)。template 只读，任何标记异常都直接报错。

    task="l1" 时额外给两根柱染同一色并写 relation/instruction/waypoint；
    task="base" 保持原几何与人任务无关的 meta 不变。
    task="forest_e1" 转给 scene_forest（E1 随机圆柱森林，density 必填，只有 default 出生）。
    task="course_base" 转给 scene_course（K 个 base 柱阵块首尾相接的长课程，blocks 必填，只有 default 出生）。
    appearance="v1" 只改头部（删固定地标、随机光照/天空），pillars 几何与 gid 不变。
    Python API 默认仍是 legacy 以复现旧调用；命令行 --appearance 默认已改为 v1，
    新代码请显式传 APPEARANCE_DEFAULT。
    """
    if appearance not in (APPEARANCE_LEGACY, APPEARANCE_V1):
        raise ValueError(f"unknown appearance {appearance!r}")
    if density is not None and task != "forest_e1":
        raise ValueError(f"density 只对 task forest_e1 有效（task={task!r}）")
    if blocks is not None and task != "course_base":
        raise ValueError(f"blocks 只对 task course_base 有效（task={task!r}）")
    if task == "campus":
        import campus
        return campus.build(template, seed, scene_id, appearance)
    if task == "campus_assets":
        import asset_campus
        return asset_campus.build(template, seed, scene_id, appearance)
    if task == "forest_e1":
        import scene_forest
        if density is None:
            raise ValueError("task forest_e1 需要 density（sparse/medium/dense）")
        if spawn_mode != "default":
            raise ValueError("forest_e1 只支持 default 出生（走廊保证按固定出生点算）")
        return scene_forest.build_scene(template, seed, scene_id, density, appearance)
    if task == "course_base":
        import scene_course
        if blocks is None:
            raise ValueError("task course_base 需要 blocks（2–6）")
        if spawn_mode != "default":
            raise ValueError("course_base 只支持 default 出生（第 0 块与 base default 逐项相同）")
        return scene_course.build_scene(template, seed, scene_id, blocks, appearance)
    if task not in ("base", "l1"):
        raise ValueError(f"unknown task {task!r}")
    for mark in (START_MARK, END_MARK):
        if template.count(mark) != 1:
            raise ValueError(f"template marker {mark!r} count != 1")
    start = template.index(START_MARK)
    end = template.index(END_MARK)
    if end <= start:
        raise ValueError("template markers out of order")
    pillars, clearance, attempts = sample_pillars(seed)
    spawn = make_spawn(pillars, seed, spawn_mode)
    target_color = relation = instruction = waypoint = None
    if task == "l1":
        target_color, relation, instruction, waypoint = landmarks.assign_landmarks(
            pillars, seed, spawn)
    lines = [f"    <!-- scene_gen: scene_id={scene_id} seed={seed} "
             f"pillars={len(pillars)}; generated, do not edit -->"]
    lines += [_pillar_xml(p) for p in pillars]
    lines.append("")
    head, appearance_used = _apply_appearance(template[:start], seed, appearance)
    world = head + "\n".join(lines) + "\n" + template[end:]
    world = _apply_spawn(world, spawn)
    vehicle = vehicle_of_template(template)
    meta = {
        "schema": "navlab.scene.v1",
        "scene_id": scene_id,
        "seed": seed,
        "generator": "sim/tools/scene_gen.py",
        # 数据版本：与 SPLIT_SALT 同源，preprocess/audit 用它拦住 v2/v3 混包。
        "geometry_version": GEOMETRY_VERSION,
        # 外观不参与 geometry_gid：legacy 数据与 v1 数据几何可以互换，
        # 但外观不同不能混在同一次训练里。
        "appearance_version": appearance,
        # requested=调用方要的，effective=实际落进 SDF 的（campus 任务强制 v1）。
        "requested_appearance": appearance,
        "effective_appearance": appearance,
        "camera_profile": "rgb320x240_fov60_v1",
        # 载具只换模型不换几何：gid/GEOMETRY_VERSION 不变，靠 vehicle/dynamics_profile 区分数据。
        "vehicle": vehicle,
        "dynamics_profile": VEHICLES[vehicle]["dynamics_profile"],
        "spawn": spawn,
        "template_sha256": sha256(template.encode("utf-8")),
        "world_sha256": sha256(world.encode("utf-8")),
        "pillar_count": len(pillars),
        "pillars": pillars,
        "sampling_attempts": attempts,
        "clearance": {
            "min_gap_required_m": MIN_GAP,
            "min_pair_gap_m": clearance["min_pair_gap_m"],
            "min_boundary_gap_m": clearance["min_boundary_gap_m"],
            "corridor_y": list(Y_RANGE),
            "pillar_x_range": list(PILLAR_X_RANGE),
            "pillar_y_range": list(PILLAR_Y_RANGE),
            "lane_sep_m": LANE_SEP,
            "lane_gap_m": round(LANE_SEP - PILLAR_SIZE, 4),
            "pillar_count_range": list(PILLAR_COUNT_RANGE),
            "gate_y_abs_max_m": GATE_Y_ABS_MAX,
            "spawn_clear_before_x_m": SPAWN_CLEAR_X,
            "pillar_height_m": PILLAR_HEIGHT,
            "cruise_height_m": CRUISE_HEIGHT,
            "ok": True,
        },
    }
    if appearance_used is not None:
        meta["appearance"] = appearance_used
        meta["appearance_schema_version"] = APPEARANCE_SCHEMA_VERSION
    if task == "l1":
        meta.update({
            # v2：goal-stop。waypoint 是最终目标，teacher 在半径内输出 SAFE。
            # v1 的 route-choice（到 waypoint 后继续飞终点）实测 0/400 到 goal，
            # 已废弃。v3 在 v2 的成功口径上把指令换成关系词（同色柱对）。
            # v4：waypoint 从「朝走廊中心偏」改成「朝目标柱外侧偏」——v3 的两个
            # 候选点都落在出生航线上（274/274 场景不需要选边），新数据必须用 v4，
            # 不能和 v3 混包（preprocess 按 task_version 拦）。
            "task_version": "l1.v4",
            "target_color": target_color,
            # 两根同色柱的左右关系：L/I/S 消融的 shuffle 就是把它反过来。
            "relation": relation,
            "instruction": instruction,
            "waypoint": waypoint,
            "stop_radius": 0.5,
        })
    return world, meta


def _write_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _read_template(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def run_self_check(template_path, n_seeds):
    template = _read_template(template_path)
    bad = []
    counts = {}
    example = None
    sun_sides = set()
    # T1 契约：横向车道缝必须 ≥1.2 m（机体外接直径 0.711 m，单侧余量 ≥0.24 m）。
    lane_gap = LANE_SEP - PILLAR_SIZE
    if lane_gap < 1.2:
        bad.append((-1, f"lane gap {lane_gap:.3f} m < 1.2 m"))
    for seed in range(n_seeds):
        try:
            w1, m1 = build_scene(template, seed, seed)
            w2, _ = build_scene(template, seed, seed)
            # 新场景默认外观必须是 v1；legacy 仅用于显式复现旧数据。
            v1a, v1m = build_scene(template, seed, seed,
                                   appearance=APPEARANCE_DEFAULT)
            v1b, _ = build_scene(template, seed, seed,
                                 appearance=APPEARANCE_DEFAULT)
        except (ValueError, RuntimeError) as e:
            bad.append((seed, str(e)))
            continue
        h1 = sha256(w1.encode("utf-8"))
        h2 = sha256(w2.encode("utf-8"))
        if h1 != h2:
            bad.append((seed, f"non-deterministic {h1} != {h2}"))
        if m1["requested_appearance"] != APPEARANCE_LEGACY:
            bad.append((seed, "legacy 显式复现的 requested_appearance 不是 legacy"))
        # v1 只准改外观：几何必须与 legacy 一致，固定地标必须消失，场景必须有天空。
        if v1a != v1b:
            bad.append((seed, "v1 non-deterministic"))
        if v1m["pillars"] != m1["pillars"]:
            bad.append((seed, "v1 changed pillar geometry"))
        if 'name="landmark_' in v1a:
            bad.append((seed, "v1 仍有固定地标"))
        if "<scene>" not in v1a:
            bad.append((seed, "v1 缺 <scene>"))
        if '<light type="directional"' not in v1a:
            bad.append((seed, "v1 缺方向光"))
        # requested/effective 与内容版本必须落盘，否则数据身份不可追溯。
        if (v1m["requested_appearance"] != APPEARANCE_DEFAULT or
                v1m["effective_appearance"] != APPEARANCE_DEFAULT):
            bad.append((seed, "v1 requested/effective_appearance 不一致"))
        if v1m.get("appearance_schema_version") != APPEARANCE_SCHEMA_VERSION:
            bad.append((seed, "v1 缺 appearance_schema_version"))
        light = v1m.get("appearance") or {}
        if not light.get("shadows") or "background" not in light:
            bad.append((seed, "v1 光照参数缺 shadows/background"))
        # 太阳方位要覆盖左右两侧：direction 的 y 分量符号两向都出现。
        sun_sides.add((light.get("light_direction") or [0, 0, 0])[1] > 0)
        if not PILLAR_COUNT_RANGE[0] <= m1["pillar_count"] <= PILLAR_COUNT_RANGE[1]:
            bad.append((seed, f"pillar_count={m1['pillar_count']}"))
        counts[m1["pillar_count"]] = counts.get(m1["pillar_count"], 0) + 1
        if seed == 0:
            example = (h1, h2, m1["pillar_count"])
    if sun_sides != {True, False}:
        bad.append((-1, f"太阳方位未覆盖左右两侧: {sorted(sun_sides)}"))
    print(f"scene_gen self-check: {n_seeds - len(bad)}/{n_seeds} seeds OK, "
          f"pillar count distribution {dict(sorted(counts.items()))}")
    if example:
        print(f"  seed 0: sha256 a={example[0]}")
        print(f"          sha256 b={example[1]}  match={example[0] == example[1]} "
              f"pillars={example[2]}")
    for seed, why in bad[:10]:
        print(f"  FAIL seed={seed}: {why}", file=sys.stderr)
    if bad:
        print(f"scene_gen self-check: FAIL ({len(bad)} seeds)", file=sys.stderr)
        return 1
    print(f"scene_gen self-check: PASS（同 seed 字节级一致，障碍 "
          f"{PILLAR_COUNT_RANGE[0]}–{PILLAR_COUNT_RANGE[1]} 根，车道缝 "
          f"{LANE_SEP - PILLAR_SIZE:.2f} m，缝隙契约全过）")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, help="随机种子（同 seed 字节级一致）")
    p.add_argument("--out", help="输出 SDF 路径；同名 .json 写场景清单")
    p.add_argument("--scene-id", type=int, default=None,
                   help="EpisodeMeta.scene_id（默认=seed）")
    p.add_argument("--spawn-mode", choices=("default", "random"), default="default",
                   help="出生位姿：default=模板原点；random=y/yaw 随机")
    p.add_argument("--task", choices=("base", "l1", "campus", "campus_assets", "forest_e1", "course_base"), default="base",
                   help="base=原柱阵；l1=彩色 landmark + instruction/waypoint；"
                        "forest_e1=E1 随机圆柱森林（scene_forest.py，需 --density，建议 --vehicle e1_mini）；"
                        "course_base=K 个 base 柱阵块首尾相接的长课程（scene_course.py，需 --blocks）")
    p.add_argument("--density", choices=("sparse", "medium", "dense"), default=None,
                   help="forest_e1 的密度档（最窄缝 0.45/0.35/0.28 m），forest_e1 必填；其它任务不接受")
    p.add_argument("--blocks", type=int, default=None,
                   help="course_base 的块数 K（2–6），course_base 必填；其它任务不接受")
    p.add_argument("--appearance", choices=(APPEARANCE_LEGACY, APPEARANCE_V1),
                   default=APPEARANCE_DEFAULT,
                   help="v1=删固定地标+随机光照/天空（默认）；legacy=模板原样，仅用于复现旧数据")
    p.add_argument("--vehicle", choices=tuple(VEHICLES), default=None,
                   help="载具：x3=默认（旧行为）；e1_mini=SITL 固件在环的 E1 MINI。"
                        "决定默认模板；几何/gid/split 不随载具变")
    p.add_argument("--template", default=None,
                   help=f"模板 SDF（默认按 --vehicle 选；x3 为 {_rel(DEFAULT_TEMPLATE)}）")
    p.add_argument("--camera-rate", type=float, default=None,
                   help="学生 RGB 与深度相机的 update_rate（Hz）；缺省用模板值"
                        "（E1 模板 10 Hz：每个 hold 窗口只消费 1 帧，两路同频配对误差≈0）")
    p.add_argument("--rtf", type=float, default=None,
                   help="世界物理的目标实时因子 <real_time_factor>（W1-0）；缺省用模板值（1.0）。只能压低、不能抬高："
                        "仿真跑不到它时无效。给了才写进 meta.real_time_factor，默认输出逐位不变")
    p.add_argument("--no-aux-cameras", action="store_true",
                   help="不生成 chase/top 两路辅助相机（采集提速；录 demo 视频时不要加）。仅 E1 模板")
    p.add_argument("--no-velocity-control", action="store_true",
                   help="删除 MulticopterVelocityControl（cf_sil 用；保留电机模型/odometry）")
    p.add_argument("--self-check", action="store_true",
                   help="确定性 + 障碍数量 + 缝隙契约自检，不写文件")
    p.add_argument("--check-seeds", type=int, default=100,
                   help="--self-check 检查的 seed 数（默认 100）")
    args = p.parse_args()
    if args.template is None:
        args.template = VEHICLES[args.vehicle or DEFAULT_VEHICLE]["template"]
    if args.camera_rate is not None and not (math.isfinite(args.camera_rate) and args.camera_rate > 0):
        p.error("--camera-rate 必须是有限正数")
    if args.rtf is not None and not (math.isfinite(args.rtf) and args.rtf > 0):
        p.error("--rtf 必须是有限正数")
    if (args.task == "forest_e1") != (args.density is not None):
        p.error("--density 与 --task forest_e1 必须同时给")
    if args.task == "forest_e1" and args.spawn_mode != "default":
        p.error("--task forest_e1 只支持 --spawn-mode default")
    if (args.task == "course_base") != (args.blocks is not None):
        p.error("--blocks 与 --task course_base 必须同时给")
    if args.task == "course_base":
        import scene_course
        if not scene_course.BLOCKS_RANGE[0] <= args.blocks <= scene_course.BLOCKS_RANGE[1]:
            p.error("--blocks 必须在 %d–%d" % scene_course.BLOCKS_RANGE)
        if args.spawn_mode != "default":
            p.error("--task course_base 只支持 --spawn-mode default")

    if args.self_check:
        if args.check_seeds < 1:
            p.error("--check-seeds must be >= 1")
        return run_self_check(args.template, args.check_seeds)
    if args.seed is None or not args.out:
        p.error("需要 --seed 和 --out（或使用 --self-check）")
    scene_id = args.seed if args.scene_id is None else args.scene_id

    template = _read_template(args.template)
    template_vehicle = vehicle_of_template(template)
    if args.vehicle and args.vehicle != template_vehicle:
        p.error(f"--vehicle {args.vehicle} 与模板里的载具 {template_vehicle} 不一致")
    world, meta = build_scene(template, args.seed, scene_id, args.spawn_mode, args.task,
                              args.appearance, density=args.density, blocks=args.blocks)
    if args.no_velocity_control:
        world = strip_velocity_control(world)
        meta["velocity_control"] = "removed"
    else:
        meta["velocity_control"] = "template"
    if args.camera_rate is not None:
        world = set_camera_rate(world, args.camera_rate)
    if args.no_aux_cameras:
        world = strip_aux_cameras(world)
    if args.rtf is not None:
        world = set_real_time_factor(world, args.rtf)
        meta["real_time_factor"] = args.rtf   # 只在给了 --rtf 时写键：默认 meta 逐位不变
    if args.camera_rate is not None or args.no_aux_cameras or args.rtf is not None:
        meta["world_sha256"] = sha256(world.encode("utf-8"))
    # campus/asset_campus 自己写 meta，这里统一补载具与相机采样信息。
    meta.setdefault("vehicle", template_vehicle)
    meta["camera_rate_hz"] = camera_rate_of(world)
    meta["aux_cameras"] = not args.no_aux_cameras
    meta["template"] = _rel(os.path.abspath(args.template))
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    _write_atomic(out, world.encode("utf-8"))
    meta_path = os.path.splitext(out)[0] + ".json"
    _write_atomic(meta_path, (json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
                  .encode("utf-8"))
    print(f"scene_id={scene_id} seed={args.seed} pillars={meta['pillar_count']} "
          f"spawn={meta['spawn']} sha256={sha256(world.encode('utf-8'))}")
    print(f"  world -> {out}")
    print(f"  meta  -> {meta_path}")
    for pillar in meta["pillars"]:
        print(f"  {pillar['name']}: x={pillar['x']:.3f} y={pillar['y']:.3f} "
              f"size={pillar['size']}")
    c = meta["clearance"]
    if args.task in ("campus", "campus_assets"):
        print(f"  map={meta['map_id']} goal_radius={meta['goal_radius']} m ok={c['ok']}")
    else:
        print(f"  clearance: min_pair={c['min_pair_gap_m']:.3f} m "
              f"min_boundary={c['min_boundary_gap_m']:.3f} m "
              f"(required {c['min_gap_required_m']} m), attempts={meta['sampling_attempts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
