#!/usr/bin/env python3
"""按 seed 确定性生成随机障碍世界（纯 stdlib + 文本替换）。

用法：
  python3 sim/tools/scene_gen.py --seed 42 --out /tmp/worlds/scene_42.sdf
  python3 sim/tools/scene_gen.py --self-check          # 100 个 seed 的自检

以 sim/worlds/uav_rgb.sdf 为模板，只替换「tall pillars」注释到 X3 模型之间的
障碍段落；四旋翼 / RGB 相机 / 深度相机 / contact 插件原样保留。同 seed 两次
生成字节级一致（sha256 相同）。

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
import os
import random
import re
import sys

# 几何语义版本：改柱子/车道/spawn 规则必须 bump。split_manifest 直接用它当
# SPLIT_SALT，collect 写进 episode manifest，preprocess 拒绝跨版本混包。
GEOMETRY_VERSION = "uavrgb.geom.v3"

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
START_MARK = "    <!-- tall pillars"
END_MARK = "    <!-- X3-equivalent quadrotor"


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
    """返回出生位姿。mode=default/random/safe。

    safe：T2 用。在柱区里找一个 plan_action 真正判 SAFE 的位姿，让「停」类样本
    由教师规则天然产生，而不是人工造标签。候选点围绕前 3 根柱子取 ±0.5/±0.25 m
    偏移；不满足 AABB 净空或 plan_action 不是 SAFE 就换下一个，顺序由 seed 决定。
    找不到才退回粗网格；再找不到就报错（不静默给 default）。
    """
    if mode == "default":
        return {"x": SPAWN_X, "y": 0.0, "z": SPAWN_Z, "yaw_deg": 0.0, "mode": mode}
    rng = random.Random(seed ^ 0x5AFE)   # 独立 RNG，不能扰动 sample_pillars 的序列
    if mode == "random":
        # 出生在门柱与侧柱之间的缝中心附近（|y|≈0.8）：直线就能穿过第一排，
        # 不再需要 1.5 m 内横移 0.8 m。y=0 的起始留给 safe/default 模式。
        y = rng.choice((-1.0, 1.0)) * round(rng.uniform(0.72, 0.88), 3)
        return {"x": SPAWN_X, "y": y, "z": SPAWN_Z,
                "yaw_deg": round(rng.uniform(-5.0, 5.0), 1), "mode": mode}
    if mode != "safe":
        raise ValueError(f"unknown spawn mode {mode!r}")

    from teacher import plan_action, SAFE_NEUTRAL
    offsets = [(dx, dy) for dx in (-0.5, 0.5)
               for dy in (-0.25, -0.125, 0.0, 0.125, 0.25)]
    candidates = [(p["x"] + dx, p["y"] + dy) for p in pillars[:3] for dx, dy in offsets]
    rng.shuffle(candidates)
    for x, y in candidates:
        x, y = round(x, 3), round(y, 3)
        if not (PILLAR_X_RANGE[0] - 0.5 <= x <= PILLAR_X_RANGE[1] + 0.5):
            continue
        if not (PILLAR_Y_RANGE[0] <= y <= PILLAR_Y_RANGE[1]):
            continue
        if _spawn_collides(x, y, pillars):
            continue
        if plan_action(x, y, 0.0, pillars) == SAFE_NEUTRAL:
            return {"x": x, "y": y, "z": SPAWN_Z, "yaw_deg": 0.0, "mode": mode}
    # 粗网格兜底：候选偏移全不中时仍要给出真实 SAFE 位姿。
    for xi in range(8, 41):
        x = round(xi * 0.125, 3)
        for yi in range(-13, 14):
            y = round(yi * 0.125, 3)
            if _spawn_collides(x, y, pillars):
                continue
            if plan_action(x, y, 0.0, pillars) == SAFE_NEUTRAL:
                return {"x": x, "y": y, "z": SPAWN_Z, "yaw_deg": 0.0, "mode": mode}
    raise RuntimeError(f"seed={seed}: 找不到 SAFE 出生位姿")


X3_POSE_RE = re.compile(r'(<model name="x3">\s*<pose>)([^<]*)(</pose>)')


def _apply_spawn(world, spawn):
    """只替换 x3 模型的第一条 pose；控制 spawn 不影响其它模板内容。"""
    def repl(m):
        return (m.group(1) + f"{spawn['x']:g} {spawn['y']:g} {spawn['z']:g} "
                f"0 0 {spawn['yaw_deg'] * 3.141592653589793 / 180.0:.6f}" + m.group(3))
    world, n = X3_POSE_RE.subn(repl, world, count=1)
    if n != 1:
        raise ValueError(f"template x3 pose 匹配到 {n} 处，应为 1")
    return world


def _pillar_xml(p):
    return f"""    <model name="{p['name']}">
      <static>true</static>
      <pose>{p['x']:.3f} {p['y']:.3f} {PILLAR_Z:.1f} 0 0 0</pose>
      <link name="link">
        <collision name="collision"><geometry><box><size>{PILLAR_SIZE:g} {PILLAR_SIZE:g} {PILLAR_HEIGHT:g}</size></box></geometry></collision>
        <visual name="visual">
          <geometry><box><size>{PILLAR_SIZE:g} {PILLAR_SIZE:g} {PILLAR_HEIGHT:g}</size></box></geometry>
          <material><ambient>0.6 0.6 0.6 1</ambient><diffuse>0.6 0.6 0.6 1</diffuse><specular>0.6 0.6 0.6 1</specular></material>
        </visual>
      </link>
    </model>"""


def build_scene(template, seed, scene_id, spawn_mode="default"):
    """返回 (world_text, meta)。template 只读，任何标记异常都直接报错。"""
    for mark in (START_MARK, END_MARK):
        if template.count(mark) != 1:
            raise ValueError(f"template marker {mark!r} count != 1")
    start = template.index(START_MARK)
    end = template.index(END_MARK)
    if end <= start:
        raise ValueError("template markers out of order")
    pillars, clearance, attempts = sample_pillars(seed)
    spawn = make_spawn(pillars, seed, spawn_mode)
    lines = [f"    <!-- scene_gen: scene_id={scene_id} seed={seed} "
             f"pillars={len(pillars)}; generated, do not edit -->"]
    lines += [_pillar_xml(p) for p in pillars]
    lines.append("")
    world = template[:start] + "\n".join(lines) + "\n" + template[end:]
    world = _apply_spawn(world, spawn)
    meta = {
        "schema": "navlab.scene.v1",
        "scene_id": scene_id,
        "seed": seed,
        "generator": "sim/tools/scene_gen.py",
        # 数据版本：与 SPLIT_SALT 同源，preprocess/audit 用它拦住 v2/v3 混包。
        "geometry_version": GEOMETRY_VERSION,
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
    # T1 契约：横向车道缝必须 ≥1.2 m（机体外接直径 0.711 m，单侧余量 ≥0.24 m）。
    lane_gap = LANE_SEP - PILLAR_SIZE
    if lane_gap < 1.2:
        bad.append((-1, f"lane gap {lane_gap:.3f} m < 1.2 m"))
    for seed in range(n_seeds):
        try:
            w1, m1 = build_scene(template, seed, seed)
            w2, _ = build_scene(template, seed, seed)
        except (ValueError, RuntimeError) as e:
            bad.append((seed, str(e)))
            continue
        h1 = sha256(w1.encode("utf-8"))
        h2 = sha256(w2.encode("utf-8"))
        if h1 != h2:
            bad.append((seed, f"non-deterministic {h1} != {h2}"))
        if not PILLAR_COUNT_RANGE[0] <= m1["pillar_count"] <= PILLAR_COUNT_RANGE[1]:
            bad.append((seed, f"pillar_count={m1['pillar_count']}"))
        counts[m1["pillar_count"]] = counts.get(m1["pillar_count"], 0) + 1
        if seed == 0:
            example = (h1, h2, m1["pillar_count"])
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
    p.add_argument("--spawn-mode", choices=("default", "random", "safe"), default="default",
                   help="出生位姿：default=模板原点；random=y/yaw 随机；safe=柱区内真 SAFE 位姿")
    p.add_argument("--template", default=DEFAULT_TEMPLATE,
                   help=f"模板 SDF（默认 {_rel(DEFAULT_TEMPLATE)}）")
    p.add_argument("--self-check", action="store_true",
                   help="确定性 + 障碍数量 + 缝隙契约自检，不写文件")
    p.add_argument("--check-seeds", type=int, default=100,
                   help="--self-check 检查的 seed 数（默认 100）")
    args = p.parse_args()

    if args.self_check:
        if args.check_seeds < 1:
            p.error("--check-seeds must be >= 1")
        return run_self_check(args.template, args.check_seeds)
    if args.seed is None or not args.out:
        p.error("需要 --seed 和 --out（或使用 --self-check）")
    scene_id = args.seed if args.scene_id is None else args.scene_id

    template = _read_template(args.template)
    world, meta = build_scene(template, args.seed, scene_id, args.spawn_mode)
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
              f"size={PILLAR_SIZE:g}x{PILLAR_SIZE:g}x{PILLAR_HEIGHT:g}")
    c = meta["clearance"]
    print(f"  clearance: min_pair={c['min_pair_gap_m']:.3f} m "
          f"min_boundary={c['min_boundary_gap_m']:.3f} m "
          f"(required {c['min_gap_required_m']} m), attempts={meta['sampling_attempts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
