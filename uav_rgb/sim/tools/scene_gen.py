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
    x∈[1.5, 5.0] 均匀递增（20 步 × hold=1.0 实测行程约 5 m），
    y∈[-1.5, 1.5] 走三条车道：中心车道 ±0.2 m、左右车道 ±1.3 m。
  - 车道间距 1.302 m（柱宽 0.4 + 缝 0.902 m）；按「中-左-右」蛇形循环，
    柱 i 与 i+3 同车道、x 间距 ≥3×0.4375=1.3125 m（≥1.302 m 最小间距）。
    走到下一根同车道障碍约 1.3 m，对应「每前进 1–1.5 m 做一次左右选择」。
  - 第一根在中心车道（|y|≤0.25）作「门柱」：teacher 只在中间 ±10.9°、
    1.5 m 内转向；柱心 x=1.5 时相机（x≈0.2）深度约 1.1 m，Step 0 就触发避障。
  - 出生区 x<1.0 无遮挡（柱体最近边 x≥1.3）。
  - 任意两柱在 x 或 y 方向至少有一条 ≥0.9 m 的缝（旋翼跨度约 0.64 m）；
    每根柱到 y=±3 走廊边界也各留 ≥0.9 m。不满足则整组重采样，最多 100 次。
  - 同次生成在 <out>.json 写 scene_id / seed / 障碍清单 / 净空检查，供 manifest 用。

注：任务要求 8–12 根，但 3.5×3.0 m 的密布区 + 全对 ≥0.9 m 缝（柱心间距
Chebyshev ≥1.3 m）最多只能放 3×3=9 根；10–12 根必然出现一对缝隙 <0.9 m，
因此采样上限取 9（详见 sim/README.md §4.6.1）。三条车道已占满 y∈[-1.5, 1.5]
的可用宽度，边界净空 3−1.5−0.2=1.3 m。
"""
import argparse
import hashlib
import json
import os
import random
import sys

PILLAR_SIZE = 0.4
PILLAR_HEIGHT = 3.0
PILLAR_Z = PILLAR_HEIGHT / 2.0
PILLAR_X_RANGE = (1.5, 5.0)
PILLAR_Y_RANGE = (-1.5, 1.5)
Y_RANGE = (-3.0, 3.0)      # 走廊边界（比密布区宽），边界净空按此校验
LANE_SEP = 1.302           # 车道中心距 = 柱宽 0.4 + 缝隙 0.902（避开 0.9 浮点抖动）
LANE_CENTER_ABS_MAX = 0.19  # 中心车道摆动上限：0.19+1.302 ≤ 1.5
PILLAR_COUNT_RANGE = (8, 9)  # 10–12 根违反全对缝隙契约（3×3 密堆上限）
GATE_Y_ABS_MAX = 0.25   # 门柱 |y| 上限：teacher 只对正前方 1.5 m 内障碍转向
MIN_GAP = 0.9
SPAWN_CLEAR_X = 1.0
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
    三条车道间距 1.3 m，柱 i 与 i+3 同车道且 x 间距 ≥1.3 m，构造上满足缝隙
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


def build_scene(template, seed, scene_id):
    """返回 (world_text, meta)。template 只读，任何标记异常都直接报错。"""
    for mark in (START_MARK, END_MARK):
        if template.count(mark) != 1:
            raise ValueError(f"template marker {mark!r} count != 1")
    start = template.index(START_MARK)
    end = template.index(END_MARK)
    if end <= start:
        raise ValueError("template markers out of order")
    pillars, clearance, attempts = sample_pillars(seed)
    lines = [f"    <!-- scene_gen: scene_id={scene_id} seed={seed} "
             f"pillars={len(pillars)}; generated, do not edit -->"]
    lines += [_pillar_xml(p) for p in pillars]
    lines.append("")
    world = template[:start] + "\n".join(lines) + "\n" + template[end:]
    meta = {
        "schema": "navlab.scene.v1",
        "scene_id": scene_id,
        "seed": seed,
        "generator": "sim/tools/scene_gen.py",
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
          f"{PILLAR_COUNT_RANGE[0]}–{PILLAR_COUNT_RANGE[1]} 根，缝隙契约全过）")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, help="随机种子（同 seed 字节级一致）")
    p.add_argument("--out", help="输出 SDF 路径；同名 .json 写场景清单")
    p.add_argument("--scene-id", type=int, default=None,
                   help="EpisodeMeta.scene_id（默认=seed）")
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
    world, meta = build_scene(template, args.seed, scene_id)
    meta["template"] = _rel(os.path.abspath(args.template))
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    _write_atomic(out, world.encode("utf-8"))
    meta_path = os.path.splitext(out)[0] + ".json"
    _write_atomic(meta_path, (json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
                  .encode("utf-8"))
    print(f"scene_id={scene_id} seed={args.seed} pillars={meta['pillar_count']} "
          f"sha256={sha256(world.encode('utf-8'))}")
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
