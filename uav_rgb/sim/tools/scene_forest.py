#!/usr/bin/env python3
"""forest_e1：E1 MINI 用的随机圆柱森林场景族（按 seed 确定性生成，纯 stdlib）。

用法：
  python3 sim/tools/scene_forest.py --seed 42 --density medium --out <dir>/forest_42.sdf
  python3 sim/tools/scene_forest.py --self-check                 # 3 档 × 30 seed：确定性 + 缝宽契约 + 走廊
  python3 sim/tools/scene_forest.py --stats-out <dir>            # 20 seed × 3 档几何统计 + 3 张俯视 PNG（要 matplotlib）

为什么要新族：scene_gen 的柱阵按 X3 外接直径 0.711 m 定 1.2 m 车道缝，E1 MINI 外径 0.204 m
在里面几乎不用转向。本族把缝宽按 E1 尺度缩成 0.45/0.35/0.28 m 三档，圆柱半径 0.05–0.15 m
随机散布——做法同 EGO-Planner-v2 map_generator 的 RandomMapGenerateCylinder（均匀撒点 +
最小间距拒绝重采，random_forest_sensing.cpp:183-202），差别是：(1) 间距按柱面缝而不是柱心距，
缝宽才是机体能不能过的量；(2) 加了网格 A* 保证出生点到目标至少一条按 E1 半径膨胀后仍连通
的走廊（map_generator 没有这步，它靠把起点放在障碍区外 2 m、规划器自己绕）；(3) 显式 seed
（map_generator 用 random_device，:474；mockamap 才有 seed 参数，maps.cpp:18）。

与 scene_gen 的关系：复用它的模板标记（START_MARK/END_MARK）、外观 v1 随机化、sha256/原子写。
几何版本独立（GEOMETRY_VERSION_FOREST），meta 显式写 split_salt，旧柱阵的 SPLIT_SALT/gid/split
不受影响。pillars 每根同时给 radius（圆）与外接方框 size：split_manifest.geometry_gid 和旧的
方框教师（teacher.py / teacher_kino.py 的 _boxes）直接能读，只是把圆当方框会偏保守（角部多
0.41r ≤ 0.06 m），不会漏判碰撞。
"""
import argparse
import heapq
import json
import math
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_gen as sg  # noqa: E402
import split_manifest as sm  # noqa: E402

SIM_DIR = sg.SIM_DIR

# ---------------------------------------------------------------- 配置区（改这里；改几何常量必 bump 版本）
GEOMETRY_VERSION_FOREST = "uavrgb.forest_e1.v1"  # 独立于 sg.GEOMETRY_VERSION；也是本族的 split_salt
FAMILY = "forest_e1"
E1_PARAMS = os.path.join(SIM_DIR, "dynamics", "e1_mini.json")  # 载具半径单源：collision_envelope.radius_from_center
DEFAULT_TEMPLATE = os.path.join(SIM_DIR, "worlds", "uav_rgb_e1.sdf")  # E1 SITL 模板（gz agent 的文件）
# 三档「最窄可通行缝」= 任意两柱面间距下限（m）。任务口径按 E1 外径约 0.18 m 的 2.5×/2×/1.5×；
# 按 e1_mini.json 的外接圆直径 0.204 m 算是 2.2×/1.7×/1.4×，dense 单侧余量只有 3.8 cm，是刻意的难度上限。
DENSITY_GAP_M = {"sparse": 0.45, "medium": 0.35, "dense": 0.28}
# 每档柱数范围：4×3 m² 森林区、半径均值 0.10 m 时随机顺序吸附（RSA）的饱和数实测 25/34/45 根
# （10 seed 均值，见 docs/worklog 2026-11-07 fastlab-forest-e1），取其 55–73%：实测重采样 1 次即过，再高接受率陡降。
DENSITY_COUNT = {"sparse": (14, 18), "medium": (20, 25), "dense": (26, 33)}
DENSITY_DEFAULT = "medium"
TRUNK_RADIUS_M = (0.05, 0.15)  # 任务给定；EGO-Planner-v2 launch_gen.py:96-97 对 0.2 m 机体用 0.25–0.35 m×[0.5,1.5]，这里按室内盆栽/立柱尺度缩小
TRUNK_HEIGHT_M = 3.0           # 同旧柱阵 PILLAR_HEIGHT：目标高 2.0 m + SITL 起飞超调 0.21 m 仍在柱内
FOREST_X = (1.5, 5.5)          # 森林段 4 m：0.3 m/s 前飞约 13 s；起点同旧柱阵 1.5，顶视相机 (2.5,0,6) 取景不用动
FOREST_Y = (-1.5, 1.5)         # 走廊半宽 1.5（旧柱阵 1.7）；A* 只在带内找路，带外视为墙，否则绕场外一定更短
SPAWN = {"x": -0.5, "y": 0.0, "z": 0.15, "yaw_deg": 0.0, "mode": "default"}  # 同旧柱阵 default；柱面全在 x≥FOREST_X[0]，起飞段 2 m 无障碍
GOAL_PAST_FOREST_M = 0.5       # A* 目标 (FOREST_X[1]+0.5, 0)：穿出森林即可，之后空旷
CORRIDOR_MARGIN_M = 0.025      # 走廊 clearance = E1 半径 + 它 = 0.127 m，刚好小于 dense 缝的一半 0.14
GRID_M = 0.025                 # A* 栅格：dense 档走廊最窄处只剩 0.28−2×0.127=0.026 m 自由带，格距必须 ≤ 它
MAX_ATTEMPTS = 100             # 同 scene_gen；计数范围取在饱和以下时实测 1–3 次就过
TRUNK_RGB = (0.45, 0.36, 0.28)  # 树皮棕；外观不进 gid
# ----------------------------------------------------------------

VEHICLE_POSE_RE = re.compile(r'(<model name="(x3|e1_mini)">\s*<pose>)([^<]*)(</pose>)')
DYNAMICS_PROFILE = {"x3": "x3_v1", "e1_mini": "cf_sitl_v1"}  # 同主仓库 scene_gen.VEHICLES（未提交）


def e1_radius(path=E1_PARAMS):
    """E1 外接圆半径（m）。缺文件/缺字段直接报错，不静默用默认值。"""
    with open(path, "r", encoding="utf-8") as f:
        return float(json.load(f)["collision_envelope"]["value"]["radius_from_center"])


E1_RADIUS_M = e1_radius()
CORRIDOR_CLEARANCE_M = E1_RADIUS_M + CORRIDOR_MARGIN_M
GOAL = (FOREST_X[1] + GOAL_PAST_FOREST_M, 0.0)


def _pair_gap(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"]) - a["radius"] - b["radius"]


def min_pair_gap(trunks):
    return min((_pair_gap(a, b) for i, a in enumerate(trunks) for b in trunks[:i]), default=math.inf)


def validate_trunks(trunks, density):
    """缝宽契约 + 区域/半径范围；不合格抛 ValueError。返回净空摘要。"""
    gap = DENSITY_GAP_M[density]
    lo, hi = DENSITY_COUNT[density]
    if not lo <= len(trunks) <= hi:
        raise ValueError(f"trunk count {len(trunks)} not in {DENSITY_COUNT[density]}")
    for i, t in enumerate(trunks):
        r = t["radius"]
        if not TRUNK_RADIUS_M[0] <= r <= TRUNK_RADIUS_M[1]:
            raise ValueError(f"{t['name']} radius {r} outside {TRUNK_RADIUS_M}")
        if not (FOREST_X[0] + r <= t["x"] <= FOREST_X[1] - r and FOREST_Y[0] + r <= t["y"] <= FOREST_Y[1] - r):
            raise ValueError(f"{t['name']} ({t['x']},{t['y']}) r={r} pokes out of forest region")
        for q in trunks[:i]:
            g = _pair_gap(t, q)
            if g < gap:
                raise ValueError(f"{t['name']} vs {q['name']}: gap {g:.3f} < {gap}")
    # 出生航线 y=0 上至少一根柱：否则盲直飞就能过线，整集没有一次避障决策（sparse 档实测 3/20 会这样）。
    blockers = sum(abs(t["y"]) < t["radius"] + E1_RADIUS_M for t in trunks)
    if blockers == 0:
        raise ValueError("no trunk blocks the straight spawn line y=0")
    boundary = min(min(t["y"] - t["radius"] - FOREST_Y[0], FOREST_Y[1] - t["y"] - t["radius"]) for t in trunks)
    return {"min_pair_gap_m": round(min_pair_gap(trunks), 4), "min_boundary_gap_m": round(boundary, 4),
            "trunks_on_spawn_line": blockers}


def corridor(trunks, clearance=CORRIDOR_CLEARANCE_M, grid=GRID_M):
    """出生点→GOAL 的网格 A*（8 邻接、禁斜穿角），障碍按圆 + clearance 精确膨胀，走廊带外视为墙。

    返回 {"path": [(x,y)...], "path_length_m", "min_clearance_m", "bottleneck_gap_m"} 或 None。
    不用 teacher.astar_path：它把障碍当方框、步长 0.1 m，对 0.28 m 缝会把能过的判成堵。
    """
    x0, y0 = SPAWN["x"], FOREST_Y[0]
    nx = int(round((GOAL[0] - x0) / grid)) + 1
    ny = int(round((FOREST_Y[1] - y0) / grid)) + 1
    blocked = bytearray(nx * ny)
    for t in trunks:
        rr = t["radius"] + clearance
        i0, i1 = max(0, int((t["x"] - rr - x0) / grid)), min(nx - 1, int((t["x"] + rr - x0) / grid) + 1)
        j0, j1 = max(0, int((t["y"] - rr - y0) / grid)), min(ny - 1, int((t["y"] + rr - y0) / grid) + 1)
        for i in range(i0, i1 + 1):
            dx2 = (x0 + i * grid - t["x"]) ** 2
            for j in range(j0, j1 + 1):
                if dx2 + (y0 + j * grid - t["y"]) ** 2 < rr * rr:
                    blocked[i * ny + j] = 1
    start = (0, int(round((SPAWN["y"] - y0) / grid)))
    goal = (nx - 1, int(round((GOAL[1] - y0) / grid)))
    if blocked[start[0] * ny + start[1]] or blocked[goal[0] * ny + goal[1]]:
        return None
    moves = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 1.41421356), (1, -1, 1.41421356), (-1, 1, 1.41421356), (-1, -1, 1.41421356))
    open_heap = [(0.0, start)]
    gscore = {start: 0.0}
    parent = {}
    while open_heap:
        _f, cur = heapq.heappop(open_heap)
        if cur == goal:
            break
        gi = gscore[cur]
        i, j = cur
        for di, dj, w in moves:
            ni, nj = i + di, j + dj
            if not (0 <= ni < nx and 0 <= nj < ny) or blocked[ni * ny + nj]:
                continue
            if di and dj and (blocked[(i + di) * ny + j] or blocked[i * ny + j + dj]):
                continue
            ng = gi + w
            if ng < gscore.get((ni, nj), math.inf):
                gscore[(ni, nj)] = ng
                parent[(ni, nj)] = cur
                heapq.heappush(open_heap, (ng + math.hypot(ni - goal[0], nj - goal[1]), (ni, nj)))
    if goal not in gscore:
        return None
    cells = [goal]
    while cells[-1] != start:
        cells.append(parent[cells[-1]])
    path = [(round(x0 + i * grid, 3), round(y0 + j * grid, 3)) for i, j in reversed(cells)]
    min_clear, bottleneck = math.inf, math.inf
    for px, py in path:
        d = sorted(math.hypot(px - t["x"], py - t["y"]) - t["radius"] for t in trunks)
        min_clear = min(min_clear, d[0])
        bottleneck = min(bottleneck, d[0] + d[1])   # 两侧最近柱面之和≈此点穿过的缝宽
    return {"path": path, "path_length_m": round(gscore[goal] * grid, 3),
            "min_clearance_m": round(min_clear, 4), "bottleneck_gap_m": round(bottleneck, 4)}


def sample_trunks(seed, density):
    """random.Random("forest_e1:<density>:<seed>") 顺序吸附采样；整组不满足缝宽或走廊则重采，最多 MAX_ATTEMPTS。

    seed 混入 density：同 seed 不同档位不共享前几根柱，避免 train 稀疏 / test 稠密共用同一骨架布局。
    候选预算 n×200：饱和时 RSA 的接受率会掉到 1% 以下，预算用尽就换一组计数重来。
    """
    if density not in DENSITY_GAP_M:
        raise ValueError(f"unknown density {density!r}, choose from {sorted(DENSITY_GAP_M)}")
    gap = DENSITY_GAP_M[density]
    rng = random.Random(f"{FAMILY}:{density}:{seed}")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        n = rng.randint(*DENSITY_COUNT[density])
        trunks, budget = [], n * 200
        while len(trunks) < n and budget > 0:
            budget -= 1
            r = round(rng.uniform(*TRUNK_RADIUS_M), 3)
            x = round(rng.uniform(FOREST_X[0] + r, FOREST_X[1] - r), 3)
            y = round(rng.uniform(FOREST_Y[0] + r, FOREST_Y[1] - r), 3)
            cand = {"x": x, "y": y, "radius": r}
            if all(_pair_gap(cand, t) >= gap for t in trunks):
                trunks.append(cand)
        if len(trunks) < n:
            continue
        trunks = [{"name": f"trunk_{k}", "x": t["x"], "y": t["y"], "z": TRUNK_HEIGHT_M / 2.0, "radius": t["radius"],
                   "shape": "cylinder", "size": [2 * t["radius"], 2 * t["radius"], TRUNK_HEIGHT_M]}
                  for k, t in enumerate(trunks)]
        try:
            clearance = validate_trunks(trunks, density)
        except ValueError:
            continue
        cor = corridor(trunks)
        if cor is None:
            continue
        return trunks, clearance, cor, attempt
    raise RuntimeError(f"seed={seed} density={density}: {MAX_ATTEMPTS} 次重采样仍无合格走廊")


def _trunk_xml(t):
    mat = f"{TRUNK_RGB[0]:g} {TRUNK_RGB[1]:g} {TRUNK_RGB[2]:g} 1"
    geom = f"<geometry><cylinder><radius>{t['radius']:.3f}</radius><length>{TRUNK_HEIGHT_M:g}</length></cylinder></geometry>"
    return f"""    <model name="{t['name']}">
      <static>true</static>
      <pose>{t['x']:.3f} {t['y']:.3f} {t['z']:.3f} 0 0 0</pose>
      <link name="link">
        <collision name="collision">{geom}</collision>
        <visual name="visual">
          {geom}
          <material><ambient>{mat}</ambient><diffuse>{mat}</diffuse><specular>{mat}</specular></material>
        </visual>
      </link>
    </model>"""


def _apply_spawn(world, spawn):
    """换掉载具模型（x3 | e1_mini）的第一条 pose；返回 (world, vehicle)。工作树的 scene_gen 只认 x3，所以自己匹配。"""
    found = []

    def repl(m):
        found.append(m.group(2))
        return (m.group(1) + f"{spawn['x']:g} {spawn['y']:g} {spawn['z']:g} 0 0 "
                f"{math.radians(spawn['yaw_deg']):.6f}" + m.group(4))
    world, n = VEHICLE_POSE_RE.subn(repl, world, count=1)
    if n != 1:
        raise ValueError("template 载具 pose（x3|e1_mini）匹配到 %d 处，应为 1" % n)
    return world, found[0]


def build_scene(template, seed, scene_id, density=DENSITY_DEFAULT, appearance=sg.APPEARANCE_DEFAULT):
    """返回 (world_text, meta)。template 只读；标记异常直接报错。"""
    for mark in (sg.START_MARK, sg.END_MARK):
        if template.count(mark) != 1:
            raise ValueError(f"template marker {mark!r} count != 1")
    start, end = template.index(sg.START_MARK), template.index(sg.END_MARK)
    if end <= start:
        raise ValueError("template markers out of order")
    trunks, clearance, cor, attempts = sample_trunks(seed, density)
    lines = [f"    <!-- scene_forest: scene_id={scene_id} seed={seed} density={density} "
             f"trunks={len(trunks)}; generated, do not edit -->"]
    lines += [_trunk_xml(t) for t in trunks] + [""]
    head, appearance_used = sg._apply_appearance(template[:start], seed, appearance)
    world, vehicle = _apply_spawn(head + "\n".join(lines) + "\n" + template[end:], SPAWN)
    if vehicle == "x3":
        print(f"warning: 模板载具是 x3（外径 0.711 m），forest_e1 的缝 {DENSITY_GAP_M[density]} m 它过不去", file=sys.stderr)
    gid = sm.geometry_gid(trunks)
    area = (FOREST_X[1] - FOREST_X[0]) * (FOREST_Y[1] - FOREST_Y[0])
    gap = DENSITY_GAP_M[density]
    meta = {
        "schema": "navlab.scene.v1",
        "scene_id": scene_id,
        "seed": seed,
        "generator": "sim/tools/scene_forest.py",
        "family": FAMILY,
        "density": density,
        "geometry_version": GEOMETRY_VERSION_FOREST,
        # split_manifest.split_salt_for 只认 uavrgb.geom.* 前缀，其它必须显式给 salt；gid 走它的 canonical_geometry(size)
        "split_salt": GEOMETRY_VERSION_FOREST,
        "geometry_gid": gid,
        "split": sm.split_for_gid(gid, GEOMETRY_VERSION_FOREST),
        "appearance_version": appearance,
        "requested_appearance": appearance,
        "effective_appearance": appearance,
        "camera_profile": "rgb320x240_fov60_v1",
        "vehicle": vehicle,
        "dynamics_profile": DYNAMICS_PROFILE[vehicle],
        "spawn": dict(SPAWN),
        "template_sha256": sg.sha256(template.encode("utf-8")),
        "world_sha256": sg.sha256(world.encode("utf-8")),
        "pillar_count": len(trunks),
        "pillars": trunks,
        "sampling_attempts": attempts,
        "clearance": {
            "min_gap_required_m": gap,
            "min_pair_gap_m": clearance["min_pair_gap_m"],
            "min_boundary_gap_m": clearance["min_boundary_gap_m"],
            "trunks_on_spawn_line": clearance["trunks_on_spawn_line"],
            "vehicle_diameter_m": round(2 * E1_RADIUS_M, 4),
            "gap_over_vehicle_diameter": round(gap / (2 * E1_RADIUS_M), 3),
            "forest_x": list(FOREST_X), "corridor_y": list(FOREST_Y),
            "trunk_radius_m": list(TRUNK_RADIUS_M), "trunk_count_range": list(DENSITY_COUNT[density]),
            "trunk_height_m": TRUNK_HEIGHT_M, "spawn_clear_before_x_m": FOREST_X[0],
            "ok": True,
        },
        "forest": {
            "area_m2": round(area, 3),
            "trunks_per_m2": round(len(trunks) / area, 3),
            "area_fraction": round(sum(math.pi * t["radius"] ** 2 for t in trunks) / area, 4),
            "mean_radius_m": round(sum(t["radius"] for t in trunks) / len(trunks), 4),
        },
        "corridor": {"clearance_m": round(CORRIDOR_CLEARANCE_M, 4), "grid_m": GRID_M, "goal": list(GOAL),
                     "straight_m": round(math.hypot(GOAL[0] - SPAWN["x"], GOAL[1] - SPAWN["y"]), 3), **cor},
    }
    if appearance_used is not None:
        meta["appearance"] = appearance_used
        meta["appearance_schema_version"] = sg.APPEARANCE_SCHEMA_VERSION
    return world, meta


def scene_rows(template, seeds, densities=tuple(DENSITY_GAP_M)):
    """逐场景统计行（--stats-out / 自检共用）。"""
    rows = []
    for density in densities:
        for seed in seeds:
            _w, m = build_scene(template, seed, seed, density)
            c, f, k = m["clearance"], m["forest"], m["corridor"]
            rows.append({"density": density, "seed": seed, "trunks": m["pillar_count"],
                         "min_pair_gap_m": c["min_pair_gap_m"], "on_spawn_line": c["trunks_on_spawn_line"],
                         "trunks_per_m2": f["trunks_per_m2"],
                         "area_fraction": f["area_fraction"], "path_length_m": k["path_length_m"],
                         "path_over_straight": round(k["path_length_m"] / k["straight_m"], 3),
                         "path_min_clearance_m": k["min_clearance_m"], "bottleneck_gap_m": k["bottleneck_gap_m"],
                         "attempts": m["sampling_attempts"], "gid": m["geometry_gid"], "split": m["split"]})
    return rows


def summarize(rows):
    out = {}
    for density in DENSITY_GAP_M:
        rs = [r for r in rows if r["density"] == density]
        if not rs:
            continue
        def agg(key):
            v = [r[key] for r in rs]
            return {"mean": round(sum(v) / len(v), 3), "min": round(min(v), 3), "max": round(max(v), 3)}
        out[density] = {"n": len(rs), "gap_required_m": DENSITY_GAP_M[density],
                        **{k: agg(k) for k in ("trunks", "min_pair_gap_m", "on_spawn_line", "trunks_per_m2",
                                               "area_fraction", "path_length_m", "path_over_straight",
                                               "path_min_clearance_m", "bottleneck_gap_m", "attempts")}}
    return out


def run_self_check(template_path, n_seeds):
    template = sg._read_template(template_path)
    bad = []
    for density in DENSITY_GAP_M:
        for seed in range(n_seeds):
            try:
                w1, m1 = build_scene(template, seed, seed, density)
                w2, m2 = build_scene(template, seed, seed, density)
                wl, ml = build_scene(template, seed, seed, density, appearance=sg.APPEARANCE_LEGACY)
            except (ValueError, RuntimeError) as e:
                bad.append((density, seed, str(e)))
                continue
            checks = [
                (w1 == w2 and m1 == m2, "non-deterministic"),
                (ml["pillars"] == m1["pillars"] and ml["geometry_gid"] == m1["geometry_gid"], "外观改变了几何/gid"),
                (m1["clearance"]["min_pair_gap_m"] >= DENSITY_GAP_M[density], "缝宽契约"),
                (m1["clearance"]["trunks_on_spawn_line"] >= 1, "出生航线上没有柱"),
                (m1["corridor"]["min_clearance_m"] >= CORRIDOR_CLEARANCE_M - 1e-6, "走廊最小净空"),
                (m1["corridor"]["bottleneck_gap_m"] >= DENSITY_GAP_M[density] - 1e-6, "走廊瓶颈 < 档位缝宽"),
                (sm.split_for_scene(m1) == m1["split"], "split_manifest 复算 split 不一致"),
                (m1["geometry_version"] != sg.GEOMETRY_VERSION, "版本串撞上旧柱阵"),
                ("<scene>" in w1 and 'name="landmark_' not in w1, "v1 外观未生效"),
                (f'<pose>{SPAWN["x"]:g} {SPAWN["y"]:g} {SPAWN["z"]:g} 0 0 0.000000</pose>' in w1, "spawn 未写进 SDF"),
                (w1.count("<model name=\"trunk_") == m1["pillar_count"], "SDF 柱数与 meta 不符"),
            ]
            bad += [(density, seed, why) for ok, why in checks if not ok]
    rows = scene_rows(template, range(n_seeds))
    for density, s in summarize(rows).items():
        print(f"  {density:7s} gap≥{s['gap_required_m']:.2f}: trunks {s['trunks']['mean']:.1f} "
              f"[{s['trunks']['min']:.0f}-{s['trunks']['max']:.0f}]  min_gap {s['min_pair_gap_m']['mean']:.3f} "
              f"(min {s['min_pair_gap_m']['min']:.3f})  on-line {s['on_spawn_line']['mean']:.1f}  "
              f"density {s['trunks_per_m2']['mean']:.2f}/m²  "
              f"path {s['path_length_m']['mean']:.2f} m (×{s['path_over_straight']['mean']:.3f})  "
              f"bottleneck {s['bottleneck_gap_m']['mean']:.3f}  attempts max {s['attempts']['max']:.0f}")
    for density, seed, why in bad[:10]:
        print(f"  FAIL {density} seed={seed}: {why}", file=sys.stderr)
    total = n_seeds * len(DENSITY_GAP_M)
    if bad:
        print(f"scene_forest self-check: FAIL ({len(bad)} / {total})", file=sys.stderr)
        return 1
    print(f"scene_forest self-check: PASS（{total} 场景同 seed 字节级一致，缝宽契约与走廊全过，"
          f"走廊 clearance {CORRIDOR_CLEARANCE_M:.3f} m）")
    return 0


def write_stats(template_path, out_dir, n_seeds):
    """20 seed × 3 档统计（json + md）和每档 seed 0 的俯视 PNG；matplotlib 只在这里用。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    template = sg._read_template(template_path)
    os.makedirs(out_dir, exist_ok=True)
    rows = scene_rows(template, range(n_seeds))
    summary = summarize(rows)
    with open(os.path.join(out_dir, "forest_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"geometry_version": GEOMETRY_VERSION_FOREST, "corridor_clearance_m": CORRIDOR_CLEARANCE_M,
                   "summary": summary, "rows": rows}, f, indent=1, ensure_ascii=False)
    cols = ("density", "seed", "trunks", "min_pair_gap_m", "on_spawn_line", "trunks_per_m2", "area_fraction",
            "path_length_m", "path_over_straight", "bottleneck_gap_m", "attempts", "split")
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    with open(os.path.join(out_dir, "forest_stats.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    pngs = []
    for density in DENSITY_GAP_M:
        _w, m = build_scene(template, 0, 0, density)
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.add_patch(Rectangle((FOREST_X[0], FOREST_Y[0]), FOREST_X[1] - FOREST_X[0], FOREST_Y[1] - FOREST_Y[0],
                               fill=False, ls="--", lw=0.8, color="gray"))
        for t in m["pillars"]:
            ax.add_patch(Circle((t["x"], t["y"]), t["radius"], color="#73503c"))
            ax.add_patch(Circle((t["x"], t["y"]), t["radius"] + CORRIDOR_CLEARANCE_M, fill=False, lw=0.4, color="#c9a"))
        px, py = zip(*m["corridor"]["path"])
        ax.plot(px, py, "-", color="tab:blue", lw=1.2, label=f"A* 走廊 {m['corridor']['path_length_m']} m")
        ax.add_patch(Circle((SPAWN["x"], SPAWN["y"]), E1_RADIUS_M, color="tab:green", label="出生点（E1 外接圆）"))
        ax.plot(*GOAL, "r*", ms=10, label="目标")
        c = m["clearance"]
        ax.set_title(f"forest_e1 {density} seed 0: {m['pillar_count']} trunks, min gap {c['min_pair_gap_m']:.3f} m "
                     f"(≥{c['min_gap_required_m']}), bottleneck {m['corridor']['bottleneck_gap_m']:.3f} m")
        ax.set_xlim(SPAWN["x"] - 0.4, GOAL[0] + 0.4)
        ax.set_ylim(FOREST_Y[0] - 0.3, FOREST_Y[1] + 0.3)
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=7)
        png = os.path.join(out_dir, f"forest_{density}_seed0.png")
        fig.tight_layout()
        fig.savefig(png, dpi=130)
        plt.close(fig)
        pngs.append(png)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print("PNG:", *pngs, sep="\n  ")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, help="随机种子（同 seed 同 density 字节级一致）")
    p.add_argument("--out", help="输出 SDF 路径；同名 .json 写场景清单")
    p.add_argument("--scene-id", type=int, default=None, help="EpisodeMeta.scene_id（默认=seed）")
    p.add_argument("--density", choices=tuple(DENSITY_GAP_M), default=DENSITY_DEFAULT,
                   help="密度档：最窄缝 " + " / ".join(f"{k}={v}" for k, v in DENSITY_GAP_M.items()) + " m")
    p.add_argument("--appearance", choices=(sg.APPEARANCE_LEGACY, sg.APPEARANCE_V1), default=sg.APPEARANCE_DEFAULT,
                   help="v1=删固定地标+随机光照/天空（默认）；legacy=模板原样")
    p.add_argument("--template", default=DEFAULT_TEMPLATE, help=f"模板 SDF（默认 {sg._rel(DEFAULT_TEMPLATE)}）")
    p.add_argument("--self-check", action="store_true", help="确定性 + 缝宽契约 + 走廊自检，不写文件")
    p.add_argument("--check-seeds", type=int, default=30, help="--self-check 每档检查的 seed 数（默认 30）")
    p.add_argument("--stats-out", help="写 20 seed × 3 档几何统计与 3 张俯视 PNG 到此目录（需 matplotlib）")
    p.add_argument("--stats-seeds", type=int, default=20, help="--stats-out 每档 seed 数（默认 20）")
    args = p.parse_args()
    if not os.path.exists(args.template):
        p.error(f"模板不存在：{args.template}（E1 模板在主仓库工作树里尚未提交，复制到同路径；或 --template）")
    if args.self_check:
        if args.check_seeds < 1:
            p.error("--check-seeds must be >= 1")
        return run_self_check(args.template, args.check_seeds)
    if args.stats_out:
        return write_stats(args.template, args.stats_out, args.stats_seeds)
    if args.seed is None or not args.out:
        p.error("需要 --seed 和 --out（或 --self-check / --stats-out）")
    scene_id = args.seed if args.scene_id is None else args.scene_id
    template = sg._read_template(args.template)
    world, meta = build_scene(template, args.seed, scene_id, args.density, args.appearance)
    meta["template"] = sg._rel(os.path.abspath(args.template))
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    sg._write_atomic(out, world.encode("utf-8"))
    meta_path = os.path.splitext(out)[0] + ".json"
    sg._write_atomic(meta_path, (json.dumps(meta, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    c, k = meta["clearance"], meta["corridor"]
    print(f"scene_id={scene_id} seed={args.seed} density={args.density} trunks={meta['pillar_count']} "
          f"vehicle={meta['vehicle']} split={meta['split']} sha256={meta['world_sha256']}")
    print(f"  world -> {out}\n  meta  -> {meta_path}")
    print(f"  clearance: min_pair={c['min_pair_gap_m']:.3f} m (required {c['min_gap_required_m']}), "
          f"boundary={c['min_boundary_gap_m']:.3f} m, attempts={meta['sampling_attempts']}")
    print(f"  corridor: path {k['path_length_m']} m (straight {k['straight_m']}), min clearance "
          f"{k['min_clearance_m']} m, bottleneck {k['bottleneck_gap_m']} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
