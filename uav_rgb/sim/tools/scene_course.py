#!/usr/bin/env python3
"""course_base：把 K 个 base 柱阵块首尾相接成一条长课程（按 seed 与 K 确定性生成，纯 stdlib）。

用法：
  python3 sim/tools/scene_course.py --seed 1 --blocks 3 --out <dir>/course_1.sdf
  python3 sim/tools/scene_course.py --self-check          # K∈{2,3,4,6} × 30 seed：确定性 + 块几何 + 缝宽 + split + 成功线

为什么：base 柱阵只有约 6.5 m，绕一根柱就过线，量不出「成功率随航程怎么衰减」。成功线（eval / collect 的
最远柱 x + 1.0）、教师目标（teacher._global_path 的最远柱 x + 2.0）、SPL 最短路都只由 meta 的 pillars 列表推出，
所以长课程只要给一份跨 K 块的柱子列表，三者自动跟着变长，评测与教师代码不用改。

构造：第 0 块就是 scene_gen.sample_pillars(seed) 的原样结果（课程 seed s 的第 0 块与 --task base --seed s 几何完全
相同，可与 K=1 的结果逐集对照）；第 k≥1 块 = 子 seed seed*1000+k*100+j（j=0,1,…）里第一个同时满足
① 它的 base 几何按 split_manifest 的规则（geometry_gid + split_for_gid，salt = scene_gen.GEOMETRY_VERSION）所属 split
与第 0 块相同（test 课程的每一块都是 test 几何，不泄漏训练几何）；② 平移后与已放置的柱子两两缝宽 ≥ MIN_GAP
的那个，整体沿 x 平移 k·BLOCK_PITCH_M。子 seed 序列与 K 无关：K=3 课程的前两块就是 K=2 课程。

与 scene_gen 的关系：复用它的模板标记、外观 v1 随机化、柱 XML、出生点与 sha256；meta 沿用 base 的字段，另加
geometry_version / split_policy / course。split 只由第 0 块决定（split_manifest.split_for_scene 的 course_block0 分支）。
shortcut: 训练打包（tools/preprocess.py、audit_dataset.py）没接课程，长课程数据只用于评测；preprocess 按
KNOWN_GEOMETRY_VERSIONS 拒收 uavrgb.course_base.*，要拿课程数据训练时再接。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_gen as sg  # noqa: E402
import split_manifest as sm  # noqa: E402

# ---------------------------------------------------------------- 配置区（改这里；改块规则 / 平移 / 选块序列必 bump 版本）
GEOMETRY_VERSION = "uavrgb.course_base.v1"    # 课程几何版本；split_manifest.COURSE_BASE_SALT 登记它建在哪个 base 版本上
SPLIT_POLICY = "course_block0"                # split 只由第 0 块的 base 几何决定
BASE_GEOMETRY_VERSION = sg.GEOMETRY_VERSION   # 各块几何与 split 的 salt；落盘进 meta.course，复算不回落当前值
BLOCKS_RANGE = (2, 6)                         # 块数 K（1 块就是 base）；电池对可飞长度的限制见 experiments/2026-11-10-long-course/README.md
BLOCK_PITCH_M = 5.0                           # 块间平移：块内柱子 x∈[1.5, 5.0]，下一块门柱 6.5，块间柱心距 1.5 m
MAX_BLOCK_TRIES = 100                         # 每块最多试的子 seed 数；test 占 10%，100 次都不中的概率 2.7e-5
SEED_STRIDE, BLOCK_STRIDE = 1000, 100         # 子 seed = seed*1000 + k*100 + j；j < 100 时各块、各课程的子 seed 互不重叠
GOAL_MARGIN_M = 1.0                           # 同 collect.GOAL_MARGIN_M（测试断言相等）：成功线 = 最远柱 x + 它
DEFAULT_TEMPLATE = sg.VEHICLES["e1_mini"]["template"]   # 课程为 E1 SITL 评测而做，电池预算按它算
# ----------------------------------------------------------------


def block_seed(seed, k, j):
    """第 k 块（k≥1）的第 j 个候选子 seed。"""
    return seed * SEED_STRIDE + k * BLOCK_STRIDE + j


def shift_block(pillars, k, pitch=BLOCK_PITCH_M):
    """第 k 块：整体沿 x 平移 k·pitch，柱名加 b<k>_ 前缀保证唯一；y / z / 尺寸原样。"""
    return [dict(p, name=f"b{k}_{p['name']}", x=round(p["x"] + k * pitch, 3), size=list(p["size"])) for p in pillars]


def cross_gap_ok(placed, cand):
    """候选块与已放置的柱子两两可通行缝宽 ≥ MIN_GAP（scene_gen._pillar_gap，x / y 取较大者）。

    pitch=5.0 时恒成立（跨块 x 向柱心距 ≥ 1.5 m → 缝 ≥ 1.1 m > 0.9 m）；留着是 BLOCK_PITCH_M 以后改小时的护栏。
    """
    return all(sg._pillar_gap(a, b) >= sg.MIN_GAP for a in cand for b in placed)


def sample_course(seed, blocks):
    """返回 (pillars, info)：全部块合并后的柱子列表，与 {attempts0, block_seeds, block_gids, block_goal_x}。"""
    if not BLOCKS_RANGE[0] <= blocks <= BLOCKS_RANGE[1]:
        raise ValueError(f"blocks={blocks} 不在 {BLOCKS_RANGE}（1 块就是 --task base）")
    base0, _clearance, attempts0 = sg.sample_pillars(seed)
    gid0 = sm.geometry_gid(base0)
    split0 = sm.split_for_gid(gid0, BASE_GEOMETRY_VERSION)
    placed = shift_block(base0, 0)
    seeds, gids = [seed], [gid0]
    goals = [round(max(p["x"] for p in placed) + GOAL_MARGIN_M, 3)]
    for k in range(1, blocks):
        for j in range(MAX_BLOCK_TRIES):
            sub = block_seed(seed, k, j)
            base = sg.sample_pillars(sub)[0]
            gid = sm.geometry_gid(base)
            if sm.split_for_gid(gid, BASE_GEOMETRY_VERSION) != split0:
                continue
            cand = shift_block(base, k)
            if cross_gap_ok(placed, cand):
                break
        else:
            raise RuntimeError(f"seed={seed} 第 {k} 块：{MAX_BLOCK_TRIES} 个子 seed 里没有 split={split0} 且缝宽达标的")
        placed += cand
        seeds.append(sub)
        gids.append(gid)
        goals.append(round(max(p["x"] for p in cand) + GOAL_MARGIN_M, 3))
    return placed, {"attempts0": attempts0, "block_seeds": seeds, "block_gids": gids, "block_goal_x": goals}


def build_scene(template, seed, scene_id, blocks, appearance=sg.APPEARANCE_DEFAULT):
    """返回 (world_text, meta)。template 只读；标记异常直接报错。出生点同 base default。"""
    for mark in (sg.START_MARK, sg.END_MARK):
        if template.count(mark) != 1:
            raise ValueError(f"template marker {mark!r} count != 1")
    start, end = template.index(sg.START_MARK), template.index(sg.END_MARK)
    if end <= start:
        raise ValueError("template markers out of order")
    pillars, info = sample_course(seed, blocks)
    spawn = sg.make_spawn(pillars, seed, "default")
    lines = [f"    <!-- scene_course: scene_id={scene_id} seed={seed} blocks={blocks} "
             f"pillars={len(pillars)}; generated, do not edit -->"]
    lines += [sg._pillar_xml(p) for p in pillars]
    lines.append("")
    head, appearance_used = sg._apply_appearance(template[:start], seed, appearance)
    world = sg._apply_spawn(head + "\n".join(lines) + "\n" + template[end:], spawn)
    vehicle = sg.vehicle_of_template(template)
    min_pair = min(sg._pillar_gap(a, b) for i, a in enumerate(pillars) for b in pillars[:i])
    min_boundary = min(min(p["y"] - sg.Y_RANGE[0] - sg.PILLAR_SIZE / 2, sg.Y_RANGE[1] - p["y"] - sg.PILLAR_SIZE / 2)
                       for p in pillars)
    meta = {
        "schema": "navlab.scene.v1",
        "scene_id": scene_id,
        "seed": seed,
        "generator": "sim/tools/scene_course.py",
        "geometry_version": GEOMETRY_VERSION,
        "split_policy": SPLIT_POLICY,
        "appearance_version": appearance,
        "requested_appearance": appearance,
        "effective_appearance": appearance,
        "camera_profile": "rgb320x240_fov60_v1",
        "vehicle": vehicle,
        "dynamics_profile": sg.VEHICLES[vehicle]["dynamics_profile"],
        "spawn": spawn,
        "template_sha256": sg.sha256(template.encode("utf-8")),
        "world_sha256": sg.sha256(world.encode("utf-8")),
        "pillar_count": len(pillars),
        "pillars": pillars,
        "sampling_attempts": info["attempts0"],
        # 字段同 base；范围类常量按块内口径，课程整体的缝宽 / 边界净空是全部柱子上的实测最小值
        "clearance": {
            "min_gap_required_m": sg.MIN_GAP,
            "min_pair_gap_m": round(min_pair, 4),
            "min_boundary_gap_m": round(min_boundary, 4),
            "corridor_y": list(sg.Y_RANGE),
            "pillar_x_range": list(sg.PILLAR_X_RANGE),
            "pillar_y_range": list(sg.PILLAR_Y_RANGE),
            "lane_sep_m": sg.LANE_SEP,
            "lane_gap_m": round(sg.LANE_SEP - sg.PILLAR_SIZE, 4),
            "pillar_count_range": list(sg.PILLAR_COUNT_RANGE),
            "gate_y_abs_max_m": sg.GATE_Y_ABS_MAX,
            "spawn_clear_before_x_m": sg.SPAWN_CLEAR_X,
            "pillar_height_m": sg.PILLAR_HEIGHT,
            "cruise_height_m": sg.CRUISE_HEIGHT,
            "ok": True,
        },
        "course": {
            "blocks": blocks,
            "block_pitch_m": BLOCK_PITCH_M,
            "block_seeds": info["block_seeds"],
            "block_gids": info["block_gids"],      # 各块 base 几何（平移前）的 geometry_gid；split 由 [0] 复算
            "base_geometry_version": BASE_GEOMETRY_VERSION,
            "block_goal_x": info["block_goal_x"],  # 第 k 块最远柱 x + GOAL_MARGIN_M；最后一个就是整条课程的成功线
        },
    }
    if appearance_used is not None:
        meta["appearance"] = appearance_used
        meta["appearance_schema_version"] = sg.APPEARANCE_SCHEMA_VERSION
    return world, meta


def run_self_check(template_path, n_seeds):
    template = sg._read_template(template_path)
    bad = []
    tries = []
    if sm.COURSE_BASE_SALT.get(GEOMETRY_VERSION) != BASE_GEOMETRY_VERSION:
        bad.append((-1, 0, f"split_manifest.COURSE_BASE_SALT 没登记 {GEOMETRY_VERSION} → {BASE_GEOMETRY_VERSION}"))
    for blocks in (2, 3, 4, 6):
        for seed in range(n_seeds):
            try:
                w1, m1 = build_scene(template, seed, seed, blocks)
                w2, m2 = build_scene(template, seed, seed, blocks)
                wl, ml = build_scene(template, seed, seed, blocks, appearance=sg.APPEARANCE_LEGACY)
            except (ValueError, RuntimeError) as e:
                bad.append((blocks, seed, str(e)))
                continue
            c, ps = m1["course"], m1["pillars"]
            base0 = sg.sample_pillars(seed)[0]
            splits = {sm.split_for_gid(sm.geometry_gid(sg.sample_pillars(s)[0]), BASE_GEOMETRY_VERSION) for s in c["block_seeds"]}
            tries += [s - (seed * SEED_STRIDE + k * BLOCK_STRIDE) + 1 for k, s in enumerate(c["block_seeds"]) if k]
            checks = [
                (w1 == w2 and m1 == m2, "non-deterministic"),
                (ml["pillars"] == m1["pillars"], "外观改变了几何"),
                ([(p["x"], p["y"], p["z"], p["size"]) for p in ps[:len(base0)]]
                 == [(p["x"], p["y"], p["z"], p["size"]) for p in base0], "第 0 块与 base 同 seed 几何不一致"),
                (len(splits) == 1 and sm.split_for_scene(m1) in splits, "各块 base split 不一致或 split_for_scene 复算不符"),
                (m1["clearance"]["min_pair_gap_m"] >= sg.MIN_GAP, "缝宽契约（含跨块）"),
                (len({p["name"] for p in ps}) == len(ps), "柱名不唯一"),
                (abs(c["block_goal_x"][-1] - (sg.PILLAR_X_RANGE[1] + GOAL_MARGIN_M + BLOCK_PITCH_M * (blocks - 1))) < 1e-9,
                 "成功线 != 6.0+5.0·(K−1)"),
                (all(g == sm.geometry_gid(sg.sample_pillars(s)[0]) for g, s in zip(c["block_gids"], c["block_seeds"])),
                 "block_gids 与子 seed 不符"),
                (w1.count('<model name="b') == m1["pillar_count"], "SDF 柱数与 meta 不符"),
                ("<scene>" in w1 and 'name="landmark_' not in w1, "v1 外观未生效"),
                (f'<pose>{sg.SPAWN_X:g} 0 {sg.SPAWN_Z:g} 0 0 0.000000</pose>' in w1, "spawn 未写进 SDF"),
            ]
            bad += [(blocks, seed, why) for ok, why in checks if not ok]
    for blocks, seed, why in bad[:10]:
        print(f"  FAIL K={blocks} seed={seed}: {why}", file=sys.stderr)
    total = 4 * n_seeds
    if bad:
        print(f"scene_course self-check: FAIL ({len(bad)} / {total})", file=sys.stderr)
        return 1
    print(f"scene_course self-check: PASS（K∈{{2,3,4,6}} × {n_seeds} seed 同 seed 字节级一致；第 0 块同 base、各块同 split、"
          f"缝宽契约与成功线全过；第 k≥1 块平均试 {sum(tries) / len(tries):.1f} 个子 seed，最多 {max(tries)}）")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, help="课程 seed（第 0 块 = base 同 seed；同 seed 同 K 字节级一致）")
    p.add_argument("--blocks", type=int, default=None, help=f"块数 K，{BLOCKS_RANGE[0]}–{BLOCKS_RANGE[1]}")
    p.add_argument("--out", help="输出 SDF 路径；同名 .json 写场景清单")
    p.add_argument("--scene-id", type=int, default=None, help="EpisodeMeta.scene_id（默认=seed）")
    p.add_argument("--appearance", choices=(sg.APPEARANCE_LEGACY, sg.APPEARANCE_V1), default=sg.APPEARANCE_DEFAULT,
                   help="v1=删固定地标+随机光照/天空（默认）；legacy=模板原样")
    p.add_argument("--template", default=DEFAULT_TEMPLATE, help=f"模板 SDF（默认 {sg._rel(DEFAULT_TEMPLATE)}）")
    p.add_argument("--self-check", action="store_true", help="确定性 + 块几何 + 缝宽 + split 自检，不写文件")
    p.add_argument("--check-seeds", type=int, default=30, help="--self-check 每个 K 检查的 seed 数（默认 30）")
    args = p.parse_args()
    if not os.path.exists(args.template):
        p.error(f"模板不存在：{args.template}")
    if args.self_check:
        if args.check_seeds < 1:
            p.error("--check-seeds must be >= 1")
        return run_self_check(args.template, args.check_seeds)
    if args.seed is None or not args.out or args.blocks is None:
        p.error("需要 --seed、--blocks 和 --out（或 --self-check）")
    if not BLOCKS_RANGE[0] <= args.blocks <= BLOCKS_RANGE[1]:
        p.error(f"--blocks 必须在 {BLOCKS_RANGE[0]}–{BLOCKS_RANGE[1]}")
    scene_id = args.seed if args.scene_id is None else args.scene_id
    world, meta = build_scene(sg._read_template(args.template), args.seed, scene_id, args.blocks, args.appearance)
    meta["template"] = sg._rel(os.path.abspath(args.template))
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    sg._write_atomic(out, world.encode("utf-8"))
    meta_path = os.path.splitext(out)[0] + ".json"
    sg._write_atomic(meta_path, (json.dumps(meta, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    c = meta["course"]
    print(f"scene_id={scene_id} seed={args.seed} blocks={args.blocks} pillars={meta['pillar_count']} "
          f"vehicle={meta['vehicle']} sha256={meta['world_sha256']}")
    print(f"  world -> {out}\n  meta  -> {meta_path}")
    print(f"  block_seeds={c['block_seeds']} block_goal_x={c['block_goal_x']} "
          f"min_pair_gap={meta['clearance']['min_pair_gap_m']:.3f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
