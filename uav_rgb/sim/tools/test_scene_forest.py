#!/usr/bin/env python3
"""forest_e1 回归：确定性、缝宽契约、走廊保证、E1 模板 pose、与 split_manifest / 方框教师的兼容。

模板：有 sim/worlds/uav_rgb_e1.sdf（gz agent 的文件，可能尚未提交）就用它，否则退到 X3 模板
（两个模板的 scene_gen 标记相同，几何/gid 与模板无关）。
"""
import os
import re
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_forest as sf  # noqa: E402
import scene_gen as sg  # noqa: E402
import split_manifest as sm  # noqa: E402
import teacher  # noqa: E402

SEEDS = range(4)


def main():
    template_path = sf.DEFAULT_TEMPLATE if os.path.exists(sf.DEFAULT_TEMPLATE) else sg.DEFAULT_TEMPLATE
    template = sg._read_template(template_path)
    want_vehicle = "e1_mini" if template_path == sf.DEFAULT_TEMPLATE else "x3"
    try:
        ET.fromstring(template)
        template_xml_ok = True
    except ET.ParseError as e:
        template_xml_ok = False
        print(f"note: 模板 {os.path.basename(template_path)} 本身不是严格 XML（{e}），整篇解析跳过", file=sys.stderr)
    gids = set()
    for density, gap in sf.DENSITY_GAP_M.items():
        for seed in SEEDS:
            w1, m1 = sf.build_scene(template, seed, seed, density)
            w2, m2 = sf.build_scene(template, seed, seed, density)
            assert w1 == w2 and m1 == m2, (density, seed, "non-deterministic")
            trunks = m1["pillars"]
            # 缝宽契约：任意两柱面间距 ≥ 档位值；半径/区域在范围内；出生航线上至少一根。
            assert sf.min_pair_gap(trunks) >= gap - 1e-9, (density, seed)
            assert all(sf.TRUNK_RADIUS_M[0] <= t["radius"] <= sf.TRUNK_RADIUS_M[1] for t in trunks)
            assert all(t["x"] - t["radius"] >= sf.FOREST_X[0] and t["x"] + t["radius"] <= sf.FOREST_X[1]
                       for t in trunks), (density, seed)
            assert m1["clearance"]["trunks_on_spawn_line"] >= 1, (density, seed)
            # 走廊：路径从出生点到目标、每点离柱面 ≥ clearance、瓶颈不窄于档位缝宽。
            cor = m1["corridor"]
            assert cor["path"][0] == (sf.SPAWN["x"], sf.SPAWN["y"]) and cor["path"][-1] == tuple(sf.GOAL)
            assert cor["min_clearance_m"] >= sf.CORRIDOR_CLEARANCE_M - 1e-6, (density, seed, cor["min_clearance_m"])
            assert cor["bottleneck_gap_m"] >= gap - 1e-6, (density, seed)
            assert cor["path_length_m"] >= cor["straight_m"] - 1e-6
            # 版本与 split：独立于旧柱阵；split_manifest 用 meta 里的 salt 复算得到同一结果。
            assert m1["geometry_version"] == sf.GEOMETRY_VERSION_FOREST != sg.GEOMETRY_VERSION
            assert m1["split_salt"] == sf.GEOMETRY_VERSION_FOREST
            assert m1["geometry_gid"] == sm.geometry_gid(trunks)
            assert sm.split_for_scene(m1) == m1["split"], (density, seed)
            gids.add(m1["geometry_gid"])
            # SDF：载具 pose 换成 spawn，柱子以 cylinder 落盘，数量一致，v1 外观生效。
            match = re.search(r'<model name="(x3|e1_mini)">\s*<pose>([^<]*)</pose>', w1)
            assert match and match.group(1) == want_vehicle == m1["vehicle"], (match and match.groups())
            parts = match.group(2).split()
            assert abs(float(parts[0]) - sf.SPAWN["x"]) < 1e-6 and abs(float(parts[1]) - sf.SPAWN["y"]) < 1e-6
            assert w1.count('<model name="trunk_') == len(trunks) == m1["pillar_count"]
            assert all(f"<radius>{t['radius']:.3f}</radius>" in w1 for t in trunks)
            assert "<scene>" in w1 and 'name="landmark_' not in w1
            # 不起 Gazebo 的底线：输出必须是合法 XML，且每根柱的 collision 与 visual 都是 cylinder。
            # 模板自身过不了 expat（E1 模板注释里有 "--"，tinyxml2 不管）时只解析生成的柱段。
            doc = w1 if template_xml_ok else "<sdf>\n" + "\n".join(sf._trunk_xml(t) for t in trunks) + "\n</sdf>"
            root = ET.fromstring(doc)
            trunk_models = [m for m in root.iter("model") if m.get("name", "").startswith("trunk_")]
            assert len(trunk_models) == len(trunks)
            assert all(m.find("link/collision/geometry/cylinder") is not None
                       and m.find("link/visual/geometry/cylinder") is not None for m in trunk_models)
            # 方框教师兼容：size 是外接方框，teacher._point_blocked 在柱心处必判堵、远处必判空。
            t0 = trunks[0]
            assert teacher._point_blocked(t0["x"], t0["y"], trunks, 0.01)
            assert not teacher._point_blocked(sf.SPAWN["x"], sf.SPAWN["y"], trunks, sf.CORRIDOR_CLEARANCE_M)
    # 不同档位、不同 seed 的几何两两不同（seed 混入了 density）。
    assert len(gids) == len(sf.DENSITY_GAP_M) * len(SEEDS)
    # 外观只改头部：legacy 与 v1 的几何、gid、spawn 一致。
    _wl, ml = sf.build_scene(template, 0, 0, "medium", appearance=sg.APPEARANCE_LEGACY)
    _wv, mv = sf.build_scene(template, 0, 0, "medium")
    assert ml["pillars"] == mv["pillars"] and ml["geometry_gid"] == mv["geometry_gid"] and ml["spawn"] == mv["spawn"]
    # 走廊检查真的会拒：一排柱横跨走廊、缝都小于 2×clearance → None；拿掉一根开口 → 有路。
    r = 0.10
    pitch = 2 * r + 2 * sf.CORRIDOR_CLEARANCE_M - 0.02
    wall = [{"x": 3.0, "y": sf.FOREST_Y[0] + r + k * pitch, "radius": r} for k in range(20)
            if sf.FOREST_Y[0] + r + k * pitch <= sf.FOREST_Y[1] - r + pitch]
    assert sf.corridor(wall) is None, "横墙应判无走廊"
    opened = wall[:len(wall) // 2 - 1] + wall[len(wall) // 2 + 1:]
    assert sf.corridor(opened) is not None, "开口后应有走廊"
    # 坏输入：未知档位报错。
    try:
        sf.sample_trunks(0, "ultra")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown density 应抛 ValueError")
    # 缝宽/外径比例与任务口径一致（2.5×/2×/1.5× 按约 0.18 m 外径；对 0.204 m 是 2.2/1.7/1.4）。
    ratios = {k: v / (2 * sf.E1_RADIUS_M) for k, v in sf.DENSITY_GAP_M.items()}
    assert ratios["sparse"] > ratios["medium"] > ratios["dense"] >= 1.3, ratios
    assert abs(sf.E1_RADIUS_M - 0.102) < 1e-9, sf.E1_RADIUS_M
    print(f"test_scene_forest: PASS（{len(SEEDS)} seeds × {len(sf.DENSITY_GAP_M)} 档；模板 {os.path.basename(template_path)}；"
          f"走廊 clearance {sf.CORRIDOR_CLEARANCE_M:.3f} m；横墙负例判无路）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
