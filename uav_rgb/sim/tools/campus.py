"""定高校园 pilot：三种地图、可见目标区；所有实体碰撞体进入教师几何。"""
import hashlib
import json
import random
from pathlib import Path

from instruction import COLORS
from landmarks import COLOR_RGB

# v1 的 480 集已冻结：本文件改布局后同 seed 会生成不同地图，旧数据不得与新数据混包。
# 自 v2 起 map_id/geometry_version/task_version 都带 campus.v2 前缀。
VERSION = "campus.v2"
FAMILIES = ("building_lane", "tree_road", "plaza")
RADIUS = 0.5
# 纹理放在 sim/assets 资源根下，SDF 只写 model:// URI；换机器/换目录不改字节。
ASSETS = Path(__file__).resolve().parents[1] / "assets" / "campus_textures"
TEXTURE_URI = "model://campus_textures/{}.png"


def material(name):
    path = ASSETS / f"{name}.png"
    if not path.is_file():
        raise ValueError(f"缺校园纹理 {path}；运行 sim/tools/make_campus_textures.py")
    return ('<material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse>'
            '<pbr><metal><metalness>0</metalness><roughness>1</roughness>'
            f'<albedo_map>{TEXTURE_URI.format(name)}</albedo_map></metal></pbr></material>')


def map_id(seed):
    # 同拓扑的尺寸微扰、四种颜色、左右任务全部同组，毫米平移不能穿过 split。
    return f"{VERSION}/{FAMILIES[(seed // 8) % len(FAMILIES)]}"


def split(seed):
    return split_map(map_id(seed))


def split_map(identity):
    assignments = dict(zip((f"{VERSION}/{f}" for f in FAMILIES), ("train", "val", "test")))
    if identity not in assignments:
        raise ValueError("未知校园拓扑组")
    return assignments[identity]


def instruction(color, relation):
    if color not in COLORS or relation not in ("left", "right"):
        raise ValueError("invalid campus instruction")
    return f"go to the {color} pillar on your {relation}"


def layout(seed):
    identity = seed // 8
    rng = random.Random(identity ^ 0xCA4F05)
    family = FAMILIES[identity % len(FAMILIES)]
    obstacles = []

    def box(name, x, y, sx, sy, sz, color=None):
        p = dict(name=name, x=round(x, 3), y=round(y, 3), z=sz / 2,
                 size=[sx, sy, sz])
        if color:
            p["color"] = color
        obstacles.append(p)

    # 场景布局随地图变，材质/语言变体不能改变几何。
    # y 下限收到 1.55：目标柱与 y=0 中央障碍既要 ≥0.9 m 净空（原柱阵
    # MIN_GAP 契约），又要从出生点看得见——y 太小时近处中央障碍会挡住
    # 目标柱中心。x 下限取 3.6，保证目标中心仍在 28° 视野余量内。
    x = round(rng.uniform(3.6, 6.5), 3)
    y = round(rng.uniform(1.55, 2.0), 3)
    color = COLORS[seed % 4]
    relation = "left" if (seed // 4) % 2 == 0 else "right"
    for side in (-1, 1):
        box(f"target_{side + 1}", x, side * y, .4, .4, 3., color)
    if family == "building_lane":
        for side in (-1, 1):
            box(f"building_{side + 1}", 3.0, side * 4.2, 6., 2., 5.)
        box("island", rng.uniform(2.0, 3.0), 0, .8, .8, 3.)
    elif family == "tree_road":
        # 树排外移到 ±3.35：目标 |y| 上限 2.0 时，±3.1 的树与目标柱只剩
        # 0.7 m 缝，低于原柱阵 0.9 m 净空契约。
        for i in range(3):
            for side in (-1, 1):
                box(f"tree_{i}_{side + 1}", 1.5 + i * 2, side * 3.35,
                    .4, .4, 4.)
        box("tree_center", rng.uniform(2.0, 3.0), 0, .4, .4, 4.)
    else:
        for i in range(3):
            box(f"plaza_edge_{i}", 1.2 + i * 1.9, -3.6, 1., .6, 3.)
        # 原 8 m 处的 1×9×5 大墙占满 60° 视场且逆光时把首帧压到 ~67 灰度。
        # 后移到 14 m、高度压到 6 m 后，墙面在屏幕上的占比从 ~47% 降到 ~20%。
        # 2026-11-04 真实渲染复核（seed 16/20/22/41）：均值 67.7 -> 79.4，
        # 有改善但仍低于 train 的 102±8；静态估算给的 93-107 偏乐观，以实测为准。
        box("building_back", 14., 0., 1., 12., 6.)
        box("island", rng.uniform(2.0, 3.0), 0, .6, .6, 3.)
    waypoint = {"x": round(x - 1.1, 3), "y": y if relation == "left" else -y}
    return family, obstacles, color, relation, waypoint


def _box_xml(p):
    sx, sy, sz = p["size"]
    rgb = COLOR_RGB.get(p.get("color"),
                        (.28, .20, .12) if p["name"].startswith("tree") else (.62, .59, .52))
    mat = " ".join(str(v) for v in (*rgb, 1))
    geom = f"<geometry><box><size>{sx} {sy} {sz}</size></box></geometry>"
    surface = (f'<material><ambient>{mat}</ambient><diffuse>{mat}</diffuse>'
               '<specular>0.05 0.05 0.05 1</specular></material>' if p.get("color") else
               material("bark" if p["name"].startswith("tree") else "brick"))
    return (f'<model name="{p["name"]}"><static>true</static>'
            f'<pose>{p["x"]} {p["y"]} {p["z"]} 0 0 0</pose><link name="link">'
            f'<collision name="collision">{geom}</collision><visual name="visual">{geom}'
            f'{surface}</visual></link></model>')


def _ground_marks(x, y, color):
    # 地面视觉与地面碰撞面重合；目标标记不新增障碍。
    rgb = " ".join(str(v) for v in (*COLOR_RGB[color], 1))
    return (f'<model name="goal_mark_{y}"><static>true</static><pose>{x} {y} 0.012 0 0 0</pose>'
            '<link name="link"><visual name="mark"><geometry><cylinder>'
            f'<radius>{RADIUS}</radius><length>0.01</length></cylinder></geometry>'
            f'<material><ambient>{rgb}</ambient><diffuse>{rgb}</diffuse></material>'
            '</visual></link></model>')


def build(template, seed, scene_id, appearance="v1"):
    import scene_gen as sg
    from teacher import astar_path
    if appearance not in (sg.APPEARANCE_LEGACY, sg.APPEARANCE_V1):
        raise ValueError(f"unknown appearance {appearance!r}")
    family, boxes, color, relation, wp = layout(seed)
    spawn = dict(x=-.5, y=0., z=.15, yaw_deg=0., mode="default")
    path = astar_path(boxes, (spawn["x"], spawn["y"]), (wp["x"], wp["y"]))
    if not path:
        raise ValueError("campus 地图无可行教师路径")
    # 校园任务固定用 v1 外观（legacy 的固定地标与校园语义冲突）；requested 如实记录。
    # 已冻结的 campus.v1 数据只见过单侧太阳，这里保持 full_sun=False，避免静默换光照。
    head, lighting = sg._apply_appearance(template[:template.index(sg.START_MARK)], seed, "v1",
                                          full_sun=False)
    models = [_box_xml(p) for p in boxes]
    models += [_ground_marks(wp["x"], side * abs(wp["y"]), color) for side in (-1, 1)]
    world = head + "\n".join(models) + template[template.index(sg.END_MARK):]
    world = sg._apply_spawn(world, spawn)
    # 只替换校园地面材质；legacy/v1 模板字节保持原样。
    start = world.index('<model name="ground_plane">')
    a = world.index('<material>', start)
    b = world.index('</material>', a) + len('</material>')
    world = world[:a] + material("pavement") + world[b:]
    meta = dict(schema="navlab.scene.v1", scene_id=scene_id, seed=seed,
                generator="sim/tools/campus.py", geometry_version=VERSION,
                task_version=VERSION, map_id=map_id(seed), family=family,
                split_policy="campus_map_v1", appearance_version="campus.v1",
                requested_appearance=appearance, effective_appearance=sg.APPEARANCE_V1,
                appearance_schema_version=sg.APPEARANCE_SCHEMA_VERSION_FROZEN,
                appearance=lighting, spawn=spawn, pillars=boxes, pillar_count=len(boxes),
                template_sha256=sg.sha256(template.encode()), world_sha256=sg.sha256(world.encode()),
                instruction=instruction(color, relation), target_color=color, relation=relation,
                waypoint=wp, stop_radius=RADIUS, goal_radius=RADIUS,
                success_criterion="visible_goal_reach_v1",
                clearance=dict(ok=True, cruise_height_m=sg.CRUISE_HEIGHT),
                camera_profile="rgb320x240_fov60_v1", dynamics_profile="x3_v1")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ASSETS.glob("*.png"))}
    meta["resource_sha256"] = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return world, meta
