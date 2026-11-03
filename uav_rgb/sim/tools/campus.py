"""定高校园 pilot：三种地图、可见目标区；所有实体碰撞体进入教师几何。"""
import hashlib
import json
import random
from pathlib import Path
from xml.sax.saxutils import escape

from instruction import COLORS
from landmarks import COLOR_RGB

VERSION = "campus.v1"
FAMILIES = ("building_lane", "tree_road", "plaza")
RADIUS = 0.5
ASSETS = Path(__file__).resolve().parents[1] / "worlds" / "campus_textures"


def material(name):
    path = ASSETS / f"{name}.png"
    if not path.is_file():
        raise ValueError(f"缺校园纹理 {path}；运行 sim/tools/make_campus_textures.py")
    return ('<material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse>'
            '<pbr><metal><metalness>0</metalness><roughness>1</roughness>'
            f'<albedo_map>{escape(str(path))}</albedo_map></metal></pbr></material>')


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
    x = round(rng.uniform(4.8, 5.4), 3)
    y = round(rng.uniform(1.4, 1.65), 3)
    color = COLORS[seed % 4]
    relation = "left" if (seed // 4) % 2 == 0 else "right"
    for side in (-1, 1):
        box(f"target_{side + 1}", x, side * y, .4, .4, 3., color)
    if family == "building_lane":
        for side in (-1, 1):
            box(f"building_{side + 1}", 3.0, side * 4.2, 6., 2., 5.)
        box("island", rng.uniform(2.0, 2.5), 0, .8, .8, 3.)
    elif family == "tree_road":
        for i in range(3):
            for side in (-1, 1):
                box(f"tree_{i}_{side + 1}", 1.5 + i * 2, side * 3.1,
                    .4, .4, 4.)
        box("tree_center", rng.uniform(2.0, 2.5), 0, .4, .4, 4.)
    else:
        for i in range(3):
            box(f"plaza_edge_{i}", 1.2 + i * 1.9, -3.6, 1., .6, 3.)
        box("building_back", 8., 0., 1., 9., 5.)
        box("island", rng.uniform(2.0, 2.5), 0, .6, .6, 3.)
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


def build(template, seed, scene_id):
    import scene_gen as sg
    from teacher import astar_path
    family, boxes, color, relation, wp = layout(seed)
    spawn = dict(x=-.5, y=0., z=.15, yaw_deg=0., mode="default")
    path = astar_path(boxes, (spawn["x"], spawn["y"]), (wp["x"], wp["y"]))
    if not path:
        raise ValueError("campus 地图无可行教师路径")
    head, lighting = sg._apply_appearance(template[:template.index(sg.START_MARK)], seed, "v1")
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
