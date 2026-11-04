"""Opt-in asset-campus task. Never mutates historical campus.v2 scenes or splits.

Visual assets are separate from authoritative collision/planning boxes. For thin
branches and building overhangs, the asset registry must specify the navigable
height band and clearance; a visual mesh does not become collision automatically.
"""
import hashlib
import json
import random
from pathlib import Path
from xml.sax.saxutils import escape

import campus
import scene_gen as sg
from landmarks import COLOR_RGB
from instruction import COLORS

VERSION = 'campus.assets.v1'
ASSET_ROOT = Path(__file__).resolve().parents[1] / 'assets'
MODEL_DIR = ASSET_ROOT / 'uavrgb_assets'
CATALOG = ASSET_ROOT / 'asset_catalog.json'


def _hash(data):
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True,
                             separators=(',', ':')).encode()).hexdigest()


def map_key(seed):
    # Same geometry, different text/appearance across all 8 variants => same split.
    group = seed // 8
    return f'{VERSION}/{campus.FAMILIES[group % len(campus.FAMILIES)]}/{group:06d}'


def split_key(key):
    if not key.startswith(VERSION + '/'):
        raise ValueError(f'foreign asset campus key: {key}')
    family = key.split('/')[1]
    if family not in campus.FAMILIES:
        raise ValueError(f'unknown family {family}')
    bucket = int(hashlib.sha256(('asset-campus-split-v1:' + key).encode()).hexdigest()[:8], 16) % 100
    return 'train' if bucket < 80 else 'val' if bucket < 90 else 'test'


def _load_catalog():
    obj = json.loads(CATALOG.read_text(encoding='utf-8'))
    if obj.get('schema') != 'uavrgb.assets.v1':
        raise ValueError('asset catalog schema mismatch')
    # 完整 provenance/sha256 校验用 `python3 sim/tools/import_asset.py --check`；
    # 运行时不做，避免旧 register_ground_texture.py 生成的 ground 条目无法加载。
    return obj


def _safe_asset(entry):
    uri = entry['mesh']
    if not uri.startswith('model://uavrgb_assets/meshes/'):
        raise ValueError('mesh must be a local model://uavrgb_assets/meshes/... URI')
    name = uri.split('/')[-1]
    p = MODEL_DIR / 'meshes' / name
    if not p.is_file():
        raise FileNotFoundError(f'asset not installed: {p}; see docs/realism_rl_v1/README.md')
    dims = entry['dimensions_m']
    if len(dims) != 3 or min(dims) <= 0:
        raise ValueError('mesh dimensions_m must be positive [x,y,z]')
    return p


def _box_geom(p):
    return f'<box><size>{" ".join(str(v) for v in p["size"])}</size></box>'


def _xml(p, catalog):
    """Collision = authoritative axis-aligned primitive; only visual uses mesh."""
    name = p['name']
    if name.startswith('tree'):
        entry = catalog['types']['tree']
    elif name.startswith('building'):
        entry = catalog['types']['building']
    else:
        entry = None
    if name.startswith('target'):
        c = COLOR_RGB[p['color']]
        mat = ' '.join(str(v) for v in (*c, 1.0))
        visual = (f'<geometry>{_box_geom(p)}</geometry><material>'
                  f'<ambient>{mat}</ambient><diffuse>{mat}</diffuse></material>')
    elif entry:
        _safe_asset(entry)
        visual_dims = entry.get('visual_size_m', p['size'])
        scales = [visual_dims[i] / entry['dimensions_m'][i] for i in range(3)]
        # All our demo/imported meshes are centered in XY and rest on local z=0;
        # the source link pose remains centered at z=size.z/2.
        s = ' '.join(f'{v:.6f}' for v in scales)
        visual = (f'<pose>0 0 {-p["size"][2]/2:.6f} 0 0 0</pose>'
                  f'<geometry><mesh><uri>{escape(entry["mesh"])}</uri><scale>{s}</scale>'
                  '</mesh></geometry>')
        if 'albedo_map' in entry:
            visual += ('<material><pbr><metal><metalness>0</metalness>'
                       '<roughness>0.9</roughness><albedo_map>'
                       f'{escape(entry["albedo_map"])}</albedo_map>'
                       '</metal></pbr></material>')
    else:
        rgb = (0.56, 0.54, 0.51)
        mat = ' '.join(str(v) for v in (*rgb, 1.0))
        visual = (f'<geometry>{_box_geom(p)}</geometry><material><ambient>{mat}</ambient>'
                  f'<diffuse>{mat}</diffuse></material>')
    return (f'<model name="{escape(name)}"><static>true</static>'
            f'<pose>{p["x"]} {p["y"]} {p["z"]} 0 0 0</pose>'
            f'<link name="link"><collision name="collision"><geometry>'
            f'{_box_geom(p)}</geometry></collision><visual name="visual">'
            f'{visual}</visual></link></model>')


def layout(seed):
    family, obstacles, _old_color, _old_relation, _old_goal = campus.layout(seed)
    # Two targets on EACH side, with different color. Color and left/right both
    # matter once same-side distractors exist; no 'color is irrelevant' shortcut.
    obstacles = [dict(p) for p in obstacles if not p['name'].startswith('target_')]
    group = seed // 8
    rng = random.Random(group ^ 0xB09C31)
    base_x = rng.uniform(3.6, 4.5)
    y = rng.uniform(1.55, 1.9)
    # Colors are position-independent over variants; geometry remains identical.
    chosen = [COLORS[(seed + j) % len(COLORS)] for j in range(4)]
    # Each side has 2 different colors: [left near, left far, right near, right far].
    for side, token, indices in ((1, 'left', (0, 1)), (-1, 'right', (2, 3))):
        for j, idx in enumerate(indices):
            obstacles.append({'name': f'target_{token}_{j}', 'x': round(base_x + j * 2.4, 3),
                              'y': round(side*y, 3), 'z': 1.5,
                              'size': [.4, .4, 3.0], 'color': chosen[idx]})
    relation = 'left' if (seed // 2) % 2 == 0 else 'right'
    target_index = seed % 2
    target_name = f'target_{relation}_{target_index}'
    target = next(p for p in obstacles if p['name'] == target_name)
    waypoint = {'x': round(target['x'] - 1.1, 3), 'y': target['y']}
    return family, obstacles, target['color'], relation, waypoint, target_name


def build(template, seed, scene_id, appearance='v1'):
    from teacher import astar_path
    if appearance not in (sg.APPEARANCE_LEGACY, sg.APPEARANCE_V1):
        raise ValueError(f'unknown appearance {appearance!r}')
    cat = _load_catalog()
    fam, obstacles, color, relation, goal, target_name = layout(seed)
    spawn = {'x': -.5, 'y': 0., 'z': .15, 'yaw_deg': 0., 'mode': 'default'}
    if not astar_path(obstacles, (spawn['x'], spawn['y']), (goal['x'], goal['y'])):
        raise ValueError('asset campus A* has no route: adjust seed/layout')
    for mark in (sg.START_MARK, sg.END_MARK):
        if template.count(mark) != 1:
            raise ValueError(f'missing or duplicate template marker {mark}')
    prefix, lighting = sg._apply_appearance(template[:template.index(sg.START_MARK)], seed,
                                           sg.APPEARANCE_V1, full_sun=False)
    models = [_xml(p, cat) for p in obstacles]
    # Colored 2D goal mark only at active target; visual object, not a collision body.
    models.append(campus._ground_marks(goal['x'], goal['y'], color))
    world = prefix + '\n'.join(models) + template[template.index(sg.END_MARK):]
    world = sg._apply_spawn(world, spawn)
    # Ground PBR is visual-only: original SDF ground collider remains unchanged.
    i = world.index('<model name="ground_plane">')
    a = world.index('<material>', i)
    b = world.index('</material>', a)+len('</material>')
    ground_uri = cat['ground']['albedo_map']
    ground_mat = ('<material><ambient>0.9 0.9 0.9 1</ambient>'
                  '<diffuse>0.9 0.9 0.9 1</diffuse><pbr><metal>'
                  '<metalness>0</metalness><roughness>0.95</roughness>'
                  f'<albedo_map>{escape(ground_uri)}</albedo_map>'
                  '</metal></pbr></material>')
    world = world[:a]+ground_mat+world[b:]
    resources = {}
    for item in MODEL_DIR.rglob('*'):
        if item.is_file():
            resources[str(item.relative_to(MODEL_DIR))] = hashlib.sha256(item.read_bytes()).hexdigest()
    resources['catalog'] = hashlib.sha256(CATALOG.read_bytes()).hexdigest()
    metadata = {'schema': 'navlab.scene.v1', 'scene_id': scene_id, 'seed': seed,
                'generator': 'sim/tools/asset_campus.py', 'geometry_version': VERSION,
                'task_version': VERSION, 'appearance_version': VERSION,
                'requested_appearance': appearance,
                'effective_appearance': sg.APPEARANCE_V1,
                'appearance_schema_version': sg.APPEARANCE_SCHEMA_VERSION_FROZEN,
                'map_id': map_key(seed), 'family': fam, 'split_policy': 'campus_assets_map_v1',
                'target_name': target_name, 'target_color': color, 'relation': relation,
                'instruction': campus.instruction(color, relation),
                'waypoint': goal, 'goal_radius': campus.RADIUS,
                'stop_radius': campus.RADIUS, 'success_criterion': 'visible_goal_reach_v1',
                'spawn': spawn, 'pillars': obstacles, 'pillar_count': len(obstacles),
                'camera_profile': 'rgb320x240_fov60_v1', 'dynamics_profile': 'x3_v1',
                'appearance': lighting, 'clearance': {'ok': True,
                  'cruise_height_m': sg.CRUISE_HEIGHT, 'note': 'A* checked, not Gazebo-collision-verified'},
                'resource_sha256': _hash(resources), 'asset_source': cat.get('source'),
                'template_sha256': sg.sha256(template.encode()),
                'world_sha256': sg.sha256(world.encode())}
    return world, metadata
