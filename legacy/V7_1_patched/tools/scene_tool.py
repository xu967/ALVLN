#!/usr/bin/env python3
"""Reproducible static geometry scene generator (NOT a flight simulator)."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ('forest', 'corridor', 'columns', 'mixed')


def validate_path(path: Path) -> Path:
    target = path.resolve()
    if not target.is_relative_to((ROOT / 'artifacts').resolve()):
        raise ValueError('SCENE_OUTPUT_PATH: 输出只能位于项目 artifacts/ 内。')
    return target


def create_obstacles(preset: str, seed: int, count: int) -> list[dict]:
    rng = random.Random(seed)
    obstacles = []
    for i in range(count):
        if preset == 'corridor':
            x = rng.uniform(-8.6, 8.6)
            y = rng.choice([-1, 1]) * rng.uniform(2.0, 3.0)
            radius = rng.uniform(.22, .52)
        elif preset == 'columns':
            x = -7 + (i % 7) * 2.3 + rng.uniform(-.35, .35)
            y = -5 + (i // 7) * 2.0 + rng.uniform(-.3, .3)
            radius = rng.uniform(.20, .45)
        else:
            x, y = rng.uniform(-8.5, 8.5), rng.uniform(-6.5, 6.5)
            radius = rng.uniform(.23, .55)
        # Reserve start, goal and a 1.4m-wide central reference corridor.
        if math.hypot(x + 8, y) < 1.4 or math.hypot(x - 8, y) < 1.4:
            continue
        if abs(y) < 1.15 + radius:
            continue
        height = round(rng.uniform(1.8, 4.0), 3)
        obstacles.append({'id': f'obs_{i:03}', 'x': round(x, 3), 'y': round(y, 3),
                          'r': round(radius, 3), 'h': height})
    return obstacles


def make_sdf(obstacles: list[dict], seed: int) -> str:
    # Static geometry / RGB-visual elements only. Physics UAV model is a separate task.
    s = ['<?xml version="1.0"?>', '<sdf version="1.9">',
         f'<world name="tinyvln_{seed}">',
         '<physics name="step" type="ignored"><max_step_size>0.002</max_step_size></physics>',
         '<gravity>0 0 -9.81</gravity>',
         '<light name="sun" type="directional"><cast_shadows>true</cast_shadows>'
         '<pose>0 0 10 0 0 0</pose><diffuse>0.85 0.85 0.85 1</diffuse>'
         '<specular>0.1 0.1 0.1 1</specular><direction>-0.4 0.3 -0.9</direction></light>',
         '<model name="ground"><static>true</static><link name="ground">'
         '<collision name="c"><geometry><box><size>22 16 .2</size></box></geometry></collision>'
         '<visual name="v"><geometry><box><size>22 16 .2</size></box></geometry>'
         '<material><diffuse>.45 .5 .45 1</diffuse></material></visual>'
         '<pose>0 0 -.1 0 0 0</pose></link></model>']
    for o in obstacles:
        name = escape(o['id'])
        s.append(f'<model name="{name}"><static>true</static><pose>{o["x"]} {o["y"]} 0 0 0 0</pose>'
                 f'<link name="body"><pose>0 0 {o["h"] / 2:.3f} 0 0 0</pose>'
                 '<collision name="c"><geometry><cylinder>'
                 f'<radius>{o["r"]}</radius><length>{o["h"]}</length>'
                 '</cylinder></geometry></collision><visual name="v"><geometry><cylinder>'
                 f'<radius>{o["r"]}</radius><length>{o["h"]}</length>'
                 '</cylinder></geometry><material><diffuse>.42 .36 .27 1</diffuse>'
                 '</material></visual></link></model>')
    # Goal is just a visual marker; no physical goal detector or policy here.
    s.append('<model name="goal_marker"><static>true</static><pose>8 0 0 0 0 0</pose>'
             '<link name="marker"><pose>0 0 .15 0 0 0</pose>'
             '<visual name="v"><geometry><sphere><radius>.15</radius></sphere></geometry>'
             '<material><diffuse>1 .12 .12 1</diffuse></material></visual></link></model>')
    s += ['</world>', '</sdf>']
    return '\n'.join(s) + '\n'


def generate(preset: str, seed: int, count: int, out: Path) -> Path:
    if preset not in PRESETS or count < 0 or count > 200:
        raise ValueError('SCENE_ARGUMENT: preset 不支持或 count 必须为 0..200。')
    out = validate_path(out)
    out.mkdir(parents=True, exist_ok=True)
    obstacles = create_obstacles(preset, seed, count)
    sdf = make_sdf(obstacles, seed)
    world_path = out / f'{preset}_{seed}.sdf'
    manifest_path = out / f'{preset}_{seed}.json'
    world_path.write_text(sdf, encoding='utf-8')
    manifest = {'schema_version': 'scene-v0.1', 'seed': seed, 'preset': preset,
                'requested_obstacles': count, 'placed_obstacles': len(obstacles),
                'start_xyz': [-8, 0, 1], 'goal_xyz': [8, 0, 1],
                'world_file': world_path.name, 'sha256': hashlib.sha256(sdf.encode()).hexdigest(),
                'notes': 'Static SDF geometry only: not RGB camera, UAV dynamics, expert policy or VLN episodes.',
                'obstacles': obstacles}
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return world_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='生成静态 SDF 场景；不等于训练闭环。')
    p.add_argument('--preset', choices=PRESETS, default='forest')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--count', type=int, default=24)
    p.add_argument('--out', default='artifacts/scenes')
    args = p.parse_args(argv)
    try:
        out = ROOT / args.out
        target = generate(args.preset, args.seed, args.count, out)
        print(f'[OK] 场景已生成: {target}')
        print('[INFO] 仅静态几何；需在 Gazebo 中接入带相机的飞行器和动作适配器。')
        return 0
    except (ValueError, OSError) as e:
        print(f'[ERR][SCENE_GENERATION] {e}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
