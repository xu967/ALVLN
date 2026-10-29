#!/usr/bin/env python3
"""V7 minimal Gazebo diagnostic + explicit not-yet-implemented features.

This command does not assert quadrotor dynamics/camera closed-loop is implemented.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

MOTOR_PLUGIN_FILENAMES = {'gz-sim-multicopter-motor-model-system', 'libgz-sim-multicopter-motor-model-system.so',
                          'ignition-gazebo-multicopter-motor-model-system'}
MOTOR_PLUGIN_NAMES = {'gz::sim::systems::MulticopterMotorModel', 'ignition::gazebo::systems::MulticopterMotorModel'}


def main() -> int:
    parser = argparse.ArgumentParser(description="TinyAerialVLN Gazebo bring-up diagnostic")
    parser.add_argument("stage", choices=["doctor", "smoke", "camera", "manual", "cf-sil", "expert", "record"])
    parser.add_argument("--world", help="Version-pinned SDF world for gazebo smoke")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gz = shutil.which("gz")
    if args.stage == "doctor":
        if not gz:
            print("[ERR][GZ_NOT_INSTALLED] gz command unavailable. Install and pin Gazebo Harmonic.", file=sys.stderr)
            return 127
        print(f"[OK][GZ_CLI_FOUND] {gz}")
        print("[BLOCKED][GZ_PHYSICS_NOT_VERIFIED] CLI presence does not prove motor, camera or control-loop behavior.")
        return 0
    if args.stage != "smoke":
        print(f"[ERR][FEATURE_NOT_IMPLEMENTED] sim {args.stage} requires GZ0/GZ1/CF gate implementation.", file=sys.stderr)
        return 3
    if not args.world:
        print("[ERR][GZ_WORLD_NOT_SPECIFIED] Supply --world with a pinned quadcopter.sdf file.", file=sys.stderr)
        return 4
    world = Path(args.world).expanduser().resolve()
    if not world.is_file():
        print(f"[ERR][GZ_WORLD_NOT_FOUND] {world}", file=sys.stderr)
        return 4
    try:
        doc = ET.parse(world).getroot()
    except ET.ParseError as exc:
        print(f"[ERR][SDF_XML_INVALID] {world}: {exc}", file=sys.stderr)
        return 4
    if doc.tag != 'sdf' or doc.find('world') is None:
        print("[ERR][SDF_WORLD_INVALID] Expected an SDF root containing a world.", file=sys.stderr)
        return 4
    # Count real rotor-force plugins; a velocity controller or a static scene is not motor dynamics.
    motors = [p for p in doc.iter('plugin')
              if p.get('filename', '') in MOTOR_PLUGIN_FILENAMES or p.get('name', '') in MOTOR_PLUGIN_NAMES]
    if len(motors) < 4:
        print(f"[ERR][GZ_MOTOR_MODEL_MISSING] found {len(motors)} MulticopterMotorModel plugin(s), need >= 4. "
              "Static scene or velocity-control plugin != flight dynamics. If the plugins live inside an "
              "<include>d model file, pass an expanded world.", file=sys.stderr)
        return 4
    if not gz:
        print("[ERR][GZ_NOT_INSTALLED] Install Gazebo Harmonic and retry.", file=sys.stderr)
        return 127
    cmd = [gz, 'sim', '-v', '4', str(world)]
    print('[INFO] Command: ' + ' '.join(cmd))
    print('[WARN][MANUAL_GATE_REQUIRED] Process launch is NOT a motor-response/camera/physics PASS; inspect and record traces.')
    if args.dry_run:
        return 0
    try:
        return subprocess.call(cmd)
    except OSError as exc:
        print(f'[ERR][GZ_PROCESS_START_FAILED] {exc}', file=sys.stderr)
        return 5


if __name__ == '__main__':
    raise SystemExit(main())
