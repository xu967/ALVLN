#!/usr/bin/env python3
"""Check Gazebo CLI and optional known-motor world XML; never fabricate a flight trace."""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

def main():
    p=argparse.ArgumentParser();p.add_argument('--world',type=Path);args=p.parse_args()
    gz=shutil.which('gz')
    if not gz:print('[BLOCKED][GZ_NOT_INSTALLED] Install Gazebo Harmonic before GZ0.');return 3
    try:
        r=subprocess.run([gz,'sim','--help'],capture_output=True,text=True,timeout=12)
        if r.returncode!=0:raise RuntimeError('gz sim --help returned nonzero')
    except (OSError,subprocess.TimeoutExpired,RuntimeError) as exc:
        print('[BLOCKED][GZ_RUNTIME_PROBE_FAILED] '+str(exc));return 4
    print('[PASS][GZ_CLI_AVAILABLE] gz sim command responds. This does NOT verify physics.')
    if args.world:
        if not args.world.exists():print('[BLOCKED][WORLD_NOT_FOUND]');return 3
        try:tree=ET.parse(args.world)
        except ET.ParseError:print('[BLOCKED][WORLD_INVALID_XML]');return 4
        motors=[]
        for x in tree.findall('.//plugin'):
            if 'MulticopterMotorModel' in (x.get('name') or ''):
                n=x.findtext('actuator_number')
                if n is not None:motors.append(int(n))
        if sorted(motors)!=[0,1,2,3]:
            print(f'[BLOCKED][GZ_MOTOR_LAYOUT_UNVERIFIED] visible motors={motors}; includes may hide model plugins. Inspect resolved SDF.')
            return 5
        print('[STATIC_CHECK] Four distinct actuator indices in this XML; physical flight remains UNVERIFIED.')
    return 0
if __name__=='__main__':sys.exit(main())
