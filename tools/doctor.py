#!/usr/bin/env python3
"""Version probes only. [FOUND] != verified Gazebo flight / CUDA training."""
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def probe(name,cmd):
    binary=shutil.which(name)
    if not binary:
        return {'tool':name,'state':'MISSING','info':'not in PATH'}
    try:
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=8)
        lines=(r.stdout+'\n'+r.stderr).splitlines()
        return {'tool':name,'state':'RESPONDED' if r.returncode==0 else 'FOUND_NOT_HEALTHY',
                'exit_code':r.returncode,'info':next((x[:180] for x in lines if x.strip()),'no output')}
    except (OSError,subprocess.TimeoutExpired) as exc:
        return {'tool':name,'state':'FOUND_NOT_HEALTHY','info':type(exc).__name__}


def main():
    checks=[probe('python3',['python3','--version']),probe('g++',['g++','--version']),
            probe('cmake',['cmake','--version']),probe('nvcc',['nvcc','--version']),
            probe('nvidia-smi',['nvidia-smi','-L']),probe('gz',['gz','--help'])]
    print(json.dumps({'system':platform.platform(),'python':sys.version.split()[0],
                      'checks':checks,'real_flight_verified':False,'npu_verified':False},
                     indent=2,ensure_ascii=False))
    return 0
if __name__=='__main__':sys.exit(main())
