#!/usr/bin/env python3
"""Use independently license-verified CC0 pavement photograph as Gazebo ground albedo.
A copy is included in the ZIP, so runtime needs no external network.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'assets'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--image',required=True)
    p.add_argument('--source',required=True,help='download provenance URL or author')
    p.add_argument('--license',default='CC0')
    a=p.parse_args()
    if a.license!='CC0':p.error('only CC0 images supported in packaged local assets')
    f=Path(a.image)
    if not f.is_file() or f.suffix.lower() not in ('.png','.jpg','.jpeg'):
        p.error('image must be local PNG/JPEG')
    if not a.source.strip():p.error('source required')
    d=ROOT/'uavrgb_assets'/'materials'/('pavement'+f.suffix.lower())
    if f.resolve()!=d.resolve():shutil.copy2(f,d)
    cat=ROOT/'asset_catalog.json'
    data=json.loads(cat.read_text())
    data['ground']={'albedo_map':'model://uavrgb_assets/materials/'+d.name,
                    'provenance':{'url':a.source,'license':'CC0',
                        'sha256':hashlib.sha256(d.read_bytes()).hexdigest()}}
    cat.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print('Registered ground albedo:',d)

if __name__=='__main__':main()
