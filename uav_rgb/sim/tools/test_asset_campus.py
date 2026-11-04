#!/usr/bin/env python3
"""Pure-stdlib offline smoke for asset campus: no Gazebo GUI required."""
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import asset_campus as ac
import scene_gen as sg
import split_manifest as sm
import teacher


def main():
    template=sg._read_template(sg.DEFAULT_TEMPLATE)
    splits={'train':set(),'val':set(),'test':set()}
    for seed in range(160):
        world,meta=sg.build_scene(template,seed,seed,task='campus_assets')
        world2,meta2=sg.build_scene(template,seed,seed,task='campus_assets')
        assert world==world2 and meta==meta2
        assert hashlib.sha256(world.encode()).hexdigest()==meta['world_sha256']
        root=ET.fromstring(world)
        models=root.find('world').findall('model')
        collision_names={m.get('name') for m in models if m.find('.//collision') is not None}
        assert collision_names-{'ground_plane','x3'}=={p['name'] for p in meta['pillars']}
        assert len([p for p in meta['pillars'] if p['name'].startswith('target_')])==4
        own=[p for p in meta['pillars'] if p['name'].startswith('target_'+meta['relation'])]
        assert len(own)==2 and own[0]['color']!=own[1]['color']
        chosen=next(p for p in own if p['name']==meta['target_name'])
        assert chosen['color']==meta['target_color']
        assert abs(chosen['y']-meta['waypoint']['y'])<1e-8
        assert not teacher._point_blocked(meta['waypoint']['x'],meta['waypoint']['y'],meta['pillars'],.5)
        assert teacher.astar_path(meta['pillars'],(-.5,0),(meta['waypoint']['x'],meta['waypoint']['y']))
        split=sm.split_for_scene(meta)
        assert split==sm.split_for_seed(seed,task='campus_assets')
        splits[split].add(meta['map_id'])
        for p in meta['pillars']:
            assert all(math.isfinite(float(v)) for v in p['size'])
        if seed%8 != 0:
            assert ac.map_key(seed)==ac.map_key(seed-1)
            assert sm.split_for_seed(seed,task='campus_assets') == sm.split_for_seed(seed-1,task='campus_assets')
        for visual in root.findall('.//visual'):
            mesh=visual.find('./geometry/mesh/uri')
            if mesh is not None:
                assert mesh.text.startswith('model://uavrgb_assets/meshes/')
                filename=mesh.text.split('/')[-1]
                assert (ac.MODEL_DIR/'meshes'/filename).is_file()
    assert all(splits.values()), {s:len(v) for s,v in splits.items()}
    forged=dict(meta,map_id='campus.assets.v1/tree_road/999999')
    try: sm.split_for_scene(forged)
    except ValueError: pass
    else: raise AssertionError('map_id forgery was not rejected')
    print('asset campus PASS: 160 worlds, offline geometry/split/render references',
          {s:len(v) for s,v in splits.items()})

def test_catalog_provenance_and_appearance_meta():
    """catalog 必须有 license/source/author/sha256；meta 记录 requested/effective。"""
    import import_asset
    obj = json.loads(ac.CATALOG.read_text(encoding='utf-8'))
    assert import_asset.catalog_problems(obj, root=ac.ASSET_ROOT) == []
    for kind in ('tree', 'building'):
        prov = obj['types'][kind]['provenance']
        assert prov['license'] and prov['source'] and prov['author'], kind
        assert prov['sha256'], kind
    assert obj['ground']['provenance']['author']
    template = sg._read_template(sg.DEFAULT_TEMPLATE)
    world, meta = sg.build_scene(template, 16, 16, task='campus_assets',
                                 appearance=sg.APPEARANCE_DEFAULT)
    assert meta['requested_appearance'] == meta['effective_appearance'] == 'v1'
    assert meta['appearance_schema_version'] == sg.APPEARANCE_SCHEMA_VERSION_FROZEN
    assert str(sg.REPO_ROOT) not in world
    print('asset campus: catalog provenance / appearance meta PASS')


if __name__=='__main__':
    main()
    test_catalog_provenance_and_appearance_meta()
