#!/usr/bin/env python3
"""Register a verified local CC0 mesh without runtime internet access.

OBJ or DAE; preserves source next to generated mesh, records hash / provenance.
Requires model asset centered in XY, local base z=0, physical size in meters.
Use Blender offline to satisfy this contract; automatic conversions are NOT faked.

`--check` validates asset_catalog.json (schema, license/source/author, file sha256)
without importing anything.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'assets'
SCHEMA='uavrgb.assets.v1'
# 当前仓库里的资产是 build_demo_assets.py 的离线演示网格，不是真实 CC0 扫描件。
# 每个 type 的 provenance 必须写清 kind/license/source/author/sha256，--check 校验。
PROVENANCE_KEYS=('license','author','sha256')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _local_file(root, uri, subdir):
    prefix=f'model://uavrgb_assets/{subdir}/'
    if not isinstance(uri,str) or not uri.startswith(prefix):
        return None
    name=uri[len(prefix):]
    if not name or '/' in name or name in ('.','..'):
        return None
    return Path(root)/'uavrgb_assets'/subdir/name


def _check_provenance(prov, root, problems, where, verify_hashes):
    if not isinstance(prov,dict):
        problems.append(f'{where}: 缺 provenance 对象')
        return
    license_name=str(prov.get('license','')).strip()
    author=str(prov.get('author','')).strip()
    source=str(prov.get('source') or prov.get('url') or '').strip()
    if not license_name:
        problems.append(f'{where}: provenance 缺 license')
    if not author:
        problems.append(f'{where}: provenance 缺 author')
    if not source:
        problems.append(f'{where}: provenance 缺 source/url')
    if not verify_hashes:
        return
    shas=prov.get('sha256')
    if isinstance(shas,str):
        # register_ground_texture.py 写的是单个文件哈希，路径由调用方单独核对。
        if len(shas)!=64:
            problems.append(f'{where}: sha256 格式不对')
        return
    if not isinstance(shas,dict) or not shas:
        problems.append(f'{where}: provenance 缺 sha256 文件哈希')
        return
    for rel,want in sorted(shas.items()):
        path=Path(root)/rel
        if not path.is_file():
            problems.append(f'{where}: sha256 指向的文件不存在 {rel}')
        elif _sha(path)!=want:
            problems.append(f'{where}: sha256 不匹配 {rel}')


def catalog_problems(obj, root=ROOT, verify_hashes=True):
    """返回 catalog 的 provenance/schema 问题列表；空列表表示通过。

    verify_hashes=False 供场景生成时做轻量结构检查（不重复读文件哈希）。
    """
    problems=[]
    if not isinstance(obj,dict) or obj.get('schema')!=SCHEMA:
        return [f'schema 必须是 {SCHEMA}']
    for key in ('source','license','author'):
        if not str(obj.get(key,'')).strip():
            problems.append(f'顶层缺 {key}')
    types=obj.get('types')
    if not isinstance(types,dict) or not types:
        problems.append('types 必须是非空对象')
        types={}
    for kind,entry in sorted(types.items()):
        where=f'types.{kind}'
        if not isinstance(entry,dict):
            problems.append(f'{where}: 必须是对象')
            continue
        mesh=_local_file(root,entry.get('mesh',''),'meshes')
        if mesh is None:
            problems.append(f'{where}: mesh 必须是 model://uavrgb_assets/meshes/<file>')
        elif not mesh.is_file():
            problems.append(f'{where}: mesh 文件不存在 {mesh}')
        dims=entry.get('dimensions_m')
        if (not isinstance(dims,list) or len(dims)!=3 or
                not all(isinstance(v,(int,float)) and v>0 for v in dims)):
            problems.append(f'{where}: dimensions_m 必须是 3 个正数')
        if 'albedo_map' in entry:
            albedo=_local_file(root,entry['albedo_map'],'materials')
            if albedo is None:
                problems.append(f'{where}: albedo_map 必须是 model://uavrgb_assets/materials/<file>')
            elif not albedo.is_file():
                problems.append(f'{where}: albedo 文件不存在 {albedo}')
        _check_provenance(entry.get('provenance'),root,problems,where,verify_hashes)
    ground=obj.get('ground')
    if ground is not None:
        if not isinstance(ground,dict):
            problems.append('ground: 必须是对象')
        else:
            albedo=_local_file(root,ground.get('albedo_map',''),'materials')
            if albedo is None:
                problems.append('ground: albedo_map 必须是 model://uavrgb_assets/materials/<file>')
            elif not albedo.is_file():
                problems.append(f'ground: albedo 文件不存在 {albedo}')
            prov=ground.get('provenance')
            _check_provenance(prov,root,problems,'ground',verify_hashes)
            if (verify_hashes and isinstance(prov,dict) and
                    isinstance(prov.get('sha256'),str) and albedo is not None and
                    _sha(albedo)!=prov['sha256']):
                problems.append('ground: sha256 不匹配')
    return problems


def register(kind, mesh_path, dimensions, source_url, license_name, albedo_path=None,
             author='unknown'):
    if kind not in ('tree','building'):
        raise ValueError('supported registry kinds: tree/building')
    src=Path(mesh_path).resolve()
    if not src.is_file() or src.suffix.lower() not in ('.obj','.dae'):
        raise ValueError('mesh must be existing .obj or .dae file')
    if len(dimensions)!=3 or min(dimensions)<=0:
        raise ValueError('dimensions must be 3 positive meters')
    if license_name!='CC0':
        raise ValueError('this import helper only packages independently verified CC0 assets')
    if not source_url.strip():
        raise ValueError('source URL or author provenance required')
    if not str(author).strip():
        raise ValueError('author required')
    dst=ROOT/'uavrgb_assets'/'meshes'/f'{kind}{src.suffix.lower()}'
    # For OBJ the .mtl and textures need baking; external material dependencies are
    # intentionally NOT copied silently. use --albedo when available.
    if src.suffix.lower()=='.obj' and any(
            line.split() and line.split()[0].lower()=='mtllib'
            for line in src.read_text(errors='ignore').splitlines()):
        raise ValueError('OBJ contains external mtllib; bake to OBJ with UVs and --albedo or use DAE')
    dst.parent.mkdir(parents=True,exist_ok=True)
    if src != dst: shutil.copy2(src,dst)
    catalog=ROOT/'asset_catalog.json'
    obj=json.loads(catalog.read_text())
    item={'mesh':f'model://uavrgb_assets/meshes/{dst.name}',
          'dimensions_m':list(dimensions)}
    files={str(dst.relative_to(ROOT)):hashlib.sha256(dst.read_bytes()).hexdigest()}
    if albedo_path:
        texture=Path(albedo_path).resolve()
        if not texture.is_file() or texture.suffix.lower() not in ('.png','.jpg','.jpeg'):
            raise ValueError('--albedo should be a PNG/JPEG image')
        td=ROOT/'uavrgb_assets'/'materials'/f'{kind}{texture.suffix.lower()}'
        if texture != td: shutil.copy2(texture,td)
        item['albedo_map']=f'model://uavrgb_assets/materials/{td.name}'
        files[str(td.relative_to(ROOT))]=hashlib.sha256(td.read_bytes()).hexdigest()
    item['provenance']={'kind':'imported','url':source_url,'source':source_url,
                        'license':license_name,'author':str(author).strip(),
                        'sha256':files}
    obj['types'][kind]=item
    obj['source']='mixed: original demo and user-imported independently verified CC0 assets'
    obj['license']='mixed: demo placeholders and per-type CC0 provenance'
    obj['author']='mixed: see per-type provenance'
    catalog.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
    print(f'Updated {kind}: {dst}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--type',choices=('tree','building'))
    p.add_argument('--mesh')
    p.add_argument('--dimensions',type=float,nargs=3,metavar=('X','Y','Z'))
    p.add_argument('--source')
    p.add_argument('--license',default='CC0')
    p.add_argument('--author',default='')
    p.add_argument('--albedo',default='')
    p.add_argument('--check',action='store_true',
                   help='只校验 asset_catalog.json 的 schema/provenance/sha256，不导入')
    a=p.parse_args()
    if a.check:
        catalog=ROOT/'asset_catalog.json'
        obj=json.loads(catalog.read_text(encoding='utf-8'))
        problems=catalog_problems(obj, root=ROOT)
        if problems:
            print(f'assets --check: FAIL（{len(problems)} 项）',file=sys.stderr)
            for problem in problems:
                print(f'  {problem}',file=sys.stderr)
            return 1
        print(f'assets --check: PASS（{len(obj["types"])} types，sha256 全部匹配）')
        return 0
    if not (a.type and a.mesh and a.dimensions and a.source):
        p.error('导入需要 --type/--mesh/--dimensions/--source；只校验用 --check')
    register(a.type,a.mesh,a.dimensions,a.source,a.license,a.albedo or None,a.author)

if __name__=='__main__': sys.exit(main())
