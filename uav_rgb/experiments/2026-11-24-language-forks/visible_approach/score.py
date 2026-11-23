from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

out=Path(__file__).resolve().parent
reg=json.loads((out/'registration.json').read_text())
read=lambda p:[json.loads(s) for s in p.read_text().splitlines()]
sha=lambda p:hashlib.sha256(Image.open(p).convert('RGB').tobytes()).hexdigest()
waypoints={r['requested']:np.asarray([r['waypoint'][k] for k in 'xy']) for r in reg['rows']}
records=[]
for model in ['baseline','dino']:
    for r in reg['rows']:
        name=f"{model}_{r['visible']}_to_{r['requested']}"
        v=json.loads((out/name/'vln_eval.json').read_text());assert not v['failures']
        ep,=read(out/name/'episodes.jsonl');d=Path(ep['dir']);aa=read(d/'actions.jsonl');tr=read(d/'trace.jsonl');man=json.loads((d/'episode_manifest.json').read_text());assert man['seed']==421 and man['requested_steps']==125 and ep['supplied_text']==r['text']
        first_rgb=d/aa[0]['frame'];check=out/(r['visible']+'_start.png');assert sha(first_rgb)==sha(check)
        xy=np.asarray([[aa[0]['pose_before'][k] for k in 'xy']]+[t['pos'][:2] for t in tr]);requested=r['requested'];other=next(k for k in waypoints if k!=requested);errors=np.linalg.norm(xy-waypoints[requested],axis=1);alternative=np.linalg.norm(xy-waypoints[other],axis=1);assert abs(errors[-1]-ep['supplied_goal_ne_m'])<1e-4
        stop=ep['stopped'];report=dict(model=model,visible=r['visible'],requested=requested,matched=r['visible']==requested,success=ep['supplied_goal_success'],osr=ep['supplied_goal_oracle_success'],spl=ep['supplied_goal_spl'],stopped=stop,collision=ep['collision'],steps=len(aa),terminal=ep['terminal_reason'],ne_m=errors[-1],start_m=errors[0],closest_m=min(errors),gain_m=errors[0]-errors[-1],other_end_m=alternative[-1],wrong_disk_stop=bool(stop and alternative[-1]<=1),far_stop=bool(stop and errors[-1]>1 and alternative[-1]>1),path_m=ep['path_m'],shortest_m=ep['supplied_goal_shortest_m'],first_rgb_sha256=sha(first_rgb),first_rgb=str(first_rgb),last_decision_rgb=str(d/aa[-1]['frame']),wall_s=v['wall_s'],movements=dict(Counter(a['action_executed'] for a in aa)))
        if model=='dino':
            supported=[a for a in aa if a['dino']['box'] is not None];report.update(boxes=len(supported),degrees_and_ranges_first=supported[0]['dino'] if supported else None,final_dino=aa[-1]['dino'],final_near_streak=aa[-1]['near_streak'],first_box_step=supported[0]['step'] if supported else None,supplied_phrase=aa[0]['dino']['phrase'],aligned_boxes=sum(abs(a['dino']['angle_deg'])<=9 for a in supported))
        records.append(report)
rollups={}
for model in ('baseline','dino'):
    rollups[model]={}
    for scope in ('all','matched','swapped'):
        subset=[r for r in records if r['model']==model and (scope=='all' or (scope=='matched')==r['matched'])];rollups[model][scope]={k:sum(r[k] for r in subset) for k in ('success','osr','stopped','collision','wrong_disk_stop','far_stop')};rollups[model][scope].update(n=len(subset),mean_ne_m=np.mean([r['ne_m'] for r in subset]),mean_spl=np.mean([r['spl'] for r in subset]))
(out/'scores.json').write_text(json.dumps(dict(results=records,aggregate=rollups),indent=2)+'\n');print(json.dumps(rollups,indent=2))

# 离线核验读取 USD 物体表面、任务航点与真实固件轨迹。所有值只用于评分。
if __name__=='__main__' and __import__('sys').argv[1:]==['geometry']:
    from pxr import Usd,UsdGeom
    source=out.parent/'exp-vln-new-scenes-20261124';usd=source/'uav_rgb/pubdata/sage3d/collision/839934_collision.usd'
    witness=next(r for r in json.loads((source/'witness/witness_audit.json').read_text())['rows'] if r['target']=='cabinet_1');prim_name=witness['cad_hits'][0]['prim'];stage=Usd.Stage.Open(str(usd));prim=stage.GetPrimAtPath(prim_name);mesh=UsdGeom.Mesh(prim);xf=np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim));verts=np.asarray(mesh.GetPointsAttr().Get());verts=(verts@xf[:3,:3]+xf[3,:3])*UsdGeom.GetStageMetersPerUnit(stage)
    sizes=mesh.GetFaceVertexCountsAttr().Get();indices=mesh.GetFaceVertexIndicesAttr().Get();faces=[];offset=0
    for n in sizes:
        for i in range(1,n-1):faces.append([indices[offset],indices[offset+i],indices[offset+i+1]])
        offset+=n
    tri=verts[faces];a,b,c=tri.transpose(1,0,2);ab=b-a;ac=c-a;normal=np.cross(ab,ac);len2=np.einsum('ij,ij->i',normal,normal);good=len2>1e-15
    def surface(point):
        d=np.einsum('ij,ij->i',point-a,normal)/np.where(good,len2,1); plane=point-d[:,None]*normal
        # 三角形垂足在内部时取平面距离，外部取三条边最近点。
        inside=good.copy();edge_distances=[]
        for p,q in ((a,b),(b,c),(c,a)):
            edge=q-p;t=np.einsum('ij,ij->i',point-p,edge)/np.maximum(1e-18,np.einsum('ij,ij->i',edge,edge)); nearest=p+np.clip(t,0,1)[:,None]*edge;edge_distances.append(np.linalg.norm(nearest-point,axis=1));inside &= np.einsum('ij,ij->i',np.cross(edge,plane-p),normal)>=-1e-12
        return float(np.minimum.reduce([np.where(inside,np.linalg.norm(plane-point,axis=1),np.inf),*edge_distances]).min())
    rec=next(r for r in records if r['model']=='dino' and r['requested']==r['visible']=='tr_69');ep=Path(rec['last_decision_rgb']).parent;acts=read(ep/'actions.jsonl');trace=read(ep/'trace.jsonl');final_action=acts[-1]; nav=np.array(json.loads((ep/'scene.json').read_text())['nav_to_world']);xywp=waypoints['tr_69']
    target=next(t for t in json.loads(next((source/'uav_rgb/pubdata/sage3d/interiorgs_labels').glob('*839934/labels.json')).read_text()) if t['ins_id']=='79');box=np.array([[p[k] for k in 'xyz'] for p in target['bounding_box']]);edges=box[[1,3,4]]-box[0];length=np.linalg.norm(edges,axis=1);basis=edges/length[:,None]
    poses={}
    for name,xyz in [('before_stop',[final_action['pose_before'][k] for k in 'xyz']),('after_stop',trace[-1]['pos'])]:
        world=np.array(xyz)@nav[:3,:3].T+nav[:3,3];q=(world-box[0])@basis.T;gap=float(np.linalg.norm(q-np.clip(q,0,length)));wp=float(np.linalg.norm(np.array(xyz)[:2]-xywp));poses[name]=dict(body_xyz_nav_m=xyz,distance_3d_to_verified_cabinet_mesh_m=surface(world),distance_to_interiorgs_3d_obb_m=gap,official_horizontal_waypoint_m=wp,outside_1m_disk_m=max(0.,wp-1))
    def compare(v):
        boxes=[q['box'] for q in v];outcome=dict(present=sum(b is not None for b in boxes))
        if outcome['present']<2:return outcome
        b=np.clip(boxes,[0,0,0,0],[320,240,320,240]);area=np.prod(b[:,2:]-b[:,:2],axis=1);inter=np.maximum(0,np.minimum(b[0,2:],b[1,2:])-np.maximum(b[0,:2],b[1,:2])).prod();return dict(present=2,area_fraction=(area/(320*240)).tolist(),inner_depth_region_fraction=(area/(4*320*240)).tolist(),iou=float(inter/(area.sum()-inter)),center_motion_px=float(np.linalg.norm(b[0,:2]+b[0,2:]-b[1,:2]-b[1,2:])/2))
    cals=json.loads((out/'calibration.json').read_text())['rows'];pairs=[dict(group=r['group'],**compare(r['views'])) for r in cals if r['condition']=='near_correct'];wrong=[]
    for rec in records:
        if rec['model']!='dino':continue
        dd=Path(rec['last_decision_rgb']).parent;aa=read(dd/'actions.jsonl');wrong.append(dict(visible=rec['visible'],requested=rec['requested'],**compare([s['dino'] for s in aa[-2:]])))
    measure=dict(cabinet_prim=prim_name,mesh_triangles=len(tri),cabinet=poses,body_origin_reference=True,source_positives_1s=pairs,closed_loop_last_two_0p5s=wrong,method='clipped DINO box area / image area, estimated central-half depth crop uses a quarter area before rounding; IoU compares final two boxes. Training pairs stationary and separated by 1s, closed loop consecutive 0.5s. Mesh distance is nearest point over exact transformed triangles.');(out/'geometry_diagnostics.json').write_text(json.dumps(measure,indent=2)+'\n');print(json.dumps(measure))
