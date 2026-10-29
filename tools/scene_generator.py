#!/usr/bin/env python3
"""Deterministic 2.5D scene/task candidate generator for later Gazebo use.

NOT a drone simulator, not an episode dataset, not a validated VLN benchmark.
Produces visible semantic landmarks + obstacle collision meshes + oracle task
manifest (train/evaluation only). Students must not read the oracle manifest.
"""
import argparse
import hashlib
import heapq
import json
import math
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
COLORS={'blue':(.17,.36,.86),'yellow':(.95,.71,.18),'red':(.82,.23,.24),'green':(.2,.68,.38),'violet':(.53,.28,.74)}


def route(start,goal,obstacles,step=.40, drone_radius=.24):
    """A* on a finite inflated occupancy grid, no privileged planner in student."""
    def grid(xy):return (round(xy[0]/step),round(xy[1]/step))
    def free(idx):
        x,y=(idx[0]*step,idx[1]*step)
        if abs(x)>10.3 or abs(y)>6.3:return False
        return all((x-a)**2+(y-b)**2 > (r+drone_radius+.10)**2 for a,b,r in obstacles)
    s=grid(start); t=grid(goal)
    if not free(s) or not free(t):return []
    pq=[(0.,0.,s)];parents={s:None};dists={s:0.}
    seen=set();seq=0
    while pq:
        _,_,u=heapq.heappop(pq)
        if u in seen:continue
        if u==t:break
        seen.add(u)
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            v=u[0]+dx,u[1]+dy
            if not free(v):continue
            if dx and dy and not (free((u[0]+dx,u[1])) and free((u[0],u[1]+dy))):continue
            ng=dists[u]+(1.414213562 if dx and dy else 1)
            if ng<dists.get(v,float('inf')):
                parents[v]=u;dists[v]=ng;seq+=1
                heapq.heappush(pq,(ng+math.hypot(v[0]-t[0],v[1]-t[1]),seq,v))
    if t not in parents:return []
    q=[];v=t
    while v is not None:q.append([round(v[0]*step,3),round(v[1]*step,3),1.0]);v=parents[v]
    return q[::-1]


def sdf_solid(name,x,y,z,r,h,rgb,collision=True):
    rgba=' '.join(f'{v:.3f}' for v in rgb)+' 1'
    collision_tag=(f'<collision name="collision"><geometry><cylinder><radius>{r:.3f}</radius>'
                    f'<length>{h:.3f}</length></cylinder></geometry></collision>') if collision else ''
    return (f'<model name="{name}"><static>true</static><pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose>'
            f'<link name="body"><visual name="visual"><geometry><cylinder><radius>{r:.3f}</radius>'
            f'<length>{h:.3f}</length></cylinder></geometry><material><diffuse>{rgba}</diffuse>'
            f'<ambient>{rgba}</ambient></material></visual>{collision_tag}</link></model>')


def create(seed:int,difficulty:str):
    rng=random.Random(seed)
    for attempt in range(120):
        start=(-8.,rng.uniform(-2,2));goal=(rng.uniform(6.5,8.5),rng.uniform(-3.2,3.2))
        obstacles=[]
        if difficulty in ('S1','S2'):
            # The straight path is deliberately blocked by a circular column.
            k=.5
            y=start[1]+(goal[1]-start[1])*((k-start[0])/(goal[0]-start[0]))
            obstacles.append((k,y,0.78))
        n={'S0':0,'S1':9,'S2':22}[difficulty]
        for _ in range(n):
            x=rng.uniform(-5.8,6.0);y=rng.uniform(-5.0,5.0);r=rng.uniform(.35,.85)
            if (x-start[0])**2+(y-start[1])**2<(r+1.5)**2:continue
            if (x-goal[0])**2+(y-goal[1])**2<(r+1.5)**2:continue
            if all(math.hypot(x-a,y-b)>r+rr+.35 for a,b,rr in obstacles):obstacles.append((x,y,r))
        path=route(start,goal,obstacles)
        if path:
            landmarks=[]
            palette=['blue','yellow','red','green','violet']
            rng.shuffle(palette)
            nums=1 if difficulty=='S0' else 2 if difficulty=='S1' else 3
            for i in range(nums):
                p=path[min(len(path)-1,int((i+1)*len(path)/(nums+1)))]
                loc=None
                for ox,oy in ((0.85,0),(0,-0.85),(-0.85,0),(0,0.85),(1.15,0),(0,1.15)):
                    x=round(p[0]+ox,3);y=round(p[1]+oy,3)
                    if abs(x)>9.8 or abs(y)>5.7:continue
                    if any(math.hypot(x-a,y-b)<r+.55 for a,b,r in obstacles):continue
                    if min(math.hypot(x-a,y-b) for a,b,_ in path)<.55:continue
                    if any(math.hypot(x-q['x'],y-q['y'])<1.0 for q in landmarks):continue
                    loc=(x,y);break
                if loc is None:
                    break
                landmarks.append({'id':f'{palette[i]}_pillar_{i+1}','color':palette[i],
                                  'x':loc[0],'y':loc[1],'radius':.18,'height':1.65})
            if len(landmarks)!=nums:continue
            # Distinct distractors; not every colored object should be target.
            for j in range(nums):
                color=palette[j]
                dx=rng.uniform(-5.5,5.5);dy=rng.choice([-1,1])*rng.uniform(3.5,5.0)
                if min(math.hypot(dx-l['x'],dy-l['y']) for l in landmarks)>1.8 and min(math.hypot(dx-a,dy-b) for a,b,_ in path)>.65:
                    landmarks.append({'id':f'distractor_{color}_{j}','color':color,'x':dx,'y':dy,'radius':.18,'height':1.65})
            instruction=('From the start, pass near '+', then pass near '.join(
              [f"the {palette[i]} pillar" for i in range(nums)])+'. Stop at the end of the route.')
            return {'seed':seed,'difficulty':difficulty,'attempts':attempt+1,'start':[start[0],start[1],1.0],
                    'goal':[goal[0],goal[1],1.0],'obstacles':obstacles,'landmarks':landmarks,
                    'oracle_path':path,'instruction':instruction,
                    'limitations':['Not a Gazebo flying episode','Terminal location not visually grounded by itself',
                    'Language relevance must be tested against blind / shuffled-instruction policies']}
    raise ValueError('SCENE_UNSOLVABLE: 120 finite retries exhausted')


def generate(seed,difficulty,out):
    obj=create(seed,difficulty)
    things=['<?xml version="1.0" encoding="UTF-8"?>','<sdf version="1.9"><world name="navlab_scene">',
           '<gravity>0 0 -9.81</gravity>',
           '<include><uri>model://ground_plane</uri></include>',
           '<include><uri>model://sun</uri></include>']
    for i,(x,y,r) in enumerate(obj['obstacles']):
        things.append(sdf_solid(f'obstacle_{i}',x,y,1.4,r,2.8,(.39,.41,.42)))
    for lm in obj['landmarks']:
        things.append(sdf_solid(lm['id'],lm['x'],lm['y'],.825,lm['radius'],lm['height'],COLORS[lm['color']]))
    things.append('</world></sdf>')
    data=('\n'.join(things)+'\n').encode('utf-8')
    out.mkdir(parents=True,exist_ok=True)
    stem=f'scene_{difficulty}_{seed}'
    (out/f'{stem}.sdf').write_bytes(data)
    obj.update({'sdf':stem+'.sdf','sdf_sha256':hashlib.sha256(data).hexdigest(),
                'schema':'navlab.scenes.v1','status':'static_world_only','camera':False,
                'physics_quadrotor':False,'closed_loop_episode':False})
    (out/f'{stem}.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return obj


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--difficulty',choices=['S0','S1','S2'],default='S1')
    ap.add_argument('--output',type=Path,default=ROOT/'artifacts/scenes')
    a=ap.parse_args()
    try:
        obj=generate(a.seed,a.difficulty,a.output)
        print(f'[SCENE_GENERATED][STATIC_ONLY] {obj["sdf"]} sha256={obj["sdf_sha256"]} '
              f'path_nodes={len(obj["oracle_path"])} obstacles={len(obj["obstacles"])}')
        print('[INFO] A* is for training/eval truth only; no drone camera/dynamics yet.')
    except (ValueError,OSError) as exc:
        print('[ERR] '+str(exc),file=sys.stderr);return 4
    return 0

if __name__=='__main__':sys.exit(main())
