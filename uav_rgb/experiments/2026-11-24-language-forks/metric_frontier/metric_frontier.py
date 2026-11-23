"""RGB depth, local visual odometry and frontier exploration in an arbitrary initial frame."""
import atexit
import collections
import math
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

HOME_DIR = Path(__file__).resolve().parents[3]
FX = 277.1281292110204
K = np.array([[FX, 0, 160], [0, FX, 120], [0, 0, 1]], dtype=np.float64)
WORKER = None


def get_depth(rgb):
    global WORKER
    if WORKER is None:
        python = HOME_DIR.parent/'baselines/venv-da2/bin/python'
        WORKER = subprocess.Popen([str(python),'-u',str(HOME_DIR/'metric_worker.py'),'serve'],
                                  stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        atexit.register(WORKER.terminate)
        if WORKER.stdout.read(4) != b'RDY!':
            raise RuntimeError('Metric depth initialization failed')
    WORKER.stdin.write(np.ascontiguousarray(rgb).tobytes())
    WORKER.stdin.flush()
    response = WORKER.stdout.read(240*320*4)
    if len(response) != 240*320*4:
        raise RuntimeError('Metric depth response truncated')
    return np.frombuffer(response, '<f4').reshape(240,320)


class Frontier:
    def __init__(self):
        self.position=np.zeros(2)
        self.yaw=0.
        self.grid=np.zeros((161,161),np.int16)
        self.visits=np.zeros_like(self.grid)
        self.previous_cell=None
        self.previous=None
        self.previous_claim=False
        self.remaining=0
        self.goal=None

    def update_pose(self,gray,depth,action):
        tracked=np.empty((0,2))
        used=False
        forward=0.15 if action==1 else 0.
        left=0.
        turn=math.radians({2:17,3:-17}.get(action,0))
        if self.previous is not None:
            old,old_depth=self.previous
            p=cv2.goodFeaturesToTrack(old,160,.015,6)
            if p is not None and len(p)>=10:
                q,ok,_=cv2.calcOpticalFlowPyrLK(old,gray,p,None,winSize=(23,23),maxLevel=3)
                mask=ok.ravel().astype(bool)
                p,q=p.reshape(-1,2)[mask],q.reshape(-1,2)[mask]
                ix=np.clip(np.rint(p).astype(int),[0,0],[319,239])
                z=old_depth[ix[:,1],ix[:,0]]
                good=np.isfinite(z)&(z>.2)&(z<6)
                xyz=np.column_stack(((p[:,0]-160)*z/FX,(p[:,1]-120)*z/FX,z))[good]
                q=q[good]
                if len(q)>=10:
                    success,r,t,inl=cv2.solvePnPRansac(xyz.astype(np.float64),q.astype(np.float64),K,None,
                        iterationsCount=80,reprojectionError=3.,flags=cv2.SOLVEPNP_EPNP)
                    if success and inl is not None and len(inl)>=10:
                        rot=cv2.Rodrigues(r)[0].T
                        motion=(-rot@t).ravel()
                        delta=math.atan2(-rot[0,2],rot[2,2])
                        if np.linalg.norm(motion)<.45 and abs(delta)<math.radians(40):
                            forward,left,turn=motion[2],-motion[0],delta
                            tracked=q[inl.ravel()]
                            used=True
        self.position += [forward*math.cos(self.yaw)-left*math.sin(self.yaw),
                          forward*math.sin(self.yaw)+left*math.cos(self.yaw)]
        self.yaw+=turn
        self.previous=(gray,depth)
        return used,tracked

    def cell(self,p):
        x,y=np.clip(np.rint(p/.15).astype(int)+80,1,159)
        return int(x),int(y)

    def update_map(self,depth):
        cx,cy=self.cell(self.position)
        if self.previous_cell!=(cx,cy):
            self.visits[cy,cx]+=1
            self.previous_cell=cx,cy
        free=np.zeros(self.grid.shape,np.uint8)
        hits=np.zeros_like(free)
        cyaw,syaw=math.cos(self.yaw),math.sin(self.yaw)
        for u in range(2,320,4):
            v=np.arange(1,240,4)
            z=depth[v,u]
            vertical=(v-120)*z/FX
            candidates=z[np.isfinite(z)&(z>.1)&(abs(vertical)<.19)]
            if not len(candidates):continue
            wall=float(np.percentile(candidates,15))
            dist=min(3.,wall)
            dx=dist;dy=-(u-160)*dist/FX
            end=self.cell(self.position+[cyaw*dx-syaw*dy,syaw*dx+cyaw*dy])
            cv2.line(free,(cx,cy),end,1)
            if wall<=3:
                hits[end[1],end[0]]=1
        self.grid=np.clip(self.grid-free.astype(np.int16)+4*hits.astype(np.int16),-12,12)
        self.grid[cy,cx]=-12

    def navigation(self):
        start=self.cell(self.position)
        solid=cv2.dilate((self.grid>=2).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))).astype(bool)
        free=(self.grid<0)&~solid
        startx,starty=start
        free[starty,startx]=True
        unknown=cv2.dilate((self.grid==0).astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        queue=collections.deque([start]); parents={start:None}; costs={start:0}
        targets=[]
        for_pop=queue.popleft
        while queue:
            x,y=for_pop(); cost=costs[x,y]
            if cost>=5 and unknown[y,x]:
                rel=math.atan2(y-starty,x-startx)-self.yaw
                targets.append((cost+.9*abs(math.remainder(rel,2*math.pi))+4*int(self.visits[y,x]),(x,y)))
            for dx,dy in ((1,0),(0,1),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                a,b=x+dx,y+dy
                if not (0<a<160 and 0<b<160) or (a,b) in parents or not free[b,a]:continue
                if dx and dy and (not free[y,a] or not free[b,x]):continue
                parents[a,b]=(x,y);costs[a,b]=cost+1;queue.append((a,b))
        if not targets:return 2
        candidates={q for _,q in targets}
        if self.goal not in candidates:self.goal=min(targets)[1]
        path=[self.goal]
        while parents[path[-1]] is not None:path.append(parents[path[-1]])
        waypoint=path[max(0,len(path)-4)]
        direction=math.atan2(waypoint[1]-starty,waypoint[0]-startx)
        error=math.remainder(direction-self.yaw,2*math.pi)
        return 1 if abs(error)<math.radians(14) else 2 if error>0 else 3

    def decide(self,rgb,previous_action,r2,grounding,tracker,confirmed):
        from .grounder import instruction_target_matches
        start=time.perf_counter()
        depth=get_depth(rgb)
        ms=(time.perf_counter()-start)*1000
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        used,tracked=self.update_pose(gray,depth,previous_action)
        self.update_map(depth)
        step=tracker.step_idx-1
        fresh=bool(tracker.trace and tracker.trace[-1]['step']==step)
        if fresh:
            vb=tracker.trace[-1]['raw_bearing_bin']
            claim=grounding.goal_mode=='final' and vb in (0,1,7) and instruction_target_matches(tracker.instruction,grounding.reason)
            lo,hi={0:(106,214),1:(0,107),7:(213,320)}.get(vb,(0,0))
            support=[(x,y) for x,y in tracked if lo<=x<hi and 0<=y<240 and .2<depth[int(y),int(x)]<4.]
            if self.previous_claim and claim and len(support)>=4:self.remaining=3
            self.previous_claim=bool(claim)
        follow=self.remaining>0
        self.remaining=max(0,self.remaining-1)
        act=r2 if follow else self.navigation()
        # Check the forward swept corridor using the same dense estimated depth at flight height.
        vv,uu=np.mgrid[2:240:4,2:320:4];zz=depth[vv,uu]
        band=(abs((uu-160)*zz/FX)<.22)&(abs((vv-120)*zz/FX)<.19)&(zz>.1)
        clearance=float(np.percentile(zz[band],8)) if band.any() else 0.
        if act==1 and clearance<.9:
            act=2 if np.median(depth[90:150,:100])>=np.median(depth[90:150,-100:]) else 3
        return act,bool(follow and confirmed),dict(metric_follow=follow,metric_vo=used,metric_covered=int((self.visits>0).sum()),metric_clear=clearance,metric_depth_ms=ms,metric_total_ms=1000*(time.perf_counter()-start),metric_r2=r2)
