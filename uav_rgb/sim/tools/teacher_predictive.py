"""Opt-in model-predictive discrete T1 teacher for cf_stab approximation.

NOT a global optimal-control or safe-flight guarantee. Uses A* distance field to
compare four short-horizon simulated actions; replans each observation.
"""
import math
from pathlib import Path
import sys

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path: sys.path.insert(0,ROOT)
from rl_teacher.core import (Phys, advance, DistanceField, signed_clearance, ACTION_DT)

VERSION='uavrgb.predictive_teacher.v1'


class PredictiveTeacher:
    def __init__(self, obstacles, waypoint, *, max_speed=.65):
        self.obstacles=obstacles
        self.goal=(waypoint['x'],waypoint['y'])
        self.field=DistanceField(obstacles,self.goal)
        self.max_speed=max_speed

    def act(self, x, y, yaw_deg, vx, vy, active=0, latency=.15):
        p0=Phys(x=x,y=y,yaw=math.radians(yaw_deg),vx=vx,vy=vy,action=active)
        start=self.field.distance(x,y)
        goal_d=math.hypot(x-self.goal[0],y-self.goal[1])
        speed=math.hypot(vx,vy)
        # Neutral is a passive coast, NOT an active brake. If drifting fast near
        # goal the teacher should have slowed earlier; this cannot be fixed by reward.
        scores=[]
        for action in (0,1,2,3):
            s=Phys(**vars(p0))
            min_gap=signed_clearance(s.x,s.y,self.obstacles)
            crashed=min_gap<=0
            for k in range(30):  # 1.2-second constant-action horizon
                advance(s,action,dt=.04,latency=latency,dispatch=(k==0))
                gap=signed_clearance(s.x,s.y,self.obstacles)
                min_gap=min(min_gap,gap)
                if gap<=0:
                    crashed=True
                    break
            end=self.field.distance(s.x,s.y)
            # heading-to-geodesic-gradient when progress alone cannot distinguish turns
            step=.25
            gx=self.field.distance(s.x+step,s.y)-self.field.distance(s.x-step,s.y)
            gy=self.field.distance(s.x,s.y+step)-self.field.distance(s.x,s.y-step)
            desired=math.atan2(-gy,-gx)
            heading=math.cos(desired-s.yaw)
            end_speed=math.hypot(s.vx,s.vy)
            risk=max(0.,.3-min_gap)
            score=1.8*(start-end)+.18*heading-10.0*risk*risk-100.0*crashed
            if goal_d<1.2:
                score-=1.8*end_speed
                score-=1.5*max(0,end_speed-self.max_speed)**2
            # Weak preference for smaller movement / fewer gratuitous switches.
            score-=.01*(action!=active)
            scores.append((score,-action,action))
        return max(scores)[2]
