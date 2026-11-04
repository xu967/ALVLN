"""Fast deterministic 2.5D small-UAV dynamics, distance field, reward, and RL observation.

This is a LIGHTWEIGHT physics approximation matched to cf_stab command semantics,
NOT PX4/Crazyflie rotor physics; validate against Gazebo before policy adoption.
No PyTorch, Gymnasium or numpy required for this module.
"""
from dataclasses import dataclass
import copy
import heapq
import math
import random

FORWARD, LEFT, RIGHT, NEUTRAL = 1, 2, 3, 0
DT = 0.04
ACTION_DT = 0.2
G = 9.81


def signed_clearance(x, y, obstacles, radius=0.356):
    """Signed minimum distance from inflated AABB; <=0 means center collision."""
    best = math.inf
    for p in obstacles:
        sx, sy = p['size'][:2]
        dx = abs(x - p['x']) - sx/2
        dy = abs(y - p['y']) - sy/2
        outside = math.hypot(max(dx, 0.0), max(dy, 0.0))
        inside = min(max(dx, dy), 0.0)
        best = min(best, outside + inside - radius)
    return best


class DistanceField:
    """Goal-rooted Dijkstra grid; blocked cells inflated by vehicle radius.

    Bilinear sampling is used only where all four neighboring cells are valid;
    otherwise nearest valid cell is selected. Map is re-built on each reset,
    NEVER computed from student-accessible measurements.
    """
    def __init__(self, obstacles, goal, resolution=0.25, radius=0.38):
        xs = [-1.5, goal[0]+2.0] + [p['x'] for p in obstacles]
        ys = [-5.5, 5.5, goal[1]] + [p['y'] for p in obstacles]
        self.r = resolution
        self.x0 = min(xs)-2.0
        self.y0 = min(ys)-2.0
        self.nx = int(math.ceil((max(xs)+2-self.x0)/resolution))+1
        self.ny = int(math.ceil((max(ys)+2-self.y0)/resolution))+1
        self.goal = goal
        self.blocked = set()
        for iy in range(self.ny):
            y = self.y0+iy*self.r
            for ix in range(self.nx):
                x = self.x0+ix*self.r
                if signed_clearance(x,y,obstacles,radius) <= 0:
                    self.blocked.add((ix,iy))
        gx,gy = self._nearest(goal)
        if (gx,gy) in self.blocked:
            raise ValueError('goal blocked under radius inflation')
        self.dist = {(gx,gy):0.0}
        q = [(0.0,gx,gy)]
        dirs = [(1,0,1),(-1,0,1),(0,1,1),(0,-1,1),
                (1,1,math.sqrt(2)),(1,-1,math.sqrt(2)),(-1,1,math.sqrt(2)),(-1,-1,math.sqrt(2))]
        while q:
            d,i,j = heapq.heappop(q)
            if d > self.dist[(i,j)]+1e-9:
                continue
            for di,dj,k in dirs:
                ni,nj=i+di,j+dj
                if not (0<=ni<self.nx and 0<=nj<self.ny) or (ni,nj) in self.blocked:
                    continue
                if di and dj and ((i+di,j) in self.blocked or (i,j+dj) in self.blocked):
                    continue
                nd = d+self.r*k
                if nd < self.dist.get((ni,nj), math.inf):
                    self.dist[ni,nj] = nd
                    heapq.heappush(q,(nd,ni,nj))

    def _nearest(self, pt):
        return int(round((pt[0]-self.x0)/self.r)),int(round((pt[1]-self.y0)/self.r))

    def distance(self,x,y):
        fx,fy=(x-self.x0)/self.r, (y-self.y0)/self.r
        ix,iy=math.floor(fx), math.floor(fy)
        tx,ty=fx-ix,fy-iy
        values = [self.dist.get((ix+di,iy+dj), math.inf)
                  for di,dj in ((0,0),(1,0),(0,1),(1,1))]
        if all(math.isfinite(v) for v in values):
            a,b,c,d=values
            return (1-tx)*(1-ty)*a+tx*(1-ty)*b+(1-tx)*ty*c+tx*ty*d
        nx,ny=self._nearest((x,y))
        if (nx,ny) in self.dist:
            return self.dist[nx,ny]
        return 60.0  # unreachable; do not pretend a short Euclidean shortcut


@dataclass
class Phys:
    x: float = -.5
    y: float = 0.0
    yaw: float = 0.0     # radians
    vx: float = 0.0      # world-frame m/s
    vy: float = 0.0
    action: int = NEUTRAL
    last_action: int = NEUTRAL
    delay_remaining: float = 0.0


def advance(state, command, dt=DT, latency=0.15, tilt_deg=2.0, drag_k=1.0,
            yaw_rate=.6, dispatch=True):
    """One physics tick, first-order drag and fixed-angle thrust.

    Caller holds a dispatched command; delayed actions remain active until their
    latency has elapsed. For speed, a single pending command is modelled; this is
    accurate when latency < ACTION_DT but not for multi-command queued delays.
    """
    if dispatch:
        state.last_action = command
        state.delay_remaining = latency
    if state.delay_remaining > 0:
        state.delay_remaining = max(0.0, state.delay_remaining-dt)
        if state.delay_remaining <= 0:
            state.action = state.last_action
    else:
        state.action = command
    a = G*math.tan(math.radians(tilt_deg)) if state.action == FORWARD else 0.0
    # Integrate yaw at midpoint: limits heading discretization drift vs Euler.
    wz = (yaw_rate if state.action==LEFT else -yaw_rate if state.action==RIGHT else 0.0)
    yaw_mid = state.yaw + .5*wz*dt
    ax,ay=a*math.cos(yaw_mid),a*math.sin(yaw_mid)
    e=math.exp(-drag_k*dt)
    lim_x,lim_y=ax/drag_k,ay/drag_k
    dx=lim_x*dt+(state.vx-lim_x)*(1-e)/drag_k
    dy=lim_y*dt+(state.vy-lim_y)*(1-e)/drag_k
    state.x+=dx
    state.y+=dy
    state.vx=lim_x+(state.vx-lim_x)*e
    state.vy=lim_y+(state.vy-lim_y)*e
    state.yaw=(state.yaw+wz*dt+math.pi)%(2*math.pi)-math.pi


def state_vector(state, waypoint, obstacles, geodesic=None):
    """13 privileged numerical features. RL and Gazebo teacher must share schema.

    dx,dy goal (body); vx,vy (body); sin/cos yaw; four directional
    clearance probes; current goal distance; last action; pending delay.
    """
    c,s=math.cos(state.yaw),math.sin(state.yaw)
    dx,dy=waypoint[0]-state.x,waypoint[1]-state.y
    gx,gy=c*dx+s*dy,-s*dx+c*dy
    ux,uy=c*state.vx+s*state.vy,-s*state.vx+c*state.vy
    probes=[]
    for phi in (0,math.pi/2,math.pi,-math.pi/2):
        theta=state.yaw+phi
        step=.15
        val=3.0
        for k in range(1,21):
            d=k*step
            if signed_clearance(state.x+d*math.cos(theta),
                                state.y+d*math.sin(theta), obstacles) <= 0:
                val=d
                break
        probes.append(val/3.0)
    return [max(-2,min(2,gx/8)),max(-2,min(2,gy/8)),
            max(-2,min(2,ux)),max(-2,min(2,uy)),s,c,
            *probes,min(2,math.hypot(dx,dy)/8),state.action/3,
            max(0,min(1,state.delay_remaining/0.5))]


@dataclass
class RewardConfig:
    gamma: float = 0.99
    potential_scale: float = 1.0
    success: float = 20.0
    collision: float = -20.0
    timeout: float = -8.0
    time_per_second: float = 0.02
    risk: float = 0.08
    action_change: float = 0.01
    safety_distance: float = 0.35
    goal_radius: float = 0.5
    goal_speed: float = .12
    goal_dwell_s: float = .6
    max_steps: int = 250


class NavigationCore:
    def __init__(self, seed=0, reward=None):
        self.seed = seed
        self.cfg = reward or RewardConfig()
        self.reset(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.seed=seed
        # Imports are restricted to on-disk existing project source (never user data).
        from pathlib import Path
        import sys
        root = Path(__file__).resolve().parents[1]/'sim'/'tools'
        if str(root) not in sys.path:
            sys.path.insert(0,str(root))
        import asset_campus
        fam, obstacles, color, relation, wp, name = asset_campus.layout(self.seed)
        self.obstacles=obstacles
        self.goal=(wp['x'],wp['y'])
        self.map_id=asset_campus.map_key(self.seed)
        self.field=DistanceField(obstacles,self.goal)
        self.p=Phys()
        self.steps=0
        self.dwell=0.0
        return self.obs()

    def obs(self):
        return state_vector(self.p,self.goal,self.obstacles)

    def step(self, action):
        if int(action) not in range(4):
            raise ValueError(f'invalid discrete action: {action}')
        action=int(action)
        before=self.field.distance(self.p.x,self.p.y)
        prev=self.p.last_action
        min_clear=signed_clearance(self.p.x,self.p.y,self.obstacles)
        collide=min_clear <= 0
        out_of_bounds=False
        # 5 x 0.04s physics ticks per decision (20 Hz/5 = 5Hz policy).
        for tick in range(5):
            advance(self.p,action,dispatch=(tick==0))
            d=signed_clearance(self.p.x,self.p.y,self.obstacles)
            min_clear=min(min_clear,d)
            if d<=0:
                collide=True
                break
            if not(-2.0<=self.p.x<=12.0 and abs(self.p.y)<=6.0):
                out_of_bounds=True
                break
        self.steps+=1
        speed=math.hypot(self.p.vx,self.p.vy)
        dist=math.hypot(self.p.x-self.goal[0],self.p.y-self.goal[1])
        if not collide and dist<self.cfg.goal_radius and speed<self.cfg.goal_speed:
            self.dwell+=ACTION_DT
        else:
            self.dwell=0.0
        success=(not collide and self.dwell>=self.cfg.goal_dwell_s)
        timed_out=(self.steps>=self.cfg.max_steps and not collide and not success)
        terminated=success or collide or out_of_bounds
        truncated=timed_out
        # Potential-based shaping with terminal potential exactly zero.
        phi_before=-self.cfg.potential_scale*before
        phi_after=(0.0 if (terminated or truncated)
                   else -self.cfg.potential_scale*self.field.distance(self.p.x,self.p.y))
        reward_progress=self.cfg.gamma*phi_after-phi_before
        reward_time=-self.cfg.time_per_second*ACTION_DT
        # One-step safety warning is too late when inertia persists. Roll forward
        # another 0.8 s under the same command, using the SAME approximate dynamics.
        future=Phys(**vars(self.p))
        future_gap=min_clear
        for j in range(20):
            advance(future,action,dispatch=(j==0))
            future_gap=min(future_gap,signed_clearance(future.x,future.y,self.obstacles))
            if future_gap<=0:
                break
        reward_risk=-self.cfg.risk*(max(0,self.cfg.safety_distance-max(future_gap,0)))**2
        reward_action=-self.cfg.action_change*float(action!=prev)
        reward_task=(self.cfg.success if success else
                     self.cfg.collision if (collide or out_of_bounds) else
                     self.cfg.timeout if timed_out else 0.)
        reward=reward_task+reward_progress+reward_time+reward_risk+reward_action
        info={'success':success,'collision':collide,'out_of_bounds':out_of_bounds,
              'timeout':timed_out,'min_clearance':min_clear,
              'speed':speed,'goal_distance':dist,'forecast_min_clearance':future_gap,
              'map_id':self.map_id,
              'reward_terms':{'task':reward_task,'progress':reward_progress,'time':reward_time,
                              'risk':reward_risk,'action':reward_action},
              'physics_profile':'cf_stab_approx_v1_dt0.04_latency0.15_k1_tilt2'}
        return self.obs(), reward, terminated, truncated, info
