#!/usr/bin/env python3
"""Numerical and integration regressions without optional Gym/Gazebo."""
import math
from .core import (Phys,advance,DistanceField,signed_clearance,NavigationCore,
                   RewardConfig,state_vector,FORWARD,NEUTRAL,delay_remaining_at)


def main():
    box=[dict(x=0.,y=0.,size=[1.,1.,3.])]
    assert signed_clearance(0,0,box,.3)<0
    assert signed_clearance(1.5,0,box,.3)>.6
    f=DistanceField(box,(3,0),resolution=.25)
    assert f.distance(-3,0)>f.distance(-1,0)
    p=Phys(x=-3,y=0)
    for tick in range(5):advance(p,FORWARD,dispatch=tick==0)
    assert 0<p.vx<0.2 and p.x>-3
    for tick in range(100):advance(p,NEUTRAL,dispatch=tick==0)
    assert p.vx<.01
    c=NavigationCore(seed=0)
    assert len(state_vector(c.p,c.goal,c.obstacles))==13
    base=c.field.distance(c.p.x,c.p.y)
    initial=c.p.x
    for _ in range(5):
        obs,r,terminated,truncated,info=c.step(FORWARD)
        assert len(obs)==13 and math.isfinite(r)
        assert all(math.isfinite(v) for v in info['reward_terms'].values())
        assert math.isclose(r,sum(info['reward_terms'].values()),abs_tol=1e-9)
        assert not terminated and not truncated
    assert c.p.x>initial
    assert c.field.distance(c.p.x,c.p.y)<base
    # Exact potential telescoping for fixed discount over two nonterminal steps.
    c2=NavigationCore(seed=0)
    phi0=-c2.cfg.potential_scale*c2.field.distance(c2.p.x,c2.p.y)
    for j in range(2):
        _,_,end,trunc,info=c2.step(NEUTRAL)
        assert not(end or trunc)
        if j==0:first=info['reward_terms']['progress']
        else:second=info['reward_terms']['progress']
    phif=-c2.cfg.potential_scale*c2.field.distance(c2.p.x,c2.p.y)
    assert abs(first+c2.cfg.gamma*second-(c2.cfg.gamma**2*phif-phi0))<1e-7
    # Success requires sustained low-speed presence, not merely passing a radius.
    assert not c2.step(FORWARD)[4]['success']
    # N1: at decision time the env has counted the dispatch delay down to 0 (0.15 < 0.2),
    # so obs[-1] is 0; the Gazebo collector must derive the same number from sim time.
    c3=NavigationCore(seed=0)
    obs=c3.obs()
    assert obs[-1]==0.0
    for _ in range(3):
        obs,_,_,_,_=c3.step(FORWARD)
        assert obs[-1]==0.0
    assert delay_remaining_at(0.0,0.2,0.15)==0.0           # a full hold later: elapsed
    assert math.isclose(delay_remaining_at(0.0,0.05,0.15),0.10)  # mid-delay observation
    assert delay_remaining_at(None,5.0,0.15)==0.0          # nothing dispatched yet
    print('rl_teacher core PASS')

if __name__=='__main__':main()
