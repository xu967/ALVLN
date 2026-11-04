#!/usr/bin/env python3
"""Independent heldout evaluation. Reports successes; does not re-train/checkpoint-select."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--split',choices=('train','val','test'),default='test')
    p.add_argument('--episodes',type=int,default=50)
    p.add_argument('--out',default='')
    a=p.parse_args()
    if a.episodes<1:p.error('--episodes must be positive')
    try:
        from stable_baselines3 import PPO
    except ImportError as e:
        p.error(f'缺少可选依赖 gymnasium/stable-baselines3：{e}；'
                '请先 python3 -m pip install gymnasium stable-baselines3 numpy')
    from .core import NavigationCore
    from pathlib import Path
    import sys
    tool_path=str(Path(__file__).resolve().parents[1]/'sim'/'tools')
    if tool_path not in sys.path: sys.path.insert(0,tool_path)
    import asset_campus
    candidates=[seed for seed in range(4000) if
                asset_campus.split_key(asset_campus.map_key(seed))==a.split and seed%8==0]
    model=PPO.load(a.checkpoint,device='cpu')
    from .core import RewardConfig
    meta_path=Path(a.checkpoint).with_suffix('.meta.json')
    cfg=RewardConfig(**json.loads(meta_path.read_text())['reward']) if meta_path.is_file() else RewardConfig()
    records=[]
    for seed in candidates[:a.episodes]:
        sim=NavigationCore(seed=seed,reward=cfg)
        done=False
        while not done:
            action,_=model.predict(__import__('numpy').asarray(sim.obs(),dtype='float32'),deterministic=True)
            _,_,terminal,truncated,info=sim.step(int(action))
            done=terminal or truncated
        records.append({'seed':seed,'success':info['success'],
                        'collision':info['collision'],'steps':sim.steps,
                        'end_speed_mps':info['speed'],'min_clearance_last_step':info['min_clearance']})
    results={'schema':'uavrgb.rl_eval.v1','split':a.split,'episodes':len(records),
             'success':sum(x['success'] for x in records),
             'collision':sum(x['collision'] for x in records),'records':records,
             'note':'lightweight dynamics ONLY, not actual Gazebo closed-loop evidence'}
    output=json.dumps(results,indent=2,ensure_ascii=False)
    if a.out:
        Path(a.out).parent.mkdir(parents=True,exist_ok=True)
        Path(a.out).write_text(output+'\n')
    print(output)

if __name__=='__main__':main()
