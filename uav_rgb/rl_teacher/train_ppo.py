#!/usr/bin/env python3
"""Offline RL teacher training; optional requirements; never claims Gazebo validation."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=100_000)
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--out',required=True)
    p.add_argument('--n-envs',type=int,default=4)
    p.add_argument('--reward-config',default=str(Path(__file__).with_name('reward_v1.json')))
    args=p.parse_args()
    if args.steps<1 or args.n_envs<1:
        p.error('--steps and --n-envs must be >0')
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as e:
        p.error(f'缺少可选依赖 gymnasium/stable-baselines3：{e}；'
                '请先 python3 -m pip install gymnasium stable-baselines3 numpy')
    from .gym_env import NavigationEnv
    from .core import RewardConfig
    from dataclasses import fields
    reward_dict=json.loads(Path(args.reward_config).read_text())
    expected={f.name for f in fields(RewardConfig)}
    if set(reward_dict)!=expected:
        p.error(f'reward config keys mismatch: missing={expected-set(reward_dict)}, extra={set(reward_dict)-expected}')
    reward_cfg=RewardConfig(**reward_dict)
    if not(0<reward_cfg.gamma<=1) or reward_cfg.max_steps<1:
        p.error('invalid gamma/max_steps')
    import sys
    tool_path=str(Path(__file__).resolve().parents[1]/'sim'/'tools')
    if tool_path not in sys.path: sys.path.insert(0,tool_path)
    import asset_campus
    seeds=[s for s in range(800) if asset_campus.split_key(asset_campus.map_key(s))=='train']
    # map variants are intentionally grouped; only split=train used for optimization.
    check_env(NavigationEnv(seeds=seeds[:64],reward_config=reward_cfg))
    env=DummyVecEnv([lambda i=i:NavigationEnv(seeds=seeds[i::args.n_envs],
                                            reward_config=reward_cfg)
                     for i in range(args.n_envs)])
    model=PPO('MlpPolicy',env,seed=args.seed,verbose=1,
              n_steps=256,batch_size=256,learning_rate=3e-4,gamma=reward_cfg.gamma,
              tensorboard_log=None)
    model.learn(total_timesteps=args.steps)
    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    model.save(str(out))
    out.with_suffix('.meta.json').write_text(json.dumps({
        'schema':'uavrgb.rlteacher.v1','observation_dim':13,'action_dim':4,
        'train_split':'campus_assets train only','steps':args.steps,'seed':args.seed,
        'physics':'cf_stab_approx_v1_dt0.04_latency0.15_k1_tilt2',
        'reward':reward_dict,
        'validation':'NOT run; model must be evaluated separately in heldout maps and Gazebo',
    },indent=2)+'\n')
    print('model saved:',out)

if __name__=='__main__':
    # python -m rl_teacher.train_ppo from uav_rgb repo root
    main()
