"""Lazy Gymnasium wrapper; RL extra is not required for core/project tests."""
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    raise SystemExit(f'rl_teacher.gym_env 需要可选依赖 gymnasium（{e}）；'
                     '请先 python3 -m pip install gymnasium stable-baselines3 numpy')
import numpy as np
from .core import NavigationCore

class NavigationEnv(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self, seeds=(0,8,16,24,32), fixed_seed=None, reward_config=None):
        super().__init__()
        self.seeds=list(seeds)
        self.fixed_seed=fixed_seed
        self.counter=0
        self.engine=NavigationCore(seed=fixed_seed if fixed_seed is not None else self.seeds[0],
                                   reward=reward_config)
        self.observation_space=spaces.Box(-3.,3.,shape=(13,),dtype=np.float32)
        self.action_space=spaces.Discrete(4)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        env_seed=self.fixed_seed if self.fixed_seed is not None else int(self.np_random.choice(self.seeds))
        obs=self.engine.reset(env_seed)
        self.counter+=1
        return np.asarray(obs,dtype=np.float32), {'map_id':self.engine.map_id}

    def step(self,action):
        obs,reward,terminated,truncated,info=self.engine.step(int(action))
        return np.asarray(obs,dtype=np.float32),float(reward),terminated,truncated,info
