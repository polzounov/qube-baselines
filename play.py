#!/usr/bin/env python3

import numpy as np
import gym

from gym_brt.envs import QubeBeginDownEnv, QubeBeginUprightEnv

from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.cmd_util import arg_parser
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import set_global_seeds
from stable_baselines import bench, logger

from stable_baselines.ppo2 import PPO2


def main():
    envs = {"down": QubeBeginDownEnv, "up": QubeBeginUprightEnv}

    # Parse command line args
    parser = arg_parser()
    parser.add_argument("-e", "--env", default="down", choices=list(envs.keys()))
    parser.add_argument("-hw", "--use-hardware", action="store_true")
    parser.add_argument("-l", "--load", type=str, default=None)
    args = parser.parse_args()

    def make_env():
        env_out = envs[args.env](use_simulator=not args.use_hardware, frequency=250)
        return env_out

    env = DummyVecEnv([make_env])

    policy = MlpPolicy
    model = PPO2(policy=policy, env=env)
    model.load_parameters(args.load)

    print("Running trained model")
    obs = np.zeros((env.num_envs,) + env.observation_space.shape)
    obs[:] = env.reset()
    while True:
        actions = model.step(obs)[0]
        obs[:], reward, done, _ = env.step(actions)
        if not args.use_hardware:
            env.render()
        if done:
            print("done")
            obs[:] = env.reset()

    env.close()


if __name__ == "__main__":
    main()
