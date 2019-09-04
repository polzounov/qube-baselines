#!/usr/bin/env python3

import numpy as np
import gym
import os

from gym_brt.envs import (
    QubeSwingupEnv,
    QubeSwingupSparseEnv,
    QubeSwingupFollowEnv,
    QubeSwingupFollowSparseEnv,
    QubeBalanceEnv,
    QubeBalanceSparseEnv,
    QubeBalanceFollowEnv,
    QubeBalanceFollowSparseEnv,
    QubeDampenEnv,
    QubeDampenSparseEnv,
    QubeDampenFollowEnv,
    QubeDampenFollowSparseEnv,
    QubeRotorEnv,
    QubeRotorFollowEnv,
)

from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.cmd_util import arg_parser
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import set_global_seeds
from stable_baselines import bench, logger
from stable_baselines.ppo2 import PPO2


def main():
    envs = {
        "QubeSwingupEnv": QubeSwingupEnv,
        "QubeSwingupSparseEnv": QubeSwingupSparseEnv,
        "QubeSwingupFollowEnv": QubeSwingupFollowEnv,
        "QubeSwingupFollowSparseEnv": QubeSwingupFollowSparseEnv,
        "QubeBalanceEnv": QubeBalanceEnv,
        "QubeBalanceSparseEnv": QubeBalanceSparseEnv,
        "QubeBalanceFollowEnv": QubeBalanceFollowEnv,
        "QubeBalanceFollowSparseEnv": QubeBalanceFollowSparseEnv,
        "QubeDampenEnv": QubeDampenEnv,
        "QubeDampenSparseEnv": QubeDampenSparseEnv,
        "QubeDampenFollowEnv": QubeDampenFollowEnv,
        "QubeDampenFollowSparseEnv": QubeDampenFollowSparseEnv,
        "QubeRotorEnv": QubeRotorEnv,
        "QubeRotorFollowEnv": QubeRotorFollowEnv,
    }

    # Parse command line args
    parser = arg_parser()
    parser.add_argument("-e", "--env", choices=list(envs.keys()), required=True)
    parser.add_argument("-ms", "--max-timesteps", type=int, default=2048)
    parser.add_argument("-ne", "--num-episodes", type=int, default=10)
    parser.add_argument("-hw", "--use-hardware", action="store_true")
    parser.add_argument("-ld", "--logdir", type=str, default="logs")
    parser.add_argument("-l", "--loaddir", type=str, required=True)
    parser.add_argument("-r", "--render", action="store_true")
    args = parser.parse_args()

    loaddir = os.path.expanduser(args.loaddir)
    max_timesteps = int(args.max_timesteps)
    n_episodes = int(args.num_episodes)
    use_hardware = args.use_hardware

    def make_env():
        env_out = envs[args.env](
            use_simulator=not use_hardware, batch_size=max_timesteps, frequency=250
        )
        return env_out

    try:
        env = DummyVecEnv([make_env])
        policy = MlpPolicy

        # Get all .pkl files
        checkpoints = []
        for (dirpath, _, files) in os.walk(loaddir):
            for file in files:
                if file.split(".")[-1] == "pkl":
                    checkpoints.append(file.split(".")[0])
        if len(checkpoints) == 0:
            print("No checkpoints to run!")
        # Note: this sorting works for files named like 12345.pkl
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split(".")[0]))
        # Add the path to the filenames
        load_files = list(map(lambda x: dirpath + x + ".pkl", checkpoints))
        print("load_files in sorted order are:\n", load_files, "\n\n")

        reward_summaries = []
        # Run each model at each checkpoint
        for checkpoint, load_file in zip(checkpoints, load_files):
            model = PPO2(policy=policy, env=env)
            model.load_parameters(load_file)
            print("Loaded model: {}".format(load_file))

            # Beyond end of episode reward is always 0
            # Extra 1 timestep is a bug with env
            rewards = np.zeros((n_episodes, max_timesteps + 1))

            for episode in range(n_episodes):
                # print("\tRunning episode: {}".format(episode))
                obs = np.zeros((env.num_envs,) + env.observation_space.shape)
                obs[:] = env.reset()
                timestep = 0
                done = False

                while not done:
                    actions = model.step(obs)[0]
                    obs[:], reward, done, _ = env.step(actions)
                    rewards[episode, timestep] = reward
                    if args.render:
                        env.render()

                episode_rewards = np.sum(rewards, axis=1)

            # Reward summary
            mean = np.mean(episode_rewards)
            var = np.var(episode_rewards)
            reward_summary = {
                "checkpoint": checkpoint,
                "rewards": episode_rewards,
                "mean": mean,
                "var": var,
            }
            print("\n", reward_summary, "\n\n\n")
            reward_summaries.append(reward_summary)

    finally:
        env.close()


if __name__ == "__main__":
    main()
