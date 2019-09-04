#!/usr/bin/env python3
import numpy as np
import argparse
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

from gym_brt.control import (
    zero_policy,
    constant_policy,
    random_policy,
    square_wave_policy,
    energy_control_policy,
    pd_control_policy,
    flip_and_hold_policy,
    square_wave_flip_and_hold_policy,
    dampen_policy,
)


def play(env_name, n_episodes, max_timesteps, hardware, logdir, policy, render):
    # Beyond end of episode reward is always 0
    # Extra 1 timestep is a bug with env
    rewards = np.zeros((n_episodes, max_timesteps + 1))  # Bug with env
    frequency = 250

    with env_name(
        use_simulator=not hardware, batch_size=max_timesteps, frequency=frequency
    ) as env:
        for episode in range(n_episodes):
            print("Running episode: {}".format(episode))
            obs = env.reset()
            timestep = 0
            done = False

            while not done:
                action = policy(obs, step=timestep, frequency=frequency)
                obs, reward, done, info = env.step(action)
                rewards[episode, timestep] = reward
                if render:
                    env.render()
                timestep += 1

    episode_rewards = np.sum(rewards, axis=1)
    print(episode_rewards)

    mean = np.mean(episode_rewards)
    var = np.var(episode_rewards)
    return {"rewards": episode_rewards, "mean": mean, "var": var}


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
    controllers = {
        "none": zero_policy,
        "zero": zero_policy,
        "const": constant_policy,
        "rand": random_policy,
        "random": random_policy,
        "sw": square_wave_policy,
        "energy": energy_control_policy,
        "pd": pd_control_policy,
        "hold": pd_control_policy,
        "flip": flip_and_hold_policy,
        "sw-hold": square_wave_flip_and_hold_policy,
        "damp": dampen_policy,
    }

    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", choices=list(envs.keys()), required=True)
    parser.add_argument("-ms", "--max-timesteps", type=int, default=2048)
    parser.add_argument("-ne", "--num-episodes", type=int, default=50)
    parser.add_argument("-hw", "--use-hardware", action="store_true")
    parser.add_argument("-ld", "--logdir", type=str, default="logs")
    parser.add_argument("-r", "--render", action="store_true")
    parser.add_argument(
        "-c",
        "--controller",
        default="flip",
        choices=list(controllers.keys()),
        help="Select what the controller to use.",
    )
    args, _ = parser.parse_known_args()

    reward_summary = play(
        env_name=envs[args.env],
        n_episodes=int(args.num_episodes),
        max_timesteps=int(args.max_timesteps),
        hardware=args.use_hardware,
        logdir=args.logdir,
        policy=controllers[args.controller],
        render=args.render,
    )

    print(reward_summary)


if __name__ == "__main__":
    main()
