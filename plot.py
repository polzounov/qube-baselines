#!/usr/bin/env python3
import argparse
import os

import matplotlib.pyplot as plt
from stable_baselines import results_plotter


def main():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-d", "--dirs", help="List of log directories", nargs="*", default=["logs1e6"]
    )
    parser.add_argument("-ns", "--num_timesteps", type=float, default=1e6)
    parser.add_argument("-x", "--xaxis", help="Varible on X-axis", default="timesteps")
    parser.add_argument("-n", "--task_name", help="Title of plot", default="Qube Sim")
    args = parser.parse_args()
    args.dirs = [os.path.abspath(folder) for folder in args.dirs]
    results_plotter.plot_results(
        args.dirs, int(args.num_timesteps), args.xaxis, args.task_name
    )
    plt.show()


if __name__ == "__main__":
    main()
