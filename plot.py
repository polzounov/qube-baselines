#!/usr/bin/env python3
import argparse
import os

import seaborn as sns
import matplotlib.pyplot as plt
from baselines.common import plot_util as pu

sns.set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dirs", help="List of log directories", required=True, nargs="*"
    )
    parser.add_argument("-ns", "--num_timesteps", type=float, default=1e6)
    parser.add_argument("-x", "--xaxis", help="Varible on X-axis", default="timesteps")
    parser.add_argument(
        "-n", "--task_name", help="Title of plot", default="Qube Simulator"
    )
    args = parser.parse_args()
    dirs = [os.path.abspath(folder) for folder in args.dirs]

    xy_list = pu.load_results(dirs)

    pu.plot_results(
        xy_list,
        # xy_fn=default_xy_fn,
        # split_fn=lambda _: "",
        # group_fn=default_split_fn,
        average_group=True,
        shaded_std=False,
        shaded_err=True,
        figsize=None,
        legend_outside=False,
        resample=0,
        smooth_step=1.0,
        tiling="vertical",
        xlabel=args.xaxis,
        ylabel="Reward",
    )
    plt.show()


if __name__ == "__main__":
    main()
