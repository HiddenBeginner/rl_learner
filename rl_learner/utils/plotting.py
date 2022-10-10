import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_datasets(logdir):
    """
    Recursively look through logdir for output files produced by EpochLogger.

    Assumes that any file "progress.txt" is a valid hit.

    Refereces:
        https://github.com/openai/spinningup/blob/master/spinup/utils/plot.py#L61
    """
    datasets = []
    for root, _, files in os.walk(logdir):
        if 'progress.txt' in files:
            exp_dir, run_id = os.path.split(root)
            exp_dir, agent_name = os.path.split(exp_dir)
            _, env_name = os.path.split(exp_dir)
            exp_data = pd.read_table(os.path.join(root, 'progress.txt'))

            # Add columns indicating experiments
            exp_data.insert(len(exp_data.columns), 'Env', env_name)
            exp_data.insert(len(exp_data.columns), 'Agent', agent_name)
            exp_data.insert(len(exp_data.columns), 'ID', run_id)

            datasets.append(exp_data)

    return datasets


def plot_data(data, target="AvgEpRet", smooth=1, hue='Agent', **kwargs):
    if smooth > 1:
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[target])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            datum[target] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)

    # Plotting
    sns.set(style='darkgrid', font_scale=1.5)
    sns.lineplot(data=data, x='Steps', y=target, hue=hue, **kwargs)
    plt.legend(loc='best').set_draggable(True)

    xscale = np.max(np.asarray(data['Steps'])) > 5e3
    if xscale:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 1))
    plt.tight_layout(pad=0.5)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--savedir', type=str, default=None)
    parser.add_argument('--x', type=str, default='Steps')
    parser.add_argument('--y', type=str, default='AvgEpRet')
    parser.add_argument('--smooth', type=int, default=1)
    parser.add_argument('--fig_width', type=float, default=8)
    parser.add_argument('--fig_height', type=float, default=5)

    args = parser.parse_args()

    return args


def main():
    args = get_arguments()
    datasets = get_datasets(args.logdir)
    plt.figure(figsize=(args.fig_width, args.fig_height))
    plot_data(datasets, target=args.y, smooth=args.smooth)
    if args.savedir is not None:
        plt.savefig(args.savedir, dpi=300)
    else:
        plt.show()


if __name__ == '__main__':
    main()
