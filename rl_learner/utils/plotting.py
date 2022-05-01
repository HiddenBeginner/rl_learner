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
            _, exp_name = os.path.split(exp_dir)
            exp_data = pd.read_table(os.path.join(root, 'progress.txt'))

            # Add columns indicating experiments
            exp_data.insert(len(exp_data.columns), 'Exp', exp_name)
            exp_data.insert(len(exp_data.columns), 'ID', run_id)
            exp_data.insert(len(exp_data.columns), 'Exp-ID', exp_name + '-' + run_id)

            datasets.append(exp_data)

    return datasets


def plot_data(data, target="AverageEpRet", smooth=1, hue='Exp', **kwargs):
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
    sns.lineplot(data=data, x='Epoch', y=target, hue=hue, **kwargs)
    plt.legend(loc='best').set_draggable(True)

    xscale = np.max(np.asarray(data['Epoch'])) > 5e3
    if xscale:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 1))
    plt.tight_layout(pad=0.5)
