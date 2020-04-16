import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import gc

from typing import Union
from tqdm.auto import tqdm as tqdm

def plot_wrmsse_by_level(weights_rmsse):
    '''
    Plot RMSSE for each of the 12 levels
    '''
    wrmsses = [np.sum(weights_rmsse[["Weight", "RMSSE"]].prod(axis=1)) / 144]
    for level in range(1, 13):
        weights_rmsse_level = weights_rmsse[weights_rmsse['Level_id'] == level]
        wrmsse = np.sum(weights_rmsse_level[["Weight", "RMSSE"]].prod(axis=1)) / 12
        wrmsses.append(wrmsse)
    labels = ['Overall'] + [f'Level {i}' for i in range(1, 13)]

    plt.figure(figsize=(12, 5))
    ax = sns.barplot(x=labels, y=wrmsses)
    ax.set(xlabel='', ylabel='WRMSSE')
    plt.title('WRMSSE by Level', fontsize=20, fontweight='bold')
    for index, val in enumerate(wrmsses):
        ax.text(index * 1, val + .01, round(val, 4), color='black',
                ha="center")


def plot_rmsse_weight(weights_rmsse, group_ids, level):
    '''
    For a specific level, plot each item's weight and rmsse
    '''
    weights_rmsse_level = weights_rmsse[weights_rmsse['Level_id'] == level]

    scores = weights_rmsse_level["RMSSE"]
    weights = weights_rmsse_level["Weight"]

    if level > 1 and level < 9:
        if level < 7:
            fig, axs = plt.subplots(1, 2, figsize=(12, 3))
        else:
            fig, axs = plt.subplots(2, 1, figsize=(12, 8))

        scores.plot.bar(width=.8, ax=axs[0], color='g')
        axs[0].set_title(f"RMSSE", size=14)
        axs[0].set(xlabel='', ylabel='RMSSE')
        if level >= 4:
            axs[0].tick_params(labelsize=8)
        for index, val in enumerate(scores):
            axs[0].text(index * 1, val + .01, round(val, 4), color='black',
                        ha="center", fontsize=10 if level == 2 else 8)

        weights.plot.bar(width=.8, ax=axs[1])
        axs[1].set_title(f"Weight", size=14)
        axs[1].set(xlabel='', ylabel='Weight')
        if level >= 4:
            axs[1].tick_params(labelsize=8)
        for index, val in enumerate(weights):
            axs[1].text(index * 1, val + .01, round(val, 2), color='black',
                        ha="center", fontsize=10 if level == 2 else 8)

        fig.suptitle(f'Level {level}: {group_ids[level]}', size=24,
                     y=1.1, fontweight='bold')
        plt.tight_layout()
        plt.show()


def create_dashboard(evaluator):
    '''
    create dashboard for an evaluator
    '''
    weights = evaluator.weights
    rmsse = evaluator.rmsse
    weights_rmsse = pd.concat([weights, rmsse], axis=1, sort=False)
    group_ids = evaluator.group_ids

    plot_wrmsse_by_level(weights_rmsse)

    for level in range(2, 9):
        plot_rmsse_weight(weights_rmsse, group_ids, level)

if __name__ == '__main__':
    args = parse_args()
    main(args)


