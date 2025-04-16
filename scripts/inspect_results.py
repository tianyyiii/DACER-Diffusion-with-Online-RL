from typing import Dict

import relax
from pathlib import Path
import re
import csv
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


sns.set_style('whitegrid')
sns.set_context(font_scale=2.6)


def plot_mean(patterns_dict: Dict, env_name, fig_name = None,
              max_steps=None):
    plt.figure(figsize=(4*1.25, 3*1.25))
    package_path = Path(relax.__file__)
    logdir = package_path.parent.parent / 'logs' / env_name
    dfs = []
    for alg, pattern in patterns_dict.items():
        matching_dir = [s for s in logdir.iterdir() if re.match(pattern, str(s))]
        for dir in matching_dir:
            csv_path = dir / 'log.csv'
            df = pd.read_csv(str(csv_path))
            print(str(dir), "dir")
            df.loc[:, ('seed')] = str(dir).split('_s')[1].split('_')[0]
            df.loc[:, ('alg')] = alg
            dfs.append(df)
    try:
        total_df = pd.concat(dfs, ignore_index=True)
    except:
        return
    # if max_steps is not None:
    plt.title(env_name)
    sns.lineplot(data=total_df, x='step', y='avg_ret', hue='alg')
    if fig_name is not None:
        plt.savefig(fig_name)
    else:
        plt.show()
    


def load_best_results(pattern, env_name, show_df=False,
              max_steps=None):
    package_path = Path(relax.__file__)
    logdir = package_path.parent.parent / 'logs' / env_name
    matching_dir = [s for s in logdir.iterdir() if re.match(pattern, str(s))]
    dfs = []
    for dir in matching_dir:
        csv_path = dir / 'log.csv'
        df = pd.read_csv(str(csv_path))
        if max_steps is not None:
            df = df[df['step'] < max_steps]
        sliced_df = df.loc[df['avg_ret'].idxmax()]
        # sliced_df.loc['seed'] = str(dir).split('_s')[1].split('_')[0]
        dfs.append(sliced_df)
    total_df = pd.concat(dfs, ignore_index=True, axis=1).T
    if show_df:
        print(total_df.to_markdown())
    print(f"${total_df['avg_ret'].mean():.0f} \pm {total_df['avg_ret'].std():.0f}$")
    return total_df

if __name__ == "__main__":
    for env in ['Ant-v4']:
        patterns_dict = {
            'lambda_0.0': r".*/sdac_soft_v2.*entropy_lambda_0.0",
            'lambda_0.01': r".*/sdac_soft_v2.*entropy_lambda_0.01",
            'lambda_0.05': r".*/sdac_soft_v2.*entropy_lambda_0.05",
            'lambda_0.1': r".*/sdac_soft_v2.*entropy_lambda_0.1",
            'lambda_0.2': r".*/sdac_soft_v2.*entropy_lambda_0.2",
            'lambda_0.5': r".*/sdac_soft_v2.*entropy_lambda_0.5",
            'lambda_1.0': r".*/sdac_soft_v2.*entropy_lambda_1.0",
        }
        plot_mean(patterns_dict, env, f"figures/{env}_lambda.png")
        for pattern in patterns_dict:
            print(pattern)
            load_best_results(patterns_dict[pattern], env)