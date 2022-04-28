import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import omegaconf

CONFIG = '.hydra/config.yaml'


def multi_plot(df, env_name, device, save_path, selects=[]):
    '''
        plot step_per_sec  of a given df and cfg.
    '''
    # filter values:
    for (label, value) in selects:
        df = df.loc[df[label] == value]

    pop_values = df['n_pop'].unique()
    other_values = {}
    row_lens = []
    for n_pop in pop_values:
        df_pop = df.loc[df['n_pop'] == n_pop]

        column_values = df_pop[['n_env', 'n_step']].values
        nenv_nstep_values = np.unique(np.array(column_values), axis=0)
        row_lens.append(len(nenv_nstep_values))
        other_values[n_pop] = nenv_nstep_values

    n_repetitions = []
    # order columns :
    for n_pop in pop_values:
        values_list = other_values[n_pop].tolist()
        values_list.sort(key=lambda tup: tup[1], reverse=True)
        other_values[n_pop] = values_list

    fig = plt.figure(figsize=(12, 6))

    rows = len(pop_values)
    columns = np.max(list(map(len, other_values.values())), axis=0)
    grid = plt.GridSpec(rows, columns, wspace=0.6, hspace=0.6)
    for i, n_pop in enumerate(pop_values):
        ax = plt.subplot(grid[i, 0])
        ax.annotate(f'Pop={n_pop}', xy=(0, 0.5), xytext=(-ax.yaxis.labelpad-5, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
        for j, (n_env, n_step) in enumerate(other_values[n_pop]):
            ax = plt.subplot(grid[i, j])

            ax.set_title("Subplot row %s \n" % i, fontsize=16)
            ax.set_title(f"n_env={n_env},n_step={n_step} ")
            ax.set_yscale('log')
            ax.set(ylim=(1e1, 1e6))
            ax.set_yticks(10**(np.arange(1, 7)))
            c_df = df.loc[(df['n_pop'] == n_pop) &
                          (df['n_env'] == n_env) &
                          (df['n_step'] == n_step)
                          ]
            g = sns.barplot(x='method', y="step_per_sec", data=c_df)
            ax.bar_label(ax.containers[0], fmt='%.1e')
            g.set_xlabel(None)
            n_repetitions.append(min(c_df.groupby('method').size()))

    fig.suptitle(f"Steps Per Second ({device}) ({env_name})" +
                 f" (mean of {min(n_repetitions)} exps) ")
    fig.savefig(save_path)
    plt.close(fig)


def plot_each_exp():
    '''
    save a plot in each of the output folders
    '''
    exp_folder = 'outputs'
    for dir_path, dir_name, filenames in os.walk(exp_folder):
        for file in filenames:
            if file.endswith('.csv'):
                csv_path = os.path.join(dir_path, file)
                save_path = os.path.join(
                    dir_path, f"{os.path.splitext(file)[0]}.png")
                cfg = omegaconf.OmegaConf.load(os.path.join(dir_path, CONFIG))
                df = pd.read_csv(csv_path)
                multi_plot(df, cfg.env_name, cfg.device, save_path,)


def plot_from_all_exps(save_filepath, selects, device=None):
    '''
    Gather exp from all folders inside outputs and plot in the target save_filepath.
    if GPU=None, merge all result withou
    '''
    exp_folder = 'outputs'
    merged_df = None
    merged_df = {}
    for dir_path, dir_name, filenames in os.walk(exp_folder):
        for file in filenames:
            if file.endswith('.csv'):
                csv_path = os.path.join(dir_path, file)
                cfg = omegaconf.OmegaConf.load(os.path.join(dir_path, CONFIG))
                device = cfg.device
                env_name = cfg.env_name
                key = (device, env_name)
                df = pd.read_csv(csv_path)
                if key in merged_df:
                    merged_df[key] = pd.concat([merged_df[key], df])
                else:
                    merged_df[key] = df

    for key in merged_df.keys():
        device, env_name = key
        multi_plot(merged_df[key], device, env_name,
                   save_filepath+f'_{device}_{env_name}.png', selects)


def main():
    plot_each_exp()
    plot_from_all_exps('no_actors/plots/spc',
                       selects=[('total_steps', 100000)])


if __name__ == '__main__':
    main()
