import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
from matplotlib.patches import Patch

CONFIG = ".hydra/config.yaml"


def multi_plot(df, env_name, device, save_path, selects={}, show=False):
    """
    plot step_per_sec  of a given df and cfg.
    """
    # filter values:
    for (label, value) in selects.items():
        df = df.loc[df[label] == value]

    pop_values = df["n_pop"].unique()
    other_values = {}
    row_lens = []
    for n_pop in pop_values:
        df_pop = df.loc[df["n_pop"] == n_pop]

        column_values = df_pop[["n_env", "n_step"]].values
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
    bar_colors = {}
    for i, n_pop in enumerate(pop_values):
        ax = plt.subplot(grid[i, 0])
        ax.annotate(
            f"Pop={n_pop}",
            xy=(0, 0.5),
            xytext=(-ax.yaxis.labelpad - 5, 0),
            xycoords=ax.yaxis.label,
            textcoords="offset points",
            size="large",
            ha="right",
            va="center",
        )
        for j, (n_env, n_step) in enumerate(other_values[n_pop]):
            ax = plt.subplot(grid[i, j])

            ax.set_title("Subplot row %s \n" % i, fontsize=16)
            ax.set_title(f"n_env={n_env},n_step={n_step} ")
            ax.set_yscale("log")
            ax.set(ylim=(1e1, 1e8))
            ax.set_yticks(10 ** (np.arange(1, 9)))
            c_df = df.loc[
                (df["n_pop"] == n_pop)
                & (df["n_env"] == n_env)
                & (df["n_step"] == n_step)
            ]
            g = sns.barplot(x="method", y="step_per_sec", data=c_df)
            # g = sns.boxplot(x="method", y="step_per_sec", data=c_df)
            ax.bar_label(ax.containers[0], fmt="%.1e", fontsize=30 / len(ax.patches))
            handle, label = ax.get_legend_handles_labels()
            g.set_xlabel(None)
            ax.axes.xaxis.set_visible(False)

            n_repetitions.append(min(c_df.groupby("method").size()))
    bar_colors = {
        label: patch.get_facecolor()
        for label, patch in zip(df["method"].unique(), ax.patches)
    }
    patches = [
        Patch(facecolor=color, label=label) for label, color in bar_colors.items()
    ]

    # add legend in empty cell:
    empty_row = np.argmin([len(other_values[pop]) for pop in pop_values])
    empty_column = len(other_values[pop_values[empty_row]])
    ax = plt.subplot(grid[empty_row, empty_column])
    # plt.legend(handle)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        handles=patches,
        loc="center",
    )

    fig.suptitle(
        f"Steps Per Second ({device}) ({env_name})"
        + f" (mean of {min(n_repetitions)} exps) "
    )
    fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)


def multi_plot_jax_vs_pytorch(
    df, env_name, device, save_path, selects={}, show=False, logscale=True
):
    """
    plot step_per_sec  of a given df and cfg.
    """

    # add labels:
    # has_jax = (df["method"] == "mms_no") | (df["method"] == "jax_256_x2")
    no_actor = (df["method"] == "mms_no") | (df["method"] == "fmm_no")

    df.loc[
        (df["method"] == "mms_no") | (df["method"] == "jax_256_x2"), "worker"
    ] = "jax"
    df.loc[
        (df["method"] == "fmm_no") | (df["method"] == "torch_256_x2"), "worker"
    ] = "pytorch"
    df.loc[
        (df["method"] == "fff_no") | (df["method"] == "gym_256_x2"), "worker"
    ] = "gym"

    # df["jax"] = np.where(has_jax, "jax", "pytorch")
    df["has_actor"] = np.where(
        no_actor,
        "no_actor",
        "actor",
    )
    # filter values:
    for (label, value) in selects.items():
        df = df.loc[df[label] == value]

    if df.empty:
        return

    pop_values = df["n_pop"].unique()
    other_values = {}
    row_lens = []
    for n_pop in pop_values:
        df_pop = df.loc[df["n_pop"] == n_pop]

        column_values = df_pop[["n_env", "n_step"]].values
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
    row_lens = np.fromiter(map(len, other_values.values()), dtype=int)
    if (row_lens[:-1] == row_lens[1:]).all():  # if all rows have the same length
        columns = row_lens[0] + 1
    else:
        columns = np.max(row_lens)
    # columns = np.max(list(map(len, other_values.values())), axis=0)
    grid = plt.GridSpec(rows, columns, wspace=0.6, hspace=0.6)

    for i, n_pop in enumerate(pop_values):
        ax = plt.subplot(grid[i, 0])
        ax.annotate(
            f"Pop={n_pop}",
            xy=(0, 0.5),
            xytext=(-ax.yaxis.labelpad - 5, 0),
            xycoords=ax.yaxis.label,
            textcoords="offset points",
            size="large",
            ha="right",
            va="center",
        )
        for j, (n_env, n_step) in enumerate(other_values[n_pop]):
            ax = plt.subplot(grid[i, j])

            ax.set_title("Subplot row %s \n" % i, fontsize=16)
            ax.set_title(f"n_env={n_env},n_step={n_step} ")
            if logscale:
                ax.set_yscale("log")
                ax.set(ylim=(1e1, 1e8))
                ax.set_yticks(10 ** (np.arange(1, 9)))
            else:
                ax.set(ylim=(0, df["step_per_sec"].unique().max() * 1.1))
            c_df = df.loc[
                (df["n_pop"] == n_pop)
                & (df["n_env"] == n_env)
                & (df["n_step"] == n_step)
            ]
            g = sns.barplot(
                x="has_actor",
                y="step_per_sec",
                hue="worker",
                order=["actor", "no_actor"],
                hue_order=["gym", "pytorch", "jax"],
                data=c_df,
            )
            # g = sns.boxplot(x="method", y="step_per_sec", data=c_df)
            [
                ax.bar_label(container, fmt="%.e", fontsize=30 / len(ax.patches))
                for container in ax.containers
            ]
            handle, label = ax.get_legend_handles_labels()
            ax.get_legend().remove()
            g.set_xlabel(None)
            # ax.axes.xaxis.set_visible(False)

            n_repetitions.append(min(c_df.groupby("method").size()))

    # add legend in empty cell:
    empty_row = np.argmin([len(other_values[pop]) for pop in pop_values])
    empty_column = len(other_values[pop_values[empty_row]])
    ax = plt.subplot(grid[empty_row, empty_column])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        handles=handle,
        loc="center",
    )

    fig.suptitle(
        f"Steps Per Second ({device}) ({env_name})"
        + f" (mean of {min(n_repetitions)} exps) "
    )
    fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_each_exp():
    """
    save a plot in each of the output folders
    """
    exp_folder = "outputs"
    for dir_path, dir_name, filenames in os.walk(exp_folder):
        for file in filenames:
            if file.endswith(".csv"):
                csv_path = os.path.join(dir_path, file)
                save_path = os.path.join(dir_path, f"{os.path.splitext(file)[0]}.png")
                cfg = omegaconf.OmegaConf.load(os.path.join(dir_path, CONFIG))
                df = pd.read_csv(csv_path)
                multi_plot_jax_vs_pytorch(
                    df,
                    cfg.env_name,
                    cfg.device,
                    save_path,
                )


def plot_from_all_exps(save_filepath, selects, device=None):
    """
    Gather exp from all folders inside outputs and plot in the target save_filepath.
    if GPU=None, merge all result withou
    """
    exp_folder = "outputs"
    merged_df = None
    merged_df = {}
    for dir_path, dir_name, filenames in os.walk(exp_folder):
        for file in filenames:
            if file.endswith(".csv"):
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
        multi_plot_jax_vs_pytorch(
            merged_df[key],
            device,
            env_name,
            save_filepath + f"_{device}_{env_name}.pdf",
            selects,
        )


def main():
    plot_each_exp()
    plot_from_all_exps("benchmark/plots/spc", selects={"total_steps": 100000})
    plot_from_all_exps(
        "benchmark/plots/spc_no_actor",
        selects={"total_steps": 100000, "has_actor": "no_actor"},
    )


if __name__ == "__main__":
    main()
