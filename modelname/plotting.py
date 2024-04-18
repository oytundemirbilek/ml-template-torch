"""Module to define plotting utilities."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_results_boxplot(
    out_path: str, metric: str = "mse", size_multiplier: int = 3
) -> None:
    """Plot the given benchmarks in a boxplot."""
    dataset_names = ["dataset1", "dataset2"]
    benchmarks = ["benchmark1", "benchmark2", "modelname"]
    custom_palette = {
        "benchmark1": "#CB3335",
        "benchmark2": "#477CA8",
        "modelname": "#905998",
    }

    # Collect data
    all_dfs = []
    for benchmark in benchmarks:
        for dataset_name in dataset_names:
            path = os.path.join(
                out_path,
                benchmark,
                f"{dataset_name}",
                f"{dataset_name}_{metric}.csv",
            )
            df = pd.DataFrame()
            result_arr = pd.read_csv(path, header=None).values

            df["Metric"] = np.squeeze(result_arr)
            df["Dataset"] = dataset_name
            df["Benchmark"] = benchmark
            if benchmark == "modelname":
                df["Benchmark"] + df["Benchmark"] + " (ours)"
            all_dfs.append(df)

    all_combined = pd.concat(all_dfs).reset_index(drop=True)

    sns.set(
        context="paper",
        style="darkgrid",
        rc={
            "figure.dpi": 100 * size_multiplier,
            "figure.figsize": (10, 5),
            # "axes.titlesize": 20 / size_multiplier,
            # "axes.labelsize": 20 / size_multiplier,
            # "axes.linewidth": 1.0 / size_multiplier,
            "xtick.labelsize": 25.0,
            # "xtick.major.pad": 3.5 / size_multiplier,
            # "xtick.minor.pad": 3.4 / size_multiplier,
            "ytick.labelsize": 15.0,
            # "ytick.major.pad": 3.5 / size_multiplier,
            # "ytick.minor.pad": 3.4 / size_multiplier,
            # "legend.fontsize": 20 / size_multiplier,
            # "legend.title_fontsize": 20 / size_multiplier,
            # "figure.titlesize": 20 / size_multiplier,
            # "lines.linewidth": 1.0 / size_multiplier,
            # "lines.markersize": 6.0 / size_multiplier,
            # "boxplot.flierprops.markersize": 6.0 / size_multiplier,
            # "boxplot.flierprops.markeredgewidth": 1.0 / size_multiplier,
            # "boxplot.flierprops.linewidth": 1.0 / size_multiplier,
            # "boxplot.boxprops.linewidth": 1.0 / size_multiplier,
            # "boxplot.whiskerprops.linewidth": 1.0 / size_multiplier,
            # "boxplot.capprops.linewidth": 1.0 / size_multiplier,
            # "boxplot.medianprops.linewidth": 1.0 / size_multiplier,
            # "boxplot.meanprops.linewidth": 1.0 / size_multiplier,
            # "boxplot.meanprops.markersize": 6.0 / size_multiplier,
            # "boxplot.showmeans": True,
        },
    )

    # Create and display the plot
    ax = sns.boxplot(
        x="Dataset",
        y="Metric",
        hue="Benchmark",
        data=all_combined,
        # palette="Set1",
        palette=custom_palette,
        width=0.8,
        dodge=True,
        legend=None,
    )
    # ax = sns.swarmplot(
    #     x="Dataset", y="Metric", hue="Benchmark", data=all_combined, dodge=True
    # )
    if metric == "mse":
        ax.set_ylim([0.0, 0.02])
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{metric}.png")
    plt.close()
    all_combined.to_csv(f"./boxplot_table_{metric}.csv", index=False)
