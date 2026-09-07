import os
import re
import warnings

import numpy as np
import pandas as pd

metrics = {
    "Accuracy": "accuracy",
    "BalancedAccuracy": "balanced accuracy",
    "LogLoss": "cross-entropy loss",
    "ROCAUC": "ROCAUC",
    "F1": "F1 score",
}


def plot_pairwise_scatter(
    results_a,
    results_b,
    method_a,
    method_b,
    metric="accuracy",
    lower_better=False,
    statistic_tests=True,
    title=None,
    figsize=(8, 8),
    color_palette="tab10",
    best_on_top=True,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.offsetbox import AnchoredText

    palette = sns.color_palette(color_palette, n_colors=3)

    if isinstance(results_a, list):
        results_a = np.array(results_a)
    if isinstance(results_b, list):
        results_b = np.array(results_b)

    if len(results_a.shape) != 1:
        raise ValueError("results_a must be a 1D array.")
    if len(results_b.shape) != 1:
        raise ValueError("results_b must be a 1D array.")

    if statistic_tests:
        fig, ax = plt.subplots(figsize=figsize, gridspec_kw=dict(bottom=0.2))
    else:
        fig, ax = plt.subplots(figsize=figsize)

    results_all = np.concatenate((results_a, results_b))
    min_value = results_all.min() * 1.05 if (results_all.min() < 0) else results_all.min() * 0.95
    max_value = results_all.max() * 1.05 if (results_all.max() > 0) else results_all.max() * 0.95

    x, y = [min_value, max_value], [min_value, max_value]
    ax.plot(x, y, color="black", alpha=0.5, zorder=1)

    if (results_a.mean() <= results_b.mean() and not lower_better) or (
        results_a.mean() >= results_b.mean() and lower_better
    ):
        first = results_b
        first_method = method_b
        second = results_a
        second_method = method_a
    else:
        first = results_a
        first_method = method_a
        second = results_b
        second_method = method_b

    if not best_on_top:
        first, second = second, first
        first_method, second_method = second_method, first_method

    differences = [0 if i - j == 0 else (1 if i - j > 0 else -1) for i, j in zip(first, second)]
    differences, first, second = map(
        np.array,
        zip(*sorted(zip(differences, first, second), key=lambda x: -abs(x[0]))),
    )

    first_median = np.median(first)
    second_median = np.median(second)

    plot = sns.scatterplot(
        x=second,
        y=first,
        hue=differences,
        hue_order=[1, 0, -1] if lower_better else [-1, 0, 1],
        palette=palette,
        zorder=2,
    )

    ax.plot(
        [first_median, min_value] if not lower_better else [first_median, max_value],
        [first_median, first_median],
        linestyle="--",
        color=palette[2],
        zorder=3,
    )

    ax.plot(
        [second_median, second_median],
        [second_median, min_value] if not lower_better else [second_median, max_value],
        linestyle="--",
        color=palette[0],
        zorder=3,
    )

    legend_median = AnchoredText(
        "* Dashed lines represent the median",
        loc="lower right" if lower_better else "upper right",
        prop=dict(size=8),
        bbox_to_anchor=(1.01, 1.037 if lower_better else -0.09),
        bbox_transform=ax.transAxes,
    )
    ax.add_artist(legend_median)

    if lower_better:
        differences = [-i for i in differences]
        ax = plt.gca()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.spines["top"].set_visible(True)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.spines["right"].set_visible(True)

    plot.set_ylabel(f"{first_method} {metric}\n(mean: {first.mean():.4f})", fontsize=13)
    plot.set_xlabel(f"{second_method} {metric}\n(mean: {second.mean():.4f})", fontsize=13)

    wins_A = losses_B = sum(i == 1 for i in differences)
    ties_A = ties_B = sum(i == 0 for i in differences)
    losses_A = wins_B = sum(i == -1 for i in differences)

    plot.set_ylim(min_value, max_value)
    plot.set_xlim(min_value, max_value)

    plot.get_legend().remove()

    anc_text = AnchoredText(
        f"{first_method} wins here\n[{wins_A}W, {ties_A}T, {losses_A}L]",
        loc="upper left" if not lower_better else "lower right",
        frameon=True,
        prop=dict(
            color=palette[2],
            fontweight="bold",
            fontsize=13,
            ha="center",
        ),
    )
    anc_text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    anc_text.patch.set_color("wheat")
    anc_text.patch.set_edgecolor("black")
    anc_text.patch.set_alpha(0.5)
    ax.add_artist(anc_text)

    anc_text = AnchoredText(
        f"{second_method} wins here\n[{wins_B}W, {ties_B}T, {losses_B}L]",
        loc="lower right" if not lower_better else "upper left",
        frameon=True,
        prop=dict(
            color=palette[0],
            fontweight="bold",
            fontsize=13,
            ha="center",
        ),
    )
    anc_text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    anc_text.patch.set_color("wheat")
    anc_text.patch.set_edgecolor("black")
    anc_text.patch.set_alpha(0.5)
    ax.add_artist(anc_text)

    if title is not None:
        plot.set_title(rf"{title}", fontsize=16)

    if statistic_tests:
        if np.all(results_a == results_b):
            warnings.warn(
                f"Estimators {method_a} and {method_b} have the same performance"
                "on all datasets. This may cause problems when forming cliques.",
                stacklevel=2,
            )

            p_value_t = 1
            p_value_w = 1

        else:
            from scipy.stats import ttest_rel, wilcoxon

            p_value_t = ttest_rel(
                first,
                second,
                alternative="less" if lower_better else "greater",
            )[1]

            p_value_w = wilcoxon(
                first,
                second,
                zero_method="wilcox",
                alternative="less" if lower_better else "greater",
            )[1]

        p_value_t_str = f"= {p_value_t:.3f}" if p_value_t > 1e-3 else "≤ 1e-3"
        p_value_w_str = f"= {p_value_w:.3f}" if p_value_w > 1e-3 else "≤ 1e-3"
        ttes = f"Paired t-test for equality of means: p-value {p_value_t_str}"
        wilcox = f"Wilcoxon test for equality of medians: p-value {p_value_w_str}"

        plt.figtext(
            0.5,
            0.03 if not lower_better else 0.13,
            f"{wilcox}\n{ttes}",
            fontsize=10,
            wrap=True,
            horizontalalignment="center",
            bbox=dict(
                facecolor="wheat",
                edgecolor="black",
                boxstyle="round,pad=0.5",
                alpha=0.5,
            ),
        )

    return fig, ax


def _slugify(label):
    return re.sub(r"[^0-9A-Za-z]+", "_", label).strip("_")


def _paired_scores(df_pivot, algorithm_a, algorithm_b):
    paired = df_pivot[[algorithm_a, algorithm_b]]
    missing = paired[paired.isna().any(axis=1)]
    if len(missing) > 0:
        warnings.warn(
            f"{len(missing)} dataset(s) missing a score for {algorithm_a!r} and/or {algorithm_b!r} "
            f"-- dropped from this comparison: {list(missing.index)}",
            stacklevel=2,
        )
        paired = paired.dropna()
    return paired[algorithm_a].to_numpy(), paired[algorithm_b].to_numpy()


def compare_quant_variants():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dir_path = os.path.join("results", "classification", "single_thread_ucr142_estimator_comparison")

    df = pd.read_csv(os.path.join(dir_path, "metrics.csv"))
    df["Estimator"] = df["Estimator"].map(
        {
            "moment_quant_approx": """MomentQuant("approx")""",
            "moment_quant_exact": """MomentQuant("exact")""",
            "quant": """Quant""",
        }
    )

    df_metrics = {}
    for metric in metrics:
        df_metrics[metric] = (
            df.groupby(["Estimator", "Dataset"])[metric]
            .mean()
            .reset_index()
            .pivot(columns="Estimator", index="Dataset", values=metric)
        )

    algorithm_a_list = ["""MomentQuant("approx")""", """MomentQuant("approx")""", """MomentQuant("exact")"""]
    algorithm_b_list = ["""Quant""", """MomentQuant("exact")""", """Quant"""]

    for algorithm_a, algorithm_b in zip(algorithm_a_list, algorithm_b_list):
        for metric, metric_name in metrics.items():
            results_a, results_b = _paired_scores(df_metrics[metric], algorithm_a, algorithm_b)
            fig, _ = plot_pairwise_scatter(
                results_a=results_a,
                results_b=results_b,
                method_a=algorithm_a,
                method_b=algorithm_b,
                metric=metric_name,
                lower_better=True if metric == "LogLoss" else False,
                figsize=(5.147, 5.147),
            )
            fig.savefig(
                os.path.join(dir_path, f"{_slugify(algorithm_a)}_vs_{_slugify(algorithm_b)}_{metric}.pdf"),
                bbox_inches="tight",
            )
            plt.close(fig)


def compare_classifiers():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dir_path = os.path.join("results", "classification", "ucr142_classifier_comparison")

    df = pd.read_csv(os.path.join(dir_path, "metrics.csv"))
    df["Classifier"] = df["Classifier"].map(
        {
            "extra_trees": "Extremely randomized trees",
            "random_forest": "Random forest",
            "ridge": "Ridge",
        }
    )

    df_metrics = {}
    for metric in metrics:
        df_metrics[metric] = (
            df.groupby(["Classifier", "Dataset"])[metric]
            .mean()
            .reset_index()
            .pivot(columns="Classifier", index="Dataset", values=metric)
        )

    algorithm_a_list = ["Extremely randomized trees", "Extremely randomized trees"]
    algorithm_b_list = ["Random forest", "Ridge"]

    for algorithm_a, algorithm_b in zip(algorithm_a_list, algorithm_b_list):
        for metric, metric_name in metrics.items():
            results_a, results_b = _paired_scores(df_metrics[metric], algorithm_a, algorithm_b)
            fig, _ = plot_pairwise_scatter(
                results_a=results_a,
                results_b=results_b,
                method_a=algorithm_a,
                method_b=algorithm_b,
                metric=metric_name,
                lower_better=True if metric == "LogLoss" else False,
                figsize=(5.147, 5.147),
            )
            fig.savefig(
                os.path.join(dir_path, f"{_slugify(algorithm_a)}_vs_{_slugify(algorithm_b)}_{metric}.pdf"),
                bbox_inches="tight",
            )
            plt.close(fig)


if __name__ == "__main__":
    compare_quant_variants()
    compare_classifiers()
