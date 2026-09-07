import math
import os

import pandas as pd

from src.classification.utils import (
    LENGTH_BIN_LABELS,
    MOMENT_QUANT_APPROX,
    MOMENT_QUANT_EXACT,
    QUANT,
    format_estimator_label,
    load_series_lengths,
    make_length_bin_edges,
)

metrics_csv_path = os.path.join("results", "classification", "single_thread_ucr142_estimator_comparison", "metrics.csv")
transform_runtimes_csv_path = os.path.join("results", "runtimes", "single_thread_ucr142", "runtimes.csv")
TRANSFORM_RUNTIME_ESTIMATOR_MAP = {
    "moment_quant_approx_auto": MOMENT_QUANT_APPROX,
    "moment_quant_exact_auto": MOMENT_QUANT_EXACT,
    "quant_float64": QUANT,
}
out_dir = os.path.join("results", "analyses", "accuracy_runtime_tradeoff")

BIN_MARKER_ACCURACY_RADIUS = 0.01
SMALL_DOT_ACCURACY_RADIUS = 0.005
A2_FIGSIZE_WIDTH = 5.147
A2_FIGSIZE_HEIGHT = A2_FIGSIZE_WIDTH * 5.5 / 7

COMPARISONS = [
    ("approx_vs_exact", MOMENT_QUANT_APPROX, MOMENT_QUANT_EXACT),
    ("approx_vs_quant", MOMENT_QUANT_APPROX, QUANT),
    ("exact_vs_quant", MOMENT_QUANT_EXACT, QUANT),
]


def load_transform_runtimes():
    raw = pd.read_csv(transform_runtimes_csv_path)
    raw = raw[raw["Estimator"].isin(TRANSFORM_RUNTIME_ESTIMATOR_MAP)]
    per_split_min = raw.groupby(["Dataset", "Split", "Estimator"])["Runtime"].min().reset_index()
    per_dataset = per_split_min.groupby(["Dataset", "Estimator"])["Runtime"].sum().reset_index(name="TransformRuntime")
    per_dataset["Estimator"] = per_dataset["Estimator"].map(TRANSFORM_RUNTIME_ESTIMATOR_MAP)
    return per_dataset


def make_figures(df, tradeoff, transform_tradeoff, speedup_tradeoff):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    estimators = sorted(transform_tradeoff["Estimator"].unique())
    colors = dict(zip(estimators, ["#4C72B0", "#DD8452", "#55A868"]))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for estimator_name in estimators:
        group = transform_tradeoff[transform_tradeoff["Estimator"] == estimator_name]
        ax.scatter(
            group["TransformRuntime"],
            group["Accuracy"],
            alpha=0.5,
            s=18,
            edgecolors="none",
            color=colors[estimator_name],
            label=format_estimator_label(estimator_name),
        )
    for estimator_name in estimators:
        group = transform_tradeoff[transform_tradeoff["Estimator"] == estimator_name]
        ax.scatter(
            group["TransformRuntime"].mean(),
            group["Accuracy"].mean(),
            color=colors[estimator_name],
            s=220,
            marker="*",
            edgecolor="black",
            linewidth=1.0,
            zorder=5,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Transform-only runtime (in seconds)")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "accuracy_vs_transform_runtime.pdf")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved {fig_path!r}")

    a2_estimators = [e for e in estimators if e != QUANT]
    fig, ax = plt.subplots(figsize=(A2_FIGSIZE_WIDTH, A2_FIGSIZE_HEIGHT))
    small_dot_collections = {}
    for estimator_name in a2_estimators:
        group = transform_tradeoff[transform_tradeoff["Estimator"] == estimator_name]
        small_dot_collections[estimator_name] = ax.scatter(
            group["SeriesLength"],
            group["Accuracy"],
            alpha=0.5,
            edgecolors="none",
            color=colors[estimator_name],
            label=format_estimator_label(estimator_name),
        )
    ax.set_xscale("log")
    ax.set_xlabel("Series length (in time points)")
    ax.set_ylabel("Accuracy")
    fig.tight_layout()

    fig.canvas.draw()
    ax_height_points = ax.get_window_extent(renderer=fig.canvas.get_renderer()).height * 72 / fig.dpi
    y_lo, y_hi = ax.get_ylim()
    points_per_accuracy_unit = ax_height_points / (y_hi - y_lo)

    small_dot_radius_points = SMALL_DOT_ACCURACY_RADIUS * points_per_accuracy_unit
    small_dot_area = math.pi * small_dot_radius_points**2
    bin_marker_radius_points = BIN_MARKER_ACCURACY_RADIUS * points_per_accuracy_unit
    bin_marker_area = math.pi * bin_marker_radius_points**2
    for collection in small_dot_collections.values():
        collection.set_sizes([small_dot_area])
    ax.legend(loc="lower left")
    print(
        f"accuracy_vs_series_length.pdf: small scatter dots drawn at radius "
        f"{SMALL_DOT_ACCURACY_RADIUS} Accuracy units (s={small_dot_area:.1f}); bin markers are "
        f"drawn at radius {BIN_MARKER_ACCURACY_RADIUS} Accuracy units (s={bin_marker_area:.1f})."
    )

    for estimator_name in a2_estimators:
        group = transform_tradeoff[transform_tradeoff["Estimator"] == estimator_name]
        binned = (
            group.groupby("LengthBin", observed=True)
            .agg(SeriesLength=("SeriesLength", "mean"), Accuracy=("Accuracy", "mean"))
            .reindex(LENGTH_BIN_LABELS)
        )
        ax.plot(
            binned["SeriesLength"],
            binned["Accuracy"],
            color=colors[estimator_name],
            linewidth=1.2,
            zorder=4,
        )
        ax.scatter(
            binned["SeriesLength"],
            binned["Accuracy"],
            s=bin_marker_area,
            marker="o",
            color=colors[estimator_name],
            edgecolor="black",
            linewidth=1.0,
            zorder=5,
        )
    fig_path = os.path.join(out_dir, "accuracy_vs_series_length.pdf")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved {fig_path!r}")

    fig, ax = plt.subplots(figsize=(A2_FIGSIZE_WIDTH, A2_FIGSIZE_HEIGHT))
    speedup_color = colors[MOMENT_QUANT_APPROX]
    ax.scatter(
        speedup_tradeoff["SeriesLength"],
        speedup_tradeoff["RuntimeSpeedup"],
        alpha=0.5,
        s=18,
        edgecolors="none",
        color=speedup_color,
    )
    binned_speedup = (
        speedup_tradeoff.groupby("LengthBin", observed=True)
        .agg(SeriesLength=("SeriesLength", "mean"), RuntimeSpeedup=("RuntimeSpeedup", "mean"))
        .reindex(LENGTH_BIN_LABELS)
    )
    ax.plot(
        binned_speedup["SeriesLength"],
        binned_speedup["RuntimeSpeedup"],
        color=speedup_color,
        linewidth=1.2,
        zorder=4,
    )
    ax.scatter(
        binned_speedup["SeriesLength"],
        binned_speedup["RuntimeSpeedup"],
        s=140,
        marker="o",
        color=speedup_color,
        edgecolor="black",
        linewidth=1.0,
        zorder=5,
    )
    ax.axhline(1, color="black", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    candidate_yticks = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4]
    y_lo, y_hi = ax.get_ylim()
    yticks = [t for t in candidate_yticks if y_lo <= t <= y_hi]
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(t) for t in yticks])
    ax.minorticks_off()
    ax.set_xlabel("Series length (in time points)")
    ax.set_ylabel("End-to-end runtime speedup (exact / approx)")
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "speedup_vs_series_length.pdf")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved {fig_path!r}")

    pivot_acc = transform_tradeoff.pivot(index="Dataset", columns="Estimator", values="Accuracy")
    pivot_rt = transform_tradeoff.pivot(index="Dataset", columns="Estimator", values="TransformRuntime")
    speedup = pivot_rt[MOMENT_QUANT_EXACT] / pivot_rt[MOMENT_QUANT_APPROX]
    delta = pivot_acc[MOMENT_QUANT_APPROX] - pivot_acc[MOMENT_QUANT_EXACT]

    log_speedup = speedup.map(math.log)
    valid = log_speedup.notna() & delta.notna()
    speedup_delta_corr = log_speedup[valid].corr(delta[valid])
    print(
        f"approx_vs_exact_dataset_tradeoff.pdf: Pearson correlation between log(speedup) and "
        f"accuracy delta = {speedup_delta_corr:.4f} (n={int(valid.sum())} datasets) -- not shown "
        "on the figure."
    )

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(speedup, delta, alpha=0.6, s=22, color="#4C72B0")
    ax.axvline(1, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    xticks = [0.25, 0.5, 1, 2, 4, 8]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in xticks])
    ax.minorticks_off()
    ax.set_xlabel("Transform-runtime speedup (exact / approx)")
    ax.set_ylabel("Accuracy delta (approx - exact)")

    quadrant_counts = {
        "faster, more accurate": int(((speedup > 1) & (delta > 0)).sum()),
        "slower, more accurate": int(((speedup <= 1) & (delta > 0)).sum()),
        "faster, less accurate": int(((speedup > 1) & (delta <= 0)).sum()),
        "slower, less accurate": int(((speedup <= 1) & (delta <= 0)).sum()),
    }
    corner_positions = {
        "faster, more accurate": (0.98, 0.98, "right", "top"),
        "slower, more accurate": (0.02, 0.98, "left", "top"),
        "faster, less accurate": (0.98, 0.02, "right", "bottom"),
        "slower, less accurate": (0.02, 0.02, "left", "bottom"),
    }
    for label, count in quadrant_counts.items():
        x, y, ha, va = corner_positions[label]
        ax.text(
            x,
            y,
            f"{label}\n(n={count})",
            transform=ax.transAxes,
            ha=ha,
            va=va,
            fontsize=8,
            color="dimgray",
        )
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "approx_vs_exact_dataset_tradeoff.pdf")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved {fig_path!r}")


def main():
    if not os.path.isfile(metrics_csv_path):
        raise FileNotFoundError(
            f"{metrics_csv_path!r} not found -- run single_thread_ucr142_estimator_comparison.py "
            "first (stage 16 in run_pipeline.sh)."
        )
    if not os.path.isfile(transform_runtimes_csv_path):
        raise FileNotFoundError(
            f"{transform_runtimes_csv_path!r} not found -- run single_thread_ucr142.py first "
            "(stage 08 in run_pipeline.sh; see module docstring for why figures (A)/(B) need this "
            "transform-only runtime source in addition to metrics.csv)."
        )
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(metrics_csv_path)

    dataset_names = sorted(df["Dataset"].unique())
    lengths = load_series_lengths(dataset_names)
    df["SeriesLength"] = df["Dataset"].map(lengths)

    ds_lengths = df.drop_duplicates("Dataset")["SeriesLength"].to_numpy()
    edges = make_length_bin_edges(ds_lengths)
    df["LengthBin"] = pd.cut(df["SeriesLength"], bins=edges, labels=LENGTH_BIN_LABELS, include_lowest=True)
    dataset_length_map = df.drop_duplicates("Dataset").set_index("Dataset")["SeriesLength"]

    summary = (
        df.groupby(["Estimator", "LengthBin"], observed=True)
        .agg(
            MeanAccuracy=("Accuracy", "mean"),
            MedianRuntime=("Runtime", "median"),
            N=("Accuracy", "size"),
            NDatasets=("Dataset", "nunique"),
        )
        .reset_index()
    )
    summary_path = os.path.join(out_dir, "summary_by_length_bin.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved {len(summary)} rows to {summary_path!r}")

    per_dataset = (
        df.groupby(["Dataset", "Estimator", "LengthBin"], observed=True)
        .agg(Accuracy=("Accuracy", "mean"), Runtime=("Runtime", "median"))
        .reset_index()
    )
    pivot_acc = per_dataset.pivot(index=["Dataset", "LengthBin"], columns="Estimator", values="Accuracy")
    pivot_run = per_dataset.pivot(index=["Dataset", "LengthBin"], columns="Estimator", values="Runtime")

    tradeoff_rows = []
    approx_vs_exact_per_dataset = None
    for comparison_name, num_est, den_est in COMPARISONS:
        accuracy_delta = pivot_acc[num_est] - pivot_acc[den_est]
        runtime_speedup = pivot_run[den_est] / pivot_run[num_est]
        per_dataset_tradeoff = pd.DataFrame(
            {
                "AccuracyDelta": accuracy_delta,
                "RuntimeSpeedup": runtime_speedup,
            }
        ).reset_index()
        per_dataset_tradeoff["Comparison"] = comparison_name
        if comparison_name == "approx_vs_exact":
            approx_vs_exact_per_dataset = per_dataset_tradeoff.copy()

        agg = (
            per_dataset_tradeoff.groupby("LengthBin", observed=True)
            .agg(
                MeanAccuracyDelta=("AccuracyDelta", "mean"),
                MeanRuntimeSpeedup=("RuntimeSpeedup", "mean"),
                NDatasets=("Dataset", "nunique"),
            )
            .reset_index()
        )
        agg["Comparison"] = comparison_name
        tradeoff_rows.append(agg)

    tradeoff = pd.concat(tradeoff_rows, ignore_index=True)
    tradeoff_path = os.path.join(out_dir, "tradeoff_by_length_bin.csv")
    tradeoff.to_csv(tradeoff_path, index=False)
    print(f"Saved {len(tradeoff)} rows to {tradeoff_path!r}")

    speedup_tradeoff = approx_vs_exact_per_dataset.copy()
    speedup_tradeoff["SeriesLength"] = speedup_tradeoff["Dataset"].map(dataset_length_map)
    speedup_tradeoff_path = os.path.join(out_dir, "speedup_tradeoff_by_dataset.csv")
    speedup_tradeoff.to_csv(speedup_tradeoff_path, index=False)
    print(f"Saved {len(speedup_tradeoff)} rows to {speedup_tradeoff_path!r}")

    print()
    print("=" * 100)
    print("Accuracy-runtime tradeoff by series-length bin")
    print(
        "(Comparison = numerator_vs_denominator; AccuracyDelta = numerator - denominator; "
        "RuntimeSpeedup = denominator / numerator, i.e. > 1 means numerator is faster)"
    )
    print("=" * 100)
    print(
        tradeoff.set_index(["Comparison", "LengthBin"])[
            ["MeanAccuracyDelta", "MeanRuntimeSpeedup", "NDatasets"]
        ].to_string()
    )

    mean_accuracy = df.groupby(["Dataset", "Estimator"])["Accuracy"].mean().reset_index()
    dataset_length_bin = df.drop_duplicates("Dataset").set_index("Dataset")["LengthBin"]
    dataset_length = df.drop_duplicates("Dataset").set_index("Dataset")["SeriesLength"]
    mean_accuracy["LengthBin"] = mean_accuracy["Dataset"].map(dataset_length_bin)
    mean_accuracy["SeriesLength"] = mean_accuracy["Dataset"].map(dataset_length)
    transform_runtimes = load_transform_runtimes()
    transform_tradeoff = mean_accuracy.merge(transform_runtimes, on=["Dataset", "Estimator"], how="inner")
    missing = set(TRANSFORM_RUNTIME_ESTIMATOR_MAP.values()) - set(transform_tradeoff["Estimator"].unique())
    if missing:
        raise ValueError(
            f"transform_tradeoff is missing estimator(s) {missing!r} after merging metrics.csv "
            f"with {transform_runtimes_csv_path!r} -- check TRANSFORM_RUNTIME_ESTIMATOR_MAP against "
            "both CSVs' actual Estimator values."
        )
    transform_tradeoff_path = os.path.join(out_dir, "transform_tradeoff.csv")
    transform_tradeoff.to_csv(transform_tradeoff_path, index=False)
    print(f"Saved {len(transform_tradeoff)} rows to {transform_tradeoff_path!r}")

    make_figures(df, tradeoff, transform_tradeoff, speedup_tradeoff)


if __name__ == "__main__":
    main()
