import os

import numpy as np
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
out_dir = os.path.join("results", "analyses", "pipeline_decomposition")

STAGE_COLUMNS = {
    "Transformation (training)": "Runtime_FeatureExtractionTrain",
    "Classification (training)": "Runtime_ClassifierFit",
    "Transformation (inference)": "Runtime_FeatureExtractionTest",
    "Classification (inference)": "Runtime_Prediction",
}
STAGES = list(STAGE_COLUMNS)
FEATURE_EXTRACTION_STAGES = ["Transformation (training)", "Transformation (inference)"]
INFERENCE_STAGES = ["Transformation (inference)", "Classification (inference)"]
ESTIMATOR_ORDER = [MOMENT_QUANT_APPROX, MOMENT_QUANT_EXACT, QUANT]


def make_figures(summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stage_colors = dict(zip(STAGES, ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]))

    def plot_stage_breakdown(stages, ylabel, fig_filename):
        scoped = summary[summary["Stage"].isin(stages)].copy()
        stage_total = scoped.groupby(["Dataset", "Estimator"])["MeanTime"].transform("sum")
        scoped["Pct"] = 100 * scoped["MeanTime"] / stage_total

        bin_level = scoped.groupby(["Estimator", "LengthBin", "Stage"], observed=True)["Pct"].mean().reset_index()
        estimators_present = [e for e in ESTIMATOR_ORDER if e in bin_level["Estimator"].unique()]
        n_bins = len(LENGTH_BIN_LABELS)
        n_est = len(estimators_present)
        bar_width = 0.25
        cluster_centers = np.arange(n_bins, dtype=float)

        fig, ax = plt.subplots(figsize=(5.147, 5.5))
        for est_idx, estimator_name in enumerate(estimators_present):
            pivot = (
                bin_level[bin_level["Estimator"] == estimator_name]
                .pivot(index="LengthBin", columns="Stage", values="Pct")
                .reindex(LENGTH_BIN_LABELS)
            )
            x_positions = cluster_centers + (est_idx - (n_est - 1) / 2) * bar_width
            bottoms = np.zeros(n_bins)
            for stage in stages:
                values = pivot[stage].fillna(0).to_numpy()
                ax.bar(
                    x_positions,
                    values,
                    width=bar_width * 0.92,
                    bottom=bottoms,
                    color=stage_colors[stage],
                    label=stage if est_idx == 0 else None,
                    edgecolor="white",
                    linewidth=0.4,
                )
                bottoms += values

        all_x, all_tags = [], []
        for c in cluster_centers:
            for est_idx, estimator_name in enumerate(estimators_present):
                all_x.append(c + (est_idx - (n_est - 1) / 2) * bar_width)
                all_tags.append(format_estimator_label(estimator_name))
        ax.set_xticks(all_x)
        ax.set_xticklabels(all_tags, fontsize=8, rotation=90)
        ax.tick_params(axis="y", labelsize=8)
        for c, length_bin in zip(cluster_centers, LENGTH_BIN_LABELS):
            ax.annotate(
                str(length_bin),
                xy=(c, 0),
                xycoords=("data", "axes fraction"),
                xytext=(0, -125),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8,
                fontweight="bold",
            )

        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_xlim(cluster_centers[0] - 0.6, cluster_centers[-1] + 0.6)
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.2),
            mode="expand",
            ncol=min(len(stages), 2),
            fontsize=8,
            frameon=True,
        )
        fig.subplots_adjust(bottom=0.30, top=0.97, left=0.10)
        fig_path = os.path.join(out_dir, fig_filename)
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fig_path!r}")

    plot_stage_breakdown(STAGES, "Mean % of total pipeline time", "stage_breakdown.pdf")

    plot_stage_breakdown(INFERENCE_STAGES, "Mean % of total inference time", "stage_breakdown_inference.pdf")

    fe_share = (
        summary[summary["Stage"].isin(FEATURE_EXTRACTION_STAGES)]
        .groupby(["Estimator", "Dataset", "SeriesLength"], observed=True)["PctOfTotal"]
        .sum()
        .reset_index(name="FeatureExtractionPct")
    )
    fig, ax = plt.subplots(figsize=(6, 7))
    colors = dict(zip(sorted(fe_share["Estimator"].unique()), ["#4C72B0", "#DD8452", "#55A868"]))
    for estimator_name, group in fe_share.groupby("Estimator"):
        group = group.sort_values("SeriesLength")
        ax.scatter(
            group["SeriesLength"],
            group["FeatureExtractionPct"],
            alpha=0.4,
            s=15,
            color=colors[estimator_name],
            label=None,
        )
        edges = make_length_bin_edges(group["SeriesLength"].to_numpy())
        binned = pd.cut(group["SeriesLength"], bins=edges, labels=LENGTH_BIN_LABELS, include_lowest=True)
        binned.name = "_LengthBinForTrend"
        trend = (
            group.groupby(binned, observed=True)
            .agg(
                SeriesLength=("SeriesLength", "median"),
                FeatureExtractionPct=("FeatureExtractionPct", "mean"),
            )
            .reset_index(drop=True)
            .sort_values("SeriesLength")
        )
        ax.plot(
            trend["SeriesLength"],
            trend["FeatureExtractionPct"],
            marker="o",
            linewidth=2,
            color=colors[estimator_name],
            label=f"{format_estimator_label(estimator_name)}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Series length (log scale)")
    ax.set_ylabel("Transformation's share of the pipeline time (%)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "feature_extraction_share_vs_length.pdf")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved {fig_path!r}")


def main():
    if not os.path.isfile(metrics_csv_path):
        raise FileNotFoundError(
            f"{metrics_csv_path!r} not found -- run single_thread_ucr142_estimator_comparison.py "
            "first (stage 16 in run_pipeline.sh)."
        )
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(metrics_csv_path)

    dataset_names = sorted(df["Dataset"].unique())
    lengths = load_series_lengths(dataset_names)
    df["SeriesLength"] = df["Dataset"].map(lengths)

    ds_lengths = df.drop_duplicates("Dataset")["SeriesLength"].to_numpy()
    edges = make_length_bin_edges(ds_lengths)
    df["LengthBin"] = pd.cut(df["SeriesLength"], bins=edges, labels=LENGTH_BIN_LABELS, include_lowest=True)

    long_df = df.melt(
        id_vars=["Dataset", "Resample", "Estimator", "SeriesLength", "LengthBin"],
        value_vars=list(STAGE_COLUMNS.values()),
        var_name="StageColumn",
        value_name="Time",
    )
    column_to_stage = {v: k for k, v in STAGE_COLUMNS.items()}
    long_df["Stage"] = long_df["StageColumn"].map(column_to_stage)

    mean_time = (
        long_df.groupby(["Dataset", "SeriesLength", "LengthBin", "Estimator", "Stage"], observed=True)["Time"]
        .mean()
        .reset_index(name="MeanTime")
    )
    total_time = mean_time.groupby(["Dataset", "Estimator"])["MeanTime"].transform("sum")
    mean_time["PctOfTotal"] = 100 * mean_time["MeanTime"] / total_time

    summary_path = os.path.join(out_dir, "summary.csv")
    mean_time.to_csv(summary_path, index=False)
    print(f"Saved {len(mean_time)} rows to {summary_path!r}")

    overall = mean_time.groupby(["Estimator", "Stage"])["PctOfTotal"].mean().unstack("Stage")[STAGES]
    print()
    print("=" * 100)
    print("Mean % of total pipeline time per stage, averaged (unweighted) across datasets")
    print("(all 142 datasets, resample-averaged; see summary.csv for the per-dataset breakdown)")
    print("=" * 100)
    print(overall.round(1).to_string())

    by_bin = (
        mean_time.groupby(["Estimator", "LengthBin", "Stage"], observed=True)["PctOfTotal"]
        .mean()
        .unstack("Stage")[STAGES]
    )
    print()
    print("=" * 100)
    print(r"Mean % of total pipeline time per stage, by series-length bin")
    print("=" * 100)
    print(by_bin.round(1).to_string())

    make_figures(mean_time)


if __name__ == "__main__":
    main()
