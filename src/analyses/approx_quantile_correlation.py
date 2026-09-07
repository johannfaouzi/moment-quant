import os

import numpy as np
import pandas as pd
from aeon.datasets import load_classification

from src.estimators.moment_quant import MomentQuantTransformer
from src.utils import DEPTH, DIV, VERBOSE

DATASET_NAMES = [
    "ItalyPowerDemand",
    "ECG200",
    "GunPoint",
    "Wafer",
    "Yoga",
    "StarLightCurves",
]

REPR_NAMES = ["raw", "diff1", "diff2", "fft_magnitude"]

out_dir = os.path.join("results", "analyses", "approx_quantile_correlation")

FIGSIZE_WIDTH = 5.147
TITLE_FONTSIZE = 7
LABEL_FONTSIZE = 7
TICK_LABELSIZE = 6


def genuinely_approximated_mask(kind, offsets):
    total = int(offsets[-1])
    mask = np.zeros(total, dtype=bool)
    n_intervals = len(offsets) - 1
    for i in range(n_intervals):
        o0, o1 = int(offsets[i]), int(offsets[i + 1])
        k = kind[i]
        if k == 1:
            mask[o0:o1] = True
        elif k == 2 and (o1 - o0) > 2:
            mask[o0 + 1 : o1 - 1] = True
    return mask


def pearson_correlation_per_column(true_mat, approx_mat):
    true_c = true_mat - true_mat.mean(axis=0, keepdims=True)
    approx_c = approx_mat - approx_mat.mean(axis=0, keepdims=True)
    num = (true_c * approx_c).sum(axis=0)
    denom = np.sqrt((true_c**2).sum(axis=0) * (approx_c**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = num / denom
    return corr


def process_dataset(name):
    X_train, y_train = load_classification(name, split="train")
    X_test, y_test = load_classification(name, split="test")
    X = np.ascontiguousarray(np.concatenate([X_train, X_test], axis=0), dtype=np.float64)

    n_samples, n_channels, length = X.shape
    if n_channels != 1:
        raise ValueError(f"{name!r}: expected a univariate dataset, got {n_channels} channels.")

    exact_est = MomentQuantTransformer(depth=DEPTH, div=DIV, mode="exact", exact_version="intervals")
    approx_est = MomentQuantTransformer(depth=DEPTH, div=DIV, mode="approx", approx_version="intervals")
    Q_true = exact_est.fit_transform(X)
    Q_approx = approx_est.fit_transform(X)

    if VERBOSE:
        print(f"{name}: n_samples={n_samples}, length={length}, " f"n_output_features={exact_est.n_output_features_}")

    rows = []
    offset = 0
    for repr_name, layout in zip(REPR_NAMES, exact_est.layouts_):
        kind, offsets = layout["kind"], layout["offsets"]
        n_quantiles = np.diff(offsets)
        depth_by_column = np.repeat(layout["depth"], n_quantiles)
        approx_mask = genuinely_approximated_mask(kind, offsets)

        width = int(offsets[-1])
        cols = np.flatnonzero(approx_mask)
        if len(cols) > 0:
            true_sub = Q_true[:, offset : offset + width][:, cols]
            approx_sub = Q_approx[:, offset : offset + width][:, cols]
            corr = pearson_correlation_per_column(true_sub, approx_sub)

            rows.append(
                pd.DataFrame(
                    {
                        "Dataset": name,
                        "SeriesLength": length,
                        "Representation": repr_name,
                        "Depth": depth_by_column[cols],
                        "Column": cols,
                        "Correlation": corr,
                    }
                )
            )
        offset += width

    return (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["Dataset", "SeriesLength", "Representation", "Depth", "Column", "Correlation"])
    )


def make_figures(result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cols = min(DEPTH, 2)
    n_rows = int(np.ceil(DEPTH / n_cols))
    figsize = (FIGSIZE_WIDTH, FIGSIZE_WIDTH * (4 * n_rows) / (5 * n_cols))

    for name, group in result.groupby("Dataset"):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        for d in range(DEPTH):
            ax = axes.flat[d]
            sub = group.loc[group["Depth"] == d, "Correlation"].dropna()
            if len(sub) == 0:
                ax.set_title(f"Depth level {d} (no data)", fontsize=TITLE_FONTSIZE)
                ax.set_xlim(-1, 1)
                ax.tick_params(labelsize=TICK_LABELSIZE)
                continue
            ax.hist(sub, bins=30, range=(-1, 1))
            ax.set_title(f"Depth level {d} (n={len(sub)}, median={sub.median():.4f})", fontsize=TITLE_FONTSIZE)
            ax.set_xlabel("Pearson correlation", fontsize=LABEL_FONTSIZE)
            ax.tick_params(labelsize=TICK_LABELSIZE)
        for extra in range(DEPTH, n_rows * n_cols):
            axes.flat[extra].axis("off")
        fig.tight_layout()
        fig_path = os.path.join(out_dir, f"{name}_correlation_histograms.pdf")
        fig.savefig(fig_path)
        plt.close(fig)
        print(f"Saved {fig_path!r}")


def main():
    os.makedirs(out_dir, exist_ok=True)
    all_rows = [process_dataset(name) for name in DATASET_NAMES]
    result = pd.concat(all_rows, ignore_index=True)

    csv_path = os.path.join(out_dir, "correlations.csv")
    result.to_csv(csv_path, index=False)
    n_nan = int(result["Correlation"].isna().sum())
    print(f"Saved {len(result)} rows ({n_nan} NaN, degenerate zero-variance columns) to {csv_path!r}")

    make_figures(result)


if __name__ == "__main__":
    main()
