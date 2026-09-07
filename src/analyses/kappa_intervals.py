import os

import numpy as np
import pandas as pd

from src.estimators.moment_quant import MomentQuantTransformer
from src.utils import APPROX_INTERVALS, DEPTH, DIV, EXACT_INTERVALS

estimators = [EXACT_INTERVALS, APPROX_INTERVALS]

csv_path = os.path.join("results", "runtimes", "single_thread_grid", "runtimes.csv")
out_dir = os.path.join("results", "analyses", "single_thread_grid")


def n_i_per_representation(depth, div, length):
    est = MomentQuantTransformer(depth=depth, div=div, mode="exact", exact_version="intervals")
    est.fit(np.zeros((1, 1, length), dtype=np.float64))
    counts = [int(np.sum(layout["kind"] != 0)) for layout in est.layouts_]
    lengths = est._representation_lengths()
    return counts, lengths


def n_i_total(depth, div, length):
    counts, _ = n_i_per_representation(depth, div, length)
    return sum(counts)


def n_i_over_l_total(depth, div, length):
    counts, lengths = n_i_per_representation(depth, div, length)
    return sum(c / l for c, l in zip(counts, lengths))


def fit_intercept_slope(n, t):
    n = np.asarray(n, dtype=float)
    t = np.asarray(t, dtype=float)
    x = 1.0 / n
    y = t / n
    intercept, slope = np.polyfit(x, y, 1)
    pred = intercept + slope * n
    ss_res = np.sum((t - pred) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return intercept, slope, r2


def fit_kappa_curve(intercepts_by_L, x1_by_L, x2_by_L):
    L_values = sorted(intercepts_by_L)
    y = np.array([intercepts_by_L[L] for L in L_values], dtype=float)
    x1 = np.array([x1_by_L[L] for L in L_values], dtype=float)
    x2 = np.array([x2_by_L[L] for L in L_values], dtype=float)
    design = np.column_stack([x1, -x2])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    a, b = (float(c) for c in coeffs)
    pred = design @ coeffs
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return a, b, float(r2)


def main():
    df = pd.read_csv(csv_path)
    df = df[df["Estimator"].isin(estimators)]

    mins = (
        df.groupby(["Series length", "Number of samples", "Estimator"])["Runtime"]
        .min()
        .reset_index()
        .rename(columns={"Series length": "L", "Number of samples": "n", "Runtime": "t_min"})
    )
    L_values = sorted(mins["L"].unique())
    x1_by_L = {L: n_i_total(DEPTH, DIV, L) for L in L_values}
    x2_by_L = {L: n_i_over_l_total(DEPTH, DIV, L) for L in L_values}

    diagnostic_rows = []
    fit_rows = []
    for estimator in estimators:
        intercepts_by_L = {}
        for L in L_values:
            sub = mins[(mins["L"] == L) & (mins["Estimator"] == estimator)].sort_values("n")
            if len(sub) < 2:
                continue
            intercept, slope, r2 = fit_intercept_slope(sub["n"].values, sub["t_min"].values)
            intercepts_by_L[L] = intercept
            diagnostic_rows.append(
                dict(
                    Estimator=estimator,
                    L=L,
                    N_i=x1_by_L[L],
                    intercept=intercept,
                    slope=slope,
                    r2_stage1=r2,
                    kappa_hat_naive=intercept / x1_by_L[L],
                )
            )

        if len(intercepts_by_L) < 2:
            raise ValueError(
                f"Need at least 2 usable series lengths to fit kappa^(I)(l) = A - B/l for "
                f"estimator {estimator!r}, got {len(intercepts_by_L)}."
            )
        a, b, r2_curve = fit_kappa_curve(intercepts_by_L, x1_by_L, x2_by_L)
        fit_rows.append(dict(Estimator=estimator, A=a, B=b, r2_curve=r2_curve, n_grid_points=len(intercepts_by_L)))

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(fit_rows).to_csv(os.path.join(out_dir, "kappa_intervals.csv"), index=False)
    pd.DataFrame(diagnostic_rows).to_csv(os.path.join(out_dir, "kappa_intervals_diagnostics.csv"), index=False)


if __name__ == "__main__":
    main()
