import os

import numpy as np
import pandas as pd

from src.estimators.moment_quant import MomentQuantTransformer
from src.utils import DEPTH, DIV, EXACT_INTERVALS, EXACT_SAMPLES

estimators = [EXACT_SAMPLES, EXACT_INTERVALS]

csv_path = os.path.join("results", "runtimes", "single_thread_grid", "runtimes.csv")
out_dir = os.path.join("results", "analyses", "single_thread_grid")


def basis_for_length(depth, div, length):
    est = MomentQuantTransformer(depth=depth, div=div, mode="exact", exact_version="intervals")
    est.fit(np.zeros((1, 1, length), dtype=np.float64))

    sortwork = nq = w = 0.0
    for layout in est.layouts_:
        starts, ends, kind, offsets = layout["starts"], layout["ends"], layout["kind"], layout["offsets"]
        for i in range(len(starts)):
            if kind[i] == 0:
                continue
            m = int(ends[i] - starts[i])
            sortwork += m * np.log2(m)
            nq += int(offsets[i + 1] - offsets[i])
            w += m
    return sortwork, nq, w


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

    basis = {L: basis_for_length(DEPTH, DIV, L) for L in L_values}
    print("=" * 100)
    print(f"BASIS FUNCTIONS (summed over all 4 representations, depth={DEPTH}, div={DIV})")
    print("=" * 100)
    print(f"{'L':>6} {'SortWork':>14} {'NQ':>10} {'W':>10} {'W/NQ':>8}")
    for L in L_values:
        sw, nq, w = basis[L]
        print(f"{L:6d} {sw:14.2f} {nq:10.0f} {w:10.0f} {w / nq:8.4f}")

    A = np.array([basis[L] for L in L_values])
    print()
    print(f"Design matrix condition number (SortWork, NQ, W): {np.linalg.cond(A):.4g}")
    print("Correlation matrix (SortWork, NQ, W):")
    print(np.corrcoef(A.T))
    print()
    print("A high condition number / near-1 off-diagonal correlations mean c_2 and c_3 are only")
    print("weakly separable from each other in the joint fit below (their sum is much more stable")
    print("than either individually) -- see the 'c2_plus_ratio_c3' row for a more robust combined")
    print("number, and treat the individual c_2/c_3 rows with appropriate caution.")

    rows = []
    for estimator in estimators:
        for L in L_values:
            sub = mins[(mins["L"] == L) & (mins["Estimator"] == estimator)].sort_values("n")
            if len(sub) < 2:
                continue
            intercept, slope, r2 = fit_intercept_slope(sub["n"].values, sub["t_min"].values)
            rows.append(dict(estimator=estimator, L=L, intercept=intercept, slope=slope, r2=r2))
    stage1 = pd.DataFrame(rows)

    results = []
    for estimator in estimators:
        sub = stage1[stage1["estimator"] == estimator].sort_values("L")
        A_est = np.array([basis[L] for L in sub["L"]])
        b_est = sub["slope"].values

        coeffs, _, _, _ = np.linalg.lstsq(A_est, b_est, rcond=None)
        c1, c2, c3 = coeffs
        pred = A_est @ coeffs
        ss_res = np.sum((b_est - pred) ** 2)
        ss_tot = np.sum((b_est - np.mean(b_est)) ** 2)
        r2_full = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        A_reduced = A_est[:, [0, 1]]
        coeffs_reduced, _, _, _ = np.linalg.lstsq(A_reduced, b_est, rcond=None)
        c1_robust, c23 = coeffs_reduced
        pred_reduced = A_reduced @ coeffs_reduced
        ss_res_reduced = np.sum((b_est - pred_reduced) ** 2)
        r2_reduced = 1.0 - ss_res_reduced / ss_tot if ss_tot > 0 else np.nan

        for name, value in [("c_1", c1), ("c_2", c2), ("c_3", c3)]:
            results.append(dict(Estimator=estimator, Constant=name, Value=value, R2=r2_full, Fit="joint_3param"))
        results.append(dict(Estimator=estimator, Constant="c_1", Value=c1_robust, R2=r2_reduced, Fit="robust_2param"))
        results.append(
            dict(Estimator=estimator, Constant="c2_plus_ratio_c3", Value=c23, R2=r2_reduced, Fit="robust_2param")
        )

    out = pd.DataFrame(results, columns=["Estimator", "Constant", "Value", "R2", "Fit"])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "constants.csv")
    out.to_csv(out_path, index=False)
    print()
    print(f"Saved {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
