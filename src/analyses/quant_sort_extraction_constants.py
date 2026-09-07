import os

import numpy as np
import pandas as pd

from src.analyses.sort_extraction_constants import basis_for_length, fit_intercept_slope
from src.utils import DEPTH, DIV

ESTIMATOR = "quant_float64"

csv_path = os.path.join("results", "runtimes", "single_thread_quant_grid", "runtimes.csv")
out_dir = os.path.join("results", "analyses", "single_thread_quant_grid")


def main():
    df = pd.read_csv(csv_path)
    df = df[df["Estimator"] == ESTIMATOR]
    if df.empty:
        raise ValueError(
            f"No rows for Estimator={ESTIMATOR!r} found in {csv_path!r}. Run "
            f"`python -m src.runtimes.single_thread_quant_grid` first."
        )

    mins = (
        df.groupby(["Series length", "Number of samples"])["Runtime"]
        .min()
        .reset_index()
        .rename(columns={"Series length": "L", "Number of samples": "n", "Runtime": "t_min"})
    )
    L_values = sorted(mins["L"].unique())

    basis = {L: basis_for_length(DEPTH, DIV, L) for L in L_values}
    print("=" * 100)
    print(f"BASIS FUNCTIONS [{ESTIMATOR}] (summed over all 4 representations, depth={DEPTH}, div={DIV})")
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
    print("weakly separable from each other in the joint fit below -- see the 'c2_plus_ratio_c3'")
    print("row for a more robust combined number, and treat the individual c_2/c_3 rows with")
    print("appropriate caution (see module docstring's fused-call caveat too).")

    rows = []
    for L in L_values:
        sub = mins[mins["L"] == L].sort_values("n")
        if len(sub) < 2:
            continue
        intercept, slope, r2 = fit_intercept_slope(sub["n"].values, sub["t_min"].values)
        rows.append(dict(L=L, intercept=intercept, slope=slope, r2=r2))
    stage1 = pd.DataFrame(rows)

    if len(stage1) < 3:
        raise ValueError(
            f"Need at least 3 usable series lengths to jointly fit (c_1, c_2, c_3), got " f"{len(stage1)}."
        )

    sub = stage1.sort_values("L")
    A_est = np.array([basis[L] for L in sub["L"]])
    b_est = sub["slope"].values

    coeffs, *_ = np.linalg.lstsq(A_est, b_est, rcond=None)
    c1, c2, c3 = coeffs
    pred = A_est @ coeffs
    ss_res = np.sum((b_est - pred) ** 2)
    ss_tot = np.sum((b_est - np.mean(b_est)) ** 2)
    r2_full = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    A_reduced = A_est[:, [0, 1]]
    coeffs_reduced, *_ = np.linalg.lstsq(A_reduced, b_est, rcond=None)
    c1_robust, c23 = coeffs_reduced
    pred_reduced = A_reduced @ coeffs_reduced
    ss_res_reduced = np.sum((b_est - pred_reduced) ** 2)
    r2_reduced = 1.0 - ss_res_reduced / ss_tot if ss_tot > 0 else np.nan

    results = []
    for name, value in [("c_1", c1), ("c_2", c2), ("c_3", c3)]:
        results.append(dict(Estimator=ESTIMATOR, Constant=name, Value=value, R2=r2_full, Fit="joint_3param"))
    results.append(dict(Estimator=ESTIMATOR, Constant="c_1", Value=c1_robust, R2=r2_reduced, Fit="robust_2param"))
    results.append(
        dict(Estimator=ESTIMATOR, Constant="c2_plus_ratio_c3", Value=c23, R2=r2_reduced, Fit="robust_2param")
    )

    out = pd.DataFrame(results, columns=["Estimator", "Constant", "Value", "R2", "Fit"])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "constants.csv")
    out.to_csv(out_path, index=False)
    print()
    print(f"Saved {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
