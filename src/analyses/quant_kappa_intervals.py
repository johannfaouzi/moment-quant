import os

import pandas as pd

from src.analyses.kappa_intervals import fit_intercept_slope, fit_kappa_curve, n_i_over_l_total, n_i_total
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
    x1_by_L = {L: n_i_total(DEPTH, DIV, L) for L in L_values}
    x2_by_L = {L: n_i_over_l_total(DEPTH, DIV, L) for L in L_values}

    diagnostic_rows = []
    intercepts_by_L = {}
    for L in L_values:
        sub = mins[mins["L"] == L].sort_values("n")
        if len(sub) < 2:
            continue
        intercept, slope, r2 = fit_intercept_slope(sub["n"].values, sub["t_min"].values)
        intercepts_by_L[L] = intercept
        diagnostic_rows.append(
            dict(
                Estimator=ESTIMATOR,
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
            f"estimator {ESTIMATOR!r}, got {len(intercepts_by_L)}."
        )
    a, b, r2_curve = fit_kappa_curve(intercepts_by_L, x1_by_L, x2_by_L)
    fit_row = dict(Estimator=ESTIMATOR, A=a, B=b, r2_curve=r2_curve, n_grid_points=len(intercepts_by_L))

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([fit_row]).to_csv(os.path.join(out_dir, "kappa_intervals.csv"), index=False)
    pd.DataFrame(diagnostic_rows).to_csv(os.path.join(out_dir, "kappa_intervals_diagnostics.csv"), index=False)
    print(
        f"Saved kappa_intervals.csv (A={a!r}, B={b!r}, r2_curve={r2_curve!r}) and "
        f"kappa_intervals_diagnostics.csv to {out_dir!r}"
    )


if __name__ == "__main__":
    main()
