import os

import numpy as np
import pandas as pd

from src.analyses.kappa_intervals import fit_kappa_curve
from src.analyses.moment_cf_constants import basis_for_length as approx_basis_for_length
from src.analyses.sort_extraction_constants import basis_for_length as exact_basis_for_length
from src.theory.cost_formulae import representation_length
from src.utils import (
    APPROX_SAMPLES,
    DEPTH,
    DIV,
    EXACT_SAMPLES,
)


runtimes_csv_path = os.path.join("results", "runtimes", "single_thread_grid", "runtimes.csv")
exact_constants_csv_path = os.path.join("results", "analyses", "single_thread_grid", "constants.csv")
microbench_moment_cf_csv_path = os.path.join("results", "runtimes", "microbench_moment_cf", "constants.csv")
out_dir = os.path.join("results", "analyses", "single_thread_grid")

modes = {
    "exact": dict(
        slope_estimator=EXACT_SAMPLES,
        constants_estimator=EXACT_SAMPLES,
        constants_csv=exact_constants_csv_path,
        constant_names=("c_1", "c_2", "c_3"),
        basis_fn=exact_basis_for_length,
        loader="joint_3param",
        curve_model="a_minus_b_over_l",
        assumption=True,
    ),
    "approx": dict(
        slope_estimator=APPROX_SAMPLES,
        constants_estimator=APPROX_SAMPLES,
        constants_csv=microbench_moment_cf_csv_path,
        constant_names=("c1_tilde", "c2_tilde", "c3_tilde"),
        basis_fn=approx_basis_for_length,
        loader="microbench",
        curve_model="power_law",
        assumption=False,
    ),
}


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


def load_processing_constants(constants_csv, estimator, constant_names):
    df = pd.read_csv(constants_csv)
    df = df[(df["Fit"] == "joint_3param") & (df["Estimator"] == estimator)]
    values = {}
    for name in constant_names:
        sub = df[df["Constant"] == name]
        if len(sub) != 1:
            raise ValueError(
                f"Expected exactly one 'joint_3param' row for Estimator={estimator!r}, "
                f"Constant={name!r} in {constants_csv!r}, found {len(sub)}."
            )
        values[name] = float(sub.iloc[0]["Value"])
    return tuple(values[name] for name in constant_names)


def load_approx_constants(csv_path, estimator):
    df = pd.read_csv(csv_path)
    df = df[df["Estimator"] == estimator]
    values = {}
    for name in ("c1_tilde", "c2_tilde", "c3_tilde"):
        sub = df[df["Constant"] == name]
        if len(sub) != 1:
            raise ValueError(
                f"Expected exactly one row for Estimator={estimator!r}, Constant={name!r} in "
                f"{csv_path!r}, found {len(sub)}. Has microbench_moment_cf.py been run?"
            )
        values[name] = float(sub.iloc[0]["Value"])
    return tuple(values[name] for name in ("c1_tilde", "c2_tilde", "c3_tilde"))


def fit_power_law_curve(residuals_by_L):
    L_values = sorted(residuals_by_L)
    L = np.array(L_values, dtype=float)
    r = np.array([residuals_by_L[Lv] for Lv in L_values], dtype=float)

    log_L = np.log(L)
    log_r = np.log(np.clip(r, 1e-300, None))
    p, log_C0 = np.polyfit(log_L, log_r, 1)

    design = np.column_stack([np.ones_like(L), L**p])
    coeffs, *_ = np.linalg.lstsq(design, r, rcond=None)
    a, c = (float(x) for x in coeffs)
    pred = design @ coeffs
    ss_res = np.sum((r - pred) ** 2)
    ss_tot = np.sum((r - np.mean(r)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return a, c, float(p), r2


def x2_over_l_total(l):
    return sum(1.0 / representation_length(l, p) for p in range(1, 5))


def main():
    runtimes = pd.read_csv(runtimes_csv_path)

    mins = (
        runtimes.groupby(["Series length", "Number of samples", "Estimator"])["Runtime"]
        .min()
        .reset_index()
        .rename(columns={"Series length": "L", "Number of samples": "n", "Runtime": "t_min"})
    )

    diagnostic_rows = []
    fit_rows = []
    for mode, spec in modes.items():
        slope_estimator = spec["slope_estimator"]
        constants_estimator = spec["constants_estimator"]
        if spec["loader"] == "microbench":
            c1, c2, c3 = load_approx_constants(spec["constants_csv"], constants_estimator)
        else:
            c1, c2, c3 = load_processing_constants(spec["constants_csv"], constants_estimator, spec["constant_names"])
        basis_fn = spec["basis_fn"]

        sub_all = mins[mins["Estimator"] == slope_estimator]
        L_values = sorted(sub_all["L"].unique())

        residuals_by_L = {}
        x2_by_L = {}
        for L in L_values:
            sub = sub_all[sub_all["L"] == L].sort_values("n")
            if len(sub) < 2:
                continue
            intercept, slope, r2_stage1 = fit_intercept_slope(sub["n"].values, sub["t_min"].values)
            predicted = sum(c * b for c, b in zip((c1, c2, c3), basis_fn(DEPTH, DIV, int(L))))
            residual = slope - predicted
            residuals_by_L[L] = residual
            x2_by_L[L] = x2_over_l_total(int(L))
            diagnostic_rows.append(
                dict(
                    Mode=mode,
                    SlopeEstimator=slope_estimator,
                    ConstantsEstimator=constants_estimator,
                    L=L,
                    slope=slope,
                    intercept=intercept,
                    r2_stage1=r2_stage1,
                    predicted_processing_cost=predicted,
                    residual=residual,
                    kappa_S_hat_naive=residual / 4.0,
                    assumption=spec["assumption"],
                )
            )

        if len(residuals_by_L) < 2:
            raise ValueError(
                f"Need at least 2 usable series lengths to fit kappa^(S)(l) = A - B/l for mode "
                f"{mode!r}, got {len(residuals_by_L)}."
            )
        curve_model = spec["curve_model"]
        if curve_model == "power_law":
            a, c, p, r2_curve = fit_power_law_curve(residuals_by_L)
            b = float("nan")
        else:
            x1_by_L = {L: 4.0 for L in residuals_by_L}
            a, b, r2_curve = fit_kappa_curve(residuals_by_L, x1_by_L, x2_by_L)
            c = p = float("nan")
        fit_rows.append(
            dict(
                Mode=mode,
                SlopeEstimator=slope_estimator,
                ConstantsEstimator=constants_estimator,
                CurveModel=curve_model,
                A=a,
                B=b,
                C=c,
                P=p,
                r2_curve=r2_curve,
                n_grid_points=len(residuals_by_L),
                assumption=spec["assumption"],
            )
        )

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(fit_rows).to_csv(os.path.join(out_dir, "kappa_series.csv"), index=False)
    pd.DataFrame(diagnostic_rows).to_csv(os.path.join(out_dir, "kappa_series_diagnostics.csv"), index=False)

    print("=" * 100)
    print("kappa^(S)(l) fit summary -- CurveModel column records which functional form was used")
    print("=" * 100)
    for row in fit_rows:
        flag = " (ASSUMPTION -- see module docstring)" if row["assumption"] else " (Theorem 32; B~=0 expected)"
        if row["CurveModel"] == "power_law":
            print(
                f"{row['Mode']:>6}: A + C*L^P model -- A={row['A']:.4e}  C={row['C']:.4e}  "
                f"P={row['P']:.4f}  r2_curve={row['r2_curve']:.4f}  "
                f"n_grid_points={row['n_grid_points']}){flag}"
            )
        else:
            print(
                f"{row['Mode']:>6}: A - B/l model -- A={row['A']:.4e}  B={row['B']:.4e}  "
                f"r2_curve={row['r2_curve']:.4f}  n_grid_points={row['n_grid_points']}){flag}"
            )
    print()
    print(f"Saved kappa_series.csv and kappa_series_diagnostics.csv to {out_dir!r}")


if __name__ == "__main__":
    main()
