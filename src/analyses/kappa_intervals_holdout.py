import os
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["KMP_WARNINGS"] = "FALSE"

import numpy as np
import pandas as pd

from src.analyses.kappa_intervals import fit_intercept_slope, n_i_total
from src.analyses.utils import HOLDOUT_L
from src.estimators.moment_quant import MomentQuantTransformer
from src.theory.cost_formulae import load_kappa_I
from src.utils import APPROX_INTERVALS, DEPTH, DIV, EXACT_INTERVALS, N_RUNS, N_SAMPLES_GRID, VERBOSE

estimators = [EXACT_INTERVALS, APPROX_INTERVALS]
mode_by_estimator = {
    EXACT_INTERVALS: "exact",
    APPROX_INTERVALS: "approx",
}

assert all(15 <= l <= 8192 for l in HOLDOUT_L)
assert not (set(HOLDOUT_L) & {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192})

csv_path = os.path.join("results", "analyses", "single_thread_grid", "kappa_intervals.csv")
out_dir = os.path.join("results", "analyses", "kappa_intervals_holdout")


def make_estimators():
    return {
        EXACT_INTERVALS: MomentQuantTransformer(
            depth=DEPTH, div=DIV, mode="exact", exact_version="intervals", parallel=False
        ),
        APPROX_INTERVALS: MomentQuantTransformer(
            depth=DEPTH, div=DIV, mode="approx", approx_version="intervals", parallel=False
        ),
    }


def benchmark_holdout_grid():
    seed = 0
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(max(N_SAMPLES_GRID), 1, max(HOLDOUT_L)))

    estimators = make_estimators()

    warmup_L, warmup_n = min(HOLDOUT_L), min(N_SAMPLES_GRID)
    X_warmup = np.ascontiguousarray(X[:warmup_n, :, :warmup_L])
    for est in estimators.values():
        est.fit(X_warmup)
        est.transform(X_warmup)

    results = []
    for L in HOLDOUT_L:
        X_fit = np.ascontiguousarray(X[:1, :, :L])
        for est in estimators.values():
            est.fit(X_fit)

        for n_samples in N_SAMPLES_GRID:
            if VERBOSE:
                print(f"[holdout] Series length = {L} | Number of samples = {n_samples}")
            X_ = np.ascontiguousarray(X[:n_samples, :, :L])

            jobs = [(name, run) for name in estimators for run in range(N_RUNS)]
            rng.shuffle(jobs)
            for name, run in jobs:
                start_time = time.perf_counter()
                estimators[name].transform(X_)
                end_time = time.perf_counter()
                results.append([L, n_samples, name, run, end_time - start_time])

    return pd.DataFrame(results, columns=["Series length", "Number of samples", "Estimator", "Run", "Runtime"])


def main():
    runtimes = benchmark_holdout_grid()

    os.makedirs(out_dir, exist_ok=True)
    runtimes.to_csv(os.path.join(out_dir, "runtimes.csv"), index=False)

    mins = (
        runtimes.groupby(["Series length", "Number of samples", "Estimator"])["Runtime"]
        .min()
        .reset_index()
        .rename(columns={"Series length": "L", "Number of samples": "n", "Runtime": "t_min"})
    )

    comparison_rows = []
    for estimator in estimators:
        mode = mode_by_estimator[estimator]
        kappa_I = load_kappa_I(mode)
        for L in HOLDOUT_L:
            sub = mins[(mins["L"] == L) & (mins["Estimator"] == estimator)].sort_values("n")
            intercept, slope, r2_stage1 = fit_intercept_slope(sub["n"].values, sub["t_min"].values)
            x1 = n_i_total(DEPTH, DIV, L)
            kappa_hat_naive = intercept / x1

            try:
                kappa_predicted = kappa_I(L)
                prediction_error_msg = ""
            except ValueError as exc:
                kappa_predicted = float("nan")
                prediction_error_msg = str(exc)

            if np.isfinite(kappa_predicted):
                abs_error = kappa_predicted - kappa_hat_naive
                rel_error = abs_error / kappa_hat_naive if kappa_hat_naive != 0 else float("nan")
            else:
                abs_error = float("nan")
                rel_error = float("nan")

            comparison_rows.append(
                dict(
                    Estimator=estimator,
                    Mode=mode,
                    L=L,
                    intercept=intercept,
                    r2_stage1=r2_stage1,
                    N_i=x1,
                    kappa_hat_naive=kappa_hat_naive,
                    kappa_predicted=kappa_predicted,
                    absolute_error=abs_error,
                    relative_error=rel_error,
                    prediction_failed=not np.isfinite(kappa_predicted),
                    prediction_error=prediction_error_msg,
                )
            )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(os.path.join(out_dir, "kappa_intervals_holdout_comparison.csv"), index=False)

    summary_rows = []
    for estimator in estimators:
        mode = mode_by_estimator[estimator]
        sub = comparison[comparison["Estimator"] == estimator]
        valid = sub[~sub["prediction_failed"]]
        n_failed = int(sub["prediction_failed"].sum())
        if len(valid) >= 2:
            y_true = valid["kappa_hat_naive"].values
            y_pred = valid["kappa_predicted"].values
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2_holdout = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            mape = float(np.mean(np.abs(valid["relative_error"])) * 100.0)
            max_ape = float(np.max(np.abs(valid["relative_error"])) * 100.0)
        else:
            r2_holdout = float("nan")
            mape = float("nan")
            max_ape = float("nan")
        summary_rows.append(
            dict(
                Estimator=estimator,
                Mode=mode,
                n_holdout_points=len(sub),
                n_predictions_failed=n_failed,
                failed_L_values=",".join(str(l) for l in sub.loc[sub["prediction_failed"], "L"]),
                r2_holdout=r2_holdout,
                mean_abs_pct_error=mape,
                max_abs_pct_error=max_ape,
            )
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(out_dir, "kappa_intervals_holdout_summary.csv"), index=False)

    if VERBOSE:
        print("=" * 100)
        print("kappa^(I)(l) held-out validation (true out-of-sample, unlike kappa_intervals.csv's r2_curve)")
        print("=" * 100)
        print(comparison.to_string(index=False))
        print()
        print(summary.to_string(index=False))
        print()
        print(f"Saved {len(comparison)} comparison rows and {len(summary)} summary rows to {out_dir!r}")


if __name__ == "__main__":
    main()
