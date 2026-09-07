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

from src.analyses.kappa_intervals import fit_intercept_slope
from src.analyses.utils import HOLDOUT_L
from src.estimators.moment_quant import MomentQuantTransformer
from src.utils import DEPTH, DIV, N_RUNS, N_SAMPLES_GRID, VERBOSE

modes = ("exact", "approx")
variants = ("samples", "intervals")

assert all(15 <= l <= 8192 for l in HOLDOUT_L)
assert not (set(HOLDOUT_L) & {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192})

out_dir = os.path.join("results", "analyses", "dispatch_thresholds_holdout")


def make_estimators():
    return {
        (mode, variant): MomentQuantTransformer(
            depth=DEPTH, div=DIV, mode=mode, exact_version=variant, approx_version=variant, parallel=False
        )
        for mode in modes
        for variant in variants
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
                print(f"[dispatch holdout] Series length = {L} | Number of samples = {n_samples}")
            X_ = np.ascontiguousarray(X[:n_samples, :, :L])

            jobs = [(key, run) for key in estimators for run in range(N_RUNS)]
            rng.shuffle(jobs)
            for key, run in jobs:
                mode, variant = key
                start_time = time.perf_counter()
                estimators[key].transform(X_)
                end_time = time.perf_counter()
                results.append([L, n_samples, mode, variant, run, end_time - start_time])

    return pd.DataFrame(results, columns=["Series length", "Number of samples", "Mode", "Variant", "Run", "Runtime"])


def solve_true_crossover(c_samples, s_samples, c_intervals, s_intervals):
    delta_slope = s_samples - s_intervals
    if delta_slope <= 0:
        return None
    n_star = (c_intervals - c_samples) / delta_slope
    if n_star <= 0:
        return None
    return float(n_star)


def main():
    runtimes = benchmark_holdout_grid()

    os.makedirs(out_dir, exist_ok=True)
    runtimes.to_csv(os.path.join(out_dir, "runtimes.csv"), index=False)

    mins = (
        runtimes.groupby(["Series length", "Number of samples", "Mode", "Variant"])["Runtime"]
        .min()
        .reset_index()
        .rename(columns={"Series length": "L", "Number of samples": "n", "Runtime": "t_min"})
    )

    MomentQuantTransformer.calibrate_dispatch_thresholds(strict=False, parallel=False)

    crossover_rows = []
    for mode in modes:
        for L in HOLDOUT_L:
            sub_samples = mins[(mins["L"] == L) & (mins["Mode"] == mode) & (mins["Variant"] == "samples")].sort_values(
                "n"
            )
            sub_intervals = mins[
                (mins["L"] == L) & (mins["Mode"] == mode) & (mins["Variant"] == "intervals")
            ].sort_values("n")
            c_s, s_s, r2_s = fit_intercept_slope(sub_samples["n"].values, sub_samples["t_min"].values)
            c_i, s_i, r2_i = fit_intercept_slope(sub_intervals["n"].values, sub_intervals["t_min"].values)
            n_true = solve_true_crossover(c_s, s_s, c_i, s_i)

            threshold = MomentQuantTransformer._nl_threshold(mode, parallel=False)
            if mode == "approx":
                basis = float(L)
            else:
                basis = L * (np.log2(L) - 2.5)
            n_predicted = threshold / basis if basis > 0 else float("nan")

            if n_true is not None:
                abs_error = n_predicted - n_true
                rel_error = abs_error / n_true
            else:
                abs_error = float("nan")
                rel_error = float("nan")

            crossover_rows.append(
                dict(
                    Mode=mode,
                    L=L,
                    c_samples=c_s,
                    s_samples=s_s,
                    r2_samples=r2_s,
                    c_intervals=c_i,
                    s_intervals=s_i,
                    r2_intervals=r2_i,
                    n_true_crossover=n_true if n_true is not None else float("nan"),
                    n_predicted_crossover=n_predicted,
                    absolute_error=abs_error,
                    relative_error=rel_error,
                )
            )
    crossover = pd.DataFrame(crossover_rows)
    crossover.to_csv(os.path.join(out_dir, "crossover_comparison.csv"), index=False)

    probe = MomentQuantTransformer(depth=DEPTH, div=DIV)
    decision_rows = []
    for L in HOLDOUT_L:
        probe.fit(np.zeros((1, 1, L), dtype=np.float64))
        for mode in modes:
            for n in N_SAMPLES_GRID:
                t_samples = mins.query("L == @L and Mode == @mode and Variant == 'samples' and n == @n")["t_min"].iloc[
                    0
                ]
                t_intervals = mins.query("L == @L and Mode == @mode and Variant == 'intervals' and n == @n")[
                    "t_min"
                ].iloc[0]
                true_winner = "samples" if t_samples < t_intervals else "intervals"
                predicted_winner = probe._decide_version(mode, n)
                match = predicted_winner == true_winner

                t_chosen = t_samples if predicted_winner == "samples" else t_intervals
                t_best = min(t_samples, t_intervals)
                pct_slower_if_wrong = (t_chosen - t_best) / t_best * 100.0 if not match else 0.0

                decision_rows.append(
                    dict(
                        Mode=mode,
                        L=L,
                        n=n,
                        t_samples=t_samples,
                        t_intervals=t_intervals,
                        true_winner=true_winner,
                        predicted_winner=predicted_winner,
                        match=match,
                        pct_slower_if_wrong=pct_slower_if_wrong,
                    )
                )
    decisions = pd.DataFrame(decision_rows)
    decisions.to_csv(os.path.join(out_dir, "decision_accuracy.csv"), index=False)

    summary_rows = []
    for mode in modes:
        c_sub = crossover[crossover["Mode"] == mode]
        d_sub = decisions[decisions["Mode"] == mode]
        valid = c_sub.dropna(subset=["relative_error"])
        wrong = d_sub[~d_sub["match"]]
        summary_rows.append(
            dict(
                Mode=mode,
                n_holdout_L=len(c_sub),
                n_true_crossovers_defined=len(valid),
                mean_abs_pct_crossover_error=(
                    float(np.mean(np.abs(valid["relative_error"])) * 100.0) if len(valid) else float("nan")
                ),
                max_abs_pct_crossover_error=(
                    float(np.max(np.abs(valid["relative_error"])) * 100.0) if len(valid) else float("nan")
                ),
                n_decisions=len(d_sub),
                decision_hit_rate_pct=float(d_sub["match"].mean() * 100.0),
                n_decisions_wrong=int(len(wrong)),
                mean_pct_slower_when_wrong=float(wrong["pct_slower_if_wrong"].mean()) if len(wrong) else 0.0,
                max_pct_slower_when_wrong=float(wrong["pct_slower_if_wrong"].max()) if len(wrong) else 0.0,
            )
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    if VERBOSE:
        print("=" * 100)
        print('Dispatch threshold ("auto" mode) held-out validation')
        print("=" * 100)
        print("--- Crossover comparison ---")
        print(crossover.to_string(index=False))
        print()
        print("--- Decision accuracy ---")
        print(decisions.to_string(index=False))
        print()
        print("--- Summary ---")
        print(summary.to_string(index=False))
        print()
        print(f"Saved results to {out_dir!r}")


if __name__ == "__main__":
    main()
