import os

import pandas as pd

from src.analyses.kappa_series import load_approx_constants, load_processing_constants
from src.runtimes.utils import MEDIAN_N_SAMPLES, MEDIAN_SERIES_LENGTH
from src.theory.cost_formulae import (
    approx_intervals_outer_cost,
    approx_series_outer_cost,
    intervals_outer_cost_all_representations,
    load_kappa_I,
    load_kappa_S,
    load_kappa_S_sum,
    series_outer_cost_all_representations,
)
from src.utils import (
    APPROX_INTERVALS,
    APPROX_SAMPLES,
    DEPTH,
    DIV,
    EXACT_INTERVALS,
    EXACT_SAMPLES,
)

QUANT = "quant"

empirical_dirs = {
    "single_thread_fixed_n_samples": os.path.join("results", "runtimes", "single_thread_fixed_n_samples"),
    "single_thread_fixed_n_timepoints": os.path.join("results", "runtimes", "single_thread_fixed_n_timepoints"),
}

exact_constants_csv = os.path.join("results", "analyses", "single_thread_grid", "constants.csv")
microbench_moment_cf_csv = os.path.join("results", "runtimes", "microbench_moment_cf", "constants.csv")
quant_constants_csv = os.path.join("results", "analyses", "single_thread_quant_grid", "constants.csv")
quant_kappa_intervals_csv = os.path.join("results", "analyses", "single_thread_quant_grid", "kappa_intervals.csv")

out_dir = os.path.join("results", "analyses", "theory_vs_actual")

estimators = [EXACT_SAMPLES, EXACT_INTERVALS, APPROX_SAMPLES, APPROX_INTERVALS, QUANT]


def load_quant_kappa_I(csv_path):
    df = pd.read_csv(csv_path)
    sub = df[df["Estimator"] == "quant_float64"]
    if len(sub) != 1:
        raise ValueError(f"Expected exactly one row in {csv_path!r}, found {len(sub)}.")
    a, b = float(sub.iloc[0]["A"]), float(sub.iloc[0]["B"])

    def kappa_I(l):
        value = a - b / l
        if value <= 0:
            raise ValueError(
                f"kappa_I(l={l}) evaluated to a non-positive value ({value!r}) for QuantFloat64 "
                f"under the fitted model A - B/l with A={a!r}, B={b!r}."
            )
        return value

    return kappa_I


def load_empirical_min(runtimes_dir, estimator):
    path = os.path.join(runtimes_dir, f"{estimator}.csv")
    df = pd.read_csv(path, index_col=0)
    run_cols = [c for c in df.columns if c.startswith("Run ")]
    return df[run_cols].min(axis=1)


def predict_runtime(estimator, n, L, constants, kappa_I_by_mode, kappa_S_by_mode, kappa_S_direct_by_mode):
    c1, c2, c3 = constants[estimator]
    if estimator == EXACT_SAMPLES:
        return series_outer_cost_all_representations(
            n, L, DEPTH, DIV, c1, c2, c3, kappa_S_by_mode["exact"], kappa_S_direct=kappa_S_direct_by_mode["exact"]
        )
    if estimator == EXACT_INTERVALS:
        return intervals_outer_cost_all_representations(n, L, DEPTH, DIV, c1, c2, c3, kappa_I_by_mode["exact"])
    if estimator == APPROX_SAMPLES:
        return approx_series_outer_cost(
            n, L, DEPTH, DIV, c1, c2, c3, kappa_S_by_mode["approx"], kappa_S_direct=kappa_S_direct_by_mode["approx"]
        )
    if estimator == APPROX_INTERVALS:
        return approx_intervals_outer_cost(n, L, DEPTH, DIV, c1, c2, c3, kappa_I_by_mode["approx"])
    if estimator == QUANT:
        return intervals_outer_cost_all_representations(n, L, DEPTH, DIV, c1, c2, c3, kappa_I_by_mode["quant"])
    raise ValueError(f"Unknown estimator {estimator!r}.")


def main():
    constants = {
        EXACT_SAMPLES: load_processing_constants(exact_constants_csv, EXACT_SAMPLES, ("c_1", "c_2", "c_3")),
        EXACT_INTERVALS: load_processing_constants(exact_constants_csv, EXACT_INTERVALS, ("c_1", "c_2", "c_3")),
        APPROX_SAMPLES: load_approx_constants(microbench_moment_cf_csv, APPROX_SAMPLES),
        APPROX_INTERVALS: load_approx_constants(microbench_moment_cf_csv, APPROX_INTERVALS),
        QUANT: load_processing_constants(quant_constants_csv, "quant_float64", ("c_1", "c_2", "c_3")),
    }

    kappa_I_by_mode = {
        "exact": load_kappa_I("exact"),
        "approx": load_kappa_I("approx"),
        "quant": load_quant_kappa_I(quant_kappa_intervals_csv),
    }

    kappa_S_by_mode = {
        "exact": load_kappa_S("exact"),
        "approx": load_kappa_S("approx"),
    }

    kappa_S_direct_by_mode = {
        "exact": load_kappa_S_sum("exact"),
        "approx": load_kappa_S_sum("approx"),
    }

    rows = []
    for sweep, runtimes_dir in empirical_dirs.items():
        for estimator in estimators:
            empirical = load_empirical_min(runtimes_dir, estimator)
            for swept_value, t_actual in empirical.items():
                if sweep == "single_thread_fixed_n_samples":
                    n, L = MEDIAN_N_SAMPLES, int(swept_value)
                else:
                    n, L = int(swept_value), MEDIAN_SERIES_LENGTH

                try:
                    t_theory = predict_runtime(
                        estimator, n, L, constants, kappa_I_by_mode, kappa_S_by_mode, kappa_S_direct_by_mode
                    )
                    prediction_failed = False
                    prediction_error = ""
                except ValueError as exc:
                    t_theory = float("nan")
                    prediction_failed = True
                    prediction_error = str(exc)

                abs_error = t_theory - t_actual if not prediction_failed else float("nan")
                rel_error = abs_error / t_actual if (not prediction_failed and t_actual) else float("nan")

                rows.append(
                    dict(
                        Sweep=sweep,
                        Estimator=estimator,
                        n=n,
                        L=L,
                        Empirical_runtime=t_actual,
                        Theoretical_runtime=t_theory,
                        Absolute_error=abs_error,
                        Relative_error=rel_error,
                        Prediction_failed=prediction_failed,
                        Prediction_error=prediction_error,
                    )
                )

    out = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "theory_vs_actual.csv")
    out.to_csv(out_path, index=False)

    print("=" * 100)
    print("THEORY VS ACTUAL -- single-threaded, minimum-over-repeats on both sides (see module docstring)")
    print("=" * 100)
    valid = out[~out["Prediction_failed"]]
    summary = (
        valid.groupby(["Sweep", "Estimator"])["Relative_error"]
        .agg(mean_signed="mean", mean_abs=lambda s: s.abs().mean(), max_abs=lambda s: s.abs().max())
        .reset_index()
    )
    print(summary.to_string(index=False))
    n_failed = int(out["Prediction_failed"].sum())
    if n_failed:
        print()
        print(
            f"{n_failed} prediction(s) failed (kappa^(I)(l) or kappa^(S)(l) extrapolated to a "
            f"non-positive value) -- see the Prediction_error column in {out_path!r}."
        )
    print()
    print(f"Saved {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
