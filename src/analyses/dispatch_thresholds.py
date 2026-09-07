import os

import numpy as np
import pandas as pd

from src.estimators.moment_quant import MomentQuantTransformer
from src.utils import (
    APPROX_INTERVALS,
    APPROX_SAMPLES,
    DEPTH,
    DIV,
    EXACT_INTERVALS,
    EXACT_SAMPLES,
    SMOKE_TEST,
)

n_intervals = 2 ** (DEPTH + 1) - DEPTH - 2

runtimes_path = os.path.join("results", "runtimes", "single_thread_grid", "runtimes.csv")
out_dir = os.path.join("results", "analyses", "single_thread_grid")
runtimes_parallel_path = os.path.join("results", "runtimes", "multiple_threads_grid", "runtimes.csv")
out_dir_parallel = os.path.join("results", "analyses", "multiple_threads_grid")

estimator_info = {
    APPROX_SAMPLES: ("approx", "samples"),
    APPROX_INTERVALS: ("approx", "intervals"),
    EXACT_SAMPLES: ("exact", "samples"),
    EXACT_INTERVALS: ("exact", "intervals"),
}

exclude_cells = {("exact", "intervals", 8192, 10_000)}


def x_approx(L):
    return DEPTH * L * (1.0 + 1.0 / DIV)


def x_exact(L):
    return DEPTH * L * (np.log2(L) - (DEPTH - 1) / 2.0)


def fit_intercept_slope(n, t):
    n = np.asarray(n, dtype=float)
    t = np.asarray(t, dtype=float)
    x = 1.0 / n
    y = t / n
    c, s = np.polyfit(x, y, 1)
    pred = c + s * n
    ss_res = np.sum((t - pred) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return c, s, r2


def ols_through_origin(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    slope = np.sum(x * y) / np.sum(x * x)
    pred = slope * x
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return slope, r2


def run_pipeline(mins, L_values, drop_excluded):
    data = mins.copy()
    if drop_excluded:
        mask = data.apply(lambda r: (r["mode"], r["variant"], r["L"], r["n"]) in exclude_cells, axis=1)
        data = data[~mask]

    stage1_rows = []
    for L in L_values:
        for mode in ("approx", "exact"):
            for variant in ("samples", "intervals"):
                sub = data[(data["L"] == L) & (data["mode"] == mode) & (data["variant"] == variant)]
                sub = sub.sort_values("n")
                intercept, slope, r2 = fit_intercept_slope(sub["n"].values, sub["t_min"].values)
                stage1_rows.append(
                    dict(L=L, mode=mode, variant=variant, intercept=intercept, slope=slope, r2=r2, n_points=len(sub))
                )
    stage1 = pd.DataFrame(stage1_rows)

    kappa_estimates = {}
    for mode in ("approx", "exact"):
        vals = []
        for L in L_values:
            c_samples = stage1.query("L == @L and mode == @mode and variant == 'samples'")["intercept"].iloc[0]
            c_intervals = stage1.query("L == @L and mode == @mode and variant == 'intervals'")["intercept"].iloc[0]
            vals.append((c_intervals - c_samples) / n_intervals)
        kappa_estimates[mode] = float(np.median(vals))

    tau_estimates = {}
    for mode in ("approx", "exact"):
        basis_func = x_approx if mode == "approx" else x_exact
        for variant in ("samples", "intervals"):
            sub = stage1.query("mode == @mode and variant == @variant").sort_values("L")
            X = basis_func(sub["L"].values)
            tau, r2 = ols_through_origin(X, sub["slope"].values)
            tau_estimates[(mode, variant)] = (tau, r2)

    return stage1, kappa_estimates, tau_estimates


def compute_thresholds(kappa_estimates, tau_estimates):
    out = {}
    for mode in ("approx", "exact"):
        tau_scalar, _ = tau_estimates[(mode, "samples")]
        tau_vec, _ = tau_estimates[(mode, "intervals")]
        delta = tau_scalar - tau_vec
        kappa = kappa_estimates[mode]
        if delta > 0:
            nl_threshold = kappa * n_intervals / delta
        else:
            nl_threshold = float("inf")
        out[mode] = dict(kappa=kappa, tau_scalar=tau_scalar, tau_vec=tau_vec, delta=delta, NL_Threshold=nl_threshold)
    return out


def fit_and_save(runtimes_path, out_dir, label):
    df = pd.read_csv(runtimes_path)
    df["mode"], df["variant"] = zip(*df["Estimator"].map(estimator_info.get))

    mins = (
        df.groupby(["Series length", "Number of samples", "mode", "variant"])["Runtime"]
        .min()
        .reset_index()
        .rename(columns={"Series length": "L", "Number of samples": "n", "Runtime": "t_min"})
    )
    L_values = sorted(mins["L"].unique())

    _, kappa_full, tau_full = run_pipeline(mins, L_values, drop_excluded=False)
    _, kappa_robust, tau_robust = run_pipeline(mins, L_values, drop_excluded=True)

    thresholds_full = compute_thresholds(kappa_full, tau_full)
    thresholds_robust = compute_thresholds(kappa_robust, tau_robust)

    print("=" * 100)
    print(
        f"DISPATCH THRESHOLDS [{label}] -- full grid vs excluding the suspect "
        f"(L=8192, n=10000, exact/intervals) cell"
    )
    print("=" * 100)
    for mode in ("approx", "exact"):
        f, r = thresholds_full[mode], thresholds_robust[mode]
        print(
            f"mode={mode:6s}  full: kappa={f['kappa']:.4g} tau_scalar={f['tau_scalar']:.4g} "
            f"tau_vec={f['tau_vec']:.4g} NL_Threshold={f['NL_Threshold']:.1f}"
        )
        print(
            f"           robust: kappa={r['kappa']:.4g} tau_scalar={r['tau_scalar']:.4g} "
            f"tau_vec={r['tau_vec']:.4g} NL_Threshold={r['NL_Threshold']:.1f}"
        )

    rows = [dict(Mode=mode, Depth=DEPTH, Div=DIV, **thresholds_robust[mode]) for mode in ("approx", "exact")]
    out = pd.DataFrame(rows)[["Mode", "Depth", "Div", "kappa", "tau_scalar", "tau_vec", "delta", "NL_Threshold"]]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dispatch_thresholds.csv")
    out.to_csv(out_path, index=False)

    print()
    print(f"Saved {len(out)} rows to {out_path}")
    return out_path


def main():
    out_path = fit_and_save(runtimes_path, out_dir, label="serial (parallel=False)")

    MomentQuantTransformer.calibrate_dispatch_thresholds(csv_path=out_path, parallel=False, save=not SMOKE_TEST)
    if SMOKE_TEST:
        print("Loaded into MomentQuantTransformer (parallel=False); NOT cached (SMOKE_TEST)")
    else:
        print(
            f"Loaded into MomentQuantTransformer (parallel=False) and cached at "
            f"{MomentQuantTransformer._dispatch_thresholds_cache_path()}"
        )

    print()
    out_path_parallel = fit_and_save(runtimes_parallel_path, out_dir_parallel, label="parallel (parallel=True)")
    MomentQuantTransformer.calibrate_dispatch_thresholds(csv_path=out_path_parallel, parallel=True, save=not SMOKE_TEST)
    if SMOKE_TEST:
        print("Loaded into MomentQuantTransformer (parallel=True); NOT cached (SMOKE_TEST)")
    else:
        print(
            f"Loaded into MomentQuantTransformer (parallel=True) and cached at "
            f"{MomentQuantTransformer._dispatch_thresholds_cache_path()}"
        )


if __name__ == "__main__":
    main()
