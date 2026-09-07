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

from src.estimators.moment_quant import (
    MomentQuantTransformer,
    _batch_interval_moments_intervals,
    _batch_interval_moments_samples,
)
from src.utils import APPROX_INTERVALS, APPROX_SAMPLES, DEPTH, DIV, N_RUNS, SMOKE_TEST, VERBOSE

n_samples = 2000

l_grid = [
    9,
    14,
    20,
    24,
    32,
    40,
    48,
    56,
    64,
    80,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
]

if SMOKE_TEST:
    l_grid = [9, 16, 32, 96, 256, 1024, 8192]

rng = np.random.default_rng(0)

moment_fn = {
    APPROX_SAMPLES: _batch_interval_moments_samples,
    APPROX_INTERVALS: _batch_interval_moments_intervals,
}


def raw_layout(depth, div, length):
    est = MomentQuantTransformer(depth=depth, div=div, mode="approx", approx_version="intervals")
    est.fit(np.zeros((1, 1, length), dtype=np.float64))
    return est.layouts_[0]


def layout_basis(layout):
    starts, ends, kind, offsets = layout["starts"], layout["ends"], layout["kind"], layout["offsets"]
    w_tilde = 0
    nq_tilde = 0
    n_i = len(starts)
    for i in range(n_i):
        w_tilde += int(ends[i] - starts[i])
        if kind[i] != 0:
            nq_tilde += int(offsets[i + 1] - offsets[i])
    return w_tilde, nq_tilde, n_i


def cf_extraction_pass(estimator_instance, mean, var, skew, exkurt, layout):
    starts = layout["starts"]
    kind, offsets, q_positions = layout["kind"], layout["offsets"], layout["q_positions"]
    n_intervals = len(starts)
    for i in range(n_intervals):
        if kind[i] == 0:
            continue
        o0, o1 = int(offsets[i]), int(offsets[i + 1])
        if kind[i] == 2:
            if o1 - o0 > 2:
                q_interior = q_positions[o0 + 1 : o1 - 1]
                estimator_instance._cornish_fisher_quantiles(
                    mean[:, i], var[:, i], skew[:, i], exkurt[:, i], q_interior
                )
        else:
            q = q_positions[o0:o1]
            estimator_instance._cornish_fisher_quantiles(mean[:, i], var[:, i], skew[:, i], exkurt[:, i], q)


def fit_intercept_slope(x, t):
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    inv_x = 1.0 / x
    y = t / x
    intercept, slope = np.polyfit(inv_x, y, 1)
    pred = intercept + slope * x
    ss_res = np.sum((t - pred) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return intercept, slope, r2


def fit_joint_2param(x1, x2, t):
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    t = np.asarray(t, dtype=float)
    design = np.column_stack([np.ones_like(x1), x1, x2])
    coeffs, *_ = np.linalg.lstsq(design, t, rcond=None)
    intercept, b1, b2 = (float(c) for c in coeffs)
    pred = design @ coeffs
    ss_res = np.sum((t - pred) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    cond = np.linalg.cond(design)
    return intercept, b1, b2, r2, cond


def main():
    estimator_instance = MomentQuantTransformer(mode="approx")

    dummy_X = np.ascontiguousarray(rng.standard_normal((3, 4)))
    dummy_starts = np.array([0], dtype=np.int64)
    dummy_ends = np.array([4], dtype=np.int64)
    _batch_interval_moments_samples(dummy_X, dummy_starts, dummy_ends)
    _batch_interval_moments_intervals(dummy_X, dummy_starts, dummy_ends)
    dummy_q = np.linspace(0.01, 0.99, 2)
    estimator_instance._cornish_fisher_quantiles(
        rng.standard_normal(3),
        np.abs(rng.standard_normal(3)) + 0.1,
        rng.standard_normal(3) * 0.1,
        rng.standard_normal(3) * 0.1,
        dummy_q,
    )
    MomentQuantTransformer._moments_to_skew_kurt(
        np.full((3, 2), 50.0),
        np.abs(rng.standard_normal((3, 2))) + 1.0,
        rng.standard_normal((3, 2)),
        np.abs(rng.standard_normal((3, 2))) + 1.0,
    )

    layouts_by_L = {L: raw_layout(DEPTH, DIV, L) for L in l_grid}
    basis_by_L = {L: layout_basis(layouts_by_L[L]) for L in l_grid}

    print("=" * 100)
    print(f"REAL RAW-REPRESENTATION LAYOUT BASIS (depth={DEPTH}, div={DIV})")
    print("=" * 100)
    print(f"{'L':>6} {'Wtilde':>10} {'NQtilde>1':>10} {'NItilde':>8}")
    for L in l_grid:
        w, nq, ni = basis_by_L[L]
        print(f"{L:6d} {w:10d} {nq:10d} {ni:8d}")
    w_arr = np.array([basis_by_L[L][0] for L in l_grid], dtype=float)
    nq_arr = np.array([basis_by_L[L][1] for L in l_grid], dtype=float)
    ni_arr = np.array([basis_by_L[L][2] for L in l_grid], dtype=float)
    print()
    print(
        f"corr(Wtilde, NQtilde^{{>1}}) = {np.corrcoef(w_arr, nq_arr)[0, 1]:.6f}  (excluded from the "
        f"c2/c3 fit below for exactly this reason if close to 1)"
    )
    print(f"corr(Wtilde, NItilde)      = {np.corrcoef(w_arr, ni_arr)[0, 1]:.6f}")
    print(
        f"corr(NQtilde^{{>1}}, NItilde) = {np.corrcoef(nq_arr, ni_arr)[0, 1]:.6f}  (the pair actually "
        f"used to jointly identify c2_tilde/c3_tilde below)"
    )

    data_by_L = {L: np.ascontiguousarray(rng.standard_normal((n_samples, L))) for L in l_grid}

    def dummy_moment_inputs(n_i):
        n = np.full((n_samples, n_i), 50.0)
        M2 = np.abs(rng.standard_normal((n_samples, n_i))) * 50.0 + 1.0
        M3 = rng.standard_normal((n_samples, n_i)) * 10.0
        M4 = np.abs(rng.standard_normal((n_samples, n_i))) * 50.0 + 1.0
        mean = rng.standard_normal((n_samples, n_i))
        var = np.abs(rng.standard_normal((n_samples, n_i))) + 0.1
        skew = rng.standard_normal((n_samples, n_i)) * 0.1
        exkurt = rng.standard_normal((n_samples, n_i)) * 0.1
        return n, M2, M3, M4, mean, var, skew, exkurt

    moment_inputs_by_L = {L: dummy_moment_inputs(basis_by_L[L][2]) for L in l_grid}

    cells = []
    for estimator in (APPROX_SAMPLES, APPROX_INTERVALS):
        for L in l_grid:
            cells.append(("c1_tilde", estimator, L))
    for L in l_grid:
        cells.append(("cf_extraction", None, L))

    jobs = [(cell, run) for cell in cells for run in range(N_RUNS)]
    rng.shuffle(jobs)

    times = {cell: [] for cell in cells}
    for j, (cell, run) in enumerate(jobs):
        quantity, estimator, L = cell
        layout = layouts_by_L[L]
        if quantity == "c1_tilde":
            X = data_by_L[L]
            starts = layout["starts"].astype(np.int64)
            ends = layout["ends"].astype(np.int64)
            fn = moment_fn[estimator]
            t0 = time.perf_counter()
            fn(X, starts, ends)
            t = time.perf_counter() - t0
        else:
            n, M2, M3, M4, mean, var, skew, exkurt = moment_inputs_by_L[L]
            t0 = time.perf_counter()
            var2, skew2, exkurt2 = MomentQuantTransformer._moments_to_skew_kurt(n, M2, M3, M4)
            cf_extraction_pass(estimator_instance, mean, var2, skew2, exkurt2, layout)
            t = time.perf_counter() - t0
        times[cell].append(t)
        if VERBOSE and (j % 200 == 0):
            print(f"[{j + 1:>{len(str(len(jobs)))}}/{len(jobs)}] {cell}")

    t_min = {cell: min(ts) for cell, ts in times.items()}

    results = []

    for estimator in (APPROX_SAMPLES, APPROX_INTERVALS):
        xs, ts = [], []
        for L in l_grid:
            w, _, _ = basis_by_L[L]
            t = t_min[("c1_tilde", estimator, L)]
            xs.append(n_samples * w)
            ts.append(t)
            if VERBOSE:
                print(f"  [{estimator}] L={L:6d}  Wtilde={w:8d}  t={t:.6f}s")
        intercept, slope, r2 = fit_intercept_slope(xs, ts)
        print(f"  -> c1_tilde[{estimator}] = {slope:.4e} s/element  (intercept={intercept:.4e}s, R2={r2:.5f})")
        results.append(
            dict(
                Estimator=estimator,
                Constant="c1_tilde",
                Value=slope,
                Intercept=intercept,
                R2=r2,
                Fit="microbench_moment_real_layout",
            )
        )

    x1s, x2s, ts = [], [], []
    for L in l_grid:
        _, nq, ni = basis_by_L[L]
        t = t_min[("cf_extraction", None, L)]
        x1s.append(n_samples * nq)
        x2s.append(n_samples * ni)
        ts.append(t)
        if VERBOSE:
            print(f"  [cf_extraction] L={L:6d}  NQtilde>1={nq:6d}  NItilde={ni:4d}  t={t:.6f}s")
    intercept, c2_tilde, c3_tilde, r2, cond = fit_joint_2param(x1s, x2s, ts)
    print(
        f"  -> c2_tilde = {c2_tilde:.4e} s/quantile, c3_tilde = {c3_tilde:.4e} s/interval "
        f"(intercept={intercept:.4e}s, R2={r2:.5f}, design cond#={cond:.4g})"
    )
    for estimator in (APPROX_SAMPLES, APPROX_INTERVALS):
        results.append(
            dict(
                Estimator=estimator,
                Constant="c2_tilde",
                Value=c2_tilde,
                Intercept=intercept,
                R2=r2,
                Fit="microbench_cf_joint_real_layout",
            )
        )
        results.append(
            dict(
                Estimator=estimator,
                Constant="c3_tilde",
                Value=c3_tilde,
                Intercept=intercept,
                R2=r2,
                Fit="microbench_cf_joint_real_layout",
            )
        )

    out = pd.DataFrame(results, columns=["Estimator", "Constant", "Value", "Intercept", "R2", "Fit"])
    path = os.path.join("results", "runtimes", "microbench_moment_cf")
    os.makedirs(path, exist_ok=True)
    out_path = os.path.join(path, "constants.csv")
    out.to_csv(out_path, index=False)
    print()
    print(f"Saved {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
