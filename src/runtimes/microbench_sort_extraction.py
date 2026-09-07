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
from numba import njit

from src.utils import N_RUNS, SMOKE_TEST, VERBOSE

n_samples = 2000
m_grid = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
q_grid = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
m_fixed_for_c2 = [64, 512, 4096]

if SMOKE_TEST:
    m_grid = [8, 128, 2048, 8192]
    q_grid = [1, 16, 256, 512]

rng = np.random.default_rng(0)


@njit("float64[:, ::1](float64[:, ::1])", cache=True)
def _samples_sort_only(X):
    n_samples, m = X.shape
    out = np.empty((n_samples, m))
    for i in range(n_samples):
        out[i] = np.sort(X[i])
    return out


@njit("float64[::1](float64[:, ::1])", cache=True)
def _samples_mean_only(X):
    n_samples, m = X.shape
    out = np.empty(n_samples)
    for i in range(n_samples):
        seg_mean = 0.0
        for t in range(m):
            seg_mean += X[i, t]
        out[i] = seg_mean / m
    return out


@njit("float64[:, ::1](float64[:, ::1], float64[::1])", cache=True)
def _samples_interp_only(seg_sorted, q_positions):
    n_samples, m = seg_sorted.shape
    n_q = q_positions.shape[0]
    out = np.empty((n_samples, n_q))
    for i in range(n_samples):
        seg = seg_sorted[i]
        for j in range(n_q):
            pos = q_positions[j] * (m - 1)
            lo = int(np.floor(pos))
            hi = min(lo + 1, m - 1)
            frac = pos - lo
            out[i, j] = seg[lo] * (1.0 - frac) + seg[hi] * frac
    return out


def _intervals_sort_only(X):
    return np.sort(X, axis=-1)


def _intervals_mean_only(X):
    return X.mean(axis=-1, keepdims=True)


def _intervals_interp_only(seg_sorted, q_positions):
    m = seg_sorted.shape[1]
    pos = q_positions * (m - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, m - 1)
    frac = pos - lo
    return seg_sorted[:, lo] * (1.0 - frac) + seg_sorted[:, hi] * frac


sort_fn = {"samples": _samples_sort_only, "intervals": _intervals_sort_only}
mean_fn = {"samples": _samples_mean_only, "intervals": _intervals_mean_only}
interp_fn = {"samples": _samples_interp_only, "intervals": _intervals_interp_only}
kernels = ["samples", "intervals"]


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


def main():
    dummy_X = np.ascontiguousarray(rng.standard_normal((3, 4)))
    dummy_Xs = np.sort(dummy_X, axis=-1)
    dummy_q = np.linspace(0.0, 1.0, 2)
    for fn in (sort_fn["samples"], sort_fn["intervals"], mean_fn["samples"], mean_fn["intervals"]):
        fn(dummy_X)
    for fn in (interp_fn["samples"], interp_fn["intervals"]):
        fn(dummy_Xs, dummy_q)

    all_m = sorted(set(m_grid) | set(m_fixed_for_c2))
    data_by_m = {m: np.ascontiguousarray(rng.standard_normal((n_samples, m))) for m in all_m}
    sorted_by_m = {m: np.sort(data_by_m[m], axis=-1) for m in m_fixed_for_c2}
    q_positions_by_q = {q: np.linspace(0.0, 1.0, q) for q in q_grid}

    cells = []
    for kernel in kernels:
        for m in m_grid:
            cells.append(("c1", kernel, m, None))
            cells.append(("c3", kernel, m, None))
        for m in m_fixed_for_c2:
            for q in q_grid:
                cells.append(("c2", kernel, m, q))

    jobs = [(cell, run) for cell in cells for run in range(N_RUNS)]
    rng.shuffle(jobs)

    times = {cell: [] for cell in cells}
    for j, (cell, run) in enumerate(jobs):
        quantity, kernel, m, q = cell
        if quantity == "c1":
            X = data_by_m[m]
            fn = sort_fn[kernel]
            t0 = time.perf_counter()
            fn(X)
            t = time.perf_counter() - t0
        elif quantity == "c3":
            X = data_by_m[m]
            fn = mean_fn[kernel]
            t0 = time.perf_counter()
            fn(X)
            t = time.perf_counter() - t0
        else:
            Xs = sorted_by_m[m]
            qp = q_positions_by_q[q]
            fn = interp_fn[kernel]
            t0 = time.perf_counter()
            fn(Xs, qp)
            t = time.perf_counter() - t0
        times[cell].append(t)
        if VERBOSE and (j % 500 == 0):
            print(f"[{j + 1:>{len(str(len(jobs)))}}/{len(jobs)}] {cell}")

    t_min = {cell: min(ts) for cell, ts in times.items()}

    results = []

    for kernel in kernels:
        xs, ts = [], []
        for m in m_grid:
            t = t_min[("c1", kernel, m, None)]
            xs.append(n_samples * m * np.log2(m))
            ts.append(t)
            if VERBOSE:
                print(f"  [{kernel:>9}] m={m:6d}  t={t:.6f}s")
        intercept, slope, r2 = fit_intercept_slope(xs, ts)
        print(f"  -> c_1[{kernel}] = {slope:.4e} s/unit  (intercept={intercept:.4e}s, R2={r2:.5f})")
        results.append(
            dict(
                Estimator=f"moment_quant_exact_{kernel}",
                Constant="c_1",
                Value=slope,
                Intercept=intercept,
                R2=r2,
                Fit="microbench_sort",
            )
        )

    for kernel in kernels:
        xs, ts = [], []
        for m in m_grid:
            t = t_min[("c3", kernel, m, None)]
            xs.append(n_samples * m)
            ts.append(t)
            if VERBOSE:
                print(f"  [{kernel:>9}] m={m:6d}  t={t:.6f}s")
        intercept, slope, r2 = fit_intercept_slope(xs, ts)
        print(f"  -> c_3[{kernel}] = {slope:.4e} s/unit  (intercept={intercept:.4e}s, R2={r2:.5f})")
        results.append(
            dict(
                Estimator=f"moment_quant_exact_{kernel}",
                Constant="c_3",
                Value=slope,
                Intercept=intercept,
                R2=r2,
                Fit="microbench_mean",
            )
        )

    for kernel in kernels:
        per_m_slopes = []
        for m in m_fixed_for_c2:
            xs, ts = [], []
            for q in q_grid:
                t = t_min[("c2", kernel, m, q)]
                xs.append(n_samples * q)
                ts.append(t)
                if VERBOSE:
                    print(f"  [{kernel:>9}] m={m:5d} q={q:4d}  t={t:.6f}s")
            intercept, slope, r2 = fit_intercept_slope(xs, ts)
            print(f"  -> c_2[{kernel}, m={m}] = {slope:.4e} s/unit  (intercept={intercept:.4e}s, R2={r2:.5f})")
            per_m_slopes.append(slope)
            results.append(
                dict(
                    Estimator=f"moment_quant_exact_{kernel}",
                    Constant="c_2",
                    Value=slope,
                    Intercept=intercept,
                    R2=r2,
                    Fit=f"microbench_interp_m{m}",
                )
            )
        spread = (max(per_m_slopes) - min(per_m_slopes)) / np.mean(per_m_slopes)
        print(
            f"  [{kernel}] c_2 spread across m: {spread:.1%} of the mean "
            f"(should be small if c_2 is genuinely m-independent, as Theorem 8 assumes)"
        )
        results.append(
            dict(
                Estimator=f"moment_quant_exact_{kernel}",
                Constant="c_2",
                Value=np.mean(per_m_slopes),
                Intercept=np.nan,
                R2=np.nan,
                Fit="microbench_interp_pooled_mean",
            )
        )

    out = pd.DataFrame(results, columns=["Estimator", "Constant", "Value", "Intercept", "R2", "Fit"])
    path = os.path.join("results", "runtimes", "microbench_sort_extraction")
    os.makedirs(path, exist_ok=True)
    out_path = os.path.join(path, "constants.csv")
    out.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
