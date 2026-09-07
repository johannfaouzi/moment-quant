import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_WARNINGS"] = "FALSE"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time

import numba
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.estimators.moment_quant import (
    MomentQuantTransformer,
    _batch_exact_features_intervals,
    _batch_exact_features_samples,
    _batch_exact_features_samples_parallel,
)
from src.runtimes.utils import LARGE_L_GRID, LARGE_N_SAMPLES
from src.utils import DEPTH, DIV, N_RUNS, VERBOSE, periodic_cooldown


def _batch_exact_features_intervals_joblib(X, starts, ends, kind, offsets, q_positions, center_mask, n_jobs):
    if n_jobs <= 1:
        return _batch_exact_features_intervals(X, starts, ends, kind, offsets, q_positions, center_mask)

    n_samples = X.shape[0]
    total = int(offsets[-1])
    out = np.empty((n_samples, total), dtype=np.float64)

    chunks = [c for c in np.array_split(np.arange(n_samples), n_jobs) if c.size > 0]
    Parallel(n_jobs=len(chunks), backend="threading")(
        delayed(_batch_exact_features_intervals)(
            X[c[0] : c[-1] + 1],
            starts,
            ends,
            kind,
            offsets,
            q_positions,
            center_mask,
            out=out[c[0] : c[-1] + 1],
        )
        for c in chunks
    )
    return out


series_length_grid = LARGE_L_GRID
n_samples = LARGE_N_SAMPLES
n_cores = os.cpu_count()
threads_grid = sorted(set([1, 2, 4, n_cores]))

rng = np.random.default_rng(0)
X_full = rng.normal(size=(n_samples, max(series_length_grid)))

_layout_builder = MomentQuantTransformer(depth=DEPTH, div=DIV)


def build_layout(length):
    starts, ends, _ = _layout_builder._make_intervals(length)
    _, kind, offsets, center_mask, q_positions = _layout_builder._build_quantile_layout(starts, ends)
    return starts, ends, kind, offsets, q_positions, center_mask


def main():
    warmup_L = min(series_length_grid)
    X_warmup = np.ascontiguousarray(X_full[:16, :warmup_L])
    warmup_layout = build_layout(warmup_L)

    _batch_exact_features_samples(X_warmup, *warmup_layout)
    numba.set_num_threads(n_cores)
    _batch_exact_features_samples_parallel(X_warmup, *warmup_layout)

    out_serial = _batch_exact_features_intervals(X_warmup, *warmup_layout)
    out_joblib = _batch_exact_features_intervals_joblib(X_warmup, *warmup_layout, n_jobs=n_cores)
    if not np.array_equal(out_serial, out_joblib):
        max_abs_diff = np.max(np.abs(out_serial - out_joblib))
        raise AssertionError(
            f"intervals_joblib output does not exactly match intervals output "
            f"(max abs diff={max_abs_diff!r}). Aborting before timing anything."
        )
    print(
        f"Correctness check passed: intervals_joblib (n_jobs={n_cores}) output is bit-for-bit "
        f"identical to serial intervals on a {X_warmup.shape} warmup array."
    )
    print(f"Detected {n_cores} CPU cores. Thread counts swept: {threads_grid}.")
    print()

    cells = []
    for L in series_length_grid:
        cells.append(("samples", L, 1))
        cells.append(("intervals", L, 1))
        for threads in threads_grid:
            cells.append(("samples_parallel", L, threads))
            cells.append(("intervals_joblib", L, threads))

    jobs = [(cell, run) for cell in cells for run in range(N_RUNS)]
    rng.shuffle(jobs)

    layouts = {L: build_layout(L) for L in series_length_grid}
    X_by_L = {L: np.ascontiguousarray(X_full[:, :L]) for L in series_length_grid}

    times = {cell: [] for cell in cells}
    last_cooldown = time.perf_counter()
    for j, (cell, run) in enumerate(jobs):
        method, L, threads = cell
        X_ = X_by_L[L]
        layout = layouts[L]

        if method == "samples":
            t0 = time.perf_counter()
            _batch_exact_features_samples(X_, *layout)
            t = time.perf_counter() - t0
        elif method == "intervals":
            t0 = time.perf_counter()
            _batch_exact_features_intervals(X_, *layout)
            t = time.perf_counter() - t0
        elif method == "samples_parallel":
            numba.set_num_threads(threads)
            t0 = time.perf_counter()
            _batch_exact_features_samples_parallel(X_, *layout)
            t = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            _batch_exact_features_intervals_joblib(X_, *layout, n_jobs=threads)
            t = time.perf_counter() - t0

        times[cell].append(t)
        last_cooldown = periodic_cooldown(last_cooldown)
        if VERBOSE and (j % 50 == 0):
            print(f"[{j + 1:>{len(str(len(jobs)))}}/{len(jobs)}] {cell}")

    t_min = {cell: min(ts) for cell, ts in times.items()}

    results = []
    for L in series_length_grid:
        n_features = int(layouts[L][3][-1])
        t_samples = t_min[("samples", L, 1)]
        t_intervals = t_min[("intervals", L, 1)]
        print("=" * 100)
        print(f"L={L}  (n_samples={n_samples}, n_features={n_features})")
        print(f"  {'samples (serial)':>22}: {t_samples:.6f}s")
        print(f"  {'intervals (serial)':>22}: {t_intervals:.6f}s")
        results.append(dict(L=L, Method="samples", Threads=1, Runtime=t_samples, NFeatures=n_features))
        results.append(dict(L=L, Method="intervals", Threads=1, Runtime=t_intervals, NFeatures=n_features))

        for method in ("samples_parallel", "intervals_joblib"):
            baseline = t_min[(method, L, 1)]
            for threads in threads_grid:
                t = t_min[(method, L, threads)]
                speedup = baseline / t
                vs_intervals = t_intervals / t
                print(
                    f"  {method:>22} [{threads:>2} threads]: {t:.6f}s  "
                    f"({speedup:.2f}x vs its own 1-thread, {vs_intervals:.2f}x vs serial intervals)"
                )
                results.append(dict(L=L, Method=method, Threads=threads, Runtime=t, NFeatures=n_features))
        print()

    out_df = pd.DataFrame(results, columns=["L", "Method", "Threads", "Runtime", "NFeatures"])
    path = os.path.join("results", "runtimes", "intervals_joblib_vs_intervals_numpy")
    os.makedirs(path, exist_ok=True)
    out_path = os.path.join(path, "runtimes.csv")
    out_df.to_csv(out_path, index=False)
    if VERBOSE:
        print(f"Saved {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()
