import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["KMP_WARNINGS"] = "FALSE"

import time

import numpy as np
import pandas as pd
from numba import njit

from src.estimators.moment_quant import (
    MomentQuantTransformer,
    _batch_exact_features_intervals,
    _batch_exact_features_samples,
)
from src.runtimes.utils import LARGE_L_GRID, LARGE_N_SAMPLES
from src.utils import DEPTH, DIV, N_RUNS, VERBOSE, periodic_cooldown


@njit(
    "float64[:, ::1](float64[:, ::1], int64[::1], int64[::1], int64[::1], int64[::1], float64[::1], boolean[::1])",
    cache=True,
)
def _batch_exact_features_intervals_numba(X, starts, ends, kind, offsets, q_positions, center_mask):
    n_samples = X.shape[0]
    total = offsets[-1]
    out = np.empty((n_samples, total), dtype=np.float64)
    n_intervals = starts.shape[0]

    for i in range(n_intervals):
        s, e = starts[i], ends[i]
        o0, o1 = offsets[i], offsets[i + 1]

        if kind[i] == 0:
            for row in range(n_samples):
                out[row, o0] = X[row, s]
            continue

        m = e - s
        for row in range(n_samples):
            seg = np.sort(X[row, s:e])
            seg_mean = 0.0
            for t in range(m):
                seg_mean += seg[t]
            seg_mean /= m
            for j in range(o1 - o0):
                pos = q_positions[o0 + j] * (m - 1)
                lo = int(np.floor(pos))
                hi = min(lo + 1, m - 1)
                frac = pos - lo
                val = seg[lo] * (1.0 - frac) + seg[hi] * frac
                if center_mask[o0 + j]:
                    val -= seg_mean
                out[row, o0 + j] = val

    return out


methods = {
    "samples": _batch_exact_features_samples,
    "intervals": _batch_exact_features_intervals,
    "intervals_numba": _batch_exact_features_intervals_numba,
}

series_length_grid = LARGE_L_GRID
n_samples = LARGE_N_SAMPLES

rng = np.random.default_rng(0)
X_full = rng.normal(size=(n_samples, max(series_length_grid)))

_layout_builder = MomentQuantTransformer(depth=DEPTH, div=DIV)


def build_layout(length):
    starts, ends, _ = _layout_builder._make_intervals(length)
    _, kind, offsets, center_mask, q_positions = _layout_builder._build_quantile_layout(starts, ends)
    return starts, ends, kind, offsets, q_positions, center_mask


warmup_L = min(series_length_grid)
X_warmup = np.ascontiguousarray(X_full[:10, :warmup_L])
warmup_layout = build_layout(warmup_L)
for method_func in methods.values():
    method_func(X_warmup, *warmup_layout)

results = []
last_cooldown = time.perf_counter()
for series_length in series_length_grid:
    if VERBOSE:
        print(f"Series length = {series_length}")
    X_ = np.ascontiguousarray(X_full[:, :series_length])
    layout = build_layout(series_length)
    n_features = int(layout[3][-1])

    jobs = [(method, run) for method in methods for run in range(N_RUNS)]
    rng.shuffle(jobs)

    for method, run in jobs:
        start_time = time.perf_counter()
        methods[method](X_, *layout)
        end_time = time.perf_counter()
        results.append([series_length, method, run, end_time - start_time, n_features])
        last_cooldown = periodic_cooldown(last_cooldown)

path = os.path.join("results", "runtimes", "intervals_numba_vs_intervals_numpy")
os.makedirs(path, exist_ok=True)
out_path = os.path.join(path, "runtimes.csv")
pd.DataFrame(results, columns=["Series length", "Method", "Run", "Runtime", "Number of features"]).to_csv(
    out_path, index=False
)

if VERBOSE:
    print(f"Saved {len(results)} rows to {out_path}")
