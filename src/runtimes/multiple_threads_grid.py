import os
import time

os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

n = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(n)
os.environ["MKL_NUM_THREADS"] = str(n)
os.environ["OPENBLAS_NUM_THREADS"] = str(n)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n)
os.environ["NUMEXPR_NUM_THREADS"] = str(n)
os.environ["NUMBA_NUM_THREADS"] = str(n)

import numpy as np
import pandas as pd

from src.estimators.moment_quant import MomentQuantTransformer
from src.utils import (
    APPROX_INTERVALS,
    APPROX_SAMPLES,
    EXACT_INTERVALS,
    EXACT_SAMPLES,
    FULL_L_GRID,
    N_RUNS,
    N_SAMPLES_GRID,
    VERBOSE,
    periodic_cooldown,
)

full_series_length_grid = FULL_L_GRID

n_relative = sorted({max(1, n // 4), max(1, n // 2), n, 2 * n, 4 * n, 8 * n})
n_samples_grid = sorted(set(N_SAMPLES_GRID) | set(n_relative))

exclude_cells = {(8192, 10_000)}

seed = 0
rng = np.random.default_rng(seed)
X = rng.normal(size=(max(n_samples_grid), 1, max(full_series_length_grid)))

estimators = {
    EXACT_SAMPLES: MomentQuantTransformer(mode="exact", exact_version="samples", parallel=True),
    EXACT_INTERVALS: MomentQuantTransformer(mode="exact", exact_version="intervals", parallel=True),
    APPROX_SAMPLES: MomentQuantTransformer(mode="approx", approx_version="samples", parallel=True),
    APPROX_INTERVALS: MomentQuantTransformer(mode="approx", approx_version="intervals", parallel=True),
}

warmup_L, warmup_n = min(full_series_length_grid), min(n_samples_grid)
X_warmup = np.ascontiguousarray(X[:warmup_n, :, :warmup_L])
for estimator in estimators.values():
    estimator.fit(X_warmup)
    estimator.transform(X_warmup)

results = []
last_cooldown = time.perf_counter()

for series_length in full_series_length_grid:
    X_fit = np.ascontiguousarray(X[:1, :, :series_length])
    for estimator in estimators.values():
        estimator.fit(X_fit)

    for n_samples in n_samples_grid:
        if (series_length, n_samples) in exclude_cells:
            if VERBOSE:
                print(
                    f"Series length = {series_length} | Number of samples = {n_samples} | SKIPPED (see exclude_cells)"
                )
            continue
        if VERBOSE:
            print(f"Series length = {series_length} | Number of samples = {n_samples}")
        X_ = np.ascontiguousarray(X[:n_samples, :, :series_length])

        jobs = [(estimator_name, run) for estimator_name in estimators for run in range(N_RUNS)]
        rng.shuffle(jobs)

        for estimator_name, run in jobs:
            estimator = estimators[estimator_name]
            start_time = time.perf_counter()
            estimator.transform(X_)
            end_time = time.perf_counter()
            results.append([series_length, n_samples, estimator_name, run, n, end_time - start_time])
            last_cooldown = periodic_cooldown(last_cooldown)

path = os.path.join("results", "runtimes", "multiple_threads_grid")
if not os.path.isdir(path):
    os.makedirs(path)

pd.DataFrame(results, columns=["Series length", "Number of samples", "Estimator", "Run", "Threads", "Runtime"]).to_csv(
    os.path.join(path, "runtimes.csv"), index=False
)
