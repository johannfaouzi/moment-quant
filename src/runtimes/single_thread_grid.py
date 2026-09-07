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

from src.estimators.moment_quant import MomentQuantTransformer
from src.utils import (
    APPROX_INTERVALS,
    APPROX_SAMPLES,
    EXACT_INTERVALS,
    EXACT_SAMPLES,
    N_RUNS,
    N_RUNS_SMALL_L,
    N_SAMPLES_GRID,
    SINGLE_THREAD_L_GRID,
    SMALL_L_REPEAT_THRESHOLD,
    VERBOSE,
    periodic_cooldown,
)

full_series_length_grid = SINGLE_THREAD_L_GRID
n_samples_grid = N_SAMPLES_GRID

seed = 0
rng = np.random.default_rng(seed)
X = rng.normal(size=(max(n_samples_grid), 1, max(full_series_length_grid)))

estimators = {
    EXACT_SAMPLES: MomentQuantTransformer(mode="exact", exact_version="samples", parallel=False),
    EXACT_INTERVALS: MomentQuantTransformer(mode="exact", exact_version="intervals", parallel=False),
    APPROX_SAMPLES: MomentQuantTransformer(mode="approx", approx_version="samples", parallel=False),
    APPROX_INTERVALS: MomentQuantTransformer(mode="approx", approx_version="intervals", parallel=False),
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
        if VERBOSE:
            print(f"Series length = {series_length} | Number of samples = {n_samples}")
        X_ = np.ascontiguousarray(X[:n_samples, :, :series_length])

        n_runs_here = N_RUNS_SMALL_L if series_length <= SMALL_L_REPEAT_THRESHOLD else N_RUNS
        jobs = [(estimator_name, run) for estimator_name in estimators for run in range(n_runs_here)]
        rng.shuffle(jobs)

        for estimator_name, run in jobs:
            estimator = estimators[estimator_name]
            start_time = time.perf_counter()
            estimator.transform(X_)
            end_time = time.perf_counter()
            results.append([series_length, n_samples, estimator_name, run, end_time - start_time])
            last_cooldown = periodic_cooldown(last_cooldown)

path = os.path.join("results", "runtimes", "single_thread_grid")
if not os.path.isdir(path):
    os.makedirs(path)

pd.DataFrame(results, columns=["Series length", "Number of samples", "Estimator", "Run", "Runtime"]).to_csv(
    os.path.join(path, "runtimes.csv"), index=False
)
