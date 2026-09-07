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
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from src.estimators.quant_float64 import QuantFloat64
from src.utils import DEPTH, DIV, N_RUNS, N_SAMPLES_GRID, SINGLE_THREAD_L_GRID, VERBOSE, periodic_cooldown

QUANT_FLOAT64 = "quant_float64"

full_series_length_grid = SINGLE_THREAD_L_GRID
n_samples_grid = N_SAMPLES_GRID

seed = 0
rng = np.random.default_rng(seed)
X = rng.normal(size=(max(n_samples_grid), 1, max(full_series_length_grid)))

warmup_L, warmup_n = min(full_series_length_grid), min(n_samples_grid)
X_warmup = torch.from_numpy(np.ascontiguousarray(X[:warmup_n, :, :warmup_L]))
warmup_model = QuantFloat64(depth=DEPTH, div=DIV)
warmup_model.fit_transform(X_warmup)
warmup_model.transform(X_warmup)

results = []
last_cooldown = time.perf_counter()

for series_length in full_series_length_grid:
    X_fit = torch.from_numpy(np.ascontiguousarray(X[:1, :, :series_length]))
    model = QuantFloat64(depth=DEPTH, div=DIV)
    model.fit_transform(X_fit)

    for n_samples in n_samples_grid:
        if VERBOSE:
            print(f"Series length = {series_length} | Number of samples = {n_samples}")
        X_ = torch.from_numpy(np.ascontiguousarray(X[:n_samples, :, :series_length]))

        for run in range(N_RUNS):
            start_time = time.perf_counter()
            model.transform(X_)
            end_time = time.perf_counter()
            results.append([series_length, n_samples, QUANT_FLOAT64, run, end_time - start_time])
            last_cooldown = periodic_cooldown(last_cooldown)

path = os.path.join("results", "runtimes", "single_thread_quant_grid")
if not os.path.isdir(path):
    os.makedirs(path)

pd.DataFrame(results, columns=["Series length", "Number of samples", "Estimator", "Run", "Runtime"]).to_csv(
    os.path.join(path, "runtimes.csv"), index=False
)
