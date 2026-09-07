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

from src.estimators.moment_quant import MomentQuantTransformer
from src.estimators.quant_float64 import QuantFloat64
from src.runtimes.utils import MEDIAN_SERIES_LENGTH, N_SAMPLES_SWEEP_GRID
from src.utils import APPROX_INTERVALS, APPROX_SAMPLES, EXACT_INTERVALS, EXACT_SAMPLES, N_RUNS, VERBOSE

seed = 0
rng = np.random.default_rng(seed)
X = rng.normal(size=(max(N_SAMPLES_SWEEP_GRID), 1, MEDIAN_SERIES_LENGTH))

estimators = {
    EXACT_SAMPLES: MomentQuantTransformer(mode="exact", exact_version="samples", parallel=False),
    EXACT_INTERVALS: MomentQuantTransformer(mode="exact", exact_version="intervals", parallel=False),
    APPROX_SAMPLES: MomentQuantTransformer(mode="approx", approx_version="samples", parallel=False),
    APPROX_INTERVALS: MomentQuantTransformer(mode="approx", approx_version="intervals", parallel=False),
    "quant": QuantFloat64(),
}

warmup_n_samples = min(N_SAMPLES_SWEEP_GRID)
X_numpy_warmup = np.ascontiguousarray(X[:warmup_n_samples].copy())
X_torch_warmup = torch.from_numpy(X_numpy_warmup)
for estimator_name, estimator in estimators.items():
    estimator.fit_transform(X_numpy_warmup if estimator_name != "quant" else X_torch_warmup)

results = {}
for estimator_name in estimators:
    results[estimator_name] = []

for i, n_samples in enumerate(N_SAMPLES_SWEEP_GRID):
    if VERBOSE:
        print(
            f"[{i + 1:>{len(str(len(N_SAMPLES_SWEEP_GRID)))}}/{len(N_SAMPLES_SWEEP_GRID)}] "
            f"Number of time series = {n_samples}"
        )

    X_numpy = np.ascontiguousarray(X[:n_samples].copy())
    X_torch = torch.from_numpy(X_numpy)

    run_times = {estimator_name: [None] * N_RUNS for estimator_name in estimators}
    jobs = [(estimator_name, run) for estimator_name in estimators for run in range(N_RUNS)]
    rng.shuffle(jobs)

    for estimator_name, run in jobs:
        estimator = estimators[estimator_name]
        X_input = X_numpy if estimator_name != "quant" else X_torch
        start_time = time.perf_counter()
        estimator.fit_transform(X_input)
        end_time = time.perf_counter()
        run_times[estimator_name][run] = end_time - start_time

    for estimator_name in estimators:
        results[estimator_name].append(run_times[estimator_name])

path = os.path.join("results", "runtimes", "single_thread_fixed_n_timepoints")
if not os.path.isdir(path):
    os.makedirs(path)

for estimator_name, estimator in estimators.items():
    pd.DataFrame(
        data=results[estimator_name],
        index=pd.Index(N_SAMPLES_SWEEP_GRID, name="Number of time series"),
        columns=[f"Run {i}" for i in range(N_RUNS)],
    ).to_csv(os.path.join(path, f"{estimator_name}.csv"))
