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
from aeon.datasets import load_classification
from aeon.datasets.tsc_datasets import redux, univariate_equal_length

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from src.classification.utils import SMOKE_TEST_DATASET_NAMES
from src.estimators.moment_quant import MomentQuantTransformer
from src.estimators.quant_float64 import QuantFloat64
from src.runtimes.utils import APPROX_AUTO, EXACT_AUTO
from src.utils import (
    APPROX_INTERVALS,
    APPROX_SAMPLES,
    EXACT_INTERVALS,
    EXACT_SAMPLES,
    N_RUNS,
    SMOKE_TEST,
    VERBOSE,
    periodic_cooldown,
)

seed = 0
rng = np.random.default_rng(seed)

estimators = {
    EXACT_INTERVALS: MomentQuantTransformer(mode="exact", exact_version="intervals", parallel=False),
    EXACT_SAMPLES: MomentQuantTransformer(mode="exact", exact_version="samples", parallel=False),
    EXACT_AUTO: MomentQuantTransformer(mode="exact", exact_version="auto", parallel=False),
    APPROX_INTERVALS: MomentQuantTransformer(mode="approx", approx_version="intervals", parallel=False),
    APPROX_SAMPLES: MomentQuantTransformer(mode="approx", approx_version="samples", parallel=False),
    APPROX_AUTO: MomentQuantTransformer(mode="approx", approx_version="auto", parallel=False),
    "quant_float64": QuantFloat64(),
}

X_warmup_train_numpy = np.ascontiguousarray(rng.normal(size=(4, 1, 16)))
X_warmup_train_torch = torch.from_numpy(X_warmup_train_numpy)
X_warmup_test_numpy = np.ascontiguousarray(rng.normal(size=(3, 1, 16)))
X_warmup_test_torch = torch.from_numpy(X_warmup_test_numpy)
for estimator_name, estimator in estimators.items():
    X_train_ = X_warmup_train_numpy if estimator_name != "quant_float64" else X_warmup_train_torch
    X_test_ = X_warmup_test_numpy if estimator_name != "quant_float64" else X_warmup_test_torch
    estimator.fit_transform(X_train_)
    estimator.transform(X_test_)
del X_warmup_train_numpy, X_warmup_train_torch, X_warmup_test_numpy, X_warmup_test_torch, X_train_, X_test_

dataset_names = SMOKE_TEST_DATASET_NAMES if SMOKE_TEST else list(univariate_equal_length) + list(redux)
rng.shuffle(dataset_names)

results = []
last_cooldown = time.perf_counter()
for i, name in enumerate(dataset_names):
    if VERBOSE:
        print(f"[{i + 1:>3}/{len(dataset_names)}] {name}")

    X_train, _ = load_classification(name, split="train", load_equal_length=True, load_no_missing=True)
    X_test, _ = load_classification(name, split="test", load_equal_length=True, load_no_missing=True)

    X_train_numpy = np.ascontiguousarray(X_train, dtype=np.float64)
    X_train_torch = torch.from_numpy(X_train_numpy)
    X_test_numpy = np.ascontiguousarray(X_test, dtype=np.float64)
    X_test_torch = torch.from_numpy(X_test_numpy)

    jobs = [(estimator_name, run) for estimator_name in estimators for run in range(N_RUNS)]
    rng.shuffle(jobs)

    for estimator_name, run in jobs:
        estimator = estimators[estimator_name]
        X_train_ = X_train_numpy if estimator_name != "quant_float64" else X_train_torch
        X_test_ = X_test_numpy if estimator_name != "quant_float64" else X_test_torch

        start_time = time.perf_counter()
        estimator.fit_transform(X_train_)
        end_time = time.perf_counter()
        results.append(
            [name, "train", X_train_numpy.shape[0], X_train_numpy.shape[2], estimator_name, run, end_time - start_time]
        )

        start_time = time.perf_counter()
        estimator.transform(X_test_)
        end_time = time.perf_counter()
        results.append(
            [name, "test", X_test_numpy.shape[0], X_test_numpy.shape[2], estimator_name, run, end_time - start_time]
        )
        last_cooldown = periodic_cooldown(last_cooldown)

path = os.path.join("results", "runtimes", "single_thread_ucr142")
if not os.path.isdir(path):
    os.makedirs(path)

pd.DataFrame(
    results,
    columns=["Dataset", "Split", "Number of samples", "Series length", "Estimator", "Run", "Runtime"],
).to_csv(os.path.join(path, "runtimes.csv"), index=False)
