import os
import time

import numpy as np
import pandas as pd
import torch

from src.estimators.moment_quant import MomentQuantTransformer
from src.estimators.quant_float64 import QuantFloat64
from src.estimators.quant_numpy import QuantNumpy
from src.estimators.quant_original import Quant
from src.runtimes.utils import LARGE_L_GRID, LARGE_N_SAMPLES
from src.utils import DEPTH, DIV, N_RUNS, VERBOSE, periodic_cooldown

series_length_grid = LARGE_L_GRID
n_samples = LARGE_N_SAMPLES

QUANT_NUMPY_MAX_L = 4096

quant_cls_by_dtype = {"float32": Quant, "float64": QuantFloat64}


def bench_quant(quant_cls, X_torch, num_threads, n_repeats=N_RUNS):
    torch.set_num_threads(num_threads)
    model = quant_cls(depth=DEPTH, div=DIV)
    Y_dummy = torch.zeros(X_torch.shape[0])
    with torch.no_grad():
        model.fit_transform(X_torch, Y_dummy)
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            model.transform(X_torch)
            times.append(time.perf_counter() - t0)
    return min(times)


def bench_moment_quant(X_numpy, n_repeats=N_RUNS):
    est = MomentQuantTransformer(depth=DEPTH, div=DIV, mode="exact", exact_version="intervals", parallel=False)
    est.fit(X_numpy)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        est.transform(X_numpy)
        times.append(time.perf_counter() - t0)
    return min(times)


def bench_quant_numpy(X_numpy, n_repeats=N_RUNS):
    model = QuantNumpy(depth=DEPTH, div=DIV)
    model.fit_transform(X_numpy)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        model.transform(X_numpy)
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    rng = np.random.default_rng(0)
    n_cores = os.cpu_count()

    print(
        f"{'L':>7} | {'dtype':>7} | {'quant 1-thread':>14} | {'quant default':>13} | {'moment_quant':>12} | "
        f"{'quant_numpy':>11} | {'speedup(1t)':>11} | {'speedup(def)':>12} | {'speedup(np)':>11}"
    )
    print("-" * 130)

    results = []
    last_cooldown = time.perf_counter()
    for L in series_length_grid:
        for dtype_name, torch_dtype in [("float32", torch.float32), ("float64", torch.float64)]:
            quant_cls = quant_cls_by_dtype[dtype_name]
            X_numpy_native = rng.normal(size=(n_samples, 1, L))
            X_torch = torch.from_numpy(X_numpy_native).to(dtype=torch_dtype)
            X_numpy_f64 = X_numpy_native.astype(np.float64)
            X_numpy_matching = X_numpy_native.astype(np.float32) if dtype_name == "float32" else X_numpy_f64

            t_quant_1 = bench_quant(quant_cls, X_torch, num_threads=1)
            t_quant_def = bench_quant(quant_cls, X_torch, num_threads=n_cores)
            t_moment = bench_moment_quant(X_numpy_f64)

            if L <= QUANT_NUMPY_MAX_L:
                t_quant_numpy = bench_quant_numpy(X_numpy_matching)
                quant_numpy_str = f"{t_quant_numpy:>11.4f}"
                speedup_np_str = f"{t_quant_numpy / t_moment:>10.2f}x"
            else:
                t_quant_numpy = float("nan")
                quant_numpy_str = f"{'skipped':>11}"
                speedup_np_str = f"{'--':>11}"

            print(
                f"{L:>7} | {dtype_name:>7} | {t_quant_1:>14.4f} | {t_quant_def:>13.4f} | {t_moment:>12.4f} | "
                f"{quant_numpy_str} | {t_quant_1 / t_moment:>10.2f}x | {t_quant_def / t_moment:>11.2f}x | "
                f"{speedup_np_str}"
            )

            results.append([L, dtype_name, t_quant_1, t_quant_def, t_moment, t_quant_numpy, n_cores])
            last_cooldown = periodic_cooldown(last_cooldown)

    path = os.path.join("results", "runtimes", "quant_pytorch_vs_moment_quant_numpy")
    os.makedirs(path, exist_ok=True)
    out_path = os.path.join(path, "runtimes.csv")
    pd.DataFrame(
        results,
        columns=[
            "Series length",
            "Dtype",
            "Runtime (quant, 1 thread)",
            "Runtime (quant, default threads)",
            "Runtime (moment_quant)",
            "Runtime (quant_numpy)",
            "Default threads (n_cores)",
        ],
    ).to_csv(out_path, index=False)
    if VERBOSE:
        print(f"Saved {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
