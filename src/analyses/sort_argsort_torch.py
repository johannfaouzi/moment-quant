import os
import time

import numpy as np
import pandas as pd
import torch

from src.utils import N_RUNS, SMOKE_TEST, VERBOSE, periodic_cooldown

# Real torch.sort() vs torch.argsort() as a function of series length l, mirroring
# src.analyses.sort_argsort_numpy (same l grid) so the two are directly comparable. PyTorch's sort
# kernel always produces both a sorted-values tensor and a permutation-indices tensor internally
# (the "keyvalue" strategy in src.analyses.quicksort_variants_numba), for both torch.sort and
# torch.argsort, unlike numpy where `sort` and `argsort` are two different kernels (quicksort_ vs
# aquicksort_).
#
# l ranges over powers of two from 2**2 to 2**15, matching the paper's l notation and covering the
# range of lengths actually encountered: as low as l=4 (close to the shortest subseries that can
# appear after repeated depth-halving from a base series -- the deepest depth level divides l by up
# to 2**5=32) up to l=32768, the largest series length used elsewhere in the paper (Table E, Table A).
#
# (l, inner) -- inner amortizes per-call/timer-resolution overhead at small l and is scaled down as
# l grows so total runtime stays reasonable; N_RUNS independent samples (from src.utils) are then
# taken per (dtype, l, method) cell.
L_GRID = [
    (2**2, 50000),  # 4
    (2**3, 50000),  # 8
    (2**4, 30000),  # 16
    (2**5, 20000),  # 32
    (2**6, 10000),  # 64
    (2**7, 5000),  # 128
    (2**8, 2000),  # 256
    (2**9, 1000),  # 512
    (2**10, 500),  # 1024
    (2**11, 200),  # 2048
    (2**12, 100),  # 4096
    (2**13, 50),  # 8192
    (2**14, 25),  # 16384
    (2**15, 12),  # 32768
]
DTYPES = [torch.float32, torch.float64]

if SMOKE_TEST:
    L_GRID = [(2**2, 5000), (2**10, 100), (2**15, 3)]

METHODS = {
    "sort": lambda a: torch.sort(a),
    "argsort": lambda a: torch.argsort(a),
}


def _time_one(func, arr, inner):
    a = arr.clone()
    t0 = time.perf_counter()
    for _ in range(inner):
        func(a)
    t1 = time.perf_counter()
    return (t1 - t0) / inner


def run_benchmark(n_threads):
    torch.set_num_threads(n_threads)

    rng = np.random.default_rng(12345)

    jobs = [
        (dtype, l, inner, method, run)
        for dtype in DTYPES
        for l, inner in L_GRID
        for method in METHODS
        for run in range(N_RUNS)
    ]
    rng.shuffle(jobs)

    results = []
    last_cooldown = time.perf_counter()

    for j, (dtype, l, inner, method, run) in enumerate(jobs):
        arr = torch.from_numpy(rng.random(l)).to(dtype)
        t = _time_one(METHODS[method], arr, inner)
        results.append([str(dtype).replace("torch.", ""), l, method, run, t])
        last_cooldown = periodic_cooldown(last_cooldown)
        # Progress only, printed periodically: N_RUNS x every (dtype, l, method) combination adds
        # up to hundreds of tiny jobs, so printing every single one would flood the console with
        # lines each carrying almost no information (these timings are microseconds to ~2 ms).
        if VERBOSE and (j % 50 == 0 or j == len(jobs) - 1):
            print(
                f"[{j + 1:>{len(str(len(jobs)))}}/{len(jobs)}] [{str(dtype):>13}] "
                f"l={l:>6d} inner={inner:<5d} method={method:<7} run={run:02d}  "
                f"t={t * 1e3:9.4f} ms  (threads={torch.get_num_threads()})"
            )

    out = pd.DataFrame(results, columns=["Dtype", "l", "Method", "Run", "Runtime"])
    path = os.path.join("results", "analyses", "sort_argsort_torch")
    os.makedirs(path, exist_ok=True)
    out.to_csv(os.path.join(path, f"runtimes_threads_{n_threads}.csv"), index=False)


if __name__ == "__main__":
    run_benchmark(n_threads=1)
