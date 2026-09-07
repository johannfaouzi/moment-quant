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

from src.analyses.quicksort_variants_numba import (
    quicksort_direct,
    quicksort_indirect,
    quicksort_keyvalue,
    verify_correctness,
)
from src.utils import SMOKE_TEST, VERBOSE, periodic_cooldown

# NOTE: intentionally not using the shared N_RUNS from src.utils here (see N_RUNS_MAIN /
# N_RUNS_TAIL below) -- the run count needed for this analysis differs by regime, and bumping the
# shared constant would slow down every other stage in the pipeline.
N_RUNS_MAIN = 40  # l=2048 (see below) showed a bimodal, noisy timing distribution across the
# default 10 runs -- roughly half the repeats landed in a "fast" cluster and
# half in a "slow" one, rather than a single tight spread. Our best guess is
# that this comes from the job shuffling below: each repeat of a given l lands
# at a different, effectively random point in the single shuffled sequence of
# every (l, variant, run) job, so some repeats happen to run right after a much
# larger l (CPU still in a different frequency/thermal state) and others after
# a much smaller one. More repeats should average this out and give a more
# reliable minimum, which is why this is bumped well above the pipeline's
# default of 10.
N_RUNS_TAIL = 12  # Fewer repeats for the l >= 2**16 tail below: those sizes are added purely to
# show where the indirect/direct cache-miss effect is headed asymptotically and
# never actually occur in this paper's pipeline, so a precise per-l estimate
# matters less there than the overall trend.

# l ranges over powers of two from 2**2 to 2**15, matching the paper's l notation and the grid used
# by src.analyses.sort_argsort_numpy / src.analyses.sort_argsort_torch: as low as l=4 (close to the
# shortest subseries that can appear after repeated depth-halving from a base series -- the deepest
# depth level divides l by up to 2**5=32) up to l=32768, the largest series length used elsewhere in
# the paper (Table E, Table A). A handful of much larger l values (2**16 to 2**22) are appended on
# top of that: they never occur in this paper's pipeline, but they let the indirect/direct ratio run
# well past the point where it's still visibly climbing at l=32768, closer to the "stable several-fold
# gap" scale that pure cache-miss effects are expected to reach asymptotically.
#
# (l, inner, n_runs) -- inner amortizes per-call/timer-resolution overhead at small l and is scaled
# down as l grows so total runtime stays reasonable; at the small end, inner is large to amortize
# python/numba call and array-allocation overhead against the (tiny) actual sort cost. n_runs
# independent samples are then taken per (variant, l) cell.
L_GRID = [
    (2**2, 50000, N_RUNS_MAIN),  # 4
    (2**3, 50000, N_RUNS_MAIN),  # 8
    (2**4, 30000, N_RUNS_MAIN),  # 16
    (2**5, 20000, N_RUNS_MAIN),  # 32
    (2**6, 10000, N_RUNS_MAIN),  # 64
    (2**7, 5000, N_RUNS_MAIN),  # 128
    (2**8, 2000, N_RUNS_MAIN),  # 256
    (2**9, 1000, N_RUNS_MAIN),  # 512
    (2**10, 500, N_RUNS_MAIN),  # 1024
    (2**11, 200, N_RUNS_MAIN),  # 2048
    (2**12, 100, N_RUNS_MAIN),  # 4096
    (2**13, 50, N_RUNS_MAIN),  # 8192
    (2**14, 25, N_RUNS_MAIN),  # 16384
    (2**15, 12, N_RUNS_MAIN),  # 32768
    (2**16, 6, N_RUNS_TAIL),  # 65536
    (2**18, 2, N_RUNS_TAIL),  # 262144
    (2**20, 1, N_RUNS_TAIL),  # 1048576
    (2**22, 1, N_RUNS_TAIL),  # 4194304
]

if SMOKE_TEST:
    L_GRID = [(2**2, 5000, 2), (2**10, 100, 2), (2**15, 3, 2), (2**22, 1, 2)]

VARIANTS = ["direct", "indirect", "keyvalue"]


def _run_direct(base):
    a = base.copy()
    return quicksort_direct(a)


def _run_indirect(base, l):
    a = base.copy()
    idx = np.empty(l, dtype=np.int64)
    return quicksort_indirect(a, idx)


def _run_keyvalue(base, l):
    a = base.copy()
    idx = np.empty(l, dtype=np.int64)
    return quicksort_keyvalue(a, idx)


def _time_one(fn, inner):
    t0 = time.perf_counter()
    for _ in range(inner):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) / inner


def main():
    rng = np.random.default_rng(42)

    verify_correctness(rng)

    # warm up JIT compilation before any timing
    warm = rng.random(1000)
    quicksort_direct(warm.copy())
    quicksort_indirect(warm.copy(), np.empty(1000, dtype=np.int64))
    quicksort_keyvalue(warm.copy(), np.empty(1000, dtype=np.int64))

    counts = []
    for l, _, _ in L_GRID:
        base = rng.random(l)

        a = base.copy()
        c_d, s_d = quicksort_direct(a)
        a = base.copy()
        idx = np.empty(l, dtype=np.int64)
        c_i, s_i = quicksort_indirect(a, idx)
        a = base.copy()
        idx = np.empty(l, dtype=np.int64)
        c_k, s_k = quicksort_keyvalue(a, idx)

        counts.append([l, "direct", c_d, s_d])
        counts.append([l, "indirect", c_i, s_i])
        counts.append([l, "keyvalue", c_k, s_k])
        if VERBOSE:
            print(
                f"l={l:>7d}  cmp(d/i/k)={c_d:>10d}/{c_i:>10d}/{c_k:>10d}  " f"swap(d/i/k)={s_d:>9d}/{s_i:>9d}/{s_k:>9d}"
            )

    jobs = [(l, inner, variant, run) for l, inner, n_runs in L_GRID for variant in VARIANTS for run in range(n_runs)]
    rng.shuffle(jobs)

    results = []
    last_cooldown = time.perf_counter()

    for j, (l, inner, variant, run) in enumerate(jobs):
        base = rng.random(l)
        if variant == "direct":
            t = _time_one(lambda: _run_direct(base), inner)
        elif variant == "indirect":
            t = _time_one(lambda: _run_indirect(base, l), inner)
        else:
            t = _time_one(lambda: _run_keyvalue(base, l), inner)
        results.append([l, variant, run, t])
        last_cooldown = periodic_cooldown(last_cooldown)
        # Progress only, printed periodically: with n_runs bumped up (see N_RUNS_MAIN/N_RUNS_TAIL
        # above), this loop runs thousands of tiny jobs, so printing every single one would flood
        # the console with lines each carrying almost no information (these timings are a few
        # microseconds to a couple of seconds).
        if VERBOSE and (j % 100 == 0 or j == len(jobs) - 1):
            print(
                f"[{j + 1:>{len(str(len(jobs)))}}/{len(jobs)}] l={l:>7d}  variant={variant:<8}  "
                f"run={run:02d}  t={t * 1e3:9.4f} ms"
            )

    path = os.path.join("results", "analyses", "sort_argsort_numba_isolation")
    os.makedirs(path, exist_ok=True)
    pd.DataFrame(results, columns=["l", "Variant", "Run", "Runtime"]).to_csv(
        os.path.join(path, "runtimes.csv"), index=False
    )
    pd.DataFrame(counts, columns=["l", "Variant", "Comparisons", "Swaps"]).to_csv(
        os.path.join(path, "counts.csv"), index=False
    )


if __name__ == "__main__":
    main()
