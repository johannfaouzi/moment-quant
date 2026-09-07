"""
Three hand-written quicksort variants, JIT-compiled with Numba, isolating the mechanism behind
the sort/argsort runtime gap discussed in the paper without any SIMD-dispatch confound (Numba does
not invoke a vectorized fast path -- AVX2/AVX-512 x86-simd-sort, or Highway/NEON QSort -- for a
branchy introsort loop like this one).

All three share the *same* median-of-three, insertion-sort-cutoff introsort skeleton (mirroring
numpy's quicksort_/aquicksort_ in numpy/_core/src/npysort/quicksort.cpp), differing only in what
gets compared and what gets swapped:

  direct    - plain value sort. Comparisons and swaps both touch `a` directly, sequentially.
              This is numpy's quicksort_ (the kernel behind numpy.sort).

  indirect  - argsort via index permutation. `a` is read-only; only `idx` is permuted.
              Comparisons go through a[idx[i]] (indirected/gather). This is numpy's aquicksort_
              (the kernel behind numpy.argsort).

  keyvalue  - argsort via paired swap. `a` IS permuted (like direct), and `idx` is carried along
              in lockstep on every swap. Comparisons stay direct/sequential on `a`, exactly like
              `direct`. This is PyTorch's sort_kernel_impl strategy (used by both torch.sort and
              torch.argsort).

Hypothesis under test: `indirect` should be markedly slower than `direct` because of
cache-unfriendly gather accesses, even though it performs the exact same number of comparisons and
swaps. `keyvalue` should stay close to `direct` (same access pattern, just carrying a second array
along), showing that the indirection -- not the "two arrays" idea itself -- is what costs.
"""

import numpy as np
from numba import njit

SMALL_QUICKSORT = 15


@njit(cache=True)
def quicksort_direct(a):
    n = a.shape[0]
    pl = 0
    pr = n - 1
    stack_l = np.empty(128, dtype=np.int64)
    stack_r = np.empty(128, dtype=np.int64)
    sp = 0
    n_cmp = 0
    n_swap = 0

    while True:
        while pr - pl > SMALL_QUICKSORT:
            pm = pl + ((pr - pl) >> 1)
            n_cmp += 1
            if a[pm] < a[pl]:
                a[pm], a[pl] = a[pl], a[pm]
                n_swap += 1
            n_cmp += 1
            if a[pr] < a[pm]:
                a[pr], a[pm] = a[pm], a[pr]
                n_swap += 1
            n_cmp += 1
            if a[pm] < a[pl]:
                a[pm], a[pl] = a[pl], a[pm]
                n_swap += 1
            vp = a[pm]
            pi = pl
            pj = pr - 1
            a[pm], a[pj] = a[pj], a[pm]
            n_swap += 1
            while True:
                pi += 1
                n_cmp += 1
                while a[pi] < vp:
                    pi += 1
                    n_cmp += 1
                pj -= 1
                n_cmp += 1
                while vp < a[pj]:
                    pj -= 1
                    n_cmp += 1
                if pi >= pj:
                    break
                a[pi], a[pj] = a[pj], a[pi]
                n_swap += 1
            a[pi], a[pr - 1] = a[pr - 1], a[pi]
            n_swap += 1
            if pi - pl < pr - pi:
                stack_l[sp] = pi + 1
                stack_r[sp] = pr
                sp += 1
                pr = pi - 1
            else:
                stack_l[sp] = pl
                stack_r[sp] = pi - 1
                sp += 1
                pl = pi + 1

        pi = pl + 1
        while pi <= pr:
            vp = a[pi]
            pj = pi
            while pj > pl:
                n_cmp += 1
                if not (vp < a[pj - 1]):
                    break
                a[pj] = a[pj - 1]
                pj -= 1
            a[pj] = vp
            pi += 1

        if sp == 0:
            break
        sp -= 1
        pl = stack_l[sp]
        pr = stack_r[sp]

    return n_cmp, n_swap


@njit(cache=True)
def quicksort_indirect(a, idx):
    """numpy aquicksort_ style: `a` is read-only, only `idx` is permuted, comparisons go through
    a[idx[i]]."""
    n = a.shape[0]
    for i in range(n):
        idx[i] = i
    pl = 0
    pr = n - 1
    stack_l = np.empty(128, dtype=np.int64)
    stack_r = np.empty(128, dtype=np.int64)
    sp = 0
    n_cmp = 0
    n_swap = 0

    while True:
        while pr - pl > SMALL_QUICKSORT:
            pm = pl + ((pr - pl) >> 1)
            n_cmp += 1
            if a[idx[pm]] < a[idx[pl]]:
                idx[pm], idx[pl] = idx[pl], idx[pm]
                n_swap += 1
            n_cmp += 1
            if a[idx[pr]] < a[idx[pm]]:
                idx[pr], idx[pm] = idx[pm], idx[pr]
                n_swap += 1
            n_cmp += 1
            if a[idx[pm]] < a[idx[pl]]:
                idx[pm], idx[pl] = idx[pl], idx[pm]
                n_swap += 1
            vp = a[idx[pm]]
            pi = pl
            pj = pr - 1
            idx[pm], idx[pj] = idx[pj], idx[pm]
            n_swap += 1
            while True:
                pi += 1
                n_cmp += 1
                while a[idx[pi]] < vp:
                    pi += 1
                    n_cmp += 1
                pj -= 1
                n_cmp += 1
                while vp < a[idx[pj]]:
                    pj -= 1
                    n_cmp += 1
                if pi >= pj:
                    break
                idx[pi], idx[pj] = idx[pj], idx[pi]
                n_swap += 1
            idx[pi], idx[pr - 1] = idx[pr - 1], idx[pi]
            n_swap += 1
            if pi - pl < pr - pi:
                stack_l[sp] = pi + 1
                stack_r[sp] = pr
                sp += 1
                pr = pi - 1
            else:
                stack_l[sp] = pl
                stack_r[sp] = pi - 1
                sp += 1
                pl = pi + 1

        pi = pl + 1
        while pi <= pr:
            vi = idx[pi]
            vp = a[vi]
            pj = pi
            while pj > pl:
                n_cmp += 1
                if not (vp < a[idx[pj - 1]]):
                    break
                idx[pj] = idx[pj - 1]
                pj -= 1
            idx[pj] = vi
            pi += 1

        if sp == 0:
            break
        sp -= 1
        pl = stack_l[sp]
        pr = stack_r[sp]

    return n_cmp, n_swap


@njit(cache=True)
def quicksort_keyvalue(a, idx):
    """PyTorch sort_kernel_impl style: `a` IS permuted (like direct), `idx` carried along on every
    swap. Comparisons stay direct/sequential on `a`."""
    n = a.shape[0]
    for i in range(n):
        idx[i] = i
    pl = 0
    pr = n - 1
    stack_l = np.empty(128, dtype=np.int64)
    stack_r = np.empty(128, dtype=np.int64)
    sp = 0
    n_cmp = 0
    n_swap = 0

    while True:
        while pr - pl > SMALL_QUICKSORT:
            pm = pl + ((pr - pl) >> 1)
            n_cmp += 1
            if a[pm] < a[pl]:
                a[pm], a[pl] = a[pl], a[pm]
                idx[pm], idx[pl] = idx[pl], idx[pm]
                n_swap += 1
            n_cmp += 1
            if a[pr] < a[pm]:
                a[pr], a[pm] = a[pm], a[pr]
                idx[pr], idx[pm] = idx[pm], idx[pr]
                n_swap += 1
            n_cmp += 1
            if a[pm] < a[pl]:
                a[pm], a[pl] = a[pl], a[pm]
                idx[pm], idx[pl] = idx[pl], idx[pm]
                n_swap += 1
            vp = a[pm]
            pi = pl
            pj = pr - 1
            a[pm], a[pj] = a[pj], a[pm]
            idx[pm], idx[pj] = idx[pj], idx[pm]
            n_swap += 1
            while True:
                pi += 1
                n_cmp += 1
                while a[pi] < vp:
                    pi += 1
                    n_cmp += 1
                pj -= 1
                n_cmp += 1
                while vp < a[pj]:
                    pj -= 1
                    n_cmp += 1
                if pi >= pj:
                    break
                a[pi], a[pj] = a[pj], a[pi]
                idx[pi], idx[pj] = idx[pj], idx[pi]
                n_swap += 1
            a[pi], a[pr - 1] = a[pr - 1], a[pi]
            idx[pi], idx[pr - 1] = idx[pr - 1], idx[pi]
            n_swap += 1
            if pi - pl < pr - pi:
                stack_l[sp] = pi + 1
                stack_r[sp] = pr
                sp += 1
                pr = pi - 1
            else:
                stack_l[sp] = pl
                stack_r[sp] = pi - 1
                sp += 1
                pl = pi + 1

        pi = pl + 1
        while pi <= pr:
            vp = a[pi]
            vi = idx[pi]
            pj = pi
            while pj > pl:
                n_cmp += 1
                if not (vp < a[pj - 1]):
                    break
                a[pj] = a[pj - 1]
                idx[pj] = idx[pj - 1]
                pj -= 1
            a[pj] = vp
            idx[pj] = vi
            pi += 1

        if sp == 0:
            break
        sp -= 1
        pl = stack_l[sp]
        pr = stack_r[sp]

    return n_cmp, n_swap


def verify_correctness(rng):
    test_lengths = [0, 1, 2, 5, 16, 17, 100, 1000, 12345]
    for n in test_lengths:
        a = rng.random(n)
        expected_sorted = np.sort(a)

        a1 = a.copy()
        quicksort_direct(a1)
        assert np.array_equal(a1, expected_sorted), f"direct failed at n={n}"

        a2 = a.copy()
        idx2 = np.empty(n, dtype=np.int64)
        quicksort_indirect(a2, idx2)
        assert np.array_equal(a[idx2], expected_sorted), f"indirect failed at n={n}"

        a3 = a.copy()
        idx3 = np.empty(n, dtype=np.int64)
        quicksort_keyvalue(a3, idx3)
        assert np.array_equal(a3, expected_sorted), f"keyvalue values failed at n={n}"
        assert np.array_equal(a[idx3], expected_sorted), f"keyvalue permutation failed at n={n}"


if __name__ == "__main__":
    verify_correctness(np.random.default_rng(0))
    print("All correctness checks passed.")
