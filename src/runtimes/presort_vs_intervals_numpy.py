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

from src.estimators.moment_quant import MomentQuantTransformer, _batch_exact_features_intervals
from src.runtimes.utils import LARGE_L_GRID, LARGE_N_SAMPLES
from src.utils import DEPTH, DIV, N_RUNS, VERBOSE, periodic_cooldown


def _make_grouped_intervals(est, length, depth):
    starts, ends, _ = est._make_intervals(length)
    exponent = min(depth, int(np.log2(length)) + 1)

    groups = []
    flat_count = 0

    for d in range(exponent):
        n = 2**d
        s, e = starts[flat_count : flat_count + n], ends[flat_count : flat_count + n]

        sizes = e - s
        cum = np.zeros(n + 1, dtype=np.int64)
        cum[1:] = np.cumsum(sizes)
        full_lookup = np.repeat(np.arange(n, dtype=np.int32), sizes)
        groups.append({"full_lookup": full_lookup, "cum": cum, "start_idx": flat_count, "n": n})
        flat_count += n

        if n > 1 and np.median(sizes) > 1:
            s_shift, e_shift = starts[flat_count : flat_count + n - 1], ends[flat_count : flat_count + n - 1]

            sizes_shift = e_shift - s_shift
            lo, hi = int(s_shift[0]), int(e_shift[-1])
            cum_shift = np.zeros(n, dtype=np.int64)
            cum_shift[1:] = np.cumsum(sizes_shift)
            full_lookup_shift = np.full(length, -1, dtype=np.int32)
            full_lookup_shift[lo:hi] = np.repeat(np.arange(n - 1, dtype=np.int32), sizes_shift)
            groups.append(
                {
                    "full_lookup": full_lookup_shift,
                    "cum": cum_shift,
                    "start_idx": flat_count,
                    "n": n - 1,
                }
            )
            flat_count += n - 1

    return starts, ends, groups


def _features_presort(X, starts, ends, kind, offsets, q_positions, center_mask, groups, out=None):
    n_samples, length = X.shape
    total = int(offsets[-1])
    if out is None:
        out = np.empty((n_samples, total), dtype=np.float64)

    perm = np.argsort(X, axis=-1)

    for g in groups:
        full_lookup = g["full_lookup"]
        cum = g["cum"]
        start_idx = g["start_idx"]
        n_local = g["n"]

        bucket_id_seq = full_lookup[perm]
        order = np.argsort(bucket_id_seq, axis=-1, kind="stable")
        sorted_perm_idx = np.take_along_axis(perm, order, axis=-1)

        n_excluded = length - int(cum[-1])
        valid_perm_idx = sorted_perm_idx[:, n_excluded:]
        sorted_vals = np.take_along_axis(X, valid_perm_idx, axis=-1)

        for b in range(n_local):
            i = start_idx + b
            o0, o1 = int(offsets[i]), int(offsets[i + 1])
            seg_sorted = sorted_vals[:, int(cum[b]) : int(cum[b + 1])]
            m = seg_sorted.shape[1]

            if kind[i] == 0:
                out[:, o0] = seg_sorted[:, 0]
                continue

            pos = q_positions[o0:o1] * (m - 1)
            lo = np.floor(pos).astype(np.int64)
            hi = np.minimum(lo + 1, m - 1)
            frac = pos - lo
            a = seg_sorted[:, lo]
            b = seg_sorted[:, hi]
            diff = b - a
            vals = np.where(frac >= 0.5, b - diff * (1.0 - frac), a + diff * frac)

            cmask = center_mask[o0:o1]
            if np.any(cmask):
                seg_mean = seg_sorted.mean(axis=-1, keepdims=True)
                vals[:, cmask] -= seg_mean

            out[:, o0:o1] = vals

    return out


def build_layouts_and_groups(length, depth, div):
    est = MomentQuantTransformer(depth=depth, div=div, mode="exact", exact_version="intervals")
    est.fit(np.zeros((1, 1, length), dtype=np.float64))

    entries = []
    for layout, repr_length in zip(est.layouts_, est._representation_lengths()):
        _, _, groups = _make_grouped_intervals(est, repr_length, depth)
        entries.append({"layout": layout, "groups": groups})
    return est, entries


def intervals_transform(X, entries, n_output_features):
    X_flat = X[:, 0, :]
    n_samples = X_flat.shape[0]
    out = np.empty((n_samples, n_output_features), dtype=np.float64)
    offset = 0
    for repr_func, entry in zip(MomentQuantTransformer._representation_funcs(), entries):
        Z = np.ascontiguousarray(repr_func(X_flat), dtype=np.float64)
        layout = entry["layout"]
        width = int(layout["offsets"][-1])
        _batch_exact_features_intervals(
            Z,
            layout["starts"],
            layout["ends"],
            layout["kind"],
            layout["offsets"],
            layout["q_positions"],
            layout["center_mask"],
            out=out[:, offset : offset + width],
        )
        offset += width
    return out


def presort_transform(X, entries, n_output_features):
    X_flat = X[:, 0, :]
    n_samples = X_flat.shape[0]
    out = np.empty((n_samples, n_output_features), dtype=np.float64)
    offset = 0
    for repr_func, entry in zip(MomentQuantTransformer._representation_funcs(), entries):
        Z = np.ascontiguousarray(repr_func(X_flat), dtype=np.float64)
        layout = entry["layout"]
        width = int(layout["offsets"][-1])
        _features_presort(
            Z,
            layout["starts"],
            layout["ends"],
            layout["kind"],
            layout["offsets"],
            layout["q_positions"],
            layout["center_mask"],
            entry["groups"],
            out=out[:, offset : offset + width],
        )
        offset += width
    return out


def verify_correctness(rng):
    test_lengths = [15, 97, 130, 1000, 2048]
    for length in test_lengths:
        n_samples = 20
        X = rng.normal(size=(n_samples, 1, length))

        est, entries = build_layouts_and_groups(length, DEPTH, DIV)
        out_ref = est.transform(X)
        out_intervals = intervals_transform(X, entries, est.n_output_features_)
        out_presort = presort_transform(X, entries, est.n_output_features_)

        d_ri = np.max(np.abs(out_ref - out_intervals))
        d_rp = np.max(np.abs(out_ref - out_presort))

        assert d_ri < 1e-10, f"intervals harness disagrees with the real estimator at length={length}"
        assert d_rp < 1e-8, f"presort disagrees with the real estimator at length={length}"


def main():
    rng = np.random.default_rng(0)
    verify_correctness(rng)

    series_length_grid = LARGE_L_GRID
    n_samples = LARGE_N_SAMPLES

    X_full = rng.normal(size=(n_samples, 1, max(series_length_grid)))

    warmup_L = min(series_length_grid)
    X_warmup = np.ascontiguousarray(X_full[:10, :, :warmup_L])
    est_warmup, entries_warmup = build_layouts_and_groups(warmup_L, DEPTH, DIV)
    intervals_transform(X_warmup, entries_warmup, est_warmup.n_output_features_)
    presort_transform(X_warmup, entries_warmup, est_warmup.n_output_features_)

    methods = {
        "intervals": intervals_transform,
        "presort": presort_transform,
    }

    results = []
    last_cooldown = time.perf_counter()

    for series_length in series_length_grid:
        if VERBOSE:
            print(f"Series length = {series_length}")
        X_ = np.ascontiguousarray(X_full[:, :, :series_length])

        est, entries = build_layouts_and_groups(series_length, DEPTH, DIV)
        n_features = est.n_output_features_

        jobs = [(method, run) for method in methods for run in range(N_RUNS)]
        rng.shuffle(jobs)

        for method, run in jobs:
            start_time = time.perf_counter()
            methods[method](X_, entries, n_features)
            end_time = time.perf_counter()
            results.append([series_length, method, run, end_time - start_time])
            last_cooldown = periodic_cooldown(last_cooldown)

    df = pd.DataFrame(results, columns=["Series length", "Method", "Run", "Runtime"])

    path = os.path.join("results", "runtimes", "presort_vs_intervals_numpy")
    os.makedirs(path, exist_ok=True)
    df.to_csv(os.path.join(path, "runtimes.csv"), index=False)

    mins = df.groupby(["Series length", "Method"])["Runtime"].min().unstack("Method")
    mins["presort / intervals"] = mins["presort"] / mins["intervals"]
    print()
    print("=" * 100)
    print("MIN-over-repeats summary (seconds)")
    print("=" * 100)
    print(mins.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
