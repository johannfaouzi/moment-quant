import os
import time

import numpy as np
import pandas as pd
import torch

from src.estimators.moment_quant import MomentQuantTransformer
from src.runtimes.utils import LARGE_L_GRID, LARGE_N_SAMPLES
from src.utils import DEPTH, DIV, N_RUNS, VERBOSE, periodic_cooldown

bucket_dtype = torch.int16


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
        full_lookup = np.repeat(np.arange(n, dtype=np.int64), sizes)
        groups.append(
            {
                "full_lookup": torch.from_numpy(full_lookup).to(bucket_dtype),
                "cum": torch.from_numpy(cum),
                "start_idx": flat_count,
                "n": n,
            }
        )
        flat_count += n

        if n > 1 and np.median(sizes) > 1:
            s_shift, e_shift = starts[flat_count : flat_count + n - 1], ends[flat_count : flat_count + n - 1]

            sizes_shift = e_shift - s_shift
            lo, hi = int(s_shift[0]), int(e_shift[-1])
            cum_shift = np.zeros(n, dtype=np.int64)
            cum_shift[1:] = np.cumsum(sizes_shift)
            full_lookup_shift = np.full(length, -1, dtype=np.int64)
            full_lookup_shift[lo:hi] = np.repeat(np.arange(n - 1, dtype=np.int64), sizes_shift)
            groups.append(
                {
                    "full_lookup": torch.from_numpy(full_lookup_shift).to(bucket_dtype),
                    "cum": torch.from_numpy(cum_shift),
                    "start_idx": flat_count,
                    "n": n - 1,
                }
            )
            flat_count += n - 1

    return starts, ends, groups


def torch_features_intervals(X, starts, ends, kind, offsets, q_positions, center_mask, out=None):
    n_samples = X.shape[0]
    total = int(offsets[-1])
    if out is None:
        out = torch.empty((n_samples, total), dtype=torch.float64)
    n_intervals = len(starts)

    for i in range(n_intervals):
        s, e = int(starts[i]), int(ends[i])
        o0, o1 = int(offsets[i]), int(offsets[i + 1])

        if kind[i] == 0:
            out[:, o0] = X[:, s]
            continue

        m = e - s
        seg = X[:, s:e]
        seg_sorted, _ = torch.sort(seg, dim=-1)

        pos = q_positions[o0:o1] * (m - 1)
        lo = torch.floor(pos).long()
        hi = torch.clamp(lo + 1, max=m - 1)
        frac = pos - lo
        a = seg_sorted[:, lo]
        b = seg_sorted[:, hi]
        diff = b - a
        vals = torch.where(frac >= 0.5, b - diff * (1.0 - frac), a + diff * frac)

        cmask = center_mask[o0:o1]
        if torch.any(cmask):
            seg_mean = seg.mean(dim=-1, keepdim=True)
            vals[:, cmask] = vals[:, cmask] - seg_mean

        out[:, o0:o1] = vals

    return out


def torch_features_intervals_msort(X, starts, ends, kind, offsets, q_positions, center_mask, out=None):
    n_samples = X.shape[0]
    total = int(offsets[-1])
    if out is None:
        out = torch.empty((n_samples, total), dtype=torch.float64)
    n_intervals = len(starts)

    for i in range(n_intervals):
        s, e = int(starts[i]), int(ends[i])
        o0, o1 = int(offsets[i]), int(offsets[i + 1])

        if kind[i] == 0:
            out[:, o0] = X[:, s]
            continue

        m = e - s
        seg = X[:, s:e]
        seg_sorted = torch.msort(seg.T).T

        pos = q_positions[o0:o1] * (m - 1)
        lo = torch.floor(pos).long()
        hi = torch.clamp(lo + 1, max=m - 1)
        frac = pos - lo
        a = seg_sorted[:, lo]
        b = seg_sorted[:, hi]
        diff = b - a
        vals = torch.where(frac >= 0.5, b - diff * (1.0 - frac), a + diff * frac)

        cmask = center_mask[o0:o1]
        if torch.any(cmask):
            seg_mean = seg.mean(dim=-1, keepdim=True)
            vals[:, cmask] = vals[:, cmask] - seg_mean

        out[:, o0:o1] = vals

    return out


def torch_features_presort(X, starts, ends, kind, offsets, q_positions, center_mask, groups, out=None):
    n_samples, length = X.shape
    total = int(offsets[-1])
    if out is None:
        out = torch.empty((n_samples, total), dtype=torch.float64)

    sorted_vals_all, perm = torch.sort(X, dim=-1)
    del sorted_vals_all

    for g in groups:
        full_lookup = g["full_lookup"]
        cum = g["cum"]
        start_idx = g["start_idx"]
        n_local = g["n"]

        bucket_id_seq = full_lookup[perm]
        _, order = torch.sort(bucket_id_seq, dim=-1, stable=True)
        sorted_perm_idx = torch.gather(perm, -1, order)

        n_excluded = length - int(cum[-1])
        valid_perm_idx = sorted_perm_idx[:, n_excluded:]
        sorted_vals = torch.gather(X, -1, valid_perm_idx)

        for b in range(n_local):
            i = start_idx + b
            o0, o1 = int(offsets[i]), int(offsets[i + 1])
            seg_sorted = sorted_vals[:, int(cum[b]) : int(cum[b + 1])]
            m = seg_sorted.shape[1]

            if kind[i] == 0:
                out[:, o0] = seg_sorted[:, 0]
                continue

            pos = q_positions[o0:o1] * (m - 1)
            lo = torch.floor(pos).long()
            hi = torch.clamp(lo + 1, max=m - 1)
            frac = pos - lo
            a = seg_sorted[:, lo]
            b = seg_sorted[:, hi]
            diff = b - a
            vals = torch.where(frac >= 0.5, b - diff * (1.0 - frac), a + diff * frac)

            cmask = center_mask[o0:o1]
            if torch.any(cmask):
                seg_mean = seg_sorted.mean(dim=-1, keepdim=True)
                vals[:, cmask] = vals[:, cmask] - seg_mean

            out[:, o0:o1] = vals

    return out


def build_layouts_and_groups(length, depth, div):
    est = MomentQuantTransformer(depth=depth, div=div, mode="exact", exact_version="intervals")
    est.fit(np.zeros((1, 1, length), dtype=np.float64))

    entries = []
    for layout, repr_length in zip(est.layouts_, est._representation_lengths()):
        _, _, groups = _make_grouped_intervals(est, repr_length, depth)
        torch_layout = {
            "starts": layout["starts"],
            "ends": layout["ends"],
            "kind": layout["kind"],
            "offsets": layout["offsets"],
            "q_positions": torch.from_numpy(layout["q_positions"]),
            "center_mask": torch.from_numpy(layout["center_mask"]),
        }
        entries.append({"layout": torch_layout, "groups": groups})
    return est, entries


def intervals_transform(X, entries, n_output_features):
    X_flat = X[:, 0, :]
    n_samples = X_flat.shape[0]
    out = torch.empty((n_samples, n_output_features), dtype=torch.float64)
    offset = 0
    for repr_func, entry in zip(MomentQuantTransformer._representation_funcs(), entries):
        Z = np.ascontiguousarray(repr_func(X_flat), dtype=np.float64)
        Z_t = torch.from_numpy(Z)
        layout = entry["layout"]
        width = int(layout["offsets"][-1])
        torch_features_intervals(
            Z_t,
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


def intervals_msort_transform(X, entries, n_output_features):
    X_flat = X[:, 0, :]
    n_samples = X_flat.shape[0]
    out = torch.empty((n_samples, n_output_features), dtype=torch.float64)
    offset = 0
    for repr_func, entry in zip(MomentQuantTransformer._representation_funcs(), entries):
        Z = np.ascontiguousarray(repr_func(X_flat), dtype=np.float64)
        Z_t = torch.from_numpy(Z)
        layout = entry["layout"]
        width = int(layout["offsets"][-1])
        torch_features_intervals_msort(
            Z_t,
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
    out = torch.empty((n_samples, n_output_features), dtype=torch.float64)
    offset = 0
    for repr_func, entry in zip(MomentQuantTransformer._representation_funcs(), entries):
        Z = np.ascontiguousarray(repr_func(X_flat), dtype=np.float64)
        Z_t = torch.from_numpy(Z)
        layout = entry["layout"]
        width = int(layout["offsets"][-1])
        torch_features_presort(
            Z_t,
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
        out_intervals = intervals_transform(X, entries, est.n_output_features_).numpy()
        out_intervals_msort = intervals_msort_transform(X, entries, est.n_output_features_).numpy()
        out_presort = presort_transform(X, entries, est.n_output_features_).numpy()

        d_ri = np.max(np.abs(out_ref - out_intervals))
        d_rm = np.max(np.abs(out_ref - out_intervals_msort))
        d_rp = np.max(np.abs(out_ref - out_presort))

        assert d_ri < 1e-8, f"torch intervals disagrees with the NumPy reference at length={length}"
        assert d_rm < 1e-8, f"torch intervals_msort disagrees with the NumPy reference at length={length}"
        assert d_rp < 1e-8, f"torch presort disagrees with the NumPy reference at length={length}"


def run_benchmark(n_threads):
    torch.set_num_threads(n_threads)

    rng = np.random.default_rng(0)
    verify_correctness(rng)

    series_length_grid = LARGE_L_GRID
    n_samples = LARGE_N_SAMPLES

    X_full = rng.normal(size=(n_samples, 1, max(series_length_grid)))

    warmup_L = min(series_length_grid)
    X_warmup = np.ascontiguousarray(X_full[:10, :, :warmup_L])
    est_warmup, entries_warmup = build_layouts_and_groups(warmup_L, DEPTH, DIV)
    intervals_transform(X_warmup, entries_warmup, est_warmup.n_output_features_)
    intervals_msort_transform(X_warmup, entries_warmup, est_warmup.n_output_features_)
    presort_transform(X_warmup, entries_warmup, est_warmup.n_output_features_)

    methods = {
        "intervals": intervals_transform,
        "intervals_msort": intervals_msort_transform,
        "presort": presort_transform,
    }

    results = []
    last_cooldown = time.perf_counter()

    for series_length in series_length_grid:
        if VERBOSE:
            print(f"Series length = {series_length}  (torch.get_num_threads()={torch.get_num_threads()})")
        X_ = np.ascontiguousarray(X_full[:, :, :series_length])

        est, entries = build_layouts_and_groups(series_length, DEPTH, DIV)
        n_features = est.n_output_features_

        jobs = [(method, run) for method in methods for run in range(N_RUNS)]
        rng.shuffle(jobs)

        for method, run in jobs:
            start_time = time.perf_counter()
            methods[method](X_, entries, n_features)
            end_time = time.perf_counter()
            results.append([series_length, n_threads, method, run, end_time - start_time])
            last_cooldown = periodic_cooldown(last_cooldown)

    path = os.path.join("results", "runtimes", "presort_vs_intervals_torch")
    os.makedirs(path, exist_ok=True)
    out_path = os.path.join(path, f"runtimes_threads_{n_threads}.csv")
    pd.DataFrame(results, columns=["Series length", "Threads", "Method", "Run", "Runtime"]).to_csv(
        out_path, index=False
    )
