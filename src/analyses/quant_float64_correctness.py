import os

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
from src.utils import DEPTH, DIV

REPR_NAMES = ("raw", "diff1_smooth", "diff2", "fft_abs")

LENGTHS = [16, 64, 97, 128, 256, 383, 512, 2047, 2048, 4096, 8191, 8192]

OFFSETS = [0.0, 1.0, 100.0, 10_000.0, 1_000_000.0]

N_SAMPLES = 50


def summarize(length, offset, name, a, b, rows):
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.where(denom == 0, 1.0, denom)
    rel = diff / denom

    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    n_disagree_32 = int(np.sum(a32 != b32))

    print(
        f"  {name:<14} shape={a.shape!s:<16} "
        f"max_abs={diff.max():.3e}  mean_abs={diff.mean():.3e}  max_rel={rel.max():.3e}  "
        f"float32_disagreements={n_disagree_32}/{a.size}"
    )
    rows.append(
        dict(
            Length=length,
            Offset=offset,
            Representation=name,
            MaxAbsDiff=float(diff.max()),
            MeanAbsDiff=float(diff.mean()),
            MaxRelDiff=float(rel.max()),
            Float32Disagreements=n_disagree_32,
            N=int(a.size),
        )
    )
    return n_disagree_32


def compare_representations(length, offset, X_numpy, X_torch, rows):
    moment_funcs = MomentQuantTransformer._representation_funcs()
    quant = QuantFloat64(depth=DEPTH, div=DIV)

    print("-- Representation functions (before any interval/quantile processing) --")
    total_disagree = 0
    for name, m_func, q_func in zip(REPR_NAMES, moment_funcs, quant.representation_functions):
        Z_moment = m_func(X_numpy)
        with torch.no_grad():
            Z_quant = q_func(X_torch).numpy()
        assert Z_moment.shape == Z_quant.shape, (
            f"{name}: shape mismatch {Z_moment.shape} vs {Z_quant.shape} -- representations "
            "disagree structurally, not just numerically; investigate before trusting anything below."
        )
        total_disagree += summarize(length, offset, name, Z_moment.astype(np.float64), Z_quant.astype(np.float64), rows)
    return total_disagree


def compare_full_transform(length, offset, X_numpy, X_torch, rows):
    est = MomentQuantTransformer(depth=DEPTH, div=DIV, mode="exact", exact_version="intervals", parallel=False)
    Z_moment = est.fit_transform(X_numpy)

    quant = QuantFloat64(depth=DEPTH, div=DIV)
    with torch.no_grad():
        Z_quant = quant.fit_transform(X_torch).numpy()

    assert Z_moment.shape == Z_quant.shape, (
        f"Full transform: shape mismatch {Z_moment.shape} vs {Z_quant.shape} -- the two estimators "
        "disagree on the number of output features, which would itself already break the "
        "classification comparison; investigate before trusting anything below."
    )
    print("-- Full fit_transform() output (what the classifier actually sees) --")
    return summarize(length, offset, "all_features", Z_moment.astype(np.float64), Z_quant.astype(np.float64), rows)


def direct_formula_diff_selftest():
    rng = np.random.default_rng(0)
    lo = rng.normal(size=200_000)
    hi = lo + rng.normal(size=200_000) * 0.01
    frac = rng.random(200_000)
    a = lo * (1.0 - frac) + hi * frac
    b = lo + frac * (hi - lo)
    n_disagree_32 = int(np.sum(a.astype(np.float32) != b.astype(np.float32)))
    print(
        f"[self-test 1/6] interpolation formula, zero-mean data: "
        f"float64 max_abs_diff={np.abs(a - b).max():.3e}, "
        f"float32 disagreements={n_disagree_32}/{len(a)}"
    )


def centering_cancellation_selftest():
    rng = np.random.default_rng(1)
    n = 200_000
    offset = 1e5
    lo = offset + rng.normal(size=n) * 1.0
    hi = lo + rng.normal(size=n) * 0.01
    frac = rng.random(n)

    quant_a = lo * (1.0 - frac) + hi * frac
    quant_b = lo + frac * (hi - lo)

    mean_val = offset + rng.normal(size=n) * 0.5
    centered_a = quant_a - mean_val
    centered_b = quant_b - mean_val

    n_disagree_32 = int(np.sum(centered_a.astype(np.float32) != centered_b.astype(np.float32)))
    print(
        f"[self-test 2/6] interpolation formula, offset~{offset:.0e} data THEN centered: "
        f"float64 max_abs_diff={np.abs(centered_a - centered_b).max():.3e}, "
        f"float32 disagreements={n_disagree_32}/{n} ({100 * n_disagree_32 / n:.3f}%)"
    )


def torch_quantile_formula_selftest():
    rng = np.random.default_rng(2)
    n_trials = 20_000
    offset = 1e6
    m = 97
    n_disagree = 0
    max_abs_diff = 0.0
    for _ in range(n_trials):
        seg = offset + rng.normal(size=m)
        seg_sorted = np.sort(seg)
        q = float(rng.random())

        with torch.no_grad():
            ref_torch = float(torch.from_numpy(seg).quantile(torch.tensor(q, dtype=torch.float64)))

        pos = q * (m - 1)
        lo = int(np.floor(pos))
        hi = min(lo + 1, m - 1)
        frac = pos - lo
        a, b = seg_sorted[lo], seg_sorted[hi]
        diff = b - a
        branched = b - diff * (1.0 - frac) if frac >= 0.5 else a + diff * frac

        d = abs(branched - ref_torch)
        max_abs_diff = max(max_abs_diff, d)
        if branched != ref_torch:
            n_disagree += 1

    print(
        f"[self-test 3/6] MomentQuant's branched-lerp formula vs real torch.quantile output "
        f"(offset~{offset:.0e}): float64 max_abs_diff={max_abs_diff:.3e}, "
        f"disagreements={n_disagree}/{n_trials}"
    )


def mean_cancellation_selftest():
    rng = np.random.default_rng(3)
    n = 200_000
    offset = 1e6
    m = 383
    seg = offset + rng.normal(size=(n, m))
    quant_val = offset + rng.normal(size=n) * 0.5

    mean_numpy = seg.mean(axis=-1)
    with torch.no_grad():
        mean_torch = torch.from_numpy(seg).mean(dim=-1).numpy()

    centered_numpy = quant_val - mean_numpy
    centered_torch = quant_val - mean_torch

    n_disagree_32 = int(np.sum(centered_numpy.astype(np.float32) != centered_torch.astype(np.float32)))
    print(
        f"[self-test 4/6] seg.mean() (NumPy) vs X.mean() (PyTorch), offset~{offset:.0e} data THEN "
        f"centered: float64 max_abs_diff={np.abs(centered_numpy - centered_torch).max():.3e}, "
        f"float32 disagreements={n_disagree_32}/{n} ({100 * n_disagree_32 / n:.3f}%)"
    )


def linspace_selftest():
    position_disagreements = 0
    position_total = 0
    worst_num, worst_diff = None, 0.0
    for num in range(2, 4097):
        y_numpy = np.linspace(0.0, 1.0, num)
        with torch.no_grad():
            y_torch = torch.linspace(0, 1, num, dtype=torch.float64).numpy()
        d = np.abs(y_numpy - y_torch).max()
        n_diff = int(np.sum(y_numpy != y_torch))
        position_disagreements += n_diff
        position_total += num
        if d > worst_diff:
            worst_diff, worst_num = float(d), num

    print(
        f"[self-test 5/6] np.linspace(0,1,num) vs real torch.linspace(0,1,num,dtype=float64), "
        f"num=2..4096: {position_disagreements}/{position_total} individual position values differ "
        f"(worst: num={worst_num}, max_abs_diff={worst_diff:.3e})"
    )

    if position_disagreements == 0:
        print("  -> position arrays always match bit-for-bit; skipping the downstream chain.")
        return

    num = worst_num
    y_numpy_pos = np.linspace(0.0, 1.0, num)
    with torch.no_grad():
        y_torch_pos = torch.linspace(0, 1, num, dtype=torch.float64).numpy()

    rng = np.random.default_rng(4)
    n = 200_000
    offset = 1e6
    m = 383
    seg = np.sort(offset + rng.normal(size=(n, m)), axis=-1)
    q_idx = rng.integers(1, num - 1, size=n)

    def interpolate(q_positions):
        q = q_positions[q_idx]
        pos = q * (m - 1)
        lo = np.floor(pos).astype(np.int64)
        hi = np.minimum(lo + 1, m - 1)
        frac = pos - lo
        a = np.take_along_axis(seg, lo[:, None], axis=-1)[:, 0]
        b = np.take_along_axis(seg, hi[:, None], axis=-1)[:, 0]
        diff = b - a
        return np.where(frac >= 0.5, b - diff * (1.0 - frac), a + diff * frac)

    vals_numpy_pos = interpolate(y_numpy_pos)
    vals_torch_pos = interpolate(y_torch_pos)
    mean_val = seg.mean(axis=-1)

    centered_a = vals_numpy_pos - mean_val
    centered_b = vals_torch_pos - mean_val
    n_disagree_32 = int(np.sum(centered_a.astype(np.float32) != centered_b.astype(np.float32)))
    print(
        f"  chained through interpolation+centering (num={num}, offset~{offset:.0e}): "
        f"float64 max_abs_diff={np.abs(centered_a - centered_b).max():.3e}, "
        f"float32 disagreements={n_disagree_32}/{n} ({100 * n_disagree_32 / n:.3f}%)"
    )


def fft_selftest():
    n_samples = 500
    offset = 1e6
    print(f"[self-test 6/6] real np.fft.rfft vs real torch.fft.rfft, offset={offset:.0e}, n_samples={n_samples}:")
    any_disagree = False
    for L in LENGTHS:
        rng = np.random.default_rng(hash(("fft_selftest", L)) % (2**32))
        X_numpy = (offset + rng.normal(size=(n_samples, 1, L))).astype(np.float64)
        Z_numpy = np.abs(np.fft.rfft(X_numpy, axis=-1))
        with torch.no_grad():
            Z_torch = torch.fft.rfft(torch.from_numpy(X_numpy)).abs().numpy()

        diff = np.abs(Z_numpy - Z_torch)
        n_disagree_32 = int(np.sum(Z_numpy.astype(np.float32) != Z_torch.astype(np.float32)))
        if n_disagree_32 > 0:
            any_disagree = True
        print(
            f"  L={L:>5}  float64 max_abs_diff={diff.max():.3e}  "
            f"float32 disagreements={n_disagree_32}/{Z_numpy.size}"
        )
    if not any_disagree:
        print(
            "  -> no disagreements at this offset/sample size; try a larger n_samples or offset "
            "to reproduce the main sweep's L=2047 finding, which was itself a low-rate event."
        )


def main():
    print("=" * 100)
    print("QuantFloat64 vs MomentQuantTransformer(mode='exact') value-level correctness check")
    print(f"(depth={DEPTH}, div={DIV}, n_samples={N_SAMPLES})")
    print("=" * 100)

    direct_formula_diff_selftest()
    centering_cancellation_selftest()
    torch_quantile_formula_selftest()
    mean_cancellation_selftest()
    linspace_selftest()
    fft_selftest()
    print()

    rows = []
    grand_total_disagree = 0
    for offset in OFFSETS:
        for L in LENGTHS:
            print(f"\nSeries length L={L}, offset={offset:g}")
            rng = np.random.default_rng(hash((L, offset)) % (2**32))
            X_numpy = (offset + rng.normal(size=(N_SAMPLES, 1, L))).astype(np.float64)
            X_torch = torch.from_numpy(X_numpy)

            grand_total_disagree += compare_representations(L, offset, X_numpy, X_torch, rows)
            grand_total_disagree += compare_full_transform(L, offset, X_numpy, X_torch, rows)

    out_dir = os.path.join("results", "analyses", "quant_float64_correctness")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "diffs.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)

    print()
    print("=" * 100)
    if grand_total_disagree == 0:
        print(
            "No float32-level disagreements found at any (length, offset) combination tested. "
            "The centering-cancellation hypothesis (see module docstring, 'ROUND 2') would not "
            "explain the metrics.csv discrepancy after all -- look at real UCR series' own value "
            "ranges/z-normalization status directly, or reconsider the FFT-representation angle at "
            "longer/different lengths, or the prediction/metric-computation path itself."
        )
    else:
        by_offset = pd.DataFrame(rows).groupby("Offset")["Float32Disagreements"].sum()
        print("Float32 disagreements by offset (0 at Offset=0.0 would confirm round 1; growth with")
        print("Offset would confirm the centering-cancellation hypothesis):")
        print(by_offset.to_string())
        print()
        print(
            f"{grand_total_disagree} float32-level disagreement(s) found in total -- see the "
            f"per-representation breakdown above, and {out_path!r}, for exactly which "
            "(length, offset, representation) combinations are responsible."
        )
    print(f"Saved {len(rows)} rows to {out_path!r}")
    print("=" * 100)


if __name__ == "__main__":
    main()
