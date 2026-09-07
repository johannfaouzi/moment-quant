import functools
import os
import warnings
from fractions import Fraction
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.estimators.moment_quant import MomentQuantTransformer
from src.utils import APPROX_INTERVALS, EXACT_INTERVALS


def interval_widths(l: int, d: int) -> np.ndarray:
    if not (isinstance(l, (int, np.integer)) and l >= 1):
        raise ValueError(f"l must be a positive integer, got {l!r}.")
    if not (isinstance(d, (int, np.integer)) and d >= 1):
        raise ValueError(f"d must be a positive integer, got {d!r}.")
    est = MomentQuantTransformer(depth=d)
    starts, ends, _ = est._make_intervals(l)
    return ends - starts


def n_intervals(l: int, d: int) -> int:
    k = min(d - 1, l.bit_length() - 1)
    if 2 * l >= 3 * (2**k):
        return 2 ** (k + 2) - k - 3
    return 3 * (2**k) - k - 2


def total_width(l: int, d: int) -> int:
    k = min(d - 1, l.bit_length() - 1)
    has_shifted_at_k = 2 * l >= 3 * (2**k)
    if has_shifted_at_k:
        leading = (2 * k + Fraction(2) ** (-k)) * l
        r_range = range(1, k + 1)
    else:
        leading = (2 * k - 1 + Fraction(2) ** (1 - k)) * l
        r_range = range(1, k)
    correction = sum((-(-l // (2**r))) - Fraction(l, 2**r) for r in r_range)
    total = leading - correction
    assert total.denominator == 1, f"total_width({l}, {d}) did not evaluate to an integer: {total}"
    return int(total)


def n_quantiles(l: int, d: int, nu: int) -> int:
    ni = n_intervals(l, d)
    w = total_width(l, d)
    e = residual_E(l, d, nu)
    total = ni + Fraction(w - ni, nu) - e
    assert total.denominator == 1, f"n_quantiles({l}, {d}, {nu}) did not evaluate to an integer: {total}"
    return int(total)


def n_trivial_intervals(l: int, d: int) -> int:
    if not (isinstance(l, (int, np.integer)) and l >= 1):
        raise ValueError(f"l must be a positive integer, got {l!r}.")
    if not (isinstance(d, (int, np.integer)) and d >= 1):
        raise ValueError(f"d must be a positive integer, got {d!r}.")
    floor_log2_l = l.bit_length() - 1
    k = min(d - 1, floor_log2_l)
    if k < floor_log2_l:
        return 0
    if 2 * l >= 3 * (2**k):
        return 2 * (2 ** (k + 1) - l)
    return 2 ** (k + 1) - l


def n_intervals_gt1(l: int, d: int) -> int:
    return n_intervals(l, d) - n_trivial_intervals(l, d)


def total_width_gt1(l: int, d: int) -> int:
    return total_width(l, d) - n_trivial_intervals(l, d)


def n_quantiles_gt1(l: int, d: int, nu: int) -> int:
    return n_quantiles(l, d, nu) - n_trivial_intervals(l, d)


def residual_E(l: int, d: int, nu: int) -> Fraction:
    if not (isinstance(nu, (int, np.integer)) and nu >= 1):
        raise ValueError(f"nu must be a positive integer, got {nu!r}.")
    widths = interval_widths(l, d)
    total = Fraction(0)
    for m in widths:
        m = int(m)
        total += Fraction(m - 1, nu) - ((m - 1) // nu)
    return total


def representation_length(l: int, p: int) -> int:
    if l < 3:
        raise ValueError(f"l must be at least 3 for every representation to be well-defined, got {l!r}.")
    if p == 1:
        return l
    if p == 2:
        return l - 1
    if p == 3:
        return l - 2
    if p == 4:
        return l // 2 + 1
    raise ValueError(f"p must be in {{1, 2, 3, 4}}, got {p!r}.")


def sum_kappa_S_all_representations(
    l: int, kappa_S: Callable[[int], float], direct_sum: Optional[float] = None
) -> float:
    if direct_sum is not None:
        return direct_sum
    return sum(kappa_S(representation_length(l, p)) for p in range(1, 5))


def sort_work(l: int, d: int) -> float:
    widths = interval_widths(l, d).astype(np.float64)
    widths = widths[widths > 0]
    return float(np.sum(widths * np.log2(widths)))


def sort_cost(l: int, d: int, c1: float) -> float:
    return c1 * sort_work(l, d)


def extraction_cost(l: int, d: int, nu: int, c2: float, c3: float) -> float:
    return c2 * n_quantiles_gt1(l, d, nu) + c3 * total_width_gt1(l, d)


def processing_cost(l: int, d: int, nu: int, c1: float, c2: float, c3: float) -> float:
    return sort_cost(l, d, c1) + extraction_cost(l, d, nu, c2, c3)


def total_sort_cost_all_representations(l: int, d: int, c1: float) -> float:
    return sum(sort_cost(representation_length(l, p), d, c1) for p in range(1, 5))


def total_extraction_cost_all_representations(l: int, d: int, nu: int, c2: float, c3: float) -> float:
    return sum(extraction_cost(representation_length(l, p), d, nu, c2, c3) for p in range(1, 5))


def total_processing_cost_all_representations(l: int, d: int, nu: int, c1: float, c2: float, c3: float) -> float:
    return total_sort_cost_all_representations(l, d, c1) + total_extraction_cost_all_representations(l, d, nu, c2, c3)


@functools.lru_cache(maxsize=None)
def _load_kappa_I_params(mode: str):
    kappa_I_estimator_by_mode = {
        "exact": EXACT_INTERVALS,
        "approx": APPROX_INTERVALS,
    }
    if mode not in kappa_I_estimator_by_mode:
        raise ValueError(f"mode must be one of {sorted(kappa_I_estimator_by_mode)}, got {mode!r}.")
    estimator = kappa_I_estimator_by_mode[mode]
    csv_path = os.path.join("results", "analyses", "single_thread_grid", "kappa_intervals.csv")
    df = pd.read_csv(csv_path)
    sub = df[df["Estimator"] == estimator]
    if sub.empty:
        raise ValueError(f"No rows for estimator {estimator!r} found in {csv_path!r}.")
    row = sub.iloc[0]
    return float(row["A"]), float(row["B"])


def load_kappa_I(mode: str) -> Callable[[int], float]:
    a, b = _load_kappa_I_params(mode)

    def kappa_I(l: int) -> float:
        value = a - b / l
        if value <= 0:
            raise ValueError(
                f"kappa_I(l={l}) evaluated to a non-positive value ({value!r}) under the fitted "
                f"model A - B/l with A={a!r}, B={b!r}. This model is only expected to be valid "
                "for representation lengths within (or not too far below) the grid used to fit "
                "it; double check l, or refit kappa_intervals.py with a wider/denser "
                "grid if this triggers for l values you actually need."
            )
        return value

    return kappa_I


@functools.lru_cache(maxsize=None)
def _load_kappa_S_params(mode: str):
    if mode not in ("exact", "approx"):
        raise ValueError(f"mode must be one of 'exact'/'approx', got {mode!r}.")
    csv_path = os.path.join("results", "analyses", "single_thread_grid", "kappa_series.csv")
    df = pd.read_csv(csv_path)
    sub = df[df["Mode"] == mode]
    if sub.empty:
        raise ValueError(f"No rows for Mode={mode!r} found in {csv_path!r}.")
    if len(sub) > 1:
        raise ValueError(f"Expected exactly one fitted row for Mode={mode!r} in {csv_path!r}, found {len(sub)}.")
    row = sub.iloc[0]
    if "A" not in row or "B" not in row:
        raise ValueError(
            f"{csv_path!r} does not have A/B columns -- is this an old-format kappa_series.csv "
            "(a single flat-mean kappa_S per mode, from before kappa_series.py was updated to fit "
            "an A - B/l closed-form curve like kappa_intervals.py)? Rerun kappa_series.py to "
            "regenerate it."
        )
    curve_model = row["CurveModel"] if "CurveModel" in row and pd.notna(row.get("CurveModel")) else "a_minus_b_over_l"
    if curve_model == "power_law":
        if "C" not in row or "P" not in row:
            raise ValueError(
                f"{csv_path!r}'s Mode={mode!r} row has CurveModel='power_law' but no C/P columns -- "
                "rerun kappa_series.py to regenerate it."
            )
        return "power_law", float(row["A"]), float(row["C"]), float(row["P"])
    return "a_minus_b_over_l", float(row["A"]), float(row["B"])


def load_kappa_S(mode: str) -> Callable[[int], float]:
    params = _load_kappa_S_params(mode)
    curve_model = params[0]

    if curve_model == "power_law":
        _, a, c, p = params

        def kappa_S(l: int) -> float:
            value = a + c * (l**p)
            if value < 0:
                warnings.warn(
                    f"kappa_S(l={l}) evaluated to a negative value ({value!r}) under the fitted "
                    f"power-law model A + C*L^P with A={a!r}, C={c!r}, P={p!r}; clipping to 0.0. "
                    "Unlike the A-B/l model, this model's r2_curve is genuinely high (see "
                    "kappa_series.py) -- a negative value here is more likely a real sign of l being "
                    "below the fitted grid's support than ordinary noise; double check l before "
                    "trusting downstream results silently.",
                    stacklevel=2,
                )
                return 0.0
            return value

        return kappa_S

    _, a, b = params

    def kappa_S(l: int) -> float:
        value = a - b / l
        if value < 0:
            warnings.warn(
                f"kappa_S(l={l}) evaluated to a negative value ({value!r}) under the fitted model "
                f"A - B/l with A={a!r}, B={b!r}; clipping to 0.0. This model's own r2_curve is close "
                "to 0 (see kappa_series.py), meaning kappa^(S) is not distinguishable from a small, "
                "near-zero constant for this mode -- a small negative dip near the fitted grid's "
                "boundary is expected noise, not a sign of unsafe extrapolation (contrast with "
                "load_kappa_I, which raises instead, because kappa^(I)'s own fit shows strong "
                "L-dependence).",
                stacklevel=2,
            )
            return 0.0
        return value

    return kappa_S


@functools.lru_cache(maxsize=None)
def _load_kappa_S_direct_table(mode: str):
    if mode not in ("exact", "approx"):
        raise ValueError(f"mode must be one of 'exact'/'approx', got {mode!r}.")
    csv_path = os.path.join("results", "analyses", "single_thread_grid", "kappa_series_diagnostics.csv")
    df = pd.read_csv(csv_path)
    sub = df[df["Mode"] == mode]
    return {int(row["L"]): float(row["residual"]) for _, row in sub.iterrows()}


def load_kappa_S_sum(mode: str) -> Callable[[int], Optional[float]]:
    direct_table = _load_kappa_S_direct_table(mode)

    def kappa_S_sum_direct(l: int) -> Optional[float]:
        return direct_table.get(l)

    return kappa_S_sum_direct


def intervals_outer_cost(
    n: int, l: int, d: int, nu: int, c1: float, c2: float, c3: float, kappa_I: Callable[[int], float]
) -> float:
    return n_intervals_gt1(l, d) * kappa_I(l) + n * processing_cost(l, d, nu, c1, c2, c3)


def series_outer_cost(
    n: int, l: int, d: int, nu: int, c1: float, c2: float, c3: float, kappa_S: Callable[[int], float]
) -> float:
    return n * (kappa_S(l) + processing_cost(l, d, nu, c1, c2, c3))


def crossover_sample_size(
    l: int,
    d: int,
    nu: int,
    c1_I: float,
    c2_I: float,
    c3_I: float,
    kappa_I: Callable[[int], float],
    c1_S: float,
    c2_S: float,
    c3_S: float,
    kappa_S: Callable[[int], float],
) -> Optional[float]:
    b_i = processing_cost(l, d, nu, c1_I, c2_I, c3_I)
    b_s = kappa_S(l) + processing_cost(l, d, nu, c1_S, c2_S, c3_S)
    a_i = n_intervals_gt1(l, d) * kappa_I(l)
    if b_s <= b_i:
        return None
    return a_i / (b_s - b_i)


def intervals_outer_cost_all_representations(
    n: int, l: int, d: int, nu: int, c1: float, c2: float, c3: float, kappa_I: Callable[[int], float]
) -> float:
    setup = sum(
        n_intervals_gt1(representation_length(l, p), d) * kappa_I(representation_length(l, p)) for p in range(1, 5)
    )
    marginal = n * total_processing_cost_all_representations(l, d, nu, c1, c2, c3)
    return setup + marginal


def series_outer_cost_all_representations(
    n: int,
    l: int,
    d: int,
    nu: int,
    c1: float,
    c2: float,
    c3: float,
    kappa_S: Callable[[int], float],
    kappa_S_direct: Optional[Callable[[int], Optional[float]]] = None,
) -> float:
    direct_sum = kappa_S_direct(l) if kappa_S_direct is not None else None
    return n * (
        sum_kappa_S_all_representations(l, kappa_S, direct_sum=direct_sum)
        + total_processing_cost_all_representations(l, d, nu, c1, c2, c3)
    )


def crossover_sample_size_all_representations(
    l: int,
    d: int,
    nu: int,
    c1_I: float,
    c2_I: float,
    c3_I: float,
    kappa_I: Callable[[int], float],
    c1_S: float,
    c2_S: float,
    c3_S: float,
    kappa_S: Callable[[int], float],
) -> Optional[float]:
    b_i = total_processing_cost_all_representations(l, d, nu, c1_I, c2_I, c3_I)
    b_s = sum_kappa_S_all_representations(l, kappa_S) + total_processing_cost_all_representations(
        l, d, nu, c1_S, c2_S, c3_S
    )
    a_i = sum(
        n_intervals_gt1(representation_length(l, p), d) * kappa_I(representation_length(l, p)) for p in range(1, 5)
    )
    if b_s <= b_i:
        return None
    return a_i / (b_s - b_i)


def moment_cost(l: int, d: int, c1_tilde: float) -> float:
    return c1_tilde * total_width(l, d)


def cf_extraction_cost(l: int, d: int, nu: int, c2_tilde: float, c3_tilde: float) -> float:
    return c2_tilde * n_quantiles_gt1(l, d, nu) + c3_tilde * n_intervals(l, d)


def approx_processing_cost(l: int, d: int, nu: int, c1_tilde: float, c2_tilde: float, c3_tilde: float) -> float:
    return moment_cost(l, d, c1_tilde) + cf_extraction_cost(l, d, nu, c2_tilde, c3_tilde)


def total_moment_cost_all_representations(l: int, d: int, c1_tilde: float) -> float:
    return sum(moment_cost(representation_length(l, p), d, c1_tilde) for p in range(1, 5))


def total_cf_extraction_cost_all_representations(l: int, d: int, nu: int, c2_tilde: float, c3_tilde: float) -> float:
    return sum(cf_extraction_cost(representation_length(l, p), d, nu, c2_tilde, c3_tilde) for p in range(1, 5))


def total_approx_processing_cost_all_representations(
    l: int, d: int, nu: int, c1_tilde: float, c2_tilde: float, c3_tilde: float
) -> float:
    return total_moment_cost_all_representations(l, d, c1_tilde) + total_cf_extraction_cost_all_representations(
        l, d, nu, c2_tilde, c3_tilde
    )


def approx_intervals_outer_cost(
    n: int,
    l: int,
    d: int,
    nu: int,
    c1_tilde: float,
    c2_tilde: float,
    c3_tilde: float,
    kappa_I: Callable[[int], float],
) -> float:
    setup = sum(
        n_intervals_gt1(representation_length(l, p), d) * kappa_I(representation_length(l, p)) for p in range(1, 5)
    )
    marginal = n * total_approx_processing_cost_all_representations(l, d, nu, c1_tilde, c2_tilde, c3_tilde)
    return setup + marginal


def approx_series_outer_cost(
    n: int,
    l: int,
    d: int,
    nu: int,
    c1_tilde: float,
    c2_tilde: float,
    c3_tilde: float,
    kappa_S: Callable[[int], float],
    kappa_S_direct: Optional[Callable[[int], Optional[float]]] = None,
) -> float:
    direct_sum = kappa_S_direct(l) if kappa_S_direct is not None else None
    return n * (
        sum_kappa_S_all_representations(l, kappa_S, direct_sum=direct_sum)
        + total_approx_processing_cost_all_representations(l, d, nu, c1_tilde, c2_tilde, c3_tilde)
    )


def approx_crossover_sample_size(
    l: int,
    d: int,
    nu: int,
    c1_tilde_I: float,
    c2_tilde_I: float,
    c3_tilde_I: float,
    kappa_I: Callable[[int], float],
    c1_tilde_S: float,
    c2_tilde_S: float,
    c3_tilde_S: float,
    kappa_S: Callable[[int], float],
) -> Optional[float]:
    b_i = total_approx_processing_cost_all_representations(l, d, nu, c1_tilde_I, c2_tilde_I, c3_tilde_I)
    b_s = sum_kappa_S_all_representations(l, kappa_S) + total_approx_processing_cost_all_representations(
        l, d, nu, c1_tilde_S, c2_tilde_S, c3_tilde_S
    )
    a_i = sum(
        n_intervals_gt1(representation_length(l, p), d) * kappa_I(representation_length(l, p)) for p in range(1, 5)
    )
    if b_s <= b_i:
        return None
    return a_i / (b_s - b_i)


def _reference_counts_via_real_class(l: int, d: int, nu: int):
    est = MomentQuantTransformer(depth=d, div=nu)
    starts, ends, _ = est._make_intervals(l)
    n_quantiles_per_interval, kind, _, _, _ = est._build_quantile_layout(starts, ends)
    ni = len(starts)
    w = int(np.sum(ends - starts))
    nq = int(np.sum(n_quantiles_per_interval))
    ni1 = int(np.sum(kind == 0))
    return ni, w, nq, ni1


if __name__ == "__main__":

    def _reference_counts_via_real_class(l: int, d: int, nu: int):
        est = MomentQuantTransformer(depth=d, div=nu)
        starts, ends, _ = est._make_intervals(l)
        n_quantiles_per_interval, kind, _, _, _ = est._build_quantile_layout(starts, ends)
        ni = len(starts)
        w = int(np.sum(ends - starts))
        nq = int(np.sum(n_quantiles_per_interval))
        ni1 = int(np.sum(kind == 0))
        return ni, w, nq, ni1

    DEFAULT_KAPPA_I_CSV_PATH = os.path.join("results", "analyses", "single_thread_grid", "kappa_intervals.csv")

    assert n_intervals(32, 6) == 89
    assert total_width(32, 6) == 290
    assert n_quantiles(32, 6, 4) == 112
    assert n_trivial_intervals(32, 6) == 32
    assert n_intervals_gt1(32, 6) == 57
    assert total_width_gt1(32, 6) == 258
    assert n_quantiles_gt1(32, 6, 4) == 80
    assert sort_work(32, 6) == 702.0
    assert extraction_cost(32, 6, 4, 1.0, 1.0) == 80.0 + 258.0
    assert processing_cost(32, 6, 4, 1.0, 1.0, 1.0) == 702.0 + 80.0 + 258.0

    assert n_intervals(64, 6) == 120
    assert total_width(64, 6) == 642
    assert n_quantiles(64, 6, 4) == 192
    assert n_trivial_intervals(64, 6) == 0
    assert n_intervals_gt1(64, 6) == 120
    assert total_width_gt1(64, 6) == 642
    assert n_quantiles_gt1(64, 6, 4) == 192
    assert sort_work(64, 6) == 2046.0
    assert extraction_cost(64, 6, 4, 1.0, 1.0) == 192.0 + 642.0
    assert processing_cost(64, 6, 4, 1.0, 1.0, 1.0) == 2046.0 + 192.0 + 642.0

    for l in [1, 2, 3, 5, 7, 8, 17, 31, 32, 33, 47, 48, 63, 100, 1000, 8191, 8192]:
        for d in [1, 2, 3, 6]:
            for nu in [1, 2, 3, 4]:
                ni_ref, w_ref, nq_ref, ni1_ref = _reference_counts_via_real_class(l, d, nu)
                assert n_intervals(l, d) == ni_ref, (l, d, nu, "N_i", n_intervals(l, d), ni_ref)
                assert total_width(l, d) == w_ref, (l, d, nu, "W", total_width(l, d), w_ref)
                assert n_quantiles(l, d, nu) == nq_ref, (l, d, nu, "N_q", n_quantiles(l, d, nu), nq_ref)
                assert n_trivial_intervals(l, d) == ni1_ref, (l, d, nu, "N_i^(1)", n_trivial_intervals(l, d), ni1_ref)
                assert n_intervals_gt1(l, d) == ni_ref - ni1_ref, (l, d, nu, "N_i^{>1}")
                assert total_width_gt1(l, d) == w_ref - ni1_ref, (l, d, nu, "W^{>1}")
                assert n_quantiles_gt1(l, d, nu) == nq_ref - ni1_ref, (l, d, nu, "N_q^{>1}")
                assert n_intervals_gt1(l, d) >= 0 and total_width_gt1(l, d) >= 0 and n_quantiles_gt1(l, d, nu) >= 0

    assert moment_cost(64, 6, 1.0) == total_width(64, 6)
    assert cf_extraction_cost(64, 6, 4, 1.0, 0.0) == n_quantiles_gt1(64, 6, 4) == n_quantiles(64, 6, 4)
    assert cf_extraction_cost(64, 6, 4, 0.0, 1.0) == n_intervals(64, 6)
    assert cf_extraction_cost(32, 6, 4, 1.0, 0.0) == n_quantiles_gt1(32, 6, 4) == 80
    assert cf_extraction_cost(32, 6, 4, 1.0, 0.0) != n_quantiles(32, 6, 4)
    assert cf_extraction_cost(32, 6, 4, 0.0, 1.0) == n_intervals(32, 6) == 89

    l_test = 200
    assert total_sort_cost_all_representations(l_test, 6, 1.0) == sum(
        sort_cost(representation_length(l_test, p), 6, 1.0) for p in range(1, 5)
    )
    assert total_approx_processing_cost_all_representations(l_test, 6, 4, 1.0, 1.0, 1.0) == sum(
        approx_processing_cost(representation_length(l_test, p), 6, 4, 1.0, 1.0, 1.0) for p in range(1, 5)
    )

    n_series = 500
    kappa_I_const = lambda length: 1e-6
    kappa_S_const = lambda length: 1e-6
    cost_I = approx_intervals_outer_cost(n_series, l_test, 6, 4, 1.0, 1.0, 1.0, kappa_I_const)
    cost_S = approx_series_outer_cost(n_series, l_test, 6, 4, 1.0, 1.0, 1.0, kappa_S_const)
    assert cost_I > 0 and cost_S > 0
    n_star = approx_crossover_sample_size(l_test, 6, 4, 1.0, 1.0, 1.0, kappa_I_const, 1.0, 1.0, 1.0, kappa_S_const)
    assert n_star is None or n_star > 0
    assert sum_kappa_S_all_representations(l_test, kappa_S_const) == 4 * 1e-6

    cost_I_exact = intervals_outer_cost(n_series, l_test, 6, 4, 1.0, 1.0, 1.0, kappa_I_const)
    cost_S_exact = series_outer_cost(n_series, l_test, 6, 4, 1.0, 1.0, 1.0, kappa_S_const)
    assert cost_I_exact > 0 and cost_S_exact > 0
    n_star_exact = crossover_sample_size(l_test, 6, 4, 1.0, 1.0, 1.0, kappa_I_const, 1.0, 1.0, 1.0, kappa_S_const)
    assert n_star_exact is None or n_star_exact > 0

    cost_I_exact_4rep = intervals_outer_cost_all_representations(n_series, l_test, 6, 4, 1.0, 1.0, 1.0, kappa_I_const)
    cost_S_exact_4rep = series_outer_cost_all_representations(n_series, l_test, 6, 4, 1.0, 1.0, 1.0, kappa_S_const)
    assert cost_I_exact_4rep > 0 and cost_S_exact_4rep > 0
    assert cost_I_exact_4rep > cost_I_exact
    assert cost_S_exact_4rep > cost_S_exact
    n_star_exact_4rep = crossover_sample_size_all_representations(
        l_test, 6, 4, 1.0, 1.0, 1.0, kappa_I_const, 1.0, 1.0, 1.0, kappa_S_const
    )
    assert n_star_exact_4rep is None or n_star_exact_4rep > 0

    DEFAULT_KAPPA_S_CSV_PATH = os.path.join("results", "analyses", "single_thread_grid", "kappa_series.csv")
    if os.path.exists(DEFAULT_KAPPA_I_CSV_PATH):
        for mode in ("exact", "approx"):
            kappa_I_fn = load_kappa_I(mode)
            for l_probe in (64, 100, 8192):
                value = kappa_I_fn(l_probe)
                assert np.isfinite(value), (mode, l_probe, value)
        print("load_kappa_I smoke test passed.")
    else:
        print(
            f"Skipping load_kappa_I smoke test: {DEFAULT_KAPPA_I_CSV_PATH!r} not found "
            "(expected in this sandbox; run this file from the repository root on a machine "
            "where kappa_intervals.py has already produced that CSV to exercise it)."
        )
    if os.path.exists(DEFAULT_KAPPA_S_CSV_PATH):
        for mode in ("exact", "approx"):
            kappa_S_fn = load_kappa_S(mode)
            for l_probe in (64, 100, 8192):
                value = kappa_S_fn(l_probe)
                assert np.isfinite(value), (mode, l_probe, value)
        print("load_kappa_S smoke test passed.")
    else:
        print(
            f"Skipping load_kappa_S smoke test: {DEFAULT_KAPPA_S_CSV_PATH!r} not found "
            "(expected in this sandbox; run this file from the repository root on a machine "
            "where kappa_series.py has already produced that CSV to exercise it)."
        )

    print("All self-tests passed.")
