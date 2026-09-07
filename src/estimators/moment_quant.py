"""Moment-Quant."""

import numpy as np
from numba import njit, prange
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


@njit("UniTuple(float64[::1], 7)(float64[::1], int64[::1], int64[::1])", cache=True)
def _interval_moments_numba(x, starts, ends):
    """Computes (n, mean, M2, M3, M4, min, max) for a list of intervals [start, end) of potentially different lengths,
    in a single sequential pass per interval using the Welford/Pebay algorithm (the running min/max are tracked
    alongside the moments in the same pass, at negligible extra cost: two comparisons per element).

    Parameters
    ----------
    x : ndarray of shape (n_timepoints,), dtype float64
        Single input time series.

    starts : ndarray of shape (n_intervals,), dtype int64
        Inclusive start index of each interval.

    ends : ndarray of shape (n_intervals,), dtype int64
        Exclusive end index of each interval.

    Returns
    -------
    n_out : ndarray of shape (n_intervals,), dtype float64
        Number of points in each interval.

    mean_out : ndarray of shape (n_intervals,), dtype float64
        Mean of each interval.

    M2_out : ndarray of shape (n_intervals,), dtype float64
        Sum of squared deviations from the mean (n * variance) of each interval.

    M3_out : ndarray of shape (n_intervals,), dtype float64
        Sum of cubed deviations from the mean (n * 3rd central moment) of each interval.

    M4_out : ndarray of shape (n_intervals,), dtype float64
        Sum of 4th-power deviations from the mean (n * 4th central moment) of each interval.

    min_out : ndarray of shape (n_intervals,), dtype float64
        Exact minimum of each interval.

    max_out : ndarray of shape (n_intervals,), dtype float64
        Exact maximum of each interval.
    """
    n_intervals = starts.shape[0]
    n_out = np.zeros(n_intervals)
    mean_out = np.zeros(n_intervals)
    M2_out = np.zeros(n_intervals)
    M3_out = np.zeros(n_intervals)
    M4_out = np.zeros(n_intervals)
    min_out = np.zeros(n_intervals)
    max_out = np.zeros(n_intervals)

    for i in range(n_intervals):
        n = 0.0
        mean = 0.0
        M2 = 0.0
        M3 = 0.0
        M4 = 0.0
        s, e = starts[i], ends[i]
        mn = x[s]
        mx = x[s]
        for t in range(s, e):
            xt = x[t]
            mn = min(mn, xt)
            mx = max(mx, xt)
            n1 = n
            n += 1.0
            delta = xt - mean
            delta_n = delta / n
            delta_n2 = delta_n * delta_n
            term1 = delta * delta_n * n1
            mean += delta_n
            M4 += term1 * delta_n2 * (n * n - 3.0 * n + 3.0) + 6.0 * delta_n2 * M2 - 4.0 * delta_n * M3
            M3 += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * M2
            M2 += term1

        n_out[i] = n
        mean_out[i] = mean
        M2_out[i] = M2
        M3_out[i] = M3
        M4_out[i] = M4
        min_out[i] = mn
        max_out[i] = mx

    return n_out, mean_out, M2_out, M3_out, M4_out, min_out, max_out


@njit("UniTuple(float64[:, ::1], 7)(float64[:, ::1], int64[::1], int64[::1])", cache=True)
def _batch_interval_moments_samples(X, starts, ends):
    """Batch version (one series per row of X) of _interval_moments_numba, looping over the sample axis.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_timepoints), dtype float64
        Batch of input time series.

    starts : ndarray of shape (n_intervals,), dtype int64
        Inclusive start index of each interval.

    ends : ndarray of shape (n_intervals,), dtype int64
        Exclusive end index of each interval.

    Returns
    -------
    n_out : ndarray of shape (n_samples, n_intervals), dtype float64
        Number of points in each interval, per sample.

    mean_out : ndarray of shape (n_samples, n_intervals), dtype float64
        Mean of each interval, per sample.

    M2_out : ndarray of shape (n_samples, n_intervals), dtype float64
        Sum of squared deviations from the mean (n * variance) of each interval, per sample.

    M3_out : ndarray of shape (n_samples, n_intervals), dtype float64
        Sum of cubed deviations from the mean (n * 3rd central moment) of each interval, per sample.

    M4_out : ndarray of shape (n_samples, n_intervals), dtype float64
        Sum of 4th-power deviations from the mean (n * 4th central moment) of each interval, per sample.

    min_out : ndarray of shape (n_samples, n_intervals), dtype float64
        Exact minimum of each interval, per sample.

    max_out : ndarray of shape (n_samples, n_intervals), dtype float64
        Exact maximum of each interval, per sample.
    """
    n_samples = X.shape[0]
    n_intervals = starts.shape[0]
    n_out = np.zeros((n_samples, n_intervals))
    mean_out = np.zeros((n_samples, n_intervals))
    M2_out = np.zeros((n_samples, n_intervals))
    M3_out = np.zeros((n_samples, n_intervals))
    M4_out = np.zeros((n_samples, n_intervals))
    min_out = np.zeros((n_samples, n_intervals))
    max_out = np.zeros((n_samples, n_intervals))
    for s in range(n_samples):
        n_s, mean_s, M2_s, M3_s, M4_s, min_s, max_s = _interval_moments_numba(X[s], starts, ends)
        n_out[s] = n_s
        mean_out[s] = mean_s
        M2_out[s] = M2_s
        M3_out[s] = M3_s
        M4_out[s] = M4_s
        min_out[s] = min_s
        max_out[s] = max_s
    return n_out, mean_out, M2_out, M3_out, M4_out, min_out, max_out


@njit("UniTuple(float64[:, ::1], 7)(float64[:, ::1], int64[::1], int64[::1])", cache=True, parallel=True)
def _batch_interval_moments_samples_parallel(X, starts, ends):
    """Twin of `_batch_interval_moments_samples`, parallelized over the sample axis via `parallel=True`/`prange`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_timepoints), dtype float64
        Batch of input time series.

    starts : ndarray of shape (n_intervals,), dtype int64
        Inclusive start index of each interval.

    ends : ndarray of shape (n_intervals,), dtype int64
        Exclusive end index of each interval.

    Returns
    -------
    n_out : ndarray of shape (n_samples, n_intervals), dtype float64
        Number of points in each interval, per sample.

    mean_out : ndarray of shape (n_samples, n_intervals), dtype float64
        Mean of each interval, per sample.

    M2_out : ndarray of shape (n_samples, n_intervals), dtype float64
        Sum of squared deviations from the mean (n * variance) of each interval, per sample.

    M3_out : ndarray of shape (n_samples, n_intervals), dtype float64
        Sum of cubed deviations from the mean (n * 3rd central moment) of each interval, per sample.

    M4_out : ndarray of shape (n_samples, n_intervals), dtype float64
        Sum of 4th-power deviations from the mean (n * 4th central moment) of each interval, per sample.

    min_out : ndarray of shape (n_samples, n_intervals), dtype float64
        Exact minimum of each interval, per sample.

    max_out : ndarray of shape (n_samples, n_intervals), dtype float64
        Exact maximum of each interval, per sample.
    """
    n_samples = X.shape[0]
    n_intervals = starts.shape[0]
    n_out = np.zeros((n_samples, n_intervals))
    mean_out = np.zeros((n_samples, n_intervals))
    M2_out = np.zeros((n_samples, n_intervals))
    M3_out = np.zeros((n_samples, n_intervals))
    M4_out = np.zeros((n_samples, n_intervals))
    min_out = np.zeros((n_samples, n_intervals))
    max_out = np.zeros((n_samples, n_intervals))
    for s in prange(n_samples):
        n_s, mean_s, M2_s, M3_s, M4_s, min_s, max_s = _interval_moments_numba(X[s], starts, ends)
        n_out[s] = n_s
        mean_out[s] = mean_s
        M2_out[s] = M2_s
        M3_out[s] = M3_s
        M4_out[s] = M4_s
        min_out[s] = min_s
        max_out[s] = max_s
    return n_out, mean_out, M2_out, M3_out, M4_out, min_out, max_out


def _batch_interval_moments_intervals(X, starts, ends):
    """Alternative approximate feature computation for long series, looping over the intervals.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_timepoints), dtype float64
        Batch of input time series.

    starts : ndarray of shape (n_intervals,), dtype int64
        Inclusive start index of each interval.

    ends : ndarray of shape (n_intervals,), dtype int64
        Exclusive end index of each interval.

    Returns
    -------
    n_out : ndarray of shape (n_samples, n_intervals), dtype float64
        Number of points in each interval, per sample.

    mean_out : ndarray of shape (n_samples, n_intervals), dtype float64
        Mean of each interval, per sample.

    M2_out : ndarray of shape (n_samples, n_intervals), dtype float64
        Sum of squared deviations from the mean (n * variance) of each interval, per sample.

    M3_out : ndarray of shape (n_samples, n_intervals), dtype float64
        Sum of cubed deviations from the mean (n * 3rd central moment) of each interval, per sample.

    M4_out : ndarray of shape (n_samples, n_intervals), dtype float64
        Sum of 4th-power deviations from the mean (n * 4th central moment) of each interval, per sample.

    min_out : ndarray of shape (n_samples, n_intervals), dtype float64
        Exact minimum of each interval, per sample.

    max_out : ndarray of shape (n_samples, n_intervals), dtype float64
        Exact maximum of each interval, per sample.
    """
    n_samples = X.shape[0]
    n_intervals = len(starts)
    n_out = np.empty((n_samples, n_intervals), dtype=np.float64)
    mean_out = np.empty((n_samples, n_intervals), dtype=np.float64)
    M2_out = np.empty((n_samples, n_intervals), dtype=np.float64)
    M3_out = np.empty((n_samples, n_intervals), dtype=np.float64)
    M4_out = np.empty((n_samples, n_intervals), dtype=np.float64)
    min_out = np.empty((n_samples, n_intervals), dtype=np.float64)
    max_out = np.empty((n_samples, n_intervals), dtype=np.float64)

    for i in range(n_intervals):
        s, e = int(starts[i]), int(ends[i])
        seg = X[:, s:e]
        m = e - s

        mean = seg.mean(axis=-1)
        centered = seg - mean[:, None]
        c2 = centered * centered

        n_out[:, i] = m
        mean_out[:, i] = mean
        M2_out[:, i] = c2.sum(axis=-1)
        M3_out[:, i] = (c2 * centered).sum(axis=-1)
        M4_out[:, i] = (c2 * c2).sum(axis=-1)
        min_out[:, i] = seg.min(axis=-1)
        max_out[:, i] = seg.max(axis=-1)

    return n_out, mean_out, M2_out, M3_out, M4_out, min_out, max_out


@njit(
    "float64[::1](float64[::1], int64[::1], int64[::1], int64[::1], int64[::1], float64[::1], boolean[::1])", cache=True
)
def _exact_features_core(x, starts, ends, kind, offsets, q_positions, center_mask):
    """Exact feature extraction (local sort + centering) for a single series.

    Parameters
    ----------
    x : ndarray of shape (n_timepoints,), dtype float64
        Single input time series.

    starts : ndarray of shape (n_intervals,), dtype int64
        Inclusive start index of each interval.

    ends : ndarray of shape (n_intervals,), dtype int64
        Exclusive end index of each interval.

    kind : ndarray of shape (n_intervals,), dtype int64
        Per-interval code:
            - 0 = raw passthrough (length-1 interval),
            - 1 = median only (no centering),
            - 2 = full quantile set (with centering).

    offsets : ndarray of shape (n_intervals + 1,), dtype int64
        CSR-style cumulative feature counts: interval `i` contributes `offsets[i+1] - offsets[i]` features, stored at
        `out[offsets[i]:offsets[i+1]]`.

    q_positions : ndarray of shape (n_features,), dtype float64
        Quantile position (in [0, 1]) requested for each output feature.

    center_mask : ndarray of shape (n_features,), dtype bool
        Whether each output feature must be centered (local interval mean subtracted) after being computed.

    Returns
    -------
    out : ndarray of shape (n_features,), dtype float64
        Flat array of extracted features, one entry per (interval, quantile position) pair, in the order defined by
        `offsets`.
    """
    total = offsets[-1]
    out = np.zeros(total)
    n_intervals = starts.shape[0]
    for i in range(n_intervals):
        s, e = starts[i], ends[i]
        o0, o1 = offsets[i], offsets[i + 1]
        if kind[i] == 0:
            out[o0] = x[s]
            continue
        seg = np.sort(x[s:e])
        m = seg.shape[0]
        seg_mean = 0.0
        for t in range(m):
            seg_mean += seg[t]
        seg_mean /= m
        for j in range(o1 - o0):
            pos = q_positions[o0 + j] * (m - 1)
            lo = int(np.floor(pos))
            hi = min(lo + 1, m - 1)
            frac = pos - lo
            a = seg[lo]
            b = seg[hi]
            diff = b - a

            if frac >= 0.5:
                val = b - diff * (1.0 - frac)
            else:
                val = a + diff * frac

            if center_mask[o0 + j]:
                val -= seg_mean

            out[o0 + j] = val
    return out


@njit(
    "float64[:, ::1](float64[:, ::1], int64[::1], int64[::1], int64[::1], int64[::1], float64[::1], boolean[::1])",
    cache=True,
)
def _batch_exact_features_samples(X, starts, ends, kind, offsets, q_positions, center_mask):
    """Batch version of _exact_features_core, looping over the samples.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_timepoints), dtype float64
        Batch of input time series.

    starts : ndarray of shape (n_intervals,), dtype int64
            Inclusive start index of each interval.

    ends : ndarray of shape (n_intervals,), dtype int64
        Exclusive end index of each interval.

    kind : ndarray of shape (n_intervals,), dtype int64
        Per-interval code:
            - 0 = raw passthrough (length-1 interval),
            - 1 = median only (no centering),
            - 2 = full quantile set (with centering).

    offsets : ndarray of shape (n_intervals + 1,), dtype int64
        CSR-style cumulative feature counts: interval `i` contributes `offsets[i+1] - offsets[i]` features, stored at
        `out[offsets[i]:offsets[i+1]]`.

    q_positions : ndarray of shape (n_features,), dtype float64
        Quantile position (in [0, 1]) requested for each output feature.

    center_mask : ndarray of shape (n_features,), dtype bool
        Whether each output feature must be centered (local interval mean subtracted) after being computed.

    Returns
    -------
    out : ndarray of shape (n_samples, n_features), dtype float64
        Extracted features, one row per sample.
    """
    n_samples = X.shape[0]
    total = offsets[-1]
    out = np.zeros((n_samples, total))
    for i in range(n_samples):
        out[i] = _exact_features_core(X[i], starts, ends, kind, offsets, q_positions, center_mask)
    return out


@njit(
    "float64[:, ::1](float64[:, ::1], int64[::1], int64[::1], int64[::1], int64[::1], float64[::1], boolean[::1])",
    cache=True,
    parallel=True,
)
def _batch_exact_features_samples_parallel(X, starts, ends, kind, offsets, q_positions, center_mask):
    """Twin of `_batch_exact_features_samples`, parallelized over the sample axis via `parallel=True`/`prange`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_timepoints), dtype float64
        Batch of input time series.

    starts : ndarray of shape (n_intervals,), dtype int64
        Inclusive start index of each interval.

    ends : ndarray of shape (n_intervals,), dtype int64
        Exclusive end index of each interval.

    kind : ndarray of shape (n_intervals,), dtype int64
        Per-interval code:
            - 0 = raw passthrough (length-1 interval),
            - 1 = median only (no centering),
            - 2 = full quantile set (with centering).

    offsets : ndarray of shape (n_intervals + 1,), dtype int64
        CSR-style cumulative feature counts: interval `i` contributes `offsets[i+1] - offsets[i]` features, stored at
        `out[offsets[i]:offsets[i+1]]`.

    q_positions : ndarray of shape (n_features,), dtype float64
        Quantile position (in [0, 1]) requested for each output feature.

    center_mask : ndarray of shape (n_features,), dtype bool
        Whether each output feature must be centered (local interval mean subtracted) after being computed.

    Returns
    -------
    out : ndarray of shape (n_samples, n_features), dtype float64
        Extracted features, one row per sample.
    """
    n_samples = X.shape[0]
    total = offsets[-1]
    out = np.zeros((n_samples, total))
    for i in prange(n_samples):
        out[i] = _exact_features_core(X[i], starts, ends, kind, offsets, q_positions, center_mask)
    return out


def _batch_exact_features_intervals(X, starts, ends, kind, offsets, q_positions, center_mask, out=None):
    """Alternative exact feature computation for long series, looping over the intervals.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_timepoints), dtype float64
            Batch of input time series.

    starts : ndarray of shape (n_intervals,), dtype int64
            Inclusive start index of each interval.

    ends : ndarray of shape (n_intervals,), dtype int64
        Exclusive end index of each interval.

    kind : ndarray of shape (n_intervals,), dtype int64
        Per-interval code:
            - 0 = raw passthrough (length-1 interval),
            - 1 = median only (no centering),
            - 2 = full quantile set (with centering).

    offsets : ndarray of shape (n_intervals + 1,), dtype int64
        CSR-style cumulative feature counts: interval `i` contributes `offsets[i+1] - offsets[i]` features, stored at
        `out[offsets[i]:offsets[i+1]]`.

    q_positions : ndarray of shape (n_features,), dtype float64
        Quantile position (in [0, 1]) requested for each output feature.

    center_mask : ndarray of shape (n_features,), dtype bool
        Whether each output feature must be centered (local interval mean subtracted) after being computed.

    out : ndarray of shape (n_samples, n_features), dtype float64, optional
        If provided, results are written directly into this array (which must already have shape
        `(n_samples, offsets[-1])`) instead of allocating a new one, and no separate "block" ever
        exists in memory beyond it -- see `MomentQuantTransformer._compute_exact_features_one_representation`.
        If None (default), a new array is allocated and returned, matching the previous behavior.

    Returns
    -------
    out : ndarray of shape (n_samples, n_features), dtype float64
        Extracted features, one row per sample. The same object as the `out` argument if one was provided.
    """
    n_samples = X.shape[0]
    total = int(offsets[-1])
    if out is None:
        out = np.empty((n_samples, total), dtype=np.float64)
    n_intervals = len(starts)

    for i in range(n_intervals):
        s, e = int(starts[i]), int(ends[i])
        o0, o1 = int(offsets[i]), int(offsets[i + 1])

        if kind[i] == 0:
            # length-1 interval: raw value for every sample at once.
            out[:, o0] = X[:, s]
            continue

        m = e - s
        seg = X[:, s:e]
        seg_sorted = np.sort(seg, axis=-1)  # one vectorized sort for all samples

        # Every requested quantile position of this interval
        pos = q_positions[o0:o1] * (m - 1)  # shape (n_q,)
        lo = np.floor(pos).astype(np.int64)
        hi = np.minimum(lo + 1, m - 1)
        frac = pos - lo
        a = seg_sorted[:, lo]
        b = seg_sorted[:, hi]
        diff = b - a

        vals = np.where(frac >= 0.5, b - diff * (1.0 - frac), a + diff * frac)

        cmask = center_mask[o0:o1]
        if np.any(cmask):
            seg_mean = seg.mean(axis=-1, keepdims=True)
            vals[:, cmask] -= seg_mean

        out[:, o0:o1] = vals

    return out


class MomentQuantTransformer(BaseEstimator, TransformerMixin):
    """Moment-Quant feature extraction algorithm.

    This algorithm is very similar to the Quant feature extraction algorithm, but with several improvements.
    The transform still involves computing quantiles over a fixed set of dyadic intervals of the input series and three
    transformations of the input time series. For each set of intervals extracted, the window is shifted by half the
    interval length to extract more intervals.

    However, it is possible to approximate the quantiles instead of using the true quantiles, resulting in a faster
    algorithm. The approximate quantiles are computed using the Cornish-Fisher expansion.

    Parameters
    ----------
    depth : int, default=6
        The depth to stop extracting intervals at. Starting with the full series, the number of intervals extracted is
        ``2 ** depth`` (starting at 0) for each level.

    div : int, default=4
        The divisor to find the number of quantiles to extract from intervals.

    mode : {"approx", "exact"}, default="approx"
        Mode used to compute the quantiles:

        - If "approx", most quantiles are approximated in O(1) per interval via a moment tree (Chan/Pebay) +
          Cornish-Fisher expansion. The two boundary positions (0 and 1, i.e. the local min/max of the interval) are
          not approximated as the Cornish-Fisher expansion cannot evaluate them anyway and  and the exact min/max only
          cost an extra O(1) reduction per interval on top of the moments already being computed (no sorting needed).
        - If "exact", every quantile (including min/max) is computed exactly via local sorting of each interval.

    approx_version : {"auto", "intervals", "samples"} or int, default="auto"
        Only used if `mode="approx"`. Which approximate version is used:

        - If "intervals", the function loops over the intervals instead of the samples.
        - If "samples", the function loops over the samples instead of the intervals.
        - If "auto", the function chooses the mode ("intervals" or "samples") that should be the faster based on the
          available information. The threshold behind this choice is calibrated for a specific (depth, div); using
          "auto" with a depth/div other than what it was calibrated at warns (see `_nl_threshold`).
        - If int, the function loops over the intervals if the length of the input series is equal to or larger than
          the provided positive integer, otherwise the function loops over the samples.

    exact_version : {"auto", "intervals", "samples"} or int, default="auto"
        Only used if `mode="exact"`. Which exact version is used:

        - If "intervals", the function loops over the intervals instead of the samples.
        - If "samples", the function loops over the samples instead of the intervals.
        - If "auto", the function chooses the mode ("intervals" or "samples") that should be the faster based on the
          available information. The threshold behind this choice is calibrated for a specific (depth, div); using
          "auto" with a depth/div other than what it was calibrated at warns (see `_nl_threshold`).
        - If int, the function loops over the intervals if the length of the input series is equal to or larger than
          the provided positive integer, otherwise the function loops over the samples.

    parallel : bool, default=False
        Enables multithreading. Only affects the "samples" versions of the "approx" and "exact" modes.
    """

    # Generic built-in defaults, fitted on a depth=6, div=4, parallel=False benchmark grid.
    _APPROX_NL_THRESHOLD_DEFAULT = 142_167
    _EXACT_NL_THRESHOLD_DEFAULT = 95_924

    # No built-in generic default for parallel=True.
    _APPROX_NL_THRESHOLD_DEFAULT_PARALLEL = None
    _EXACT_NL_THRESHOLD_DEFAULT_PARALLEL = None

    # Machine-specific overrides.
    _APPROX_NL_THRESHOLD_CALIBRATED = None
    _EXACT_NL_THRESHOLD_CALIBRATED = None
    _APPROX_NL_THRESHOLD_CALIBRATED_PARALLEL = None
    _EXACT_NL_THRESHOLD_CALIBRATED_PARALLEL = None

    _APPROX_NL_THRESHOLD_CALIBRATED_DEPTH_DIV = None
    _EXACT_NL_THRESHOLD_CALIBRATED_DEPTH_DIV = None
    _APPROX_NL_THRESHOLD_CALIBRATED_PARALLEL_DEPTH_DIV = None
    _EXACT_NL_THRESHOLD_CALIBRATED_PARALLEL_DEPTH_DIV = None

    # Default values for depth and div when fitting the dispatch thresholds
    _DISPATCH_THRESHOLD_DEFAULT_FIT_DEPTH = 6
    _DISPATCH_THRESHOLD_DEFAULT_FIT_DIV = 4

    # Whether this process has already settled its dispatch-threshold state
    _DISPATCH_THRESHOLDS_AUTOLOADED = False

    def __init__(self, depth=6, div=4, mode="approx", approx_version="auto", exact_version="auto", parallel=False):
        if not (isinstance(mode, str) and (mode in ("approx", "exact"))):
            raise ValueError(f"mode must be 'approx' or 'exact', got {mode!r}.")
        if not (isinstance(depth, (int, np.int_)) and (depth >= 1)):
            raise ValueError(f"depth must be a positive integer, got {depth!r}.")
        if not (isinstance(div, (int, np.int_)) and (div >= 1)):
            raise ValueError(f"div must be a positive integer, got {div!r}.")
        if not (
            (isinstance(approx_version, str) and approx_version in ("intervals", "samples", "auto"))
            or (isinstance(approx_version, (int, np.int_)) and approx_version > 0)
        ):
            raise ValueError(
                f"approx_version must be either a positive integer or one of the following strings: "
                f"'intervals', 'samples' or 'auto'. Got {approx_version!r}."
            )
        if not (
            (isinstance(exact_version, str) and exact_version in ("intervals", "samples", "auto"))
            or (isinstance(exact_version, (int, np.int_)) and exact_version > 0)
        ):
            raise ValueError(
                f"exact_version must be either a positive integer or one of the following strings: "
                f"'intervals', 'samples' or 'auto'. Got {exact_version!r}."
            )
        if not isinstance(parallel, (bool, np.bool_)):
            raise TypeError(f"parallel must be a boolean, got {parallel!r}.")

        self.depth = depth
        self.div = div
        self.mode = mode
        self.approx_version = approx_version
        self.exact_version = exact_version
        self.parallel = parallel

    @classmethod
    def _warn_depth_div_mismatch(cls, mode, parallel, fit_depth_div, depth, div):
        """Warns if the calling instance's own (depth, div) doesn't match the (depth, div) the dispatch threshold
        `_nl_threshold` is about to return was actually fitted at."""
        if fit_depth_div is None or (depth is None and div is None):
            return
        fit_depth, fit_div = fit_depth_div
        depth_mismatched = depth is not None and fit_depth is not None and depth != fit_depth
        div_mismatched = div is not None and fit_div is not None and div != fit_div
        if not (depth_mismatched or div_mismatched):
            return
        import warnings

        warnings.warn(
            f"'auto' dispatch for mode={mode!r}, parallel={parallel!r} is using a threshold fitted "
            f"at depth={fit_depth!r}, div={fit_div!r}, but this MomentQuantTransformer instance was "
            f"constructed with depth={depth!r}, div={div!r}. The fitted cost basis is depth-dependent "
            f"(and, for approx mode, div-dependent too -- see dispatch_thresholds.py's module "
            f"docstring), so this threshold's samples-vs-intervals crossover is not guaranteed "
            f"correct here. Re-run the matching grid script and dispatch_thresholds.py at this "
            f"instance's own depth/div to get a matching threshold, or pass an explicit "
            f"approx_version='samples'/'intervals' (or exact_version=...) instead of 'auto' to "
            f"sidestep dispatch entirely.",
            stacklevel=3,
        )

    @classmethod
    def _nl_threshold(cls, mode, parallel, depth=None, div=None):
        """Resolve the class's currently-effective dispatch threshold for one (mode, parallel) combination.

        Parameters
        ----------
        mode : {"approx", "exact"}

        parallel : bool
            Which threshold family to resolve -- should be the calling instance's own `parallel`
            attribute (see `_decide_version`).

        depth, div : int, optional
            The calling instance's own `depth`/`div` attributes, for the mismatch check described
            above. Independently optional -- passing only one still checks that one.

        Returns
        -------
        threshold : float
        """
        if not cls._DISPATCH_THRESHOLDS_AUTOLOADED:
            cls.load_cached_dispatch_thresholds()

        if mode == "approx":
            calibrated, default = cls._APPROX_NL_THRESHOLD_CALIBRATED, cls._APPROX_NL_THRESHOLD_DEFAULT
            calibrated_p, default_p = (
                cls._APPROX_NL_THRESHOLD_CALIBRATED_PARALLEL,
                cls._APPROX_NL_THRESHOLD_DEFAULT_PARALLEL,
            )
            calibrated_dd, calibrated_p_dd = (
                cls._APPROX_NL_THRESHOLD_CALIBRATED_DEPTH_DIV,
                cls._APPROX_NL_THRESHOLD_CALIBRATED_PARALLEL_DEPTH_DIV,
            )
        else:
            calibrated, default = cls._EXACT_NL_THRESHOLD_CALIBRATED, cls._EXACT_NL_THRESHOLD_DEFAULT
            calibrated_p, default_p = (
                cls._EXACT_NL_THRESHOLD_CALIBRATED_PARALLEL,
                cls._EXACT_NL_THRESHOLD_DEFAULT_PARALLEL,
            )
            calibrated_dd, calibrated_p_dd = (
                cls._EXACT_NL_THRESHOLD_CALIBRATED_DEPTH_DIV,
                cls._EXACT_NL_THRESHOLD_CALIBRATED_PARALLEL_DEPTH_DIV,
            )
        default_dd = (cls._DISPATCH_THRESHOLD_DEFAULT_FIT_DEPTH, cls._DISPATCH_THRESHOLD_DEFAULT_FIT_DIV)

        serial_threshold = calibrated if calibrated is not None else default
        serial_dd = calibrated_dd if calibrated is not None else default_dd
        if not parallel:
            cls._warn_depth_div_mismatch(mode, parallel, serial_dd, depth, div)
            return serial_threshold
        if calibrated_p is not None:
            cls._warn_depth_div_mismatch(mode, parallel, calibrated_p_dd, depth, div)
            return calibrated_p
        if default_p is not None:
            return default_p
        cls._warn_depth_div_mismatch(mode, parallel, serial_dd, depth, div)
        return serial_threshold

    def _decide_version(self, mode, n_samples):
        """Decide between "samples" and "intervals" for the "auto" setting.

        Parameters
        ----------
        mode : {"approx", "exact"}
            Which cost model (and threshold) to use.

        n_samples : int
            Number of samples in the batch about to be processed (the batch passed to whichever
            of `fit()`/`transform()` is calling this, not necessarily the batch used to fit).

        Returns
        -------
        version : {"samples", "intervals"}
        """
        L = self.n_timepoints_
        threshold = self._nl_threshold(mode, self.parallel, self.depth, self.div)
        if mode == "approx":
            return "intervals" if n_samples * L > threshold else "samples"
        else:
            basis = L * (np.log2(L) - 2.5)
            return "intervals" if n_samples * basis > threshold else "samples"

    @staticmethod
    def _dispatch_thresholds_cache_path():
        """Path to the small JSON cache file `calibrate_dispatch_thresholds()` (when `save=True`)
        and `load_cached_dispatch_thresholds()` use to persist machine-specific thresholds across
        processes, so a user does not need to recalibrate every session on the same hardware.

        Returns
        -------
        cache_path : pathlib.Path
        """
        import os
        from pathlib import Path

        if os.name == "nt":
            base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        else:
            base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
        return Path(base) / "moment_quant" / "dispatch_thresholds.json"

    @classmethod
    def calibrate_dispatch_thresholds(
        cls, csv_path=None, modes=("approx", "exact"), parallel=False, strict=True, save=True, cache_path=None
    ):
        """Fit the `_CALIBRATED` dispatch thresholds at the class level with machine-specific
        values fitted on this hardware, from `src/analyses/dispatch_thresholds.py`'s output
        CSV. The built-in `_DEFAULT` constants are never modified by this method.

        Parameters
        ----------
        csv_path : str, optional
            Path to `dispatch_thresholds.csv`. Defaults to
            `os.path.join("results", "analyses", "multiple_threads_grid" if parallel else
            "single_thread_grid", "dispatch_thresholds.csv")`, i.e.
            `dispatch_thresholds.py`'s own default output location for the requested
            `parallel` value, assuming the process is run from the repository root (same assumption
            every script in this project makes about relative paths).

        modes : iterable of {"approx", "exact"}, default=("approx", "exact")
            Which threshold(s) to calibrate. Defaults to both; pass e.g. `("exact",)` to leave the
            approx threshold unset (i.e. `_nl_threshold("approx", parallel)` keeps resolving to its
            fallback) while only calibrating exact mode.

        parallel : bool, default=False
            Which threshold family to calibrate: the `parallel=False` (serial) `_CALIBRATED`
            attributes, fit from `single_thread_grid.py`'s output, or the `parallel=True`
            `_CALIBRATED_PARALLEL` attributes, fit from `multiple_threads_grid.py`'s output. These
            are fit and stored completely independently -- calibrating one never touches the other
            (see `_nl_threshold`'s serial-fallback behavior for `parallel=True` when only the
            serial one has been calibrated).

        strict : bool, default=True
            If True (default) and `csv_path` does not exist, raises `FileNotFoundError` naming the
            two scripts to run first (the grid script matching `parallel`, then
            `dispatch_thresholds.py`) rather than failing later with a confusing error. If
            False, silently leaves the existing threshold in place when the CSV is missing --
            useful for scripts that want to opportunistically calibrate when the file happens to
            exist, without hard-failing otherwise.

        save : bool, default=True
            If True (default), also write the fitted value(s) to the user-cache JSON file (see
            `_dispatch_thresholds_cache_path()`), merging with -- rather than overwriting -- any
            other mode/parallel combination already present in that file. If False, only the
            in-process class attributes are set; nothing is persisted to disk.

        cache_path : str or pathlib.Path, optional
            Where to write the cache file if `save=True`. Defaults to
            `_dispatch_thresholds_cache_path()`.

        Returns
        -------
        calibrated : bool
            True if the class attribute(s) were set, False if `strict=False` and the CSV was
            missing (existing thresholds left untouched).
        """
        import json
        import os
        from pathlib import Path

        import pandas as pd

        cls._DISPATCH_THRESHOLDS_AUTOLOADED = True

        if csv_path is None:
            csv_path = os.path.join(
                "results",
                "analyses",
                "multiple_threads_grid" if parallel else "single_thread_grid",
                "dispatch_thresholds.csv",
            )

        if not os.path.isfile(csv_path):
            if strict:
                parallel_diff = (
                    'the serial threshold as a stopgap (see _APPROX/_EXACT_NL_THRESHOLD_DEFAULT_PARALLEL)'
                    if parallel else 'the built-in defaults (_APPROX_NL_THRESHOLD_DEFAULT/_EXACT_NL_THRESHOLD_DEFAULT)'
                )
                grid_script = "multiple_threads_grid" if parallel else "single_thread_grid"
                raise FileNotFoundError(
                    f"Could not find {csv_path!r}. Dispatch thresholds are only known once you have "
                    f"run, from the repository root: `python -m src.runtimes.{grid_script}` "
                    f"then `python -m src.analyses.dispatch_thresholds`. Until then, "
                    f"'auto' with parallel={parallel!r} uses {parallel_diff}, "
                    f"which {'were' if not parallel else 'was'} fitted on a different machine (or a "
                    f"different execution mode) and may not be optimal for yours. Pass strict=False "
                    f"to silently keep the existing threshold instead of raising."
                )
            return False

        df = pd.read_csv(csv_path)
        attr_by_mode = (
            {"approx": "_APPROX_NL_THRESHOLD_CALIBRATED_PARALLEL", "exact": "_EXACT_NL_THRESHOLD_CALIBRATED_PARALLEL"}
            if parallel
            else {"approx": "_APPROX_NL_THRESHOLD_CALIBRATED", "exact": "_EXACT_NL_THRESHOLD_CALIBRATED"}
        )
        attr_by_mode_dd = (
            {
                "approx": "_APPROX_NL_THRESHOLD_CALIBRATED_PARALLEL_DEPTH_DIV",
                "exact": "_EXACT_NL_THRESHOLD_CALIBRATED_PARALLEL_DEPTH_DIV",
            }
            if parallel
            else {
                "approx": "_APPROX_NL_THRESHOLD_CALIBRATED_DEPTH_DIV",
                "exact": "_EXACT_NL_THRESHOLD_CALIBRATED_DEPTH_DIV",
            }
        )
        fitted = {}
        fitted_depth_div = {}
        for mode in modes:
            if mode not in attr_by_mode:
                raise ValueError(f"modes must only contain 'approx'/'exact', got {mode!r}.")
            sub = df[df["Mode"] == mode]
            if len(sub) != 1:
                raise ValueError(f"Expected exactly one row for Mode={mode!r} in {csv_path!r}, found {len(sub)}.")
            row = sub.iloc[0]
            threshold = float(row["NL_Threshold"])

            if "delta" in row and row["delta"] <= 0 and not np.isinf(threshold):
                threshold = float("inf")

            fitted[mode] = threshold
            setattr(cls, attr_by_mode[mode], fitted[mode])

            if "Depth" in row.index and "Div" in row.index and pd.notna(row["Depth"]) and pd.notna(row["Div"]):
                depth_div = (int(row["Depth"]), int(row["Div"]))
            else:
                depth_div = None
            fitted_depth_div[mode] = depth_div
            setattr(cls, attr_by_mode_dd[mode], depth_div)

        if save:
            path = Path(cache_path) if cache_path is not None else cls._dispatch_thresholds_cache_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            cached = {}
            if path.is_file():
                cached = json.loads(path.read_text())
            cache_keys = {mode: (f"{mode}_parallel" if parallel else mode) for mode in fitted}
            cached.update(
                {
                    cache_keys[mode]: {
                        "value": fitted[mode],
                        "depth": fitted_depth_div[mode][0] if fitted_depth_div[mode] is not None else None,
                        "div": fitted_depth_div[mode][1] if fitted_depth_div[mode] is not None else None,
                    }
                    for mode in fitted
                }
            )
            path.write_text(json.dumps(cached))

        return True

    @classmethod
    def load_cached_dispatch_thresholds(cls, cache_path=None, modes=("approx", "exact")):
        """Load previously-calibrated thresholds from the user-cache JSON file written by
        `calibrate_dispatch_thresholds(save=True)`, without re-reading or re-fitting from a
        benchmark CSV.

        Parameters
        ----------
        cache_path : str or pathlib.Path, optional
            Defaults to `_dispatch_thresholds_cache_path()`.

        modes : iterable of {"approx", "exact"}, default=("approx", "exact")
            Which threshold(s) to load, among those present in the cache file. Defaults to both.
            Applies to both the serial and parallel families -- there is no separate `parallel`
            argument here since both are always loaded together from the single cache file.

        Returns
        -------
        loaded : bool
            True if at least one class attribute (serial or parallel) was set from the cache, False
            if the cache file does not exist or contains none of the requested `modes`.
        """
        import json
        import warnings
        from pathlib import Path

        cls._DISPATCH_THRESHOLDS_AUTOLOADED = True

        path = Path(cache_path) if cache_path is not None else cls._dispatch_thresholds_cache_path()
        if not path.is_file():
            return False

        cached = json.loads(path.read_text())
        attr_by_mode = {"approx": "_APPROX_NL_THRESHOLD_CALIBRATED", "exact": "_EXACT_NL_THRESHOLD_CALIBRATED"}
        attr_by_mode_parallel = {
            "approx": "_APPROX_NL_THRESHOLD_CALIBRATED_PARALLEL",
            "exact": "_EXACT_NL_THRESHOLD_CALIBRATED_PARALLEL",
        }
        attr_by_mode_dd = {
            "approx": "_APPROX_NL_THRESHOLD_CALIBRATED_DEPTH_DIV",
            "exact": "_EXACT_NL_THRESHOLD_CALIBRATED_DEPTH_DIV",
        }
        attr_by_mode_parallel_dd = {
            "approx": "_APPROX_NL_THRESHOLD_CALIBRATED_PARALLEL_DEPTH_DIV",
            "exact": "_EXACT_NL_THRESHOLD_CALIBRATED_PARALLEL_DEPTH_DIV",
        }

        def _unpack(entry):
            if isinstance(entry, dict):
                depth, div = entry.get("depth"), entry.get("div")
                depth_div = (depth, div) if depth is not None and div is not None else None
                return entry["value"], depth_div
            return entry, None

        def _sanitize(mode, key, value):
            value = float(value)
            if value < 0:
                warnings.warn(
                    f"Cached dispatch threshold for mode={mode!r} ({key!r}) in {path} is negative "
                    f"({value!r}), which is almost always a sign that it was calibrated before a "
                    f"sign-guard fix for delta <= 0 (see compute_thresholds() in "
                    f'src/analyses/dispatch_thresholds.py) -- treating it as +inf ("always '
                    f'samples") instead of trusting it as-is. Re-run dispatch_thresholds.py to '
                    f"replace this cache entry with a freshly re-fit value.",
                    stacklevel=2,
                )
                return float("inf")
            return value

        loaded = False
        for mode in modes:
            if mode not in attr_by_mode:
                raise ValueError(f"modes must only contain 'approx'/'exact', got {mode!r}.")
            if mode in cached:
                value, depth_div = _unpack(cached[mode])
                setattr(cls, attr_by_mode[mode], _sanitize(mode, mode, value))
                setattr(cls, attr_by_mode_dd[mode], depth_div)
                loaded = True
            parallel_key = f"{mode}_parallel"
            if parallel_key in cached:
                value, depth_div = _unpack(cached[parallel_key])
                setattr(cls, attr_by_mode_parallel[mode], _sanitize(mode, parallel_key, value))
                setattr(cls, attr_by_mode_parallel_dd[mode], depth_div)
                loaded = True
        return loaded

    @classmethod
    def reset_dispatch_thresholds(cls, modes=("approx", "exact"), parallel=None, delete_cache=False, cache_path=None):
        """Clear calibrated threshold(s) back to unset.

        Parameters
        ----------
        modes : iterable of {"approx", "exact"}, default=("approx", "exact")
            Which threshold(s) to reset. Defaults to both.

        parallel : {None, True, False}, default=None
            Which threshold family to reset. None (default) resets BOTH the serial and parallel
            calibrated values, matching this method's pre-existing "reset everything" behavior
            before parallel-awareness was added. Pass True/False to reset only that one family and
            leave the other's calibration untouched.

        delete_cache : bool, default=False
            If True, also delete the user-cache JSON file (or the requested `modes`'/`parallel`
            entries from it, leaving the file in place if any other entry remains), so a subsequent
            `load_cached_dispatch_thresholds()` in a new process does not restore the calibration.

        cache_path : str or pathlib.Path, optional
            Only used if `delete_cache=True`. Defaults to `_dispatch_thresholds_cache_path()`.
        """
        import json
        from pathlib import Path

        cls._DISPATCH_THRESHOLDS_AUTOLOADED = True

        attr_by_mode = {"approx": "_APPROX_NL_THRESHOLD_CALIBRATED", "exact": "_EXACT_NL_THRESHOLD_CALIBRATED"}
        attr_by_mode_parallel = {
            "approx": "_APPROX_NL_THRESHOLD_CALIBRATED_PARALLEL",
            "exact": "_EXACT_NL_THRESHOLD_CALIBRATED_PARALLEL",
        }
        attr_by_mode_dd = {
            "approx": "_APPROX_NL_THRESHOLD_CALIBRATED_DEPTH_DIV",
            "exact": "_EXACT_NL_THRESHOLD_CALIBRATED_DEPTH_DIV",
        }
        attr_by_mode_parallel_dd = {
            "approx": "_APPROX_NL_THRESHOLD_CALIBRATED_PARALLEL_DEPTH_DIV",
            "exact": "_EXACT_NL_THRESHOLD_CALIBRATED_PARALLEL_DEPTH_DIV",
        }
        reset_serial = parallel is None or parallel is False
        reset_parallel = parallel is None or parallel is True
        for mode in modes:
            if mode not in attr_by_mode:
                raise ValueError(f"modes must only contain 'approx'/'exact', got {mode!r}.")
            if reset_serial:
                setattr(cls, attr_by_mode[mode], None)
                setattr(cls, attr_by_mode_dd[mode], None)
            if reset_parallel:
                setattr(cls, attr_by_mode_parallel[mode], None)
                setattr(cls, attr_by_mode_parallel_dd[mode], None)

        if delete_cache:
            path = Path(cache_path) if cache_path is not None else cls._dispatch_thresholds_cache_path()
            if path.is_file():
                cached = json.loads(path.read_text())
                for mode in modes:
                    if reset_serial:
                        cached.pop(mode, None)
                    if reset_parallel:
                        cached.pop(f"{mode}_parallel", None)
                if cached:
                    path.write_text(json.dumps(cached))
                else:
                    path.unlink()

    def _determine_approx_version(self, n_samples):
        """Determines which approximate version is used.

        Parameters
        ----------
        n_samples : int
            Number of samples in the batch about to be processed by `fit()` or `transform()`.
            Only used by the "auto" setting (see `_decide_version`); ignored otherwise.
        """
        samples_func = _batch_interval_moments_samples_parallel if self.parallel else _batch_interval_moments_samples
        if isinstance(self.approx_version, str):  # case where approx_version is a string
            if self.approx_version == "samples":
                self._approx_func = samples_func
            elif self.approx_version == "intervals":
                self._approx_func = _batch_interval_moments_intervals
            else:  # 'auto' case
                version = self._decide_version("approx", n_samples)
                self._approx_func = samples_func if version == "samples" else _batch_interval_moments_intervals
        else:  # case where approx_version is a positive integer
            if self.n_timepoints_ < self.approx_version:
                self._approx_func = samples_func
            else:
                self._approx_func = _batch_interval_moments_intervals

    def _determine_exact_version(self, n_samples):
        """Determines which exact version is used.

        Parameters
        ----------
        n_samples : int
            Number of samples in the batch about to be processed by `fit()` or `transform()`.
            Only used by the "auto" setting (see `_decide_version`); ignored otherwise.
        """
        samples_func = _batch_exact_features_samples_parallel if self.parallel else _batch_exact_features_samples
        if isinstance(self.exact_version, str):  # case where exact_version is a string
            if self.exact_version == "samples":
                self._exact_func = samples_func
            elif self.exact_version == "intervals":
                self._exact_func = _batch_exact_features_intervals
            else:  # 'auto' case
                version = self._decide_version("exact", n_samples)
                self._exact_func = samples_func if version == "samples" else _batch_exact_features_intervals
        else:  # case where exact_version is a positive integer
            if self.n_timepoints_ < self.exact_version:
                self._exact_func = samples_func
            else:
                self._exact_func = _batch_exact_features_intervals

    @staticmethod
    def _norm_ppf(p):
        """Percent point function of the standard normal distribution with Peter Acklam's rational approximation.

        Parameters
        ----------
        p : array_like of shape (...,)
            Probabilities in (0, 1) at which to evaluate the standard normal quantile function. Any shape is accepted.

        Returns
        -------
        out : ndarray of shape (...,), dtype float64
            Approximated standard normal quantiles, same shape as `p`.
        """
        p = np.asarray(p, dtype=np.float64)

        a = [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
        b = [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
        c = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]

        p_low = 0.02425
        out = np.empty_like(p)

        low = p < p_low
        high = p > 1 - p_low
        mid = ~(low | high)

        # Lower tail
        q = np.sqrt(-2 * np.log(np.clip(p[low], 1e-300, None)))
        out[low] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )

        # Central region
        q = p[mid] - 0.5
        r = q * q
        out[mid] = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )

        # Upper tail
        q = np.sqrt(-2 * np.log(np.clip(1 - p[high], 1e-300, None)))
        out[high] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )

        return out

    @staticmethod
    def _moments_to_skew_kurt(n, M2, M3, M4, eps=1e-12):
        """Converts (n, mean, M2, M3, M4) into (mean, variance, skewness, excess kurtosis), vectorized.

        Parameters
        ----------
        n : ndarray
            Number of points used to compute each set of moments.

        M2 : ndarray of shape (...,)
            Sum of squared deviations from the mean (n * variance), same shape as `n`.

        M3 : ndarray of shape (...,)
            Sum of cubed deviations from the mean (n * 3rd central moment), same shape as `n`.

        M4 : ndarray of shape (...,)
            Sum of 4th-power deviations from the mean (n * 4th central moment), same shape as `n`.

        eps : float, default=1e-12
            Numerical floor added to variance-like quantities to avoid division by zero.

        Returns
        -------
        var : ndarray of shape (...,)
            Variance (M2 / n).

        skew : ndarray of shape (...,)
            Skewness.

        exkurt : ndarray of shape (...,)
            Excess kurtosis (kurtosis minus 3).
        """
        var = M2 / n
        std = np.sqrt(np.maximum(var, eps))
        skew = (M3 / n) / (std**3 + eps)
        exkurt = (M4 / n) / (var**2 + eps) - 3.0
        return var, skew, exkurt

    def _cornish_fisher_quantiles(self, mean, var, skew, exkurt, q):
        """Approximates quantiles q (scalar or array) from the first 4 moments.

        Parameters
        ----------
        mean : ndarray of shape (...,)
            Mean of each distribution to approximate.

        var : ndarray of shape (...,)
            Variance of each distribution, same shape as `mean`.

        skew : ndarray of shape (...,)
            Skewness of each distribution, same shape as `mean`.

        exkurt : ndarray of shape (...,)
            Excess kurtosis of each distribution, same shape as `mean`.

        q : array_like of shape (Q,)
            Quantile positions (in [0, 1]) at which to evaluate the Cornish-Fisher expansion.

        Returns
        -------
        out : ndarray of shape (..., Q)
            Approximated quantiles, for every combination of the leading shape of `mean`/`var`/`skew`/`exkurt` and the
            Q requested quantile positions.
        """
        z = self._norm_ppf(np.asarray(q, dtype=np.float64))  # (Q,)
        std = np.sqrt(np.maximum(var, 0.0))[..., None]  # (..., 1)
        skew_ = skew[..., None]
        exkurt_ = exkurt[..., None]
        mean_ = mean[..., None]

        w = (
            z
            + (z**2 - 1.0) * skew_ / 6.0
            + (z**3 - 3.0 * z) * exkurt_ / 24.0
            - (2.0 * z**3 - 5.0 * z) * (skew_**2) / 36.0
        )
        return mean_ + std * w

    @staticmethod
    def _repr_raw(X):
        """Raw representation."""
        return X

    @staticmethod
    def _repr_diff1_smooth(X):
        """Smoothed first-order difference representation."""
        d1 = np.diff(X, axis=-1)
        pad_width = [(0, 0)] * (d1.ndim - 1) + [(2, 2)]
        padded = np.pad(d1, pad_width, mode="edge")
        window = 5
        csum = np.cumsum(padded, axis=-1)
        csum = np.concatenate([np.zeros(csum.shape[:-1] + (1,)), csum], axis=-1)
        smoothed = (csum[..., window:] - csum[..., :-window]) / window
        return np.ascontiguousarray(smoothed, dtype=np.float64)

    @staticmethod
    def _repr_diff2(X):
        """Second-order difference representation."""
        return np.ascontiguousarray(np.diff(X, n=2, axis=-1), dtype=np.float64)

    @staticmethod
    def _repr_fft_abs(X):
        """Magnitude of the real Fourier transform representation."""
        return np.ascontiguousarray(np.abs(np.fft.rfft(X, axis=-1)), dtype=np.float64)

    @classmethod
    def _representation_funcs(cls):
        """The 4 representation transform functions."""
        return (cls._repr_raw, cls._repr_diff1_smooth, cls._repr_diff2, cls._repr_fft_abs)

    def _representation_lengths(self):
        """Closed-form lengths for the 4 representations.

        Returns
        -------
        lengths : tuple of 4 int
            Lengths of the raw, smoothed first-difference, second-difference, and FFT-magnitude representations,
            respectively: (L, L - 1, L - 2, L // 2 + 1), where L is the length of each raw input time series.
        """
        return (self.n_timepoints_, self.n_timepoints_ - 1, self.n_timepoints_ - 2, self.n_timepoints_ // 2 + 1)

    def _make_intervals(self, length):
        """Method to generate the intervals.

        Returns
        -------
        starts, ends : ndarray of shape (n_intervals,), dtype int64

        depths : ndarray of shape (n_intervals,), dtype int64
            Depth level d (0 .. exponent - 1) each interval belongs to. The base set of 2**d
            intervals and, where present, its shifted counterpart (same loop iteration) share the
            same depth label. Kept as a genuine return value (not reconstructed elsewhere) so that
            `fit()` can store it in `layouts_` as the single source of truth -- see
            `get_feature_names_out` and approx_quantile_correlation.py for two consumers.
        """
        exponent = min(self.depth, int(np.log2(length)) + 1)

        starts_list, ends_list, depth_list = [], [], []
        for d in range(exponent):
            n = 2**d
            indices = np.linspace(0, length, n + 1).astype(np.int64)
            s, e = indices[:-1], indices[1:]
            starts_list.append(s)
            ends_list.append(e)
            depth_list.append(np.full(len(s), d, dtype=np.int64))

            if n > 1 and np.median(e - s) > 1:
                shift = int(np.ceil(length / n / 2))
                starts_list.append(s[:-1] + shift)
                ends_list.append(e[:-1] + shift)
                depth_list.append(np.full(len(s) - 1, d, dtype=np.int64))

        starts = np.concatenate(starts_list).astype(np.int64)
        ends = np.concatenate(ends_list).astype(np.int64)
        depths = np.concatenate(depth_list).astype(np.int64)
        return starts, ends, depths

    def _build_quantile_layout(self, starts, ends):
        """Determines the quantiles to compute in all the (sub-)intervals."""
        lengths = (ends - starts).astype(np.int64)
        n_intervals = len(starts)

        n_quantiles = np.zeros(n_intervals, dtype=np.int64)
        kind = np.zeros(n_intervals, dtype=np.int64)

        for i, n in enumerate(lengths):
            if n == 1:
                n_quantiles[i] = 1
                kind[i] = 0
            else:
                nq = 1 + (n - 1) // self.div
                n_quantiles[i] = nq
                kind[i] = 1 if nq == 1 else 2

        offsets = np.zeros(n_intervals + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(n_quantiles)
        total = int(offsets[-1])

        q_positions = np.full(total, 0.5, dtype=np.float64)
        center_mask = np.zeros(total, dtype=np.bool_)

        for i in range(n_intervals):
            o0, o1 = offsets[i], offsets[i + 1]
            if kind[i] == 2:
                q_positions[o0:o1] = np.linspace(0.0, 1.0, n_quantiles[i])
                center_mask[o0 + 1 : o1 : 2] = True
            elif kind[i] == 1:
                q_positions[o0:o1] = 0.5

        return n_quantiles, kind, offsets, center_mask, q_positions

    @staticmethod
    def _build_channel_permutation(offsets, n_channels):
        """Builds the index permutation that reorders a per-channel-then-per-feature layout into QUANT's actual
        multivariate concatenation order.

        Parameters
        ----------
        offsets : ndarray of shape (n_intervals + 1,), dtype int64
            CSR-style cumulative single-channel feature counts (see `_build_quantile_layout`).

        n_channels : int
            Number of channels.

        Returns
        -------
        perm : ndarray of shape (n_channels * total,), dtype int64
            Such that, for `flat` of shape (n_samples, n_channels * total) laid out as
            `flat[:, c * total + f]` = channel `c`, single-channel feature `f`, `flat[:, perm]` is laid out in
            QUANT's channel-major-per-interval order (`total = offsets[-1]`, the single-channel feature count).
        """
        total = int(offsets[-1])
        n_intervals = len(offsets) - 1
        perm = np.empty(n_channels * total, dtype=np.int64)
        for i in range(n_intervals):
            o0, o1 = int(offsets[i]), int(offsets[i + 1])
            width = o1 - o0
            for c in range(n_channels):
                dest = o0 * n_channels + c * width
                source = c * total + o0
                perm[dest : dest + width] = np.arange(source, source + width)
        return perm

    def _compute_approx_features_one_representation(self, Z, layout, out):
        """Compute approximate features for a single representation, writing them into `out`.

        Parameters
        ----------
        Z : ndarray of shape (n_samples, n_channels, n_timepoints), dtype float64
            Batch of input time series (`n_channels == 1` for a univariate input, normalized to 3D by `fit`/
            `transform`).

        layout : dict
            One entry of `self.layouts_` (keys "starts", "ends", "kind", "offsets", "center_mask", "q_positions",
            "channel_perm"), matching `Z`'s length.

        out : ndarray of shape (n_samples, n_channels * n_features_this_repr), dtype float64
            Destination slice (a view into the shared, pre-allocated output array built by
            `_compute_approx_features`) that this representation's features are written into. Channel-major
            then quantile-minor within each interval's block (matching QUANT's own multivariate concatenation
            order -- see `_build_channel_permutation`).
        """
        starts, ends = layout["starts"], layout["ends"]
        kind, offsets = layout["kind"], layout["offsets"]
        q_positions, center_mask = layout["q_positions"], layout["center_mask"]

        n_samples, n_channels, length = Z.shape
        Z_flat = Z.reshape(n_samples * n_channels, length)

        n, mean, M2, M3, M4, mn, mx = self._approx_func(Z_flat, starts, ends)
        n_intervals = n.shape[1]
        n, mean, M2, M3, M4, mn, mx = (
            a.reshape(n_samples, n_channels, n_intervals) for a in (n, mean, M2, M3, M4, mn, mx)
        )
        var, skew, exkurt = self._moments_to_skew_kurt(n, M2, M3, M4)

        total = int(offsets[-1])
        dest = out[:, None, :] if n_channels == 1 else np.zeros((n_samples, n_channels, total))

        for i in range(n_intervals):
            o0, o1 = int(offsets[i]), int(offsets[i + 1])
            if kind[i] == 0:
                dest[:, :, o0] = mean[:, :, i]
                continue

            if kind[i] == 2:
                # First position (o0) is exactly q=0.0 (the min), never centered.
                dest[:, :, o0] = mn[:, :, i]

                # Last position (o1 - 1) is exactly q=1.0 (the max), possibly centered
                # depending on parity (matches QUANT's `quantiles[..., 1::2] -= mean`).
                last_val = mx[:, :, i]
                if center_mask[o1 - 1]:
                    last_val = last_val - mean[:, :, i]
                dest[:, :, o1 - 1] = last_val

                # Strictly interior positions, if any (none when num_quantiles == 2,
                # i.e. the only positions are 0.0 and 1.0).
                if o1 - o0 > 2:
                    q_interior = q_positions[o0 + 1 : o1 - 1]
                    feats = self._cornish_fisher_quantiles(
                        mean[:, :, i], var[:, :, i], skew[:, :, i], exkurt[:, :, i], q_interior
                    )
                    cmask_interior = center_mask[o0 + 1 : o1 - 1]
                    if cmask_interior.any():
                        feats = feats.copy()
                        feats[..., cmask_interior] -= mean[:, :, i : i + 1]
                    dest[:, :, o0 + 1 : o1 - 1] = feats
            else:
                # kind == 1 (median only, q=0.5): a single strictly interior position handled directly by Cornish-Fisher
                q = q_positions[o0:o1]
                feats = self._cornish_fisher_quantiles(mean[:, :, i], var[:, :, i], skew[:, :, i], exkurt[:, :, i], q)
                cmask = center_mask[o0:o1]
                if cmask.any():
                    feats = feats.copy()
                    feats[..., cmask] -= mean[:, :, i : i + 1]
                dest[:, :, o0:o1] = feats

        if n_channels > 1:
            out[:, :] = dest.reshape(n_samples, n_channels * total)[:, layout["channel_perm"]]

    def _compute_approx_features(self, X):
        """Compute approximate features for the 4 representations.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_channels, n_timepoints), dtype float64, C-contiguous
            Batch of raw input time series.

        Returns
        -------
        out : ndarray of shape (n_samples, n_output_features_), dtype float64
            Approximate QUANT-style features, concatenated over the 4 representations.
        """
        n_samples = X.shape[0]
        out = np.empty((n_samples, self.n_output_features_), dtype=np.float64)
        offset = 0
        for repr_func, layout in zip(self._representation_funcs(), self.layouts_):
            width = int(layout["offsets"][-1]) * self.n_channels_
            self._compute_approx_features_one_representation(repr_func(X), layout, out[:, offset : offset + width])
            offset += width
        return out

    def _compute_exact_features_one_representation(self, Z, layout, out):
        """Compute exact features for a single representation, writing them into `out`.

        Parameters
        ----------
        Z : ndarray of shape (n_samples, n_channels, n_timepoints), dtype float64
            Batch of input time series (`n_channels == 1` for a univariate input, normalized to 3D by `fit`/
            `transform`).

        layout : dict
            One entry of `self.layouts_` (keys "starts", "ends", "kind", "offsets", "center_mask", "q_positions",
            "channel_perm"), matching `Z`'s length.

        out : ndarray of shape (n_samples, n_channels * n_features_this_repr), dtype float64
            Destination slice (a view into the shared, pre-allocated output array built by
            `_compute_exact_features`) that this representation's features are written into. Channel-major
            then quantile-minor within each interval's block (see `_build_channel_permutation`).
        """
        n_samples, n_channels, length = Z.shape
        Z_flat = Z.reshape(n_samples * n_channels, length)

        if n_channels == 1 and self._exact_func is _batch_exact_features_intervals:
            self._exact_func(
                Z_flat,
                layout["starts"],
                layout["ends"],
                layout["kind"],
                layout["offsets"],
                layout["q_positions"],
                layout["center_mask"],
                out,
            )
            return

        block = self._exact_func(
            Z_flat,
            layout["starts"],
            layout["ends"],
            layout["kind"],
            layout["offsets"],
            layout["q_positions"],
            layout["center_mask"],
        )  # (n_samples * n_channels, total)

        if n_channels == 1:
            out[:, :] = block
        else:
            total = block.shape[1]
            out[:, :] = block.reshape(n_samples, n_channels * total)[:, layout["channel_perm"]]
        del block

    def _compute_exact_features(self, X):
        """Compute exact quantiles.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_channels, n_timepoints), dtype float64, C-contiguous
            Batch of input time series.

        Returns
        -------
        out : ndarray of shape (n_samples, n_output_features_), dtype float64
            Exact quantiles.
        """
        n_samples = X.shape[0]
        out = np.empty((n_samples, self.n_output_features_), dtype=np.float64)
        offset = 0
        for repr_func, layout in zip(self._representation_funcs(), self.layouts_):
            width = int(layout["offsets"][-1]) * self.n_channels_
            self._compute_exact_features_one_representation(repr_func(X), layout, out[:, offset : offset + width])
            offset += width
        return out

    def _validate_input_shape(self, X, transform):
        """Validates the dimensions of X."""
        if X.ndim != 3:
            raise ValueError(
                f"X must be 3D (n_samples, n_channels, n_timepoints), got an array with {X.ndim} dimensions."
            )
        if transform:
            if X.shape[1] != self.n_channels_:
                raise ValueError(
                    f"Number of channels inconsistent with fit(): expected {self.n_channels_}, got {X.shape[1]}."
                )
            if X.shape[2] != self.n_timepoints_:
                raise ValueError(
                    f"Series length inconsistent with fit(): expected {self.n_timepoints_}, got {X.shape[2]}."
                )

    def fit(self, X, y=None):
        """Fit the estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_channels, n_timepoints)
            Batch of input time series, all of the same length.

        y : array-like of shape (n_samples,) or None, default=None
            Ignored. Present for scikit-learn API compatibility.

        Returns
        -------
        self : MomentQuantTransformer
            The fitted transformer.
        """
        X = check_array(X, dtype=np.float64, order="C", ensure_2d=False, allow_nd=True)
        self._validate_input_shape(X, transform=False)
        self.n_channels_ = X.shape[1]
        self.n_timepoints_ = X.shape[2]

        if self.mode == "approx":
            self._determine_approx_version(X.shape[0])
        else:
            self._determine_exact_version(X.shape[0])

        self.layouts_ = []
        for length in self._representation_lengths():
            starts, ends, depths = self._make_intervals(length)
            _, kind, offsets, center_mask, q_positions = self._build_quantile_layout(starts, ends)
            channel_perm = self._build_channel_permutation(offsets, self.n_channels_) if self.n_channels_ > 1 else None
            self.layouts_.append(
                {
                    "starts": starts,
                    "ends": ends,
                    "depth": depths,
                    "kind": kind,
                    "offsets": offsets,
                    "center_mask": center_mask,
                    "q_positions": q_positions,
                    "channel_perm": channel_perm,
                }
            )

        self.n_output_features_ = sum(int(layout["offsets"][-1]) for layout in self.layouts_) * self.n_channels_

        return self

    def transform(self, X):
        """Applies feature extraction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_channels, n_timepoints)
            Batch of input time series, of the same length and the same number of channels as seen in fit().

        Returns
        -------
        out : ndarray of shape (n_samples, n_output_features_), dtype float64
            Extracted QUANT-style features (approximate or exact, depending on `self.mode`).
        """
        check_is_fitted(self, "layouts_")
        X = check_array(X, dtype=np.float64, order="C", ensure_2d=False, allow_nd=True)
        self._validate_input_shape(X, transform=True)

        if self.mode == "approx":
            self._determine_approx_version(X.shape[0])
            return self._compute_approx_features(X)
        else:
            self._determine_exact_version(X.shape[0])
            return self._compute_exact_features(X)

    def get_feature_names_out(self):
        """Explicit feature names (representation, channel, interval, depth, quantile position, optional centering).

        Returns
        -------
        names : ndarray of shape (n_output_features_,), dtype object
            One descriptive string per output feature, e.g. "raw_iv[0:100]_d0_q0.50_centered_approx" for a
            univariate input, or "raw_ch0_iv[0:100]_d0_q0.50_centered_approx" for a multivariate one (the channel
            tag is only included when `n_channels_ > 1`, to keep univariate names unchanged). "d0" is the depth
            level the interval belongs to (see `_make_intervals`'s own docstring: the base 2**d intervals and,
            where present, their shifted counterpart share the same depth label). Names are generated in the
            same channel-major, per-interval order as the actual output columns.
        """
        check_is_fitted(self, "layouts_")

        names = []
        for repr_func, layout in zip(self._representation_funcs(), self.layouts_):
            # Strip the "_repr_" prefix off the static method's own name, e.g.
            repr_name = repr_func.__name__[len("_repr_") :]

            starts, ends, depths = layout["starts"], layout["ends"], layout["depth"]
            kind, offsets = layout["kind"], layout["offsets"]
            q_positions, center_mask = layout["q_positions"], layout["center_mask"]

            for i in range(len(starts)):
                s, e, d = int(starts[i]), int(ends[i]), int(depths[i])
                o0, o1 = int(offsets[i]), int(offsets[i + 1])
                for c in range(self.n_channels_):
                    ch_tag = f"_ch{c}" if self.n_channels_ > 1 else ""
                    for j in range(o0, o1):
                        if kind[i] == 0:
                            tag = "raw"
                        else:
                            tag = f"q{q_positions[j]:.2f}"
                            if center_mask[j]:
                                tag += "_centered"
                        names.append(f"{repr_name}{ch_tag}_iv[{s}:{e}]_d{d}_{tag}_{self.mode}")

        return np.asarray(names, dtype=object)
