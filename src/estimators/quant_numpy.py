"""One-to-one NumPy translation of quant_float64.py, but in NumPy instead of PyTorch."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def make_intervals(input_length, depth):
    exponent = min(depth, int(np.log2(input_length)) + 1)
    intervals = []

    for n in 2 ** np.arange(exponent):
        indices = np.linspace(0, input_length, n + 1).astype(np.int64)
        intervals_n = np.stack((indices[:-1], indices[1:]), axis=1)
        intervals.append(intervals_n)

        if n > 1 and np.median(np.diff(indices)) > 1:
            shift = int(np.ceil(input_length / n / 2))
            intervals.append(intervals_n[:-1] + shift)

    return np.concatenate(intervals, axis=0)


def f_quantile(X, div=4):
    n = X.shape[-1]

    if n == 1:
        return X.reshape(X.shape[0], 1, X.shape[1] * X.shape[2])

    num_quantiles = 1 + (n - 1) // div

    if num_quantiles == 1:
        quantiles = np.quantile(X, [0.5], axis=-1)
        quantiles = np.moveaxis(quantiles, 0, -1)
        return quantiles.reshape(quantiles.shape[0], 1, quantiles.shape[1] * quantiles.shape[2])

    q_positions = np.linspace(0.0, 1.0, num_quantiles)
    quantiles = np.quantile(X, q_positions, axis=-1)
    quantiles = np.moveaxis(quantiles, 0, -1).copy()
    quantiles[..., 1::2] -= X.mean(axis=-1, keepdims=True)
    return quantiles.reshape(quantiles.shape[0], 1, quantiles.shape[1] * quantiles.shape[2])


class IntervalModel:

    def __init__(self, input_length, depth=6, div=4):
        assert div >= 1
        assert depth >= 1
        self.div = div
        self.intervals = make_intervals(input_length=input_length, depth=depth)

    def fit(self, X, Y):
        pass

    def transform(self, X):
        features = []
        for a, b in self.intervals:
            features.append(np.squeeze(f_quantile(X[..., a:b], div=self.div), axis=1))
        return np.concatenate(features, axis=-1)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


def _smoothed_diff1(X):
    """NumPy translation of `F.avg_pool1d(F.pad(X.diff(), (2, 2), "replicate"), 5, 1)`."""
    d = np.diff(X, axis=-1)
    padded = np.pad(d, [(0, 0)] * (d.ndim - 1) + [(2, 2)], mode="edge")
    windows = sliding_window_view(padded, window_shape=5, axis=-1)
    return windows.mean(axis=-1)


class QuantNumpy:

    def __init__(self, depth=6, div=4):
        assert depth >= 1
        assert div >= 1
        self.depth = depth
        self.div = div

        self.representation_functions = (
            lambda X: X,
            lambda X: _smoothed_diff1(X),
            lambda X: np.diff(X, n=2, axis=-1),
            lambda X: np.abs(np.fft.rfft(X, axis=-1)),
        )
        self.models = {}
        self.fitted = False

    def transform(self, X):
        assert self.fitted, "not fitted"

        features = []
        for index, function in enumerate(self.representation_functions):
            Z = function(X)
            features.append(self.models[index].transform(Z))

        return np.concatenate(features, axis=-1)

    def fit_transform(self, X, y=None):
        features = []
        for index, function in enumerate(self.representation_functions):
            Z = function(X)
            self.models[index] = IntervalModel(input_length=Z.shape[-1], depth=self.depth, div=self.div)
            features.append(self.models[index].fit_transform(Z, y))

        self.fitted = True

        return np.concatenate(features, axis=-1)
