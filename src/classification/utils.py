import os

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, roc_auc_score

from src.utils import SMOKE_TEST

RESAMPLE_INDICES_PATH = "PythonResampleIndices"

LENGTH_BIN_LABELS = ["Short", "Medium-short", "Medium-long", "Long"]

MOMENT_QUANT_EXACT = "moment_quant_exact"
MOMENT_QUANT_APPROX = "moment_quant_approx"
QUANT = "quant"


def format_estimator_label(name):
    if name == QUANT:
        return "Quant"
    for mode in ("exact", "approx"):
        prefix = f"moment_quant_{mode}"
        if name == prefix:
            return f'MomentQuant("{mode}")'
        for version in ("samples", "intervals", "auto"):
            if name == f"{prefix}_{version}":
                return f'MomentQuant("{mode}", "{version}")'
    return name


NUM_ESTIMATORS = 200
MAX_FEATURES = 0.1
NUM_RESAMPLES = 30

if SMOKE_TEST:
    NUM_ESTIMATORS = 10
    NUM_RESAMPLES = 3

SMOKE_TEST_DATASET_NAMES = [
    "SmoothSubspace",
    "ECG200",
    "GunPoint",
    "Colposcopy",
    "Coffee",
    "BeetleFly",
    "Rock",
    "KeplerLightCurves",
]


def load_resample(
    dataset, resample, X_train_full, y_train_full, X_test_full, y_test_full, resample_indices_path=RESAMPLE_INDICES_PATH
):
    if resample == 0:
        return X_train_full, y_train_full, X_test_full, y_test_full

    X_pooled = np.concatenate([X_train_full, X_test_full])
    y_pooled = np.concatenate([y_train_full, y_test_full])
    indices_pooled = np.arange(X_pooled.shape[0])

    train_indices_path = os.path.join(resample_indices_path, dataset, f"resample{resample}Indices_TRAIN.txt")
    if not os.path.isfile(train_indices_path):
        raise FileNotFoundError(f"Could not find resample indices file {train_indices_path!r}.")
    indices_train = np.loadtxt(train_indices_path, dtype=np.int64)
    indices_test = np.setdiff1d(indices_pooled, indices_train)

    X_train, y_train = X_pooled[indices_train], y_pooled[indices_train]
    X_test, y_test = X_pooled[indices_test], y_pooled[indices_test]
    return X_train, y_train, X_test, y_test


def predict_with_proba(classifier, X, n_classes):
    if hasattr(classifier, "predict_proba"):
        y_proba = classifier.predict_proba(X)
        y_pred = np.argmax(y_proba, axis=1)
    else:
        y_pred = classifier.predict(X)
        y_proba = np.zeros((len(y_pred), n_classes))
        y_proba[np.arange(len(y_pred)), y_pred] = 1.0
    return y_pred, y_proba


def compute_classification_metrics(y_true, y_pred, y_proba, n_classes):
    labels = np.arange(n_classes)

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    logloss = log_loss(y_true, y_proba, labels=labels)

    if n_classes == 2:
        auroc = roc_auc_score(y_true, y_proba[:, 1], labels=labels, average="weighted", multi_class="ovr")
    else:
        auroc = roc_auc_score(y_true, y_proba, labels=labels, average="weighted", multi_class="ovr")

    unique, counts = np.unique(y_true, return_counts=True)
    sort_order = np.argsort(unique)
    unique, counts = unique[sort_order], counts[sort_order]
    minority_class = unique[np.flatnonzero(counts == counts.min())[0]]

    if n_classes == 2:
        f1 = f1_score(y_true, y_pred, average="binary", pos_label=minority_class, zero_division=0.0)
    else:
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0.0)

    return dict(
        Accuracy=accuracy,
        BalancedAccuracy=balanced_accuracy,
        LogLoss=logloss,
        ROCAUC=auroc,
        F1=f1,
    )


def load_series_lengths(dataset_names):
    from aeon.datasets import load_classification

    lengths = {}
    for name in dataset_names:
        X, _ = load_classification(name, split="train", load_equal_length=True, load_no_missing=True)
        lengths[name] = X.shape[2]
    return lengths


def make_length_bin_edges(lengths):
    log_edges = np.quantile(np.log(lengths), [0, 0.25, 0.5, 0.75, 1.0])
    edges = np.exp(log_edges)
    edges[0] -= 1
    edges[-1] += 1
    return edges
