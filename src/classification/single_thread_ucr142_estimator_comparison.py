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
import torch
from aeon.datasets import load_classification
from aeon.datasets.tsc_datasets import redux, univariate_equal_length
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from src.classification.utils import (
    MAX_FEATURES,
    MOMENT_QUANT_APPROX,
    MOMENT_QUANT_EXACT,
    NUM_ESTIMATORS,
    NUM_RESAMPLES,
    QUANT,
    SMOKE_TEST_DATASET_NAMES,
    compute_classification_metrics,
    load_resample,
    predict_with_proba,
)
from src.estimators.moment_quant import MomentQuantTransformer
from src.estimators.quant_float64 import QuantFloat64
from src.utils import DEPTH, DIV, SMOKE_TEST, VERBOSE, periodic_cooldown

n_jobs = 1

estimators = {
    MOMENT_QUANT_EXACT: MomentQuantTransformer(
        depth=DEPTH, div=DIV, mode="exact", exact_version="auto", parallel=False
    ),
    MOMENT_QUANT_APPROX: MomentQuantTransformer(
        depth=DEPTH, div=DIV, mode="approx", approx_version="auto", parallel=False
    ),
    QUANT: QuantFloat64(depth=DEPTH, div=DIV),
}

dataset_names = SMOKE_TEST_DATASET_NAMES if SMOKE_TEST else list(univariate_equal_length) + list(redux)

results = []
last_cooldown = time.perf_counter()
for i, name in enumerate(dataset_names):
    if VERBOSE:
        print(f"[{i + 1:>3}/{len(dataset_names)}] {name}")

    X_train_full, y_train_full = load_classification(name, split="train", load_equal_length=True, load_no_missing=True)
    X_test_full, y_test_full = load_classification(name, split="test", load_equal_length=True, load_no_missing=True)

    for resample in range(NUM_RESAMPLES):
        X_train, y_train, X_test, y_test = load_resample(
            name, resample, X_train_full, y_train_full, X_test_full, y_test_full
        )

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)
        n_classes = len(le.classes_)

        X_train_numpy = np.ascontiguousarray(X_train, dtype=np.float64)
        X_test_numpy = np.ascontiguousarray(X_test, dtype=np.float64)
        X_train_torch = torch.from_numpy(X_train_numpy)
        X_test_torch = torch.from_numpy(X_test_numpy)

        for estimator_name, estimator in estimators.items():
            X_train_ = X_train_numpy if estimator_name != QUANT else X_train_torch
            X_test_ = X_test_numpy if estimator_name != QUANT else X_test_torch

            classifier = ExtraTreesClassifier(
                n_estimators=NUM_ESTIMATORS,
                max_features=MAX_FEATURES,
                criterion="entropy",
                n_jobs=n_jobs,
                random_state=resample,
            )

            t0 = time.perf_counter()
            Z_train = estimator.fit_transform(X_train_)
            t1 = time.perf_counter()
            classifier.fit(Z_train, y_train_enc)
            t2 = time.perf_counter()
            Z_test = estimator.transform(X_test_)
            t3 = time.perf_counter()
            y_pred, y_proba = predict_with_proba(classifier, Z_test, n_classes)
            t4 = time.perf_counter()

            runtime_feature_extraction_train = t1 - t0
            runtime_classifier_fit = t2 - t1
            runtime_feature_extraction_test = t3 - t2
            runtime_prediction = t4 - t3

            metrics = compute_classification_metrics(y_test_enc, y_pred, y_proba, n_classes)

            results.append(
                [
                    name,
                    resample,
                    estimator_name,
                    metrics["Accuracy"],
                    metrics["BalancedAccuracy"],
                    metrics["LogLoss"],
                    metrics["ROCAUC"],
                    metrics["F1"],
                    runtime_feature_extraction_train
                    + runtime_classifier_fit
                    + runtime_feature_extraction_test
                    + runtime_prediction,
                    runtime_feature_extraction_train,
                    runtime_classifier_fit,
                    runtime_feature_extraction_test,
                    runtime_prediction,
                ]
            )
            last_cooldown = periodic_cooldown(last_cooldown)

path = os.path.join("results", "classification", "single_thread_ucr142_estimator_comparison")
if not os.path.isdir(path):
    os.makedirs(path)

pd.DataFrame(
    results,
    columns=[
        "Dataset",
        "Resample",
        "Estimator",
        "Accuracy",
        "BalancedAccuracy",
        "LogLoss",
        "ROCAUC",
        "F1",
        "Runtime",
        "Runtime_FeatureExtractionTrain",
        "Runtime_ClassifierFit",
        "Runtime_FeatureExtractionTest",
        "Runtime_Prediction",
    ],
).to_csv(os.path.join(path, "metrics.csv"), index=False)
