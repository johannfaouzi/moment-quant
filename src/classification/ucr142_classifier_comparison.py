import os
import time

os.environ["KMP_WARNINGS"] = "FALSE"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
from aeon.datasets import load_classification
from aeon.datasets.tsc_datasets import redux, univariate_equal_length
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import LabelEncoder

from src.classification.utils import (
    MAX_FEATURES,
    NUM_ESTIMATORS,
    NUM_RESAMPLES,
    SMOKE_TEST_DATASET_NAMES,
    compute_classification_metrics,
    load_resample,
    predict_with_proba,
)
from src.estimators.moment_quant import MomentQuantTransformer
from src.utils import DEPTH, DIV, SMOKE_TEST, VERBOSE, periodic_cooldown

n_jobs = -1
ridge_alphas = np.logspace(-3, 3, 10)

feature_extractor = MomentQuantTransformer(depth=DEPTH, div=DIV, mode="approx", approx_version="auto", parallel=True)


def make_classifiers(random_state):
    return {
        "extra_trees": ExtraTreesClassifier(
            n_estimators=NUM_ESTIMATORS,
            max_features=MAX_FEATURES,
            criterion="entropy",
            n_jobs=n_jobs,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=NUM_ESTIMATORS,
            max_features=MAX_FEATURES,
            criterion="entropy",
            n_jobs=n_jobs,
            random_state=random_state,
        ),
        "ridge": RidgeClassifierCV(alphas=ridge_alphas),
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

        Z_train = feature_extractor.fit_transform(X_train_numpy)
        Z_test = feature_extractor.transform(X_test_numpy)

        for classifier_name, classifier in make_classifiers(random_state=resample).items():
            classifier.fit(Z_train, y_train_enc)
            y_pred, y_proba = predict_with_proba(classifier, Z_test, n_classes)
            metrics = compute_classification_metrics(y_test_enc, y_pred, y_proba, n_classes)

            results.append(
                [
                    name,
                    resample,
                    classifier_name,
                    metrics["Accuracy"],
                    metrics["BalancedAccuracy"],
                    metrics["LogLoss"],
                    metrics["ROCAUC"],
                    metrics["F1"],
                ]
            )
            last_cooldown = periodic_cooldown(last_cooldown)

path = os.path.join("results", "classification", "ucr142_classifier_comparison")
if not os.path.isdir(path):
    os.makedirs(path)

pd.DataFrame(
    results,
    columns=[
        "Dataset",
        "Resample",
        "Classifier",
        "Accuracy",
        "BalancedAccuracy",
        "LogLoss",
        "ROCAUC",
        "F1",
    ],
).to_csv(os.path.join(path, "metrics.csv"), index=False)
