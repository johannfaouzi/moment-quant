import os

import pandas as pd
from aeon.datasets import load_classification
from aeon.datasets.tsc_datasets import redux, univariate_equal_length

from src.classification.utils import NUM_RESAMPLES, load_resample

out_dir = os.path.join("results", "analyses", "ucr142_loading_check")


def main():
    dataset_names = list(univariate_equal_length) + list(redux)
    redux_set = set(redux)

    shape_rows = []
    resample_failures = []

    for i, name in enumerate(dataset_names):
        print(f"[{i + 1:>3}/{len(dataset_names)}] {name}")

        try:
            X_train_full, y_train_full = load_classification(
                name, split="train", load_equal_length=True, load_no_missing=True
            )
            X_test_full, y_test_full = load_classification(
                name, split="test", load_equal_length=True, load_no_missing=True
            )
        except Exception as exc:
            resample_failures.append((name, None, f"{type(exc).__name__}: {exc}"))
            print(f"    FAILED to load base train/test split: {type(exc).__name__}: {exc}")
            continue

        shape_rows.append(
            dict(
                Dataset=name,
                Archive="redux_30" if name in redux_set else "original_112",
                SeriesLength=int(X_train_full.shape[-1]),
                NumSamplesTrain=int(X_train_full.shape[0]),
                NumSamplesTest=int(X_test_full.shape[0]),
                NumClasses=int(len(set(y_train_full.tolist()) | set(y_test_full.tolist()))),
            )
        )

        for resample in range(NUM_RESAMPLES):
            try:
                load_resample(name, resample, X_train_full, y_train_full, X_test_full, y_test_full)
            except Exception as exc:
                resample_failures.append((name, resample, f"{type(exc).__name__}: {exc}"))
                print(f"    FAILED resample {resample}: {type(exc).__name__}: {exc}")

    os.makedirs(out_dir, exist_ok=True)
    shapes_path = os.path.join(out_dir, "dataset_shapes.csv")
    pd.DataFrame(shape_rows).to_csv(shapes_path, index=False)
    print()
    print(f"Saved {len(shape_rows)} rows to {shapes_path!r}")

    total_combinations = len(dataset_names) * NUM_RESAMPLES
    print()
    print("=" * 100)
    print(
        f"{len(dataset_names)} datasets x {NUM_RESAMPLES} resamples = {total_combinations} "
        f"(dataset, resample) combinations checked -- {len(resample_failures)} failure(s)"
    )
    print("=" * 100)
    if resample_failures:
        for name, resample, error in resample_failures:
            label = "base split" if resample is None else f"resample {resample}"
            print(f"  {name} [{label}]: {error}")
    else:
        print("All combinations loaded successfully.")


if __name__ == "__main__":
    main()
