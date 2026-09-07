import os

import pandas as pd

single_csv_path = os.path.join("results", "runtimes", "single_thread_grid", "runtimes.csv")
multi_csv_path = os.path.join("results", "runtimes", "multiple_threads_grid", "runtimes.csv")
out_dir = os.path.join("results", "analyses", "multiple_threads_grid")


def reduce_to_min(df):
    return (
        df.groupby(["Series length", "Number of samples", "Estimator"])["Runtime"]
        .min()
        .reset_index()
        .rename(columns={"Runtime": "Runtime_min"})
    )


def main():
    single = pd.read_csv(single_csv_path)
    multi = pd.read_csv(multi_csv_path)

    threads = int(multi["Threads"].iloc[0]) if "Threads" in multi.columns else None

    single_min = reduce_to_min(single).rename(columns={"Runtime_min": "Runtime_single"})
    multi_min = reduce_to_min(multi).rename(columns={"Runtime_min": "Runtime_multi"})

    speedup = pd.merge(single_min, multi_min, on=["Series length", "Number of samples", "Estimator"], how="inner")
    speedup["Speedup"] = speedup["Runtime_single"] / speedup["Runtime_multi"]
    speedup = speedup.sort_values(["Estimator", "Series length", "Number of samples"])

    os.makedirs(out_dir, exist_ok=True)
    speedup.to_csv(os.path.join(out_dir, "speedup.csv"), index=False)

    print("=" * 100)
    title = "SPEEDUP = T_single_thread / T_multi_thread"
    if threads is not None:
        title += f"  (multi-thread run used Threads={threads})"
    print(title)
    print("=" * 100)
    print(
        speedup.pivot_table(
            index="Series length", columns=["Estimator", "Number of samples"], values="Speedup"
        ).to_string(float_format=lambda v: f"{v:.2f}")
    )

    marginal = multi_min.copy()
    marginal["Runtime_per_sample"] = marginal["Runtime_multi"] / marginal["Number of samples"]
    marginal = marginal.sort_values(["Estimator", "Series length", "Number of samples"])
    marginal.to_csv(os.path.join(out_dir, "marginal_cost.csv"), index=False)

    print()
    print("=" * 100)
    title = "MARGINAL PER-SAMPLE COST (multi-thread only, full n grid): Runtime_min / n"
    if threads is not None:
        title += f"  (Threads={threads})"
    print(title)
    print("=" * 100)
    print(
        marginal.pivot_table(
            index="Series length", columns=["Estimator", "Number of samples"], values="Runtime_per_sample"
        ).to_string(float_format=lambda v: f"{v:.3g}")
    )

    print()
    print("Reading table 2: for a '*_samples' estimator, Runtime_per_sample should fall as n grows")
    print("(more of the parallel work is amortized across threads) then flatten once all threads")
    print("are saturated -- the n at which it flattens is the practical 'multi-threading has paid")
    print("for itself' point. For '*_intervals' estimators, Runtime_per_sample should already be")
    print("roughly flat across n even at small n (same shape as in the single-thread grid), since")
    print("there is no parallel kernel to saturate in the first place.")


if __name__ == "__main__":
    main()
