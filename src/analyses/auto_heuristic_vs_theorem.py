import os

import numpy as np
import pandas as pd

from src.analyses.kappa_series import load_approx_constants
from src.estimators.moment_quant import MomentQuantTransformer
from src.theory.cost_formulae import (
    approx_crossover_sample_size,
    crossover_sample_size_all_representations,
    load_kappa_I,
    load_kappa_S,
)
from src.utils import (
    APPROX_INTERVALS,
    APPROX_SAMPLES,
    DEPTH,
    DIV,
    EXACT_INTERVALS,
    EXACT_SAMPLES,
    FULL_L_GRID,
)


exact_constants_csv_path = os.path.join("results", "analyses", "single_thread_grid", "constants.csv")
approx_constants_csv_path = os.path.join("results", "runtimes", "microbench_moment_cf", "constants.csv")
out_dir = os.path.join("results", "analyses", "single_thread_grid")

l_grid = FULL_L_GRID


def load_joint_constants(csv_path, estimator, constant_names):
    df = pd.read_csv(csv_path)
    df = df[(df["Fit"] == "joint_3param") & (df["Estimator"] == estimator)]
    values = {}
    for name in constant_names:
        sub = df[df["Constant"] == name]
        if len(sub) != 1:
            raise ValueError(
                f"Expected exactly one 'joint_3param' row for Estimator={estimator!r}, "
                f"Constant={name!r} in {csv_path!r}, found {len(sub)}."
            )
        values[name] = float(sub.iloc[0]["Value"])
    return tuple(values[name] for name in constant_names)


def main():
    exact_threshold = MomentQuantTransformer._nl_threshold("exact", parallel=False)
    approx_threshold = MomentQuantTransformer._nl_threshold("approx", parallel=False)

    c1_I_exact, c2_I_exact, c3_I_exact = load_joint_constants(
        exact_constants_csv_path, EXACT_INTERVALS, ("c_1", "c_2", "c_3")
    )
    c1_S_exact, c2_S_exact, c3_S_exact = load_joint_constants(
        exact_constants_csv_path, EXACT_SAMPLES, ("c_1", "c_2", "c_3")
    )
    c1_I_approx, c2_I_approx, c3_I_approx = load_approx_constants(approx_constants_csv_path, APPROX_INTERVALS)
    c1_S_approx, c2_S_approx, c3_S_approx = load_approx_constants(approx_constants_csv_path, APPROX_SAMPLES)

    kappa_I_exact = load_kappa_I("exact")
    kappa_I_approx = load_kappa_I("approx")
    kappa_S_exact = load_kappa_S("exact")
    kappa_S_approx = load_kappa_S("approx")

    rows = []
    for L in l_grid:
        basis_exact = L * (np.log2(L) - 2.5)
        n_heuristic_exact = exact_threshold / basis_exact if basis_exact > 0 else float("inf")
        n_heuristic_approx = approx_threshold / L

        exact_error = approx_error = ""
        try:
            n_theorem_exact = crossover_sample_size_all_representations(
                L,
                DEPTH,
                DIV,
                c1_I_exact,
                c2_I_exact,
                c3_I_exact,
                kappa_I_exact,
                c1_S_exact,
                c2_S_exact,
                c3_S_exact,
                kappa_S_exact,
            )
        except ValueError as exc:
            n_theorem_exact = None
            exact_error = str(exc)
        try:
            n_theorem_approx = approx_crossover_sample_size(
                L,
                DEPTH,
                DIV,
                c1_I_approx,
                c2_I_approx,
                c3_I_approx,
                kappa_I_approx,
                c1_S_approx,
                c2_S_approx,
                c3_S_approx,
                kappa_S_approx,
            )
        except ValueError as exc:
            n_theorem_approx = None
            approx_error = str(exc)

        for mode, n_heur, n_thm, prediction_error in [
            ("exact", n_heuristic_exact, n_theorem_exact, exact_error),
            ("approx", n_heuristic_approx, n_theorem_approx, approx_error),
        ]:
            n_thm_value = n_thm if n_thm is not None else float("nan")
            ratio = (n_heur / n_thm) if (n_thm is not None and n_thm > 0) else float("nan")
            rows.append(
                dict(
                    Mode=mode,
                    L=L,
                    n_heuristic=n_heur,
                    n_theorem=n_thm_value,
                    ratio_heuristic_over_theorem=ratio,
                    absolute_difference=n_heur - n_thm_value,
                    Prediction_error=prediction_error,
                )
            )

    out = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "auto_heuristic_vs_theorem.csv")
    out.to_csv(out_path, index=False)

    print("=" * 100)
    print("Auto-heuristic vs Theorem-predicted crossover sample size")
    print("=" * 100)
    print("NOTE (exact mode): the exact-mode crossover below uses")
    print("crossover_sample_size_all_representations, the all-4-representations ENGINEERING")
    print("EXTENSION of Theorem 21 (see cost_formulae.py's own section banner) -- matching")
    print("_decide_version's own scope (the FULL 4-representation transform() call), unlike")
    print("Theorem 21 itself (raw representation only, by deliberate choice in the paper). This is")
    print("a validated extension, not (yet) a proven theorem -- approx mode's comparison remains")
    print("the only one backed end to end by a proven theorem (Theorem 33).")
    print()
    print(out.to_string(index=False))
    print()
    print(f"Saved {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
