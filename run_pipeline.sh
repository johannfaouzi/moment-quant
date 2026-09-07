#!/usr/bin/env bash
# Full pipeline runner for the MomentQuant project's src/ scripts.
#
# Platforms: macOS and Linux (any bash >= 4, needed for the associative-array-free but still
# bash-4-specific constructs below, e.g. "${!STAGES[@]}"). NOT usable on native Windows -- there is
# no bash, no caffeinate/systemd-inhibit, and no eval/case/array support there without WSL or Git
# Bash. Use run_pipeline.ps1 (PowerShell) on Windows instead; both scripts define the exact same 37
# stages, --list/--from flags, and cooldown behavior, so pick whichever matches the OS you're on.
#
# Usage
# -----
#   ./run_pipeline.sh                 # run everything, in dependency order
#   ./run_pipeline.sh --from STAGE    # resume starting at STAGE (see stage names below)
#   ./run_pipeline.sh --list          # print stage names and exit
#
# Must be run from the repository root

set -euo pipefail

COOLDOWN_SHORT=10   # between quick analysis scripts (seconds)
COOLDOWN_LONG=60    # after a sustained multi-minute/hour benchmark stage (seconds)

# Prevent the machine from sleeping during a run that can take hours: caffeinate on macOS, and systemd-inhibit on Linux distributions
if [[ "$(uname)" == "Darwin" && -z "${_PIPELINE_CAFFEINATED:-}" ]]; then
    export _PIPELINE_CAFFEINATED=1
    exec caffeinate -i "$0" "$@"
elif [[ "$(uname)" == "Linux" && -z "${_PIPELINE_CAFFEINATED:-}" ]] && command -v systemd-inhibit >/dev/null 2>&1; then
    export _PIPELINE_CAFFEINATED=1
    exec systemd-inhibit --what=sleep:idle --why="run_pipeline.sh benchmark run" "$0" "$@"
fi

cd "$(dirname "$0")"

if [[ ! -d "src" || ! -d "PythonResampleIndices" ]]; then
    echo "ERROR: run this script from the repository root (src/ and PythonResampleIndices/ not found here)." >&2
    exit 1
fi

STAGES=(
    "01_single_thread_grid:python -m src.runtimes.single_thread_grid"
    "02_multiple_threads_grid:python -m src.runtimes.multiple_threads_grid"
    "03_dispatch_thresholds:python -m src.analyses.dispatch_thresholds"
    "04_microbench_sort_extraction:python -m src.runtimes.microbench_sort_extraction"
    "05_microbench_moment_cf:python -m src.runtimes.microbench_moment_cf"
    "06_single_thread_fixed_n_samples:python -m src.runtimes.single_thread_fixed_n_samples"
    "07_single_thread_fixed_n_timepoints:python -m src.runtimes.single_thread_fixed_n_timepoints"
    "08_single_thread_ucr142:python -m src.runtimes.single_thread_ucr142"
    "09_multiple_threads_ucr142:python -m src.runtimes.multiple_threads_ucr142"
    "10_presort_vs_intervals_torch_singlethread:python -m src.runtimes.presort_vs_intervals_torch_singlethread"
    "11_presort_vs_intervals_torch_multithread:python -m src.runtimes.presort_vs_intervals_torch_multithread"
    "12_intervals_numba_vs_intervals_numpy:python -m src.runtimes.intervals_numba_vs_intervals_numpy"
    "13_quant_pytorch_vs_moment_quant_numpy:python -m src.runtimes.quant_pytorch_vs_moment_quant_numpy"
    "14_presort_vs_intervals_numpy:python -m src.runtimes.presort_vs_intervals_numpy"
    "15_single_thread_quant_grid:python -m src.runtimes.single_thread_quant_grid"
    "16_single_thread_ucr142_estimator_comparison:python -m src.classification.single_thread_ucr142_estimator_comparison"
    "17_multiple_threads_ucr142_estimator_comparison:python -m src.classification.multiple_threads_ucr142_estimator_comparison"
    "18_ucr142_classifier_comparison:python -m src.classification.ucr142_classifier_comparison"
    "19_quant_float64_correctness:python -m src.analyses.quant_float64_correctness"
    "20_quant_sort_extraction_constants:python -m src.analyses.quant_sort_extraction_constants"
    "21_quant_kappa_intervals:python -m src.analyses.quant_kappa_intervals"
    "22_sort_extraction_constants:python -m src.analyses.sort_extraction_constants"
    "23_moment_cf_constants:python -m src.analyses.moment_cf_constants"
    "24_kappa_intervals:python -m src.analyses.kappa_intervals"
    "25_kappa_series:python -m src.analyses.kappa_series"
    "26_theory_vs_actual:python -m src.analyses.theory_vs_actual"
    "27_auto_heuristic_vs_theorem:python -m src.analyses.auto_heuristic_vs_theorem"
    "28_speedup:python -m src.analyses.speedup"
    "29_cost_formulae_selftest:python -m src.theory.cost_formulae"
    "30_approx_quantile_correlation:python -m src.analyses.approx_quantile_correlation"
    "31_pipeline_decomposition:python -m src.analyses.pipeline_decomposition"
    "32_accuracy_runtime_tradeoff:python -m src.analyses.accuracy_runtime_tradeoff"
    "33_classification_comparisons:python -m src.analyses.classification_comparisons"
    "34_sort_argsort_numpy:python -m src.analyses.sort_argsort_numpy"
    "35_sort_argsort_numba_isolation:python -m src.analyses.sort_argsort_numba_isolation"
    "36_sort_argsort_torch_singlethread:python -m src.analyses.sort_argsort_torch_singlethread"
    "37_sort_argsort_torch_multithread:python -m src.analyses.sort_argsort_torch_multithread"
)

# ---------------------------------------------------------------------------
# --list / --from handling
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--list" ]]; then
    for entry in "${STAGES[@]}"; do
        echo "${entry%%:*}"
    done
    exit 0
fi

START_INDEX=0
if [[ "${1:-}" == "--from" ]]; then
    target="${2:-}"
    found=0
    for i in "${!STAGES[@]}"; do
        if [[ "${STAGES[$i]%%:*}" == "$target" ]]; then
            START_INDEX=$i
            found=1
            break
        fi
    done
    if [[ "$found" -eq 0 ]]; then
        echo "ERROR: unknown stage '$target'. Run with --list to see valid stage names." >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
TOTAL=${#STAGES[@]}
START_TIME=$(date +%s)

for i in "${!STAGES[@]}"; do
    if (( i < START_INDEX )); then
        continue
    fi

    entry="${STAGES[$i]}"
    name="${entry%%:*}"
    cmd="${entry#*:}"

    echo
    echo "================================================================================"
    echo "[$((i + 1))/${TOTAL}] ${name}  ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "  \$ ${cmd}"
    echo "================================================================================"

    eval "$cmd"

    # Cooldown before the next script, if any remain.
    if (( i + 1 < TOTAL )); then
        case "$name" in
            01_*|02_*|08_*|09_*|10_*|11_*|12_*|13_*|14_*|15_*|16_*|17_*|18_*)
                echo "--- cooldown ${COOLDOWN_LONG}s ---"
                sleep "$COOLDOWN_LONG"
                ;;
            *)
                echo "--- cooldown ${COOLDOWN_SHORT}s ---"
                sleep "$COOLDOWN_SHORT"
                ;;
        esac
    fi
done

ELAPSED=$(( $(date +%s) - START_TIME ))
echo
echo "================================================================================"
echo "Pipeline complete in $(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m $(( ELAPSED % 60 ))s."
echo "================================================================================"
