import os
import time

DEPTH = 6
DIV = 4

EXACT_SAMPLES = "moment_quant_exact_samples"
EXACT_INTERVALS = "moment_quant_exact_intervals"
APPROX_SAMPLES = "moment_quant_approx_samples"
APPROX_INTERVALS = "moment_quant_approx_intervals"

SMOKE_TEST = os.environ.get("MOMENT_QUANT_SMOKE_TEST", "0") == "1"

N_RUNS = 2 if SMOKE_TEST else 10

VERBOSE = True

FULL_L_GRID = [16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 8192]

N_SAMPLES_GRID = [1, 3, 10, 30, 100, 300, 1_000, 3_000, 10_000]

_KAPPA_I_SUPPORT_BASE_L = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

if SMOKE_TEST:
    FULL_L_GRID = [16, 64, 128]
    N_SAMPLES_GRID = [1, 10, 100]
    _KAPPA_I_SUPPORT_BASE_L = [16, 64, 256]

_KAPPA_I_SUPPORT_EXTRA_L = sorted({val for _L in _KAPPA_I_SUPPORT_BASE_L for val in (_L - 1, _L - 2, _L // 2 + 1)})

_CONSTANTS_FIT_EXTRA_L = [20, 24, 28, 40, 48, 56, 80, 88, 96]

if SMOKE_TEST:
    _CONSTANTS_FIT_EXTRA_L = [24, 48]

SINGLE_THREAD_L_GRID = sorted(set(FULL_L_GRID) | set(_KAPPA_I_SUPPORT_EXTRA_L) | set(_CONSTANTS_FIT_EXTRA_L))

SMALL_L_REPEAT_THRESHOLD = 96
N_RUNS_SMALL_L = N_RUNS if SMOKE_TEST else 30

INNER_COOLDOWN_WORK_SECONDS = 300
INNER_COOLDOWN_SECONDS = 20

if SMOKE_TEST:
    INNER_COOLDOWN_WORK_SECONDS = 300_000_000
    INNER_COOLDOWN_SECONDS = 1


def periodic_cooldown(last_time, work_seconds=INNER_COOLDOWN_WORK_SECONDS, cooldown_seconds=INNER_COOLDOWN_SECONDS):
    now = time.perf_counter()
    if now - last_time >= work_seconds:
        time.sleep(cooldown_seconds)
        return time.perf_counter()
    return last_time
