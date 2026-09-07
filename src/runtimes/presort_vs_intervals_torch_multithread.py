import os

from src.runtimes.presort_vs_intervals_torch import run_benchmark

if __name__ == "__main__":
    run_benchmark(n_threads=os.cpu_count())
