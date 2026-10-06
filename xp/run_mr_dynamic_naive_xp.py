#!/usr/bin/env python3
import os
import sys
import argparse

# Ensure execution from xp directory so common.py does not fail
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(os.getcwd()) != "xp":
    os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from expTools import execute
from common import EXEC_OMP, EXEC_MPI, EXEC_STARPU, PERF_DIR, MPI_PATH, ROOT_PROJECT_PATH

# Configuration variables
DEBUG = False  # Set to False to run the actual benchmarks, True to debug command lines
TASKS_PER_UNIT = 1  # Number of StarPU tasks per calculation unit

min_levels = [6, 6, 6]
max_levels = [11, 12, 13]
max_iter = [60, 40, 20]
unit_counts = [1, 2, 4, 8, 16]
repetitions = 3

# Define experiment-specific executable paths
OMP_BIN = os.path.join(EXEC_OMP, "finite-volume-advection-2d-mra-dynamic")
MPI_BIN = os.path.join(EXEC_MPI, "finite-volume-advection-2d-mra-dynamic")
STARPU_BIN = os.path.join(EXEC_STARPU, "finite-volume-advection-2d-starpu-mr-dynamic-naive")

# Define experiment-specific performance CSV path
csv_file = os.path.join(PERF_DIR, "perf_results_mr_dynamic_naive.csv")

options = {}
options["--min-level, --max-level, --max-iter"] = min_levels, max_levels, max_iter
options["--Tf"] = [1_000_000.0]
options["--no-save"] = [True]
options["--save-perf"] = [csv_file]

def run_benchmarks(reps=repetitions, reset_csv=False):
    if reset_csv and os.path.exists(csv_file):
        print(f"Removing old results file: {csv_file}")
        os.remove(csv_file)

    env = {}

    ###################### OpenMP ######################
    print("\n--- Running OpenMP (MR Dynamic) ---")
    env["OMP_NUM_THREADS"] = unit_counts
    options_omp = options.copy()
    options_omp["--label"] = ["omp"]

    execute(OMP_BIN, env, options_omp, nbruns=reps, verbose=True, pwd=ROOT_PROJECT_PATH, debug=DEBUG)

    ####################### MPI ######################
    print("\n--- Running MPI (MR Dynamic) ---")
    for units in unit_counts:
        options_mpi = options.copy()
        options_mpi["--label"] = ["mpi"]
        cmd = f"{MPI_PATH} -np {units} {MPI_BIN}"
        execute(cmd, {}, options_mpi, nbruns=reps, verbose=True, pwd=ROOT_PROJECT_PATH, debug=DEBUG)

    ###################### StarPU ######################
    print("\n--- Running StarPU (MR Dynamic Naive) ---")
    for units in unit_counts:
        env_sp = {"STARPU_NCPU": [units]}
        nb_tasks = units * TASKS_PER_UNIT
        options_sp = options.copy()
        options_sp["--nb-task"] = [nb_tasks]
        options_sp["--label"] = ["starpu"]
        execute(STARPU_BIN, env_sp, options_sp, nbruns=reps, verbose=True, pwd=ROOT_PROJECT_PATH, debug=DEBUG)

    print("\nAll MR dynamic benchmark runs completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MR Dynamic Naive benchmark suite.")
    parser.add_argument("--reps", type=int, default=repetitions, help=f"Number of repetitions per config (default: {repetitions})")
    parser.add_argument("--reset", action="store_true", help="Reset/remove existing perf_results CSV before benchmarking")
    args = parser.parse_args()

    run_benchmarks(reps=args.reps, reset_csv=args.reset)
