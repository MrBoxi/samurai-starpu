#!/usr/bin/env python3
import os
from expTools import execute
from common import EXEC_OMP, EXEC_MPI, EXEC_STARPU, PERF_DIR, MPI_PATH, ROOT_PROJECT_PATH

DEBUG = False  # Set to False to run the actual benchmarks, True to debug command lines

levels = [11, 12, 13]
max_iter = [200, 100, 50]
unit_counts = [1, 2, 4, 8, 16]
repetitions = 1

# Define experiment-specific executable paths
OMP_BIN = os.path.join(EXEC_OMP, "finite-volume-advection-2d-mra-uniform")
MPI_BIN = os.path.join(EXEC_MPI, "finite-volume-advection-2d-mra-uniform")
STARPU_BIN = os.path.join(EXEC_STARPU, "finite-volume-advection-2d-starpu")

# Define experiment-specific performance CSV path
csv_file = os.path.join(PERF_DIR, "perf_results.csv")

options = {}
options["--min-level, --max-level, --max-iter"] = levels, levels, max_iter
options["--Tf"] = [1_000_000.0]
options["--no-save"] = [True]
options["--save-perf"] = [csv_file]

env = {}

###################### OpenMP ######################
print("\n--- Running OpenMP ---")
env["OMP_NUM_THREADS"] = unit_counts
options_omp = options.copy()
options_omp["--label"] = ["omp"]

execute(OMP_BIN, env, options_omp, nbruns=repetitions, verbose=True, pwd=ROOT_PROJECT_PATH, debug=DEBUG)


####################### MPI ######################
print("\n--- Running MPI ---")
for units in unit_counts:
    options_mpi = options.copy()
    options_mpi["--label"] = ["mpi"]
    cmd = f"{MPI_PATH} -np {units} {MPI_BIN}"
    execute(cmd, {}, options_mpi, nbruns=repetitions, verbose=True, pwd=ROOT_PROJECT_PATH, debug=DEBUG)


###################### StarPU ######################
print("\n--- Running StarPU ---")
for units in unit_counts:
    env_sp = {"STARPU_NCPU": [units]}
    nb_tasks = max(4, units * 4)
    options_sp = options.copy()
    options_sp["--nb-task"] = [nb_tasks]
    options_sp["--label"] = ["starpu"]
    execute(STARPU_BIN, env_sp, options_sp, nbruns=repetitions, verbose=True, pwd=ROOT_PROJECT_PATH, debug=DEBUG)

print("\nAll benchmark runs completed.")
