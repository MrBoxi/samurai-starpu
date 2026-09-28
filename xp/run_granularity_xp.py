#!/usr/bin/env python3
import os
from expTools import execute
from common import EXEC_STARPU, PERF_DIR, ROOT_PROJECT_PATH

DEBUG = False

# Fixed to 16 threads to study granularity impact
num_threads = 16
repetitions = 3

task_counts = [16, 32, 64, 128, 256]

# Mesh configurations
levels_uniform = [11, 12, 13]
max_iter_uniform = [200, 100, 50]

min_levels_mr = [6, 6, 6]
levels_mr = [13, 14, 15]
max_iter_mr = [200, 100, 50]

STARPU_UNIFORM_BIN = os.path.join(EXEC_STARPU, "finite-volume-advection-2d-starpu")
STARPU_MR_STATIC_BIN = os.path.join(EXEC_STARPU, "finite-volume-advection-2d-starpu-mr-static")

csv_file = os.path.join(PERF_DIR, "perf_results_granularity.csv")

env = {"STARPU_NCPU": [num_threads]}

###################### StarPU Uniform ######################
print("\n--- Running StarPU Uniform Granularity Study (16 threads) ---")
options_uniform = {}
options_uniform["--min-level, --max-level, --max-iter"] = levels_uniform, levels_uniform, max_iter_uniform
options_uniform["--Tf"] = [1_000_000.0]
options_uniform["--no-save"] = [True]
options_uniform["--save-perf"] = [csv_file]
options_uniform["--label"] = ["starpu_uniform"]

for nb_tasks in task_counts:
    print(f"\n[Uniform] Testing nb_tasks = {nb_tasks}...")
    opts = options_uniform.copy()
    opts["--nb-task"] = [nb_tasks]
    execute(STARPU_UNIFORM_BIN, env, opts, nbruns=repetitions, verbose=True, pwd=ROOT_PROJECT_PATH, debug=DEBUG)

###################### StarPU MR Static ######################
print("\n--- Running StarPU MR Static Granularity Study (16 threads) ---")
options_mr = {}
options_mr["--min-level, --max-level, --max-iter"] = min_levels_mr, levels_mr, max_iter_mr
options_mr["--Tf"] = [1_000_000.0]
options_mr["--no-save"] = [True]
options_mr["--save-perf"] = [csv_file]
options_mr["--label"] = ["starpu_mr_static"]

for nb_tasks in task_counts:
    print(f"\n[MR Static] Testing nb_tasks = {nb_tasks}...")
    opts = options_mr.copy()
    opts["--nb-task"] = [nb_tasks]
    execute(STARPU_MR_STATIC_BIN, env, opts, nbruns=repetitions, verbose=True, pwd=ROOT_PROJECT_PATH, debug=DEBUG)

print("\nAll granularity benchmark runs completed.")
