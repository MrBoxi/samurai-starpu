import os

# Check if the user executes this script from here /xp
if os.path.basename(os.getcwd()) != "xp":
    print("Please, execute this script from the 'samurai-starpu/xp' folder")
    exit(1)

# Absolute path to the project root directory
ROOT_PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Absolute paths to the build directories for Finite Volume demos
EXEC_OMP = os.path.abspath(os.path.join(ROOT_PROJECT_PATH, "build_omp", "demos", "FiniteVolume"))
EXEC_MPI = os.path.abspath(os.path.join(ROOT_PROJECT_PATH, "build_mpi", "demos", "FiniteVolume"))
EXEC_STARPU = os.path.abspath(os.path.join(ROOT_PROJECT_PATH, "build", "demos", "FiniteVolume"))

# Absolute path to the performance output directory
PERF_DIR = os.path.abspath(os.path.join(ROOT_PROJECT_PATH, "xp", "perf"))

# Absolute path to Conda mpirun
MPI_PATH = os.path.join(os.environ['HOME'], "miniforge3/envs/samurai-mpi-env/bin/mpirun")
