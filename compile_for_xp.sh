#!/bin/bash

# Find and source conda.sh to define the conda function
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    . "/opt/conda/etc/profile.d/conda.sh"
else
    # Fallback: check if conda command is available on PATH and use hook
    if command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)"
    else
        echo "Error: conda could not be found or initialized." >&2
        exit 1
    fi
fi

conda activate samurai-mpi-env || { echo "Failed to activate conda environment 'samurai-mpi-env'. Please ensure that the environment exists and is properly configured." >&2; exit 1; }

cmake . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install_starpu/ -DWITH_MPI=OFF -DWITH_OPENMP=OFF -DWITH_STARPU=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && \
cmake --build ./build --target finite-volume-advection-2d-starpu -j 8

cmake . -B build_omp -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install_omp/ -DWITH_MPI=OFF -DWITH_OPENMP=ON -DWITH_STARPU=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && \
cmake --build ./build_omp --target finite-volume-advection-2d-mra-uniform -j 8

cmake . -B build_mpi/ -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install_mpi/ -DWITH_MPI=ON -DWITH_OPENMP=OFF -DWITH_STARPU=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && \
cmake --build ./build_mpi/ --target finite-volume-advection-2d-mra-uniform -j 8





