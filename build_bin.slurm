#!/bin/bash -l
#SBATCH --job-name=LAMA_build
#SBATCH --partition=normal
#SBATCH --output=LAMA_build_%j.out
#SBATCH --error=LAMA_build_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00


cd "$SLURM_SUBMIT_DIR"

module --quiet purge
module load hosts/hopper
module load singularity

CMAKE_DIR="/scratch/dblackle/cmake_portable"
LLVM_DIR="/scratch/dblackle/llvm_portable"
CONTAINER="/scratch/dblackle/rust_env.sif"

# Reuse the toolchains installed by your existing setup.
for required in \
    "$ORTOOLS_DIR/lib" \
    "$CMAKE_DIR/bin/cmake" \
    "$LLVM_DIR/lib" \
    "$CONTAINER"
do
    if [ ! -e "$required" ]; then
        echo "Missing build dependency: $required" >&2
        exit 1
    fi
done

export SINGULARITYENV_ORTOOLS_PREFIX="$ORTOOLS_DIR"
export SINGULARITYENV_LD_LIBRARY_PATH="$ORTOOLS_DIR/lib"
export SINGULARITYENV_LIBRARY_PATH="$ORTOOLS_DIR/lib"
export SINGULARITYENV_PREPEND_PATH="$CMAKE_DIR/bin"
export SINGULARITYENV_LIBCLANG_PATH="$LLVM_DIR/lib"

source /home/dblackle/miniconda3/etc/profile.d/conda.sh
conda activate main
export PYTHON_EXEC="/home/dblackle/.conda/envs/main/bin/python3.10"

# Limit both Cargo and CMake parallelism to the allocated CPUs.
export SINGULARITYENV_CARGO_BUILD_JOBS="$SLURM_CPUS_PER_TASK"
export SINGULARITYENV_CMAKE_BUILD_PARALLEL_LEVEL="$SLURM_CPUS_PER_TASK"

echo "Building project inside container..."

singularity exec \
    --bind /scratch/dblackle \
    --bind /scratch/dblackle/cargo:/usr/local/cargo \
    "$CONTAINER" \
    cargo rustc \
        --jobs "$SLURM_CPUS_PER_TASK" \
        --bin frequency_analysis_limits \
        --release \
        -- -Clink-arg=-lprotobuf

echo "Build finished: $SLURM_SUBMIT_DIR/target/release/frequency_analysis_limits"