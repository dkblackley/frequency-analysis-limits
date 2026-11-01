#!/bin/bash -l
#SBATCH --job-name=LAMA_perfect
#SBATCH --partition=normal
##SBATCH --nodelist=amd049
##SBATCH --reservation=dblackle_288
#SBATCH --output=LAMA_perfect.out
#SBATCH --error=LAMA_perfect.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=480G
#SBATCH --time=5-00:00:00

# Perfect knowledge (100% of queries, exact distribution):
#   - every 2D dataset at 50x50 with t=3, on uniform/gaussian/beta
#   - the 3D nh dataset (16x16x14) with t=4, on uniform/gaussian/beta
# This is also the script that BUILDS the binary. The other three scripts
# (run_LAMA_probs_2d, run_LAMA_probs_highdim, run_LAMA_noise) only use it.

cd "$SLURM_SUBMIT_DIR"

module --quiet purge
module load hosts/hopper
module load singularity

# ---------------------------------------
# 1. Toolchain (only downloaded if missing, so concurrent jobs don't wipe each other)
# ---------------------------------------
ORTOOLS_DIR="/scratch/dblackle/ortools"
CMAKE_DIR="/scratch/dblackle/cmake_portable"
LLVM_DIR="/scratch/dblackle/llvm_portable"
CONTAINER="/scratch/dblackle/rust_env.sif"

if [ ! -d "$ORTOOLS_DIR/lib" ]; then
    echo "Setting up OR-Tools..."
    mkdir -p "$ORTOOLS_DIR" && cd "$ORTOOLS_DIR"
    wget -q https://github.com/google/or-tools/releases/download/v9.15/or-tools_amd64_debian-12_cpp_v9.15.6755.tar.gz
    tar -xf or-tools_amd64_debian-12_cpp_v9.15.6755.tar.gz --strip-components=1
    rm or-tools_amd64_debian-12_cpp_v9.15.6755.tar.gz
    cd "$SLURM_SUBMIT_DIR"
fi

if [ ! -x "$CMAKE_DIR/bin/cmake" ]; then
    echo "Setting up portable CMake..."
    mkdir -p "$CMAKE_DIR" && cd "$CMAKE_DIR"
    wget -q https://github.com/Kitware/CMake/releases/download/v3.29.3/cmake-3.29.3-linux-x86_64.tar.gz
    tar -xf cmake-3.29.3-linux-x86_64.tar.gz --strip-components=1
    rm cmake-3.29.3-linux-x86_64.tar.gz
    cd "$SLURM_SUBMIT_DIR"
fi

if [ ! -d "$LLVM_DIR/lib" ]; then
    echo "Setting up portable LLVM/Clang..."
    mkdir -p "$LLVM_DIR" && cd "$LLVM_DIR"
    wget -q https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.4/clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz
    tar -xf clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz --strip-components=1
    rm clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz
    cd "$SLURM_SUBMIT_DIR"
fi

export SINGULARITYENV_ORTOOLS_PREFIX="$ORTOOLS_DIR"
export SINGULARITYENV_LD_LIBRARY_PATH="$ORTOOLS_DIR/lib"
export SINGULARITYENV_LIBRARY_PATH="$ORTOOLS_DIR/lib"
export SINGULARITYENV_PREPEND_PATH="$CMAKE_DIR/bin"
export SINGULARITYENV_LIBCLANG_PATH="$LLVM_DIR/lib"

source /home/dblackle/miniconda3/etc/profile.d/conda.sh
conda activate main
export PYTHON_EXEC="/home/dblackle/.conda/envs/main/bin/python3.10"

export SINGULARITYENV_RUST_LOG="debug"
export SINGULARITYENV_RUST_BACKTRACE="1"
export SINGULARITYENV_RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK

SG_RUN="singularity exec --bind /scratch/dblackle --bind /scratch/dblackle/cargo:/usr/local/cargo $CONTAINER"

# ---------------------------------------
# 2. Build (no `cargo clean`: other jobs may be running the binary)
# ---------------------------------------
echo "Building project inside container..."
$SG_RUN cargo rustc --bin frequency_analysis_limits --release -- -Clink-arg=-lprotobuf
if [ $? -ne 0 ]; then
    echo "Build failed."
    exit 1
fi
echo "Build finished."

# ---------------------------------------
# 3. Runs
# ---------------------------------------
padding=0
DISTS=("uniform" "gaussian" "beta")
FAILED_RUNS=()

# run_one <data_dir> <name> <dist> <t>
# Retries once with t-1 if the run fails (e.g. out of memory at the higher t).
run_one() {
    local data_dir=$1 name=$2 dist=$3 t_val=$4
    echo "========================================"
    echo "Running configuration:"
    echo "  Data Dir: $data_dir"
    echo "  Name:     $name"
    echo "  Dist:     $dist"
    echo "  t:        $t_val"
    echo "========================================"

    $SG_RUN ./target/release/frequency_analysis_limits \
        --dir-path "$data_dir" \
        --name "$name" \
        --t "$t_val" \
        --dist "$dist" \
        --padding "$padding" \
        --query-percent 1.0

    if [ $? -ne 0 ]; then
        local retry_t=$((t_val - 1))
        echo "Error: run failed for $name (Dist: $dist, t=$t_val). Retrying with t=$retry_t..."
        $SG_RUN ./target/release/frequency_analysis_limits \
            --dir-path "$data_dir" \
            --name "$name" \
            --t "$retry_t" \
            --dist "$dist" \
            --padding "$padding" \
            --query-percent 1.0
        if [ $? -ne 0 ]; then
            echo "Error: Retry failed."
            FAILED_RUNS+=("$name | Dist: $dist | Dir: $data_dir | t: $t_val")
        else
            echo "Completed on retry with t=$retry_t."
            FAILED_RUNS+=("$name | Dist: $dist | Dir: $data_dir | only reached t=$retry_t")
        fi
    else
        echo "Successfully completed."
    fi
    echo ""
}

# 2D datasets at 50x50, t=3
NAMES=("drink" "busstop" "highway" "shopparis" "cali" "spitz")
for name in "${NAMES[@]}"; do
    for dist in "${DISTS[@]}"; do
        run_one "$SLURM_SUBMIT_DIR/databases/50x50" "$name" "$dist" 3
    done
done

# 3D dataset, t=4
for dist in "${DISTS[@]}"; do
    run_one "$SLURM_SUBMIT_DIR/databases/16x16x14" "nh" "$dist" 4
done

echo "All runs completed."
if [ ${#FAILED_RUNS[@]} -ne 0 ]; then
    echo "The following configurations failed or fell back to a lower t:"
    for failed in "${FAILED_RUNS[@]}"; do
        echo "  - $failed"
    done
    exit 1
else
    echo "All configurations processed successfully."
fi
