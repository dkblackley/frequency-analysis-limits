#!/bin/bash -l
#SBATCH --job-name=LAMA_probs_2d
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=210G
#SBATCH --time=0-04:30:00

# Probabilistic setting on every 15x15 grid (15x15_run_1 .. 15x15_run_10):
# 6 names * 3 dists * 10 dirs * 3 probs = 540 combinations (indexes 0 through 539)
#SBATCH --array=0-539

#SBATCH --output=logs/LAMA_probs_2d_%A_%a.out
#SBATCH --error=logs/LAMA_probs_2d_%A_%a.err

# Needs the binary built by run_LAMA_perfect.slurm (this script does not build).

cd "$SLURM_SUBMIT_DIR"

module --quiet purge
module load hosts/hopper
module load singularity

ORTOOLS_DIR="/scratch/dblackle/ortools"
CMAKE_DIR="/scratch/dblackle/cmake_portable"
LLVM_DIR="/scratch/dblackle/llvm_portable"
CONTAINER="/scratch/dblackle/rust_env.sif"

export SINGULARITYENV_ORTOOLS_PREFIX="$ORTOOLS_DIR"
export SINGULARITYENV_LD_LIBRARY_PATH="$ORTOOLS_DIR/lib"
export SINGULARITYENV_LIBRARY_PATH="$ORTOOLS_DIR/lib"
export SINGULARITYENV_RUST_LOG="debug"
export SINGULARITYENV_RUST_BACKTRACE="1"
export SINGULARITYENV_PREPEND_PATH="$CMAKE_DIR/bin"
export SINGULARITYENV_LIBCLANG_PATH="$LLVM_DIR/lib"
export SINGULARITYENV_RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK

SG_RUN="singularity exec --bind /scratch/dblackle --bind /scratch/dblackle/cargo:/usr/local/cargo $CONTAINER"

if [ ! -x ./target/release/frequency_analysis_limits ]; then
    echo "Binary not found: build it first (run_LAMA_perfect.slurm builds it)."
    exit 1
fi

NAMES=("drink" "busstop" "highway" "shopparis" "cali" "spitz")
DISTS=("gaussian" "beta" "uniform")
DATA_DIRS=( "${SLURM_SUBMIT_DIR}/databases/15x15_run_"{1..10} )
PROBS=("0.3" "0.2" "0.1")

# --- PARAMETER DECODING MATH ---
task_id=$SLURM_ARRAY_TASK_ID

num_probs=${#PROBS[@]}
num_dists=${#DISTS[@]}
num_names=${#NAMES[@]}
num_dirs=${#DATA_DIRS[@]}

total=$((num_probs * num_dists * num_names * num_dirs))
if [ "$task_id" -ge "$total" ]; then
    echo "Task $task_id is outside the $total combinations, nothing to do."
    exit 0
fi

prob_idx=$((task_id % num_probs))
task_id=$((task_id / num_probs))

dist_idx=$((task_id % num_dists))
task_id=$((task_id / num_dists))

name_idx=$((task_id % num_names))
dir_idx=$((task_id / num_names))

prob=${PROBS[$prob_idx]}
dist=${DISTS[$dist_idx]}
name=${NAMES[$name_idx]}
data_dir=${DATA_DIRS[$dir_idx]}
padding=0
t_val=2

echo "========================================"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running configuration:"
echo "  Data Dir: $data_dir"
echo "  Name:     $name"
echo "  Dist:     $dist"
echo "  Prob:     $prob"
echo "========================================"

$SG_RUN ./target/release/frequency_analysis_limits \
    --dir-path "$data_dir" \
    --name "$name" \
    --t "$t_val" \
    --dist "$dist" \
    --padding "$padding" \
    --query-percent "$prob"

if [ $? -ne 0 ]; then
    echo "Error: run failed. Exiting with failure code."
    exit 1
fi

echo "Job completed successfully."
exit 0
