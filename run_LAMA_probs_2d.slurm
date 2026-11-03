#!/bin/bash -l
#SBATCH --job-name=LAMA_probs_2d
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=180G
#SBATCH --time=0-05:30:00

# Probabilistic setting on the 50x50 grid:
# 6 names * 3 dists * 1 dir * 5 probs = 90 combinations (indexes 0 through 89)
# If you add DATA_DIRS or PROBS, update the array to 0-(total-1).
#SBATCH --array=0-89

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
# export SINGULARITYENV_RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SINGULARITYENV_RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

SG_RUN="singularity exec --bind /scratch/dblackle --bind /scratch/dblackle/cargo:/usr/local/cargo $CONTAINER"

if [ ! -x ./target/release/frequency_analysis_limits ]; then
    echo "Binary not found: build it first (run_LAMA_perfect.slurm builds it)."
    exit 1
fi

NAMES=("drink" "busstop" "highway" "shopparis" "cali" "spitz")
DISTS=("gaussian" "beta" "uniform")
#DATA_DIRS=( "${SLURM_SUBMIT_DIR}/databases/15x15_run_"{1..10} )
# DATA_DIRS=( "${SLURM_SUBMIT_DIR}/databases/15x15_run_1" )
DATA_DIRS=(
    #"${SLURM_SUBMIT_DIR}/databases/15x15_run_1"
    "${SLURM_SUBMIT_DIR}/databases/50x50"
)
PROBS=("0.3" "0.2" "0.1" "0.01" "0.05")

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

# Try each configuration up to MAX_ATTEMPTS times (one retry). Query
# selection uses thread_rng, so a retry draws a fresh query sample.
# Note: a run killed by the --time limit ends the whole task, so no retry then.
MAX_ATTEMPTS=2
status=1
for attempt in $(seq 1 $MAX_ATTEMPTS); do
    echo "Attempt $attempt of $MAX_ATTEMPTS"
    $SG_RUN ./target/release/frequency_analysis_limits \
        --dir-path "$data_dir" \
        --name "$name" \
        --t "$t_val" \
        --dist "$dist" \
        --padding "$padding" \
        --query-percent "$prob"
    status=$?
    if [ $status -eq 0 ]; then
        break
    fi
    echo "Attempt $attempt failed with exit code $status."
done

if [ $status -ne 0 ]; then
    echo "Error: run failed after $MAX_ATTEMPTS attempts ($name | $dist | p=$prob). Exiting with failure code."
    exit 1
fi

echo "Job completed successfully."
exit 0
