#!/bin/bash -l
#SBATCH --job-name=LAMA_noise_sweep
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=480G
#SBATCH --time=0-04:30:00

# Noise sweep with every query observed (--query-percent 1.0) on the single 15x15 grid.
# The attacker knows --dist, queries really come from a distribution --dist-noise away in
# total variation. With noise > 0 it uses the exact largest error as eps (no VC bound) and
# enumerates every DB within it (perfect-world path). Noise 0.0 is the noiseless baseline.
# Above ~0.02-0.03 noise the number of DBs within eps explodes on 15x15, so keep noise small.
# 6 names * 3 dists * 4 noises = 72 combinations (indexes 0 through 71)
#SBATCH --array=0-71

#SBATCH --output=logs/LAMA_noise_sweep_%A_%a.out
#SBATCH --error=logs/LAMA_noise_sweep_%A_%a.err

# Needs the binary built by run_LAMA_perfect.slurm (this script does not build).
# Results: <db>/limits/<db>_<dist>_p1_noise<noise>_* (noise 0.0 -> <db>_<dist>_p1_*).

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
if ! $SG_RUN ./target/release/frequency_analysis_limits --help | grep -q -- "--dist-noise"; then
    echo "This binary has no --dist-noise flag"
    exit 1
fi

NAMES=("drink" "busstop" "highway" "shopparis" "cali" "spitz")
DISTS=("gaussian" "beta" "uniform")
DATA_DIRS=( "${SLURM_SUBMIT_DIR}/databases/15x15" )
NOISES=("0.0" "0.01" "0.02" "0.03")
prob="1.0"

# --- PARAMETER DECODING MATH ---
task_id=$SLURM_ARRAY_TASK_ID

num_noises=${#NOISES[@]}
num_dists=${#DISTS[@]}
num_names=${#NAMES[@]}
num_dirs=${#DATA_DIRS[@]}

total=$((num_noises * num_dists * num_names * num_dirs))
if [ "$task_id" -ge "$total" ]; then
    echo "Task $task_id is outside the $total combinations, nothing to do."
    exit 0
fi

noise_idx=$((task_id % num_noises))
task_id=$((task_id / num_noises))

dist_idx=$((task_id % num_dists))
task_id=$((task_id / num_dists))

name_idx=$((task_id % num_names))
dir_idx=$((task_id / num_names))

noise=${NOISES[$noise_idx]}
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
echo "  Noise:    $noise"
echo "========================================"

$SG_RUN ./target/release/frequency_analysis_limits \
    --dir-path "$data_dir" \
    --name "$name" \
    --t "$t_val" \
    --dist "$dist" \
    --padding "$padding" \
    --query-percent "$prob" \
    --dist-noise "$noise"

if [ $? -ne 0 ]; then
    echo "Error: run failed. Exiting with failure code."
    exit 1
fi

echo "Job completed successfully."
exit 0
