#!/bin/bash -l
#SBATCH --job-name=LAMA_Array
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=350G
#SBATCH --time=2-00:30:00

# NEW: GMU Array syntax.
# 5 names * 4 dists * 1 dir * 1 prob = 20 total combinations. (Indexes 0 through 19)
#SBATCH --array=0-19

# NEW: GMU best practice for array logging (%A = Master Array ID, %a = Task ID)
#SBATCH --output=logs/LAMA_%A_%a.out
#SBATCH --error=logs/LAMA_%A_%a.err

cd "$SLURM_SUBMIT_DIR"

module --quiet purge
module load hosts/hopper
module load singularity

# ... [Keep ALL your setup code here: OR-Tools, Cmake, LLVM, Cargo build] ...

# Define your parameter arrays
NAMES=("drink" "busstop" "shopparis" "cali" "spitz" )
DISTS=("gaussian" "beta" "uniform")
DATA_DIRS=( "$SLURM_SUBMIT_DIR/databases/15x15" )
PROBS=("0.3" "0.2" "0.1")

# --- PARAMETER DECODING MATH ---
# Map the 1D SLURM_ARRAY_TASK_ID to our 4D parameter space
task_id=$SLURM_ARRAY_TASK_ID

# Calculate array lengths
num_probs=${#PROBS[@]}
num_dists=${#DISTS[@]}
num_names=${#NAMES[@]}
num_dirs=${#DATA_DIRS[@]}

# Extract indices using modulo arithmetic
prob_idx=$((task_id % num_probs))
task_id=$((task_id / num_probs))

dist_idx=$((task_id % num_dists))
task_id=$((task_id / num_dists))

name_idx=$((task_id % num_names))
dir_idx=$((task_id / num_names))

# Assign the specific parameters for THIS isolated job
prob=${PROBS[$prob_idx]}
dist=${DISTS[$dist_idx]}
name=${NAMES[$name_idx]}
data_dir=${DATA_DIRS[$dir_idx]}
padding=3

echo "========================================"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running configuration:"
echo "  Data Dir: $data_dir"
echo "  Name:     $name"
echo "  Dist:     $dist"
echo "  Prob:     $prob"
echo "========================================"

# Determine initial 't' value based on distribution type
if [ "$dist" == "flat" ]; then
    t_val=2
else
    t_val=2
fi

# Run the single configuration assigned to this node
$SG_RUN ./target/release/frequency_analysis_limits \
    --dir-path "$data_dir" \
    --name "$name" \
    --t "$t_val" \
    --dist "$dist" \
    --padding "$padding" \
    --query-percent "$prob"

# Retry logic for this specific configuration
if [ $? -ne 0 ]; then
    echo "Error: Initial run failed. Retrying with t=4..."

    $SG_RUN ./target/release/frequency_analysis_limits \
            --dir-path "$data_dir" \
            --name "$name" \
            --t 2 \
            --dist "$dist" \
            --padding "$padding" \
            --query-percent "$prob"

    if [ $? -ne 0 ]; then
        echo "Error: Retry failed. Exiting with failure code."
        exit 1
    fi
fi

echo "Job completed successfully."
exit 0