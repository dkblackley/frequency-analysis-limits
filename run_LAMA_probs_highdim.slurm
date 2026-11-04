#!/bin/bash -l
#SBATCH --job-name=LAMA_probs_highdim
#SBATCH --partition=bigmem
#SBATCH --output=LAMA_probs_highdim.out
#SBATCH --error=LAMA_probs_highdim.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=3200G
#SBATCH --time=2-00:30:00

# Probabilistic setting (30% of queries, Gaussian only) on the higher-dim data:
#   - the 3D nh dataset (16x16x14)
#   - the synthetic grid DBs 6x6, 6x6x6, ..., 6x6x6x6x6x6 (--name grid --dim 2..6)
# Needs the binary built by build_bin.slurm (this script does not build).

cd "$SLURM_SUBMIT_DIR"

module --quiet purge
module load hosts/hopper
module load singularity

CONTAINER="/scratch/dblackle/rust_env.sif"

export SINGULARITYENV_RUST_LOG="debug"
export SINGULARITYENV_RUST_BACKTRACE="1"
export SINGULARITYENV_RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Same container and flags as build_bin.slurm; all runtime libs are in the image.
SG_RUN="singularity exec --cleanenv --bind /scratch/dblackle $CONTAINER"

if [ ! -x ./target/release/frequency_analysis_limits ]; then
    echo "Binary not found: build it first with build_bin.slurm."
    exit 1
fi

padding=0
dist="gaussian"
prob="0.3"
t_val=2
FAILED_RUNS=()

# run_one <data_dir> <name> <dim>
run_one() {
    local data_dir=$1 name=$2 dim=$3
    echo "========================================"
    echo "Running configuration:"
    echo "  Data Dir: $data_dir"
    echo "  Name:     $name"
    echo "  Dim:      $dim"
    echo "  Dist:     $dist"
    echo "  Prob:     $prob"
    echo "========================================"

    $SG_RUN ./target/release/frequency_analysis_limits \
        --dir-path "$data_dir" \
        --name "$name" \
        --t "$t_val" \
        --dim "$dim" \
        --dist "$dist" \
        --padding "$padding" \
        --query-percent "$prob"

    if [ $? -ne 0 ]; then
        echo "Error: run failed for $name (Dir: $data_dir)."
        FAILED_RUNS+=("$name | Dim: $dim | Dir: $data_dir")
    else
        echo "Successfully completed."
    fi
    echo ""
}

# 3D dataset (--dim is only used by the synthetic grid, but pass the true value)
run_one "$SLURM_SUBMIT_DIR/databases/16x16x14" "nh" 3

# Synthetic grids: each dimension has size 6
for dim in {2..6}; do
    dim_str="6"
    for ((i=1; i<dim; i++)); do
        dim_str+="x6"
    done
    run_one "$SLURM_SUBMIT_DIR/databases/$dim_str" "grid" "$dim"
done

echo "All runs completed."
if [ ${#FAILED_RUNS[@]} -ne 0 ]; then
    echo "The following configurations failed:"
    for failed in "${FAILED_RUNS[@]}"; do
        echo "  - $failed"
    done
    exit 1
else
    echo "All configurations processed successfully."
fi
