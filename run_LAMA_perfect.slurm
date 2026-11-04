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
# Needs the binary built by build_bin.slurm (this script does not build).

cd "$SLURM_SUBMIT_DIR"

module --quiet purge
module load hosts/hopper
module load singularity

CONTAINER="/scratch/dblackle/rust_env.sif"

source /home/dblackle/miniconda3/etc/profile.d/conda.sh
conda activate main
export PYTHON_EXEC="/home/dblackle/.conda/envs/main/bin/python3.10"

export SINGULARITYENV_RUST_LOG="debug"
export SINGULARITYENV_RUST_BACKTRACE="1"
export SINGULARITYENV_RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Same container and flags as build_bin.slurm; all runtime libs are in the image.
SG_RUN="singularity exec --cleanenv --bind /scratch/dblackle $CONTAINER"

if [ ! -x ./target/release/frequency_analysis_limits ]; then
    echo "Binary not found: build it first with build_bin.slurm."
    exit 1
fi

# ---------------------------------------
# Runs
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
