#!/bin/bash -l
#SBATCH --job-name=rust_ortools_batch
#SBATCH --partition=bigmem
#SBATCH --output=LAMA.out
#SBATCH --error=LAMA.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=2-00:30:00

cd "$SLURM_SUBMIT_DIR"

module --quiet purge
module load hosts/hopper
module load singularity

export SINGULARITYENV_ORTOOLS_PREFIX="/scratch/dblackle/ortools"
export SINGULARITYENV_LD_LIBRARY_PATH="/scratch/dblackle/ortools/lib"
export SINGULARITYENV_LIBRARY_PATH="/scratch/dblackle/ortools/lib"

CONTAINER="/scratch/dblackle/rust_env.sif"
SG_RUN="singularity exec --bind /scratch/dblackle --bind /scratch/dblackle/cargo:/usr/local/cargo $CONTAINER"

echo "Building project inside container..."
# Use cargo rustc to link protobuf only to the final executable
$SG_RUN cargo rustc --release -- -Clink-arg=-lprotobuf

NAMES=("paris" "manhattan" "shanghai" "amsterdam" "spitz" "cali")
DATA_DIR="./databases"

echo "Starting batch processing..."

for name in "${NAMES[@]}"; do
    echo "========================================"
    echo "Running configuration for: $name"
    echo "========================================"

    $SG_RUN ./target/release/frequency_analysis_limits \
        --dir-path "$DATA_DIR/$name" \
        --name "$name" \
        --save \
        --plot

    if [ $? -ne 0 ]; then
        echo "Error: Run failed for $name"
    else
        echo "Successfully completed $name"
    fi
done

echo "All runs completed."