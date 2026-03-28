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

# Setup OR-Tools paths using the SINGULARITYENV_ prefix.
export SINGULARITYENV_ORTOOLS_PREFIX="/scratch/dblackle/ortools"
export SINGULARITYENV_LD_LIBRARY_PATH="/scratch/dblackle/ortools/lib"

# Tell the compiler where to find C++ libraries at build time without poisoning RUSTFLAGS
export SINGULARITYENV_LIBRARY_PATH="/scratch/dblackle/ortools/lib"

# Define the container run command with Bind Mounts
CONTAINER="/scratch/dblackle/rust_env.sif"
SG_RUN="singularity exec --bind /scratch/dblackle --bind /scratch/dblackle/cargo:/usr/local/cargo $CONTAINER"

echo "Building project inside container..."
$SG_RUN cargo build --release

NAMES=("paris" "manhattan" "shanghai" "amsterdam" "spitz" "cali")
DATA_DIR="./databases"

echo "Starting batch processing..."

for name in "${NAMES[@]}"; do
    echo "========================================"
    echo "Running configuration for: $name"
    echo "========================================"

    # Run the compiled binary inside the container
    # Notice we append /$name to the DATA_DIR so the Rust code evaluates to databases/cali/cali.json
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