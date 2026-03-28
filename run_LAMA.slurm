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

# We no longer load Hopper's protobuf module; the container handles it.

# Set Cargo home to scratch so the container doesn't try to write to its read-only internal system
export CARGO_HOME=/scratch/dblackle/cargo
export ORTOOLS_PREFIX="/scratch/dblackle/ortools"

# Make sure the container can find Cargo and the 'protoc' compiler included in OR-Tools
export PATH="$CARGO_HOME/bin:$ORTOOLS_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$ORTOOLS_PREFIX/lib:$LD_LIBRARY_PATH"

# Link the Protobuf library included in your OR-Tools download
export RUSTFLAGS='-Clink-arg=-lprotobuf'

# Define how we call the container (binding scratch ensures the container has read/write access to your files)
CONTAINER="/scratch/dblackle/rust_env.sif"
SG_RUN="singularity exec --bind /scratch/dblackle $CONTAINER"

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
    $SG_RUN ./target/release/frequency_analysis_limits \
        --dir-path "$DATA_DIR" \
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