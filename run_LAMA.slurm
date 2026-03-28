#!/bin/bash -l
#SBATCH --job-name=rust_ortools_batch
#SBATCH --partition=bigmem
#SBATCH --output=rust_batch.out
#SBATCH --error=rust_batch.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=2-00:30:00

cd "$SLURM_SUBMIT_DIR"

module --quiet purge
module load hosts/hopper

# Setup custom Rust paths on scratch
export RUSTUP_HOME=/scratch/dblackle/rustup
export CARGO_HOME=/scratch/dblackle/cargo
export PATH="$CARGO_HOME/bin:$PATH"

# Setup OR-Tools linker paths
export ORTOOLS_DIR="$SLURM_SUBMIT_DIR/or-tools"
export LD_LIBRARY_PATH="$ORTOOLS_DIR/lib:$LD_LIBRARY_PATH"

# Build the Rust project once before executing the loop
echo "Building project..."
cargo build --release

# Define the targets extracted from your Rust match/if logic
NAMES=("paris" "manhattan" "shanghai" "amsterdam" "spitz" "cali")

# Define your data directory (update this to point to where your .json files live)
DATA_DIR="./databases"

echo "Starting batch processing..."

# Iterate over all names
for name in "${NAMES[@]}"; do
    echo "========================================"
    echo "Running configuration for: $name"
    echo "========================================"

    # Execute the binary.
    # Note: --save and --plot are included to set those boolean flags to true.
    # Add --load here if you also want to load checkpoints.
    ./target/release/your_rust_binary \
        --dir-path "$DATA_DIR" \
        --name "$name" \
        --save \
        --plot

    # Check the exit status of the Rust program
    if [ $? -ne 0 ]; then
        echo "Error: Run failed for $name"
    else
        echo "Successfully completed $name"
    fi
done

echo "All runs completed."