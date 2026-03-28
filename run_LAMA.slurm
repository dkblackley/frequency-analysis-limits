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

# --- 1. Download and Unpack OR-Tools ---
ORTOOLS_DIR="/scratch/dblackle/ortools"
echo "Setting up OR-Tools..."
rm -rf "$ORTOOLS_DIR"
mkdir -p "$ORTOOLS_DIR"
cd "$ORTOOLS_DIR"

wget -q https://github.com/google/or-tools/releases/download/v9.14/or-tools_amd64_debian-12_cpp_v9.14.6206.tar.gz
tar -xf or-tools_amd64_debian-12_cpp_v9.14.6206.tar.gz --strip-components=1
rm or-tools_amd64_debian-12_cpp_v9.14.6206.tar.gz

cd "$SLURM_SUBMIT_DIR"
# ---------------------------------------

export SINGULARITYENV_ORTOOLS_PREFIX="$ORTOOLS_DIR"
export SINGULARITYENV_LD_LIBRARY_PATH="$ORTOOLS_DIR/lib"
export SINGULARITYENV_LIBRARY_PATH="$ORTOOLS_DIR/lib"

# Enable Rust logging and backtraces
export SINGULARITYENV_RUST_LOG="debug"
export SINGULARITYENV_RUST_BACKTRACE="1"

CONTAINER="/scratch/dblackle/rust_env.sif"
SG_RUN="singularity exec --bind /scratch/dblackle --bind /scratch/dblackle/cargo:/usr/local/cargo $CONTAINER"

echo "Cleaning previous cached builds..."
$SG_RUN cargo clean

echo "Building project inside container..."
$SG_RUN cargo rustc --bin frequency_analysis_limits --release -- -Clink-arg=-lprotobuf

NAMES=("paris" "manhattan" "shanghai" "amsterdam" "spitz" "cali")

# Use an absolute path to guarantee file resolution inside the container
DATA_DIR="$SLURM_SUBMIT_DIR/databases"

echo "Starting batch processing..."

for name in "${NAMES[@]}"; do
    echo "========================================"
    echo "Running configuration for: $name"
    echo "========================================"

    $SG_RUN ./target/release/frequency_analysis_limits \
        --dir-path "$DATA_DIR/" \
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