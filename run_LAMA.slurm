#!/bin/bash -l
#SBATCH --job-name=rust_ortools_batch
#SBATCH --partition=bigmem
#SBATCH --output=LAMA.out
#SBATCH --error=LAMA.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=900G
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

CMAKE_DIR="/scratch/dblackle/cmake_portable"
echo "Setting up portable CMake..."
rm -rf "$CMAKE_DIR"
mkdir -p "$CMAKE_DIR"
cd "$CMAKE_DIR"

wget -q https://github.com/Kitware/CMake/releases/download/v3.29.3/cmake-3.29.3-linux-x86_64.tar.gz
tar -xf cmake-3.29.3-linux-x86_64.tar.gz --strip-components=1
rm cmake-3.29.3-linux-x86_64.tar.gz

cd "$SLURM_SUBMIT_DIR"
# ---------------------------------------

# Safely prepend the portable binary to the container's internal PATH
export SINGULARITYENV_PREPEND_PATH="$CMAKE_DIR/bin"

CONTAINER="/scratch/dblackle/rust_env.sif"


SG_RUN="singularity exec --bind /scratch/dblackle --bind /scratch/dblackle/cargo:/usr/local/cargo $CONTAINER"
echo "Wiping corrupted global Cargo registry cache..."
rm -rf /scratch/dblackle/cargo/registry

echo "Cleaning previous cached builds..."
$SG_RUN cargo clean

echo "Building project inside container..."
$SG_RUN cargo rustc --bin frequency_analysis_limits --release -- -Clink-arg=-lprotobuf

NAMES=("shopparis" "highway" "busstop" "drink" "spitz" "cali")
DISTS=("uniform" "gaussian" "beta")
DATA_DIRS=("$SLURM_SUBMIT_DIR/databases/25x25" "$SLURM_SUBMIT_DIR/databases/50x50" "$SLURM_SUBMIT_DIR/databases/75x75")

echo "Starting batch processing..."

# Array to keep track of specific configurations that completely fail
FAILED_RUNS=()

# Loop through each data directory
for data_dir in "${DATA_DIRS[@]}"; do
    # Loop through each name
    for name in "${NAMES[@]}"; do
        # Loop through each distribution type
        for dist in "${DISTS[@]}"; do
            echo "========================================"
            echo "Running configuration:"
            echo "  Data Dir: $data_dir"
            echo "  Name:     $name"
            echo "  Dist:     $dist"
            echo "========================================"

            # Determine initial 't' value based on distribution type
            if [ "$dist" == "uniform" ]; then
                t_val=3
            else
                t_val=2
            fi

            # Initial run
            $SG_RUN ./target/release/frequency_analysis_limits \
                --dir-path "$data_dir/" \
                --name "$name" \
                --t "$t_val" \
                --dist "$dist" \
                --plot

            # Check if the initial run failed
            if [ $? -ne 0 ]; then
                echo "Error: Initial run failed for $name (Dist: $dist, Dir: $data_dir). Retrying with t=4..."

                # Retry one more time with t=4
                $SG_RUN ./target/release/frequency_analysis_limits \
                        --dir-path "$data_dir/" \
                        --name "$name" \
                        --t 4 \
                        --dist "$dist" \
                        --plot

                # Check if the retry also failed
                if [ $? -ne 0 ]; then
                    echo "Error: Retry failed."
                    FAILED_RUNS+=("$name | Dist: $dist | Dir: $data_dir")
                else
                    echo "Successfully completed on retry."
                fi
            else
                echo "Successfully completed."
            fi
            echo "" # Add a blank line for terminal readability
        done
    done
done

echo "All runs completed."

# Report any failures
if [ ${#FAILED_RUNS[@]} -ne 0 ]; then
    echo "The following configurations failed to process completely:"
    for failed in "${FAILED_RUNS[@]}"; do
        echo "  - $failed"
    done
    exit 1
else
    echo "All configurations processed successfully."
fi