#!/bin/bash -l
#SBATCH --job-name=rust_ortools_batch
#SBATCH --partition=bigmem
#SBATCH --output=Plot.out
#SBATCH --error=Plot.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=64G
#SBATCH --time=0-04:30:00

cd "$SLURM_SUBMIT_DIR"

module --quiet purge
module load hosts/hopper
module load singularity

## --- 1. Download and Unpack OR-Tools ---
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
#
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

## --- 3. Download and Unpack portable LLVM/Clang ---
LLVM_DIR="/scratch/dblackle/llvm_portable"
echo "Setting up portable LLVM/Clang..."
rm -rf "$LLVM_DIR"
mkdir -p "$LLVM_DIR"
cd "$LLVM_DIR"

# Download official pre-compiled LLVM binaries (compatible with Debian 12)
wget -q https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.4/clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz
tar -xf clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz --strip-components=1
rm clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz

cd "$SLURM_SUBMIT_DIR"
# ---------------------------------------

# Point Rust's bindgen tool directly to the downloaded libclang.so
export SINGULARITYENV_LIBCLANG_PATH="$LLVM_DIR/lib"

CONTAINER="/scratch/dblackle/rust_env.sif"

export SINGULARITYENV_RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK

SG_RUN="singularity exec --bind /scratch/dblackle --bind /scratch/dblackle/cargo:/usr/local/cargo $CONTAINER"
echo "Wiping corrupted global Cargo registry cache..."
rm -rf /scratch/dblackle/cargo/registry

echo "Cleaning previous cached builds..."
$SG_RUN cargo clean

echo "Building project inside container..."
$SG_RUN cargo rustc --bin frequency_analysis_limits --release -- -Clink-arg=-lprotobuf

$SG_RUN ./target/release/frequency_analysis_limits \
                     --dir-path "databases" \
                     --name "spitz" \
                     --t 2 \
                     --dist "uniform" \
                     --dim "2" \
                     --skip-lama \
                     --plot
