#!/bin/bash -l
#SBATCH --job-name=LAMA_build
#SBATCH --partition=normal
#SBATCH --output=LAMA_build_%j.out
#SBATCH --error=LAMA_build_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# Builds ./target/release/frequency_analysis_limits inside the container made by
# build_and_push_container.sh. All system deps (cmake, clang/libclang, protobuf,
# CBC) are in the image, so nothing else needs to be installed in /scratch.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module --quiet purge
module load hosts/hopper
module load singularity

CONTAINER="/scratch/dblackle/rust_env.sif"
# Writable cargo home for the crate registry/git cache (the .sif is read-only).
CARGO_CACHE="/scratch/dblackle/cargo_home"

if [ ! -e "$CONTAINER" ]; then
    echo "Missing container: $CONTAINER (run build_and_push_container.sh locally first)" >&2
    exit 1
fi
mkdir -p "$CARGO_CACHE"

# .cargo/config.toml points target-dir at a local home directory; override it so the
# binary lands in ./target/release where the run_*.slurm scripts look for it.
export SINGULARITYENV_CARGO_TARGET_DIR="$SLURM_SUBMIT_DIR/target"
export SINGULARITYENV_CARGO_HOME="$CARGO_CACHE"
# Limit both Cargo and CMake (HiGHS) parallelism to the allocated CPUs.
export SINGULARITYENV_CARGO_BUILD_JOBS="$SLURM_CPUS_PER_TASK"
export SINGULARITYENV_CMAKE_BUILD_PARALLEL_LEVEL="$SLURM_CPUS_PER_TASK"

echo "Building project inside container..."

# --cleanenv keeps host module paths (LD_LIBRARY_PATH etc.) out of the build;
# only the SINGULARITYENV_* variables above are passed in.
singularity exec \
    --cleanenv \
    --bind /scratch/dblackle \
    "$CONTAINER" \
    cargo build \
        --jobs "$SLURM_CPUS_PER_TASK" \
        --bin frequency_analysis_limits \
        --release

BIN="$SLURM_SUBMIT_DIR/target/release/frequency_analysis_limits"
if [ ! -x "$BIN" ]; then
    echo "Build reported success but $BIN is missing." >&2
    exit 1
fi
echo "Build finished: $BIN"
