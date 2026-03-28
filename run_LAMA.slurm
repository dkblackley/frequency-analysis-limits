#!/bin/bash -l
#SBATCH --job-name=rust_ortools_job
#SBATCH --partition=bigmem
#SBATCH --output=rust.out
#SBATCH --error=rust.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=2-00:30:00

cd "$SLURM_SUBMIT_DIR"

module --quiet purge
module load hosts/hopper

export CARGO_HOME=/scratch/dblackle/cargo
export PATH="$CARGO_HOME/bin:$PATH"

# Make Rust toolchain available to the Slurm node
# source "$HOME/.cargo/env"

# Point the linker to your downloaded OR-Tools libraries
export ORTOOLS_DIR="$SLURM_SUBMIT_DIR/or-tools"
export LD_LIBRARY_PATH="$ORTOOLS_DIR/lib:$LD_LIBRARY_PATH"

# Build and run
cargo build --release
./target/release/your_rust_binary