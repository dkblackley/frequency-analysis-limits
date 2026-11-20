#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# One LAMa configuration. Submitted by run_me.sh, which sets the partition, CPUs, memory, time,
# job name, log files and LAMA_ARGS (the binary's arguments) on the sbatch command line.
cd "$SLURM_SUBMIT_DIR"

module --quiet purge
module load hosts/hopper
module load singularity

CONTAINER="/scratch/dblackle/rust_env.sif"
export SINGULARITYENV_RUST_LOG="${LAMA_LOG_LEVEL:-info}"
export SINGULARITYENV_RUST_BACKTRACE="1"
export SINGULARITYENV_RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# The snapshot bin/lama points at NOW (run_me.sh makes this job wait for a pending build)
BIN=$(readlink -f bin/lama)
if [ ! -x "$BIN" ]; then
    echo "No binary at bin/lama: run build_bin.slurm (run_me.sh does it for you)." >&2
    exit 1
fi
echo "host:   $(hostname)"
echo "binary: $BIN"
echo "args:   $LAMA_ARGS"
# shellcheck disable=SC2086
singularity exec --cleanenv --bind /scratch/dblackle "$CONTAINER" "$BIN" $LAMA_ARGS
