#!/bin/bash -l

# ============================================================
# Local version of run_LAMA_noise_sweep.slurm: every 15x15 dataset,
# 100% of queries, with the true query distribution 0 / 0.01 / 0.02 / 0.03
# away (TV) from --dist. With noise > 0 the attacker uses the exact
# largest error as eps (no VC bound) and enumerates every DB within it.
#
# Override from the command line (all optional):
#   ./local_run_noise_sweep.sh "cali spitz" "gaussian" "0.005 0.02"
#     $1 = names, $2 = dists, $3 = noises
# RUN_TIMEOUT (default 4h30m, like the slurm job) caps each run.
# ============================================================
NAMES=(${1:-drink busstop highway shopparis cali spitz})
DISTS=(${2:-gaussian beta uniform})
NOISES=(${3:-0.05})
RUN_TIMEOUT="${RUN_TIMEOUT:-4h30m}"

export RUST_LOG="debug"
export RUST_BACKTRACE="1"

BIN="$(cargo metadata --format-version 1 --no-deps | grep -o '"target_directory":"[^"]*"' | cut -d'"' -f4)/release/frequency_analysis_limits"
DATA_DIR="databases/15x15"
FAILED_RUNS=()

echo "Building project..."
cargo rustc --bin frequency_analysis_limits --release -- -Clink-arg=-lprotobuf
if [ $? -ne 0 ]; then
    echo "Build failed."
    exit 1
fi
if ! $BIN --help | grep -q -- "--dist-noise"; then
    echo "This binary has no --dist-noise flag: check out the branch with it and rebuild."
    exit 1
fi

for name in "${NAMES[@]}"; do
    for dist in "${DISTS[@]}"; do
        for noise in "${NOISES[@]}"; do
            label="$name | $dist | noise $noise"
            echo "========================================"
            echo "Experiment: $label"
            echo "========================================"
            $BIN \
                --dir-path "$DATA_DIR" \
                --name "$name" \
                --t 2 \
                --dist "$dist" \
                --padding 0 \
                --query-percent 1.0 \
                --dist-noise "$noise"
            rc=$?
            if [ $rc -eq 124 ]; then
                echo "Error: $label timed out after $RUN_TIMEOUT."
                FAILED_RUNS+=("$label (timeout)")
            elif [ $rc -ne 0 ]; then
                echo "Error: $label failed (exit code $rc)."
                FAILED_RUNS+=("$label")
            else
                echo "$label completed successfully."
            fi
            echo ""
        done
    done
done

echo "All runs completed."
if [ ${#FAILED_RUNS[@]} -ne 0 ]; then
    echo "The following experiments failed:"
    for failed in "${FAILED_RUNS[@]}"; do
        echo "  - $failed"
    done
    exit 1
else
    echo "All experiments processed successfully."
fi
