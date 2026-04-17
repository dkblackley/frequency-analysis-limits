#!/bin/bash -l

#cargo clean

echo "Building project..."
cargo rustc --bin frequency_analysis_limits --release -- -Clink-arg=-lprotobuf

NAMES=("shopparis" "highway" "busstop" "drink" "spitz" "cali")
DISTS=("uniform" "gaussian" "beta")
DATA_DIRS=("databases/25x25" "databases/50x50" "databases/75x75")

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
                t_val=2
            else
                t_val=2
            fi


            # Initial run
            $SG_RUN ./target/release/frequency_analysis_limits \
                --dir-path "$data_dir" \
                --name "$name" \
                --t "$t_val" \
                --dist "$dist"

            # Check if the initial run failed
            if [ $? -ne 0 ]; then
                echo "Error: Initial run failed for $name (Dist: $dist, Dir: $data_dir). Retrying with t=4..."

                # Retry one more time with t=4
                $SG_RUN ./target/release/frequency_analysis_limits \
                        --dir-path "$data_dir" \
                        --name "$name" \
                        --t 3 \
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

# Do specific spitz run at the end.
$SG_RUN ./target/release/frequency_analysis_limits \
                        --dir-path "$SLURM_SUBMIT_DIR/databases/350x50" \
                        --name "spitz" \
                        --t 2 \
                        --dist "gaussian" \
                        --plot

$SG_RUN ./target/release/frequency_analysis_limits \
                        --dir-path "$SLURM_SUBMIT_DIR/databases/350x50" \
                        --name "spitz" \
                        --t 2 \
                        --dist "uniform" \
                        --plot

$SG_RUN ./target/release/frequency_analysis_limits \
                        --dir-path "$SLURM_SUBMIT_DIR/databases/350x50" \
                        --name "spitz" \
                        --t 2 \
                        --dist "beta" \
                        --plot

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
