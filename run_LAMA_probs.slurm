#!/bin/bash -l
#SBATCH --job-name=LAMA_probs
#SBATCH --partition=bigmem
#SBATCH --output=prob_LAMA.out
#SBATCH --error=prob_LAMA.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=600G
#SBATCH --time=2-00:30:00

cd "$SLURM_SUBMIT_DIR"

module --quiet purge
module load hosts/hopper
module load singularity

cd "$SLURM_SUBMIT_DIR"
# ---------------------------------------

ORTOOLS_DIR="/scratch/dblackle/ortools"
echo "Setting up OR-Tools..."
rm -rf "$ORTOOLS_DIR"
mkdir -p "$ORTOOLS_DIR"
cd "$ORTOOLS_DIR"

wget -q https://github.com/google/or-tools/releases/download/v9.15/or-tools_amd64_debian-12_cpp_v9.15.6755.tar.gz
tar -xf or-tools_amd64_debian-12_cpp_v9.15.6755.tar.gz --strip-components=1
rm or-tools_amd64_debian-12_cpp_v9.15.6755.tar.gz

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

source /home/dblackle/miniconda3/etc/profile.d/conda.sh
conda activate main
export PYTHON_EXEC="/home/dblackle/.conda/envs/main/bin/python3.10"

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

padding=0

# NAMES=("spitz" "cali")
# DISTS=("uniform" "gaussian" "beta")
#DATA_DIRS=("$SLURM_SUBMIT_DIR/databases/350x50" "$SLURM_SUBMIT_DIR/databases/175x25")


 NAMES=("shopparis" "highway" "busstop" "drink" "spitz" "cali")
 # DISTS=("uniform" "gaussian" "beta" "flat")
# DISTS=("flat")
#DATA_DIRS=(
#     "$SLURM_SUBMIT_DIR/databases/20x20"
#     "$SLURM_SUBMIT_DIR/databases/25x25"
#     "$SLURM_SUBMIT_DIR/databases/30x30"
#     "$SLURM_SUBMIT_DIR/databases/35x35"
#     "$SLURM_SUBMIT_DIR/databases/40x40"
#     "$SLURM_SUBMIT_DIR/databases/45x45"
#     "$SLURM_SUBMIT_DIR/databases/50x50"
# )

#NAMES=("nh")
#DISTS=("uniform" "gaussian" "beta")
#DATA_DIRS=("$SLURM_SUBMIT_DIR/databases/16x16x14")


DISTS=("uniform" "gaussian" "beta")
DATA_DIRS=(
     "$SLURM_SUBMIT_DIR/databases/15x15")

PROBS=("0.35" "0.25" "0.15")
# PROBS=("1.0")

echo "Running canary"
$SG_RUN ./target/release/frequency_analysis_limits \
                --dir-path "$SLURM_SUBMIT_DIR/databases/20x20" \
                --name "spitz" \
                --t "2" \
                --dist "gaussian" \
                --padding "0"


echo "Starting batch processing..."

# Array to keep track of specific configurations that completely fail
FAILED_RUNS=()

# Loop through each data directory
for data_dir in "${DATA_DIRS[@]}"; do
    # Loop through each name
    for name in "${NAMES[@]}"; do
        # Loop through each distribution type
        for dist in "${DISTS[@]}"; do
          for prob in "${PROBS[@]}"; do
            echo "========================================"
            echo "Running configuration:"
            echo "  Data Dir: $data_dir"
            echo "  Name:     $name"
            echo "  Dist:     $dist"
            echo "========================================"

            # Determine initial 't' value based on distribution type
            if [ "$dist" == "flat" ]; then
                t_val=2
            else
                t_val=2
            fi


            # Initial run
            $SG_RUN ./target/release/frequency_analysis_limits \
                --dir-path "$data_dir" \
                --name "$name" \
                --t "$t_val" \
                --dist "$dist" \
                --padding "$padding" \
                --query-percent "$prob"

            # Check if the initial run failed
            if [ $? -ne 0 ]; then
                echo "Error: Initial run failed for $name (Dist: $dist, Dir: $data_dir). Retrying with t=4..."

                # Retry one more time with t=4
                $SG_RUN ./target/release/frequency_analysis_limits \
                        --dir-path "$data_dir" \
                        --name "albert" \
                        --t 2 \
                        --dist "$dist" \
                        --padding "$padding" \
                        --query-percent "$prob"

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


 # Define the main variables
 name="grid"
 DISTS=("uniform" "gaussian" "beta")

 # Initialize an array to track failures
 FAILED_RUNS=()

 # 1. Loop through dimensions 2, 3, 4, 5, and 6
 for dim in {2..6}; do

     # 2. Dynamically build the "10x10x..." string based on the dimension
     dim_str="5"
     for ((i=1; i<dim; i++)); do
         dim_str+="x5"
     done

     # 3. Set the directory path for this dimension
     data_dir="$SLURM_SUBMIT_DIR/databases/$dim_str"

     echo "================================================================"
     echo "Starting runs for Dimension: $dim"
     echo "Data Dir: $data_dir"
     echo "================================================================"
     echo ""

     # 4. Loop through each distribution type
     for dist in "${DISTS[@]}"; do
         echo "----------------------------------------"
         echo "Running configuration:"
         echo "  Name:     $name"
         echo "  Dist:     $dist"
         echo "----------------------------------------"

         # Determine initial 't' value based on distribution type
         if [ "$dist" == "uniform" ]; then
             t_val=3
         else
             t_val=2
         fi

         # Initial run
         $SG_RUN ./target/release/frequency_analysis_limits \
             --dir-path "$data_dir" \
             --name "$name" \
             --t "$t_val" \
             --dim "$dim" \
             --dist "$dist"

         # Check if the initial run failed ($? checks the exit code of the last command)
         if [ $? -ne 0 ]; then
             echo "Error: Initial run failed for $name (Dist: $dist, Dir: $data_dir). Retrying with t=4..."

             # If it fails, retry with t=4
             $SG_RUN ./target/release/frequency_analysis_limits \
                     --dir-path "$data_dir" \
                     --name "$name" \
                     --t 2 \
                     --dist "$dist" \
                     --dim "$dim"

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

 # 5. Print a clean summary of failures at the very end
 if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
     echo "========================================"
     echo "SUMMARY OF FAILED RUNS:"
     for failure in "${FAILED_RUNS[@]}"; do
         echo "- $failure"
     done
     echo "========================================"
 else
     echo "========================================"
     echo "ALL RUNS COMPLETED SUCCESSFULLY."
     echo "========================================"
 fi