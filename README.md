# Leakage Abuse via Matching

This repository contains the source code for running frequency analysis limit experiments. The project is written in
Rust and requires manual environment setup before building and executing the binaries.

### Prerequisites

To build and run this project, you must have the following development tools and libraries installed on your system:

- Python: A standard Python 3 installation. To specify a specific python environment, set the PYTHON_EXEC variable.

- OR-Tools (v9.15): You must install version 9.15 of the ortools Python package (python -m pip install
  ortools==9.15.6755). Depending on your OS and environment, you may also need the OR-Tools C++ binary release available
  on your system path for Rust to link against during compilation.

- Rust & Cargo: Install the latest stable toolchain via rustup.

### Usage

The primary executable is built to the ./target/release/ directory.
Command Structure

Execute the binary using the following params:

```bash
./target/release/frequency_analysis_limits \
--dir-path <path_to_database_directory> \
--name <dataset_name> \
--t <t_value> \
--dist <dist_type> \
--padding <padding_value> \
--query-percent <query_percentage_as_decimal> \
[--dim <dimension_value>]
```

Parameters:

- <path_to_database_directory>: Relative or absolute path to the database directory.

- <dataset_name>: Target dataset identifier (Name of folder in database dir).

- <t_value>: For how high a t-value should we run this.

- <dist_type>: Distribution configuration (uniform, gaussian, beta, flat).

- <padding_value>: Padding to add - Helps demonstrate hot 'flat' works.

- <query_percentage_as_decimal>: Float representing the percentage of queries to execute (e.g., 0.20 for 20%).

- <dimension_value>: Optional integer required for multi-dim grid

### Basic Example:

To run the cali dataset located in the databases/20x20 directory, using a Gaussian distribution, a 20% query rate, a
threshold of 2, and padding enabled:

```bash
export PYTHON_EXEC="/home/.conda/envs/main/bin/python3"


./target/release/frequency_analysis_limits \
--dir-path "databases/20x20" \
--name "cali" \
--t "3" \
--dist "gaussian" \
--padding "1" \
--query-percent "0.20"
```