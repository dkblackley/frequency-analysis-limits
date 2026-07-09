# Leakage Abuse via Matching (LAMa)

This repository contains the source code for running frequency analysis limit experiments with LAMa. The project is
written in
Rust and requires manual environment setup before building and executing the binaries.

Specifically this artifact includes:

- The full source code for LAMa
- The Cali, Manhattan, Paris, Shanghai and Amsterdam datasets scaled to 20x20
- A singularity container file to allow for a consistent execution environment
- An integration test that can be run on smaller machines

Due to the complexity of LAMa it may not be possible to run LAMa unless certain hardware requirements are satisfied.
This README will specify the full environment, build instructions, and the workflow for generating
the results in our paper.

### Hardware Requirements

LAMa was compiled and ran on an AMD Zen 2 CPU with 64 cores and 512GB of RAM. We increased the RAM size to 3TB for
larger datasets.
LAMa was compiled and ran on Debian 12 Linux. A singularity file rust_env.dev has been provided and can be run as
follows:

```bash
sudo singularity build rust_env.sif rust_env.dev
```

This container then contains all the libraries necessary to build and run LAMa on linux machines.

### Prerequisites

To build and run this project, you must have the following development tools and libraries installed on your system:

- Python: A standard Python 3 installation. To specify a specific python environment, set the PYTHON_EXEC variable.

- OR-Tools (v9.15): You must install version 9.15 of the ortools Python package (python -m pip install
  ortools==9.15.6755). Depending on your OS and environment, you may also need the OR-Tools C++ binary release available
  on your system path for Rust to link against during compilation.

All libraries required to build are specified in the rust_env.dev container. Once compiled, Rust should statically link
all libraries and produce a single binary at ./target/release/frequency_analysis_limits

To run plotting code, results from [REMIN](https://github.com/ZIMUQIN-L/REMIN-attack) mush be dumped and transferred to
a database folder with the format: "<root_dir>/<db_name>/<method_name>/" where root_name is the directory holding all
the db files, db_name is cali/shopparis/etc. and method name is: limits, remin, reminp or even_less.

### Determinism and replicability

All results are designed to be exactly replicable and random engines were seeded were possible. LAMa is also defined to
utilise as many cores as possible, with substantial focus on multithreading for solving hard combinatorial problems.
Unfortunately, libraries like ORTOOLS (of which are usually fully deterministic) may rely on the order that constraints
are added to begin searching for a solution. As a result and specifically in the case where query probabilities are
provided, results may see some variance. In non probabilistic settings, ortools should consistently generate the
solution.

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

- <padding_value>: Padding to add - Helps demonstrate how 'flat' works.

- <query_percentage_as_decimal>: Float representing the percentage of queries to perform (e.g., 0.20 for 20%).

- <dimension_value>: Optional integer required for multi-dim grid

### Basic Example:

Note: LAMa requires a substantial amount of RAM to work. In the non-approximate setting we would recommend 100 GB of RAM
for a 20x20 across all datasets. We have included 5 databases in our code. If you want to run the approximate/flat
setting, you would require roughly 3TB of RAM. For most distributions t=2 is enough. For uniform or flat, up to 2 times
the number of dimensions may
help.

To run the cali dataset located in the databases/20x20 directory, using a Uniform distribution, a 20% query rate and a t
val
of 3:

```bash
export PYTHON_EXEC="/home/.conda/envs/main/bin/python3"


./target/release/frequency_analysis_limits \
--dir-path "databases/20x20" \
--name "cali" \
--t "3" \
--dist "uniform" \
--query-percent "0.20"
```