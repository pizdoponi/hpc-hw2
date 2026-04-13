#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=lenia_cpu_bench
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --output=lenia_cpu_benchmark.log

set -euo pipefail

module load CUDA

workdir="${SLURM_SUBMIT_DIR:-}"
if [[ -z "$workdir" ]]; then
    workdir="$(cd "$(dirname "$0")" && pwd)"
fi

cd "$workdir"

if [[ ! -f Makefile || ! -f benchmark_run_cpu.sh ]]; then
    echo "Error: run this script from src/lenia (Makefile and benchmark_run_cpu.sh must be present)." >&2
    echo "Current directory: $PWD" >&2
    exit 2
fi

make clean
make

results_root="${1:-benchmarks_cpu}"
bash benchmark_run_cpu.sh "$results_root"
