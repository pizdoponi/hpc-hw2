#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lenia_mem_mode
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --output=lenia_memory_mode.log

set -euo pipefail

module load CUDA

workdir="${SLURM_SUBMIT_DIR:-}"
if [[ -z "$workdir" ]]; then
    workdir="$(cd "$(dirname "$0")" && pwd)"
fi

cd "$workdir"

if [[ ! -f Makefile || ! -f benchmark_memory_mode.sh ]]; then
    echo "Error: run this script from src/lenia (Makefile and benchmark_memory_mode.sh must be present)." >&2
    echo "Current directory: $PWD" >&2
    exit 2
fi

make clean
make

results_root="${1:-benchmarks_memory_mode}"
bash benchmark_memory_mode.sh "$results_root"
