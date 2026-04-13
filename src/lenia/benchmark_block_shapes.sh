#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

steps=100
sizes=(512 1024 2048 4096)
shapes=("16 16" "32 8" "8 32" "32 16")
out_root="${1:-benchmarks_block_shapes}"

ts=$(date +%Y%m%d_%H%M%S)
outdir="$out_root/$ts"
mkdir -p "$outdir/final_states"

raw_csv="$outdir/raw_times.csv"
avg_csv="$outdir/avg_times.csv"

printf "size,block_x,block_y,run,t_total\n" > "$raw_csv"

run_one() {
    local size=$1
    local block_x=$2
    local block_y=$3
    local run_id=$4
    local output_path=""

    if [[ "$run_id" -eq 1 ]]; then
        output_path="$outdir/final_states/final_gpu_${size}_${block_x}x${block_y}.pgm"
    fi

    local cmd_output
    if [[ -n "$output_path" ]]; then
        cmd_output=$(./lenia.out gpu "$size" "$steps" "$output_path" "$block_x" "$block_y")
    else
        cmd_output=$(./lenia.out gpu "$size" "$steps" "" "$block_x" "$block_y")
    fi

    local total
    total=$(printf "%s\n" "$cmd_output" | awk -F= '/^T_TOTAL=/{print $2}')

    if [[ -z "$total" ]]; then
        echo "Failed to parse T_TOTAL for size=$size block=${block_x}x${block_y} run=$run_id" >&2
        echo "$cmd_output" >&2
        exit 1
    fi

    printf "%s,%s,%s,%s,%s\n" "$size" "$block_x" "$block_y" "$run_id" "$total" >> "$raw_csv"
    printf "[GPU] size=%s block=%sx%s run=%s T_TOTAL=%s\n" "$size" "$block_x" "$block_y" "$run_id" "$total"
}

for size in "${sizes[@]}"; do
    repeats=5
    if [[ "$size" -eq 4096 ]]; then
        repeats=1
    fi

    for shape in "${shapes[@]}"; do
        read -r block_x block_y <<< "$shape"
        for ((r=1; r<=repeats; r++)); do
            run_one "$size" "$block_x" "$block_y" "$r"
        done
    done
done

printf "size,block_x,block_y,t_avg\n" > "$avg_csv"
awk -F, 'NR>1 { k=$1","$2","$3; sum[k]+=$5; cnt[k]++ } END { for (k in sum) printf "%s,%.6f\n", k, sum[k]/cnt[k] }' "$raw_csv" | sort -t, -k1,1n -k2,2n -k3,3n >> "$avg_csv"

echo "Block-shape benchmark complete."
echo "Raw data:     $raw_csv"
echo "Averages:     $avg_csv"
echo "Final states: $outdir/final_states"
