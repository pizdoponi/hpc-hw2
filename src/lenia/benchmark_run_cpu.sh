#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

steps=100
sizes=(256 512 1024 2048 4096)
out_root="${1:-benchmarks_cpu}"

ts=$(date +%Y%m%d_%H%M%S)
outdir="$out_root/$ts"
mkdir -p "$outdir/final_states"

raw_csv="$outdir/raw_times.csv"
avg_csv="$outdir/avg_times.csv"

printf "size,device,run,t_total\n" > "$raw_csv"

run_one() {
    local size=$1
    local run_id=$2
    local output_path=""

    if [[ "$run_id" -eq 1 ]]; then
        output_path="$outdir/final_states/final_cpu_${size}.pgm"
    fi

    local cmd_output
    if [[ -n "$output_path" ]]; then
        cmd_output=$(./lenia.out cpu "$size" "$steps" "$output_path")
    else
        cmd_output=$(./lenia.out cpu "$size" "$steps")
    fi

    local total
    total=$(printf "%s\n" "$cmd_output" | awk -F= '/^T_TOTAL=/{print $2}')

    if [[ -z "$total" ]]; then
        echo "Failed to parse T_TOTAL for cpu size=$size run=$run_id" >&2
        echo "$cmd_output" >&2
        exit 1
    fi

    printf "%s,CPU,%s,%s\n" "$size" "$run_id" "$total" >> "$raw_csv"
    printf "[CPU] size=%s run=%s T_TOTAL=%s\n" "$size" "$run_id" "$total"
}

for size in "${sizes[@]}"; do
    repeats=5
    if [[ "$size" -eq 4096 ]]; then
        repeats=1
    fi

    for ((r=1; r<=repeats; r++)); do
        run_one "$size" "$r"
    done
done

printf "size,device,t_avg\n" > "$avg_csv"
awk -F, 'NR>1 { k=$1","$2; sum[k]+=$4; cnt[k]++ } END { for (k in sum) printf "%s,%.6f\n", k, sum[k]/cnt[k] }' "$raw_csv" | sort -t, -k1,1n >> "$avg_csv"

echo "CPU benchmark complete."
echo "Raw data:    $raw_csv"
echo "Averages:    $avg_csv"
echo "Final states: $outdir/final_states"
