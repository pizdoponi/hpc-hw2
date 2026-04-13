#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

steps=100
sizes=(256 512 1024 2048 4096)
modes=(global shared)
out_root="${1:-benchmarks_memory_mode}"
block_x=16
block_y=16

ts=$(date +%Y%m%d_%H%M%S)
outdir="$out_root/$ts"
mkdir -p "$outdir/final_states"

raw_csv="$outdir/raw_times.csv"
avg_csv="$outdir/avg_times.csv"
speedup_csv="$outdir/speedup.csv"

printf "size,memory_mode,run,t_total\n" > "$raw_csv"

run_one() {
    local size=$1
    local mode=$2
    local run_id=$3
    local output_path=""

    if [[ "$run_id" -eq 1 ]]; then
        output_path="$outdir/final_states/final_gpu_${mode}_${size}.pgm"
    fi

    local cmd_output
    if [[ -n "$output_path" ]]; then
        cmd_output=$(./lenia.out gpu "$size" "$steps" "$output_path" "$block_x" "$block_y" "$mode")
    else
        cmd_output=$(./lenia.out gpu "$size" "$steps" "" "$block_x" "$block_y" "$mode")
    fi

    local total
    total=$(printf "%s\n" "$cmd_output" | awk -F= '/^T_TOTAL=/{print $2}')

    if [[ -z "$total" ]]; then
        echo "Failed to parse T_TOTAL for mode=$mode size=$size run=$run_id" >&2
        echo "$cmd_output" >&2
        exit 1
    fi

    printf "%s,%s,%s,%s\n" "$size" "${mode^^}" "$run_id" "$total" >> "$raw_csv"
    printf "[GPU-%s] size=%s run=%s T_TOTAL=%s\n" "${mode^^}" "$size" "$run_id" "$total"
}

for size in "${sizes[@]}"; do
    repeats=5
    if [[ "$size" -eq 4096 ]]; then
        repeats=1
    fi

    for mode in "${modes[@]}"; do
        for ((r=1; r<=repeats; r++)); do
            run_one "$size" "$mode" "$r"
        done
    done
done

printf "size,memory_mode,t_avg\n" > "$avg_csv"
awk -F, 'NR>1 { k=$1","$2; sum[k]+=$4; cnt[k]++ } END { for (k in sum) printf "%s,%.6f\n", k, sum[k]/cnt[k] }' "$raw_csv" | sort -t, -k1,1n -k2,2 >> "$avg_csv"

printf "size,t_global_avg,t_shared_avg,speedup_shared_vs_global\n" > "$speedup_csv"
awk -F, 'NR>1 {
    if ($2 == "GLOBAL") global[$1] = $3;
    if ($2 == "SHARED") shared[$1] = $3;
}
END {
    for (s in global) {
        if (shared[s] != "") {
            printf "%s,%.6f,%.6f,%.6f\n", s, global[s], shared[s], global[s]/shared[s];
        }
    }
}' "$avg_csv" | sort -t, -k1,1n >> "$speedup_csv"

echo "Memory-mode benchmark complete."
echo "Raw data:     $raw_csv"
echo "Averages:     $avg_csv"
echo "Speedups:     $speedup_csv"
echo "Final states: $outdir/final_states"
