#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

cpu_root="${1:-benchmarks_cpu}"
gpu_root="${2:-benchmarks_gpu}"
out_root="${3:-benchmarks_combined}"

cpu_dir=$(ls -1dt "$cpu_root"/* 2>/dev/null | head -n 1 || true)
gpu_dir=$(ls -1dt "$gpu_root"/* 2>/dev/null | head -n 1 || true)

if [[ -z "$cpu_dir" || -z "$gpu_dir" ]]; then
    echo "Error: could not find benchmark folders." >&2
    echo "CPU root: $cpu_root" >&2
    echo "GPU root: $gpu_root" >&2
    exit 1
fi

cpu_avg="$cpu_dir/avg_times.csv"
gpu_avg="$gpu_dir/avg_times.csv"

if [[ ! -f "$cpu_avg" ]]; then
    echo "Error: missing $cpu_avg" >&2
    exit 1
fi

if [[ ! -f "$gpu_avg" ]]; then
    echo "Error: missing $gpu_avg" >&2
    exit 1
fi

ts=$(date +%Y%m%d_%H%M%S)
outdir="$out_root/$ts"
mkdir -p "$outdir"

combined_csv="$outdir/combined_avg.csv"
speedup_csv="$outdir/speedup.csv"

printf "size,t_cpu_avg,t_gpu_avg\n" > "$combined_csv"
awk -F, 'FNR>1 {
    if (ARGIND == 1 && $2 == "CPU") cpu[$1] = $3;
    if (ARGIND == 2 && $2 == "GPU") gpu[$1] = $3;
}
END {
    for (s in cpu) {
        if (gpu[s] != "") {
            printf "%s,%.6f,%.6f\n", s, cpu[s], gpu[s];
        }
    }
}' "$cpu_avg" "$gpu_avg" | sort -t, -k1,1n >> "$combined_csv"

printf "size,t_cpu_avg,t_gpu_avg,speedup_cpu_over_gpu\n" > "$speedup_csv"
awk -F, 'NR>1 { printf "%s,%.6f,%.6f,%.6f\n", $1, $2, $3, $2/$3 }' "$combined_csv" >> "$speedup_csv"

cp "$cpu_avg" "$outdir/cpu_avg_times.csv"
cp "$gpu_avg" "$outdir/gpu_avg_times.csv"

cat > "$outdir/inputs.txt" <<EOF
cpu_avg=$cpu_avg
gpu_avg=$gpu_avg
EOF

echo "Combined results created."
echo "CPU source:  $cpu_avg"
echo "GPU source:  $gpu_avg"
echo "Combined:    $combined_csv"
echo "Speedup:     $speedup_csv"
