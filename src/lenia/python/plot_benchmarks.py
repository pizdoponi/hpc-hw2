#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_avg_times(csv_path: Path):
    sizes = []
    times = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sizes.append(float(row["size"]))
            times.append(float(row["t_avg"]))
    return np.array(sizes), np.array(times)


def linear_fit(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x + intercept
    return (slope, intercept), y_fit


def power_law_fit(x, y):
    # Fit y = a * x^b by linear regression in log-log space.
    log_x = np.log(x)
    log_y = np.log(y)
    b, log_a = np.polyfit(log_x, log_y, 1)
    a = np.exp(log_a)
    y_fit = a * np.power(x, b)
    return (a, b), y_fit


def r_squared(y, y_fit):
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot


def main():
    parser = argparse.ArgumentParser(
        description="Plot Lenia CPU/GPU benchmark times and fitted curves."
    )
    parser.add_argument(
        "--cpu-csv",
        type=Path,
        default=Path("src/lenia/results_cpu/20260412_155548/avg_times.csv"),
        help="Path to CPU avg_times.csv",
    )
    parser.add_argument(
        "--gpu-csv",
        type=Path,
        default=Path("src/lenia/results_gpu/20260412_164740/avg_times.csv"),
        help="Path to GPU avg_times.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/lenia/python/benchmark_fits.png"),
        help="Output image path",
    )
    args = parser.parse_args()

    cpu_sizes, cpu_times = read_avg_times(args.cpu_csv)
    gpu_sizes, gpu_times = read_avg_times(args.gpu_csv)

    cpu_pixels = cpu_sizes ** 2
    gpu_pixels = gpu_sizes ** 2

    (gpu_slope, gpu_intercept), gpu_fit = linear_fit(gpu_pixels, gpu_times)
    (cpu_a, cpu_b), cpu_fit = power_law_fit(cpu_pixels, cpu_times)

    gpu_r2 = r_squared(gpu_times, gpu_fit)
    cpu_r2 = r_squared(cpu_times, cpu_fit)

    x_dense_cpu = np.linspace(cpu_pixels.min(), cpu_pixels.max(), 300)
    x_dense_gpu = np.linspace(gpu_pixels.min(), gpu_pixels.max(), 300)

    cpu_curve = cpu_a * np.power(x_dense_cpu, cpu_b)
    gpu_curve = gpu_slope * x_dense_gpu + gpu_intercept

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    axes[0].scatter(cpu_pixels, cpu_times, color="tab:blue", label="Measured CPU")
    axes[0].plot(
        x_dense_cpu,
        cpu_curve,
        color="tab:red",
        label=f"Fit: t = {cpu_a:.3e} * pixels^{cpu_b:.3f}\n$R^2$ = {cpu_r2:.5f}",
    )
    axes[0].set_title("CPU Time vs Pixel Count")
    axes[0].set_xlabel("Pixels (size^2)")
    axes[0].set_ylabel("Time (s)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].scatter(gpu_pixels, gpu_times, color="tab:green", label="Measured GPU")
    axes[1].plot(
        x_dense_gpu,
        gpu_curve,
        color="tab:red",
        label=f"Fit: t = {gpu_slope:.3e} * pixels + {gpu_intercept:.3e}\n$R^2$ = {gpu_r2:.5f}",
    )
    axes[1].set_title("GPU Time vs Pixel Count")
    axes[1].set_xlabel("Pixels (size^2)")
    axes[1].set_ylabel("Time (s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)

    print(f"Saved plot to {args.output}")
    print(f"CPU fit: t = {cpu_a:.6e} * pixels^{cpu_b:.6f}  (R^2 = {cpu_r2:.6f})")
    print(f"GPU fit: t = {gpu_slope:.6e} * pixels + {gpu_intercept:.6e}  (R^2 = {gpu_r2:.6f})")


if __name__ == "__main__":
    main()
