#!/usr/bin/env python3
"""Generate benchmark plots from conv benchmark output.

Usage:
    python3 scripts/plot_results.py results/benchmark.log [results/plots]

Reads lines tagged "RESULT ..." written by print_benchmark_result() and
produces five plots saved as both PNG and PDF.
"""

import os
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

COLORS = {
    "naive": "steelblue",
    "tiled": "darkorange",
    "const_mem": "forestgreen",
    "cudnn": "crimson",
}

LABELS = {
    "naive": "Naive",
    "tiled": "Tiled Shared Mem",
    "const_mem": "Const Memory",
    "cudnn": "cuDNN",
}

KERNEL_ORDER = ["naive", "tiled", "const_mem", "cudnn"]


_RESULT_RE = re.compile(
    r"RESULT"
    r" kernel=(?P<kernel>\w+)"
    r" H=(?P<H>\d+)"
    r" W=(?P<W>\d+)"
    r" filter_radius=(?P<filter_radius>\d+)"
    r" filter_width=(?P<filter_width>\d+)"
    r" runs=(?P<runs>\d+)"
    r" mean_ms=(?P<mean_ms>[\d.]+)"
    r" stddev_ms=(?P<stddev_ms>[\d.]+)"
    r" gflops=(?P<gflops>[\d.]+)"
    r" bw_gbs=(?P<bw_gbs>[\d.]+)"
)


def parse_results(path: str) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            m = _RESULT_RE.search(line)
            if not m:
                continue
            records.append({
                "kernel":        m.group("kernel"),
                "H":             int(m.group("H")),
                "W":             int(m.group("W")),
                "filter_radius": int(m.group("filter_radius")),
                "filter_width":  int(m.group("filter_width")),
                "timed_runs":    int(m.group("runs")),
                "mean_ms":       float(m.group("mean_ms")),
                "stddev_ms":     float(m.group("stddev_ms")),
                "gflops":        float(m.group("gflops")),
                "bandwidth_gbs": float(m.group("bw_gbs")),
            })
    return pd.DataFrame(records)


def save_plot(fig: plt.Figure, name: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"{name}.{ext}"), bbox_inches="tight", dpi=150)
    plt.close(fig)


# Plot 1 — Execution time vs image size (fixed filter).
def plot_time_vs_image_size(df: pd.DataFrame, output_dir: str, filter_radius: int = 2) -> None:
    sub = df[df["filter_radius"] == filter_radius]
    fw = 2 * filter_radius + 1

    fig, ax = plt.subplots(figsize=(8, 5))
    for k in KERNEL_ORDER:
        kdf = sub[sub["kernel"] == k].sort_values("H")
        if kdf.empty:
            continue
        ax.errorbar(
            kdf["H"], kdf["mean_ms"], yerr=kdf["stddev_ms"],
            label=LABELS[k], color=COLORS[k], marker="o", linewidth=2, capsize=3,
        )

    ax.set_xlabel("Image size (pixels, square)")
    ax.set_ylabel("Execution time (ms)")
    ax.set_title(f"Execution Time vs Image Size  (filter {fw}×{fw})")
    ax.set_xticks(sorted(sub["H"].unique()))
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "plot1_time_vs_image_size", output_dir)
    print("  Saved plot1_time_vs_image_size")


# Plot 2 — Execution time vs filter size (fixed image).
def plot_time_vs_filter_size(df: pd.DataFrame, output_dir: str, image_size: int = 1024) -> None:
    sub = df[df["H"] == image_size]

    fig, ax = plt.subplots(figsize=(8, 5))
    for k in KERNEL_ORDER:
        kdf = sub[sub["kernel"] == k].sort_values("filter_width")
        if kdf.empty:
            continue
        ax.errorbar(
            kdf["filter_width"], kdf["mean_ms"], yerr=kdf["stddev_ms"],
            label=LABELS[k], color=COLORS[k], marker="o", linewidth=2, capsize=3,
        )

    ax.set_xlabel("Filter size")
    ax.set_ylabel("Execution time (ms)")
    ax.set_title(f"Execution Time vs Filter Size  ({image_size}×{image_size} image)")
    ax.set_xticks(sorted(sub["filter_width"].unique()))
    ax.set_xticklabels([f"{fw}×{fw}" for fw in sorted(sub["filter_width"].unique())])
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "plot2_time_vs_filter_size", output_dir)
    print("  Saved plot2_time_vs_filter_size")


# Plot 3 — Speedup over naive (grouped bar chart).
def plot_speedup(df: pd.DataFrame, output_dir: str, filter_radius: int = 2) -> None:
    sub = df[df["filter_radius"] == filter_radius]
    fw = 2 * filter_radius + 1
    sizes = sorted(sub["H"].unique())
    compare_kernels = ["tiled", "const_mem", "cudnn"]

    x = np.arange(len(sizes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, k in enumerate(compare_kernels):
        speedups = []
        for size in sizes:
            naive_t = sub[(sub["kernel"] == "naive") & (sub["H"] == size)]["mean_ms"]
            kernel_t = sub[(sub["kernel"] == k) & (sub["H"] == size)]["mean_ms"]
            if naive_t.empty or kernel_t.empty:
                speedups.append(0.0)
            else:
                speedups.append(float(naive_t.values[0]) / float(kernel_t.values[0]))
        ax.bar(x + i * width, speedups, width, label=LABELS[k], color=COLORS[k])

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, label="Naive baseline")
    ax.set_xlabel("Image size")
    ax.set_ylabel("Speedup over naive")
    ax.set_title(f"Speedup over Naive Kernel  (filter {fw}×{fw})")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{s}×{s}" for s in sizes])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    save_plot(fig, "plot3_speedup", output_dir)
    print("  Saved plot3_speedup")


# Plot 4 — Effective memory bandwidth (bar chart, single config).
def plot_bandwidth(
    df: pd.DataFrame, output_dir: str, image_size: int = 1024, filter_radius: int = 2
) -> None:
    sub = df[(df["H"] == image_size) & (df["filter_radius"] == filter_radius)]
    fw = 2 * filter_radius + 1

    bws = []
    labels = []
    colors = []
    for k in KERNEL_ORDER:
        row = sub[sub["kernel"] == k]
        if row.empty:
            continue
        bws.append(float(row["bandwidth_gbs"].values[0]))
        labels.append(LABELS[k])
        colors.append(COLORS[k])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, bws, color=colors)
    ax.set_ylabel("Effective bandwidth (GB/s)")
    ax.set_title(f"Memory Bandwidth  ({image_size}×{image_size}, filter {fw}×{fw})")
    ax.grid(True, axis="y", alpha=0.3)
    save_plot(fig, "plot4_bandwidth", output_dir)
    print("  Saved plot4_bandwidth")


# Plot 5 — GFLOPS vs image size (line chart).
def plot_gflops(df: pd.DataFrame, output_dir: str, filter_radius: int = 2) -> None:
    sub = df[df["filter_radius"] == filter_radius]
    fw = 2 * filter_radius + 1

    fig, ax = plt.subplots(figsize=(8, 5))
    for k in KERNEL_ORDER:
        kdf = sub[sub["kernel"] == k].sort_values("H")
        if kdf.empty:
            continue
        ax.plot(kdf["H"], kdf["gflops"], label=LABELS[k], color=COLORS[k], marker="o", linewidth=2)

    ax.set_xlabel("Image size (pixels, square)")
    ax.set_ylabel("Achieved GFLOPS")
    ax.set_title(f"GFLOPS vs Image Size  (filter {fw}×{fw})")
    ax.set_xticks(sorted(sub["H"].unique()))
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "plot5_gflops", output_dir)
    print("  Saved plot5_gflops")


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} RESULTS_FILE [OUTPUT_DIR]")
        sys.exit(1)

    results_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "results/plots"

    df = parse_results(results_file)
    if df.empty:
        print("No RESULT lines found in input file.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(df)} benchmark records.")
    print(f"  Kernels:      {sorted(df['kernel'].unique())}")
    print(f"  Image sizes:  {sorted(df['H'].unique())}")
    print(f"  Filter radii: {sorted(df['filter_radius'].unique())}")
    print(f"Writing plots to {output_dir}/")

    plot_time_vs_image_size(df, output_dir)
    plot_time_vs_filter_size(df, output_dir)
    plot_speedup(df, output_dir)
    plot_bandwidth(df, output_dir)
    plot_gflops(df, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
