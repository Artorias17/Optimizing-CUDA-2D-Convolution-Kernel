#!/usr/bin/env python3
"""Run the full benchmark sweep and optionally generate plots.

Usage:
    python3 main.py                  # benchmark only
    python3 main.py --plot           # benchmark + generate plots
    python3 main.py --plot-only      # skip benchmark, plot existing results
"""

import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

RESULTS_FILE = os.path.join(PROJECT_ROOT, "results", "benchmark.log")
BINARY      = os.path.join(PROJECT_ROOT, "conv")

IMAGE_SIZES  = [256, 512, 1024, 2048, 4096]
FILTER_RADII = [1, 2, 3, 4, 5]


def runs_for_size(size: int) -> int:
    if size <= 512:
        return 100
    if size <= 2048:
        return 50
    return 20


def generate_data() -> None:
    print("Preparing benchmark data...")
    subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "scripts", "generate_benchmark_data.py")],
        check=True,
    )


def run_benchmarks() -> None:
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    open(RESULTS_FILE, "w").close()  # truncate

    for size in IMAGE_SIZES:
        runs  = runs_for_size(size)
        image = os.path.join(PROJECT_ROOT, "data", "images", f"test_{size}x{size}.pgm")

        for radius in FILTER_RADII:
            filter_width = 2 * radius + 1
            filt = os.path.join(PROJECT_ROOT, "data", "filters", f"gaussian_{filter_width}x{filter_width}.txt")

            print(f"\nRunning all kernels: {size}x{size}, filter {filter_width}x{filter_width}, runs={runs}")

            result = subprocess.run(
                [BINARY, "--image", image, "--filter", filt, "--runs", str(runs)],
                capture_output=False,
                text=True,
                stdout=subprocess.PIPE,
            )

            print(result.stdout, end="")

            with open(RESULTS_FILE, "a") as f:
                f.write(result.stdout)

            if result.returncode != 0:
                print(result.stderr, end="", file=sys.stderr)
                sys.exit(result.returncode)

    print(f"\nBenchmarks completed. Results saved to: {RESULTS_FILE}")


def generate_plots() -> None:
    print(f"\nGenerating plots from {RESULTS_FILE} ...")
    subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "scripts", "plot_results.py"), RESULTS_FILE],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CUDA convolution benchmark runner")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--plot",      action="store_true", help="run benchmarks then generate plots")
    group.add_argument("--plot-only", action="store_true", help="generate plots from existing results")
    args = parser.parse_args()

    if args.plot_only:
        generate_plots()
        return

    generate_data()
    run_benchmarks()

    if args.plot:
        generate_plots()


if __name__ == "__main__":
    main()
