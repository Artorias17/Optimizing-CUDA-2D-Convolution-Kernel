#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

OUTPUT_FILE="results/benchmark.csv"

IMAGE_SIZES=(256 512 1024 2048 4096)
FILTER_RADII=(1 2 3 4 5)
KERNELS=(naive cudnn)

# Generate benchmark data files if not already present.
echo "Preparing benchmark data..."
python3 scripts/generate_benchmark_data.py

mkdir -p results
rm -f "$OUTPUT_FILE"

for kernel in "${KERNELS[@]}"; do
    for size in "${IMAGE_SIZES[@]}"; do
        if (( size <= 512 )); then
            runs=100
        elif (( size <= 2048 )); then
            runs=50
        else
            runs=20
        fi

        image="data/images/test_${size}x${size}.pgm"

        for radius in "${FILTER_RADII[@]}"; do
            filter_size=$(( 2 * radius + 1 ))
            filter="data/filters/gaussian_${filter_size}x${filter_size}.txt"

            echo
            echo "Running ${kernel}: ${size}x${size}, filter ${filter_size}x${filter_size}, runs=${runs}"

            ./conv \
                --kernel "$kernel" \
                --image  "$image"  \
                --filter "$filter" \
                --runs   "$runs"   \
                --output "$OUTPUT_FILE"
        done
    done
done

echo
echo "Benchmarks completed."
echo "Results saved to: $OUTPUT_FILE"
