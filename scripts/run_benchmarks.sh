#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

RESULTS_FILE="results/benchmark.log"

IMAGE_SIZES=(256 512 1024 2048 4096)
FILTER_RADII=(1 2 3 4 5)

# Generate benchmark data files if not already present.
echo "Preparing benchmark data..."
python3 scripts/generate_benchmark_data.py

mkdir -p results
rm -f "$RESULTS_FILE"

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
        echo "Running all kernels: ${size}x${size}, filter ${filter_size}x${filter_size}, runs=${runs}"

        ./conv \
            --image  "$image"  \
            --filter "$filter" \
            --runs   "$runs"   \
            | tee -a "$RESULTS_FILE"
    done
done

echo
echo "Benchmarks completed."
echo "Results saved to: $RESULTS_FILE"
echo
echo "To generate plots:"
echo "  python3 scripts/plot_results.py $RESULTS_FILE"
