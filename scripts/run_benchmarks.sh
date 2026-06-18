#!/usr/bin/env bash

set -euo pipefail

OUTPUT_FILE="results/benchmark.csv"

IMAGE_SIZES=(256 512 1024 2048 4096)
FILTER_RADII=(1 2 3 4 5)

rm -f "$OUTPUT_FILE"

for size in "${IMAGE_SIZES[@]}"; do
    for radius in "${FILTER_RADII[@]}"; do
        if (( size <= 512 )); then
            runs=100
        elif (( size <= 2048 )); then
            runs=50
        else
            runs=20
        fi

        echo
        echo "Running naive: ${size}x${size}, radius=${radius}, runs=${runs}"

        ./conv \
            --kernel naive \
            --H "$size" \
            --W "$size" \
            --filter-radius "$radius" \
            --runs "$runs" \
            --output "$OUTPUT_FILE"
    done
done

echo
echo "Benchmarks completed."
echo "Results saved to: $OUTPUT_FILE"
