#!/usr/bin/env python3
"""Generate PGM test images and Gaussian filter TXT files for benchmarking."""

import math
import os
import sys

IMAGES_DIR = "data/images"
FILTERS_DIR = "data/filters"

IMAGE_SIZES = [256, 512, 1024, 2048, 4096]
FILTER_RADII = [1, 2, 3, 4, 5]


def write_pgm(path: str, width: int, height: int) -> None:
    if os.path.exists(path):
        return
    maxval = 255
    with open(path, "w") as f:
        f.write(f"P2\n{width} {height}\n{maxval}\n")
        pixels = []
        for row in range(height):
            for col in range(width):
                cx, cy = col - width // 2, row - height // 2
                dist = math.sqrt(cx * cx + cy * cy)
                val = int((math.sin(dist / 20.0) * 0.5 + 0.5) * maxval)
                pixels.append(str(val))
        for i in range(0, len(pixels), 16):
            f.write(" ".join(pixels[i : i + 16]) + "\n")
    print(f"  Written {path}")


def gaussian_kernel(radius: int) -> list[float]:
    size = 2 * radius + 1
    sigma = size / 6.0
    kernel = []
    total = 0.0
    for row in range(size):
        for col in range(size):
            x, y = col - radius, row - radius
            val = math.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
            kernel.append(val)
            total += val
    return [v / total for v in kernel]


def write_filter(path: str, radius: int) -> None:
    if os.path.exists(path):
        return
    size = 2 * radius + 1
    values = gaussian_kernel(radius)
    with open(path, "w") as f:
        f.write(f"{size} {size}\n")
        for row in range(size):
            row_vals = values[row * size : (row + 1) * size]
            f.write(" ".join(f"{v:.6f}" for v in row_vals) + "\n")
    print(f"  Written {path}")


def main() -> None:
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(FILTERS_DIR, exist_ok=True)

    print("Generating test images...")
    for size in IMAGE_SIZES:
        path = os.path.join(IMAGES_DIR, f"test_{size}x{size}.pgm")
        write_pgm(path, size, size)

    print("Generating Gaussian filter files...")
    for radius in FILTER_RADII:
        size = 2 * radius + 1
        path = os.path.join(FILTERS_DIR, f"gaussian_{size}x{size}.txt")
        write_filter(path, radius)

    print("Done.")


if __name__ == "__main__":
    main()
