#!/usr/bin/env python3
"""Generate PGM test images and Gaussian filter TXT files for benchmarking."""

import math
import os

IMAGES_DIR = "data/images"
FILTERS_DIR = "data/filters"

IMAGE_SIZES  = [256, 512, 1024, 2048, 4096]
FILTER_RADII = [1, 2, 3, 4, 5]


def write_pgm(path: str, width: int, height: int) -> None:
    maxval = 255
    pixels = [
        str(int((math.sin(math.sqrt((col - width // 2) ** 2 + (row - height // 2) ** 2) / 20.0) * 0.5 + 0.5) * maxval))
        for row in range(height)
        for col in range(width)
    ]
    with open(path, "w") as f:
        f.write(f"P2\n{width} {height}\n{maxval}\n")
        for i in range(0, len(pixels), 16):
            f.write(" ".join(pixels[i : i + 16]) + "\n")


def gaussian_kernel(radius: int) -> list[float]:
    size  = 2 * radius + 1
    sigma = size / 6.0
    vals  = [
        math.exp(-((col - radius) ** 2 + (row - radius) ** 2) / (2.0 * sigma * sigma))
        for row in range(size)
        for col in range(size)
    ]
    total = sum(vals)
    return [v / total for v in vals]


def write_filter(path: str, radius: int) -> None:
    size   = 2 * radius + 1
    values = gaussian_kernel(radius)
    with open(path, "w") as f:
        f.write(f"{size} {size}\n")
        for row in range(size):
            f.write(" ".join(f"{v:.6f}" for v in values[row * size : (row + 1) * size]) + "\n")


def main() -> None:
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(FILTERS_DIR, exist_ok=True)

    print("Generating test images...")
    for size in IMAGE_SIZES:
        path = os.path.join(IMAGES_DIR, f"test_{size}x{size}.pgm")
        if not os.path.exists(path):
            write_pgm(path, size, size)
            print(f"  Written {path}")

    print("Generating Gaussian filter files...")
    for radius in FILTER_RADII:
        size = 2 * radius + 1
        path = os.path.join(FILTERS_DIR, f"gaussian_{size}x{size}.txt")
        if not os.path.exists(path):
            write_filter(path, radius)
            print(f"  Written {path}")

    print("Done.")


if __name__ == "__main__":
    main()
