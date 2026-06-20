# Optimizing a CUDA 2D Convolution Kernel

Benchmarks four implementations of 2D image convolution on the GPU:

| Kernel | Description |
| --- | --- |
| `naive` | Each thread reads directly from global memory |
| `tiled` | Threads cooperatively load input tiles into shared memory |
| `const_mem` | Tiled input + filter stored in constant memory (broadcast cache) |
| `cudnn` | cuDNN baseline (NVIDIA's tuned library) |

Results are swept across image sizes (256–4096) and filter sizes (3×3–11×11).

---

## Requirements

- CUDA 12.x (`nvcc`)
- Python 3.11+
- GPU with compute capability ≥ 8.0

Install Python dependencies:

```bash
uv sync
```

---

## Build

```bash
make clean && make
```

Override the compute architecture if needed (default `sm_89` for RTX 40-series):

```bash
make clean && make CUDA_ARCH=sm_86
```

---

## Run

**Full benchmark sweep (all kernels × all image sizes × all filter sizes):**

```bash
python3 main.py
```

**Benchmark + generate plots:**

```bash
python3 main.py --plot
```

**Re-generate plots from existing results without re-running benchmarks:**

```bash
python3 main.py --plot-only
```

Results are written to `results/benchmark.log`. Plots are saved as PNG and PDF in `results/plots/`.

---

## Project Structure

```text
├── src/
│   ├── naive_conv.cu        # Naive global memory kernel
│   ├── tiled_conv.cu        # Tiled shared memory kernel
│   ├── const_mem_conv.cu    # Constant memory kernel
│   ├── cudnn_conv.cpp       # cuDNN wrapper
│   ├── cpu_reference.cpp    # CPU reference (correctness baseline)
│   ├── benchmark.cpp        # Timing harness for all kernels
│   ├── tests.cpp            # Correctness checks vs CPU reference
│   ├── gpu_buffers.cpp      # Device memory allocation helpers
│   └── main.cpp             # Entry point and CLI
├── include/
│   ├── conv.h               # Shared types and function declarations
│   ├── utils.h              # CudaTimer, CHECK_CUDA, file I/O, printing
│   ├── gpu_buffers.h
│   └── tests.h
├── scripts/
│   ├── generate_benchmark_data.py   # Generates PGM images and filter files
│   └── plot_results.py              # Produces plots from benchmark.log
├── data/
│   ├── images/              # Generated PGM test images
│   └── filters/             # Generated Gaussian filter files
├── results/                 # Created at runtime (gitignored)
│   ├── benchmark.log
│   └── plots/
├── main.py                  # Benchmark runner (replaces shell script)
└── Makefile
```

---

## Plots

`plot_results.py` produces five plots from `results/benchmark.log`:

1. **Execution time vs image size** — fixed 5×5 filter, one line per kernel
2. **Execution time vs filter size** — fixed 1024×1024 image, one line per kernel
3. **Speedup over naive** — grouped bar chart across image sizes
4. **Memory bandwidth** — effective GB/s at 1024×1024, 5×5 filter
5. **GFLOPS** — achieved throughput vs image size

---

## Hardware

Benchmarks run on:

- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB, compute capability 8.9)
- **CUDA:** 12.9
- **Driver:** 591.86
