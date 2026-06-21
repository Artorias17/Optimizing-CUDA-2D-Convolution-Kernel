#!/usr/bin/env python3
"""Generate PowerPoint slides for the CUDA 2D convolution benchmark project.

Usage:
    python3 scripts/make_slides.py [output_path]

Produces a .pptx file covering each implementation and benchmark results.
"""

import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = ROOT / "results" / "plots"

PLOT1 = PLOTS_DIR / "plot1_time_vs_image_size.png"
PLOT2 = PLOTS_DIR / "plot2_time_vs_filter_size.png"
PLOT3 = PLOTS_DIR / "plot3_speedup.png"
PLOT4 = PLOTS_DIR / "plot4_bandwidth.png"
PLOT5 = PLOTS_DIR / "plot5_gflops.png"

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

DARK_BG = RGBColor(0x1A, 0x1A, 0x2E)
ACCENT = RGBColor(0x16, 0x21, 0x3E)
HIGHLIGHT = RGBColor(0x0F, 0x3D, 0x66)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xCC, 0xCC, 0xCC)
YELLOW = RGBColor(0xFF, 0xD7, 0x00)
GREEN = RGBColor(0x4C, 0xAF, 0x50)
ORANGE = RGBColor(0xFF, 0x98, 0x00)
BLUE = RGBColor(0x42, 0xA5, 0xF5)
RED = RGBColor(0xEF, 0x53, 0x50)

# Slide dimensions (16:9 widescreen)
W = Inches(13.33)
H_SLIDE = Inches(7.5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def new_prs() -> Presentation:
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H_SLIDE
    return prs


def blank_slide(prs: Presentation):
    blank_layout = prs.slide_layouts[6]  # completely blank
    return prs.slides.add_slide(blank_layout)


def fill_bg(slide, color: RGBColor) -> None:
    from pptx.oxml.ns import qn
    import lxml.etree as etree

    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(slide, left, top, width, height, fill_color, line_color=None):
    from pptx.util import Emu

    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        left,
        top,
        width,
        height,
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line_color
    return shape


def add_text(
    slide,
    text: str,
    left,
    top,
    width,
    height,
    font_size: int = 18,
    bold: bool = False,
    color: RGBColor = WHITE,
    align=PP_ALIGN.LEFT,
    wrap: bool = True,
) -> None:
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color


def add_bullet_box(
    slide,
    bullets: list[str],
    left,
    top,
    width,
    height,
    font_size: int = 16,
    color: RGBColor = WHITE,
    indent: bool = True,
) -> None:
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True

    for i, bullet in enumerate(bullets):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.space_before = Pt(4)
        run = p.add_run()
        prefix = "• " if indent else ""
        run.text = prefix + bullet
        run.font.size = Pt(font_size)
        run.font.color.rgb = color


def add_image(slide, path, left, top, width, height=None):
    if height is not None:
        slide.shapes.add_picture(str(path), left, top, width, height)
    else:
        slide.shapes.add_picture(str(path), left, top, width)


def add_code_block(
    slide, code: str, left, top, width, height, font_size: int = 11
) -> None:
    add_rect(slide, left, top, width, height, RGBColor(0x0D, 0x0D, 0x1A))
    txBox = slide.shapes.add_textbox(
        left + Inches(0.1),
        top + Inches(0.1),
        width - Inches(0.2),
        height - Inches(0.2),
    )
    tf = txBox.text_frame
    tf.word_wrap = False
    for i, line in enumerate(code.splitlines()):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        run = p.add_run()
        run.text = line
        run.font.size = Pt(font_size)
        run.font.color.rgb = RGBColor(0xA8, 0xD8, 0xEA)
        run.font.name = "Courier New"


def section_header(slide, title: str, color: RGBColor = HIGHLIGHT) -> None:
    add_rect(slide, Inches(0), Inches(0), W, Inches(1.0), color)
    add_text(
        slide,
        title,
        Inches(0.4),
        Inches(0.15),
        Inches(12),
        Inches(0.7),
        font_size=28,
        bold=True,
        color=WHITE,
        align=PP_ALIGN.LEFT,
    )


# ---------------------------------------------------------------------------
# Individual slides
# ---------------------------------------------------------------------------


def slide_title(prs: Presentation) -> None:
    slide = blank_slide(prs)
    fill_bg(slide, DARK_BG)

    # Accent bar
    add_rect(slide, Inches(0), Inches(0), Inches(0.25), H_SLIDE, YELLOW)

    # Title
    add_text(
        slide,
        "CUDA 2D Convolution",
        Inches(0.6),
        Inches(1.8),
        Inches(11.5),
        Inches(1.1),
        font_size=44,
        bold=True,
        color=WHITE,
        align=PP_ALIGN.LEFT,
    )
    add_text(
        slide,
        "Benchmarking Naive, Tiled, Constant Memory & cuDNN Kernels",
        Inches(0.6),
        Inches(2.9),
        Inches(11.5),
        Inches(0.8),
        font_size=24,
        color=YELLOW,
        align=PP_ALIGN.LEFT,
    )

    # Divider line
    add_rect(slide, Inches(0.6), Inches(3.8), Inches(10), Inches(0.03), LIGHT_GRAY)

    add_text(
        slide,
        "Hardware: RTX 4060 Laptop GPU  •  CUDA 12.9  •  Driver 591.86",
        Inches(0.6),
        Inches(4.0),
        Inches(11),
        Inches(0.5),
        font_size=16,
        color=LIGHT_GRAY,
        align=PP_ALIGN.LEFT,
    )
    add_text(
        slide,
        "Parallel Programming — Final Project",
        Inches(0.6),
        Inches(5.5),
        Inches(11),
        Inches(0.5),
        font_size=18,
        bold=True,
        color=LIGHT_GRAY,
        align=PP_ALIGN.LEFT,
    )


def slide_overview(prs: Presentation) -> None:
    slide = blank_slide(prs)
    fill_bg(slide, DARK_BG)
    section_header(slide, "Problem Overview")

    # Left column — what
    add_text(
        slide,
        "2D Convolution",
        Inches(0.4),
        Inches(1.2),
        Inches(5.5),
        Inches(0.5),
        font_size=20,
        bold=True,
        color=YELLOW,
    )
    add_bullet_box(
        slide,
        [
            "Core primitive in image processing & deep learning",
            "For each output pixel: multiply every filter element\n  by the corresponding input neighbor and sum",
            "Arithmetic intensity scales as O(F²) per pixel",
            "Entirely data-parallel — ideal for GPU acceleration",
        ],
        Inches(0.4),
        Inches(1.75),
        Inches(5.8),
        Inches(3.0),
        font_size=15,
    )

    # Formula box
    add_rect(slide, Inches(0.4), Inches(4.9), Inches(5.8), Inches(0.9), HIGHLIGHT)
    add_text(
        slide,
        "output[r,c] = Σ  input[r+i, c+j] · filter[i,j]",
        Inches(0.5),
        Inches(5.0),
        Inches(5.6),
        Inches(0.7),
        font_size=15,
        color=YELLOW,
        align=PP_ALIGN.CENTER,
    )

    # Right column — what we measure
    add_text(
        slide,
        "What We Measure",
        Inches(7.0),
        Inches(1.2),
        Inches(5.5),
        Inches(0.5),
        font_size=20,
        bold=True,
        color=YELLOW,
    )
    add_bullet_box(
        slide,
        [
            "Execution time (ms) — 20 timed runs, mean ± stddev",
            "Effective GFLOPS — arithmetic throughput",
            "Effective bandwidth (GB/s) — memory throughput",
            "Speedup vs naive baseline",
        ],
        Inches(7.0),
        Inches(1.75),
        Inches(5.8),
        Inches(2.5),
        font_size=15,
    )

    add_text(
        slide,
        "Test matrix",
        Inches(7.0),
        Inches(4.2),
        Inches(5.8),
        Inches(0.4),
        font_size=16,
        bold=True,
        color=LIGHT_GRAY,
    )
    add_bullet_box(
        slide,
        [
            "Image sizes: 256² · 512² · 1024² · 2048² · 4096²",
            "Filter sizes: 3×3 · 5×5 · 7×7 · 9×9 · 11×11 (Gaussian)",
            "Padding: zero-pad to preserve output dimensions",
        ],
        Inches(7.0),
        Inches(4.65),
        Inches(5.8),
        Inches(2.1),
        font_size=14,
        color=LIGHT_GRAY,
    )


def slide_naive(prs: Presentation) -> None:
    slide = blank_slide(prs)
    fill_bg(slide, DARK_BG)
    section_header(slide, "Implementation 1 — Naive Global Memory")

    add_bullet_box(
        slide,
        [
            "Block size: 16×16 threads, each computes one output pixel",
            "All reads go directly to global memory (DRAM)",
            "Filter elements re-read by every block independently — no sharing",
            "Simple, correct, and a fair starting baseline",
        ],
        Inches(0.4),
        Inches(1.15),
        Inches(5.5),
        Inches(2.2),
        font_size=15,
    )

    code = """\
__global__ void naive_conv_kernel(
    const float *d_input, const float *d_filter,
    float *d_output, int H, int W,
    int filter_radius, int pad_mode)
{
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    if (out_row >= H || out_col >= W) return;

    int fw = 2 * filter_radius + 1;
    float sum = 0.f;
    for (int fr = 0; fr < fw; ++fr)
        for (int fc = 0; fc < fw; ++fc) {
            int ir = out_row - filter_radius + fr;
            int ic = out_col - filter_radius + fc;
            // boundary check / clamp ...
            sum += d_input[ir*W+ic] * d_filter[fr*fw+fc];
        }
    d_output[out_row*W+out_col] = sum;
}"""
    add_code_block(slide, code, Inches(0.4), Inches(3.35), Inches(5.5), Inches(3.75), font_size=10)

    # Diagram on right
    add_text(
        slide,
        "Memory Access Pattern",
        Inches(6.5),
        Inches(1.15),
        Inches(6.4),
        Inches(0.4),
        font_size=18,
        bold=True,
        color=YELLOW,
    )
    add_bullet_box(
        slide,
        [
            "Each thread loads F² filter values from global DRAM",
            "Adjacent threads load overlapping input regions\n  → redundant DRAM traffic",
            "GPU L1/L2 cache may absorb some reuse",
            "Bottleneck: filter read bandwidth for large filters",
        ],
        Inches(6.5),
        Inches(1.65),
        Inches(6.4),
        Inches(2.5),
        font_size=14,
        color=LIGHT_GRAY,
    )

    add_rect(slide, Inches(6.5), Inches(4.2), Inches(6.4), Inches(2.9), ACCENT)
    labels = [
        ("Block 0", BLUE, Inches(6.9), Inches(4.45)),
        ("Block 1", ORANGE, Inches(9.1), Inches(4.45)),
        ("Overlapping filter reads", WHITE, Inches(7.5), Inches(5.5)),
    ]
    for label, color, lx, ly in labels:
        add_text(slide, label, lx, ly, Inches(2.5), Inches(0.4), font_size=13, color=color)

    # draw simple boxes
    add_rect(slide, Inches(6.9), Inches(4.85), Inches(2.0), Inches(1.5), BLUE)
    add_rect(slide, Inches(8.5), Inches(4.85), Inches(2.0), Inches(1.5), ORANGE)
    add_rect(slide, Inches(8.2), Inches(4.85), Inches(0.6), Inches(1.5), RGBColor(0x80, 0x40, 0x80))

    add_text(
        slide,
        "↑ Overlap",
        Inches(8.15),
        Inches(6.45),
        Inches(1.0),
        Inches(0.4),
        font_size=11,
        color=LIGHT_GRAY,
        align=PP_ALIGN.CENTER,
    )


def slide_tiled(prs: Presentation) -> None:
    slide = blank_slide(prs)
    fill_bg(slide, DARK_BG)
    section_header(slide, "Implementation 2 — Tiled Shared Memory")

    add_bullet_box(
        slide,
        [
            "Each block loads a shared-memory tile including a halo border",
            "Halo width = filter_radius on every side",
            "Block size = (TILE_WIDTH + 2R) × (TILE_WIDTH + 2R)  — larger than output",
            "All threads load input cooperatively, then inner threads convolve",
            "__syncthreads() ensures tile is fully populated before use",
        ],
        Inches(0.4),
        Inches(1.15),
        Inches(6.0),
        Inches(2.6),
        font_size=14,
    )

    code = """\
// Block covers output tile + halo
const int tile_dim = TILE_WIDTH + 2 * filter_radius;
dim3 block(tile_dim, tile_dim);

// Each thread loads one cell (possibly halo)
int in_row = blockIdx.y*TILE_WIDTH - R + threadIdx.y;
int in_col = blockIdx.x*TILE_WIDTH - R + threadIdx.x;
s_tile[threadIdx.y*tile_dim + threadIdx.x]
    = d_input[in_row*W + in_col];  // with boundary check

__syncthreads();  // wait for full tile

// Halo threads exit; inner threads compute
if (threadIdx.y >= TILE_WIDTH ||
    threadIdx.x >= TILE_WIDTH) return;

for (int fr = 0; fr < fw; ++fr)
  for (int fc = 0; fc < fw; ++fc)
    sum += s_tile[(threadIdx.y+fr)*tile_dim
                 + (threadIdx.x+fc)]
         * d_filter[fr*fw+fc];"""
    add_code_block(slide, code, Inches(0.4), Inches(3.85), Inches(5.6), Inches(3.3), font_size=10)

    # Right side — diagram
    add_text(
        slide,
        "Tile Layout (TILE_WIDTH=16, R=2)",
        Inches(6.5),
        Inches(1.15),
        Inches(6.4),
        Inches(0.4),
        font_size=17,
        bold=True,
        color=YELLOW,
    )

    # draw tile diagram
    tile_left = Inches(7.2)
    tile_top = Inches(1.75)
    tile_full = Inches(4.8)
    inner = Inches(3.4)
    halo_w = (tile_full - inner) / 2

    add_rect(slide, tile_left, tile_top, tile_full, tile_full, RGBColor(0x3A, 0x3A, 0x5C))
    add_rect(
        slide,
        tile_left + halo_w,
        tile_top + halo_w,
        inner,
        inner,
        HIGHLIGHT,
    )

    add_text(slide, "Halo\n(border)", tile_left + Inches(0.05), tile_top + Inches(0.05), Inches(1.0), Inches(0.7), font_size=11, color=ORANGE)
    add_text(slide, "Output\ntile\n(16×16)", tile_left + halo_w + Inches(0.7), tile_top + halo_w + Inches(0.8), Inches(1.5), Inches(0.9), font_size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    add_bullet_box(
        slide,
        [
            "Input reads: shared memory (on-chip, ~100× faster than DRAM)",
            "Filter reads: still from global memory",
            "Larger blocks reduce SM occupancy — fewer blocks in flight",
            "On RTX 4060 (large L1): tiled is slightly slower than naive",
        ],
        Inches(6.5),
        Inches(5.0),
        Inches(6.4),
        Inches(2.2),
        font_size=13,
        color=LIGHT_GRAY,
    )


def slide_const_mem(prs: Presentation) -> None:
    slide = blank_slide(prs)
    fill_bg(slide, DARK_BG)
    section_header(slide, "Implementation 3 — Constant Memory")

    add_bullet_box(
        slide,
        [
            "Extends tiled kernel: filter stored in __constant__ memory",
            "Constant memory: 64 KB, cached, optimized for uniform access",
            "All threads in a warp read the same filter[fr][fc] each iteration",
            "Hardware broadcasts the value in a single transaction (no bank conflicts)",
            "Filter uploaded with cudaMemcpyToSymbol before kernel launch",
        ],
        Inches(0.4),
        Inches(1.15),
        Inches(6.0),
        Inches(2.7),
        font_size=14,
    )

    code = """\
// Declared in global scope — resides in constant cache
__constant__ float d_const_filter[MAX_CONST_MEM_FILTER
                                  * MAX_CONST_MEM_FILTER];

void launch_const_mem_conv(...) {
    // Copy filter device→constant memory before launch
    cudaMemcpyToSymbol(d_const_filter, d_filter,
        fw*fw*sizeof(float), 0,
        cudaMemcpyDeviceToDevice);

    // Block/grid same as tiled
    const_mem_conv_kernel<<<grid, block, smem>>>(...);
}

// Inside kernel — filter read broadcasts to entire warp:
sum += s_tile[(threadIdx.y+fr)*tile_dim
             + (threadIdx.x+fc)]
     * d_const_filter[fr*fw+fc];  // ← constant cache"""
    add_code_block(slide, code, Inches(0.4), Inches(3.95), Inches(5.8), Inches(3.25), font_size=10)

    # Right column
    add_text(
        slide,
        "Why Constant Memory Wins",
        Inches(6.5),
        Inches(1.15),
        Inches(6.4),
        Inches(0.4),
        font_size=18,
        bold=True,
        color=YELLOW,
    )

    # Comparison diagram
    scenarios = [
        ("Global Memory", "32 threads × 1 DRAM read\n= 32 serial transactions", RED),
        ("Constant Memory", "32 threads × same address\n= 1 broadcast transaction", GREEN),
    ]
    for i, (title, desc, color) in enumerate(scenarios):
        bx = Inches(6.5)
        by = Inches(1.75 + i * 1.6)
        add_rect(slide, bx, by, Inches(6.4), Inches(1.35), ACCENT)
        add_text(slide, title, bx + Inches(0.15), by + Inches(0.1), Inches(3.0), Inches(0.4), font_size=14, bold=True, color=color)
        add_text(slide, desc, bx + Inches(0.15), by + Inches(0.5), Inches(6.1), Inches(0.75), font_size=13, color=WHITE)

    add_text(
        slide,
        "Performance note",
        Inches(6.5),
        Inches(5.1),
        Inches(6.4),
        Inches(0.35),
        font_size=15,
        bold=True,
        color=YELLOW,
    )
    add_bullet_box(
        slide,
        [
            "Consistently fastest kernel in this benchmark",
            "Best advantage for large filters (7×7 and above)",
            "Up to 1.4× speedup over naive at 4096×4096, 11×11",
            "Exceeds 1000 GFLOPS for 4096×4096 with 11×11 filter",
        ],
        Inches(6.5),
        Inches(5.5),
        Inches(6.4),
        Inches(1.7),
        font_size=13,
        color=LIGHT_GRAY,
    )


def slide_cudnn(prs: Presentation) -> None:
    slide = blank_slide(prs)
    fill_bg(slide, DARK_BG)
    section_header(slide, "Implementation 4 — cuDNN")

    add_bullet_box(
        slide,
        [
            "NVIDIA's production convolution library (cuDNN)",
            "Auto-selects the best algorithm for the given configuration",
            "Supports NCHW tensor format (batch × channels × H × W)",
            "Our benchmark: batch=1, channels=1 — a worst-case for cuDNN",
        ],
        Inches(0.4),
        Inches(1.15),
        Inches(6.0),
        Inches(2.2),
        font_size=14,
    )

    code = """\
// Context lifecycle around the timing loop
CudnnConvContext *ctx =
    create_cudnn_conv_context(d_input, d_filter,
                              d_output, params);

// Warmup
for (int i = 0; i < warmup_runs; ++i)
    run_cudnn_conv(ctx);
cudaDeviceSynchronize();

// Timed runs
for (int run = 0; run < timed_runs; ++run) {
    timer.start();
    run_cudnn_conv(ctx);
    times.push_back(timer.stop());
}

destroy_cudnn_conv_context(ctx);"""
    add_code_block(slide, code, Inches(0.4), Inches(3.5), Inches(5.8), Inches(3.75), font_size=10)

    # Right — why cuDNN is slower
    add_text(
        slide,
        "Why cuDNN Is Slower Here",
        Inches(6.5),
        Inches(1.15),
        Inches(6.4),
        Inches(0.4),
        font_size=18,
        bold=True,
        color=YELLOW,
    )
    add_bullet_box(
        slide,
        [
            "cuDNN is designed for deep learning workloads:",
            "  — Large batches (N ≫ 1)",
            "  — Many input/output channels (C ≫ 1)",
            "  — Repeated calls with fixed shapes",
            "",
            "Our test case (N=1, C=1) is its worst scenario:",
            "  — Algorithm selection overhead dominates",
            "  — Cannot amortize setup cost over batches",
            "  — Tensor descriptor overhead per call",
            "",
            "In real CNNs (e.g. 64 channels, batch 32),",
            "cuDNN would be significantly faster than custom kernels.",
        ],
        Inches(6.5),
        Inches(1.65),
        Inches(6.4),
        Inches(4.2),
        font_size=13,
        color=LIGHT_GRAY,
    )

    add_rect(slide, Inches(6.5), Inches(5.95), Inches(6.4), Inches(1.2), RGBColor(0x4A, 0x20, 0x20))
    add_text(
        slide,
        "cuDNN is a baseline, not a target — its strength is batch/channel parallelism, not single-image throughput.",
        Inches(6.6),
        Inches(6.05),
        Inches(6.2),
        Inches(1.0),
        font_size=13,
        color=RGBColor(0xFF, 0xAA, 0xAA),
        align=PP_ALIGN.LEFT,
    )


def slide_results_time_size(prs: Presentation) -> None:
    slide = blank_slide(prs)
    fill_bg(slide, DARK_BG)
    section_header(slide, "Results — Execution Time vs Image Size  (11×11 Gaussian filter)")

    add_image(slide, PLOT1, Inches(1.5), Inches(1.1), Inches(10.3))

    add_text(
        slide,
        "const_mem is fastest across all sizes  •  cuDNN overhead grows with image area  •  tiled consistently slower than naive",
        Inches(0.4),
        Inches(6.9),
        Inches(12.5),
        Inches(0.45),
        font_size=13,
        color=LIGHT_GRAY,
        align=PP_ALIGN.CENTER,
    )


def slide_results_time_filter(prs: Presentation) -> None:
    slide = blank_slide(prs)
    fill_bg(slide, DARK_BG)
    section_header(slide, "Results — Execution Time vs Filter Size  (4096×4096 image)")

    add_image(slide, PLOT2, Inches(1.5), Inches(1.1), Inches(10.3))

    add_text(
        slide,
        "cuDNN overhead spikes sharply from 7×7 onward  •  const_mem advantage grows with filter size",
        Inches(0.4),
        Inches(6.9),
        Inches(12.5),
        Inches(0.45),
        font_size=13,
        color=LIGHT_GRAY,
        align=PP_ALIGN.CENTER,
    )


def slide_results_speedup(prs: Presentation) -> None:
    slide = blank_slide(prs)
    fill_bg(slide, DARK_BG)
    section_header(slide, "Results — Speedup Over Naive  (11×11 Gaussian filter)")

    add_image(slide, PLOT3, Inches(1.5), Inches(1.1), Inches(10.3))

    add_text(
        slide,
        "const_mem clearly above 1× for all sizes  •  tiled is slower due to occupancy cost  •  cuDNN < 1× for most sizes",
        Inches(0.4),
        Inches(6.9),
        Inches(12.5),
        Inches(0.45),
        font_size=13,
        color=LIGHT_GRAY,
        align=PP_ALIGN.CENTER,
    )


def slide_results_bandwidth(prs: Presentation) -> None:
    slide = blank_slide(prs)
    fill_bg(slide, DARK_BG)
    section_header(slide, "Results — Memory Bandwidth & GFLOPS")

    add_image(slide, PLOT4, Inches(0.3), Inches(1.1), Inches(6.4))
    add_image(slide, PLOT5, Inches(6.9), Inches(1.1), Inches(6.1))

    add_text(
        slide,
        "Bandwidth (GB/s) — 4096×4096 image, 11×11 filter",
        Inches(0.3),
        Inches(6.5),
        Inches(6.4),
        Inches(0.4),
        font_size=12,
        color=LIGHT_GRAY,
        align=PP_ALIGN.CENTER,
    )
    add_text(
        slide,
        "GFLOPS vs image size — 11×11 filter",
        Inches(6.9),
        Inches(6.5),
        Inches(6.1),
        Inches(0.4),
        font_size=12,
        color=LIGHT_GRAY,
        align=PP_ALIGN.CENTER,
    )


def slide_key_numbers(prs: Presentation) -> None:
    slide = blank_slide(prs)
    fill_bg(slide, DARK_BG)
    section_header(slide, "Key Numbers at a Glance")

    rows = [
        ("Config", "Kernel", "Time (ms)", "GFLOPS", "BW (GB/s)", "vs Naive"),
        ("4096² · 11×11", "const_mem", "4.02", "1009", "4052", "1.44×"),
        ("4096² · 11×11", "naive", "5.80", "701", "2814", "1.00×"),
        ("4096² · 11×11", "tiled", "7.93", "512", "2057", "0.73×"),
        ("4096² · 11×11", "cuDNN", "5.56", "730", "2933", "1.04×"),
        ("", "", "", "", "", ""),
        ("2048² · 11×11", "const_mem", "1.32", "767", "3080", "1.10×"),
        ("2048² · 11×11", "naive", "1.46", "695", "2790", "1.00×"),
        ("2048² · 11×11", "tiled", "2.40", "423", "1697", "0.61×"),
        ("2048² · 11×11", "cuDNN", "1.78", "570", "2287", "0.82×"),
    ]

    col_widths = [Inches(2.8), Inches(1.8), Inches(1.6), Inches(1.4), Inches(1.8), Inches(1.4)]
    col_x = [Inches(0.4)]
    for w in col_widths[:-1]:
        col_x.append(col_x[-1] + w)

    row_h = Inches(0.46)

    for ri, row in enumerate(rows):
        y = Inches(1.05) + ri * row_h

        if ri == 0:
            bg = HIGHLIGHT
            fc = YELLOW
            bold = True
        elif row[0] == "":
            continue
        elif "const_mem" in row[1]:
            bg = RGBColor(0x1A, 0x3A, 0x1A)
            fc = GREEN
            bold = True
        else:
            bg = ACCENT if ri % 2 == 0 else RGBColor(0x1E, 0x1E, 0x3C)
            fc = WHITE
            bold = False

        for ci, (cell, cw, cx) in enumerate(zip(row, col_widths, col_x)):
            add_rect(slide, cx, y, cw - Inches(0.05), row_h - Inches(0.04), bg)
            add_text(
                slide,
                cell,
                cx + Inches(0.06),
                y + Inches(0.07),
                cw - Inches(0.12),
                row_h - Inches(0.1),
                font_size=12,
                bold=bold,
                color=fc,
                align=PP_ALIGN.LEFT if ci < 2 else PP_ALIGN.CENTER,
            )


def slide_discussion(prs: Presentation) -> None:
    slide = blank_slide(prs)
    fill_bg(slide, DARK_BG)
    section_header(slide, "Discussion & Conclusions")

    findings = [
        (GREEN, "Constant memory is the winner",
         "Broadcasts filter reads to all warp threads in one transaction.\n"
         "Advantage grows with filter size — up to 1.44× over naive at 4096²/11×11."),
        (ORANGE, "Tiled shared memory is slower than naive on RTX 4060",
         "Larger blocks (e.g. 676 threads for R=5 vs 256 for naive) reduce SM occupancy.\n"
         "RTX 4060's large L1 cache already captures input reuse — shared memory adds less benefit."),
        (RED, "cuDNN underperforms for single-channel, single-image workloads",
         "cuDNN is designed for batch + multi-channel parallelism.\n"
         "Algorithm selection and descriptor overhead dominate when N=C=1."),
        (BLUE, "Arithmetic throughput scales with filter size",
         "Larger filters = more FMAs per pixel = higher GFLOPS utilization.\n"
         "const_mem exceeds 1 TFLOPS for 4096×4096 with 11×11 filter."),
    ]

    for i, (color, title, body) in enumerate(findings):
        y = Inches(1.15) + i * Inches(1.48)
        add_rect(slide, Inches(0.4), y, Inches(0.1), Inches(1.25), color)
        add_text(
            slide, title,
            Inches(0.65), y,
            Inches(12.0), Inches(0.4),
            font_size=16, bold=True, color=color,
        )
        add_text(
            slide, body,
            Inches(0.65), y + Inches(0.42),
            Inches(12.0), Inches(0.85),
            font_size=13, color=LIGHT_GRAY,
        )


def slide_future_work(prs: Presentation) -> None:
    slide = blank_slide(prs)
    fill_bg(slide, DARK_BG)
    section_header(slide, "Future Work & Takeaways")

    left = [
        "Multi-channel convolution (C > 1)",
        "Batched execution (N > 1) — where cuDNN shines",
        "Separable filters: decompose F×F into two F×1 passes",
        "Warp-level primitives (__shfl_sync) for filter broadcast",
        "Half-precision (FP16) for doubled throughput on Tensor Cores",
        "Larger tile widths with register blocking",
    ]

    right = [
        "Constant memory is a simple, high-impact optimization",
        "Occupancy matters — more threads ≠ always faster",
        "Library APIs (cuDNN) need the right workload to shine",
        "Profiling first: RTX 4060 L1 already captures many reuses",
        "Benchmark with realistic workloads before generalizing",
    ]

    add_text(slide, "Future Directions", Inches(0.4), Inches(1.1), Inches(6.0), Inches(0.4), font_size=18, bold=True, color=YELLOW)
    add_bullet_box(slide, left, Inches(0.4), Inches(1.6), Inches(6.0), Inches(4.5), font_size=14)

    add_text(slide, "Key Takeaways", Inches(7.0), Inches(1.1), Inches(5.9), Inches(0.4), font_size=18, bold=True, color=YELLOW)
    add_bullet_box(slide, right, Inches(7.0), Inches(1.6), Inches(5.9), Inches(4.5), font_size=14)

    # Bottom bar
    add_rect(slide, Inches(0), Inches(6.8), W, Inches(0.7), HIGHLIGHT)
    add_text(
        slide,
        "Hardware: NVIDIA RTX 4060 Laptop GPU  •  CUDA 12.9  •  Driver 591.86  •  All kernels validated for correctness",
        Inches(0.4),
        Inches(6.88),
        Inches(12.5),
        Inches(0.45),
        font_size=12,
        color=LIGHT_GRAY,
        align=PP_ALIGN.CENTER,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "results" / "slides.pptx")

    prs = new_prs()

    slide_title(prs)
    slide_overview(prs)
    slide_naive(prs)
    slide_tiled(prs)
    slide_const_mem(prs)
    slide_cudnn(prs)
    slide_results_time_size(prs)
    slide_results_time_filter(prs)
    slide_results_speedup(prs)
    slide_results_bandwidth(prs)
    slide_key_numbers(prs)
    slide_discussion(prs)
    slide_future_work(prs)

    prs.save(out_path)
    print(f"Saved {len(prs.slides)} slides → {out_path}")


if __name__ == "__main__":
    main()
