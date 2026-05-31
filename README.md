# TurboSplat: SH Overfitting Analysis and Training-Free 3DGS Compression

**"SH Coefficients Are Spurious Harmonics: Overfitting Analysis and Free Compression in 3DGS"**

*Target venue: SIGGRAPH Asia 2026 Technical Papers*

---

## Overview

This repository contains the code and paper for **TurboSplat**, an analysis and compression framework for 3D Gaussian Splatting (3DGS). We make two interconnected contributions:

1. **SH Overfitting Analysis**: We show that spherical harmonic (SH) coefficients in 3DGS massively overfit — contributing 2–7 dB more to *training* views than *test* views across 21 standard scenes. Band l=3 has a 2.45× overfitting ratio (≈60% noise).

2. **Provably Near-Optimal Compression (TurboSplat)**: We apply [TurboQuant](https://arxiv.org/abs/2504.19874) (data-oblivious VQ with provable MSE bounds) to 3DGS and achieve **9.1× compression at 0.32 dB** average quality loss — CPU-only, sub-second, no training or fine-tuning required.

**Key finding**: On high-overfitting scenes, compression is effectively *free* — the overfitting gap absorbs quantization damage:

| Scene | Ratio | ΔPSNR |
|---|---|---|
| drjohnson | 9.3× | +0.01 dB |
| playroom | 9.0× | −0.04 dB |
| treehill | 9.9× | +0.02 dB |
| garden | 9.2× | +0.05 dB |

---

## Method

TurboSplat applies [TurboQuant (arXiv:2504.19874, ICLR 2026)](https://arxiv.org/abs/2504.19874) to all 3DGS attributes:

- **Random rotation** decorrelates SH coefficient vectors; after rotation, coordinates are approximately Beta-distributed
- **Per-coordinate scalar quantization** using the Lloyd-Max codebook — provably near-optimal (MSE within 2.7× of Shannon bound)
- **Entropy coding** via zstd on bit-packed indices
- **Position**: 16-bit uniform quantization (0.86 dB/bit sensitivity below 16 bits)

The compression pipeline is CPU-only with no GPU requirement and runs in 1–40 seconds per scene.

### Compression Pareto Points

| Config | Ratio | ΔPSNR |
|---|---|---|
| Quality (b=3) | 5.6× | 0.07 dB |
| Balanced (b=2, default) | 9.1× | 0.32 dB |
| +Merge (v3 pipeline) | 12.5× | 0.58 dB |

### Comparison with Training-Free Baselines

| Method | Ratio | ΔPSNR | Device | Time | Provable Bounds |
|---|---|---|---|---|---|
| HAC++ | 100× | ~0 dB | GPU | min | No |
| EntropyGS | 30× | 0.04 dB | CPU | 16s | No |
| FlexGaussian | ~20× | <1 dB | GPU | 25s | No |
| FCGS (λ=4e-4) | 16.9× | 0.15 dB | GPU | 14s | No |
| **TurboSplat** | **9.1×** | **0.32 dB** | **CPU** | **<1s** | **Yes** |

---

## Repository Structure

```
TurboQuant/
├── gaussian-splatting/          # Main codebase (fork of 3DGS)
│   ├── turbo_quant/             # Core TurboQuant module
│   │   ├── quantizer.py         # TurboQuantizer: random rotation + scalar codebook
│   │   └── codebook.py          # Lloyd-Max codebook generation
│   │
│   ├── compress.py              # v1/v2: TurboQuant SH compression pipeline
│   ├── compress_v3.py           # v3: Voxel merging + anchor coding + TurboQuant SH
│   ├── compress_nuclear.py      # Nuclear norm compression experiment
│   ├── decompress.py            # Reconstruct PLY from .npz/.tsv4 files
│   │
│   ├── diagnosis/               # SH overfitting analysis scripts
│   │   ├── train_test_gap.py    # Measure train/test PSNR gap per scene
│   │   ├── sh_band_analysis.py  # Per-band overfitting ratios (R1/R2/R3)
│   │   └── colmap_overfitting_15views.py  # COLMAP sparse-view overfitting
│   │
│   ├── sqr/                     # Stochastic Quantization Regularization (training)
│   │   └── sqr_module.py        # SQR: inject TurboQuant noise during training
│   │
│   ├── NanoGS/                  # Gaussian merging module
│   │   └── simplification.py
│   ├── nanogs_merge.py          # Merge script for v3 pipeline
│   │
│   ├── eval_compression.py      # Batch compression evaluation (PSNR/SSIM/LPIPS)
│   ├── eval_full_metrics.py     # Full metrics on trained models
│   ├── eval_colmap_full.py      # COLMAP scene evaluation
│   ├── entropy_utils.py         # Entropy coding utilities (zstd, bit-packing)
│   │
│   ├── train.py                 # Standard 3DGS training
│   ├── render.py                # Standard 3DGS rendering
│   ├── metrics.py               # PSNR/SSIM/LPIPS
│   │
│   ├── scene/                   # 3DGS scene utilities
│   ├── gaussian_renderer/       # Differentiable Gaussian rasterizer
│   ├── tests/                   # Unit tests
│   └── data/                    # Datasets (symlinks/downloaded separately)
│       ├── nerf_synthetic/      # NeRF Synthetic (8 scenes)
│       ├── 360_v2/              # MipNeRF360 (9 scenes)
│       ├── tandt/               # Tanks & Temples (truck, train)
│       └── db/                  # Deep Blending (drjohnson, playroom)
│
├── paper/                       # LaTeX source (SIGGRAPH Asia 2026)
│   ├── main.tex
│   └── references.bib
│
├── paper_blueprint.md           # Detailed paper outline and full results
├── directions.md                # Research directions and competitive analysis
└── paper_strategy.md            # Paper strategy and rebuttal prep
```

---

## Installation

### Prerequisites

- CUDA 11.6+ compatible GPU (for training and rendering)
- Conda

### Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/JaeLee18/3dgs_compression
cd 3dgs_compression/gaussian-splatting

# Create conda environment
conda env create -f environment.yml
conda activate gaussian_splatting

# Install diff-gaussian-rasterization + simple-knn
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/fused-ssim
```

The compression pipeline (`compress.py`, `decompress.py`) requires only NumPy, SciPy, and plyfile — no GPU needed.

---

## Usage

All commands are run from `gaussian-splatting/`.

### 1. Train a 3DGS model (standard)

```bash
python train.py -s data/nerf_synthetic/lego -m output/lego
```

### 2. Diagnose SH Overfitting

Measure the train/test PSNR gap for a trained model:

```bash
python -m diagnosis.train_test_gap -m output/lego -s data/nerf_synthetic/lego
```

Measure per-band overfitting ratios (R1/R2/R3):

```bash
python -m diagnosis.sh_band_analysis -m output/lego -s data/nerf_synthetic/lego
```

### 3. Compress

**Standard compression (v2, recommended):**

```bash
# Default balanced config: 9.1× / ~0.32 dB
python compress.py -m output/lego -o compressed/lego.npz

# Quality config: 5.6× / ~0.07 dB
python compress.py -m output/lego -o compressed/lego.npz --sh_bits 3 --pos_bits 16

# Aggressive: higher ratio
python compress.py -m output/lego -o compressed/lego.npz --sh_bits 2 --prune 0.1
```

**v3 pipeline (voxel merging + anchor coding):**

```bash
# Default
python compress_v3.py -m output/lego

# With merging (targets 12–15× compression)
python compress_v3.py -m output/lego --merge_threshold 5 --sh_bits 2
```

### 4. Decompress

```bash
python decompress.py -i compressed/lego.npz -o decompressed/lego.ply
```

### 5. Evaluate Compression

Evaluate PSNR/SSIM/LPIPS on a set of scenes at multiple bit-widths:

```bash
python eval_compression.py --scenes lego chair ficus --sh_bits 2 3 4 \
    --source_root data/nerf_synthetic
```

### 6. Run Full Metrics

```bash
python eval_full_metrics.py -m output/lego -s data/nerf_synthetic/lego
```

---

## Key Results

### SH Overfitting (21 scenes)

| Dataset | Avg Train/Test Gap | Avg R₃ (band 3) |
|---|---|---|
| NeRF Synthetic | 3.64 dB | 2.45 |
| MipNeRF360 | 2.85 dB | 2.49 |
| drjohnson | 7.90 dB | 8.30 |
| playroom | 4.02 dB | 7.24 |

R₃ = 2.45 means band l=3 contributes 2.45× more to training than test PSNR → approximately 60% of band 3 content is overfitting noise.

### Compression Results (balanced config: b=2 p16 d10 s10 r10 o8 +zstd)

| Dataset | Ratio | ΔPSNR |
|---|---|---|
| NeRF Synthetic (8 scenes) | 8.6× | 0.43 dB |
| MipNeRF360 + T&T + DB (13 scenes) | 9.5× | 0.23 dB |
| **Overall (21 scenes)** | **9.1×** | **0.32 dB** |

---

## Datasets

Download the standard benchmarks and place them under `data/`:

- [NeRF Synthetic](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) → `data/nerf_synthetic/`
- [MipNeRF360](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip) → `data/360_v2/`
- [Tanks & Temples](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt.zip) → `data/tandt/`
- [Deep Blending](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/db.zip) → `data/db/`

---

## Tests

```bash
cd gaussian-splatting
python -m pytest tests/ -v
```

Tests cover the TurboQuant quantizer, codebook generation, entropy utilities, and compression round-trips.

---

## Theory

TurboQuant provides a provable MSE bound per coordinate after random rotation:

```
D_mse ≤ sqrt(3π)/2 · (1/4^b)
```

where `b` is the bit-width. This is within 2.7× of the Shannon information-theoretic optimum for Beta-distributed coordinates.

**Why compression can improve test quality**: Let `G` = overfitting gap (train PSNR − test PSNR), `D` = compression distortion. When `G > D`, the quantization noise partially cancels overfitting artifacts, improving test PSNR. This is why drjohnson (G=7.90 dB) gains quality at 9.3× compression.

---

## Paper

The LaTeX source is in `paper/main.tex`. To build:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

See `paper_blueprint.md` for the complete paper outline, all results, and rebuttal preparation.

---

## Citation

```bibtex
@inproceedings{turbosplat2026,
  title     = {SH Coefficients Are Spurious Harmonics: Overfitting Analysis and Free Compression in 3DGS},
  author    = {JaeLee18},
  booktitle = {SIGGRAPH Asia 2026 Technical Papers},
  year      = {2026},
}
```

The compression method builds on:

```bibtex
@inproceedings{turboquant2026,
  title     = {TurboQuant: Near-Optimal Data-Oblivious Vector Quantization},
  author    = {...},
  booktitle = {ICLR 2026},
  note      = {arXiv:2504.19874},
}
```

---

## Acknowledgements

This project builds on the original [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) codebase by Kerbl et al. (SIGGRAPH 2023).
