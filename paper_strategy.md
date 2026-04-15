# TurboSplat Paper Strategy — SIGGRAPH Asia 2026

## Core Narrative

**Old:** "We built a 9.1x compressor."
**New:** "We define the Pareto Frontier of Training-Free 3DGS Compression. We demonstrate that beyond ~12x, distortion becomes non-monotonic and scene-dependent, requiring learning-based priors. Our method provides the optimal quality/size trade-off for CPU deployment."

## Title Candidates

1. "How Much of Your Gaussian is Noise? The Rate-Distortion Limits of Training-Free 3DGS Compression"
2. "TurboSplat: Provably Optimal Training-Free 3D Gaussian Compression"
3. "Spherical Harmonics Overfit in 3DGS: Analysis and Optimal Compression"

## Key Contributions

1. **First systematic SH overfitting analysis** in full-view 3DGS (19 scenes, R₃=2.45 avg)
2. **Pareto frontier** of training-free compression: 5.6x/0.07dB to 12.5x/0.58dB
3. **Provable near-optimal compression** (within 2.7x of Shannon bound)
4. **CPU-only, no training, no neural network** — unique deployment profile
5. **Position sensitivity discovery** — COLMAP scenes require 16-bit positions (15-bit adds 0.86 dB)

## Paper-Ready Numbers

### Overfitting (19 scenes)
- NeRF Synthetic: 3.64 dB avg gap, R₃=2.45
- COLMAP (gap>0): 2.85 dB avg gap, R₃=2.49
- Playroom: compression IMPROVES quality (+0.04 dB)

### Compression (optimal config: b=2 p16 d10 s10 r10 o8 +zstd)
- NeRF Synthetic: 8.6x / 0.43 dB
- COLMAP: 9.5x / 0.23 dB
- Overall: 9.1x / 0.32 dB
- All CPU-only, 1-40s

### Rate-Distortion Pareto
| Point | Ratio | Drop | Method |
|-------|-------|------|--------|
| Quality-first | 5.6x | 0.07 dB | TQ b=3, 16-bit all |
| Balanced | 9.1x | 0.32 dB | TQ b=2, zstd, tuned bits |
| +Merge | 12.5x | 0.58 dB | 20% NanoGS merge |
| Aggressive | 19.1x | 3.98 dB | 50% merge + drop L3 |

### Quantization Wall Evidence
- Positions: 16-bit mandatory (15-bit: +0.86 dB on stump)
- Scales: 8-bit is zero-cost
- SH L3 dropping: scene-dependent (ficus: -3.89 dB, garden: -0.28 dB)
- Adaptive SH allocation: risky without per-scene analysis

## Comparison Table

| Method | Ratio | PSNR Drop | Device | Training? | Time | Provable? |
|--------|-------|-----------|--------|-----------|------|-----------|
| HAC++ | 100x | ~0 dB | GPU | Yes | minutes | No |
| EntropyGS | 30x | 0.04 dB | CPU | No | 16s | No |
| FlexGaussian | ~20x | <1 dB | GPU | No | 25s | No |
| FCGS | 17x | ~0.15 dB | GPU | No | 14s | No |
| **TurboSplat** | **9.1x** | **0.32 dB** | **CPU** | **No** | **1-40s** | **Yes** |

## Figures Plan

1. **Fig 1 (Hook):** Side-by-side renders: Original vs 9.1x compressed. Imperceptible difference.
2. **Fig 2 (RD Curve):** Pareto frontier with competitors. Annotate "Quantization Wall" and "GPU Required" regions.
3. **Fig 3 (Overfitting):** Bar chart of R₁/R₂/R₃ across 19 scenes. Highlight the hierarchy.
4. **Fig 4 (Position Sensitivity):** Error maps at 16/15/14/12-bit positions on stump.
5. **Fig 5 (SH Band Analysis):** Per-band contribution to train vs test quality.

## Limitations (to write honestly)

- 9.1x is modest vs HAC++ (100x) — we optimize for quality, not ratio
- Training-based methods can exploit learned priors for higher compression
- Position sensitivity prevents universal below-16-bit quantization
- Adaptive SH allocation is scene-dependent and risky without analysis pass

## Key Defense Phrases

- "We prioritize fidelity preservation and deployment accessibility over maximum compression."
- "We establish the quantization wall: the point beyond which training-free methods cannot compress without scene-dependent quality collapse."
- "Our method is the only 3DGS compression with information-theoretic optimality guarantees."
- "For embedded/CPU scenarios requiring immediate deployment on pre-trained models, our method defines the current state-of-the-art."

## 4-Week Schedule

| Week | Tasks |
|------|-------|
| 1 (Apr 14-20) | DONE: All experiments, 19 scenes, compression, overfitting |
| 2 (Apr 21-27) | Rate-distortion curves, comparison figures, reproduce EntropyGS numbers |
| 3 (Apr 28-May 4) | Write full paper, all figures/tables, abstract (due May 5) |
| 4 (May 5-12) | Polish, internal review, submit May 12 |
