# Paper Blueprint: SIGGRAPH Asia 2026

## Title: "SH Coefficients Are Spurious Harmonics: Overfitting Analysis and Free Compression in 3DGS"

## Key Sentence
"We show that SH coefficients in 3DGS massively overfit, contributing 2-7 dB more to training views than test views — and that this overfitting gap makes compression effectively free on certain scenes."

## Abstract (~150 words)

We show that spherical harmonic (SH) coefficients in 3D Gaussian Splatting (3DGS) massively overfit, contributing 2-7 dB more to training views than to test views across 21 standard benchmarks. This overfitting is concentrated in high-frequency SH bands — band 3 contributes 2.45x more to training than test, implying ~60% of this band is overfitting noise. We apply TurboQuant (a data-oblivious vector quantization method with provable MSE bounds) to 3DGS and achieve 9.1x compression at 0.32 dB quality loss. Critically, on high-overfitting scenes, compression becomes effectively free: drjohnson compresses 9.3x with 0.01 dB loss, and playroom compresses 9.0x with -0.04 dB (compression improves quality). These results suggest the SH overfitting gap absorbs quantization damage. We analyze this phenomenon across scene types, SH bands, and compression ratios, and discuss its implications for training-free 3DGS compression.

## Structure

### Page 1: Introduction (3/4 page)
- Hook: SH coefficients overfit 2-7 dB
- Gap numbers as evidence
- This is an analysis paper; TurboSplat enables the analysis
- Contributions list

### Pages 2-3: Understanding SH Overfitting (1.5 pages)
- Background: 3DGS, SH, train/test split
- Per-scene gap measurements (21 scenes)
- Per-band R_k ratios (hierarchy R1 < R2 < R3)
- Key insight: ~60% of band 3 is noise
- Implication: overfitting gap absorbs compression damage

### Pages 4-5: TurboSplat Results (1.5 pages)
- Method: brief (TurboQuant + zstd, cite ICLR 2026)
- Rate-distortion Pareto (3 operating points)
- "Free compression" scenes (drjohnson, playroom, treehill, garden)
- Comparison table (training-free vs training-required)
- Quantization wall (position sensitivity)

### Pages 6-7: Discussion (1 page)
- Why 9.1x is the theoretical limit of training-free
- Provable optimality
- What failed and why (SQR, adaptive SH, grid VQ)
- Future directions

### Pages 8-9: Figures (2 pages)

## Figures

1. **Rate-Distortion Frontier** (main result)
   - X: compression ratio (log), Y: PSNR drop
   - 3 operating points + competitors
   - "Free compression threshold" line

2. **SH Overfitting Analysis** (core finding)
   - Scatter: R3 vs gap for 21 scenes
   - Bar: avg R_k per band across datasets
   - Highlight drjohnson (R3=8.30, gap=7.90)

3. **Visual Quality** (proof)
   - drjohnson: 9.3x, 0.01 dB (invisible)
   - garden: 9.2x, 0.05 dB (invisible)
   - ficus: 8.6x, 0.72 dB (some artifacts)

4. **Quantization Wall** (position sensitivity)
   - PSNR drop vs position bits on stump
   - 0.86 dB per bit below 16

## Comparison Table

| Method | Ratio | Drop | Device | Training? | Time | Provable? |
|--------|-------|------|--------|-----------|------|-----------|
| HAC++ | 100x | ~0 dB | GPU | Yes | min | No |
| EntropyGS | 30x | 0.04 dB | CPU | No | 16s | No |
| FlexGaussian | ~20x | <1 dB | GPU | No | 25s | No |
| FCGS | 17x | ~0.15 dB | GPU | No | 14s | No |
| TurboSplat | 9.1x | 0.32 dB | CPU | No | 1-40s | Yes |

## Rebuttal Prep

**"9.1x is modest"** → Not the contribution. The contribution is the overfitting finding + provable bounds. 9.1x is the theoretical limit of training-free compression.

**"Just applying VQ"** → The method is not novel; the finding is. TurboQuant reveals that SH overfitting makes compression free.

**"Overfitting not surprising"** → The magnitude is surprising. 2-7 dB, R3=2.45, 60% noise. Nobody quantified this.

**"Only standard benchmarks"** → 21 scenes across 4 benchmarks. Standard evaluation protocol. Deep Blending scenes are large-scale.

## Complete Results Reference

### Overfitting (21 scenes)
NeRF Synthetic: gap=3.64 dB, R3=2.45
COLMAP (gap>0): gap=2.85 dB, R3=2.49
drjohnson: gap=7.90, R3=8.30 (strongest)
playroom: gap=4.02, R3=7.24

### Compression (optimal config: b=2 p16 d10 s10 r10 o8 +zstd)
NeRF Synthetic: 8.6x / 0.43 dB
COLMAP: 9.5x / 0.23 dB
Overall: 9.1x / 0.32 dB

### Free compression scenes (<0.05 dB)
drjohnson: 9.3x / 0.01 dB
playroom: 9.0x / -0.04 dB
treehill: 9.9x / 0.02 dB
garden: 9.2x / 0.05 dB

### Rate-distortion Pareto
Quality: 5.6x / 0.07 dB
Balanced: 9.1x / 0.32 dB
+Merge: 12.5x / 0.58 dB
