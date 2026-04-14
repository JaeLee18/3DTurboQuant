# TurboSplat: SIGGRAPH Paper Design Spec

**Date:** 2026-04-14
**Working title:** *How Much Do Spherical Harmonics Overfit? Provably Optimal Compression as Diagnosis and Cure*
**Target venue:** SIGGRAPH 2027 (~January 2027 deadline), with SIGGRAPH Asia 2026 (May 12-13) as earlier option

---

## 1. Core Thesis

3D Gaussian Splatting massively overfits its Spherical Harmonic coefficients. We quantify this (2.6-7.6 dB train/test gap), explain it per-band (band l=3 is 60% noise), and show that provably optimal compression via TurboQuant acts as denoising — improving test quality on high-overfitting scenes. This leads to both a training-free compression method (TurboSplat) and a training-time regularizer (SQR).

## 2. Paper Structure

### Section 1: Introduction
- 3DGS is state-of-the-art but stores redundant/overfit information in SH coefficients
- Nobody has quantified how much SH overfits in standard (full-view) settings
- We show: 2.6-7.6 dB train/test gap, band l=3 is 60% noise
- Compression with provable guarantees acts as principled denoising
- Contributions: (1) overfitting diagnosis, (2) provably optimal compression, (3) theory connecting overfitting to compression benefit, (4) SQR regularizer

### Section 2: SH Overfitting Analysis (Pillar 1 — DIAGNOSIS)
- Measure train/test PSNR gap across 16+ scenes (NeRF Synthetic, MipNeRF360, T&T, Deep Blending)
- Per-band zeroing experiment: remove band l=k, measure train PSNR drop vs test PSNR drop
- Define overfitting ratio R_k = (train_drop_k) / (test_drop_k) for each band k
- Key finding: R_1 ~ 1.77, R_2 ~ 2.03, R_3 ~ 2.43
- Interpretation: higher bands overfit more; band l=3 contributes least to test quality despite largest norms
- Iteration sweep: show overfitting accumulates (ratio grows from 1.37x at 3K to 2.52x at 30K iterations)

### Section 3: TurboQuant Background + Theory (Pillar 2 — THEORY)

#### 3.1 TurboQuant Review
- Random rotation Pi maps input to uniform hypersphere → Beta-distributed coordinates
- Optimal scalar quantizer per coordinate (Lloyd-Max on Beta distribution)
- MSE bound: D_mse <= sqrt(3*pi)/2 * 1/4^b (Theorem 1 from TurboQuant paper)
- For inner products: two-stage with QJL on residual (Theorem 2)
- Information-theoretic lower bound: D_mse >= 1/4^b (Theorem 3)
- Near-optimal: upper/lower differ by factor sqrt(3*pi)/2 ~ 2.7

#### 3.2 When Does Compression Improve Test Quality? (NEW THEOREM)
- Define: G = train_PSNR - test_PSNR (overfitting gap in dB)
- Define: D(b) = compression distortion at bit-width b (from TurboQuant bounds)
- Theorem: If G > 10*log10(1 + D(b)/sigma^2_signal), then compression at bit-width b improves test PSNR
- Intuition: the overfitting gap provides a "damage budget" — compression removes overfit noise while the budget absorbs the reconstruction error
- The random rotation is key: it distributes the damage uniformly across all coefficients, preventing catastrophic damage to any single component
- Validate: show predicted vs actual test PSNR improvement across scenes

### Section 4: TurboSplat Method (Pillar 3 — METHOD)

#### 4.1 Compression Pipeline
- Input: trained 3DGS model (.ply file with all attributes)
- Per-attribute quantization:
  - **SH coefficients** (45D for degree 3): TurboQuant with random rotation. This is the main target — highest dimension, most overfitting
  - **Scales** (3D): log-transform then uniform scalar quantization
  - **Rotations** (4D quaternion): normalize then TurboQuant
  - **Positions** (3D): bounding-box normalization then uniform scalar quantization
  - **Opacity** (1D): sigmoid-space uniform scalar quantization
- Bit-width selection: b=3 for SH (quality-optimal point from theory), b=8 for positions, b=4 for scales/rotations/opacity
- Output: compressed .npz with bit-packed indices + rotation matrix + codebook (precomputed, same for all scenes)

#### 4.2 Decompression
- Load indices, look up precomputed codebook centroids
- Inverse rotation: multiply by Pi^T
- Reconstruct all attributes
- Render with standard 3DGS rasterizer (no modifications needed)

#### 4.3 Complexity Analysis
- Compression: O(N * d * log(d)) per Gaussian (dominated by random rotation via fast Walsh-Hadamard)
- Decompression: O(N * d) (table lookup + matrix multiply)
- No GPU required. No neural network. No data-dependent preprocessing.
- Wall-clock: ~0.35s for 100K Gaussians on CPU

### Section 5: Stochastic Quantization Regularization (Pillar 4 — CURE)

#### 5.1 Motivation
- Diagnosis shows SH overfits → prevent it during training
- Quant-Noise (Meta 2020) showed noise injection regularizes DNNs
- SQR: inject TurboQuant-style quantization noise into SH coefficients during training

#### 5.2 Method
- After densification is complete (~15K iterations), enable SQR
- Each forward pass: with probability p_sqr=0.5 (ablated in experiments), replace SH coefficients with their quantized version
  - Apply random rotation (new rotation each iteration for diversity)
  - Quantize to bit-width b_sqr
  - Dequantize and rotate back
  - Use straight-through estimator for backpropagation
- Annealing schedule: b_sqr starts at 2 (coarse, strong regularization), anneals to 4 (fine, mild regularization) over training
- Warmup: noise scale ramps from 0 to 1 over first 1K SQR iterations
- Gradient clipping: clip SH gradients during SQR phase to prevent NaN (max_norm=1.0)

#### 5.3 Expected Outcome
- Test PSNR improves by 0.3-0.5 dB (based on overfitting gap analysis — removing 60% noise from band l=3 should recover significant quality)
- Train PSNR decreases slightly (less overfitting)
- Models trained with SQR compress better (already regularized, less to denoise)

### Section 6: Experiments

#### 6.1 Datasets
- **NeRF Synthetic** (8 scenes): Lego, Chair, Drums, Ficus, Hotdog, Materials, Mic, Ship
- **Mip-NeRF 360** (9 scenes): bicycle, bonsai, counter, flowers, garden, kitchen, room, stump, treehill
- **Tanks & Temples** (2 scenes): truck, train
- **Deep Blending** (2 scenes): playroom, drjohnson
- Total: 21 scenes

#### 6.2 Experiment 1: Overfitting Diagnosis
- Train/test PSNR gap for all 21 scenes
- Per-band zeroing: overfitting ratio R_k for bands k=1,2,3
- Iteration sweep: overfitting ratio vs training iteration (checkpoints at 1K/3K/7K/15K/30K)
- Visualization: bar charts, line plots

#### 6.3 Experiment 2: TurboSplat Compression
- Full-attribute compression at b=2,3,4 for SH
- Metrics: PSNR, SSIM, LPIPS, compression ratio, compression time, file size
- Comparison table vs:
  - HAC++ (use published numbers, 100x compression)
  - FCGS (reproduce or use published, 17x)
  - FlexGaussian (reproduce or published, ~25x)
  - EntropyGS (published, ~30x)
  - SPZ format (simple baseline, ~10x)
  - CompGS (published, ~45x)
- Rate-distortion curves: PSNR vs bits-per-Gaussian
- Speed comparison: wall-clock time bar chart

#### 6.4 Experiment 3: Theory Validation
- Predicted distortion (from TurboQuant bounds) vs actual distortion
- Predicted "compression improves quality" threshold vs actual improvement
- Scatter plot: overfitting gap (x) vs test PSNR change from compression (y)
- Show: scenes above the theoretical threshold improve, scenes below degrade

#### 6.5 Experiment 4: SQR Regularization
- Retrain all 21 scenes with SQR enabled
- Compare: vanilla 3DGS test PSNR vs SQR test PSNR
- Compare: SQR + post-hoc compression vs vanilla + post-hoc compression
- Ablation: SQR start iteration, annealing schedule, noise probability p_sqr

#### 6.6 Experiment 5: Ablations
- Random rotation vs no rotation (per-channel scalar quantization)
- Bit-width sweep: b=1,2,3,4,5 for SH
- Per-attribute analysis: which attributes benefit most from TurboQuant
- Codebook: precomputed (Beta) vs data-dependent (k-means) vs uniform

#### 6.7 Experiment 6: Edge Deployment
- Compression + decompression timing on: desktop CPU, laptop CPU, Raspberry Pi / ARM
- Memory footprint: compressed model size vs original

### Section 7: Limitations and Discussion
- Compression ratio (6.5x) is modest vs HAC++ (100x) — we optimize for quality, not ratio
- SQR adds training time (quantization overhead per forward pass)
- Random rotation matrix must be stored (d*d floats) — negligible for d=45 but grows with dimension
- Theory assumes worst-case; actual data may be easier than worst case

## 3. Codebase Structure

```
/mnt/ssd1/idea/TurboQuant/
  gaussian-splatting/           # Clone of original 3DGS repo
    turbo_quant/                # Core TurboQuant implementation
      __init__.py
      codebook.py               # Precomputed optimal codebooks for Beta distribution
      quantizer.py              # Quantize/dequantize with random rotation
      fast_hadamard.py          # Fast Walsh-Hadamard transform (optional speedup)
    diagnosis/                  # Overfitting analysis tools
      train_test_gap.py         # Measure train/test PSNR gap
      sh_band_analysis.py       # Per-band zeroing experiment
      iteration_sweep.py        # Overfitting vs training iteration
    sqr/                        # Stochastic Quantization Regularization
      sqr_module.py             # SQR noise injection module
      train_sqr.py              # Modified training script with SQR
    compress.py                 # Full-attribute compression CLI
    decompress.py               # Decompression + render
    eval_compression.py         # Benchmark compression across all scenes
    eval_comparisons.py         # Run/collect comparison method results
  paper/                        # LaTeX source
    main.tex
    figures/
  directions.md                 # This file's parent
  docs/superpowers/specs/       # This spec
```

## 4. Implementation Order

### Phase 1: Infrastructure (Days 1-3)
1. Clone 3DGS repo, set up conda env, verify training works
2. Implement turbo_quant/ module (codebook.py, quantizer.py)
3. Implement compress.py and decompress.py
4. Verify compression pipeline works end-to-end on one scene

### Phase 2: Diagnosis (Days 4-7)
5. Train all 21 scenes (or download pre-trained if available)
6. Run train/test gap analysis on all scenes
7. Run per-band SH zeroing experiment
8. Run iteration sweep on 2-3 representative scenes
9. Generate diagnosis figures

### Phase 3: Compression Experiments (Days 8-12)
10. Full-attribute compression on all 21 scenes at b=2,3,4
11. Collect comparison method results (published numbers + reproduce key ones)
12. Theory validation: predicted vs actual distortion
13. Rate-distortion curves and comparison tables
14. Speed benchmarks

### Phase 4: SQR (Days 13-18)
15. Implement SQR module with straight-through estimator
16. Test on single scene (Lego) with gradient monitoring
17. Debug NaN issues (gradient clipping, warmup, learning rate)
18. If SQR works: train all 21 scenes with SQR
19. If SQR fails: fall back to Approach 2 (theory-first, no cure)

### Phase 5: Paper (Days 19-25)
20. Write paper draft in LaTeX
21. Generate all figures and tables
22. Internal review, polish, submit

## 5. Success Criteria

- **Minimum viable paper (Approach 2 fallback):** Overfitting diagnosis (16+ scenes) + TurboSplat compression with provable bounds + theory validation + comparisons. No SQR needed.
- **Full paper (Approach 1):** All of the above + SQR improving test PSNR by >= 0.3 dB on average.
- **SIGGRAPH-worthy bar:** The "compression improves quality" finding is validated with theory, and either SQR works or the theory-practice gap analysis is compelling enough to stand alone.

## 6. Key Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| SQR NaN during training | Loses "cure" pillar | Gradient clipping, warmup, LR reduction; fallback to Approach 2 |
| Overfitting gap smaller on MipNeRF360 | Weakens diagnosis | Already confirmed 0.99-7.58 dB on 16 scenes; focus on high-gap scenes |
| 6.5x ratio dismissed by reviewers | Method seems weak | Frame as quality-optimal; show rate-distortion curve; speed advantage |
| Theorem doesn't predict well | Theory pillar weak | Use as approximation/intuition rather than tight bound; empirical validation |
| DropAnSH-GS reviewer overlap | Novelty questioned | Clear differentiation: full-view, quantitative, compression-as-denoising, theory |
