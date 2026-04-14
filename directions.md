# TurboSplat: SIGGRAPH Paper Directions

## Research Context (April 2026)

Applying TurboQuant (arXiv:2504.19874, ICLR 2026) — a data-oblivious online vector quantization method — to compress 3D Gaussian Splatting. The method uses random rotation to induce Beta-distributed coordinates, then applies optimal scalar quantizers per coordinate. Provably near-optimal MSE (within 2.7x of Shannon bound). CPU-only, sub-second, no training needed.

## Competitive Landscape

- **20+ 3DGS compression papers** exist (HAC++, FCGS, CompGS, FlexGaussian, etc.)
- HAC++ achieves 100x compression; TurboSplat achieves 6.5x — ratio is modest
- **No 3DGS paper has information-theoretic guarantees** — this is novel
- **No one has applied data-oblivious VQ to 3DGS** — all existing VQ methods use learned codebooks
- **Progressive streaming is killed** — LapisGS, PCGS, LTS, ProGS (5+ papers)
- **SH overfitting in full-view is novel** — DropAnSH-GS only covers sparse-view
- **"Compression improves quality" is novel** — SA-3DGS observes it but doesn't explain why

## Key Prior Art to Differentiate From

| Paper | Overlap | Differentiation |
|-------|---------|----------------|
| DropAnSH-GS (Feb 2026) | SH overfitting | Sparse-view only; no quantitative per-band analysis; no denoising framing |
| SA-3DGS (Aug 2025) | Compression improves PSNR (+0.54 dB) | Doesn't explain WHY; no overfitting theory |
| EntropyGS (Aug 2025) | Training-free compression | 47x slower (16s CPU); no provable bounds |
| FlexGaussian (ACM MM 2025) | Training-free, fast | Requires GPU; no theory; TurboSplat beats it (+0.28 dB, 21% smaller) |
| FCGS (ICLR 2025) | Fast compression | Requires GPU, neural network; 40x slower |
| QReg (2022) | Quantization-as-regularization theory | For standard DNNs, not 3DGS |
| Quant-Noise (Meta 2020) | Noise injection during training | For DNNs; precedent for SQR but different domain |
| SPZ format (Niantic) | CPU-only, simple quantization+gzip | No provable bounds; must compare against |

---

## Approach 1: "Diagnosis-First" (CHOSEN)

**Title:** *How Much Do Spherical Harmonics Overfit? Provably Optimal Compression as Diagnosis and Cure*

**Core narrative:** We discovered something fundamental about 3DGS — SH coefficients massively overfit — and information theory explains why compression can actually improve quality. This leads to both a compression method (TurboQuant, post-hoc) and a training improvement (SQR, during training).

### Paper Structure

1. **Discovery:** 3DGS overfits 2.6-7.6 dB across 16+ scenes (NeRF Synthetic, Mip-NeRF 360, T&T, Deep Blending). Per-band analysis shows band l=3 has 2.43x overfitting ratio — 60% of its content is noise.

2. **Theory:** TurboQuant provides provable MSE distortion bounds (D_mse <= sqrt(3*pi)/2 * 1/4^b). Derive theorem: post-hoc compression improves test PSNR when the scene's overfitting gap exceeds the quantization distortion. The overfitting gap acts as a "budget" that absorbs compression damage.

3. **Method (TurboSplat):** Apply TurboQuant to all 3DGS attributes. Random rotation decorrelates SH coefficients → per-coordinate optimal scalar quantization. CPU-only, sub-second, no training. Provably near-optimal.

4. **Cure (SQR):** Stochastic Quantization Regularization — inject TurboQuant noise during training (after densification, ~15K iterations). Anneal bit-width b from 2→4. Prevents SH overfitting from forming. The diagnosis motivates the cure.

5. **Experiments:**
   - Overfitting analysis: 16+ scenes, train/test gaps, per-band hierarchy
   - Compression: full-attribute results on standard benchmarks (MipNeRF360, T&T, DB, NeRF Syn)
   - Comparison: vs HAC++, FCGS, FlexGaussian, EntropyGS, SPZ, CompGS
   - SQR: test PSNR improvement from quantization regularization
   - Ablations: bit-width sweep, rotation vs no-rotation, per-attribute analysis
   - Edge deployment: CPU timing on various devices

### Novelty Claims (Defensible)

1. First quantitative SH overfitting analysis in full-view 3DGS (per-band, 16+ scenes)
2. First 3DGS compression with information-theoretic optimality guarantees
3. Theory connecting overfitting gap to compression benefit (when compression helps)
4. SQR: quantization-aware regularization for 3DGS training
5. CPU-only sub-second compression (40-50x faster than nearest training-free GPU method)

### Risks

- **SQR must work.** Previous attempt failed with NaN. Likely a gradient scaling issue. Fallback: drop SQR and go with Approach 2.
- **6.5x ratio is modest.** Frame as "quality-optimal operating point" — we compress exactly as much as theory says is safe.
- **"Applying existing VQ" criticism.** Counter: TurboQuant is the tool, the insight is the contribution.

---

## Approach 2: "Theory-First" (Fallback)

**Title:** *Information-Theoretic Limits of 3D Gaussian Compression*

**Core narrative:** We derive rate-distortion bounds specific to 3DGS. Theory predicts compression should damage quality by X dB — but measured damage is LESS. Why? Because SH overfit. The gap between theoretical damage and actual damage reveals the overfitting.

### Paper Structure

1. Theory: rate-distortion bounds for 3DGS via TurboQuant (Beta distribution, optimal scalar quantizer)
2. Observation: theory vs reality gap → overfitting explanation
3. Method: TurboQuant compression (CPU, sub-second, provable bounds)
4. Experiments: theory validation, full benchmarks, comparisons

**Pros:** No SQR dependency. Pure analysis + method. Publishable if SQR fails.
**Cons:** Weaker narrative. May feel thin for SIGGRAPH ("theory paper applying existing bounds").

---

## Approach 3: "Speed-First" (Most Practical, Weakest Novelty)

**Title:** *Sub-Second 3DGS Compression with Provable Quality Guarantees*

**Core narrative:** All 3DGS compression needs GPU + minutes/hours. We achieve sub-second CPU compression with provable bounds. Surprise: quality loss is less than theory predicts because SH overfit.

### Paper Structure

1. Problem: GPU requirement blocks edge/mobile deployment
2. Method: TurboQuant (data-oblivious, CPU, 0.35s, provable bounds)
3. Insight: compression as denoising (supporting observation)
4. Experiments: speed benchmarks, quality, edge deployment

**Pros:** Easy to implement. Clear practical contribution.
**Cons:** "Applied existing method" — weakest for SIGGRAPH. Better suited for SIGGRAPH Asia or systems venue.

---

## Essential References

- TurboQuant (arXiv:2504.19874, ICLR 2026) — the VQ method
- DropAnSH-GS (arXiv:2602.20933) — SH overfitting (sparse-view)
- SA-3DGS (arXiv:2508.03017) — compression improves PSNR
- QReg (arXiv:2206.12372) — quantization-as-regularization theory
- Quant-Noise (arXiv:2004.07320) — noise injection during training
- "Why Quantization Improves Generalization" (arXiv:2206.05916) — NTK analysis
- "When Less is More" (arXiv:2512.18934) — quantization improves continual learning
- EntropyGS (arXiv:2508.10227) — training-free entropy coding
- FlexGaussian (arXiv:2507.06671) — training-free mixed-precision
- FCGS (arXiv:2410.08017) — feed-forward compression
- HAC++ (arXiv:2501.12255) — 100x compression
- SPZ format (Niantic) — simple CPU baseline
- CompGS (arXiv:2311.18159) — VQ on Gaussian attributes
- LightGaussian — SH distillation
- MDL / compression-generalization bounds (NeurIPS 2023, COLT 2022)
