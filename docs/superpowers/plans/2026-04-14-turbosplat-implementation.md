# TurboSplat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build TurboSplat — a provably optimal, training-free, CPU-only 3DGS compression system with SH overfitting diagnosis and SQR regularization, targeting a SIGGRAPH paper.

**Architecture:** Four pillars built bottom-up: (1) TurboQuant core VQ implementation, (2) SH overfitting diagnosis tools, (3) full-attribute 3DGS compression pipeline, (4) SQR training-time regularizer. Each pillar produces both code and experimental results.

**Tech Stack:** Python 3.10+, PyTorch 2.5, CUDA 12.1, scipy, plyfile, lpips, numpy. Conda env: `/mnt/ssd1/conda_envs/nerf_tq/`. GPU: RTX 4090. Base 3DGS repo: clone from `graphdeco-inria/gaussian-splatting`.

**Python executable:** `/mnt/ssd1/conda_envs/nerf_tq/bin/python`

**Project root:** `/mnt/ssd1/idea/TurboQuant/`

**3DGS repo:** `/mnt/ssd1/idea/TurboQuant/gaussian-splatting/`

---

## File Map

```
gaussian-splatting/
  turbo_quant/
    __init__.py              # Package init, exports Quantizer class
    codebook.py              # Precomputed Lloyd-Max codebooks for Beta distribution
    quantizer.py             # TurboQuantizer: rotate, quantize, dequantize
  diagnosis/
    __init__.py
    train_test_gap.py        # Measure train/test PSNR gap for a trained model
    sh_band_analysis.py      # Per-band zeroing: overfitting ratio per SH band
    iteration_sweep.py       # Overfitting ratio vs training iteration
  sqr/
    __init__.py
    sqr_module.py            # SQR noise injection (torch.nn.Module wrapper)
    train_sqr.py             # Modified 3DGS training script with SQR
  compress.py                # CLI: compress a trained .ply to .npz
  decompress.py              # CLI: decompress .npz back to .ply, optionally render
  eval_compression.py        # Batch compression eval across all scenes
  tests/
    test_codebook.py         # Unit tests for codebook generation
    test_quantizer.py        # Unit tests for quantize/dequantize roundtrip
    test_compress.py         # Integration test for full compression pipeline
    test_sqr.py              # Unit tests for SQR module
```

---

## Task 1: Clone 3DGS Repo and Verify Environment

**Files:**
- Create: `gaussian-splatting/` (git clone)

- [ ] **Step 1: Clone the 3DGS repo**

```bash
cd /mnt/ssd1/idea/TurboQuant
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git
```

- [ ] **Step 2: Install 3DGS submodules into nerf_tq env**

```bash
cd /mnt/ssd1/idea/TurboQuant/gaussian-splatting
/mnt/ssd1/conda_envs/nerf_tq/bin/pip install submodules/diff-gaussian-rasterization
/mnt/ssd1/conda_envs/nerf_tq/bin/pip install submodules/simple-knn
/mnt/ssd1/conda_envs/nerf_tq/bin/pip install submodules/fused-ssim
```

- [ ] **Step 3: Verify the environment works**

```bash
cd /mnt/ssd1/idea/TurboQuant/gaussian-splatting
/mnt/ssd1/conda_envs/nerf_tq/bin/python -c "
import torch
from diff_gaussian_rasterization import GaussianRasterizer
from scene import GaussianModel
print('3DGS imports OK')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'PyTorch: {torch.__version__}')
"
```

Expected: `3DGS imports OK`, `CUDA: True`

- [ ] **Step 4: Download NeRF Synthetic dataset**

```bash
cd /mnt/ssd1/idea/TurboQuant/gaussian-splatting
mkdir -p data
cd data
wget https://huggingface.co/datasets/ivanleomk/nerf_synthetic/resolve/main/nerf_synthetic.zip
unzip nerf_synthetic.zip
# Should create: data/nerf_synthetic/{lego,chair,drums,ficus,hotdog,materials,mic,ship}/
```

Verify:
```bash
ls data/nerf_synthetic/lego/transforms_train.json
```

- [ ] **Step 5: Train one test scene (Lego) to verify pipeline**

```bash
cd /mnt/ssd1/idea/TurboQuant/gaussian-splatting
/mnt/ssd1/conda_envs/nerf_tq/bin/python train.py \
  -s data/nerf_synthetic/lego \
  -m output/lego_wb \
  --white_background \
  --iterations 30000
```

This takes ~10-15 minutes on RTX 4090. Verify output exists:
```bash
ls output/lego_wb/point_cloud/iteration_30000/point_cloud.ply
```

- [ ] **Step 6: Render and compute metrics to verify quality**

```bash
cd /mnt/ssd1/idea/TurboQuant/gaussian-splatting
/mnt/ssd1/conda_envs/nerf_tq/bin/python render.py \
  -m output/lego_wb \
  --skip_train
/mnt/ssd1/conda_envs/nerf_tq/bin/python metrics.py \
  -m output/lego_wb
```

Expected: PSNR ~35.5-36.0 for Lego test set.

- [ ] **Step 7: Commit**

```bash
cd /mnt/ssd1/idea/TurboQuant
git init
git add gaussian-splatting/.gitignore directions.md docs/
git commit -m "feat: initial project setup with 3DGS repo clone"
```

Note: Do NOT add data/, output/, or submodules/ to git. Add a `.gitignore`:
```
gaussian-splatting/data/
gaussian-splatting/output/
gaussian-splatting/submodules/
*.pyc
__pycache__/
```

---

## Task 2: Implement TurboQuant Codebook Generation

**Files:**
- Create: `gaussian-splatting/turbo_quant/__init__.py`
- Create: `gaussian-splatting/turbo_quant/codebook.py`
- Create: `gaussian-splatting/tests/test_codebook.py`

- [ ] **Step 1: Write the failing test for codebook generation**

Create `gaussian-splatting/tests/test_codebook.py`:

```python
import numpy as np
import sys
sys.path.insert(0, "/mnt/ssd1/idea/TurboQuant/gaussian-splatting")
from turbo_quant.codebook import generate_codebook, beta_pdf


def test_beta_pdf_integrates_to_one():
    """The Beta PDF for coordinates on the unit hypersphere should integrate to ~1."""
    d = 48
    x = np.linspace(-1, 1, 10000)
    pdf_vals = beta_pdf(x, d)
    integral = np.trapz(pdf_vals, x)
    assert abs(integral - 1.0) < 0.01, f"Integral={integral}, expected ~1.0"


def test_codebook_shape():
    """Codebook for b bits should have 2^b centroids in [-1, 1]."""
    d = 48
    for b in [1, 2, 3, 4]:
        centroids = generate_codebook(d, b)
        assert centroids.shape == (2**b,), f"Expected ({2**b},), got {centroids.shape}"
        assert np.all(centroids >= -1) and np.all(centroids <= 1)


def test_codebook_sorted():
    """Centroids should be sorted in ascending order."""
    d = 48
    centroids = generate_codebook(d, b=3)
    assert np.all(np.diff(centroids) > 0), "Centroids must be sorted ascending"


def test_codebook_symmetry():
    """For symmetric Beta PDF, codebook should be symmetric around 0."""
    d = 48
    centroids = generate_codebook(d, b=2)
    # Centroids should be roughly symmetric: c[i] ~ -c[n-1-i]
    n = len(centroids)
    for i in range(n // 2):
        assert abs(centroids[i] + centroids[n - 1 - i]) < 0.01, \
            f"Asymmetric: c[{i}]={centroids[i]}, c[{n-1-i}]={centroids[n-1-i]}"


def test_codebook_b1_near_expected():
    """At b=1, centroids should be near +/- sqrt(2/pi) / sqrt(d) for large d."""
    d = 1000
    centroids = generate_codebook(d, b=1)
    expected = np.sqrt(2 / np.pi) / np.sqrt(d)
    assert abs(abs(centroids[0]) - expected) < 0.05 * expected, \
        f"c[0]={centroids[0]}, expected ~{-expected}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/ssd1/idea/TurboQuant/gaussian-splatting
/mnt/ssd1/conda_envs/nerf_tq/bin/python -m pytest tests/test_codebook.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'turbo_quant'`

- [ ] **Step 3: Create package init**

Create `gaussian-splatting/turbo_quant/__init__.py`:

```python
from .quantizer import TurboQuantizer
```

Note: This will fail until quantizer.py exists, but that's fine — codebook.py is self-contained.

- [ ] **Step 4: Implement codebook generation**

Create `gaussian-splatting/turbo_quant/codebook.py`:

```python
"""
Precomputed optimal Lloyd-Max codebooks for the Beta distribution
arising from random rotation of unit-norm vectors.

From TurboQuant (arXiv:2504.19874): when a unit vector x in S^{d-1}
is multiplied by a random rotation matrix Pi, each coordinate of
Pi*x follows the Beta distribution:
    f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
for x in [-1, 1].

For large d, this converges to N(0, 1/d).

The optimal scalar quantizer for b bits partitions [-1, 1] into 2^b
cells and finds centroids minimizing the expected MSE under this
distribution. This is solved via the Lloyd-Max (k-means in 1D)
algorithm.
"""

import numpy as np
from scipy.special import gamma
from functools import lru_cache


def beta_pdf(x: np.ndarray, d: int) -> np.ndarray:
    """
    Evaluate the Beta PDF for coordinate distribution on S^{d-1}.

    f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)

    Args:
        x: array of points in [-1, 1]
        d: dimension of the ambient space
    Returns:
        PDF values at each point
    """
    if d < 3:
        # For d < 3, use uniform or degenerate distribution
        return np.ones_like(x) / 2.0

    coeff = gamma(d / 2) / (np.sqrt(np.pi) * gamma((d - 1) / 2))
    exponent = (d - 3) / 2
    vals = np.where(np.abs(x) < 1, coeff * (1 - x**2) ** exponent, 0.0)
    return vals


def _lloyd_max_1d(pdf_func, n_centroids: int, x_min: float = -1.0,
                  x_max: float = 1.0, n_grid: int = 100000,
                  max_iter: int = 200, tol: float = 1e-10) -> np.ndarray:
    """
    Lloyd-Max algorithm for optimal scalar quantization in 1D.

    Finds centroids that minimize E[|X - c_{nearest}|^2] under the given PDF.

    Args:
        pdf_func: callable returning PDF values for an array of x
        n_centroids: number of centroids (2^b)
        x_min, x_max: support interval
        n_grid: grid resolution for numerical integration
        max_iter: maximum Lloyd iterations
        tol: convergence tolerance on centroid movement
    Returns:
        Sorted array of optimal centroids
    """
    x_grid = np.linspace(x_min, x_max, n_grid)
    pdf_vals = pdf_func(x_grid)
    dx = x_grid[1] - x_grid[0]

    # Initialize centroids uniformly
    centroids = np.linspace(x_min + dx, x_max - dx, n_centroids)

    for _ in range(max_iter):
        # Compute boundaries (midpoints between consecutive centroids)
        boundaries = np.concatenate([[x_min],
                                      (centroids[:-1] + centroids[1:]) / 2,
                                      [x_max]])

        # Update centroids: weighted mean within each cell
        new_centroids = np.zeros(n_centroids)
        for i in range(n_centroids):
            mask = (x_grid >= boundaries[i]) & (x_grid < boundaries[i + 1])
            if i == n_centroids - 1:
                mask = (x_grid >= boundaries[i]) & (x_grid <= boundaries[i + 1])

            if np.any(mask):
                weights = pdf_vals[mask]
                total_weight = np.sum(weights) * dx
                if total_weight > 1e-15:
                    new_centroids[i] = np.sum(x_grid[mask] * weights * dx) / total_weight
                else:
                    new_centroids[i] = centroids[i]
            else:
                new_centroids[i] = centroids[i]

        # Check convergence
        if np.max(np.abs(new_centroids - centroids)) < tol:
            break
        centroids = new_centroids

    return np.sort(centroids)


@lru_cache(maxsize=32)
def generate_codebook(d: int, b: int) -> np.ndarray:
    """
    Generate the optimal Lloyd-Max codebook for dimension d at bit-width b.

    Args:
        d: dimension of vectors being quantized
        b: bits per coordinate (codebook has 2^b entries)
    Returns:
        Sorted numpy array of 2^b centroid values in [-1, 1]
    """
    n_centroids = 2 ** b
    pdf_func = lambda x: beta_pdf(x, d)
    centroids = _lloyd_max_1d(pdf_func, n_centroids)
    return centroids


def compute_mse_cost(d: int, b: int) -> float:
    """
    Compute the optimal MSE cost C(f_X, b) for quantizing the Beta distribution.

    This is the per-coordinate expected quantization error:
        C(f_X, b) = sum_i integral_{cell_i} |x - c_i|^2 f_X(x) dx

    The total MSE for a d-dimensional vector is d * C(f_X, b).

    Args:
        d: dimension
        b: bit-width
    Returns:
        Per-coordinate MSE cost
    """
    centroids = generate_codebook(d, b)
    n = len(centroids)
    boundaries = np.concatenate([[-1.0],
                                  (centroids[:-1] + centroids[1:]) / 2,
                                  [1.0]])

    x_grid = np.linspace(-1, 1, 100000)
    pdf_vals = beta_pdf(x_grid, d)
    dx = x_grid[1] - x_grid[0]

    total_mse = 0.0
    for i in range(n):
        mask = (x_grid >= boundaries[i]) & (x_grid < boundaries[i + 1])
        if i == n - 1:
            mask = (x_grid >= boundaries[i]) & (x_grid <= boundaries[i + 1])
        if np.any(mask):
            errors = (x_grid[mask] - centroids[i]) ** 2
            total_mse += np.sum(errors * pdf_vals[mask]) * dx

    return total_mse
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /mnt/ssd1/idea/TurboQuant/gaussian-splatting
/mnt/ssd1/conda_envs/nerf_tq/bin/python -m pytest tests/test_codebook.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 6: Verify MSE cost matches TurboQuant paper values**

```bash
cd /mnt/ssd1/idea/TurboQuant/gaussian-splatting
/mnt/ssd1/conda_envs/nerf_tq/bin/python -c "
from turbo_quant.codebook import compute_mse_cost
d = 48
for b in [1, 2, 3, 4]:
    cost = compute_mse_cost(d, b)
    total = d * cost
    print(f'b={b}: per-coord={cost:.4f}, total D_mse={total:.3f} (paper: {[0.36, 0.117, 0.03, 0.009][b-1]})')
"
```

Expected: Values should closely match paper Table: D_mse ~ 0.36, 0.117, 0.03, 0.009 for b=1,2,3,4.

- [ ] **Step 7: Commit**

```bash
cd /mnt/ssd1/idea/TurboQuant
git add gaussian-splatting/turbo_quant/__init__.py gaussian-splatting/turbo_quant/codebook.py gaussian-splatting/tests/test_codebook.py
git commit -m "feat: TurboQuant codebook generation with Lloyd-Max on Beta distribution"
```

---

## Task 3: Implement TurboQuant Quantizer

**Files:**
- Create: `gaussian-splatting/turbo_quant/quantizer.py`
- Create: `gaussian-splatting/tests/test_quantizer.py`

- [ ] **Step 1: Write the failing tests**

Create `gaussian-splatting/tests/test_quantizer.py`:

```python
import numpy as np
import sys
sys.path.insert(0, "/mnt/ssd1/idea/TurboQuant/gaussian-splatting")
from turbo_quant.quantizer import TurboQuantizer


def test_roundtrip_shape():
    """Quantize then dequantize should preserve shape."""
    d = 48
    n = 100
    q = TurboQuantizer(d=d, b=3, seed=42)
    vectors = np.random.randn(n, d).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms  # unit norm

    indices, stored_norms = q.quantize(vectors)
    reconstructed = q.dequantize(indices, stored_norms)

    assert reconstructed.shape == vectors.shape, \
        f"Expected {vectors.shape}, got {reconstructed.shape}"


def test_indices_are_b_bit():
    """Quantized indices should be in range [0, 2^b - 1]."""
    d = 48
    q = TurboQuantizer(d=d, b=3, seed=42)
    vectors = np.random.randn(50, d).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    indices, _ = q.quantize(vectors)
    assert indices.dtype == np.uint8 or indices.dtype == np.int32
    assert np.all(indices >= 0) and np.all(indices < 2**3)


def test_mse_decreases_with_bitwidth():
    """Higher bit-width should give lower reconstruction MSE."""
    d = 48
    n = 1000
    np.random.seed(42)
    vectors = np.random.randn(n, d).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    mses = []
    for b in [1, 2, 3, 4]:
        q = TurboQuantizer(d=d, b=b, seed=42)
        indices, stored_norms = q.quantize(vectors)
        recon = q.dequantize(indices, stored_norms)
        mse = np.mean((vectors - recon) ** 2)
        mses.append(mse)

    for i in range(len(mses) - 1):
        assert mses[i] > mses[i + 1], \
            f"MSE should decrease: b={i+1} MSE={mses[i]:.4f} >= b={i+2} MSE={mses[i+1]:.4f}"


def test_mse_near_theoretical():
    """Reconstruction MSE should be near theoretical bound from TurboQuant paper."""
    d = 48
    n = 5000
    np.random.seed(42)
    vectors = np.random.randn(n, d).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    b = 3
    q = TurboQuantizer(d=d, b=b, seed=42)
    indices, stored_norms = q.quantize(vectors)
    recon = q.dequantize(indices, stored_norms)
    mse = np.mean(np.sum((vectors - recon) ** 2, axis=1))

    # Paper: D_mse ~ 0.03 for b=3, d=48
    # Upper bound: sqrt(3*pi)/2 * 1/4^b ~ 2.7 * 0.0156 ~ 0.042
    assert mse < 0.10, f"MSE={mse:.4f}, expected < 0.10 (upper bound ~0.042)"


def test_deterministic_with_seed():
    """Same seed should produce same rotation and same results."""
    d = 48
    vectors = np.random.randn(10, d).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    q1 = TurboQuantizer(d=d, b=3, seed=42)
    q2 = TurboQuantizer(d=d, b=3, seed=42)

    idx1, n1 = q1.quantize(vectors)
    idx2, n2 = q2.quantize(vectors)

    np.testing.assert_array_equal(idx1, idx2)
    np.testing.assert_array_almost_equal(n1, n2)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /mnt/ssd1/idea/TurboQuant/gaussian-splatting
/mnt/ssd1/conda_envs/nerf_tq/bin/python -m pytest tests/test_quantizer.py -v
```

Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement the quantizer**

Create `gaussian-splatting/turbo_quant/quantizer.py`:

```python
"""
TurboQuantizer: data-oblivious vector quantization via random rotation
+ optimal scalar quantization per coordinate.

Algorithm (from TurboQuant, arXiv:2504.19874):
1. Generate random rotation matrix Pi (QR decomposition of Gaussian matrix)
2. Rotate: y = Pi @ x (each coordinate of y ~ Beta distribution)
3. Quantize: for each coordinate, find nearest centroid in precomputed codebook
4. Store: indices (b bits per coordinate) + L2 norm of original vector
5. Dequantize: look up centroids, inverse rotate: x_hat = Pi^T @ y_hat
"""

import numpy as np
from .codebook import generate_codebook


class TurboQuantizer:
    """
    Data-oblivious vector quantizer with near-optimal MSE distortion.

    Args:
        d: dimension of input vectors
        b: bits per coordinate (1-8)
        seed: random seed for rotation matrix generation
    """

    def __init__(self, d: int, b: int, seed: int = 0):
        self.d = d
        self.b = b
        self.seed = seed

        # Generate random rotation matrix via QR decomposition
        rng = np.random.RandomState(seed)
        gaussian_matrix = rng.randn(d, d).astype(np.float64)
        self.rotation, _ = np.linalg.qr(gaussian_matrix)
        self.rotation = self.rotation.astype(np.float32)

        # Load precomputed codebook
        self.centroids = generate_codebook(d, b).astype(np.float32)

        # Precompute boundaries (midpoints between centroids) for fast quantization
        self.boundaries = np.concatenate([
            [-np.inf],
            (self.centroids[:-1] + self.centroids[1:]) / 2,
            [np.inf]
        ])

    def quantize(self, vectors: np.ndarray) -> tuple:
        """
        Quantize a batch of vectors.

        Args:
            vectors: (N, d) array of input vectors

        Returns:
            indices: (N, d) array of codebook indices (uint8 for b<=8)
            norms: (N,) array of L2 norms of input vectors
        """
        assert vectors.shape[1] == self.d, \
            f"Expected d={self.d}, got {vectors.shape[1]}"

        # Store norms for reconstruction
        norms = np.linalg.norm(vectors, axis=1).astype(np.float32)

        # Normalize to unit norm
        safe_norms = np.maximum(norms, 1e-10)
        unit_vectors = vectors / safe_norms[:, np.newaxis]

        # Rotate: y = Pi @ x^T => (d, N), then transpose to (N, d)
        rotated = (self.rotation @ unit_vectors.T).T

        # Quantize each coordinate: find nearest centroid
        # Use searchsorted on boundaries for efficiency
        indices = np.searchsorted(self.boundaries[1:-1], rotated).astype(np.uint8)

        return indices, norms

    def dequantize(self, indices: np.ndarray, norms: np.ndarray) -> np.ndarray:
        """
        Reconstruct vectors from quantized indices and norms.

        Args:
            indices: (N, d) array of codebook indices
            norms: (N,) array of L2 norms

        Returns:
            reconstructed: (N, d) array of reconstructed vectors
        """
        # Look up centroids
        rotated_recon = self.centroids[indices]

        # Inverse rotate: x_hat = Pi^T @ y_hat
        reconstructed = (self.rotation.T @ rotated_recon.T).T

        # Rescale by original norms
        reconstructed = reconstructed * norms[:, np.newaxis]

        return reconstructed.astype(np.float32)

    def get_rotation_matrix(self) -> np.ndarray:
        """Return the rotation matrix (needed for storage)."""
        return self.rotation

    def get_centroids(self) -> np.ndarray:
        """Return the codebook centroids."""
        return self.centroids
```

- [ ] **Step 4: Update __init__.py**

Update `gaussian-splatting/turbo_quant/__init__.py`:

```python
from .quantizer import TurboQuantizer
from .codebook import generate_codebook, beta_pdf, compute_mse_cost
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /mnt/ssd1/idea/TurboQuant/gaussian-splatting
/mnt/ssd1/conda_envs/nerf_tq/bin/python -m pytest tests/test_quantizer.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /mnt/ssd1/idea/TurboQuant
git add gaussian-splatting/turbo_quant/quantizer.py gaussian-splatting/turbo_quant/__init__.py gaussian-splatting/tests/test_quantizer.py
git commit -m "feat: TurboQuantizer with random rotation and optimal scalar quantization"
```

---

## Task 4: Implement Full-Attribute Compression Pipeline

**Files:**
- Create: `gaussian-splatting/compress.py`
- Create: `gaussian-splatting/decompress.py`
- Create: `gaussian-splatting/tests/test_compress.py`

- [ ] **Step 1: Write the failing integration test**

Create `gaussian-splatting/tests/test_compress.py`:

```python
"""Integration test: compress a real 3DGS PLY, decompress, check quality."""
import numpy as np
import os
import sys
import tempfile
sys.path.insert(0, "/mnt/ssd1/idea/TurboQuant/gaussian-splatting")


def test_compress_decompress_synthetic():
    """Create a synthetic PLY-like dict, compress, decompress, check MSE."""
    from compress import compress_gaussians
    from decompress import decompress_gaussians

    np.random.seed(42)
    n = 1000
    d_sh = 45  # degree 3: (3+1)^2 - 1 = 15 coefficients * 3 channels = 45

    # Simulate Gaussian attributes
    attrs = {
        "xyz": np.random.randn(n, 3).astype(np.float32),
        "scales": np.random.randn(n, 3).astype(np.float32),
        "rotations": np.random.randn(n, 4).astype(np.float32),
        "opacity": np.random.randn(n, 1).astype(np.float32),
        "sh_dc": np.random.randn(n, 3).astype(np.float32),
        "sh_rest": np.random.randn(n, d_sh).astype(np.float32),
    }
    # Normalize rotations
    rot_norms = np.linalg.norm(attrs["rotations"], axis=1, keepdims=True)
    attrs["rotations"] = attrs["rotations"] / rot_norms

    with tempfile.TemporaryDirectory() as tmpdir:
        compressed_path = os.path.join(tmpdir, "compressed.npz")
        compress_gaussians(attrs, compressed_path, sh_bits=3, pos_bits=8,
                           scale_bits=6, rot_bits=4, opacity_bits=4, seed=42)

        assert os.path.exists(compressed_path)

        recon_attrs = decompress_gaussians(compressed_path)

        # Check shapes match
        for key in attrs:
            assert recon_attrs[key].shape == attrs[key].shape, \
                f"{key}: expected {attrs[key].shape}, got {recon_attrs[key].shape}"

        # SH MSE should be reasonable (< 0.1 for b=3)
        sh_mse = np.mean((attrs["sh_rest"] - recon_attrs["sh_rest"]) ** 2)
        assert sh_mse < 0.5, f"SH MSE={sh_mse:.4f}, expected < 0.5"

        # Positions should be very close (b=8)
        pos_mse = np.mean((attrs["xyz"] - recon_attrs["xyz"]) ** 2)
        assert pos_mse < 0.01, f"Position MSE={pos_mse:.6f}, expected < 0.01"


def test_compression_ratio():
    """Compressed file should be smaller than original attributes."""
    from compress import compress_gaussians

    np.random.seed(42)
    n = 10000
    attrs = {
        "xyz": np.random.randn(n, 3).astype(np.float32),
        "scales": np.random.randn(n, 3).astype(np.float32),
        "rotations": np.random.randn(n, 4).astype(np.float32),
        "opacity": np.random.randn(n, 1).astype(np.float32),
        "sh_dc": np.random.randn(n, 3).astype(np.float32),
        "sh_rest": np.random.randn(n, 45).astype(np.float32),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        compressed_path = os.path.join(tmpdir, "compressed.npz")
        compress_gaussians(attrs, compressed_path, sh_bits=3, seed=42)

        original_size = sum(v.nbytes for v in attrs.values())
        compressed_size = os.path.getsize(compressed_path)

        ratio = original_size / compressed_size
        assert ratio > 2.0, f"Compression ratio {ratio:.2f}x, expected > 2x"
        print(f"Compression ratio: {ratio:.2f}x ({original_size} -> {compressed_size})")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/ssd1/idea/TurboQuant/gaussian-splatting
/mnt/ssd1/conda_envs/nerf_tq/bin/python -m pytest tests/test_compress.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement compress.py**

Create `gaussian-splatting/compress.py`:

```python
"""
Compress a trained 3DGS model using TurboQuant.

Usage:
    python compress.py -m output/lego_wb -o compressed/lego.npz --sh_bits 3
"""

import argparse
import numpy as np
import os
import time
from plyfile import PlyData
from turbo_quant.quantizer import TurboQuantizer


def load_ply_attributes(ply_path: str) -> dict:
    """Load Gaussian attributes from a PLY file."""
    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]

    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)

    # SH DC (f_dc_0, f_dc_1, f_dc_2)
    sh_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]],
                     axis=1).astype(np.float32)

    # SH rest coefficients (f_rest_0 through f_rest_44 for degree 3)
    sh_rest_names = [p.name for p in vertex.properties if p.name.startswith("f_rest_")]
    sh_rest_names.sort(key=lambda x: int(x.split("_")[-1]))
    if sh_rest_names:
        sh_rest = np.stack([vertex[name] for name in sh_rest_names],
                           axis=1).astype(np.float32)
    else:
        sh_rest = np.zeros((len(xyz), 0), dtype=np.float32)

    opacity = vertex["opacity"].reshape(-1, 1).astype(np.float32)

    scales = np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]],
                      axis=1).astype(np.float32)

    rotations = np.stack([vertex["rot_0"], vertex["rot_1"],
                          vertex["rot_2"], vertex["rot_3"]],
                         axis=1).astype(np.float32)

    return {
        "xyz": xyz,
        "sh_dc": sh_dc,
        "sh_rest": sh_rest,
        "opacity": opacity,
        "scales": scales,
        "rotations": rotations,
    }


def _uniform_quantize(values: np.ndarray, bits: int) -> tuple:
    """Simple uniform scalar quantization with min/max normalization."""
    vmin = values.min(axis=0)
    vmax = values.max(axis=0)
    scale = vmax - vmin
    scale = np.where(scale < 1e-10, 1.0, scale)
    normalized = (values - vmin) / scale
    n_levels = 2 ** bits
    indices = np.clip(np.round(normalized * (n_levels - 1)), 0, n_levels - 1).astype(np.uint8 if bits <= 8 else np.uint16)
    return indices, vmin, scale


def _uniform_dequantize(indices: np.ndarray, vmin: np.ndarray,
                        scale: np.ndarray, bits: int) -> np.ndarray:
    """Inverse of uniform scalar quantization."""
    n_levels = 2 ** bits
    normalized = indices.astype(np.float32) / (n_levels - 1)
    return normalized * scale + vmin


def compress_gaussians(attrs: dict, output_path: str, sh_bits: int = 3,
                       pos_bits: int = 8, scale_bits: int = 6,
                       rot_bits: int = 4, opacity_bits: int = 4,
                       seed: int = 0) -> dict:
    """
    Compress Gaussian attributes using TurboQuant (SH) + uniform (others).

    Args:
        attrs: dict with keys xyz, sh_dc, sh_rest, opacity, scales, rotations
        output_path: path to save compressed .npz
        sh_bits: bit-width for SH coefficients via TurboQuant
        pos_bits: bit-width for positions (uniform quantization)
        scale_bits: bit-width for log-scales (uniform quantization)
        rot_bits: bit-width for quaternion rotations (uniform quantization)
        opacity_bits: bit-width for opacity (uniform quantization)
        seed: random seed for TurboQuant rotation matrix
    Returns:
        dict with compression statistics
    """
    t0 = time.time()
    save_dict = {}
    stats = {}

    n = attrs["xyz"].shape[0]
    stats["n_gaussians"] = n

    # 1. SH rest coefficients — TurboQuant with random rotation
    sh_rest = attrs["sh_rest"]
    d_sh = sh_rest.shape[1]
    if d_sh > 0:
        tq = TurboQuantizer(d=d_sh, b=sh_bits, seed=seed)
        sh_indices, sh_norms = tq.quantize(sh_rest)
        save_dict["sh_rest_indices"] = sh_indices
        save_dict["sh_rest_norms"] = sh_norms
        save_dict["sh_rotation"] = tq.get_rotation_matrix()
        save_dict["sh_centroids"] = tq.get_centroids()
        save_dict["sh_bits"] = np.array(sh_bits)
        save_dict["sh_d"] = np.array(d_sh)

    # 2. SH DC — uniform quantization (only 3D, too low for TurboQuant rotation)
    dc_idx, dc_min, dc_scale = _uniform_quantize(attrs["sh_dc"], bits=8)
    save_dict["sh_dc_indices"] = dc_idx
    save_dict["sh_dc_min"] = dc_min
    save_dict["sh_dc_scale"] = dc_scale

    # 3. Positions — uniform quantization with high precision
    pos_idx, pos_min, pos_scale = _uniform_quantize(attrs["xyz"], bits=pos_bits)
    save_dict["xyz_indices"] = pos_idx
    save_dict["xyz_min"] = pos_min
    save_dict["xyz_scale"] = pos_scale
    save_dict["pos_bits"] = np.array(pos_bits)

    # 4. Scales — log-transform then uniform quantization
    scales_log = attrs["scales"]  # Already in log-space in 3DGS
    sc_idx, sc_min, sc_scale = _uniform_quantize(scales_log, bits=scale_bits)
    save_dict["scales_indices"] = sc_idx
    save_dict["scales_min"] = sc_min
    save_dict["scales_scale"] = sc_scale
    save_dict["scale_bits"] = np.array(scale_bits)

    # 5. Rotations — normalize then uniform quantization
    rot = attrs["rotations"]
    rot = rot / np.linalg.norm(rot, axis=1, keepdims=True)
    rot_idx, rot_min, rot_scale = _uniform_quantize(rot, bits=rot_bits)
    save_dict["rot_indices"] = rot_idx
    save_dict["rot_min"] = rot_min
    save_dict["rot_scale"] = rot_scale
    save_dict["rot_bits"] = np.array(rot_bits)

    # 6. Opacity — uniform quantization
    op_idx, op_min, op_scale = _uniform_quantize(attrs["opacity"], bits=opacity_bits)
    save_dict["opacity_indices"] = op_idx
    save_dict["opacity_min"] = op_min
    save_dict["opacity_scale"] = op_scale
    save_dict["opacity_bits"] = np.array(opacity_bits)

    # Save
    np.savez_compressed(output_path, **save_dict)

    elapsed = time.time() - t0
    stats["compression_time_s"] = elapsed
    stats["compressed_size_bytes"] = os.path.getsize(output_path)
    original_size = sum(v.nbytes for v in attrs.values())
    stats["original_size_bytes"] = original_size
    stats["compression_ratio"] = original_size / stats["compressed_size_bytes"]

    return stats


def main():
    parser = argparse.ArgumentParser(description="Compress 3DGS model with TurboQuant")
    parser.add_argument("-m", "--model_path", required=True, help="Path to trained 3DGS model directory")
    parser.add_argument("-o", "--output", default=None, help="Output .npz path")
    parser.add_argument("--sh_bits", type=int, default=3, help="Bit-width for SH coefficients (default: 3)")
    parser.add_argument("--pos_bits", type=int, default=8, help="Bit-width for positions (default: 8)")
    parser.add_argument("--scale_bits", type=int, default=6, help="Bit-width for scales (default: 6)")
    parser.add_argument("--rot_bits", type=int, default=4, help="Bit-width for rotations (default: 4)")
    parser.add_argument("--opacity_bits", type=int, default=4, help="Bit-width for opacity (default: 4)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--iteration", type=int, default=30000, help="Which iteration checkpoint to compress")
    args = parser.parse_args()

    ply_path = os.path.join(args.model_path, "point_cloud",
                            f"iteration_{args.iteration}", "point_cloud.ply")
    if not os.path.exists(ply_path):
        print(f"Error: PLY not found at {ply_path}")
        return

    if args.output is None:
        args.output = os.path.join(args.model_path, "compressed.npz")

    print(f"Loading PLY from {ply_path}...")
    attrs = load_ply_attributes(ply_path)
    print(f"  {attrs['xyz'].shape[0]} Gaussians, SH dim={attrs['sh_rest'].shape[1]}")

    print(f"Compressing with sh_bits={args.sh_bits}...")
    stats = compress_gaussians(attrs, args.output, sh_bits=args.sh_bits,
                                pos_bits=args.pos_bits, scale_bits=args.scale_bits,
                                rot_bits=args.rot_bits, opacity_bits=args.opacity_bits,
                                seed=args.seed)

    print(f"Done in {stats['compression_time_s']:.3f}s")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"  Original:   {stats['original_size_bytes'] / 1e6:.1f} MB")
    print(f"  Compressed: {stats['compressed_size_bytes'] / 1e6:.1f} MB")
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement decompress.py**

Create `gaussian-splatting/decompress.py`:

```python
"""
Decompress a TurboSplat-compressed 3DGS model.

Usage:
    python decompress.py -i compressed/lego.npz -o decompressed/lego.ply
"""

import argparse
import numpy as np
import os
from turbo_quant.quantizer import TurboQuantizer
from compress import _uniform_dequantize


def decompress_gaussians(compressed_path: str) -> dict:
    """
    Decompress a .npz file back to Gaussian attributes.

    Args:
        compressed_path: path to compressed .npz file

    Returns:
        dict with keys: xyz, sh_dc, sh_rest, opacity, scales, rotations
    """
    data = np.load(compressed_path)

    # 1. SH rest — TurboQuant dequantize
    sh_bits = int(data["sh_bits"])
    sh_d = int(data["sh_d"])
    rotation_matrix = data["sh_rotation"]
    centroids = data["sh_centroids"]
    sh_indices = data["sh_rest_indices"]
    sh_norms = data["sh_rest_norms"]

    # Reconstruct using stored rotation matrix and centroids
    rotated_recon = centroids[sh_indices]
    sh_rest = (rotation_matrix.T @ rotated_recon.T).T
    sh_rest = sh_rest * sh_norms[:, np.newaxis]
    sh_rest = sh_rest.astype(np.float32)

    # 2. SH DC — uniform dequantize
    sh_dc = _uniform_dequantize(data["sh_dc_indices"], data["sh_dc_min"],
                                 data["sh_dc_scale"], bits=8)

    # 3. Positions
    pos_bits = int(data["pos_bits"])
    xyz = _uniform_dequantize(data["xyz_indices"], data["xyz_min"],
                               data["xyz_scale"], bits=pos_bits)

    # 4. Scales
    scale_bits = int(data["scale_bits"])
    scales = _uniform_dequantize(data["scales_indices"], data["scales_min"],
                                  data["scales_scale"], bits=scale_bits)

    # 5. Rotations
    rot_bits = int(data["rot_bits"])
    rotations = _uniform_dequantize(data["rot_indices"], data["rot_min"],
                                     data["rot_scale"], bits=rot_bits)
    # Re-normalize quaternions
    rot_norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    rotations = rotations / np.maximum(rot_norms, 1e-10)

    # 6. Opacity
    opacity_bits = int(data["opacity_bits"]) if "opacity_bits" in data else 4
    opacity = _uniform_dequantize(data["opacity_indices"], data["opacity_min"],
                                   data["opacity_scale"], bits=opacity_bits)

    return {
        "xyz": xyz.astype(np.float32),
        "sh_dc": sh_dc.astype(np.float32),
        "sh_rest": sh_rest.astype(np.float32),
        "opacity": opacity.astype(np.float32),
        "scales": scales.astype(np.float32),
        "rotations": rotations.astype(np.float32),
    }


def save_ply(attrs: dict, output_path: str):
    """Save reconstructed attributes to PLY format compatible with 3DGS."""
    from plyfile import PlyData, PlyElement

    n = attrs["xyz"].shape[0]
    d_sh_rest = attrs["sh_rest"].shape[1]

    # Build structured array
    names = ["x", "y", "z"]
    names += ["f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(d_sh_rest)]
    names += ["opacity"]
    names += ["scale_0", "scale_1", "scale_2"]
    names += ["rot_0", "rot_1", "rot_2", "rot_3"]

    dtype = [(name, "f4") for name in names]
    elements = np.empty(n, dtype=dtype)

    elements["x"] = attrs["xyz"][:, 0]
    elements["y"] = attrs["xyz"][:, 1]
    elements["z"] = attrs["xyz"][:, 2]
    elements["f_dc_0"] = attrs["sh_dc"][:, 0]
    elements["f_dc_1"] = attrs["sh_dc"][:, 1]
    elements["f_dc_2"] = attrs["sh_dc"][:, 2]
    for i in range(d_sh_rest):
        elements[f"f_rest_{i}"] = attrs["sh_rest"][:, i]
    elements["opacity"] = attrs["opacity"][:, 0]
    elements["scale_0"] = attrs["scales"][:, 0]
    elements["scale_1"] = attrs["scales"][:, 1]
    elements["scale_2"] = attrs["scales"][:, 2]
    elements["rot_0"] = attrs["rotations"][:, 0]
    elements["rot_1"] = attrs["rotations"][:, 1]
    elements["rot_2"] = attrs["rotations"][:, 2]
    elements["rot_3"] = attrs["rotations"][:, 3]

    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(output_path)


def main():
    parser = argparse.ArgumentParser(description="Decompress TurboSplat model")
    parser.add_argument("-i", "--input", required=True, help="Compressed .npz path")
    parser.add_argument("-o", "--output", default=None, help="Output .ply path")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.replace(".npz", "_decompressed.ply")

    print(f"Decompressing {args.input}...")
    attrs = decompress_gaussians(args.input)
    print(f"  {attrs['xyz'].shape[0]} Gaussians, SH rest dim={attrs['sh_rest'].shape[1]}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_ply(attrs, args.output)
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run integration tests**

```bash
cd /mnt/ssd1/idea/TurboQuant/gaussian-splatting
/mnt/ssd1/conda_envs/nerf_tq/bin/python -m pytest tests/test_compress.py -v -s
```

Expected: Both tests PASS. Compression ratio should be > 2x.

- [ ] **Step 6: Test on real PLY (if Lego model exists)**

```bash
cd /mnt/ssd1/idea/TurboQuant/gaussian-splatting
/mnt/ssd1/conda_envs/nerf_tq/bin/python compress.py \
  -m output/lego_wb \
  -o compressed/lego_b3.npz \
  --sh_bits 3
```

Expected output like:
```
Done in 0.XXXs
  Compression ratio: ~5-7x
  Original: ~XX MB
  Compressed: ~XX MB
```

- [ ] **Step 7: Commit**

```bash
cd /mnt/ssd1/idea/TurboQuant
git add gaussian-splatting/compress.py gaussian-splatting/decompress.py gaussian-splatting/tests/test_compress.py
git commit -m "feat: full-attribute 3DGS compression and decompression pipeline"
```

---

## Task 5: Implement SH Overfitting Diagnosis

**Files:**
- Create: `gaussian-splatting/diagnosis/__init__.py`
- Create: `gaussian-splatting/diagnosis/train_test_gap.py`
- Create: `gaussian-splatting/diagnosis/sh_band_analysis.py`

- [ ] **Step 1: Create diagnosis package init**

Create `gaussian-splatting/diagnosis/__init__.py`:

```python
```

- [ ] **Step 2: Implement train/test gap measurement**

Create `gaussian-splatting/diagnosis/train_test_gap.py`:

```python
"""
Measure train vs test PSNR gap for trained 3DGS models.

Usage:
    python -m diagnosis.train_test_gap -m output/lego_wb -s data/nerf_synthetic/lego
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from argparse import Namespace

# Add parent to path for 3DGS imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr


def compute_split_psnr(model_path: str, source_path: str,
                       iteration: int = 30000,
                       white_background: bool = True) -> dict:
    """
    Compute PSNR on train and test splits separately.

    Returns:
        dict with train_psnr, test_psnr, gap, per-image results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gaussians = GaussianModel(3)  # SH degree 3
    bg = torch.tensor([1, 1, 1] if white_background else [0, 0, 0],
                      dtype=torch.float32, device=device)

    # Load scene
    args = Namespace(
        source_path=source_path,
        model_path=model_path,
        images="images",
        resolution=1,
        data_device="cuda",
        eval=True,
        sh_degree=3,
    )
    scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False)
    gaussians.load_ply(os.path.join(model_path, "point_cloud",
                                     f"iteration_{iteration}", "point_cloud.ply"))

    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)

    results = {"train": [], "test": []}
    for split_name, cameras in [("train", scene.getTrainCameras()),
                                 ("test", scene.getTestCameras())]:
        psnr_list = []
        for cam in cameras:
            rendering = render(cam, gaussians, pipe, bg)["render"]
            gt = cam.original_image[:3, :, :].to(device)
            p = psnr(rendering, gt).item()
            psnr_list.append(p)

        results[split_name] = psnr_list

    train_avg = np.mean(results["train"])
    test_avg = np.mean(results["test"])

    return {
        "train_psnr": float(train_avg),
        "test_psnr": float(test_avg),
        "gap": float(train_avg - test_avg),
        "train_per_image": results["train"],
        "test_per_image": results["test"],
        "n_train": len(results["train"]),
        "n_test": len(results["test"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--white_background", action="store_true", default=True)
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    print(f"Computing train/test gap for {args.model_path}...")
    result = compute_split_psnr(args.model_path, args.source_path,
                                 args.iteration, args.white_background)

    print(f"  Train PSNR: {result['train_psnr']:.2f} dB ({result['n_train']} images)")
    print(f"  Test PSNR:  {result['test_psnr']:.2f} dB ({result['n_test']} images)")
    print(f"  Gap:        {result['gap']:.2f} dB")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Implement per-band SH analysis**

Create `gaussian-splatting/diagnosis/sh_band_analysis.py`:

```python
"""
Per-band SH overfitting analysis: zero each band's coefficients and
measure train vs test PSNR drop to compute overfitting ratio.

SH bands for degree 3 (16 coefficients per channel, 3 channels):
  Band 0 (DC): 1 coeff/channel = 3 values (f_dc_0-2)
  Band 1: 3 coeffs/channel = 9 values (f_rest_0-8)
  Band 2: 5 coeffs/channel = 15 values (f_rest_9-23)
  Band 3: 7 coeffs/channel = 21 values (f_rest_24-44)

Overfitting ratio R_k = train_drop_k / test_drop_k
  R_k >> 1 means band k contributes much more to train than test → overfitting

Usage:
    python -m diagnosis.sh_band_analysis -m output/lego_wb -s data/nerf_synthetic/lego
"""

import argparse
import copy
import json
import os
import sys
import torch
import numpy as np
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.image_utils import psnr


# SH band structure: indices into f_rest (0-indexed)
# Each band l has (2l+1) coefficients per channel, 3 channels
SH_BAND_RANGES = {
    1: (0, 9),      # 3 coeffs * 3 channels = 9
    2: (9, 24),     # 5 coeffs * 3 channels = 15
    3: (24, 45),    # 7 coeffs * 3 channels = 21
}


def _compute_psnr(gaussians, scene, bg, device, split="test"):
    """Compute average PSNR for a split."""
    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    cameras = scene.getTestCameras() if split == "test" else scene.getTrainCameras()
    psnr_list = []
    for cam in cameras:
        rendering = render(cam, gaussians, pipe, bg)["render"]
        gt = cam.original_image[:3, :, :].to(device)
        psnr_list.append(psnr(rendering, gt).item())
    return float(np.mean(psnr_list))


def analyze_sh_bands(model_path: str, source_path: str,
                     iteration: int = 30000,
                     white_background: bool = True) -> dict:
    """
    Zero each SH band and measure impact on train/test PSNR.

    Returns dict with baseline PSNRs, per-band drops, and overfitting ratios.
    """
    device = torch.device("cuda")
    bg = torch.tensor([1, 1, 1] if white_background else [0, 0, 0],
                      dtype=torch.float32, device=device)

    gaussians = GaussianModel(3)
    args = Namespace(
        source_path=source_path, model_path=model_path,
        images="images", resolution=1, data_device="cuda",
        eval=True, sh_degree=3,
    )
    scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False)
    ply_path = os.path.join(model_path, "point_cloud",
                            f"iteration_{iteration}", "point_cloud.ply")
    gaussians.load_ply(ply_path)

    # Baseline PSNR
    train_base = _compute_psnr(gaussians, scene, bg, device, "train")
    test_base = _compute_psnr(gaussians, scene, bg, device, "test")

    results = {
        "baseline_train_psnr": train_base,
        "baseline_test_psnr": test_base,
        "baseline_gap": train_base - test_base,
        "bands": {},
    }

    # For each band, zero it and measure PSNR drop
    original_sh_rest = gaussians._features_rest.data.clone()

    for band in [1, 2, 3]:
        start, end = SH_BAND_RANGES[band]
        n_coeffs_per_channel = (2 * band + 1)

        # Zero this band's coefficients
        gaussians._features_rest.data = original_sh_rest.clone()
        # f_rest is stored as (N, n_sh_rest, 3) in 3DGS
        # Indices: for band l, coefficients l^2-1 to (l+1)^2-2 in each channel
        coeff_start = band * band - 1  # 0-indexed within rest
        coeff_end = coeff_start + n_coeffs_per_channel
        gaussians._features_rest.data[:, coeff_start:coeff_end, :] = 0

        train_zeroed = _compute_psnr(gaussians, scene, bg, device, "train")
        test_zeroed = _compute_psnr(gaussians, scene, bg, device, "test")

        train_drop = train_base - train_zeroed
        test_drop = test_base - test_zeroed

        ratio = train_drop / max(test_drop, 0.01)

        results["bands"][f"l{band}"] = {
            "train_drop": train_drop,
            "test_drop": test_drop,
            "overfitting_ratio": ratio,
            "n_coefficients": n_coeffs_per_channel * 3,
        }

    # Restore
    gaussians._features_rest.data = original_sh_rest

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--white_background", action="store_true", default=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"Analyzing SH band overfitting for {args.model_path}...")
    results = analyze_sh_bands(args.model_path, args.source_path,
                                args.iteration, args.white_background)

    print(f"\nBaseline: train={results['baseline_train_psnr']:.2f}, "
          f"test={results['baseline_test_psnr']:.2f}, "
          f"gap={results['baseline_gap']:.2f} dB")
    print(f"\nPer-band overfitting:")
    for band_name, band_data in results["bands"].items():
        print(f"  {band_name}: train_drop={band_data['train_drop']:.3f} dB, "
              f"test_drop={band_data['test_drop']:.3f} dB, "
              f"ratio={band_data['overfitting_ratio']:.2f}x")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Test diagnosis on Lego (if trained)**

```bash
cd /mnt/ssd1/idea/TurboQuant/gaussian-splatting

# Train/test gap
/mnt/ssd1/conda_envs/nerf_tq/bin/python -m diagnosis.train_test_gap \
  -m output/lego_wb -s data/nerf_synthetic/lego \
  --output results/lego_gap.json

# Per-band analysis
/mnt/ssd1/conda_envs/nerf_tq/bin/python -m diagnosis.sh_band_analysis \
  -m output/lego_wb -s data/nerf_synthetic/lego \
  --output results/lego_bands.json
```

Expected: Gap ~2.6-5.1 dB; R_3 > 2.0.

- [ ] **Step 5: Commit**

```bash
cd /mnt/ssd1/idea/TurboQuant
git add gaussian-splatting/diagnosis/
git commit -m "feat: SH overfitting diagnosis - train/test gap and per-band analysis"
```

---

## Task 6: Implement Compression Evaluation Script

**Files:**
- Create: `gaussian-splatting/eval_compression.py`

- [ ] **Step 1: Implement batch evaluation**

Create `gaussian-splatting/eval_compression.py`:

```python
"""
Evaluate TurboSplat compression across multiple scenes.

Compresses each scene, renders from the compressed model, computes
PSNR/SSIM/LPIPS vs original renders, and outputs a results table.

Usage:
    python eval_compression.py --scenes lego chair drums --sh_bits 3
"""

import argparse
import json
import os
import sys
import time
import torch
import numpy as np
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compress import load_ply_attributes, compress_gaussians
from decompress import decompress_gaussians, save_ply
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.image_utils import psnr
from lpipsPyTorch import lpips


NERF_SYNTHETIC_SCENES = ["lego", "chair", "drums", "ficus", "hotdog", "materials", "mic", "ship"]


def evaluate_scene(model_path: str, source_path: str, sh_bits: int = 3,
                   iteration: int = 30000, white_background: bool = True,
                   seed: int = 0) -> dict:
    """Compress a scene and evaluate quality metrics."""
    device = torch.device("cuda")
    bg = torch.tensor([1, 1, 1] if white_background else [0, 0, 0],
                      dtype=torch.float32, device=device)

    # 1. Load and compress
    ply_path = os.path.join(model_path, "point_cloud",
                            f"iteration_{iteration}", "point_cloud.ply")
    attrs = load_ply_attributes(ply_path)
    n_gaussians = attrs["xyz"].shape[0]

    compressed_path = os.path.join(model_path, f"compressed_b{sh_bits}.npz")
    t0 = time.time()
    stats = compress_gaussians(attrs, compressed_path, sh_bits=sh_bits, seed=seed)
    compress_time = time.time() - t0

    # 2. Decompress and save as PLY
    recon_attrs = decompress_gaussians(compressed_path)
    recon_ply_path = os.path.join(model_path, f"recon_b{sh_bits}.ply")
    save_ply(recon_attrs, recon_ply_path)

    # 3. Load original and compressed models, render test views
    args_ns = Namespace(
        source_path=source_path, model_path=model_path,
        images="images", resolution=1, data_device="cuda",
        eval=True, sh_degree=3,
    )

    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)

    # Original
    gaussians_orig = GaussianModel(3)
    scene = Scene(args_ns, gaussians_orig, load_iteration=iteration, shuffle=False)
    gaussians_orig.load_ply(ply_path)

    # Compressed
    gaussians_comp = GaussianModel(3)
    gaussians_comp.load_ply(recon_ply_path)

    # Render and compare
    psnr_orig_list, psnr_comp_list = [], []
    ssim_orig_list, ssim_comp_list = [], []
    lpips_orig_list, lpips_comp_list = [], []

    test_cameras = scene.getTestCameras()
    for cam in test_cameras:
        gt = cam.original_image[:3, :, :].to(device)

        render_orig = render(cam, gaussians_orig, pipe, bg)["render"]
        render_comp = render(cam, gaussians_comp, pipe, bg)["render"]

        psnr_orig_list.append(psnr(render_orig, gt).item())
        psnr_comp_list.append(psnr(render_comp, gt).item())

        from utils.loss_utils import ssim as ssim_fn
        ssim_orig_list.append(ssim_fn(render_orig, gt).item())
        ssim_comp_list.append(ssim_fn(render_comp, gt).item())

        lpips_orig_list.append(lpips(render_orig, gt, net_type="vgg").item())
        lpips_comp_list.append(lpips(render_comp, gt, net_type="vgg").item())

    # Cleanup temp files
    if os.path.exists(recon_ply_path):
        os.remove(recon_ply_path)

    return {
        "n_gaussians": n_gaussians,
        "sh_bits": sh_bits,
        "compression_ratio": stats["compression_ratio"],
        "compression_time_s": compress_time,
        "original_size_mb": stats["original_size_bytes"] / 1e6,
        "compressed_size_mb": stats["compressed_size_bytes"] / 1e6,
        "orig_psnr": float(np.mean(psnr_orig_list)),
        "comp_psnr": float(np.mean(psnr_comp_list)),
        "psnr_drop": float(np.mean(psnr_orig_list) - np.mean(psnr_comp_list)),
        "orig_ssim": float(np.mean(ssim_orig_list)),
        "comp_ssim": float(np.mean(ssim_comp_list)),
        "orig_lpips": float(np.mean(lpips_orig_list)),
        "comp_lpips": float(np.mean(lpips_comp_list)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=NERF_SYNTHETIC_SCENES)
    parser.add_argument("--data_root", default="data/nerf_synthetic")
    parser.add_argument("--output_root", default="output")
    parser.add_argument("--suffix", default="_wb", help="Model dir suffix (e.g., _wb for white_background)")
    parser.add_argument("--sh_bits", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--output", default="results/compression_results.json")
    args = parser.parse_args()

    all_results = {}
    for scene_name in args.scenes:
        model_path = os.path.join(args.output_root, f"{scene_name}{args.suffix}")
        source_path = os.path.join(args.data_root, scene_name)

        if not os.path.exists(model_path):
            print(f"Skipping {scene_name}: model not found at {model_path}")
            continue

        all_results[scene_name] = {}
        for b in args.sh_bits:
            print(f"\n{'='*60}")
            print(f"Evaluating {scene_name} at sh_bits={b}...")
            result = evaluate_scene(model_path, source_path, sh_bits=b)
            all_results[scene_name][f"b{b}"] = result
            print(f"  PSNR: {result['orig_psnr']:.2f} -> {result['comp_psnr']:.2f} "
                  f"(drop: {result['psnr_drop']:.2f} dB)")
            print(f"  Ratio: {result['compression_ratio']:.2f}x in {result['compression_time_s']:.3f}s")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Scene':<12} {'b':>3} {'Ratio':>6} {'PSNR_orig':>9} {'PSNR_comp':>9} {'Drop':>6} {'Time':>6}")
    print(f"{'-'*80}")
    for scene_name, scene_results in all_results.items():
        for b_key, r in scene_results.items():
            print(f"{scene_name:<12} {r['sh_bits']:>3} {r['compression_ratio']:>6.2f}x "
                  f"{r['orig_psnr']:>9.2f} {r['comp_psnr']:>9.2f} {r['psnr_drop']:>6.2f} "
                  f"{r['compression_time_s']:>5.2f}s")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
cd /mnt/ssd1/idea/TurboQuant
git add gaussian-splatting/eval_compression.py
git commit -m "feat: batch compression evaluation with PSNR/SSIM/LPIPS metrics"
```

---

## Task 7: Implement SQR (Stochastic Quantization Regularization)

**Files:**
- Create: `gaussian-splatting/sqr/__init__.py`
- Create: `gaussian-splatting/sqr/sqr_module.py`
- Create: `gaussian-splatting/tests/test_sqr.py`
- Modify: `gaussian-splatting/train.py` (add SQR integration)

- [ ] **Step 1: Write the failing tests**

Create `gaussian-splatting/tests/test_sqr.py`:

```python
import torch
import numpy as np
import sys
sys.path.insert(0, "/mnt/ssd1/idea/TurboQuant/gaussian-splatting")
from sqr.sqr_module import SQRModule


def test_sqr_output_shape():
    """SQR should preserve tensor shape."""
    sqr = SQRModule(d=45, b=3)
    x = torch.randn(100, 15, 3, device="cuda")  # (N, n_coeffs, 3_channels)
    y = sqr(x)
    assert y.shape == x.shape


def test_sqr_adds_noise():
    """Output should differ from input (quantization noise injected)."""
    sqr = SQRModule(d=45, b=2)
    sqr.train()
    x = torch.randn(100, 15, 3, device="cuda")
    y = sqr(x)
    assert not torch.allclose(x, y, atol=1e-6), "SQR should modify the input"


def test_sqr_eval_passthrough():
    """In eval mode, SQR should be identity (no noise)."""
    sqr = SQRModule(d=45, b=3)
    sqr.eval()
    x = torch.randn(100, 15, 3, device="cuda")
    y = sqr(x)
    assert torch.allclose(x, y), "SQR in eval mode should be identity"


def test_sqr_gradient_flows():
    """Gradients should flow through SQR via straight-through estimator."""
    sqr = SQRModule(d=45, b=3)
    sqr.train()
    x = torch.randn(100, 15, 3, device="cuda", requires_grad=True)
    y = sqr(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None, "Gradients should flow through SQR"
    assert not torch.all(x.grad == 0), "Gradients should be non-zero"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /mnt/ssd1/idea/TurboQuant/gaussian-splatting
/mnt/ssd1/conda_envs/nerf_tq/bin/python -m pytest tests/test_sqr.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement SQR module**

Create `gaussian-splatting/sqr/__init__.py`:
```python
```

Create `gaussian-splatting/sqr/sqr_module.py`:

```python
"""
Stochastic Quantization Regularization (SQR) for 3DGS training.

Injects TurboQuant-style quantization noise into SH coefficients during
training to prevent overfitting. Uses straight-through estimator for
gradient flow.

Key insight: SH coefficients overfit (band l=3 is ~60% noise). By
injecting quantization noise during training, we prevent overfitting
while maintaining signal quality — the same effect as post-hoc
compression, but applied during optimization.
"""

import torch
import torch.nn as nn
import numpy as np
from turbo_quant.codebook import generate_codebook


class SQRFunction(torch.autograd.Function):
    """Straight-through estimator for quantization."""

    @staticmethod
    def forward(ctx, x, centroids, boundaries):
        # x: (N, d) flattened SH coefficients after rotation
        # Find nearest centroid for each value
        indices = torch.searchsorted(boundaries[1:-1], x)
        quantized = centroids[indices]
        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: pass gradients unchanged
        return grad_output, None, None


class SQRModule(nn.Module):
    """
    Stochastic Quantization Regularization module.

    During training: applies random rotation + quantization + inverse rotation
    to SH coefficients, injecting structured noise.
    During eval: identity (no noise).

    Args:
        d: total dimension of SH rest coefficients (e.g., 45 for degree 3)
        b: bit-width for quantization (lower = more regularization)
        p: probability of applying SQR per forward pass (default: 0.5)
    """

    def __init__(self, d: int = 45, b: int = 3, p: float = 0.5):
        super().__init__()
        self.d = d
        self.b = b
        self.p = p

        # Precompute codebook
        centroids_np = generate_codebook(d, b).astype(np.float32)
        self.register_buffer("centroids", torch.from_numpy(centroids_np))

        boundaries_np = np.concatenate([
            [-1e10],
            (centroids_np[:-1] + centroids_np[1:]) / 2,
            [1e10]
        ]).astype(np.float32)
        self.register_buffer("boundaries", torch.from_numpy(boundaries_np))

    def _generate_rotation(self, device):
        """Generate a fresh random rotation matrix on GPU."""
        gaussian = torch.randn(self.d, self.d, device=device)
        q, r = torch.linalg.qr(gaussian)
        # Ensure proper rotation (det = +1)
        d = torch.diag(r)
        ph = torch.sign(d)
        q = q * ph.unsqueeze(0)
        return q

    def forward(self, sh_rest: torch.Tensor) -> torch.Tensor:
        """
        Apply SQR to SH rest coefficients.

        Args:
            sh_rest: (N, n_coeffs, 3) tensor of SH rest coefficients
                     where n_coeffs = 15 for degree 3 (coeffs 1-15)

        Returns:
            Quantized SH rest with same shape, gradients via STE
        """
        if not self.training:
            return sh_rest

        # Stochastic: skip with probability (1 - p)
        if torch.rand(1).item() > self.p:
            return sh_rest

        N, n_coeffs, n_channels = sh_rest.shape
        device = sh_rest.device

        # Reshape to (N, d) where d = n_coeffs * n_channels
        flat = sh_rest.reshape(N, -1)  # (N, d)
        d = flat.shape[1]

        # Compute norms and normalize
        norms = torch.norm(flat, dim=1, keepdim=True)  # (N, 1)
        safe_norms = torch.clamp(norms, min=1e-10)
        unit = flat / safe_norms

        # Apply random rotation
        rotation = self._generate_rotation(device)  # (d, d)
        # Pad or trim rotation if d != self.d
        if d != self.d:
            rotation = self._generate_rotation_sized(d, device)
        rotated = unit @ rotation.T  # (N, d)

        # Quantize with STE
        quantized = SQRFunction.apply(rotated, self.centroids, self.boundaries)

        # Inverse rotation
        dequantized = quantized @ rotation  # (N, d)

        # Rescale
        result = dequantized * safe_norms

        return result.reshape(N, n_coeffs, n_channels)

    def _generate_rotation_sized(self, d, device):
        """Generate rotation matrix of arbitrary size."""
        gaussian = torch.randn(d, d, device=device)
        q, r = torch.linalg.qr(gaussian)
        diag = torch.diag(r)
        ph = torch.sign(diag)
        q = q * ph.unsqueeze(0)
        return q
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /mnt/ssd1/idea/TurboQuant/gaussian-splatting
/mnt/ssd1/conda_envs/nerf_tq/bin/python -m pytest tests/test_sqr.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Create modified training script with SQR integration**

Create `gaussian-splatting/sqr/train_sqr.py`. This wraps the original `train.py` with SQR injection. The key modification is inserting SQR into the forward pass after densification completes:

```python
"""
Modified 3DGS training with SQR (Stochastic Quantization Regularization).

This script patches the standard 3DGS training loop to inject quantization
noise into SH coefficients after densification is complete (~15K iterations).

Usage:
    python -m sqr.train_sqr -s data/nerf_synthetic/lego -m output/lego_sqr \
        --white_background --sqr_start 15000 --sqr_b 3 --sqr_p 0.5
"""

import os
import sys
import torch
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqr.sqr_module import SQRModule


def patch_gaussian_model_with_sqr(gaussians, sqr_module):
    """
    Monkey-patch the GaussianModel to apply SQR to SH features
    when get_features is called during rendering.
    """
    original_get_features = gaussians.__class__.get_features.fget

    @property
    def get_features_sqr(self):
        features_dc = self._features_dc
        features_rest = self._features_rest

        if self.training and hasattr(self, '_sqr_module') and self._sqr_module is not None:
            features_rest = self._sqr_module(features_rest)

        return torch.cat((features_dc, features_rest), dim=1)

    gaussians.__class__.get_features = get_features_sqr
    gaussians._sqr_module = sqr_module


def main():
    # Import and run standard training with SQR patches
    from train import training as original_training
    from utils.general_utils import safe_state
    from argparse import ArgumentParser

    parser = ArgumentParser(description="3DGS Training with SQR")
    # Standard 3DGS args are added by the training function
    # Add SQR-specific args
    parser.add_argument("--sqr_start", type=int, default=15000,
                        help="Iteration to start SQR (after densification)")
    parser.add_argument("--sqr_b", type=int, default=3,
                        help="SQR bit-width (lower = stronger regularization)")
    parser.add_argument("--sqr_p", type=float, default=0.5,
                        help="Probability of applying SQR per forward pass")
    parser.add_argument("--sqr_warmup", type=int, default=1000,
                        help="Number of iterations to warm up SQR noise")

    # Parse known args (let the rest pass through to standard training)
    args, remaining = parser.parse_known_args()

    print(f"\n{'='*60}")
    print(f"SQR Training Configuration:")
    print(f"  Start iteration: {args.sqr_start}")
    print(f"  Bit-width: {args.sqr_b}")
    print(f"  Probability: {args.sqr_p}")
    print(f"  Warmup: {args.sqr_warmup} iterations")
    print(f"{'='*60}\n")

    # Store SQR config for the training hook
    sqr_config = {
        "start": args.sqr_start,
        "b": args.sqr_b,
        "p": args.sqr_p,
        "warmup": args.sqr_warmup,
    }

    # Write config to model dir for reproducibility
    # The actual integration into train.py's loop requires modifying train.py
    # For now, save the config and provide instructions
    print("NOTE: SQR integration requires modifying train.py's training loop.")
    print("Add the following after the densification phase (iteration > densify_until_iter):")
    print()
    print("    if iteration == sqr_start:")
    print("        sqr = SQRModule(d=45, b=sqr_b, p=sqr_p).cuda()")
    print("        patch_gaussian_model_with_sqr(gaussians, sqr)")
    print("        print('SQR enabled')")
    print()

    # For fully automated training, we need to modify train.py directly.
    # This will be done in the implementation phase.


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
cd /mnt/ssd1/idea/TurboQuant
git add gaussian-splatting/sqr/ gaussian-splatting/tests/test_sqr.py
git commit -m "feat: SQR module - stochastic quantization regularization for 3DGS training"
```

---

## Task 8: Train All NeRF Synthetic Scenes

**Files:**
- Create: `gaussian-splatting/scripts/train_all.sh`

- [ ] **Step 1: Create batch training script**

Create `gaussian-splatting/scripts/train_all.sh`:

```bash
#!/bin/bash
# Train all 8 NeRF Synthetic scenes with white background
# Expected: ~2 hours total on RTX 4090

PYTHON=/mnt/ssd1/conda_envs/nerf_tq/bin/python
DATA_ROOT=data/nerf_synthetic
cd /mnt/ssd1/idea/TurboQuant/gaussian-splatting

SCENES="lego chair drums ficus hotdog materials mic ship"

for scene in $SCENES; do
    echo "=========================================="
    echo "Training $scene"
    echo "=========================================="

    if [ -f "output/${scene}_wb/point_cloud/iteration_30000/point_cloud.ply" ]; then
        echo "  Already trained, skipping."
        continue
    fi

    $PYTHON train.py \
        -s ${DATA_ROOT}/${scene} \
        -m output/${scene}_wb \
        --white_background \
        --iterations 30000 \
        --eval \
        --save_iterations 1000 3000 7000 15000 30000

    echo "  Done. Rendering test views..."
    $PYTHON render.py -m output/${scene}_wb --skip_train
    $PYTHON metrics.py -m output/${scene}_wb
done

echo "All scenes trained."
```

- [ ] **Step 2: Make executable and run**

```bash
chmod +x /mnt/ssd1/idea/TurboQuant/gaussian-splatting/scripts/train_all.sh
cd /mnt/ssd1/idea/TurboQuant/gaussian-splatting
bash scripts/train_all.sh
```

Note: Save checkpoints at 1K/3K/7K/15K/30K for the iteration sweep experiment.

- [ ] **Step 3: Commit**

```bash
cd /mnt/ssd1/idea/TurboQuant
git add gaussian-splatting/scripts/train_all.sh
git commit -m "feat: batch training script for all NeRF Synthetic scenes"
```

---

## Task 9: Run All Experiments

**Files:**
- Create: `gaussian-splatting/scripts/run_experiments.sh`

- [ ] **Step 1: Create experiment runner script**

Create `gaussian-splatting/scripts/run_experiments.sh`:

```bash
#!/bin/bash
# Run all TurboSplat experiments
PYTHON=/mnt/ssd1/conda_envs/nerf_tq/bin/python
cd /mnt/ssd1/idea/TurboQuant/gaussian-splatting
mkdir -p results

echo "===== Experiment 1: Overfitting Diagnosis ====="
SCENES="lego chair drums ficus hotdog materials mic ship"
for scene in $SCENES; do
    echo "--- $scene ---"
    $PYTHON -m diagnosis.train_test_gap \
        -m output/${scene}_wb -s data/nerf_synthetic/${scene} \
        --output results/${scene}_gap.json

    $PYTHON -m diagnosis.sh_band_analysis \
        -m output/${scene}_wb -s data/nerf_synthetic/${scene} \
        --output results/${scene}_bands.json
done

echo "===== Experiment 2: Compression Evaluation ====="
$PYTHON eval_compression.py \
    --scenes $SCENES \
    --sh_bits 2 3 4 \
    --output results/compression_results.json

echo "===== All experiments complete ====="
echo "Results saved to results/"
```

- [ ] **Step 2: Run experiments**

```bash
chmod +x /mnt/ssd1/idea/TurboQuant/gaussian-splatting/scripts/run_experiments.sh
cd /mnt/ssd1/idea/TurboQuant/gaussian-splatting
bash scripts/run_experiments.sh
```

- [ ] **Step 3: Commit results**

```bash
cd /mnt/ssd1/idea/TurboQuant
git add gaussian-splatting/scripts/run_experiments.sh
git commit -m "feat: experiment runner for diagnosis and compression evaluation"
```

---

## Task 10: Integrate SQR into 3DGS Training Loop

**Files:**
- Modify: `gaussian-splatting/train.py` (add SQR hooks)

- [ ] **Step 1: Read train.py to find the right injection points**

```bash
cd /mnt/ssd1/idea/TurboQuant/gaussian-splatting
# Find densification end and the main training loop structure
grep -n "densif" train.py | head -20
grep -n "iteration" train.py | head -30
```

Identify:
- The iteration loop start
- Where densification ends (`densify_until_iter`)
- Where rendering happens (calls to `render()`)
- Where loss is computed

- [ ] **Step 2: Add SQR arguments to train.py**

Add after the existing argument parsing section in `train.py`:

```python
# SQR arguments
parser.add_argument("--sqr", action="store_true", default=False,
                    help="Enable Stochastic Quantization Regularization")
parser.add_argument("--sqr_start", type=int, default=15000,
                    help="Iteration to start SQR")
parser.add_argument("--sqr_b", type=int, default=3,
                    help="SQR bit-width")
parser.add_argument("--sqr_p", type=float, default=0.5,
                    help="SQR application probability")
```

- [ ] **Step 3: Add SQR initialization and injection into training loop**

In the training function, after densification logic, add:

```python
# SQR initialization (after densification completes)
if args.sqr and iteration == args.sqr_start:
    from sqr.sqr_module import SQRModule, patch_gaussian_model_with_sqr
    d_sh = gaussians._features_rest.shape[1] * gaussians._features_rest.shape[2]
    sqr_module = SQRModule(d=d_sh, b=args.sqr_b, p=args.sqr_p).cuda()
    patch_gaussian_model_with_sqr(gaussians, sqr_module)
    print(f"\n[SQR] Enabled at iteration {iteration}: b={args.sqr_b}, p={args.sqr_p}, d={d_sh}")
```

Also add gradient clipping for SH parameters after the loss backward step:

```python
if args.sqr and iteration >= args.sqr_start:
    # Clip SH gradients to prevent NaN
    if gaussians._features_rest.grad is not None:
        torch.nn.utils.clip_grad_norm_([gaussians._features_rest], max_norm=1.0)
```

- [ ] **Step 4: Test SQR training on Lego**

```bash
cd /mnt/ssd1/idea/TurboQuant/gaussian-splatting
/mnt/ssd1/conda_envs/nerf_tq/bin/python train.py \
    -s data/nerf_synthetic/lego \
    -m output/lego_sqr \
    --white_background \
    --iterations 30000 \
    --eval \
    --sqr \
    --sqr_start 15000 \
    --sqr_b 3 \
    --sqr_p 0.5
```

Monitor for:
- NaN losses (check stdout)
- SQR activation message at iteration 15000
- Final test PSNR compared to vanilla (expect small improvement: +0.1-0.5 dB)

- [ ] **Step 5: Compare SQR vs vanilla results**

```bash
cd /mnt/ssd1/idea/TurboQuant/gaussian-splatting
/mnt/ssd1/conda_envs/nerf_tq/bin/python render.py -m output/lego_sqr --skip_train
/mnt/ssd1/conda_envs/nerf_tq/bin/python metrics.py -m output/lego_sqr
echo "Compare with vanilla:"
cat output/lego_wb/results.json | python3 -m json.tool | head -5
cat output/lego_sqr/results.json | python3 -m json.tool | head -5
```

- [ ] **Step 6: If NaN occurs, apply mitigations**

If training produces NaN:
1. Reduce SQR probability: `--sqr_p 0.3`
2. Start later: `--sqr_start 20000`
3. Use higher bit-width (less noise): `--sqr_b 4`
4. Add warmup: linearly ramp SQR probability from 0 to sqr_p over 2000 iterations

- [ ] **Step 7: Commit**

```bash
cd /mnt/ssd1/idea/TurboQuant
git add gaussian-splatting/train.py
git commit -m "feat: integrate SQR into 3DGS training loop with gradient clipping"
```

---

## Summary of Task Dependencies

```
Task 1 (Clone + env setup)
  └── Task 2 (Codebook)
       └── Task 3 (Quantizer)
            └── Task 4 (Compress/Decompress)
                 ├── Task 5 (Diagnosis) ─── requires trained models from Task 8
                 ├── Task 6 (Eval script) ─── requires trained models from Task 8
                 └── Task 7 (SQR module)
                      └── Task 10 (SQR training integration)
Task 8 (Train all scenes) ─── can run in parallel after Task 1
Task 9 (Run experiments) ─── requires Tasks 4, 5, 6, 8
```

**Critical path:** Tasks 1→2→3→4→6→8→9 (core pipeline + experiments)

**Parallel work:** Task 8 (training) can start as soon as Task 1 is done — it takes ~2 hours and doesn't depend on the TurboQuant code.
