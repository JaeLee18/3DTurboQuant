"""Tests for TurboQuantizer (Task 3).

Tests follow TDD: written before implementation.
"""

import numpy as np
import pytest
import sys
import os

# Ensure the gaussian-splatting directory is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from turbo_quant.quantizer import TurboQuantizer


def _make_random_vectors(N: int, d: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N, d)).astype(np.float32)


def test_roundtrip_shape():
    """quantize + dequantize must return (N, d) float32 array."""
    N, d, b = 100, 48, 3
    tq = TurboQuantizer(d=d, b=b, seed=0)
    vecs = _make_random_vectors(N, d)

    indices, norms = tq.quantize(vecs)
    reconstructed = tq.dequantize(indices, norms)

    assert reconstructed.shape == (N, d), (
        f"Expected shape {(N, d)}, got {reconstructed.shape}"
    )
    assert reconstructed.dtype == np.float32, (
        f"Expected float32, got {reconstructed.dtype}"
    )


def test_indices_are_b_bit():
    """All quantized indices must lie in [0, 2^b - 1]."""
    N, d, b = 200, 48, 3
    tq = TurboQuantizer(d=d, b=b, seed=0)
    vecs = _make_random_vectors(N, d)

    indices, norms = tq.quantize(vecs)

    assert indices.shape == (N, d), f"Expected indices shape {(N, d)}, got {indices.shape}"
    assert indices.dtype == np.uint8, f"Expected uint8 dtype, got {indices.dtype}"
    assert int(indices.min()) >= 0, f"Negative index found: {indices.min()}"
    assert int(indices.max()) <= 2**b - 1, (
        f"Index {indices.max()} exceeds max {2**b - 1} for b={b}"
    )


def test_mse_decreases_with_bitwidth():
    """Higher bit-width should give strictly lower reconstruction MSE.

    MSE(b=1) > MSE(b=2) > MSE(b=3) > MSE(b=4)
    """
    N, d = 1000, 48
    vecs = _make_random_vectors(N, d, seed=7)

    prev_mse = float("inf")
    for b in [1, 2, 3, 4]:
        tq = TurboQuantizer(d=d, b=b, seed=0)
        indices, norms = tq.quantize(vecs)
        recon = tq.dequantize(indices, norms)
        mse = float(np.mean((vecs - recon) ** 2))
        assert mse < prev_mse, (
            f"MSE did not decrease: b={b} MSE={mse:.5f} >= previous {prev_mse:.5f}"
        )
        prev_mse = mse


def test_mse_near_theoretical():
    """For b=3, d=48, per-element MSE should be < 0.10 (paper reports ~0.03).

    We use unit vectors (norm=1) so MSE is purely from quantization error.
    """
    N, d, b = 2000, 48, 3
    rng = np.random.default_rng(99)
    # Generate unit vectors on the sphere
    raw = rng.standard_normal((N, d)).astype(np.float32)
    unit_vecs = raw / np.linalg.norm(raw, axis=1, keepdims=True)

    tq = TurboQuantizer(d=d, b=b, seed=0)
    indices, norms = tq.quantize(unit_vecs)
    recon = tq.dequantize(indices, norms)

    mse = float(np.mean((unit_vecs - recon) ** 2))
    assert mse < 0.10, f"MSE={mse:.4f} exceeds threshold 0.10 for b={b}, d={d}"


def test_deterministic_with_seed():
    """Same seed must produce identical results across two independent instances."""
    N, d, b = 50, 48, 3
    vecs = _make_random_vectors(N, d, seed=123)

    tq1 = TurboQuantizer(d=d, b=b, seed=42)
    idx1, norms1 = tq1.quantize(vecs)
    recon1 = tq1.dequantize(idx1, norms1)

    tq2 = TurboQuantizer(d=d, b=b, seed=42)
    idx2, norms2 = tq2.quantize(vecs)
    recon2 = tq2.dequantize(idx2, norms2)

    np.testing.assert_array_equal(idx1, idx2, err_msg="Indices differ across seeds")
    np.testing.assert_array_almost_equal(
        recon1, recon2, decimal=6,
        err_msg="Reconstructions differ across seeds"
    )
