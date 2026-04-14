"""Tests for turbo_quant codebook generation (Task 2)."""

import numpy as np
import pytest
import sys
import os

# Ensure the gaussian-splatting directory is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from turbo_quant.codebook import beta_pdf, generate_codebook, compute_mse_cost


def test_beta_pdf_integrates_to_one():
    """PDF should integrate to ~1.0 for d=48 (numerical integration on dense grid)."""
    d = 48
    x = np.linspace(-1 + 1e-9, 1 - 1e-9, 100_000)
    dx = x[1] - x[0]
    pdf_vals = np.array([beta_pdf(xi, d) for xi in x])
    integral = np.trapz(pdf_vals, x)
    assert abs(integral - 1.0) < 1e-3, f"PDF integral = {integral:.6f}, expected ~1.0"


def test_codebook_shape():
    """Codebook should have exactly 2^b entries for b in [1,2,3,4]."""
    d = 48
    for b in [1, 2, 3, 4]:
        cb = generate_codebook(d, b)
        assert len(cb) == 2**b, f"b={b}: expected {2**b} entries, got {len(cb)}"


def test_codebook_sorted():
    """Centroids should be sorted in ascending order."""
    d = 48
    for b in [1, 2, 3, 4]:
        cb = generate_codebook(d, b)
        assert np.all(np.diff(cb) > 0), f"b={b}: codebook is not sorted ascending"


def test_codebook_symmetry():
    """Centroids should be symmetric around 0 for d=48, b=2.

    The Beta PDF is symmetric (even function), so the optimal codebook should
    satisfy c[k] = -c[n-1-k].
    """
    d = 48
    b = 2
    cb = generate_codebook(d, b)
    n = len(cb)
    for i in range(n // 2):
        assert abs(cb[i] + cb[n - 1 - i]) < 1e-4, (
            f"Asymmetry: cb[{i}]={cb[i]:.6f}, cb[{n-1-i}]={cb[n-1-i]:.6f}, "
            f"sum={cb[i]+cb[n-1-i]:.2e}"
        )


def test_codebook_b1_near_expected():
    """For d=1000, b=1, centroids should be near ±sqrt(2/pi)/sqrt(d).

    In very high dimensions the Beta PDF converges to N(0, 1/d).
    The 1-bit Lloyd-Max quantizer for N(0, sigma^2) has centroids at
    ±sigma * sqrt(2/pi), i.e., ±sqrt(2/pi)/sqrt(d).
    """
    d = 1000
    b = 1
    cb = generate_codebook(d, b)
    expected_pos = np.sqrt(2 / np.pi) / np.sqrt(d)
    expected_neg = -expected_pos

    assert len(cb) == 2, f"Expected 2 centroids, got {len(cb)}"
    assert abs(cb[0] - expected_neg) < 0.005, (
        f"Negative centroid: {cb[0]:.5f}, expected ~{expected_neg:.5f}"
    )
    assert abs(cb[1] - expected_pos) < 0.005, (
        f"Positive centroid: {cb[1]:.5f}, expected ~{expected_pos:.5f}"
    )
