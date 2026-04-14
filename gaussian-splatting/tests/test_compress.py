"""Tests for the full-attribute compression pipeline (Task 4).

Tests compress.py and decompress.py: round-trip fidelity and compression ratio.
"""

import numpy as np
import pytest
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from compress import load_ply_attributes, compress_gaussians, _uniform_quantize, _uniform_dequantize
from decompress import decompress_gaussians, save_ply


def _make_synthetic_attrs(N=1000, sh_dim=45, seed=42):
    """Create a synthetic attributes dict mimicking a 3DGS model."""
    rng = np.random.default_rng(seed)
    attrs = {
        "xyz": rng.standard_normal((N, 3)).astype(np.float32) * 2.0,
        "sh_dc": rng.standard_normal((N, 3)).astype(np.float32) * 0.5,
        "sh_rest": rng.standard_normal((N, sh_dim)).astype(np.float32) * 0.1,
        "opacity": rng.uniform(-5, 5, (N, 1)).astype(np.float32),
        "scales": rng.standard_normal((N, 3)).astype(np.float32),
        "rotations": rng.standard_normal((N, 4)).astype(np.float32),
    }
    return attrs


class TestUniformQuantize:
    """Unit tests for the uniform scalar quantizer."""

    def test_roundtrip_shape(self):
        values = np.random.randn(100).astype(np.float32)
        indices, vmin, scale = _uniform_quantize(values, bits=8)
        recon = _uniform_dequantize(indices, vmin, scale, bits=8)
        assert recon.shape == values.shape

    def test_8bit_low_error(self):
        values = np.random.randn(10000).astype(np.float32)
        indices, vmin, scale = _uniform_quantize(values, bits=8)
        recon = _uniform_dequantize(indices, vmin, scale, bits=8)
        mse = np.mean((values - recon) ** 2)
        # 8-bit uniform should give very low MSE relative to data variance
        assert mse < 0.01 * np.var(values), f"8-bit MSE={mse:.6f} too high"

    def test_index_range(self):
        values = np.random.randn(500).astype(np.float32)
        for bits in [2, 4, 6, 8]:
            indices, _, _ = _uniform_quantize(values, bits=bits)
            assert indices.min() >= 0
            assert indices.max() <= 2**bits - 1


class TestCompressDecompress:
    """Integration tests for the full compression pipeline."""

    def test_compress_decompress_synthetic(self):
        """Compress and decompress synthetic attrs; verify shapes match and MSE is reasonable."""
        N = 2000
        attrs = _make_synthetic_attrs(N=N, sh_dim=45)

        with tempfile.TemporaryDirectory() as tmpdir:
            compressed_path = os.path.join(tmpdir, "test.npz")
            stats = compress_gaussians(attrs, compressed_path, sh_bits=3,
                                       pos_bits=8, scale_bits=6, rot_bits=4,
                                       opacity_bits=4, seed=0)

            assert stats["n_gaussians"] == N
            assert stats["compression_ratio"] > 1.0
            assert os.path.exists(compressed_path)

            # Decompress
            recon = decompress_gaussians(compressed_path)

            # Check all keys present and shapes match
            for key in attrs:
                assert key in recon, f"Missing key: {key}"
                assert recon[key].shape == attrs[key].shape, (
                    f"Shape mismatch for {key}: {recon[key].shape} vs {attrs[key].shape}"
                )

            # Check MSE is reasonable for each attribute
            for key in attrs:
                mse = np.mean((attrs[key] - recon[key]) ** 2)
                variance = np.var(attrs[key])
                # MSE should be well below variance (signal is not destroyed)
                assert mse < variance, (
                    f"MSE for {key} ({mse:.6f}) exceeds variance ({variance:.6f})"
                )
                print(f"  {key:12s}: MSE={mse:.6f}, var={variance:.6f}, "
                      f"NMSE={mse/variance:.4f}")

    def test_compression_ratio(self):
        """10K Gaussians should achieve > 2x compression."""
        N = 10_000
        attrs = _make_synthetic_attrs(N=N, sh_dim=45)

        with tempfile.TemporaryDirectory() as tmpdir:
            compressed_path = os.path.join(tmpdir, "test.npz")
            stats = compress_gaussians(attrs, compressed_path, sh_bits=3, seed=0)

            ratio = stats["compression_ratio"]
            print(f"\n  Compression ratio: {ratio:.2f}x")
            print(f"  Original size:    {stats['original_size_bytes'] / 1024:.1f} KB")
            print(f"  Compressed size:  {stats['compressed_size_bytes'] / 1024:.1f} KB")
            print(f"  Compression time: {stats['compression_time_s']*1000:.1f} ms")
            assert ratio > 2.0, f"Compression ratio {ratio:.2f}x is below 2x threshold"

    def test_ply_roundtrip(self):
        """Compress, decompress, save to PLY, reload from PLY, verify shapes."""
        N = 500
        attrs = _make_synthetic_attrs(N=N, sh_dim=45)

        with tempfile.TemporaryDirectory() as tmpdir:
            compressed_path = os.path.join(tmpdir, "test.npz")
            ply_path = os.path.join(tmpdir, "test.ply")

            compress_gaussians(attrs, compressed_path, sh_bits=3, seed=0)
            recon = decompress_gaussians(compressed_path)
            save_ply(recon, ply_path)

            # Reload from PLY
            reloaded = load_ply_attributes(ply_path)
            for key in attrs:
                assert key in reloaded, f"Missing key after PLY roundtrip: {key}"
                assert reloaded[key].shape == attrs[key].shape, (
                    f"Shape mismatch after PLY roundtrip for {key}: "
                    f"{reloaded[key].shape} vs {attrs[key].shape}"
                )
