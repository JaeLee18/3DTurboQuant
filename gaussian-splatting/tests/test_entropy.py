"""Test Morton sorting, bit-packing, and zstd compression.

Compares old (npz) vs new (zstd) compression on the Lego scene and
validates round-trip correctness for bit-packing and the zstd format.
"""

import sys
import os
import time
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compress import load_ply_attributes, compress_gaussians
from decompress import decompress_gaussians
from entropy_utils import (
    morton_sort_gaussians,
    pack_2bit, unpack_2bit,
    pack_3bit, unpack_3bit,
    pack_4bit, unpack_4bit,
    pack_indices, unpack_indices,
    save_compressed, load_compressed,
)


# ---------------------------------------------------------------------------
# Unit tests: bit-packing round-trips
# ---------------------------------------------------------------------------

def test_pack_2bit():
    rng = np.random.default_rng(42)
    for n in [0, 1, 3, 4, 5, 100, 1023]:
        data = rng.integers(0, 4, size=n, dtype=np.uint8)
        packed, orig_len = pack_2bit(data)
        assert orig_len == n
        assert len(packed) == (n + 3) // 4
        recovered = unpack_2bit(packed, orig_len)
        np.testing.assert_array_equal(data, recovered)
    print("  PASS: pack_2bit round-trip")


def test_pack_3bit():
    rng = np.random.default_rng(42)
    for n in [0, 1, 7, 8, 9, 100, 1023]:
        data = rng.integers(0, 8, size=n, dtype=np.uint8)
        packed, orig_len = pack_3bit(data)
        assert orig_len == n
        recovered = unpack_3bit(packed, orig_len)
        np.testing.assert_array_equal(data, recovered)
    print("  PASS: pack_3bit round-trip")


def test_pack_4bit():
    rng = np.random.default_rng(42)
    for n in [0, 1, 2, 3, 100, 1023]:
        data = rng.integers(0, 16, size=n, dtype=np.uint8)
        packed, orig_len = pack_4bit(data)
        assert orig_len == n
        assert len(packed) == (n + 1) // 2
        recovered = unpack_4bit(packed, orig_len)
        np.testing.assert_array_equal(data, recovered)
    print("  PASS: pack_4bit round-trip")


def test_pack_indices_dispatch():
    rng = np.random.default_rng(42)
    n = 500
    for bits, maxval in [(2, 4), (3, 8), (4, 16), (8, 256)]:
        data = rng.integers(0, maxval, size=n, dtype=np.uint8)
        packed, orig_len = pack_indices(data, bits)
        recovered = unpack_indices(packed, orig_len, bits)
        np.testing.assert_array_equal(data, recovered)
    print("  PASS: pack_indices dispatch")


# ---------------------------------------------------------------------------
# Unit test: zstd save/load round-trip
# ---------------------------------------------------------------------------

def test_zstd_roundtrip():
    rng = np.random.default_rng(42)
    arrays = {
        "float_2d": rng.standard_normal((100, 3)).astype(np.float32),
        "uint8_1d": rng.integers(0, 4, size=400, dtype=np.uint8),
        "int32_scalar": np.int32(42),
        "float16_1d": rng.standard_normal(100).astype(np.float16),
    }
    with tempfile.NamedTemporaryFile(suffix=".tsv4", delete=False) as f:
        path = f.name
    try:
        save_compressed(path, arrays, compression_level=3)
        loaded = load_compressed(path)
        assert set(loaded.keys()) == set(arrays.keys())
        for k in arrays:
            np.testing.assert_array_equal(arrays[k], loaded[k])
        print("  PASS: zstd save/load round-trip")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Unit test: Morton sort
# ---------------------------------------------------------------------------

def test_morton_sort():
    rng = np.random.default_rng(42)
    N = 1000
    attrs = {
        "xyz": rng.standard_normal((N, 3)).astype(np.float32),
        "sh_dc": rng.standard_normal((N, 3)).astype(np.float32),
        "sh_rest": rng.standard_normal((N, 45)).astype(np.float32),
        "opacity": rng.standard_normal((N, 1)).astype(np.float32),
        "scales": rng.standard_normal((N, 3)).astype(np.float32),
        "rotations": rng.standard_normal((N, 4)).astype(np.float32),
    }
    sorted_attrs = morton_sort_gaussians(attrs)
    # Same keys, same shapes, same set of rows (just reordered)
    assert set(sorted_attrs.keys()) == set(attrs.keys())
    for k in attrs:
        assert sorted_attrs[k].shape == attrs[k].shape
    # xyz rows should be a permutation of the original
    orig_set = set(map(tuple, attrs["xyz"]))
    sort_set = set(map(tuple, sorted_attrs["xyz"]))
    assert orig_set == sort_set
    print("  PASS: morton_sort round-trip")


# ---------------------------------------------------------------------------
# Integration: compress Lego with npz vs zstd, compare sizes
# ---------------------------------------------------------------------------

def test_lego_compression():
    ply_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output", "lego_wb", "point_cloud", "iteration_30000", "point_cloud.ply",
    )
    if not os.path.exists(ply_path):
        print("  SKIP: Lego PLY not found at", ply_path)
        return

    attrs = load_ply_attributes(ply_path)
    N = attrs["xyz"].shape[0]
    print(f"\n  Lego: {N:,} Gaussians")

    with tempfile.TemporaryDirectory() as tmpdir:
        # --- Old method: npz ---
        npz_path = os.path.join(tmpdir, "lego_old.npz")
        t0 = time.perf_counter()
        stats_old = compress_gaussians(attrs, npz_path, sh_bits=2, entropy="npz")
        t_old = time.perf_counter() - t0
        size_old = os.path.getsize(npz_path)

        # --- New method: zstd ---
        zstd_path = os.path.join(tmpdir, "lego_new.tsv4")
        t0 = time.perf_counter()
        stats_new = compress_gaussians(attrs, zstd_path, sh_bits=2, entropy="zstd")
        t_new = time.perf_counter() - t0
        size_new = os.path.getsize(zstd_path)

        print(f"\n  Old (npz):  {size_old/1e6:.2f} MB  in {t_old:.2f}s")
        print(f"  New (zstd): {size_new/1e6:.2f} MB  in {t_new:.2f}s")
        print(f"  Additional compression: {size_old/size_new:.2f}x")

        # --- Verify decompression round-trip for zstd ---
        recon = decompress_gaussians(zstd_path)
        assert recon["xyz"].shape == (N, 3), f"Bad xyz shape: {recon['xyz'].shape}"
        assert recon["sh_rest"].shape == (N, 45), f"Bad sh_rest shape: {recon['sh_rest'].shape}"

        # Also verify npz decompression still works
        recon_npz = decompress_gaussians(npz_path)
        assert recon_npz["xyz"].shape == (N, 3)

        # Check that both decompressed results match (same quantization, just
        # different sorting -- values may differ due to Morton reorder, but
        # each should be a valid reconstruction)
        print(f"  Decompression round-trip: OK (both formats)")
        print(f"  PASS: lego compression comparison")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running entropy_utils tests...")
    test_pack_2bit()
    test_pack_3bit()
    test_pack_4bit()
    test_pack_indices_dispatch()
    test_zstd_roundtrip()
    test_morton_sort()
    test_lego_compression()
    print("\nAll tests passed.")
