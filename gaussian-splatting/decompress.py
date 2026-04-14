"""Decompress TurboQuant-compressed 3DGS models.

Reads .npz or .tsv4 files produced by compress.py and reconstructs all Gaussian
attributes. Can optionally save the result to a PLY file compatible with the
standard 3DGS renderer.

Handles v1 (full SH), v2 (pruned/truncated SH), and v4 (zstd) compressed files.

Usage:
    python decompress.py -i compressed/lego.npz -o decompressed/lego.ply
    python decompress.py -i compressed/lego.tsv4 -o decompressed/lego.ply
"""

import argparse
import os
import numpy as np
from plyfile import PlyData, PlyElement

from compress import _uniform_dequantize, SH_BAND_DIMS
from entropy_utils import (
    load_compressed as _load_zstd,
    unpack_indices,
    delta_decode,
    byte_unshuffle,
)


# ---------------------------------------------------------------------------
# Decompression
# ---------------------------------------------------------------------------

def _load_data(compressed_path):
    """Load compressed data from either .npz or .tsv4 format.

    Returns a dict-like object (np.NpzFile or plain dict).
    """
    # Peek at magic bytes to decide format
    with open(compressed_path, "rb") as f:
        magic = f.read(4)
    if magic == b"TSv4":
        return _load_zstd(compressed_path)
    else:
        return np.load(compressed_path)


def decompress_gaussians(compressed_path: str) -> dict:
    """Decompress a .npz or .tsv4 file back to 3DGS attributes.

    SH rest is reconstructed via centroid lookup + inverse rotation (no
    TurboQuantizer needed at decompression time). Other attributes are
    reconstructed via uniform dequantization.

    Handles truncated SH: if sh_degree < 3, sh_rest will have fewer
    dimensions. The remaining (dropped) bands are zero-padded to restore
    the full 45-dim sh_rest for renderer compatibility.

    Args:
        compressed_path: Path to the .npz or .tsv4 file.

    Returns:
        Dict with keys: xyz (N,3), sh_dc (N,3), sh_rest (N,45),
        opacity (N,1), scales (N,3), rotations (N,4).
    """
    data = _load_data(compressed_path)

    def _scalar(key, default=None):
        """Extract a scalar value from data, handling 0-d arrays."""
        if key not in data:
            return default
        v = data[key]
        return v.item() if hasattr(v, 'item') else v

    N = int(_scalar("n_gaussians"))
    d = int(_scalar("sh_d"))

    # Read sh_degree if present (v2+), default to 3 (v1 files)
    sh_degree = int(_scalar("sh_degree", 3))
    sh_bits = int(_scalar("sh_bits", 3))

    # Check if SH indices are bit-packed (zstd format)
    sh_packed = bool(_scalar("sh_packed", 0))
    is_byte_shuffled = bool(_scalar("byte_shuffled", 0))

    # --- SH rest: TurboQuant inverse ---
    if d > 0:
        sh_indices_raw = data["sh_indices"]        # possibly packed
        sh_norms_raw = data["sh_norms"]
        if is_byte_shuffled:
            sh_norms_raw = np.frombuffer(
                byte_unshuffle(sh_norms_raw.flatten(), np.float16).tobytes(),
                dtype=np.float16,
            )
        sh_norms = sh_norms_raw.astype(np.float32)  # (N,) may be float16
        sh_rotation = data["sh_rotation"]      # (D, D) float32
        sh_centroids = data["sh_centroids"]    # (2^b,) float32

        if sh_packed:
            # Unpack bit-packed SH indices
            orig_shape = tuple(data["sh_indices_shape"])
            orig_len = 1
            for s in orig_shape:
                orig_len *= s
            sh_indices = unpack_indices(sh_indices_raw.flatten(), orig_len, sh_bits)
            sh_indices = sh_indices.reshape(orig_shape)
        else:
            sh_indices = sh_indices_raw  # (N, D) uint8

        # Centroid lookup: map each index to its centroid value
        y_hat = sh_centroids[sh_indices]  # (N, D) float32

        # Inverse rotation: x_hat = Y_hat @ R  (R is orthogonal, so R^{-1} = R^T,
        # but we stored R directly, and the forward pass was X @ R^T, so inverse is Y @ R)
        x_hat = y_hat @ sh_rotation  # (N, D)

        # Rescale by norms
        sh_rest_truncated = (x_hat * sh_norms[:, np.newaxis]).astype(np.float32)
    else:
        sh_rest_truncated = np.zeros((N, 0), dtype=np.float32)

    # Zero-pad to full 45-dim sh_rest if truncated
    full_sh_dim = SH_BAND_DIMS[3]  # 45
    if d < full_sh_dim:
        sh_rest = np.zeros((N, full_sh_dim), dtype=np.float32)
        if d > 0:
            sh_rest[:, :d] = sh_rest_truncated
    else:
        sh_rest = sh_rest_truncated

    # Check if uniform-quantized indices are delta-coded (zstd format)
    is_delta = bool(_scalar("delta_coded", 0))

    # --- Uniform dequantize other attributes ---
    def _dequant(name):
        idx_raw = data[f"{name}_idx"]
        shape = tuple(data[f"{name}_shape"])
        bits = int(_scalar(f"{name}_bits"))

        if is_byte_shuffled:
            # Recover original dtype from stored string
            dtype_key = f"{name}_idx_dtype"
            if dtype_key in data:
                orig_dtype = bytes(data[dtype_key].tolist()).decode("utf-8")
            else:
                orig_dtype = "uint16"
            idx = np.frombuffer(
                byte_unshuffle(idx_raw.flatten(), np.dtype(orig_dtype)).tobytes(),
                dtype=np.dtype(orig_dtype),
            ).reshape(shape)
        else:
            idx = idx_raw

        if is_delta and name == "pos":
            idx = delta_decode(idx, target_dtype=np.uint16)

        vmin = float(_scalar(f"{name}_vmin"))
        scale = float(_scalar(f"{name}_scale"))
        recon = _uniform_dequantize(idx, vmin, scale, bits)
        return recon.reshape(shape)

    xyz = _dequant("pos")
    sh_dc = _dequant("dc")
    scales = _dequant("scale")
    rotations = _dequant("rot")
    opacity = _dequant("opacity")

    return {
        "xyz": xyz,
        "sh_dc": sh_dc,
        "sh_rest": sh_rest,
        "opacity": opacity,
        "scales": scales,
        "rotations": rotations,
    }


# ---------------------------------------------------------------------------
# PLY saving
# ---------------------------------------------------------------------------

def save_ply(attrs: dict, output_path: str) -> None:
    """Save reconstructed attributes to a PLY file compatible with 3DGS.

    The PLY format matches the original 3DGS save format: x,y,z, nx,ny,nz,
    f_dc_0..2, f_rest_0..44, opacity, scale_0..2, rot_0..3, all as float32.

    Args:
        attrs: Dict from decompress_gaussians or load_ply_attributes.
        output_path: Where to write the PLY file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    xyz = attrs["xyz"]
    sh_dc = attrs["sh_dc"]
    sh_rest = attrs["sh_rest"]
    opacity = attrs["opacity"]
    scales = attrs["scales"]
    rotations = attrs["rotations"]
    N = xyz.shape[0]

    normals = np.zeros((N, 3), dtype=np.float32)

    # Build property names list
    prop_names = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(sh_dc.shape[1]):
        prop_names.append(f"f_dc_{i}")
    for i in range(sh_rest.shape[1]):
        prop_names.append(f"f_rest_{i}")
    prop_names.append("opacity")
    for i in range(scales.shape[1]):
        prop_names.append(f"scale_{i}")
    for i in range(rotations.shape[1]):
        prop_names.append(f"rot_{i}")

    dtype_full = [(name, "f4") for name in prop_names]
    elements = np.empty(N, dtype=dtype_full)

    # Concatenate all attributes in order
    all_attrs = np.concatenate(
        [xyz, normals, sh_dc, sh_rest, opacity, scales, rotations],
        axis=1,
    ).astype(np.float32)
    elements[:] = list(map(tuple, all_attrs))

    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Decompress a TurboQuant .npz file to PLY"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to compressed .npz file",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output PLY path",
    )
    args = parser.parse_args()

    print(f"Decompressing: {args.input}")
    attrs = decompress_gaussians(args.input)
    print(f"  {attrs['xyz'].shape[0]} Gaussians, "
          f"SH rest dim={attrs['sh_rest'].shape[1]}")

    save_ply(attrs, args.output)
    ply_size = os.path.getsize(args.output)
    npz_size = os.path.getsize(args.input)
    print(f"\nSaved to: {args.output}")
    print(f"  PLY size: {ply_size/1024:.1f} KB")
    print(f"  NPZ size: {npz_size/1024:.1f} KB")


if __name__ == "__main__":
    main()
