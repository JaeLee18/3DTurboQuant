"""Decompress TurboQuant-compressed 3DGS models.

Reads .npz files produced by compress.py and reconstructs all Gaussian
attributes. Can optionally save the result to a PLY file compatible with the
standard 3DGS renderer.

Usage:
    python decompress.py -i compressed/lego.npz -o decompressed/lego.ply
"""

import argparse
import os
import numpy as np
from plyfile import PlyData, PlyElement

from compress import _uniform_dequantize


# ---------------------------------------------------------------------------
# Decompression
# ---------------------------------------------------------------------------

def decompress_gaussians(compressed_path: str) -> dict:
    """Decompress a .npz file back to 3DGS attributes.

    SH rest is reconstructed via centroid lookup + inverse rotation (no
    TurboQuantizer needed at decompression time). Other attributes are
    reconstructed via uniform dequantization.

    Args:
        compressed_path: Path to the .npz file.

    Returns:
        Dict with keys: xyz (N,3), sh_dc (N,3), sh_rest (N,D),
        opacity (N,1), scales (N,3), rotations (N,4).
    """
    data = np.load(compressed_path)

    N = int(data["n_gaussians"])

    # --- SH rest: TurboQuant inverse ---
    sh_indices = data["sh_indices"]        # (N, D) uint8
    sh_norms = data["sh_norms"]            # (N,) float32
    sh_rotation = data["sh_rotation"]      # (D, D) float32
    sh_centroids = data["sh_centroids"]    # (2^b,) float32

    # Centroid lookup: map each index to its centroid value
    y_hat = sh_centroids[sh_indices]  # (N, D) float32

    # Inverse rotation: x_hat = Y_hat @ R  (R is orthogonal, so R^{-1} = R^T,
    # but we stored R directly, and the forward pass was X @ R^T, so inverse is Y @ R)
    x_hat = y_hat @ sh_rotation  # (N, D)

    # Rescale by norms
    sh_rest = (x_hat * sh_norms[:, np.newaxis]).astype(np.float32)

    # --- Uniform dequantize other attributes ---
    def _dequant(name):
        idx = data[f"{name}_idx"]
        vmin = float(data[f"{name}_vmin"])
        scale = float(data[f"{name}_scale"])
        bits = int(data[f"{name}_bits"])
        shape = tuple(data[f"{name}_shape"])
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
