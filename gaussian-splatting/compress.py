"""Compress 3D Gaussian Splatting models using TurboQuant.

SH coefficients are compressed with TurboQuant (random rotation + optimal scalar
quantization). Other attributes (positions, DC, scales, rotations, opacity) use
simple uniform scalar quantization. Output is a compressed .npz file.

Usage:
    python compress.py -m output/lego_wb -o compressed/lego.npz --sh_bits 3
"""

import argparse
import os
import time
import numpy as np
from plyfile import PlyData, PlyElement

from turbo_quant.quantizer import TurboQuantizer


# ---------------------------------------------------------------------------
# PLY loading
# ---------------------------------------------------------------------------

def load_ply_attributes(ply_path: str) -> dict:
    """Load 3DGS attributes from a PLY file.

    Args:
        ply_path: Path to a 3DGS PLY file.

    Returns:
        Dict with keys: xyz (N,3), sh_dc (N,3), sh_rest (N,D),
        opacity (N,1), scales (N,3), rotations (N,4).
    """
    plydata = PlyData.read(ply_path)
    vertex = plydata.elements[0]
    N = vertex.count

    # Positions
    xyz = np.stack([
        np.asarray(vertex["x"]),
        np.asarray(vertex["y"]),
        np.asarray(vertex["z"]),
    ], axis=1).astype(np.float32)  # (N, 3)

    # SH DC coefficients
    sh_dc = np.stack([
        np.asarray(vertex["f_dc_0"]),
        np.asarray(vertex["f_dc_1"]),
        np.asarray(vertex["f_dc_2"]),
    ], axis=1).astype(np.float32)  # (N, 3)

    # SH rest coefficients
    f_rest_names = sorted(
        [p.name for p in vertex.properties if p.name.startswith("f_rest_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    sh_rest = np.zeros((N, len(f_rest_names)), dtype=np.float32)
    for i, name in enumerate(f_rest_names):
        sh_rest[:, i] = np.asarray(vertex[name])

    # Opacity
    opacity = np.asarray(vertex["opacity"])[..., np.newaxis].astype(np.float32)  # (N, 1)

    # Scales
    scale_names = sorted(
        [p.name for p in vertex.properties if p.name.startswith("scale_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    scales = np.zeros((N, len(scale_names)), dtype=np.float32)
    for i, name in enumerate(scale_names):
        scales[:, i] = np.asarray(vertex[name])

    # Rotations
    rot_names = sorted(
        [p.name for p in vertex.properties if p.name.startswith("rot_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    rotations = np.zeros((N, len(rot_names)), dtype=np.float32)
    for i, name in enumerate(rot_names):
        rotations[:, i] = np.asarray(vertex[name])

    return {
        "xyz": xyz,
        "sh_dc": sh_dc,
        "sh_rest": sh_rest,
        "opacity": opacity,
        "scales": scales,
        "rotations": rotations,
    }


# ---------------------------------------------------------------------------
# Uniform scalar quantization
# ---------------------------------------------------------------------------

def _uniform_quantize(values: np.ndarray, bits: int):
    """Min/max uniform scalar quantization.

    Args:
        values: Float array of arbitrary shape.
        bits: Number of bits (1..16).

    Returns:
        indices: Integer array (same shape), values in [0, 2^bits - 1].
        vmin: Float scalar, minimum of values.
        scale: Float scalar, (vmax - vmin) / (2^bits - 1).
    """
    n_levels = 2 ** bits - 1
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax == vmin:
        # Constant input: all indices are 0, scale is 1 (arbitrary nonzero)
        return np.zeros_like(values, dtype=np.uint16), vmin, 1.0

    scale = (vmax - vmin) / n_levels
    indices = np.clip(np.round((values - vmin) / scale), 0, n_levels).astype(np.uint16)
    return indices, vmin, scale


def _uniform_dequantize(indices: np.ndarray, vmin: float, scale: float, bits: int):
    """Reconstruct values from uniform quantization parameters.

    Args:
        indices: Integer array of quantized indices.
        vmin: Minimum value from quantization.
        scale: Scale factor from quantization.
        bits: Number of bits used (unused in computation, kept for API symmetry).

    Returns:
        Reconstructed float32 array.
    """
    return (indices.astype(np.float32) * scale + vmin).astype(np.float32)


# ---------------------------------------------------------------------------
# Main compression function
# ---------------------------------------------------------------------------

def compress_gaussians(
    attrs: dict,
    output_path: str,
    sh_bits: int = 3,
    pos_bits: int = 8,
    scale_bits: int = 6,
    rot_bits: int = 4,
    opacity_bits: int = 4,
    seed: int = 0,
) -> dict:
    """Compress 3DGS attributes to a .npz file.

    Args:
        attrs: Dict from load_ply_attributes (or synthetic).
        output_path: Where to save the .npz file.
        sh_bits: Bit-width for SH rest coefficients (TurboQuant).
        pos_bits: Bit-width for xyz positions (uniform).
        scale_bits: Bit-width for scales (uniform).
        rot_bits: Bit-width for rotations (uniform).
        opacity_bits: Bit-width for opacity (uniform).
        seed: Random seed for TurboQuantizer.

    Returns:
        Stats dict with keys: n_gaussians, compression_time_s,
        compressed_size_bytes, original_size_bytes, compression_ratio.
    """
    t0 = time.perf_counter()
    N = attrs["xyz"].shape[0]

    # Original size: all float32 attributes
    original_size = sum(a.nbytes for a in attrs.values())

    # --- SH rest: TurboQuant ---
    sh_rest = attrs["sh_rest"]  # (N, D)
    d = sh_rest.shape[1]
    tq = TurboQuantizer(d=d, b=sh_bits, seed=seed)
    sh_indices, sh_norms = tq.quantize(sh_rest)  # (N, D) uint8, (N,) float32
    sh_rotation = tq.get_rotation_matrix()  # (D, D) float64
    sh_centroids = tq.get_centroids()  # (2^b,) float64

    # --- Uniform quantize other attributes ---
    def _quant(values, bits, name):
        indices, vmin, scale = _uniform_quantize(values, bits)
        return {
            f"{name}_idx": indices,
            f"{name}_vmin": np.float32(vmin),
            f"{name}_scale": np.float32(scale),
            f"{name}_bits": np.uint8(bits),
            f"{name}_shape": np.array(values.shape, dtype=np.int32),
        }

    pos_q = _quant(attrs["xyz"], pos_bits, "pos")
    dc_q = _quant(attrs["sh_dc"], pos_bits, "dc")  # reuse pos_bits for DC
    scale_q = _quant(attrs["scales"], scale_bits, "scale")
    rot_q = _quant(attrs["rotations"], rot_bits, "rot")
    opacity_q = _quant(attrs["opacity"], opacity_bits, "opacity")

    # --- Pack and save ---
    save_dict = {
        # SH rest (TurboQuant)
        "sh_indices": sh_indices,
        "sh_norms": sh_norms,
        "sh_rotation": sh_rotation.astype(np.float32),
        "sh_centroids": sh_centroids.astype(np.float32),
        "sh_bits": np.uint8(sh_bits),
        "sh_d": np.int32(d),
        # Metadata
        "n_gaussians": np.int32(N),
    }
    # Add all uniform-quantized attributes
    for q_dict in [pos_q, dc_q, scale_q, rot_q, opacity_q]:
        save_dict.update(q_dict)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(output_path, **save_dict)

    compression_time = time.perf_counter() - t0
    compressed_size = os.path.getsize(output_path)

    return {
        "n_gaussians": N,
        "compression_time_s": compression_time,
        "compressed_size_bytes": compressed_size,
        "original_size_bytes": original_size,
        "compression_ratio": original_size / compressed_size,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compress a 3DGS model to .npz using TurboQuant"
    )
    parser.add_argument(
        "-m", "--model_path", required=True,
        help="Path to 3DGS output directory (contains point_cloud/iteration_*/point_cloud.ply)",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output .npz path",
    )
    parser.add_argument("--iteration", type=int, default=None,
                        help="Iteration to load (default: latest)")
    parser.add_argument("--sh_bits", type=int, default=3)
    parser.add_argument("--pos_bits", type=int, default=8)
    parser.add_argument("--scale_bits", type=int, default=6)
    parser.add_argument("--rot_bits", type=int, default=4)
    parser.add_argument("--opacity_bits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Find PLY file
    pc_dir = os.path.join(args.model_path, "point_cloud")
    if args.iteration is not None:
        ply_path = os.path.join(pc_dir, f"iteration_{args.iteration}", "point_cloud.ply")
    else:
        # Find latest iteration
        iters = []
        if os.path.isdir(pc_dir):
            for d in os.listdir(pc_dir):
                if d.startswith("iteration_"):
                    try:
                        iters.append(int(d.split("_")[1]))
                    except ValueError:
                        pass
        if not iters:
            # Maybe the model_path is directly a PLY file
            if args.model_path.endswith(".ply"):
                ply_path = args.model_path
            else:
                raise FileNotFoundError(
                    f"No point_cloud iterations found in {pc_dir}"
                )
        else:
            latest = max(iters)
            ply_path = os.path.join(pc_dir, f"iteration_{latest}", "point_cloud.ply")

    print(f"Loading: {ply_path}")
    attrs = load_ply_attributes(ply_path)
    print(f"  {attrs['xyz'].shape[0]} Gaussians, "
          f"SH rest dim={attrs['sh_rest'].shape[1]}")

    stats = compress_gaussians(
        attrs, args.output,
        sh_bits=args.sh_bits,
        pos_bits=args.pos_bits,
        scale_bits=args.scale_bits,
        rot_bits=args.rot_bits,
        opacity_bits=args.opacity_bits,
        seed=args.seed,
    )

    print(f"\nCompressed to: {args.output}")
    print(f"  Gaussians:        {stats['n_gaussians']:,}")
    print(f"  Original size:    {stats['original_size_bytes']/1024:.1f} KB")
    print(f"  Compressed size:  {stats['compressed_size_bytes']/1024:.1f} KB")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"  Time:             {stats['compression_time_s']*1000:.1f} ms")


if __name__ == "__main__":
    main()
