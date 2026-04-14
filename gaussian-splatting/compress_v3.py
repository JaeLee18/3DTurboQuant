"""TurboSplat v3: Voxel merging + anchor position coding + TurboQuant SH.

Implements a full compression pipeline:
    Input PLY -> Voxel Grid Assignment -> Merge Dense Voxels ->
    Morton Sort -> Anchor+Residual Position Coding ->
    Delta Rotation Coding -> TurboQuant SH ->
    Bit-Pack + zstd -> Output .npz

Target: 15-20x compression at <0.5 dB loss.

Usage:
    python compress_v3.py -m output/lego_wb
    python compress_v3.py -m output/lego_wb --merge_threshold 5
    python compress_v3.py -m output/lego_wb --merge_threshold 2 --sh_bits 3
"""

import argparse
import math
import os
import time
from collections import defaultdict

import numpy as np

from compress import (
    SH_BAND_DIMS,
    _prune_gaussians,
    _truncate_sh,
    _uniform_dequantize,
    _uniform_quantize,
    load_ply_attributes,
)
from turbo_quant.quantizer import TurboQuantizer


# ---------------------------------------------------------------------------
# Utility: downcast quantized indices to minimal dtype
# ---------------------------------------------------------------------------

def _downcast_indices(indices, bits):
    """Downcast quantized uint16 indices to uint8 if bits <= 8."""
    if bits <= 8:
        return indices.astype(np.uint8)
    return indices


# ---------------------------------------------------------------------------
# Step 1: Voxel Grid Assignment
# ---------------------------------------------------------------------------

def assign_voxels(positions, voxel_size=None, grid_resolution=512):
    """Assign each Gaussian to a voxel in a sparse grid.

    Args:
        positions: (N, 3) float32 array of Gaussian positions.
        voxel_size: float, or None to auto-compute from bounding box.
        grid_resolution: int, max grid cells per axis (used when voxel_size is None).

    Returns:
        keys: (N,) int64 - linearized voxel key for each Gaussian.
        voxel_indices: dict mapping voxel_key -> list of Gaussian indices.
        bbox_min: (3,) float32 - bounding box minimum.
        voxel_size: float - computed or provided voxel size.
    """
    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    bbox_size = bbox_max - bbox_min

    if voxel_size is None:
        voxel_size = float(bbox_size.max()) / grid_resolution

    # Quantize positions to grid coordinates
    grid_coords = ((positions - bbox_min) / voxel_size).astype(np.int32)
    grid_coords = np.clip(grid_coords, 0, grid_resolution - 1)

    # Linearized key: x * R^2 + y * R + z
    R = np.int64(grid_resolution)
    keys = (grid_coords[:, 0].astype(np.int64) * R * R +
            grid_coords[:, 1].astype(np.int64) * R +
            grid_coords[:, 2].astype(np.int64))

    # Build voxel -> gaussian index mapping
    voxel_indices = defaultdict(list)
    for i, k in enumerate(keys):
        voxel_indices[int(k)].append(i)

    return keys, dict(voxel_indices), bbox_min.astype(np.float32), float(voxel_size)


# ---------------------------------------------------------------------------
# Step 2: Voxel Merging
# ---------------------------------------------------------------------------

def _sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def _inverse_sigmoid(x):
    """Inverse sigmoid (logit). Input must be in (0, 1)."""
    x = np.clip(x, 1e-7, 1.0 - 1e-7)
    return np.log(x / (1.0 - x))


def merge_gaussians(attrs, voxel_indices, merge_threshold=3, variance_threshold=None,
                    voxel_size=1.0):
    """Merge Gaussians within dense voxels.

    For voxels with > merge_threshold Gaussians where position variance is low,
    merge into a single representative Gaussian:
    - Position: opacity-weighted average
    - SH DC: opacity-weighted average
    - SH rest: opacity-weighted average
    - Scale: mean in log-space (geometric mean in linear space)
    - Rotation: opacity-weighted average quaternion (LERP + normalize)
    - Opacity: combined transparency: 1 - prod(1 - o_i)

    Sparse voxels (<= merge_threshold) keep all Gaussians unchanged.

    Args:
        attrs: dict with xyz, sh_dc, sh_rest, opacity, scales, rotations.
        voxel_indices: dict mapping voxel_key -> list of Gaussian indices.
        merge_threshold: minimum Gaussians in a voxel to trigger merging.
        variance_threshold: max position variance for merging (default: 2*voxel_size^2).
        voxel_size: voxel size in world units (for default variance_threshold).

    Returns:
        merged_attrs: dict with same keys but potentially fewer Gaussians.
        merge_stats: dict with n_original, n_merged, n_kept, ratio.
    """
    if variance_threshold is None:
        variance_threshold = 2.0 * voxel_size ** 2

    xyz = attrs['xyz']
    sh_dc = attrs['sh_dc']
    sh_rest = attrs['sh_rest']
    opacity_raw = attrs['opacity']  # pre-sigmoid, (N, 1)
    scales = attrs['scales']        # log-space, (N, 3)
    rotations = attrs['rotations']  # quaternions, (N, 4)

    N = xyz.shape[0]

    # Track which Gaussians are kept as-is vs merged
    kept_indices = []       # individual Gaussians kept unchanged
    merged_results = []     # list of merged Gaussian attribute tuples

    n_voxels_merged = 0
    n_gaussians_merged = 0

    for voxel_key, indices in voxel_indices.items():
        indices = np.array(indices, dtype=np.int64)

        if len(indices) <= merge_threshold:
            # Sparse voxel: keep all Gaussians
            kept_indices.extend(indices.tolist())
            continue

        # Check position variance within voxel
        voxel_xyz = xyz[indices]
        pos_var = np.var(voxel_xyz, axis=0).sum()

        if pos_var > variance_threshold:
            # High variance: don't merge, keep all
            kept_indices.extend(indices.tolist())
            continue

        # Merge this voxel's Gaussians into one
        n_voxels_merged += 1
        n_gaussians_merged += len(indices)

        # Compute opacity weights (sigmoid space)
        opa_sigmoid = _sigmoid(opacity_raw[indices].flatten())  # (K,)
        weights = opa_sigmoid / (opa_sigmoid.sum() + 1e-10)     # normalized

        # Position: opacity-weighted average
        merged_xyz = np.sum(weights[:, None] * voxel_xyz, axis=0)

        # SH DC: opacity-weighted average
        merged_dc = np.sum(weights[:, None] * sh_dc[indices], axis=0)

        # SH rest: opacity-weighted average
        merged_sh_rest = np.sum(weights[:, None] * sh_rest[indices], axis=0)

        # Scale: mean in log-space (= geometric mean in linear space)
        merged_scales = np.mean(scales[indices], axis=0)

        # Rotation: opacity-weighted LERP + normalize
        # Handle quaternion sign flips: align all to first quaternion
        voxel_rots = rotations[indices].copy()
        ref_q = voxel_rots[0]
        for j in range(1, len(voxel_rots)):
            if np.dot(voxel_rots[j], ref_q) < 0:
                voxel_rots[j] = -voxel_rots[j]
        q_avg = np.sum(weights[:, None] * voxel_rots, axis=0)
        q_norm = np.linalg.norm(q_avg)
        if q_norm < 1e-10:
            q_avg = ref_q
        else:
            q_avg = q_avg / q_norm

        # Opacity: combined transparency = 1 - prod(1 - o_i)
        combined_opacity = 1.0 - np.prod(1.0 - opa_sigmoid)
        combined_opacity = np.clip(combined_opacity, 1e-7, 1.0 - 1e-7)
        merged_opacity = _inverse_sigmoid(combined_opacity)

        merged_results.append((
            merged_xyz.astype(np.float32),
            merged_dc.astype(np.float32),
            merged_sh_rest.astype(np.float32),
            np.float32(merged_opacity),
            merged_scales.astype(np.float32),
            q_avg.astype(np.float32),
        ))

    # Assemble output arrays
    kept_indices = np.array(kept_indices, dtype=np.int64)
    n_kept = len(kept_indices)
    n_new = len(merged_results)
    n_total = n_kept + n_new

    out_xyz = np.empty((n_total, 3), dtype=np.float32)
    out_dc = np.empty((n_total, sh_dc.shape[1]), dtype=np.float32)
    out_sh_rest = np.empty((n_total, sh_rest.shape[1]), dtype=np.float32)
    out_opacity = np.empty((n_total, 1), dtype=np.float32)
    out_scales = np.empty((n_total, scales.shape[1]), dtype=np.float32)
    out_rotations = np.empty((n_total, rotations.shape[1]), dtype=np.float32)

    # Copy kept Gaussians
    if n_kept > 0:
        out_xyz[:n_kept] = xyz[kept_indices]
        out_dc[:n_kept] = sh_dc[kept_indices]
        out_sh_rest[:n_kept] = sh_rest[kept_indices]
        out_opacity[:n_kept] = opacity_raw[kept_indices]
        out_scales[:n_kept] = scales[kept_indices]
        out_rotations[:n_kept] = rotations[kept_indices]

    # Copy merged Gaussians
    for i, (m_xyz, m_dc, m_sh, m_opa, m_sc, m_rot) in enumerate(merged_results):
        idx = n_kept + i
        out_xyz[idx] = m_xyz
        out_dc[idx] = m_dc
        out_sh_rest[idx] = m_sh
        out_opacity[idx, 0] = m_opa
        out_scales[idx] = m_sc
        out_rotations[idx] = m_rot

    merged_attrs = {
        'xyz': out_xyz,
        'sh_dc': out_dc,
        'sh_rest': out_sh_rest,
        'opacity': out_opacity,
        'scales': out_scales,
        'rotations': out_rotations,
    }

    merge_stats = {
        'n_original': N,
        'n_kept': n_kept,
        'n_merged_gaussians': n_gaussians_merged,
        'n_merged_voxels': n_voxels_merged,
        'n_new_from_merge': n_new,
        'n_output': n_total,
        'reduction_ratio': 1.0 - n_total / N if N > 0 else 0.0,
    }

    return merged_attrs, merge_stats


# ---------------------------------------------------------------------------
# Step 3: Anchor + Residual Position Coding
# ---------------------------------------------------------------------------

def encode_positions_anchor(positions, bbox_min, voxel_size, grid_resolution,
                            offset_bits=4):
    """Encode positions as voxel anchor (grid coords) + local offset (residual).

    Anchor: which voxel (stored as uint16 grid coordinates).
    Offset: position within voxel, normalized to [0, 1)^3, quantized to offset_bits.

    Total per position: 3 * 2 bytes (anchor) + 3 * ceil(offset_bits/8) bytes (offset)
    At offset_bits=4: 6 + 3 = 9 bytes (but offset is packed 4 bits each -> 1.5 bytes)

    Args:
        positions: (N, 3) float32 positions.
        bbox_min: (3,) float32 bounding box minimum.
        voxel_size: float voxel side length.
        grid_resolution: int max grid dimension.
        offset_bits: int bits per offset coordinate (default 4).

    Returns:
        grid_coords: (N, 3) uint16 voxel coordinates.
        offset_quantized: (N, 3) uint8 local offset indices.
    """
    grid_coords = ((positions - bbox_min) / voxel_size).astype(np.int32)
    grid_coords = np.clip(grid_coords, 0, grid_resolution - 1)

    # Local offset within voxel [0, 1)
    voxel_origin = bbox_min + grid_coords * voxel_size
    local_offset = (positions - voxel_origin) / voxel_size
    local_offset = np.clip(local_offset, 0.0, 1.0 - 1e-6)

    # Quantize offset
    n_levels = 2 ** offset_bits
    offset_quantized = (local_offset * n_levels).astype(np.uint8)
    offset_quantized = np.clip(offset_quantized, 0, n_levels - 1)

    return grid_coords.astype(np.uint16), offset_quantized


def decode_positions_anchor(grid_coords, offset_quantized, bbox_min, voxel_size,
                            offset_bits=4):
    """Reconstruct positions from anchor + offset encoding.

    Args:
        grid_coords: (N, 3) uint16 voxel coordinates.
        offset_quantized: (N, 3) uint8 local offset indices.
        bbox_min: (3,) float32 bounding box minimum.
        voxel_size: float voxel side length.
        offset_bits: int bits per offset coordinate.

    Returns:
        positions: (N, 3) float32 reconstructed positions.
    """
    n_levels = 2 ** offset_bits
    # Dequantize offset to [0, 1) center of each quantization bin
    local_offset = (offset_quantized.astype(np.float32) + 0.5) / n_levels

    # Reconstruct: bbox_min + grid_coord * voxel_size + local_offset * voxel_size
    positions = (bbox_min +
                 grid_coords.astype(np.float32) * voxel_size +
                 local_offset * voxel_size)
    return positions.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 4: Delta Rotation Coding
# ---------------------------------------------------------------------------

def encode_rotations_delta(rotations, sort_order):
    """Delta-encode quaternion rotations after Morton sorting.

    After Morton sorting, consecutive Gaussians are spatially nearby, so their
    rotations tend to be similar. Delta encoding exploits this for better
    compression under zstd/deflate.

    Args:
        rotations: (N, 4) quaternions (not necessarily normalized).
        sort_order: (N,) int indices that put Gaussians in Morton order.

    Returns:
        deltas: (N, 4) float32 rotation deltas (first row is absolute).
    """
    sorted_rots = rotations[sort_order].copy()

    # Ensure consistent hemisphere: flip if dot product with previous is negative
    for i in range(1, len(sorted_rots)):
        if np.dot(sorted_rots[i], sorted_rots[i - 1]) < 0:
            sorted_rots[i] = -sorted_rots[i]

    # Delta encode
    deltas = np.empty_like(sorted_rots)
    deltas[0] = sorted_rots[0]
    deltas[1:] = sorted_rots[1:] - sorted_rots[:-1]

    return deltas


def decode_rotations_delta(deltas):
    """Reconstruct rotations from delta encoding.

    Args:
        deltas: (N, 4) float32 rotation deltas.

    Returns:
        rotations: (N, 4) float32 reconstructed quaternions.
    """
    rotations = np.cumsum(deltas, axis=0)
    return rotations.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 5: Full Compression Pipeline
# ---------------------------------------------------------------------------

def compress_v3(ply_path, output_path,
                grid_resolution=512, merge_threshold=3,
                sh_bits=2, sh_degree=2,
                pos_offset_bits=4,
                scale_bits=8, rot_bits=8, opacity_bits=8,
                dc_bits=8, seed=0, prune_ratio=0.0):
    """TurboSplat v3: Voxel merge + anchor coding + TurboQuant SH.

    Args:
        ply_path: Path to input PLY file.
        output_path: Path for output .npz file.
        grid_resolution: Voxel grid resolution per axis.
        merge_threshold: Min Gaussians in a voxel to trigger merging.
        sh_bits: Bit-width for TurboQuant SH quantization.
        sh_degree: Max SH degree to keep (0-3).
        pos_offset_bits: Bits for position offset within voxel.
        scale_bits: Bits for scale uniform quantization.
        rot_bits: Bits for rotation uniform quantization.
        opacity_bits: Bits for opacity uniform quantization.
        dc_bits: Bits for SH DC uniform quantization.
        seed: Random seed for TurboQuantizer.
        prune_ratio: Fraction of low-importance Gaussians to remove (0.0=none).

    Returns:
        stats dict with compression results.
    """
    t0 = time.perf_counter()

    # Load PLY
    attrs = load_ply_attributes(ply_path)
    n_original = attrs['xyz'].shape[0]
    original_size = sum(a.nbytes for a in attrs.values())

    print(f"  Loaded: {n_original:,} Gaussians, {original_size / 1e6:.1f} MB")

    # Step 0: Prune low-importance Gaussians
    if prune_ratio > 0:
        attrs = _prune_gaussians(attrs, prune_ratio)
        n_after_prune = attrs['xyz'].shape[0]
        print(f"  Pruned: {n_original:,} -> {n_after_prune:,} Gaussians "
              f"({prune_ratio * 100:.0f}% removed)")
    else:
        n_after_prune = n_original

    # Step 1: Voxel assignment
    voxel_keys, voxel_indices, bbox_min, voxel_size = assign_voxels(
        attrs['xyz'], grid_resolution=grid_resolution)
    n_voxels = len(voxel_indices)
    print(f"  Voxel grid: resolution={grid_resolution}, "
          f"voxel_size={voxel_size:.6f}, {n_voxels:,} occupied voxels")

    # Step 2: Merge dense voxels
    if merge_threshold > 0:
        attrs, merge_stats = merge_gaussians(
            attrs, voxel_indices,
            merge_threshold=merge_threshold,
            voxel_size=voxel_size)
        n_merged = attrs['xyz'].shape[0]
        print(f"  Merged: {n_after_prune:,} -> {n_merged:,} Gaussians "
              f"({merge_stats['reduction_ratio'] * 100:.1f}% reduction, "
              f"{merge_stats['n_merged_voxels']:,} voxels merged)")
    else:
        merge_stats = {'n_original': n_after_prune, 'n_output': n_after_prune,
                       'reduction_ratio': 0.0, 'n_merged_voxels': 0,
                       'n_merged_gaussians': 0, 'n_kept': n_after_prune,
                       'n_new_from_merge': 0}
        n_merged = n_after_prune

    # Step 3: SH truncation
    if sh_degree < 3:
        old_d = attrs['sh_rest'].shape[1]
        attrs = _truncate_sh(attrs, sh_degree)
        new_d = attrs['sh_rest'].shape[1]
        print(f"  SH truncated: {old_d} -> {new_d} dims (degree {sh_degree})")

    # Step 4: Re-assign voxels after merge + Morton sort
    new_keys, _, new_bbox_min, new_voxel_size = assign_voxels(
        attrs['xyz'], voxel_size=voxel_size, grid_resolution=grid_resolution)
    sort_order = np.argsort(new_keys)
    for key in attrs:
        attrs[key] = attrs[key][sort_order]
    print(f"  Morton-sorted {n_merged:,} Gaussians")

    # Step 5: Encode positions as anchor + offset
    grid_coords, offset_q = encode_positions_anchor(
        attrs['xyz'], bbox_min, voxel_size, grid_resolution,
        offset_bits=pos_offset_bits)

    # Step 6: Rotation encoding
    # Note: 3DGS stores un-normalized quaternions (renderer normalizes internally).
    # Delta coding via cumsum amplifies quantization error over long sequences,
    # so we use direct uniform quantization. Morton sorting still benefits zstd
    # compression by improving spatial coherence of adjacent values.

    # Step 7: TurboQuant on SH rest
    sh_rest = attrs['sh_rest']
    d_sh = sh_rest.shape[1]
    if d_sh > 0:
        tq = TurboQuantizer(d=d_sh, b=sh_bits, seed=seed)
        sh_idx, sh_norms = tq.quantize(sh_rest)
        sh_norms = sh_norms.astype(np.float16)
    else:
        sh_idx = np.zeros((n_merged, 0), dtype=np.uint8)
        sh_norms = np.zeros(n_merged, dtype=np.float16)

    # Step 8: Uniform quantize other attributes (downcast to uint8 when <= 8 bits)
    dc_idx, dc_min, dc_scale = _uniform_quantize(attrs['sh_dc'], dc_bits)
    dc_idx = _downcast_indices(dc_idx, dc_bits)
    scale_idx, sc_min, sc_scale = _uniform_quantize(attrs['scales'], scale_bits)
    scale_idx = _downcast_indices(scale_idx, scale_bits)
    opa_idx, op_min, op_scale = _uniform_quantize(attrs['opacity'], opacity_bits)
    opa_idx = _downcast_indices(opa_idx, opacity_bits)

    # Quantize rotations directly (Morton sort helps zstd compress spatial coherence)
    rot_idx, rt_min, rt_scale = _uniform_quantize(attrs['rotations'], rot_bits)
    rot_idx = _downcast_indices(rot_idx, rot_bits)

    # Step 9: Build save dict
    save_dict = {
        # Position encoding
        'grid_coords': grid_coords,
        'offset_q': offset_q,
        # SH TurboQuant
        'sh_idx': sh_idx,
        'sh_norms': sh_norms,
        # DC uniform quantization
        'dc_idx': dc_idx,
        'dc_min': np.float32(dc_min),
        'dc_scale': np.float32(dc_scale),
        # Scale uniform quantization
        'scale_idx': scale_idx,
        'sc_min': np.float32(sc_min),
        'sc_scale': np.float32(sc_scale),
        # Rotation delta uniform quantization
        'rot_idx': rot_idx,
        'rt_min': np.float32(rt_min),
        'rt_scale': np.float32(rt_scale),
        # Opacity uniform quantization
        'opa_idx': opa_idx,
        'op_min': np.float32(op_min),
        'op_scale': np.float32(op_scale),
        # Metadata for decompression
        'bbox_min': bbox_min.astype(np.float32),
        'voxel_size': np.float32(voxel_size),
        'grid_resolution': np.int32(grid_resolution),
        'n_gaussians': np.int32(n_merged),
        'n_original': np.int32(n_original),
        'sh_bits': np.uint8(sh_bits),
        'sh_degree': np.uint8(sh_degree),
        'dc_bits': np.uint8(dc_bits),
        'scale_bits': np.uint8(scale_bits),
        'rot_bits': np.uint8(rot_bits),
        'opacity_bits': np.uint8(opacity_bits),
        'pos_offset_bits': np.uint8(pos_offset_bits),
        'prune_ratio': np.float32(prune_ratio),
        'version': np.uint8(3),  # v3 format marker
    }

    # Add TurboQuant metadata for SH dequantization
    if d_sh > 0:
        save_dict['sh_rotation'] = tq.get_rotation_matrix().astype(np.float32)
        save_dict['sh_centroids'] = tq.get_centroids().astype(np.float32)
        save_dict['sh_d'] = np.int32(d_sh)
    else:
        save_dict['sh_rotation'] = np.zeros((0, 0), dtype=np.float32)
        save_dict['sh_centroids'] = np.zeros(0, dtype=np.float32)
        save_dict['sh_d'] = np.int32(0)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(output_path, **save_dict)

    compression_time = time.perf_counter() - t0
    compressed_size = os.path.getsize(output_path)
    ratio = original_size / compressed_size

    print(f"\n  TurboSplat v3 Compression Results:")
    print(f"    Original:    {n_original:,} Gaussians, {original_size / 1e6:.1f} MB")
    print(f"    After merge: {n_merged:,} Gaussians "
          f"({merge_stats['reduction_ratio'] * 100:.1f}% fewer)")
    print(f"    Compressed:  {compressed_size / 1e6:.2f} MB")
    print(f"    Ratio:       {ratio:.1f}x")
    print(f"    Time:        {compression_time * 1000:.0f} ms")

    return {
        'n_original': n_original,
        'n_merged': n_merged,
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_ratio': ratio,
        'merge_reduction': merge_stats['reduction_ratio'],
        'merge_stats': merge_stats,
        'compression_time_s': compression_time,
    }


# ---------------------------------------------------------------------------
# Step 6: Decompression
# ---------------------------------------------------------------------------

def decompress_v3(compressed_path):
    """Decompress a v3 .npz file back to 3DGS attributes.

    Reconstructs:
    - Positions from anchor grid_coords + quantized offset
    - SH rest via TurboQuant inverse (centroid lookup + inverse rotation)
    - SH DC, scales, opacity via uniform dequantization
    - Rotations via cumulative sum of delta-decoded quaternions

    Zero-pads SH rest back to 45 dims for renderer compatibility.

    Args:
        compressed_path: Path to the .npz file.

    Returns:
        Dict with keys: xyz (N,3), sh_dc (N,3), sh_rest (N,45),
        opacity (N,1), scales (N,3), rotations (N,4).
    """
    data = np.load(compressed_path)

    N = int(data['n_gaussians'])
    d_sh = int(data['sh_d'])
    sh_degree = int(data['sh_degree'])

    # --- Positions: anchor + offset ---
    bbox_min = data['bbox_min']
    voxel_size = float(data['voxel_size'])
    pos_offset_bits = int(data['pos_offset_bits'])
    grid_coords = data['grid_coords']
    offset_q = data['offset_q']

    xyz = decode_positions_anchor(grid_coords, offset_q, bbox_min, voxel_size,
                                  offset_bits=pos_offset_bits)

    # --- SH rest: TurboQuant inverse ---
    if d_sh > 0:
        sh_indices = data['sh_idx']                              # (N, D) uint8
        sh_norms = data['sh_norms'].astype(np.float32)           # (N,)
        sh_rotation = data['sh_rotation'].astype(np.float64)     # (D, D)
        sh_centroids = data['sh_centroids'].astype(np.float64)   # (2^b,)

        # Centroid lookup
        y_hat = sh_centroids[sh_indices]  # (N, D)

        # Inverse rotation: x_hat = Y_hat @ R
        x_hat = y_hat @ sh_rotation  # (N, D)

        # Rescale by norms
        sh_rest_truncated = (x_hat * sh_norms[:, np.newaxis]).astype(np.float32)
    else:
        sh_rest_truncated = np.zeros((N, 0), dtype=np.float32)

    # Zero-pad to full 45-dim sh_rest
    full_sh_dim = SH_BAND_DIMS[3]  # 45
    if d_sh < full_sh_dim:
        sh_rest = np.zeros((N, full_sh_dim), dtype=np.float32)
        if d_sh > 0:
            sh_rest[:, :d_sh] = sh_rest_truncated
    else:
        sh_rest = sh_rest_truncated

    # --- Uniform dequantize DC, scales, opacity ---
    dc_bits = int(data['dc_bits'])
    sh_dc = _uniform_dequantize(data['dc_idx'], float(data['dc_min']),
                                float(data['dc_scale']), dc_bits)
    sh_dc = sh_dc.reshape(N, 3)

    scale_bits = int(data['scale_bits'])
    scales = _uniform_dequantize(data['scale_idx'], float(data['sc_min']),
                                 float(data['sc_scale']), scale_bits)
    scales = scales.reshape(N, 3)

    opacity_bits = int(data['opacity_bits'])
    opacity = _uniform_dequantize(data['opa_idx'], float(data['op_min']),
                                  float(data['op_scale']), opacity_bits)
    opacity = opacity.reshape(N, 1)

    # --- Rotations: direct uniform dequantization ---
    rot_bits = int(data['rot_bits'])
    rotations = _uniform_dequantize(data['rot_idx'], float(data['rt_min']),
                                    float(data['rt_scale']), rot_bits)
    rotations = rotations.reshape(N, 4)

    return {
        'xyz': xyz,
        'sh_dc': sh_dc,
        'sh_rest': sh_rest,
        'opacity': opacity,
        'scales': scales,
        'rotations': rotations,
    }


# ---------------------------------------------------------------------------
# PLY saving (reuse decompress.py logic)
# ---------------------------------------------------------------------------

def save_ply(attrs, output_path):
    """Save reconstructed attributes to a PLY file compatible with 3DGS."""
    from decompress import save_ply as _save_ply
    _save_ply(attrs, output_path)


# ---------------------------------------------------------------------------
# Quality evaluation (quick PSNR on test views)
# ---------------------------------------------------------------------------

def evaluate_quick(model_path, source_path, decompressed_ply_path,
                   iteration=30000, white_background=True, max_views=10):
    """Quick PSNR evaluation using test views.

    Args:
        model_path: Path to original trained model directory.
        source_path: Path to scene data directory.
        decompressed_ply_path: Path to decompressed PLY file.
        iteration: Training iteration of the original model.
        white_background: Whether to use white background.
        max_views: Maximum number of test views to render.

    Returns:
        dict with psnr_orig, psnr_comp, psnr_drop, n_views.
    """
    try:
        import torch
        from eval_compression import (
            _load_gaussians_from_ply,
            _load_scene_cameras,
            psnr,
        )
        from gaussian_renderer import render as gs_render
    except ImportError as e:
        return {'error': f'Missing dependency: {e}'}

    try:
        with torch.no_grad():
            scene, gaussians_orig, pp, background = _load_scene_cameras(
                model_path, source_path, white_background,
                sh_degree=3, iteration=iteration)

            gaussians_comp = _load_gaussians_from_ply(decompressed_ply_path, sh_degree=3)
            gaussians_comp.active_sh_degree = gaussians_orig.active_sh_degree

            test_cameras = scene.getTestCameras()
            if len(test_cameras) == 0:
                return {'error': 'No test cameras found'}

            # Limit views
            views = test_cameras[:max_views]

            psnr_orig_list = []
            psnr_comp_list = []

            for view in views:
                out_orig = gs_render(view, gaussians_orig, pp, background)
                rendered_orig = out_orig['render']
                gt = view.original_image[0:3, :, :].cuda()

                out_comp = gs_render(view, gaussians_comp, pp, background)
                rendered_comp = out_comp['render']

                psnr_orig_list.append(psnr(rendered_orig, gt))
                psnr_comp_list.append(psnr(rendered_comp, gt))

            p_orig = float(np.mean(psnr_orig_list))
            p_comp = float(np.mean(psnr_comp_list))

            return {
                'psnr_orig': p_orig,
                'psnr_comp': p_comp,
                'psnr_drop': p_comp - p_orig,
                'n_views': len(views),
            }
    except Exception as e:
        import traceback
        return {'error': f'{type(e).__name__}: {e}',
                'traceback': traceback.format_exc()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TurboSplat v3: Voxel merging + anchor coding + TurboQuant SH compression"
    )
    parser.add_argument(
        "-m", "--model_path", required=True,
        help="Path to trained 3DGS model directory",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output .npz path (default: compressed/<model_name>_v3.npz)",
    )
    parser.add_argument("--iteration", type=int, default=30000,
                        help="Training iteration to load (default: 30000)")
    parser.add_argument("--grid_resolution", type=int, default=512,
                        help="Voxel grid resolution per axis (default: 512)")
    parser.add_argument("--merge_threshold", type=int, default=3,
                        help="Min Gaussians per voxel to trigger merge (default: 3)")
    parser.add_argument("--sh_bits", type=int, default=2,
                        help="TurboQuant bit-width for SH rest (default: 2)")
    parser.add_argument("--sh_degree", type=int, default=2,
                        help="Max SH degree to keep: 0-3 (default: 2)")
    parser.add_argument("--scale_bits", type=int, default=8,
                        help="Bit-width for scale quantization (default: 8)")
    parser.add_argument("--rot_bits", type=int, default=8,
                        help="Bit-width for rotation delta quantization (default: 8)")
    parser.add_argument("--opacity_bits", type=int, default=8,
                        help="Bit-width for opacity quantization (default: 8)")
    parser.add_argument("--dc_bits", type=int, default=8,
                        help="Bit-width for DC coefficient quantization (default: 8)")
    parser.add_argument("--pos_offset_bits", type=int, default=4,
                        help="Bits for position offset within voxel (default: 4)")
    parser.add_argument("--prune_ratio", type=float, default=0.0,
                        help="Fraction of low-importance Gaussians to prune (default: 0.0)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for TurboQuantizer (default: 0)")
    parser.add_argument("--eval", action="store_true",
                        help="Run quick PSNR evaluation after compression")
    parser.add_argument("--data_root", default="data/nerf_synthetic",
                        help="Root directory for scene data (for --eval)")
    parser.add_argument("--max_views", type=int, default=10,
                        help="Max test views for quick eval (default: 10)")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep merge_threshold values [0, 2, 3, 5, 8] and print table")
    args = parser.parse_args()

    # Resolve PLY path
    pc_dir = os.path.join(args.model_path, "point_cloud")
    if args.iteration is not None:
        ply_path = os.path.join(pc_dir, f"iteration_{args.iteration}",
                                "point_cloud.ply")
    else:
        iters = []
        if os.path.isdir(pc_dir):
            for d in os.listdir(pc_dir):
                if d.startswith("iteration_"):
                    try:
                        iters.append(int(d.split("_")[1]))
                    except ValueError:
                        pass
        if not iters:
            if args.model_path.endswith(".ply"):
                ply_path = args.model_path
            else:
                raise FileNotFoundError(f"No iterations found in {pc_dir}")
        else:
            ply_path = os.path.join(pc_dir, f"iteration_{max(iters)}",
                                    "point_cloud.ply")

    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    # Auto-generate output path
    if args.output is None:
        model_name = os.path.basename(os.path.normpath(args.model_path))
        args.output = os.path.join("compressed", f"{model_name}_v3.npz")

    # Resolve paths relative to script dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.output):
        args.output = os.path.join(script_dir, args.output)
    if not os.path.isabs(args.data_root):
        args.data_root = os.path.join(script_dir, args.data_root)

    if args.sweep:
        # Sweep multiple merge_threshold values
        thresholds = [0, 2, 3, 5, 8]
        results = []

        for mt in thresholds:
            # Generate unique output path per threshold
            base, ext = os.path.splitext(args.output)
            out_path = f"{base}_mt{mt}{ext}"

            print(f"\n{'=' * 60}")
            print(f"  merge_threshold = {mt}")
            print(f"{'=' * 60}")

            stats = compress_v3(
                ply_path, out_path,
                grid_resolution=args.grid_resolution,
                merge_threshold=mt,
                sh_bits=args.sh_bits,
                sh_degree=args.sh_degree,
                pos_offset_bits=args.pos_offset_bits,
                scale_bits=args.scale_bits,
                rot_bits=args.rot_bits,
                opacity_bits=args.opacity_bits,
                dc_bits=args.dc_bits,
                seed=args.seed,
                prune_ratio=args.prune_ratio,
            )

            row = {
                'merge_threshold': mt,
                'n_original': stats['n_original'],
                'n_merged': stats['n_merged'],
                'merge_pct': stats['merge_reduction'] * 100,
                'compressed_MB': stats['compressed_size'] / 1e6,
                'ratio': stats['compression_ratio'],
                'output_path': out_path,
            }

            # Optional quick PSNR eval
            if args.eval:
                import tempfile
                with tempfile.TemporaryDirectory() as tmpdir:
                    dec_ply = os.path.join(tmpdir, "decompressed.ply")
                    dec_attrs = decompress_v3(out_path)
                    save_ply(dec_attrs, dec_ply)

                    scene_name = os.path.basename(os.path.normpath(args.model_path))
                    # Strip suffix like _wb
                    for suffix in ['_wb', '_sqr', '_test']:
                        if scene_name.endswith(suffix):
                            scene_name = scene_name[:-len(suffix)]
                            break
                    source_path = os.path.join(args.data_root, scene_name)

                    qresult = evaluate_quick(
                        args.model_path, source_path, dec_ply,
                        iteration=args.iteration, max_views=args.max_views)

                    if 'error' not in qresult:
                        row['psnr_orig'] = qresult['psnr_orig']
                        row['psnr_comp'] = qresult['psnr_comp']
                        row['psnr_drop'] = qresult['psnr_drop']
                    else:
                        row['error'] = qresult['error']

            results.append(row)

        # Print summary table
        print(f"\n{'=' * 80}")
        print("  TurboSplat v3 Sweep Results")
        print(f"{'=' * 80}")

        has_psnr = any('psnr_comp' in r for r in results)
        if has_psnr:
            header = (f"{'MT':>3}  {'N_orig':>9}  {'N_merged':>9}  {'Merge%':>6}  "
                      f"{'Size_MB':>7}  {'Ratio':>6}  {'PSNR_orig':>9}  "
                      f"{'PSNR_comp':>9}  {'Drop':>6}")
        else:
            header = (f"{'MT':>3}  {'N_orig':>9}  {'N_merged':>9}  {'Merge%':>6}  "
                      f"{'Size_MB':>7}  {'Ratio':>6}")
        print(header)
        print("-" * len(header))

        for r in results:
            line = (f"{r['merge_threshold']:>3}  {r['n_original']:>9,}  "
                    f"{r['n_merged']:>9,}  {r['merge_pct']:>5.1f}%  "
                    f"{r['compressed_MB']:>7.2f}  {r['ratio']:>5.1f}x")
            if has_psnr and 'psnr_comp' in r:
                line += (f"  {r['psnr_orig']:>9.2f}  {r['psnr_comp']:>9.2f}  "
                         f"{r['psnr_drop']:>+5.2f}")
            elif has_psnr:
                err = r.get('error', 'N/A')[:30]
                line += f"  {err}"
            print(line)

    else:
        # Single compression run
        print(f"\nTurboSplat v3 Compression")
        print(f"  Input:  {ply_path}")
        print(f"  Output: {args.output}")
        print(f"  Settings: grid={args.grid_resolution}, merge_threshold={args.merge_threshold}, "
              f"sh_bits={args.sh_bits}, sh_degree={args.sh_degree}")
        print(f"  Quant: dc={args.dc_bits}b, scale={args.scale_bits}b, "
              f"rot={args.rot_bits}b, opacity={args.opacity_bits}b, "
              f"pos_offset={args.pos_offset_bits}b")

        stats = compress_v3(
            ply_path, args.output,
            grid_resolution=args.grid_resolution,
            merge_threshold=args.merge_threshold,
            sh_bits=args.sh_bits,
            sh_degree=args.sh_degree,
            pos_offset_bits=args.pos_offset_bits,
            scale_bits=args.scale_bits,
            rot_bits=args.rot_bits,
            opacity_bits=args.opacity_bits,
            dc_bits=args.dc_bits,
            seed=args.seed,
            prune_ratio=args.prune_ratio,
        )

        # Optional quick eval
        if args.eval:
            import tempfile
            print(f"\n  Running quick PSNR evaluation ({args.max_views} views)...")
            with tempfile.TemporaryDirectory() as tmpdir:
                dec_ply = os.path.join(tmpdir, "decompressed.ply")
                dec_attrs = decompress_v3(args.output)
                save_ply(dec_attrs, dec_ply)

                scene_name = os.path.basename(os.path.normpath(args.model_path))
                for suffix in ['_wb', '_sqr', '_test']:
                    if scene_name.endswith(suffix):
                        scene_name = scene_name[:-len(suffix)]
                        break
                source_path = os.path.join(args.data_root, scene_name)

                qresult = evaluate_quick(
                    args.model_path, source_path, dec_ply,
                    iteration=args.iteration, max_views=args.max_views)

                if 'error' not in qresult:
                    print(f"    PSNR orig:       {qresult['psnr_orig']:.2f} dB")
                    print(f"    PSNR compressed: {qresult['psnr_comp']:.2f} dB")
                    print(f"    PSNR drop:       {qresult['psnr_drop']:+.2f} dB")
                    print(f"    Views:           {qresult['n_views']}")
                else:
                    print(f"    Eval error: {qresult['error']}")

        print(f"\n  Output saved to: {args.output}")


if __name__ == "__main__":
    main()
