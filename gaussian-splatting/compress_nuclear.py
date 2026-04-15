"""Nuclear compression for 3D Gaussian Splatting: PCA SH + Low-bit Scalar Quant.

Three training-free, CPU-only compression techniques stacked on top of TurboSplat:
  1. SH PCA Dimensionality Reduction (45D -> ~12D) then TurboQuant on fewer dims
  2. Low-bit scalar quantization for all attributes (Scale/Rot/Opacity/DC/Pos)
  3. Morton sorting for improved npz deflate compression

The main innovation is PCA reducing the 45D SH coefficients to ~12D before
TurboQuant, dramatically cutting the dominant storage term. Optionally,
grid-local K-means VQ can replace scalar quant for scale/rot/opacity.

Usage:
    python compress_nuclear.py -m output/lego_wb --pca 12 --sh_bits 2
    python compress_nuclear.py -m output/stump --pca 12 --sh_bits 2
    python compress_nuclear.py --sweep_scalar -m output/lego_wb
    python compress_nuclear.py --sweep -m output/lego_wb --use_vq
"""

import argparse
import json
import math
import os
import sys
import tempfile
import time

import numpy as np
from collections import defaultdict

from compress import load_ply_attributes, _uniform_quantize, _uniform_dequantize
from turbo_quant.quantizer import TurboQuantizer


# ---------------------------------------------------------------------------
# Technique 1: SH PCA Dimensionality Reduction
# ---------------------------------------------------------------------------

def sh_pca_compress(sh_rest, n_components=12):
    """Reduce SH dimensionality via PCA.

    Args:
        sh_rest: (N, D) float32 - SH rest coefficients (D is typically 45).
        n_components: number of PCA components to keep.

    Returns:
        projected: (N, n_components) float32 - projected coefficients.
        mean: (D,) float32 - mean vector.
        components: (n_components, D) float32 - PCA basis.
        explained_variance_ratio: float - cumulative variance explained.
    """
    D = sh_rest.shape[1]
    n_components = min(n_components, D)

    mean = sh_rest.mean(axis=0)
    centered = sh_rest - mean

    # Covariance: (D, D)
    cov = (centered.T @ centered) / max(len(centered) - 1, 1)

    # Eigendecomposition (eigh returns ascending order)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Keep top n_components
    components = eigenvectors[:, :n_components].T  # (n_components, D)

    # Project
    projected = centered @ components.T  # (N, n_components)

    # Variance explained
    total_var = max(eigenvalues.sum(), 1e-12)
    explained = np.cumsum(eigenvalues[:n_components]) / total_var

    return projected, mean.astype(np.float32), components.astype(np.float32), float(explained[-1])


def sh_pca_decompress(projected, mean, components):
    """Reconstruct SH from PCA projection."""
    return (projected @ components + mean).astype(np.float32)


# ---------------------------------------------------------------------------
# Technique 2: Grid-Local K-Means VQ for Scale/Rotation/Opacity
# ---------------------------------------------------------------------------

def grid_vq_compress(attrs, grid_resolution=32, n_clusters=256):
    """Vector-quantize Scale/Rot/Opacity using grid-local K-means.

    Args:
        attrs: dict with xyz, scales, rotations, opacity.
        grid_resolution: voxel grid resolution.
        n_clusters: clusters per voxel (256 = 8 bits).

    Returns:
        voxel_keys: (N,) int32 - which voxel each Gaussian belongs to.
        cluster_indices: (N,) uint8 - index within voxel's codebook.
        codebooks: dict mapping voxel_id -> (n_clusters, 8) centroids.
    """
    from scipy.cluster.vq import kmeans2

    xyz = attrs['xyz']
    attr_vectors = np.concatenate([
        attrs['scales'],       # (N, 3)
        attrs['rotations'],    # (N, 4)
        attrs['opacity'],      # (N, 1)
    ], axis=1).astype(np.float64)  # (N, 8)

    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    span = (bbox_max - bbox_min).max()
    voxel_size = max(span / grid_resolution, 1e-6)
    grid_coords = ((xyz - bbox_min) / voxel_size).astype(np.int32)
    grid_coords = np.clip(grid_coords, 0, grid_resolution - 1)
    voxel_keys = (grid_coords[:, 0].astype(np.int64) * grid_resolution * grid_resolution +
                  grid_coords[:, 1].astype(np.int64) * grid_resolution +
                  grid_coords[:, 2].astype(np.int64)).astype(np.int32)

    voxel_groups = defaultdict(list)
    for i, k in enumerate(voxel_keys):
        voxel_groups[int(k)].append(i)

    cluster_indices = np.zeros(len(xyz), dtype=np.uint8)
    codebooks = {}

    for voxel_id, indices in voxel_groups.items():
        indices = np.array(indices)
        vectors = attr_vectors[indices]

        actual_k = min(n_clusters, len(indices))
        if len(indices) <= actual_k:
            centroids = vectors.copy()
            labels = np.arange(len(indices))
        else:
            col_std = vectors.std(axis=0)
            col_std[col_std < 1e-8] = 1.0
            normalized = vectors / col_std
            try:
                centroids_norm, labels = kmeans2(
                    normalized, actual_k, minit='points', iter=10, seed=42
                )
                centroids = centroids_norm * col_std
            except Exception:
                sample_idx = np.linspace(0, len(indices) - 1, actual_k, dtype=int)
                centroids = vectors[sample_idx].copy()
                col_std_safe = col_std.copy()
                col_std_safe[col_std_safe < 1e-8] = 1.0
                norm_v = vectors / col_std_safe
                norm_c = centroids / col_std_safe
                dists = np.linalg.norm(
                    norm_v[:, np.newaxis, :] - norm_c[np.newaxis, :, :], axis=2
                )
                labels = np.argmin(dists, axis=1)

        codebooks[int(voxel_id)] = centroids.astype(np.float32)
        cluster_indices[indices] = labels.astype(np.uint8)

    return voxel_keys, cluster_indices, codebooks


def grid_vq_decompress(voxel_keys, cluster_indices, codebooks):
    """Reconstruct Scale/Rot/Opacity from VQ indices."""
    n = len(voxel_keys)
    reconstructed = np.zeros((n, 8), dtype=np.float32)

    unique_voxels = np.unique(voxel_keys)
    for voxel_id in unique_voxels:
        voxel_id = int(voxel_id)
        mask = voxel_keys == voxel_id
        cb = codebooks.get(voxel_id)
        if cb is None:
            continue
        local_indices = cluster_indices[mask].astype(np.int32)
        local_indices = np.clip(local_indices, 0, len(cb) - 1)
        reconstructed[mask] = cb[local_indices]

    scales = reconstructed[:, :3]
    rotations = reconstructed[:, 3:7]
    opacity = reconstructed[:, 7:8]
    return scales, rotations, opacity


# ---------------------------------------------------------------------------
# Morton Sort for compression-friendly ordering
# ---------------------------------------------------------------------------

def _morton_sort_order(positions, grid_bits=10):
    """Compute Morton-curve sort order for positions.

    Reordering Gaussians spatially improves deflate compression of
    quantized indices because nearby Gaussians tend to have similar values.
    """
    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    span = np.maximum(bbox_max - bbox_min, 1e-6)
    grid_max = (1 << grid_bits) - 1

    normalized = (positions - bbox_min) / span
    grid_coords = (normalized * grid_max).astype(np.int64)
    grid_coords = np.clip(grid_coords, 0, grid_max)

    R = np.int64(grid_max + 1)
    keys = grid_coords[:, 0] * R * R + grid_coords[:, 1] * R + grid_coords[:, 2]
    return np.argsort(keys)


# ---------------------------------------------------------------------------
# Full Nuclear Pipeline: Compress
# ---------------------------------------------------------------------------

def compress_nuclear(ply_path, output_path,
                     pca_components=12, sh_bits=2,
                     dc_bits=10, pos_bits=16,
                     scale_bits=8, rot_bits=8, opacity_bits=8,
                     use_vq=False, grid_resolution=32, vq_clusters=256):
    """Nuclear compression pipeline: PCA + scalar quant + Morton sort.

    The main compression gain comes from PCA reducing SH from 45D to ~12D
    before TurboQuant. Additional savings from low-bit scalar quantization
    and Morton sorting for better deflate.

    Args:
        ply_path: Path to input PLY file.
        output_path: Path to output .npz file.
        pca_components: Number of PCA components to keep.
        sh_bits: Bit-width for TurboQuant on PCA-projected SH.
        dc_bits: Bit-width for SH DC.
        pos_bits: Bit-width for positions.
        scale_bits: Bit-width for scales.
        rot_bits: Bit-width for rotations.
        opacity_bits: Bit-width for opacity.
        use_vq: If True, use grid-local VQ for scale/rot/opacity.
        grid_resolution: Voxel grid resolution for VQ.
        vq_clusters: K-means clusters per voxel for VQ.

    Returns:
        Dict with compression stats.
    """
    t0 = time.time()

    attrs = load_ply_attributes(ply_path)
    n_original = attrs['xyz'].shape[0]
    original_size = sum(a.nbytes for a in attrs.values())

    print(f"  Loaded: {n_original:,} Gaussians, {original_size / 1e6:.1f} MB")

    # 1. SH PCA
    sh_rest = attrs['sh_rest']
    sh_dim = sh_rest.shape[1]
    actual_pca = min(pca_components, sh_dim)
    projected, sh_mean, sh_components, var_explained = sh_pca_compress(sh_rest, actual_pca)
    print(f"  PCA: {sh_dim}D -> {actual_pca}D ({var_explained * 100:.1f}% variance)")

    # 2. TurboQuant on PCA-projected SH
    tq = TurboQuantizer(d=actual_pca, b=sh_bits, seed=0)
    sh_idx, sh_norms = tq.quantize(projected)
    sh_norms = sh_norms.astype(np.float16)

    # 3. Morton sort for compression-friendly ordering
    sort_order = _morton_sort_order(attrs['xyz'])

    # 4. Scalar quantize positions, DC
    pos_idx, pos_vmin, pos_vscale = _uniform_quantize(attrs['xyz'], pos_bits)
    dc_idx, dc_vmin, dc_vscale = _uniform_quantize(attrs['sh_dc'], dc_bits)

    # 5. Scale/Rot/Opacity
    if use_vq:
        print(f"  VQ: grid={grid_resolution}, K={vq_clusters} ...")
        t_vq = time.time()
        voxel_keys, vq_indices, codebooks = grid_vq_compress(
            attrs, grid_resolution, vq_clusters
        )
        print(f"  VQ done: {len(codebooks)} voxels in {time.time() - t_vq:.1f}s")
    else:
        scale_idx, scale_vmin, scale_vscale = _uniform_quantize(attrs['scales'], scale_bits)
        rot_idx, rot_vmin, rot_vscale = _uniform_quantize(attrs['rotations'], rot_bits)
        opacity_idx, opacity_vmin, opacity_vscale = _uniform_quantize(attrs['opacity'], opacity_bits)

    # 6. Apply Morton sort to all per-Gaussian arrays
    sh_idx = sh_idx[sort_order]
    sh_norms = sh_norms[sort_order]
    pos_idx = pos_idx[sort_order]
    dc_idx = dc_idx[sort_order]

    if use_vq:
        vq_indices = vq_indices[sort_order]
        voxel_keys = voxel_keys[sort_order]
    else:
        scale_idx = scale_idx[sort_order]
        rot_idx = rot_idx[sort_order]
        opacity_idx = opacity_idx[sort_order]

    # 7. Save
    save_dict = {
        # Positions (scalar quantized)
        'pos_idx': pos_idx,
        'pos_vmin': np.float32(pos_vmin),
        'pos_vscale': np.float32(pos_vscale),
        'pos_bits': np.uint8(pos_bits),
        'pos_shape': np.array(attrs['xyz'].shape, dtype=np.int32),
        # SH (PCA + TurboQuant)
        'sh_idx': sh_idx,
        'sh_norms': sh_norms,
        'sh_mean': sh_mean.astype(np.float32),
        'sh_components': sh_components.astype(np.float32),
        'sh_rotation': tq.get_rotation_matrix().astype(np.float32),
        'sh_centroids': tq.get_centroids().astype(np.float32),
        'sh_bits': np.uint8(sh_bits),
        'pca_components': np.int32(actual_pca),
        'sh_original_dim': np.int32(sh_dim),
        # DC
        'dc_idx': dc_idx,
        'dc_vmin': np.float32(dc_vmin),
        'dc_vscale': np.float32(dc_vscale),
        'dc_bits': np.uint8(dc_bits),
        'dc_shape': np.array(attrs['sh_dc'].shape, dtype=np.int32),
        # Metadata
        'n_gaussians': np.int32(n_original),
        'sort_order': sort_order.astype(np.int32),
        'use_vq': np.uint8(1 if use_vq else 0),
    }

    if use_vq:
        save_dict['vq_indices'] = vq_indices
        save_dict['vq_voxel_keys'] = voxel_keys.astype(np.int32)
        save_dict['vq_grid_resolution'] = np.int32(grid_resolution)
        save_dict['vq_n_clusters'] = np.int32(vq_clusters)

        all_voxel_ids = sorted(codebooks.keys())
        codebook_data = []
        codebook_offsets = [0]
        for vid in all_voxel_ids:
            cb = codebooks[vid]
            codebook_data.append(cb)
            codebook_offsets.append(codebook_offsets[-1] + len(cb))

        if codebook_data:
            save_dict['vq_codebook_data'] = np.concatenate(codebook_data, axis=0).astype(np.float16)
        else:
            save_dict['vq_codebook_data'] = np.zeros((0, 8), dtype=np.float16)
        save_dict['vq_codebook_offsets'] = np.array(codebook_offsets, dtype=np.int32)
        save_dict['vq_codebook_voxel_ids'] = np.array(all_voxel_ids, dtype=np.int32)
    else:
        for name, idx, vmin, vscale, bits, shape in [
            ('scale', scale_idx, scale_vmin, scale_vscale, scale_bits, attrs['scales'].shape),
            ('rot', rot_idx, rot_vmin, rot_vscale, rot_bits, attrs['rotations'].shape),
            ('opacity', opacity_idx, opacity_vmin, opacity_vscale, opacity_bits, attrs['opacity'].shape),
        ]:
            save_dict[f'{name}_idx'] = idx
            save_dict[f'{name}_vmin'] = np.float32(vmin)
            save_dict[f'{name}_vscale'] = np.float32(vscale)
            save_dict[f'{name}_bits'] = np.uint8(bits)
            save_dict[f'{name}_shape'] = np.array(shape, dtype=np.int32)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(output_path, **save_dict)

    elapsed = time.time() - t0
    compressed_size = os.path.getsize(output_path)
    ratio = original_size / compressed_size

    print(f"  Nuclear Compression:")
    print(f"    Original: {n_original:,} Gaussians, {original_size / 1e6:.1f} MB")
    print(f"    Compressed: {compressed_size / 1e6:.1f} MB")
    print(f"    Ratio: {ratio:.1f}x in {elapsed:.1f}s")

    return {
        'ratio': ratio,
        'original_size': original_size,
        'compressed_size': compressed_size,
        'time': elapsed,
        'pca_variance': var_explained,
        'pca_components': actual_pca,
        'sh_bits': sh_bits,
        'pos_bits': pos_bits,
        'dc_bits': dc_bits,
        'scale_bits': scale_bits if not use_vq else 0,
        'rot_bits': rot_bits if not use_vq else 0,
        'opacity_bits': opacity_bits if not use_vq else 0,
        'use_vq': use_vq,
        'vq_clusters': vq_clusters if use_vq else 0,
        'n_voxels': len(codebooks) if use_vq else 0,
    }


# ---------------------------------------------------------------------------
# Full Nuclear Pipeline: Decompress
# ---------------------------------------------------------------------------

def decompress_nuclear(npz_path):
    """Decompress nuclear-compressed model back to attributes dict.

    Returns:
        Dict with keys: xyz, sh_dc, sh_rest, scales, rotations, opacity.
        All arrays are (N, ...) float32 in the original Gaussian order.
    """
    data = np.load(npz_path, allow_pickle=True)

    n = int(data['n_gaussians'])
    sort_order = data['sort_order']
    inverse_order = np.argsort(sort_order)
    use_vq = bool(data['use_vq'])

    # 1. Positions
    xyz = _uniform_dequantize(
        data['pos_idx'], float(data['pos_vmin']), float(data['pos_vscale']),
        int(data['pos_bits'])
    ).reshape(tuple(data['pos_shape']))

    # 2. SH (PCA + TurboQuant)
    sh_idx = data['sh_idx']
    sh_norms = data['sh_norms'].astype(np.float32)
    sh_mean = data['sh_mean']
    sh_components = data['sh_components']
    rotation = data['sh_rotation']
    centroids = data['sh_centroids']
    pca_components = int(data['pca_components'])

    # Dequantize: centroid lookup
    y_hat = centroids[sh_idx]  # (N, pca_components)

    # Inverse rotation: x_hat = Y_hat @ R (since forward was X @ R^T)
    x_hat = y_hat @ rotation  # (N, pca_components)

    # Rescale by norms
    projected_recon = x_hat * sh_norms[:, np.newaxis]

    # Inverse PCA
    sh_rest_truncated = sh_pca_decompress(projected_recon, sh_mean, sh_components)

    # Zero-pad to full 45D if needed
    if sh_rest_truncated.shape[1] < 45:
        sh_rest = np.zeros((n, 45), dtype=np.float32)
        sh_rest[:, :sh_rest_truncated.shape[1]] = sh_rest_truncated
    else:
        sh_rest = sh_rest_truncated

    # 3. Scale/Rot/Opacity
    if use_vq:
        vq_indices = data['vq_indices']
        voxel_keys = data['vq_voxel_keys']
        codebook_data = data['vq_codebook_data'].astype(np.float32)
        codebook_offsets = data['vq_codebook_offsets']
        codebook_voxel_ids = data['vq_codebook_voxel_ids']

        codebooks = {}
        for i, vid in enumerate(codebook_voxel_ids):
            start = codebook_offsets[i]
            end = codebook_offsets[i + 1]
            codebooks[int(vid)] = codebook_data[start:end]

        scales, rotations, opacity = grid_vq_decompress(voxel_keys, vq_indices, codebooks)
    else:
        scales = _uniform_dequantize(
            data['scale_idx'], float(data['scale_vmin']),
            float(data['scale_vscale']), int(data['scale_bits']),
        ).reshape(tuple(data['scale_shape']))

        rotations = _uniform_dequantize(
            data['rot_idx'], float(data['rot_vmin']),
            float(data['rot_vscale']), int(data['rot_bits']),
        ).reshape(tuple(data['rot_shape']))

        opacity = _uniform_dequantize(
            data['opacity_idx'], float(data['opacity_vmin']),
            float(data['opacity_vscale']), int(data['opacity_bits']),
        ).reshape(tuple(data['opacity_shape']))

    # 4. DC
    sh_dc = _uniform_dequantize(
        data['dc_idx'], float(data['dc_vmin']), float(data['dc_vscale']),
        int(data['dc_bits'])
    ).reshape(tuple(data['dc_shape']))

    # 5. Undo Morton sort
    return {
        'xyz': xyz[inverse_order].astype(np.float32),
        'sh_dc': sh_dc[inverse_order].astype(np.float32),
        'sh_rest': sh_rest[inverse_order].astype(np.float32),
        'scales': scales[inverse_order].astype(np.float32),
        'rotations': rotations[inverse_order].astype(np.float32),
        'opacity': opacity[inverse_order].astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Quality Evaluation
# ---------------------------------------------------------------------------

def evaluate_nuclear(model_path, source_path, output_path,
                     pca_components=12, sh_bits=2,
                     dc_bits=10, pos_bits=16,
                     scale_bits=8, rot_bits=8, opacity_bits=8,
                     use_vq=False, grid_resolution=32, vq_clusters=256,
                     iteration=30000, white_background=True,
                     max_test_views=0):
    """Compress, decompress, render, and compute PSNR."""
    from decompress import save_ply

    ply_path = os.path.join(model_path, "point_cloud",
                            f"iteration_{iteration}", "point_cloud.ply")
    if not os.path.exists(ply_path):
        return {"error": f"PLY not found: {ply_path}"}

    result = {}

    try:
        # 1. Compress
        stats = compress_nuclear(
            ply_path, output_path,
            pca_components=pca_components, sh_bits=sh_bits,
            dc_bits=dc_bits, pos_bits=pos_bits,
            scale_bits=scale_bits, rot_bits=rot_bits, opacity_bits=opacity_bits,
            use_vq=use_vq, grid_resolution=grid_resolution, vq_clusters=vq_clusters,
        )
        result.update(stats)

        # 2. Decompress
        recon_attrs = decompress_nuclear(output_path)

        # 3. Save to temp PLY for rendering
        with tempfile.TemporaryDirectory() as tmpdir:
            comp_ply_path = os.path.join(tmpdir, "compressed.ply")
            save_ply(recon_attrs, comp_ply_path)

            # 4. Render and compute PSNR
            import torch
            from eval_compression import (
                psnr as compute_psnr,
                _load_scene_cameras,
                _load_gaussians_from_ply,
            )
            from gaussian_renderer import render as gs_render

            with torch.no_grad():
                scene, gaussians_orig, pp, background = _load_scene_cameras(
                    model_path, source_path, white_background,
                    sh_degree=3, iteration=iteration,
                )
                gaussians_comp = _load_gaussians_from_ply(comp_ply_path, sh_degree=3)
                gaussians_comp.active_sh_degree = gaussians_orig.active_sh_degree

                test_cameras = scene.getTestCameras()
                if len(test_cameras) == 0:
                    result['error'] = "No test cameras found"
                    return result

                if max_test_views > 0:
                    test_cameras = test_cameras[:max_test_views]

                psnr_orig_list = []
                psnr_comp_list = []

                for view in test_cameras:
                    out_orig = gs_render(view, gaussians_orig, pp, background)
                    rendered_orig = out_orig["render"]
                    gt = view.original_image[0:3, :, :].cuda()

                    out_comp = gs_render(view, gaussians_comp, pp, background)
                    rendered_comp = out_comp["render"]

                    psnr_orig_list.append(compute_psnr(rendered_orig, gt))
                    psnr_comp_list.append(compute_psnr(rendered_comp, gt))

                result['psnr_orig'] = float(np.mean(psnr_orig_list))
                result['psnr_comp'] = float(np.mean(psnr_comp_list))
                result['psnr_drop'] = result['psnr_comp'] - result['psnr_orig']
                result['n_test_views'] = len(test_cameras)

    except Exception as e:
        import traceback
        result['error'] = f"{type(e).__name__}: {e}"
        result['traceback'] = traceback.format_exc()

    return result


# ---------------------------------------------------------------------------
# PCA Variance Analysis
# ---------------------------------------------------------------------------

def analyze_pca_variance(ply_path, components_list=None):
    """Analyze PCA variance explained for different component counts."""
    if components_list is None:
        components_list = [4, 8, 12, 16, 20, 24, 32, 45]

    attrs = load_ply_attributes(ply_path)
    sh_rest = attrs['sh_rest']
    D = sh_rest.shape[1]

    mean = sh_rest.mean(axis=0)
    centered = sh_rest - mean
    cov = (centered.T @ centered) / max(len(centered) - 1, 1)
    eigenvalues, _ = np.linalg.eigh(cov)
    eigenvalues = eigenvalues[::-1]

    total_var = max(eigenvalues.sum(), 1e-12)
    cumvar = np.cumsum(eigenvalues) / total_var

    results = {}
    for nc in components_list:
        nc = min(nc, D)
        results[nc] = float(cumvar[nc - 1])

    return results


# ---------------------------------------------------------------------------
# Sweep functions
# ---------------------------------------------------------------------------

def run_scalar_sweep(model_path, source_path, iteration=30000, white_background=True,
                     pca_list=None, sh_bits=2, max_test_views=0):
    """Sweep PCA component counts with scalar quantization."""
    if pca_list is None:
        pca_list = [8, 12, 16, 20]

    ply_path = os.path.join(model_path, "point_cloud",
                            f"iteration_{iteration}", "point_cloud.ply")

    print("\n=== PCA Variance Analysis ===")
    var_results = analyze_pca_variance(ply_path, pca_list + [4, 24, 32, 45])
    for nc in sorted(var_results.keys()):
        print(f"  {nc:>3} components: {var_results[nc] * 100:.2f}% variance")

    all_results = []
    scene_name = os.path.basename(os.path.normpath(model_path))
    total = len(pca_list)

    for i, pca in enumerate(pca_list):
        print(f"\n=== [{i+1}/{total}] PCA={pca}, sh_bits={sh_bits}, scalar quant ===")

        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = os.path.join(tmpdir, "nuclear.npz")
            result = evaluate_nuclear(
                model_path, source_path, npz_path,
                pca_components=pca, sh_bits=sh_bits,
                dc_bits=10, pos_bits=16,
                scale_bits=8, rot_bits=8, opacity_bits=8,
                use_vq=False,
                iteration=iteration,
                white_background=white_background,
                max_test_views=max_test_views,
            )
            result['scene'] = scene_name
            all_results.append(result)

            if 'error' in result:
                print(f"  ERROR: {result['error']}")
            else:
                print(f"  Ratio: {result['ratio']:.1f}x, "
                      f"PSNR: {result['psnr_orig']:.2f} -> {result['psnr_comp']:.2f} "
                      f"({result['psnr_drop']:+.2f} dB)")

    _print_scalar_table(all_results)
    return all_results


def run_vq_sweep(model_path, source_path, iteration=30000, white_background=True,
                 pca_list=None, vq_list=None, sh_bits=2, max_test_views=0):
    """Sweep PCA components x VQ clusters."""
    if pca_list is None:
        pca_list = [8, 12, 16, 20]
    if vq_list is None:
        vq_list = [8, 16, 32]

    ply_path = os.path.join(model_path, "point_cloud",
                            f"iteration_{iteration}", "point_cloud.ply")

    print("\n=== PCA Variance Analysis ===")
    var_results = analyze_pca_variance(ply_path, pca_list + [4, 24, 32, 45])
    for nc in sorted(var_results.keys()):
        print(f"  {nc:>3} components: {var_results[nc] * 100:.2f}% variance")

    all_results = []
    scene_name = os.path.basename(os.path.normpath(model_path))
    total = len(pca_list) * len(vq_list)
    done = 0

    for pca in pca_list:
        for vq in vq_list:
            done += 1
            print(f"\n=== [{done}/{total}] PCA={pca}, VQ_K={vq}, sh_bits={sh_bits} ===")

            with tempfile.TemporaryDirectory() as tmpdir:
                npz_path = os.path.join(tmpdir, "nuclear.npz")
                result = evaluate_nuclear(
                    model_path, source_path, npz_path,
                    pca_components=pca, sh_bits=sh_bits,
                    use_vq=True, vq_clusters=vq,
                    iteration=iteration,
                    white_background=white_background,
                    max_test_views=max_test_views,
                )
                result['scene'] = scene_name
                all_results.append(result)

                if 'error' in result:
                    print(f"  ERROR: {result['error']}")
                else:
                    print(f"  Ratio: {result['ratio']:.1f}x, "
                          f"PSNR: {result['psnr_orig']:.2f} -> {result['psnr_comp']:.2f} "
                          f"({result['psnr_drop']:+.2f} dB)")

    _print_vq_table(all_results)
    return all_results


def _print_scalar_table(all_results):
    """Print sweep results table for scalar quantization."""
    print("\n" + "=" * 80)
    print(f"{'Scene':<12} {'PCA':>4} {'Ratio':>7} {'PSNR_orig':>10} "
          f"{'PSNR_comp':>10} {'Drop':>7} {'Var%':>7} {'Time':>7}")
    print("-" * 80)
    for r in all_results:
        if 'error' in r:
            print(f"  ERROR: {r.get('error', '?')[:70]}")
            continue
        print(f"{r.get('scene', '?'):<12} "
              f"{r.get('pca_components', '?'):>4} "
              f"{r['ratio']:>7.1f}x "
              f"{r.get('psnr_orig', float('nan')):>10.2f} "
              f"{r.get('psnr_comp', float('nan')):>10.2f} "
              f"{r.get('psnr_drop', float('nan')):>+7.2f} "
              f"{r.get('pca_variance', 0) * 100:>6.1f}% "
              f"{r['time']:>6.1f}s")
    print("=" * 80)


def _print_vq_table(all_results):
    """Print sweep results table for VQ."""
    print("\n" + "=" * 90)
    print(f"{'Scene':<12} {'PCA':>4} {'VQ':>4} {'Ratio':>7} {'PSNR_orig':>10} "
          f"{'PSNR_comp':>10} {'Drop':>7} {'Time':>7} {'Voxels':>7}")
    print("-" * 90)
    for r in all_results:
        if 'error' in r:
            print(f"  ERROR: {r.get('error', '?')[:70]}")
            continue
        print(f"{r.get('scene', '?'):<12} "
              f"{r.get('pca_components', '?'):>4} "
              f"{r.get('vq_clusters', '?'):>4} "
              f"{r['ratio']:>7.1f}x "
              f"{r.get('psnr_orig', float('nan')):>10.2f} "
              f"{r.get('psnr_comp', float('nan')):>10.2f} "
              f"{r.get('psnr_drop', float('nan')):>+7.2f} "
              f"{r['time']:>6.1f}s "
              f"{r.get('n_voxels', 0):>7}")
    print("=" * 90)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Nuclear compression: PCA SH + Low-bit Quant + Morton Sort"
    )
    parser.add_argument("-m", "--model_path", required=True,
                        help="Path to 3DGS output directory")
    parser.add_argument("-o", "--output", default=None,
                        help="Output .npz path")
    parser.add_argument("--iteration", type=int, default=30000)

    # Compression params
    parser.add_argument("--pca", type=int, default=12,
                        help="Number of PCA components (default: 12)")
    parser.add_argument("--sh_bits", type=int, default=2,
                        help="Bit-width for TurboQuant on PCA SH (default: 2)")
    parser.add_argument("--dc_bits", type=int, default=10)
    parser.add_argument("--pos_bits", type=int, default=16)
    parser.add_argument("--scale_bits", type=int, default=8)
    parser.add_argument("--rot_bits", type=int, default=8)
    parser.add_argument("--opacity_bits", type=int, default=8)
    parser.add_argument("--use_vq", action="store_true",
                        help="Use grid-local VQ for scale/rot/opacity")
    parser.add_argument("--grid_resolution", type=int, default=32)
    parser.add_argument("--vq_clusters", type=int, default=256)

    # Evaluation
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate quality (render + PSNR)")
    parser.add_argument("--data_root", default="data/nerf_synthetic")
    parser.add_argument("--data_root_360", default="data/360_v2")
    parser.add_argument("--white_bg", action="store_true", default=True)
    parser.add_argument("--no_white_bg", dest="white_bg", action="store_false")
    parser.add_argument("--max_test_views", type=int, default=0)

    # Sweep
    parser.add_argument("--sweep_scalar", action="store_true",
                        help="Sweep PCA components (scalar quant)")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep PCA x VQ configs")
    parser.add_argument("--pca_list", nargs="+", type=int, default=[8, 12, 16, 20])
    parser.add_argument("--vq_list", nargs="+", type=int, default=[8, 16, 32])

    # PCA analysis
    parser.add_argument("--pca_analysis", action="store_true")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Find PLY
    pc_dir = os.path.join(args.model_path, "point_cloud")
    ply_path = os.path.join(pc_dir, f"iteration_{args.iteration}", "point_cloud.ply")
    if not os.path.exists(ply_path):
        print(f"ERROR: PLY not found at {ply_path}")
        sys.exit(1)

    # Determine scene name and source path
    scene_name = os.path.basename(os.path.normpath(args.model_path))
    base_scene = scene_name.split("_")[0] if "_" in scene_name else scene_name

    source_candidates = [
        os.path.join(script_dir, args.data_root, base_scene),
        os.path.join(script_dir, args.data_root_360, base_scene),
        os.path.join(script_dir, args.data_root, scene_name),
        os.path.join(script_dir, args.data_root_360, scene_name),
    ]
    source_path = None
    for sp in source_candidates:
        if os.path.isdir(sp):
            source_path = sp
            break

    # PCA analysis mode
    if args.pca_analysis:
        print(f"PCA Variance Analysis for: {ply_path}")
        var_results = analyze_pca_variance(ply_path)
        for nc in sorted(var_results.keys()):
            print(f"  {nc:>3} components: {var_results[nc] * 100:.2f}% variance")
        return

    # Sweep modes
    if args.sweep_scalar or args.sweep:
        if source_path is None:
            print(f"ERROR: source data not found for scene '{base_scene}'")
            print(f"  Tried: {source_candidates}")
            # Compression-only fallback
            all_results = []
            for pca in args.pca_list:
                print(f"\n--- PCA={pca} ---")
                with tempfile.TemporaryDirectory() as tmpdir:
                    npz_path = os.path.join(tmpdir, "nuclear.npz")
                    result = compress_nuclear(
                        ply_path, npz_path,
                        pca_components=pca, sh_bits=args.sh_bits,
                        use_vq=args.use_vq or args.sweep,
                        vq_clusters=args.vq_clusters,
                    )
                    result['scene'] = scene_name
                    all_results.append(result)
            return

        if args.sweep:
            results = run_vq_sweep(
                args.model_path, source_path,
                iteration=args.iteration,
                white_background=args.white_bg,
                pca_list=args.pca_list,
                vq_list=args.vq_list,
                sh_bits=args.sh_bits,
                max_test_views=args.max_test_views,
            )
        else:
            results = run_scalar_sweep(
                args.model_path, source_path,
                iteration=args.iteration,
                white_background=args.white_bg,
                pca_list=args.pca_list,
                sh_bits=args.sh_bits,
                max_test_views=args.max_test_views,
            )

        out_json = os.path.join(script_dir, "results",
                                f"nuclear_sweep_{scene_name}.json")
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {out_json}")
        return

    # Single compression
    if args.output is None:
        args.output = os.path.join("compressed", f"{scene_name}_nuclear.npz")

    print(f"Nuclear compression: {ply_path}")
    print(f"  PCA={args.pca}, sh_bits={args.sh_bits}, "
          f"dc={args.dc_bits}b, pos={args.pos_bits}b, "
          f"scale={args.scale_bits}b, rot={args.rot_bits}b, "
          f"opacity={args.opacity_bits}b")

    stats = compress_nuclear(
        ply_path, args.output,
        pca_components=args.pca, sh_bits=args.sh_bits,
        dc_bits=args.dc_bits, pos_bits=args.pos_bits,
        scale_bits=args.scale_bits, rot_bits=args.rot_bits,
        opacity_bits=args.opacity_bits,
        use_vq=args.use_vq, grid_resolution=args.grid_resolution,
        vq_clusters=args.vq_clusters,
    )

    if args.eval:
        if source_path is None:
            print(f"\nWARNING: source data not found for '{base_scene}', skipping eval")
        else:
            print(f"\n  Evaluating quality ...")
            result = evaluate_nuclear(
                args.model_path, source_path, args.output,
                pca_components=args.pca, sh_bits=args.sh_bits,
                dc_bits=args.dc_bits, pos_bits=args.pos_bits,
                scale_bits=args.scale_bits, rot_bits=args.rot_bits,
                opacity_bits=args.opacity_bits,
                use_vq=args.use_vq, grid_resolution=args.grid_resolution,
                vq_clusters=args.vq_clusters,
                iteration=args.iteration,
                white_background=args.white_bg,
                max_test_views=args.max_test_views,
            )
            if 'error' in result:
                print(f"  ERROR: {result['error']}")
            else:
                print(f"  PSNR: {result['psnr_orig']:.2f} -> {result['psnr_comp']:.2f} "
                      f"({result['psnr_drop']:+.2f} dB)")


if __name__ == "__main__":
    main()
