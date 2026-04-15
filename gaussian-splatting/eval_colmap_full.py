#!/usr/bin/env python3
"""Full COLMAP evaluation: overfitting diagnosis + compression for 11 scenes.

Part 1: Overfitting diagnosis (train/test gap + per-band SH ratios)
Part 2: Compression evaluation (best config with zstd)

Uses data_device='cpu' and only 5 train + 5 test views for memory safety.
"""

import gc
import json
import math
import os
import sys
import tempfile
import time
import traceback

import torch
import numpy as np
from argparse import Namespace

# Ensure project root is in path
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render as gs_render
from utils.image_utils import psnr as psnr_fn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCENES = [
    "bonsai", "counter", "flowers", "garden", "kitchen",
    "room", "stump", "treehill", "truck", "train", "playroom",
]

SH_BAND_SLICES = {
    1: (0, 3),
    2: (3, 8),
    3: (8, 15),
}

COMPRESSION_CONFIG = dict(
    sh_bits=2, pos_bits=14, dc_bits=10,
    scale_bits=8, rot_bits=8, opacity_bits=6,
    entropy="zstd",
)


def _cleanup():
    """Aggressively free GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()


def _load_cfg(scene_name):
    """Load cfg_args for a scene."""
    cfg_path = os.path.join(_root, "output", scene_name, "cfg_args")
    with open(cfg_path) as f:
        cfg = eval(f.read())
    source_path = cfg.source_path
    white_bg = getattr(cfg, "white_background", False)
    return source_path, white_bg


def _make_args(model_path, source_path, white_bg):
    """Build ModelParams-compatible Namespace with data_device='cpu'."""
    return Namespace(
        sh_degree=3,
        source_path=os.path.abspath(source_path),
        model_path=os.path.abspath(model_path),
        images="images",
        depths="",
        resolution=-1,
        white_background=white_bg,
        train_test_exp=False,
        data_device="cpu",   # IMPORTANT: avoid OOM on large COLMAP scenes
        eval=True,
    )


def _make_pipe():
    return Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False,
    )


def _avg_psnr(cameras, gaussians, pipe, bg, max_views=5):
    """Render up to max_views cameras and return average PSNR."""
    total = 0.0
    n = min(len(cameras), max_views)
    with torch.no_grad():
        for i in range(n):
            cam = cameras[i]
            rendering = gs_render(cam, gaussians, pipe, bg)["render"]
            gt = cam.original_image[:3, :, :].to("cuda")
            total += psnr_fn(rendering, gt).mean().item()
            del rendering, gt
    return total / n if n > 0 else float("nan")


# ---------------------------------------------------------------------------
# Part 1: Overfitting diagnosis
# ---------------------------------------------------------------------------

def diagnose_scene(scene_name):
    """Run overfitting diagnosis for a single COLMAP scene.

    Returns dict with train_psnr, test_psnr, gap, R1, R2, R3.
    """
    print(f"\n{'='*60}")
    print(f"OVERFITTING DIAGNOSIS: {scene_name}")
    print(f"{'='*60}")

    model_path = os.path.join(_root, "output", scene_name)
    source_path, white_bg = _load_cfg(scene_name)

    args = _make_args(model_path, source_path, white_bg)
    pipe = _make_pipe()

    bg_color = [1, 1, 1] if white_bg else [0, 0, 0]
    bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians = GaussianModel(3)
    scene = Scene(args, gaussians, load_iteration=30000, shuffle=False)

    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()

    print(f"  Train views: {len(train_cams)}, Test views: {len(test_cams)}")
    print(f"  Using 5 train + 5 test views")

    with torch.no_grad():
        # Baseline
        baseline_train = _avg_psnr(train_cams, gaussians, pipe, bg, max_views=5)
        baseline_test = _avg_psnr(test_cams, gaussians, pipe, bg, max_views=5)
        gap = baseline_train - baseline_test
        print(f"  Baseline train: {baseline_train:.2f}, test: {baseline_test:.2f}, gap: {gap:.2f}")

        # Per-band analysis
        orig_rest = gaussians._features_rest.data.clone()
        band_ratios = {}

        for band, (start, end) in SH_BAND_SLICES.items():
            gaussians._features_rest.data[:, start:end, :] = 0.0
            zeroed_train = _avg_psnr(train_cams, gaussians, pipe, bg, max_views=5)
            zeroed_test = _avg_psnr(test_cams, gaussians, pipe, bg, max_views=5)

            train_drop = baseline_train - zeroed_train
            test_drop = baseline_test - zeroed_test
            ratio = train_drop / test_drop if abs(test_drop) > 1e-6 else float("inf")
            band_ratios[band] = ratio
            print(f"  Band {band}: train_drop={train_drop:+.3f}, test_drop={test_drop:+.3f}, R={ratio:.3f}")

            gaussians._features_rest.data.copy_(orig_rest)

    result = {
        "scene": scene_name,
        "train_psnr": baseline_train,
        "test_psnr": baseline_test,
        "gap": gap,
        "R1": band_ratios[1],
        "R2": band_ratios[2],
        "R3": band_ratios[3],
    }

    # Cleanup
    del gaussians, scene, train_cams, test_cams, orig_rest, bg
    _cleanup()

    return result


# ---------------------------------------------------------------------------
# Part 2: Compression evaluation
# ---------------------------------------------------------------------------

def compress_scene(scene_name):
    """Compress a scene and measure quality impact.

    Returns dict with scene, ratio, orig_psnr, comp_psnr, drop, time.
    """
    from compress import load_ply_attributes, compress_gaussians
    from decompress import decompress_gaussians, save_ply

    print(f"\n{'='*60}")
    print(f"COMPRESSION EVAL: {scene_name}")
    print(f"{'='*60}")

    model_path = os.path.join(_root, "output", scene_name)
    source_path, white_bg = _load_cfg(scene_name)
    ply_path = os.path.join(model_path, "point_cloud", "iteration_30000", "point_cloud.ply")

    pipe = _make_pipe()
    # COLMAP scenes: black background
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    t0 = time.perf_counter()

    # 1. Compress
    attrs = load_ply_attributes(ply_path)
    n_gaussians = attrs["xyz"].shape[0]
    print(f"  {n_gaussians:,} Gaussians")

    with tempfile.TemporaryDirectory() as tmpdir:
        comp_path = os.path.join(tmpdir, f"{scene_name}.tsv4")
        stats = compress_gaussians(attrs, comp_path, **COMPRESSION_CONFIG)
        del attrs

        # 2. Decompress to PLY
        recon_attrs = decompress_gaussians(comp_path)
        comp_ply = os.path.join(tmpdir, "compressed.ply")
        save_ply(recon_attrs, comp_ply)
        del recon_attrs

        # 3. Load scene + cameras
        args = _make_args(model_path, source_path, white_bg)
        gaussians_orig = GaussianModel(3)
        scene_obj = Scene(args, gaussians_orig, load_iteration=30000, shuffle=False)
        test_cams = scene_obj.getTestCameras()

        # 4. Load compressed model
        gaussians_comp = GaussianModel(3)
        gaussians_comp.load_ply(comp_ply)
        gaussians_comp.active_sh_degree = gaussians_orig.active_sh_degree

        # 5. Render and compare
        with torch.no_grad():
            orig_psnr = _avg_psnr(test_cams, gaussians_orig, pipe, bg, max_views=5)
            comp_psnr = _avg_psnr(test_cams, gaussians_comp, pipe, bg, max_views=5)

        del gaussians_orig, gaussians_comp, scene_obj, test_cams, bg
        _cleanup()

    elapsed = time.perf_counter() - t0
    drop = comp_psnr - orig_psnr

    result = {
        "scene": scene_name,
        "n_gaussians": stats["n_gaussians"],
        "compression_ratio": stats["compression_ratio"],
        "original_size_bytes": stats["original_size_bytes"],
        "compressed_size_bytes": stats["compressed_size_bytes"],
        "compression_time_s": stats["compression_time_s"],
        "orig_psnr": orig_psnr,
        "comp_psnr": comp_psnr,
        "psnr_drop": drop,
        "total_time_s": elapsed,
    }

    print(f"  Ratio: {stats['compression_ratio']:.2f}x, "
          f"PSNR: {orig_psnr:.2f} -> {comp_psnr:.2f} ({drop:+.2f} dB), "
          f"Time: {elapsed:.1f}s")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(os.path.join(_root, "results"), exist_ok=True)
    overfit_path = os.path.join(_root, "results", "colmap_overfitting_full.json")
    compress_path = os.path.join(_root, "results", "colmap_compression.json")

    # ---- Part 1: Overfitting ----
    print("\n" + "=" * 70)
    print("PART 1: OVERFITTING DIAGNOSIS (11 scenes)")
    print("=" * 70)

    overfit_results = []
    overfit_failures = []
    for scene in SCENES:
        try:
            r = diagnose_scene(scene)
            overfit_results.append(r)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  FAILED: {e}")
            print(tb)
            overfit_failures.append({"scene": scene, "error": str(e), "traceback": tb})
            _cleanup()

        # Save intermediate
        with open(overfit_path, "w") as f:
            json.dump({"results": overfit_results, "failures": overfit_failures}, f, indent=2)

    # Print overfitting summary
    print("\n" + "=" * 70)
    print("OVERFITTING SUMMARY")
    print("=" * 70)
    hdr = f"{'Scene':<12} {'Gap':>6} {'R1':>7} {'R2':>7} {'R3':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in overfit_results:
        print(f"{r['scene']:<12} {r['gap']:>+6.2f} {r['R1']:>7.3f} {r['R2']:>7.3f} {r['R3']:>7.3f}")
    if overfit_results:
        avg_gap = np.mean([r["gap"] for r in overfit_results])
        avg_r1 = np.mean([r["R1"] for r in overfit_results if np.isfinite(r["R1"])])
        avg_r2 = np.mean([r["R2"] for r in overfit_results if np.isfinite(r["R2"])])
        avg_r3 = np.mean([r["R3"] for r in overfit_results if np.isfinite(r["R3"])])
        print(f"{'AVERAGE':<12} {avg_gap:>+6.2f} {avg_r1:>7.3f} {avg_r2:>7.3f} {avg_r3:>7.3f}")

    # ---- Part 2: Compression ----
    print("\n" + "=" * 70)
    print("PART 2: COMPRESSION EVALUATION (11 scenes)")
    print(f"Config: {COMPRESSION_CONFIG}")
    print("=" * 70)

    comp_results = []
    comp_failures = []
    for scene in SCENES:
        try:
            r = compress_scene(scene)
            comp_results.append(r)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  FAILED: {e}")
            print(tb)
            comp_failures.append({"scene": scene, "error": str(e), "traceback": tb})
            _cleanup()

        # Save intermediate
        with open(compress_path, "w") as f:
            json.dump({"config": COMPRESSION_CONFIG, "results": comp_results, "failures": comp_failures}, f, indent=2)

    # Print compression summary
    print("\n" + "=" * 70)
    print("COMPRESSION SUMMARY")
    print("=" * 70)
    hdr = f"{'Scene':<12} {'Ratio':>7} {'Orig':>8} {'Comp':>8} {'Drop':>7} {'Time':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in comp_results:
        print(f"{r['scene']:<12} {r['compression_ratio']:>6.2f}x {r['orig_psnr']:>8.2f} "
              f"{r['comp_psnr']:>8.2f} {r['psnr_drop']:>+7.2f} {r['total_time_s']:>5.1f}s")
    if comp_results:
        avg_ratio = np.mean([r["compression_ratio"] for r in comp_results])
        avg_drop = np.mean([r["psnr_drop"] for r in comp_results])
        print(f"{'AVERAGE':<12} {avg_ratio:>6.2f}x {'':>8} {'':>8} {avg_drop:>+7.2f}")

    # Final save
    with open(overfit_path, "w") as f:
        json.dump({"results": overfit_results, "failures": overfit_failures}, f, indent=2)
    with open(compress_path, "w") as f:
        json.dump({"config": COMPRESSION_CONFIG, "results": comp_results, "failures": comp_failures}, f, indent=2)

    print(f"\nResults saved to:")
    print(f"  {overfit_path}")
    print(f"  {compress_path}")

    # Report failures
    all_fail = overfit_failures + comp_failures
    if all_fail:
        print(f"\n{len(all_fail)} failure(s):")
        for f_item in all_fail:
            print(f"  {f_item['scene']}: {f_item['error']}")


if __name__ == "__main__":
    main()
