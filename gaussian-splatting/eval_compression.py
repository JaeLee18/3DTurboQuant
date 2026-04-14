"""Batch evaluation script: compress each trained 3DGS scene, render from
the compressed model, and compute PSNR/SSIM/LPIPS vs. original.

Usage:
    python eval_compression.py --scenes lego chair --sh_bits 2 3 4
"""

import argparse
import json
import math
import os
import sys
import tempfile
import time

import torch
import numpy as np

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """PSNR between two (C, H, W) tensors in [0, 1]."""
    mse = ((img1 - img2) ** 2).mean().item()
    if mse == 0:
        return float("inf")
    return -10.0 * math.log10(mse)


# ---------------------------------------------------------------------------
# Scene / model loading helpers
# ---------------------------------------------------------------------------

def _make_namespace(**kwargs):
    """Build a Namespace-like object with arbitrary attributes."""
    from argparse import Namespace
    return Namespace(**kwargs)


def _load_scene_cameras(model_path: str, source_path: str, white_background: bool,
                         sh_degree: int = 3, iteration: int = 30000):
    """Load a Scene and a GaussianModel from the given paths.

    Returns (scene, gaussians, pipeline_params, background).
    """
    from arguments import ModelParams, PipelineParams
    from scene import Scene
    from scene.gaussian_model import GaussianModel
    from argparse import ArgumentParser

    # Build minimal argument objects that Scene / ModelParams expect
    parser = ArgumentParser()
    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    # Parse empty args to get defaults
    args = parser.parse_args([])

    # Override relevant fields
    args.sh_degree = sh_degree
    args.source_path = os.path.abspath(source_path)
    args.model_path = os.path.abspath(model_path)
    args.images = "images"
    args.depths = ""
    args.resolution = -1
    args.white_background = white_background
    args.train_test_exp = False
    args.data_device = "cuda"
    args.eval = True

    # Extract typed param groups
    mp = model_params.extract(args)
    mp.source_path = os.path.abspath(source_path)
    mp.model_path = os.path.abspath(model_path)
    mp.white_background = white_background
    mp.eval = True
    mp.train_test_exp = False
    mp.sh_degree = sh_degree
    mp.images = "images"
    mp.depths = ""
    mp.resolution = -1
    mp.data_device = "cuda"

    pp = pipeline_params.extract(args)

    gaussians = GaussianModel(sh_degree)
    scene = Scene(mp, gaussians, load_iteration=iteration, shuffle=False)

    bg_color = [1, 1, 1] if white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    return scene, gaussians, pp, background


def _load_gaussians_from_ply(ply_path: str, sh_degree: int = 3) -> "GaussianModel":
    """Load a GaussianModel from an arbitrary PLY file (no scene needed)."""
    from scene.gaussian_model import GaussianModel
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(ply_path)
    return gaussians


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_scene(
    model_path: str,
    source_path: str,
    sh_bits: int = 3,
    iteration: int = 30000,
    white_background: bool = True,
    seed: int = 0,
) -> dict:
    """Evaluate compression quality for a single scene.

    Steps:
        1. Load PLY attributes
        2. Compress to temp .npz
        3. Decompress back to attrs
        4. Save decompressed attrs to temp PLY
        5. Load original GaussianModel (via Scene)
        6. Load compressed GaussianModel from temp PLY
        7. Render test views, compute metrics
        8. Return stats + quality metrics

    Returns:
        dict with keys: scene, sh_bits, n_gaussians, compression_ratio,
        compression_time_s, psnr_orig, psnr_comp, ssim_orig, ssim_comp,
        lpips_orig, lpips_comp, render_time_s, error (if failed)
    """
    from compress import load_ply_attributes, compress_gaussians
    from decompress import decompress_gaussians, save_ply
    from gaussian_renderer import render as gs_render
    from utils.loss_utils import ssim as ssim_fn
    from lpipsPyTorch import lpips as lpips_fn

    result = {"scene": os.path.basename(model_path), "sh_bits": sh_bits}

    # --- Locate PLY ---
    ply_path = os.path.join(model_path, "point_cloud",
                            f"iteration_{iteration}", "point_cloud.ply")
    if not os.path.exists(ply_path):
        result["error"] = f"PLY not found: {ply_path}"
        return result

    try:
        # 1. Load PLY attributes
        attrs = load_ply_attributes(ply_path)

        # 2. Compress to temp .npz
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = os.path.join(tmpdir, "compressed.npz")
            stats = compress_gaussians(attrs, npz_path, sh_bits=sh_bits, seed=seed)

            result["n_gaussians"] = stats["n_gaussians"]
            result["compression_ratio"] = stats["compression_ratio"]
            result["compression_time_s"] = stats["compression_time_s"]
            result["original_size_bytes"] = stats["original_size_bytes"]
            result["compressed_size_bytes"] = stats["compressed_size_bytes"]

            # 3. Decompress
            recon_attrs = decompress_gaussians(npz_path)

            # 4. Save to temp PLY
            comp_ply_path = os.path.join(tmpdir, "compressed.ply")
            save_ply(recon_attrs, comp_ply_path)

            # 5 & 6. Load scene cameras + both gaussians
            with torch.no_grad():
                scene, gaussians_orig, pp, background = _load_scene_cameras(
                    model_path, source_path, white_background,
                    sh_degree=3, iteration=iteration
                )

                # Compressed gaussians: load from temp PLY
                gaussians_comp = _load_gaussians_from_ply(comp_ply_path, sh_degree=3)
                # Copy active_sh_degree from original
                gaussians_comp.active_sh_degree = gaussians_orig.active_sh_degree

                test_cameras = scene.getTestCameras()
                if len(test_cameras) == 0:
                    result["error"] = "No test cameras found"
                    return result

                # 7. Render and compute metrics
                psnr_orig_list, psnr_comp_list = [], []
                ssim_orig_list, ssim_comp_list = [], []
                lpips_orig_list, lpips_comp_list = [], []

                t_render_start = time.perf_counter()
                for view in test_cameras:
                    # Render original
                    out_orig = gs_render(view, gaussians_orig, pp, background)
                    rendered_orig = out_orig["render"].unsqueeze(0)  # (1, C, H, W)
                    gt = view.original_image[0:3, :, :].cuda().unsqueeze(0)  # (1, C, H, W)

                    # Render compressed
                    out_comp = gs_render(view, gaussians_comp, pp, background)
                    rendered_comp = out_comp["render"].unsqueeze(0)  # (1, C, H, W)

                    # PSNR
                    psnr_orig_list.append(psnr(rendered_orig[0], gt[0]))
                    psnr_comp_list.append(psnr(rendered_comp[0], gt[0]))

                    # SSIM
                    ssim_orig_list.append(ssim_fn(rendered_orig, gt).item())
                    ssim_comp_list.append(ssim_fn(rendered_comp, gt).item())

                    # LPIPS
                    lpips_orig_list.append(lpips_fn(rendered_orig, gt, net_type="vgg").item())
                    lpips_comp_list.append(lpips_fn(rendered_comp, gt, net_type="vgg").item())

                render_time = time.perf_counter() - t_render_start

                result["psnr_orig"] = float(np.mean(psnr_orig_list))
                result["psnr_comp"] = float(np.mean(psnr_comp_list))
                result["ssim_orig"] = float(np.mean(ssim_orig_list))
                result["ssim_comp"] = float(np.mean(ssim_comp_list))
                result["lpips_orig"] = float(np.mean(lpips_orig_list))
                result["lpips_comp"] = float(np.mean(lpips_comp_list))
                result["render_time_s"] = render_time
                result["n_test_views"] = len(test_cameras)

    except Exception as e:
        import traceback
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()

    return result


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

NERF_SYNTHETIC_SCENES = [
    "chair", "drums", "ficus", "hotdog",
    "lego", "materials", "mic", "ship",
]


def print_summary_table(all_results: list):
    """Print a formatted summary table to stdout."""
    header = f"{'Scene':<12} {'b':>2}  {'Ratio':>6}  {'PSNR_orig':>9}  {'PSNR_comp':>9}  {'Drop':>6}  {'SSIM_d':>7}  {'LPIPS_d':>7}  {'Time':>6}"
    print()
    print(header)
    print("-" * len(header))
    for r in all_results:
        if "error" in r:
            print(f"{'ERROR':<12} {r.get('sh_bits','?'):>2}  {r.get('scene','?')}  {r['error'][:60]}")
            continue
        scene = r.get("scene", "?")[:11]
        b = r.get("sh_bits", "?")
        ratio = r.get("compression_ratio", float("nan"))
        p_orig = r.get("psnr_orig", float("nan"))
        p_comp = r.get("psnr_comp", float("nan"))
        drop = p_comp - p_orig
        s_drop = r.get("ssim_comp", float("nan")) - r.get("ssim_orig", float("nan"))
        l_drop = r.get("lpips_comp", float("nan")) - r.get("lpips_orig", float("nan"))
        rtime = r.get("render_time_s", float("nan"))
        print(
            f"{scene:<12} {b:>2}  {ratio:>6.2f}x  {p_orig:>9.2f}  {p_comp:>9.2f}  "
            f"{drop:>+6.2f}  {s_drop:>+7.4f}  {l_drop:>+7.4f}  {rtime:>5.1f}s"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Batch compression + quality evaluation for 3DGS scenes"
    )
    parser.add_argument(
        "--scenes", nargs="+", default=NERF_SYNTHETIC_SCENES,
        help="Scene names to evaluate (default: all 8 NeRF Synthetic)",
    )
    parser.add_argument(
        "--data_root", default="data/nerf_synthetic",
        help="Root directory containing scene data folders (default: data/nerf_synthetic)",
    )
    parser.add_argument(
        "--output_root", default="output",
        help="Root directory containing trained model folders (default: output)",
    )
    parser.add_argument(
        "--suffix", default="_wb",
        help="Suffix appended to scene name to form model dir (default: _wb)",
    )
    parser.add_argument(
        "--sh_bits", nargs="+", type=int, default=[2, 3, 4],
        help="Bit-widths for SH quantization to test (default: 2 3 4)",
    )
    parser.add_argument(
        "--iteration", type=int, default=30000,
        help="Model iteration to load (default: 30000)",
    )
    parser.add_argument(
        "--white_background", action="store_true", default=True,
        help="Use white background (default: True)",
    )
    parser.add_argument(
        "--no_white_background", dest="white_background", action="store_false",
        help="Use black background",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for TurboQuantizer (default: 0)",
    )
    parser.add_argument(
        "--output", default="results/compression_results.json",
        help="Output JSON path (default: results/compression_results.json)",
    )
    args = parser.parse_args()

    # Resolve to absolute paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(script_dir, args.data_root) if not os.path.isabs(args.data_root) else args.data_root
    output_root = os.path.join(script_dir, args.output_root) if not os.path.isabs(args.output_root) else args.output_root
    out_json = os.path.join(script_dir, args.output) if not os.path.isabs(args.output) else args.output

    all_results = []

    total = len(args.scenes) * len(args.sh_bits)
    done = 0
    for scene in args.scenes:
        model_path = os.path.join(output_root, scene + args.suffix)
        source_path = os.path.join(data_root, scene)

        # Check if model exists; skip gracefully if not
        ply_check = os.path.join(model_path, "point_cloud",
                                 f"iteration_{args.iteration}", "point_cloud.ply")
        if not os.path.exists(ply_check):
            print(f"[SKIP] {scene}: model not found at {ply_check}")
            for b in args.sh_bits:
                all_results.append({
                    "scene": scene,
                    "sh_bits": b,
                    "error": f"Model not found: {ply_check}",
                })
            done += len(args.sh_bits)
            continue

        for b in args.sh_bits:
            done += 1
            print(f"\n[{done}/{total}] Evaluating scene={scene}, sh_bits={b} ...")
            t0 = time.perf_counter()
            result = evaluate_scene(
                model_path=model_path,
                source_path=source_path,
                sh_bits=b,
                iteration=args.iteration,
                white_background=args.white_background,
                seed=args.seed,
            )
            elapsed = time.perf_counter() - t0
            result["total_time_s"] = elapsed

            if "error" in result:
                print(f"  ERROR: {result['error']}")
                if "traceback" in result:
                    print(result["traceback"])
            else:
                drop = result["psnr_comp"] - result["psnr_orig"]
                print(
                    f"  ratio={result['compression_ratio']:.2f}x  "
                    f"PSNR: {result['psnr_orig']:.2f} -> {result['psnr_comp']:.2f} ({drop:+.2f} dB)  "
                    f"SSIM: {result['ssim_orig']:.4f} -> {result['ssim_comp']:.4f}  "
                    f"LPIPS: {result['lpips_orig']:.4f} -> {result['lpips_comp']:.4f}  "
                    f"({elapsed:.1f}s)"
                )
            all_results.append(result)

            # Save intermediate results after each scene/bit combo
            os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
            with open(out_json, "w") as f:
                json.dump(all_results, f, indent=2)

    # Final summary
    print_summary_table(all_results)
    print(f"\nResults saved to: {out_json}")


if __name__ == "__main__":
    main()
