#!/usr/bin/env python3
"""Compute PSNR, SSIM, LPIPS for all 21 scenes at balanced compression config.

Config: sh_bits=2, pos_bits=16, dc_bits=10, scale_bits=10, rot_bits=10,
        opacity_bits=8, entropy='zstd'

Outputs: results/full_metrics_21_scenes.json
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
from utils.loss_utils import ssim as ssim_fn
from compress import load_ply_attributes, compress_gaussians
from decompress import decompress_gaussians, save_ply

# ---------------------------------------------------------------------------
# Persistent LPIPS model (avoid re-creating VGG net every call)
# ---------------------------------------------------------------------------
_lpips_model = None

def _get_lpips_model():
    global _lpips_model
    if _lpips_model is None:
        from lpipsPyTorch.modules.lpips import LPIPS
        _lpips_model = LPIPS('vgg', '0.1').cuda().eval()
    return _lpips_model

def lpips_fn(x, y):
    """Compute LPIPS between (1,C,H,W) or (C,H,W) tensors."""
    model = _get_lpips_model()
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if y.dim() == 3:
        y = y.unsqueeze(0)
    with torch.no_grad():
        return model(x, y).item()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG = dict(
    sh_bits=2, pos_bits=16, dc_bits=10, scale_bits=10,
    rot_bits=10, opacity_bits=8, entropy='zstd',
)

NERF_SYN_SCENES = [
    "lego", "chair", "drums", "ficus", "hotdog", "materials", "mic", "ship",
]
COLMAP_SCENES = [
    "bicycle", "bonsai", "counter", "flowers", "garden", "kitchen",
    "room", "stump", "treehill", "truck", "train", "playroom", "drjohnson",
]

MAX_TEST_VIEWS = 5  # limit for speed

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def psnr(img1, img2):
    """PSNR between two (C,H,W) tensors in [0,1]."""
    mse = ((img1 - img2) ** 2).mean().item()
    if mse == 0:
        return float("inf")
    return -10.0 * math.log10(mse)


def _load_cfg(scene_name):
    """Load cfg_args for a COLMAP scene."""
    cfg_path = os.path.join(_root, "output", scene_name, "cfg_args")
    with open(cfg_path) as f:
        cfg = eval(f.read())
    return cfg.source_path, getattr(cfg, "white_background", False)


def _make_pipe():
    return Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False,
    )


def _make_args(model_path, source_path, white_bg, data_device="cuda"):
    return Namespace(
        sh_degree=3,
        source_path=os.path.abspath(source_path),
        model_path=os.path.abspath(model_path),
        images="images",
        depths="",
        resolution=-1,
        white_background=white_bg,
        train_test_exp=False,
        data_device=data_device,
        eval=True,
    )


# ---------------------------------------------------------------------------
# Core: evaluate one scene
# ---------------------------------------------------------------------------

def evaluate_scene(scene_name, model_path, source_path, white_bg, data_device="cuda"):
    """Compress, decompress, render, compute PSNR/SSIM/LPIPS for one scene."""

    ply_path = os.path.join(model_path, "point_cloud", "iteration_30000", "point_cloud.ply")
    if not os.path.exists(ply_path):
        return {"error": f"PLY not found: {ply_path}"}

    pipe = _make_pipe()
    bg_color = [1, 1, 1] if white_bg else [0, 0, 0]
    bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 1. Compress
    attrs = load_ply_attributes(ply_path)
    n_gauss = attrs["xyz"].shape[0]
    print(f"  {n_gauss:,} Gaussians, compressing...")

    with tempfile.TemporaryDirectory() as tmpdir:
        comp_path = os.path.join(tmpdir, f"{scene_name}.tsv4")
        stats = compress_gaussians(attrs, comp_path, **CONFIG)
        del attrs

        # 2. Decompress to PLY
        recon_attrs = decompress_gaussians(comp_path)
        comp_ply = os.path.join(tmpdir, "compressed.ply")
        save_ply(recon_attrs, comp_ply)
        del recon_attrs

        # 3. Load scene + cameras
        args = _make_args(model_path, source_path, white_bg, data_device)
        gaussians_orig = GaussianModel(3)
        scene_obj = Scene(args, gaussians_orig, load_iteration=30000, shuffle=False)
        test_cams = scene_obj.getTestCameras()

        # 4. Load compressed model
        gaussians_comp = GaussianModel(3)
        gaussians_comp.load_ply(comp_ply)
        gaussians_comp.active_sh_degree = gaussians_orig.active_sh_degree

        # 5. Render and compute metrics
        n_views = min(len(test_cams), MAX_TEST_VIEWS)
        if n_views == 0:
            return {"error": "No test cameras found"}

        psnr_orig_list, psnr_comp_list = [], []
        ssim_orig_list, ssim_comp_list = [], []
        lpips_orig_list, lpips_comp_list = [], []

        with torch.no_grad():
            for i in range(n_views):
                cam = test_cams[i]
                gt = cam.original_image[:3, :, :].cuda()

                # Original render
                render_orig = gs_render(cam, gaussians_orig, pipe, bg)["render"]
                psnr_orig_list.append(psnr(render_orig, gt))
                ssim_orig_list.append(ssim_fn(render_orig.unsqueeze(0), gt.unsqueeze(0)).item())
                lpips_orig_list.append(lpips_fn(render_orig, gt))

                # Compressed render
                render_comp = gs_render(cam, gaussians_comp, pipe, bg)["render"]
                psnr_comp_list.append(psnr(render_comp, gt))
                ssim_comp_list.append(ssim_fn(render_comp.unsqueeze(0), gt.unsqueeze(0)).item())
                lpips_comp_list.append(lpips_fn(render_comp, gt))

                del render_orig, render_comp, gt
                torch.cuda.empty_cache()

        # Clean up
        del gaussians_orig, gaussians_comp, scene_obj, test_cams, bg
        _cleanup()

    result = {
        "n_gaussians": stats["n_gaussians"],
        "compression_ratio": stats["compression_ratio"],
        "original_size_bytes": stats["original_size_bytes"],
        "compressed_size_bytes": stats["compressed_size_bytes"],
        "n_test_views": n_views,
        "orig_psnr": float(np.mean(psnr_orig_list)),
        "comp_psnr": float(np.mean(psnr_comp_list)),
        "delta_psnr": float(np.mean(psnr_comp_list)) - float(np.mean(psnr_orig_list)),
        "orig_ssim": float(np.mean(ssim_orig_list)),
        "comp_ssim": float(np.mean(ssim_comp_list)),
        "delta_ssim": float(np.mean(ssim_comp_list)) - float(np.mean(ssim_orig_list)),
        "orig_lpips": float(np.mean(lpips_orig_list)),
        "comp_lpips": float(np.mean(lpips_comp_list)),
        "delta_lpips": float(np.mean(lpips_comp_list)) - float(np.mean(lpips_orig_list)),
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(os.path.join(_root, "results"), exist_ok=True)
    out_path = os.path.join(_root, "results", "full_metrics_21_scenes.json")

    all_results = {}
    total = len(NERF_SYN_SCENES) + len(COLMAP_SCENES)
    done = 0

    # --- NeRF Synthetic ---
    for scene_name in NERF_SYN_SCENES:
        done += 1
        print(f"\n[{done}/{total}] {scene_name} (NeRF Synthetic)")
        model_path = os.path.join(_root, "output", f"{scene_name}_wb")
        source_path = os.path.join(_root, "data", "nerf_synthetic", scene_name)
        t0 = time.perf_counter()
        try:
            result = evaluate_scene(scene_name, model_path, source_path,
                                     white_bg=True, data_device="cuda")
            elapsed = time.perf_counter() - t0
            result["time_s"] = elapsed
            result["dataset"] = "nerf_synthetic"
            if "error" not in result:
                print(f"  Ratio={result['compression_ratio']:.2f}x  "
                      f"PSNR: {result['orig_psnr']:.2f}->{result['comp_psnr']:.2f} ({result['delta_psnr']:+.2f})  "
                      f"SSIM: {result['orig_ssim']:.4f}->{result['comp_ssim']:.4f} ({result['delta_ssim']:+.4f})  "
                      f"LPIPS: {result['orig_lpips']:.4f}->{result['comp_lpips']:.4f} ({result['delta_lpips']:+.4f})  "
                      f"({elapsed:.1f}s)")
            else:
                print(f"  ERROR: {result['error']}")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            tb = traceback.format_exc()
            print(f"  FAILED: {e}\n{tb}")
            result = {"error": str(e), "traceback": tb, "dataset": "nerf_synthetic", "time_s": elapsed}
            _cleanup()

        all_results[scene_name] = result
        # Save intermediate
        with open(out_path, "w") as f:
            json.dump({"config": CONFIG, "results": all_results}, f, indent=2)

    # --- COLMAP ---
    for scene_name in COLMAP_SCENES:
        done += 1
        print(f"\n[{done}/{total}] {scene_name} (COLMAP)")
        model_path = os.path.join(_root, "output", scene_name)
        source_path, white_bg = _load_cfg(scene_name)
        t0 = time.perf_counter()
        try:
            result = evaluate_scene(scene_name, model_path, source_path,
                                     white_bg=white_bg, data_device="cpu")
            elapsed = time.perf_counter() - t0
            result["time_s"] = elapsed
            result["dataset"] = "colmap"
            if "error" not in result:
                print(f"  Ratio={result['compression_ratio']:.2f}x  "
                      f"PSNR: {result['orig_psnr']:.2f}->{result['comp_psnr']:.2f} ({result['delta_psnr']:+.2f})  "
                      f"SSIM: {result['orig_ssim']:.4f}->{result['comp_ssim']:.4f} ({result['delta_ssim']:+.4f})  "
                      f"LPIPS: {result['orig_lpips']:.4f}->{result['comp_lpips']:.4f} ({result['delta_lpips']:+.4f})  "
                      f"({elapsed:.1f}s)")
            else:
                print(f"  ERROR: {result['error']}")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            tb = traceback.format_exc()
            print(f"  FAILED: {e}\n{tb}")
            result = {"error": str(e), "traceback": tb, "dataset": "colmap", "time_s": elapsed}
            _cleanup()

        all_results[scene_name] = result
        # Save intermediate
        with open(out_path, "w") as f:
            json.dump({"config": CONFIG, "results": all_results}, f, indent=2)

    # --- Summary table ---
    print("\n" + "=" * 90)
    print("FULL METRICS SUMMARY (21 scenes)")
    print(f"Config: {CONFIG}")
    print("=" * 90)
    hdr = f"{'Scene':<12} {'Ratio':>6} {'PSNR_o':>7} {'PSNR_c':>7} {'dPSNR':>7}  {'SSIM_o':>7} {'SSIM_c':>7} {'dSSIM':>8}  {'LPIPS_o':>7} {'LPIPS_c':>7} {'dLPIPS':>8}"
    print(hdr)
    print("-" * len(hdr))

    syn_results = []
    colmap_results = []

    for name in NERF_SYN_SCENES + COLMAP_SCENES:
        r = all_results.get(name, {})
        if "error" in r:
            print(f"{name:<12} ERROR: {r['error'][:60]}")
            continue

        ratio = r.get("compression_ratio", float("nan"))
        po = r.get("orig_psnr", float("nan"))
        pc = r.get("comp_psnr", float("nan"))
        dp = r.get("delta_psnr", float("nan"))
        so = r.get("orig_ssim", float("nan"))
        sc = r.get("comp_ssim", float("nan"))
        ds = r.get("delta_ssim", float("nan"))
        lo = r.get("orig_lpips", float("nan"))
        lc = r.get("comp_lpips", float("nan"))
        dl = r.get("delta_lpips", float("nan"))

        print(f"{name:<12} {ratio:>5.1f}x {po:>7.2f} {pc:>7.2f} {dp:>+7.2f}  "
              f"{so:>7.4f} {sc:>7.4f} {ds:>+8.5f}  "
              f"{lo:>7.4f} {lc:>7.4f} {dl:>+8.5f}")

        if r.get("dataset") == "nerf_synthetic":
            syn_results.append(r)
        else:
            colmap_results.append(r)

    # Averages
    def _avg(results, key):
        vals = [r[key] for r in results if key in r and not np.isnan(r[key])]
        return np.mean(vals) if vals else float("nan")

    if syn_results:
        print(f"\n{'NeRF-Syn AVG':<12} {_avg(syn_results, 'compression_ratio'):>5.1f}x "
              f"{_avg(syn_results, 'orig_psnr'):>7.2f} {_avg(syn_results, 'comp_psnr'):>7.2f} {_avg(syn_results, 'delta_psnr'):>+7.2f}  "
              f"{_avg(syn_results, 'orig_ssim'):>7.4f} {_avg(syn_results, 'comp_ssim'):>7.4f} {_avg(syn_results, 'delta_ssim'):>+8.5f}  "
              f"{_avg(syn_results, 'orig_lpips'):>7.4f} {_avg(syn_results, 'comp_lpips'):>7.4f} {_avg(syn_results, 'delta_lpips'):>+8.5f}")

    if colmap_results:
        print(f"{'COLMAP AVG':<12} {_avg(colmap_results, 'compression_ratio'):>5.1f}x "
              f"{_avg(colmap_results, 'orig_psnr'):>7.2f} {_avg(colmap_results, 'comp_psnr'):>7.2f} {_avg(colmap_results, 'delta_psnr'):>+7.2f}  "
              f"{_avg(colmap_results, 'orig_ssim'):>7.4f} {_avg(colmap_results, 'comp_ssim'):>7.4f} {_avg(colmap_results, 'delta_ssim'):>+8.5f}  "
              f"{_avg(colmap_results, 'orig_lpips'):>7.4f} {_avg(colmap_results, 'comp_lpips'):>7.4f} {_avg(colmap_results, 'delta_lpips'):>+8.5f}")

    all_valid = syn_results + colmap_results
    if all_valid:
        print(f"{'ALL AVG':<12} {_avg(all_valid, 'compression_ratio'):>5.1f}x "
              f"{_avg(all_valid, 'orig_psnr'):>7.2f} {_avg(all_valid, 'comp_psnr'):>7.2f} {_avg(all_valid, 'delta_psnr'):>+7.2f}  "
              f"{_avg(all_valid, 'orig_ssim'):>7.4f} {_avg(all_valid, 'comp_ssim'):>7.4f} {_avg(all_valid, 'delta_ssim'):>+8.5f}  "
              f"{_avg(all_valid, 'orig_lpips'):>7.4f} {_avg(all_valid, 'comp_lpips'):>7.4f} {_avg(all_valid, 'delta_lpips'):>+8.5f}")

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
