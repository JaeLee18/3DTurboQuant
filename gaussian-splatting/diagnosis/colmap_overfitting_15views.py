#!/usr/bin/env python3
"""
Overfitting diagnosis on all 13 COLMAP scenes using 15 views.

Measures train/test gap and per-band R_k ratios (overfitting ratio).
Uses 15 evenly-spaced train views and min(15, all) test views for
reduced noise compared to the earlier 5-view measurements.

Usage:
    cd /mnt/ssd1/idea/TurboQuant/gaussian-splatting
    /mnt/ssd1/conda_envs/gs_compression/bin/python -m diagnosis.colmap_overfitting_15views
"""

import gc
import json
import os
import sys
import torch
from argparse import Namespace

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.image_utils import psnr

# --- Config ---
N_EVAL_VIEWS = 15
ITERATION = 30000

SCENES = {
    # name: (source_path_relative_to_gs_root, white_background)
    "bicycle":   ("data/360_v2/bicycle",   False),
    "bonsai":    ("data/360_v2/bonsai",    False),
    "counter":   ("data/360_v2/counter",   False),
    "flowers":   ("data/360_v2/flowers",   False),
    "garden":    ("data/360_v2/garden",    False),
    "kitchen":   ("data/360_v2/kitchen",   False),
    "room":      ("data/360_v2/room",      False),
    "stump":     ("data/360_v2/stump",     False),
    "treehill":  ("data/360_v2/treehill",  False),
    "truck":     ("data/tandt/truck",      False),
    "train":     ("data/tandt/train",      False),
    "playroom":  ("data/db/playroom",      False),
    "drjohnson": ("data/db/drjohnson",     False),
}

SH_BAND_SLICES = {
    1: (0, 3),
    2: (3, 8),
    3: (8, 15),
}

OUTPUT_PATH = os.path.join(_root, "results", "colmap_overfitting_15views.json")


def _evenly_spaced(lst, n):
    """Select n evenly-spaced items from lst."""
    total = len(lst)
    if n >= total:
        return list(lst)
    indices = [int(round(i * (total - 1) / (n - 1))) for i in range(n)]
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    return [lst[i] for i in unique]


def _avg_psnr(cameras, gaussians, pipe, bg):
    """Average PSNR over cameras. Expects torch.no_grad() context outside."""
    total = 0.0
    for cam in cameras:
        rendering = render(cam, gaussians, pipe, bg, separate_sh=False)["render"]
        gt = cam.original_image[:3, :, :].to("cuda")
        total += psnr(rendering, gt).mean().item()
    return total / len(cameras) if cameras else float("nan")


def process_scene(scene_name, source_rel, white_bg):
    """Run full overfitting diagnosis on one scene."""
    model_path = os.path.join(_root, "output", scene_name)
    source_path = os.path.join(_root, source_rel)

    args = Namespace(
        sh_degree=3,
        source_path=os.path.abspath(source_path),
        model_path=os.path.abspath(model_path),
        images="images",
        depths="",
        resolution=-1,
        white_background=white_bg,
        train_test_exp=False,
        data_device="cpu",  # CRITICAL: keep images on CPU
        eval=True,
    )

    pipe = Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False,
    )

    bg_color = [1, 1, 1] if white_bg else [0, 0, 0]
    bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians, load_iteration=ITERATION, shuffle=False)

    all_train = scene.getTrainCameras()
    all_test = scene.getTestCameras()

    train_cams = _evenly_spaced(all_train, N_EVAL_VIEWS)
    test_cams = _evenly_spaced(all_test, N_EVAL_VIEWS)

    n_train_used = len(train_cams)
    n_test_used = len(test_cams)

    print(f"  Using {n_train_used}/{len(all_train)} train, {n_test_used}/{len(all_test)} test views")

    with torch.no_grad():
        # Baseline
        baseline_train = _avg_psnr(train_cams, gaussians, pipe, bg)
        baseline_test = _avg_psnr(test_cams, gaussians, pipe, bg)
        gap = baseline_train - baseline_test
        print(f"  Baseline: train={baseline_train:.2f}, test={baseline_test:.2f}, gap={gap:+.2f}")

        orig_features_rest = gaussians._features_rest.data.clone()

        band_results = {}
        for band, (start, end) in SH_BAND_SLICES.items():
            # Zero band k
            gaussians._features_rest.data[:, start:end, :] = 0.0

            zeroed_train = _avg_psnr(train_cams, gaussians, pipe, bg)
            zeroed_test = _avg_psnr(test_cams, gaussians, pipe, bg)

            train_drop = baseline_train - zeroed_train
            test_drop = baseline_test - zeroed_test

            # Handle division carefully
            if abs(test_drop) < 1e-4:
                ratio = float("inf") if train_drop > 0 else float("nan")
            elif test_drop < 0:
                # Negative test drop means zeroing HELPED test => strong overfitting signal
                ratio = float("inf") if train_drop > 0 else train_drop / test_drop
            else:
                ratio = train_drop / test_drop

            print(f"  Band {band}: train_drop={train_drop:+.3f}, test_drop={test_drop:+.3f}, R={ratio:.2f}")

            band_results[band] = {
                "train_drop": train_drop,
                "test_drop": test_drop,
                "ratio": ratio,
            }

            # Restore
            gaussians._features_rest.data.copy_(orig_features_rest)
            torch.cuda.empty_cache()

    result = {
        "gap": round(gap, 4),
        "R1": round(band_results[1]["ratio"], 4) if not (band_results[1]["ratio"] == float("inf") or band_results[1]["ratio"] != band_results[1]["ratio"]) else str(band_results[1]["ratio"]),
        "R2": round(band_results[2]["ratio"], 4) if not (band_results[2]["ratio"] == float("inf") or band_results[2]["ratio"] != band_results[2]["ratio"]) else str(band_results[2]["ratio"]),
        "R3": round(band_results[3]["ratio"], 4) if not (band_results[3]["ratio"] == float("inf") or band_results[3]["ratio"] != band_results[3]["ratio"]) else str(band_results[3]["ratio"]),
        "train_psnr": round(baseline_train, 4),
        "test_psnr": round(baseline_test, 4),
        "n_train_used": n_train_used,
        "n_test_used": n_test_used,
    }

    # Clean up
    del gaussians, scene, all_train, all_test, train_cams, test_cams, orig_features_rest
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main():
    results = {}
    scene_order = [
        "bicycle", "bonsai", "counter", "flowers", "garden",
        "kitchen", "room", "stump", "treehill",
        "truck", "train", "playroom", "drjohnson",
    ]

    for i, scene_name in enumerate(scene_order):
        source_rel, white_bg = SCENES[scene_name]
        print(f"\n[{i+1}/{len(scene_order)}] Processing {scene_name} ...")
        try:
            results[scene_name] = process_scene(scene_name, source_rel, white_bg)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[scene_name] = {"error": str(e)}

    # Save JSON
    output = {
        "n_eval_views": N_EVAL_VIEWS,
        "results": results,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")

    # Print summary table
    print(f"\n{'Scene':<12} {'Gap':>7} {'R1':>7} {'R2':>7} {'R3':>7}  {'TrainPSNR':>10} {'TestPSNR':>10}")
    print("-" * 72)
    for scene_name in scene_order:
        r = results.get(scene_name, {})
        if "error" in r:
            print(f"{scene_name:<12} ERROR: {r['error']}")
            continue

        def _fmt(v):
            if isinstance(v, str):
                return f"{v:>7}"
            return f"{v:>7.2f}"

        print(
            f"{scene_name:<12} "
            f"{r['gap']:>+7.2f} "
            f"{_fmt(r['R1'])} "
            f"{_fmt(r['R2'])} "
            f"{_fmt(r['R3'])}  "
            f"{r['train_psnr']:>10.2f} "
            f"{r['test_psnr']:>10.2f}"
        )


if __name__ == "__main__":
    main()
