#
# diagnosis/sh_band_analysis.py
# Per-band SH overfitting analysis for a trained 3DGS model.
#
# SH band structure stored in gaussians._features_rest  (N, 15, 3):
#   Band 1  — indices 0:3   (3 coefficients per channel)
#   Band 2  — indices 3:8   (5 coefficients per channel)
#   Band 3  — indices 8:15  (7 coefficients per channel)
#
# Overfitting ratio R_k = train_drop_k / test_drop_k
#   A ratio > 1 means the band contributes more to train quality than test
#   quality, i.e., it encodes scene-specific overfitting rather than
#   view-dependent appearance.
#
# Usage:
#   python -m diagnosis.sh_band_analysis -m output/lego_wb -s data/nerf_synthetic/lego
#

import os
import sys
import copy
import torch
from argparse import Namespace

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.image_utils import psnr


# SH band layout inside _features_rest (degree 3 → 15 coefficients)
SH_BAND_SLICES = {
    1: (0, 3),   # l=1: 3 coefficients
    2: (3, 8),   # l=2: 5 coefficients
    3: (8, 15),  # l=3: 7 coefficients
}


def _make_args(model_path: str, source_path: str, white_background: bool) -> Namespace:
    return Namespace(
        sh_degree=3,
        source_path=os.path.abspath(source_path),
        model_path=os.path.abspath(model_path),
        images="images",
        depths="",
        resolution=-1,
        white_background=white_background,
        train_test_exp=False,
        data_device="cuda",
        eval=True,
    )


def _avg_psnr(cameras, gaussians, pipe, bg) -> float:
    """Return average PSNR over a camera list (no_grad context expected outside)."""
    total = 0.0
    for cam in cameras:
        rendering = render(cam, gaussians, pipe, bg, separate_sh=False)["render"]
        gt = cam.original_image[:3, :, :].to("cuda")
        total += psnr(rendering, gt).mean().item()
    return total / len(cameras) if cameras else float("nan")


def analyze_sh_bands(
    model_path: str,
    source_path: str,
    iteration: int = 30000,
    white_background: bool = True,
) -> dict:
    """
    Measure per-band SH overfitting by zeroing each band and observing PSNR drop.

    Returns a dict with:
        baseline_train_psnr  – float
        baseline_test_psnr   – float
        bands                – dict keyed by band index (1, 2, 3), each containing:
            train_psnr_zeroed   float  – train PSNR with this band zeroed
            test_psnr_zeroed    float  – test PSNR with this band zeroed
            train_drop          float  – baseline_train - train_psnr_zeroed  (dB)
            test_drop           float  – baseline_test  - test_psnr_zeroed   (dB)
            overfitting_ratio   float  – train_drop / test_drop
    """
    args = _make_args(model_path, source_path, white_background)

    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False)

    bg_color = [1, 1, 1] if white_background else [0, 0, 0]
    bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    pipe = Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False,
    )

    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()

    with torch.no_grad():
        print("Computing baseline PSNRs ...")
        baseline_train = _avg_psnr(train_cams, gaussians, pipe, bg)
        baseline_test = _avg_psnr(test_cams, gaussians, pipe, bg)
        print(f"  Baseline train: {baseline_train:.2f} dB")
        print(f"  Baseline test : {baseline_test:.2f} dB")

        # Save original _features_rest so we can restore after each zeroing
        orig_features_rest = gaussians._features_rest.data.clone()

        band_results = {}
        for band, (start, end) in SH_BAND_SLICES.items():
            print(f"\nZeroing band l={band} (coeffs {start}:{end}) ...")
            # Zero the target band
            gaussians._features_rest.data[:, start:end, :] = 0.0

            zeroed_train = _avg_psnr(train_cams, gaussians, pipe, bg)
            zeroed_test = _avg_psnr(test_cams, gaussians, pipe, bg)

            train_drop = baseline_train - zeroed_train
            test_drop = baseline_test - zeroed_test
            ratio = train_drop / test_drop if test_drop != 0 else float("inf")

            print(f"  Train PSNR (zeroed): {zeroed_train:.2f} dB  (drop {train_drop:+.2f} dB)")
            print(f"  Test  PSNR (zeroed): {zeroed_test:.2f} dB  (drop {test_drop:+.2f} dB)")
            print(f"  Overfitting ratio  : {ratio:.3f}")

            band_results[band] = {
                "train_psnr_zeroed": zeroed_train,
                "test_psnr_zeroed": zeroed_test,
                "train_drop": train_drop,
                "test_drop": test_drop,
                "overfitting_ratio": ratio,
            }

            # Restore original coefficients for the next band test
            gaussians._features_rest.data.copy_(orig_features_rest)

    return {
        "baseline_train_psnr": baseline_train,
        "baseline_test_psnr": baseline_test,
        "bands": band_results,
    }


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Per-band SH overfitting analysis for a trained 3DGS model"
    )
    parser.add_argument("-m", "--model_path", required=True, help="Path to trained model output directory")
    parser.add_argument("-s", "--source_path", required=True, help="Path to dataset source directory")
    parser.add_argument("--iteration", type=int, default=30000, help="Checkpoint iteration (default: 30000)")
    parser.add_argument("--no_white_background", action="store_true", help="Use black background instead of white")
    parser.add_argument("--output", type=str, default=None, help="Path to save results as JSON")
    args = parser.parse_args()

    white_background = not args.no_white_background
    results = analyze_sh_bands(
        model_path=args.model_path,
        source_path=args.source_path,
        iteration=args.iteration,
        white_background=white_background,
    )

    print("\n=== SH Band Overfitting Analysis ===")
    print(f"  Baseline train PSNR : {results['baseline_train_psnr']:.2f} dB")
    print(f"  Baseline test  PSNR : {results['baseline_test_psnr']:.2f} dB")
    print(f"  Train/test gap      : {results['baseline_train_psnr'] - results['baseline_test_psnr']:.2f} dB\n")

    print(f"  {'Band':<6} {'Train drop':>12} {'Test drop':>12} {'Ratio (R_k)':>14}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*14}")
    for band in sorted(results["bands"]):
        b = results["bands"][band]
        print(
            f"  l={band:<4} "
            f"{b['train_drop']:>+11.2f} dB "
            f"{b['test_drop']:>+11.2f} dB "
            f"{b['overfitting_ratio']:>13.3f}"
        )
    print()
    print("  R_k > 1 → band encodes overfitting; R_k >> 1 → strongly overfit.")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        # Convert int keys to strings for JSON
        json_results = {
            "baseline_train_psnr": results["baseline_train_psnr"],
            "baseline_test_psnr": results["baseline_test_psnr"],
            "bands": {str(k): v for k, v in results["bands"].items()},
        }
        with open(args.output, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
