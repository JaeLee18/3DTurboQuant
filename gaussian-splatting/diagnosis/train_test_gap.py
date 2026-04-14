#
# diagnosis/train_test_gap.py
# Measure train vs test PSNR gap for a trained 3DGS model.
#
# Usage:
#   python -m diagnosis.train_test_gap -m output/lego_wb -s data/nerf_synthetic/lego
#

import os
import sys
import torch
from argparse import Namespace

# Make sure we can import from the project root when run as a module
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.image_utils import psnr


def _make_args(model_path: str, source_path: str, white_background: bool) -> Namespace:
    """Build a minimal ModelParams-compatible Namespace."""
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


def _render_cameras(cameras, gaussians, pipe, bg):
    """Render a list of cameras and return per-image PSNR values."""
    psnr_values = []
    with torch.no_grad():
        for cam in cameras:
            rendering = render(cam, gaussians, pipe, bg, separate_sh=False)["render"]
            gt = cam.original_image[:3, :, :].to("cuda")
            p = psnr(rendering, gt).mean().item()
            psnr_values.append(p)
    return psnr_values


def compute_split_psnr(
    model_path: str,
    source_path: str,
    iteration: int = 30000,
    white_background: bool = True,
) -> dict:
    """
    Load a trained 3DGS model and measure train/test PSNR gap.

    Returns:
        dict with keys:
            train_psnr  – average PSNR over training views (dB)
            test_psnr   – average PSNR over test views (dB)
            gap         – train_psnr - test_psnr (dB)
            train_psnrs – per-image PSNR list for train set
            test_psnrs  – per-image PSNR list for test set
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

    print("Rendering train set ...")
    train_psnrs = _render_cameras(scene.getTrainCameras(), gaussians, pipe, bg)

    print("Rendering test set ...")
    test_psnrs = _render_cameras(scene.getTestCameras(), gaussians, pipe, bg)

    train_avg = sum(train_psnrs) / len(train_psnrs) if train_psnrs else float("nan")
    test_avg = sum(test_psnrs) / len(test_psnrs) if test_psnrs else float("nan")
    gap = train_avg - test_avg

    return {
        "train_psnr": train_avg,
        "test_psnr": test_avg,
        "gap": gap,
        "train_psnrs": train_psnrs,
        "test_psnrs": test_psnrs,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Measure train/test PSNR gap for a 3DGS model")
    parser.add_argument("-m", "--model_path", required=True, help="Path to trained model output directory")
    parser.add_argument("-s", "--source_path", required=True, help="Path to dataset source directory")
    parser.add_argument("--iteration", type=int, default=30000, help="Checkpoint iteration (default: 30000)")
    parser.add_argument("--no_white_background", action="store_true", help="Use black background instead of white")
    args = parser.parse_args()

    white_background = not args.no_white_background
    results = compute_split_psnr(
        model_path=args.model_path,
        source_path=args.source_path,
        iteration=args.iteration,
        white_background=white_background,
    )

    print("\n=== Train / Test PSNR Gap ===")
    print(f"  Train PSNR : {results['train_psnr']:.2f} dB  (n={len(results['train_psnrs'])})")
    print(f"  Test  PSNR : {results['test_psnr']:.2f} dB  (n={len(results['test_psnrs'])})")
    print(f"  Gap        : {results['gap']:.2f} dB")
    print()

    if results["train_psnrs"]:
        print("Per-image train PSNRs:")
        for i, p in enumerate(results["train_psnrs"]):
            print(f"  train[{i:03d}]: {p:.2f} dB")
    if results["test_psnrs"]:
        print("Per-image test PSNRs:")
        for i, p in enumerate(results["test_psnrs"]):
            print(f"  test[{i:03d}]: {p:.2f} dB")


if __name__ == "__main__":
    main()
