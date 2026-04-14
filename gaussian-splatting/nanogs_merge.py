"""Wrapper around NanoGS's MPMM (Moment-matched Pairwise Merge and Map)
algorithm for Gaussian merging in the TurboSplat compression pipeline.

Usage:
    from nanogs_merge import merge_with_nanogs
    stats = merge_with_nanogs('input.ply', 'output.ply', ratio=0.5)
"""

import os
import sys
import time

# Ensure NanoGS's root is on the path so its internal `utils.*` imports resolve.
_NANOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NanoGS")
if _NANOGS_DIR not in sys.path:
    sys.path.insert(0, _NANOGS_DIR)

from NanoGS.simplification import simplify
from NanoGS.utils.params import RunParams, CostParams


def merge_with_nanogs(
    ply_path: str,
    output_ply_path: str,
    ratio: float = 0.5,
    k: int = 16,
    merge_cap: float = 0.5,
    opacity_threshold: float = 0.1,
    lam_geo: float = 1.0,
    lam_sh: float = 1.0,
) -> dict:
    """Merge Gaussians using NanoGS's MPMM algorithm.

    Args:
        ply_path: Path to input PLY with all Gaussians.
        output_ply_path: Path for output PLY with merged Gaussians.
        ratio: Fraction of splats to KEEP (0.5 = keep 50%, merge away 50%).
        k: Number of nearest neighbours for the merge graph.
        merge_cap: Max merges per pass as fraction of original count (0.01-0.5).
        opacity_threshold: Prune splats with opacity below this before merging.
            Effective threshold is min(threshold, median(opacity)).
        lam_geo: Weight for geometric (KL divergence) cost term.
        lam_sh: Weight for SH (L2 distance) cost term.

    Returns:
        dict with keys:
            n_original  - number of Gaussians in input PLY
            n_merged    - number of Gaussians in output PLY
            merge_time_s - wall-clock seconds for the merge
            output_path  - path to the written PLY
    """
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"Input PLY not found: {ply_path}")
    if not (0.0 < ratio < 1.0):
        raise ValueError(f"ratio must be in (0, 1), got {ratio}")

    # Count input Gaussians by reading the header
    n_original = _count_ply_vertices(ply_path)

    rp = RunParams(
        ratio=ratio,
        merge_cap=max(0.01, min(0.5, merge_cap)),
        k=k,
        opacity_threshold=opacity_threshold,
    )
    cp = CostParams(lam_geo=lam_geo, lam_sh=lam_sh)

    os.makedirs(os.path.dirname(os.path.abspath(output_ply_path)), exist_ok=True)

    t0 = time.time()
    simplify(ply_path, output_ply_path, rp, cp)
    elapsed = time.time() - t0

    n_merged = _count_ply_vertices(output_ply_path)

    return {
        "n_original": n_original,
        "n_merged": n_merged,
        "merge_time_s": elapsed,
        "output_path": output_ply_path,
    }


def _count_ply_vertices(path: str) -> int:
    """Read vertex count from a PLY header without loading the full file."""
    with open(path, "rb") as f:
        for raw_line in f:
            line = raw_line.decode("ascii", errors="replace").strip()
            if line.startswith("element vertex"):
                return int(line.split()[2])
            if line == "end_header":
                break
    raise ValueError(f"Could not find vertex count in PLY header: {path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="NanoGS merge wrapper")
    ap.add_argument("input_ply", help="Input PLY path")
    ap.add_argument("output_ply", help="Output PLY path")
    ap.add_argument("--ratio", type=float, default=0.5, help="Fraction to keep")
    ap.add_argument("--k", type=int, default=16, help="KNN neighbours")
    ap.add_argument("--merge_cap", type=float, default=0.5)
    ap.add_argument("--opacity_threshold", type=float, default=0.1)
    args = ap.parse_args()

    stats = merge_with_nanogs(
        args.input_ply,
        args.output_ply,
        ratio=args.ratio,
        k=args.k,
        merge_cap=args.merge_cap,
        opacity_threshold=args.opacity_threshold,
    )
    print(f"Done: {stats['n_original']} -> {stats['n_merged']} "
          f"({stats['merge_time_s']:.1f}s)")
