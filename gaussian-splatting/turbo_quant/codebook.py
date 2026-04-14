"""TurboQuant codebook generation via Lloyd-Max algorithm.

Based on Lemma 1 of TurboQuant (arXiv:2504.19874): when a unit vector in S^{d-1}
is uniformly distributed, each coordinate follows the Beta marginal distribution.
The optimal quantizer (minimizing MSE) is found by the Lloyd-Max algorithm under
this distribution.
"""

import numpy as np
from functools import lru_cache
from scipy.special import gammaln


# Dense grid for numerical integration
_N_GRID = 100_000
_EPS = 1e-9


def beta_pdf(x: float, d: int) -> float:
    """Evaluate the Beta marginal PDF for a coordinate of a uniform unit vector.

    For a uniformly distributed unit vector in S^{d-1}, each coordinate x follows:
        f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)

    For d < 3, the distribution is uniform: f_X(x) = 1/2.

    Args:
        x: Coordinate value in [-1, 1].
        d: Ambient dimension.

    Returns:
        PDF value at x.
    """
    if d < 3:
        return 0.5

    # Clamp x to avoid numerical issues at the boundary
    x_sq = float(x) ** 2
    x_sq = min(x_sq, 1.0 - _EPS)

    # Use log-gamma to avoid overflow for large d
    log_norm = gammaln(d / 2) - 0.5 * np.log(np.pi) - gammaln((d - 1) / 2)
    exponent = (d - 3) / 2
    return np.exp(log_norm) * (1.0 - x_sq) ** exponent


def _beta_pdf_vectorized(x: np.ndarray, d: int) -> np.ndarray:
    """Vectorized version of beta_pdf for arrays."""
    if d < 3:
        return np.full_like(x, 0.5)

    # Use log-gamma to avoid overflow for large d
    log_norm = gammaln(d / 2) - 0.5 * np.log(np.pi) - gammaln((d - 1) / 2)
    norm_const = np.exp(log_norm)
    exponent = (d - 3) / 2
    x_sq = np.clip(x ** 2, 0.0, 1.0 - _EPS)
    return norm_const * (1.0 - x_sq) ** exponent


@lru_cache(maxsize=256)
def generate_codebook(d: int, b: int) -> np.ndarray:
    """Generate optimal Lloyd-Max codebook for dimension d at bit-width b.

    The codebook minimizes E[|X - c_nearest|^2] under the Beta marginal PDF
    for coordinates of uniform unit vectors in S^{d-1}.

    Algorithm:
        1. Initialize 2^b centroids uniformly in [-1, 1].
        2. Repeat until convergence:
           a. Compute cell boundaries as midpoints between adjacent centroids.
           b. Update each centroid to the PDF-weighted mean within its cell.
        3. Return sorted centroids.

    Args:
        d: Ambient dimension.
        b: Bit-width (number of bits per coordinate).

    Returns:
        Sorted numpy array of 2^b centroids in [-1, 1].
    """
    n_centroids = 2 ** b

    # Dense grid for numerical integration
    x_grid = np.linspace(-1.0 + _EPS, 1.0 - _EPS, _N_GRID)
    dx = x_grid[1] - x_grid[0]
    pdf_vals = _beta_pdf_vectorized(x_grid, d)

    # Initialize centroids uniformly, excluding endpoints
    centroids = np.linspace(-1.0 + 2.0 / (n_centroids + 1),
                             1.0 - 2.0 / (n_centroids + 1),
                             n_centroids)

    max_iters = 500
    tol = 1e-10

    for _ in range(max_iters):
        old_centroids = centroids.copy()

        # Step 1: Compute cell boundaries (midpoints between adjacent centroids)
        boundaries = np.empty(n_centroids + 1)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])

        # Step 2: Update centroids as PDF-weighted mean in each cell
        new_centroids = np.empty(n_centroids)
        for k in range(n_centroids):
            lo, hi = boundaries[k], boundaries[k + 1]
            mask = (x_grid >= lo) & (x_grid < hi)
            w = pdf_vals[mask]
            xk = x_grid[mask]
            total_weight = w.sum()
            if total_weight > 0:
                new_centroids[k] = (w * xk).sum() / total_weight
            else:
                # Fallback: keep the midpoint of the cell
                new_centroids[k] = 0.5 * (lo + hi)

        centroids = new_centroids

        # Check convergence
        if np.max(np.abs(centroids - old_centroids)) < tol:
            break

    return np.sort(centroids)


@lru_cache(maxsize=256)
def compute_mse_cost(d: int, b: int) -> float:
    """Compute per-coordinate MSE cost for quantizing the Beta distribution.

    This is E[|X - Q(X)|^2] where Q is the Lloyd-Max quantizer defined by the
    codebook for dimension d at bit-width b.

    Args:
        d: Ambient dimension.
        b: Bit-width.

    Returns:
        Per-coordinate MSE cost C(f_X, b).
        Multiply by d to get total MSE for a unit-norm vector.
    """
    centroids = generate_codebook(d, b)
    n_centroids = len(centroids)

    x_grid = np.linspace(-1.0 + _EPS, 1.0 - _EPS, _N_GRID)
    pdf_vals = _beta_pdf_vectorized(x_grid, d)

    # Compute cell boundaries
    boundaries = np.empty(n_centroids + 1)
    boundaries[0] = -1.0
    boundaries[-1] = 1.0
    boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])

    mse = 0.0
    for k in range(n_centroids):
        lo, hi = boundaries[k], boundaries[k + 1]
        mask = (x_grid >= lo) & (x_grid < hi)
        w = pdf_vals[mask]
        xk = x_grid[mask]
        mse += (w * (xk - centroids[k]) ** 2).sum()

    # Normalize: integrate pdf_vals * dx should be ~1
    dx = x_grid[1] - x_grid[0]
    mse *= dx

    return float(mse)
