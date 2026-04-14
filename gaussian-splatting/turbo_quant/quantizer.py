"""TurboQuant vector quantizer.

Implements data-oblivious vector quantization via random rotation followed by
per-coordinate scalar quantization using the Lloyd-Max codebook.

Key insight (TurboQuant, arXiv:2504.19874): After a random rotation, coordinates
of a unit vector are nearly independent and Beta-distributed, making per-coordinate
scalar quantization near-optimal.
"""

import numpy as np
from .codebook import generate_codebook


class TurboQuantizer:
    """Data-oblivious vector quantizer using random rotation + scalar codebook.

    Args:
        d: Dimension of the vectors to quantize.
        b: Bit-width per coordinate (1..8; uint8 indices, so max is 8).
        seed: Random seed for reproducible rotation matrix generation.
    """

    def __init__(self, d: int, b: int, seed: int = 0) -> None:
        self.d = d
        self.b = b

        # Generate random rotation matrix via QR decomposition of a d×d Gaussian matrix
        rng = np.random.default_rng(seed)
        G = rng.standard_normal((d, d))
        Q, _ = np.linalg.qr(G)
        # QR is only defined up to sign; ensure it is a proper rotation (det > 0)
        if np.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        self._rotation = Q.astype(np.float64)  # (d, d)

        # Load codebook and precompute boundaries
        centroids = generate_codebook(d, b)  # sorted, shape (2^b,)
        self._centroids = centroids.astype(np.float64)

        # Boundaries are midpoints between adjacent centroids; outer bounds are ±∞
        n = len(centroids)
        self._boundaries = np.empty(n - 1, dtype=np.float64)
        self._boundaries[:] = 0.5 * (centroids[:-1] + centroids[1:])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quantize(self, vectors: np.ndarray) -> tuple:
        """Quantize vectors to per-coordinate codebook indices plus L2 norms.

        Args:
            vectors: Float array of shape (N, d).

        Returns:
            indices: uint8 array of shape (N, d), each value in [0, 2^b - 1].
            norms:   float32 array of shape (N,), L2 norm of each input vector.
        """
        vectors = np.asarray(vectors, dtype=np.float64)
        N, d = vectors.shape
        assert d == self.d, f"Expected d={self.d}, got {d}"

        # Store L2 norms
        norms = np.linalg.norm(vectors, axis=1)  # (N,)

        # Normalize to unit sphere; guard against zero vectors
        safe_norms = np.where(norms == 0.0, 1.0, norms)
        unit_vecs = vectors / safe_norms[:, np.newaxis]  # (N, d)

        # Rotate: y[i] = R @ x[i]   →   Y = X @ R^T  (equivalent, row-major friendly)
        rotated = unit_vecs @ self._rotation.T  # (N, d)

        # Per-coordinate scalar quantization via searchsorted on boundaries
        # searchsorted returns index k such that boundaries[k-1] <= coord < boundaries[k]
        # which maps to centroid index k in [0, n_centroids-1]
        indices = np.searchsorted(self._boundaries, rotated).astype(np.uint8)  # (N, d)

        return indices, norms.astype(np.float32)

    def dequantize(self, indices: np.ndarray, norms: np.ndarray) -> np.ndarray:
        """Reconstruct vectors from indices and norms.

        Args:
            indices: uint8 array of shape (N, d), values in [0, 2^b - 1].
            norms:   float32 or float64 array of shape (N,).

        Returns:
            Reconstructed float32 array of shape (N, d).
        """
        indices = np.asarray(indices, dtype=np.uint8)
        norms = np.asarray(norms, dtype=np.float64)

        # Look up centroids for each index
        y_hat = self._centroids[indices]  # (N, d), float64

        # Inverse rotate: x_hat[i] = R^T @ y_hat[i]   →   X_hat = Y_hat @ R
        x_hat = y_hat @ self._rotation  # (N, d)

        # Rescale by norms
        x_hat *= norms[:, np.newaxis]

        return x_hat.astype(np.float32)

    # ------------------------------------------------------------------
    # Accessors for serialization
    # ------------------------------------------------------------------

    def get_rotation_matrix(self) -> np.ndarray:
        """Return the (d, d) float64 rotation matrix."""
        return self._rotation.copy()

    def get_centroids(self) -> np.ndarray:
        """Return the sorted (2^b,) float64 centroid array."""
        return self._centroids.copy()
