"""SQR: Stochastic Quantization Regularization for 3DGS SH coefficients.

Injects TurboQuant-style quantization noise during training using a
straight-through estimator (STE), following the Quant-Noise paradigm
(Meta 2020) but with TurboQuant's principled Lloyd-Max codebook noise.

Usage during training (after densification, ~15K iterations):
    sqr = SQRModule(d=45, b=3, p=0.5).cuda()
    ...
    sh_noisy = sqr(gaussians.get_features[:, 1:, :])  # skip DC band
"""

import torch
import torch.nn as nn
from turbo_quant.codebook import generate_codebook


class SQRFunction(torch.autograd.Function):
    """Straight-through quantizer: quantize forward, pass gradients unchanged."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, centroids: torch.Tensor,
                boundaries: torch.Tensor) -> torch.Tensor:
        """Quantize each element of x to the nearest centroid.

        Args:
            x: Input tensor of arbitrary shape, values expected in [-1, 1].
            centroids: 1-D tensor of sorted centroid values (2^b entries).
            boundaries: 1-D tensor of decision boundaries (2^b + 1 entries),
                        where boundaries[k] and boundaries[k+1] bracket centroid k.

        Returns:
            Tensor with the same shape as x; each value replaced by its
            nearest centroid.
        """
        # searchsorted returns index i such that boundaries[i-1] <= x < boundaries[i]
        # Clamp to [0, n_centroids-1] to handle values outside the boundary range.
        flat = x.reshape(-1)
        indices = torch.searchsorted(boundaries.contiguous(), flat.contiguous())
        # boundaries has length n_centroids+1; valid centroid indices are 0..n_centroids-1
        indices = indices.clamp(0, centroids.shape[0] - 1)
        quantized = centroids[indices].reshape(x.shape)
        return quantized

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Straight-through: pass gradient directly to x, None for buffers."""
        return grad_output, None, None


class SQRModule(nn.Module):
    """Stochastic Quantization Regularization module.

    Randomly injects quantization noise into SH rest coefficients during
    training using TurboQuant's Lloyd-Max codebook.  At inference the module
    is a strict identity (no noise, no overhead).

    The noise is applied in a randomly-rotated coordinate frame so that no
    fixed axis is systematically degraded across training steps.

    Args:
        d: Ambient dimension for the codebook (d = n_coeffs * 3 = 45 for
           degree-3 SH rest bands).
        b: Bit-width (number of bits per coordinate).  2^b centroids.
        p: Probability of *skipping* noise injection on any given forward
           call.  p=0.5 means noise is injected 50 % of the time.
    """

    def __init__(self, d: int = 45, b: int = 3, p: float = 0.5):
        super().__init__()
        self.d = d
        self.b = b
        self.p = p

        # Build codebook (numpy) and derive boundaries
        centroids_np = generate_codebook(d, b)          # shape (2^b,)
        import numpy as np
        n = len(centroids_np)
        boundaries_np = np.empty(n + 1, dtype=np.float32)
        boundaries_np[0] = -1.0
        boundaries_np[-1] = 1.0
        boundaries_np[1:-1] = 0.5 * (centroids_np[:-1] + centroids_np[1:])

        # Register as buffers so they are moved to GPU with .cuda() / .to(device)
        self.register_buffer(
            "centroids",
            torch.from_numpy(centroids_np.astype("float32"))
        )
        self.register_buffer(
            "boundaries",
            torch.from_numpy(boundaries_np)
        )

    def forward(self, sh_rest: torch.Tensor) -> torch.Tensor:
        """Apply stochastic quantization noise to SH rest coefficients.

        Args:
            sh_rest: Tensor of shape (N, n_coeffs, 3).  Typically the
                     non-DC SH bands (n_coeffs=15 for degree-3 SH).

        Returns:
            Tensor of the same shape.  In eval mode or when the stochastic
            skip fires, returns the input unchanged (no copy, same object).
            Otherwise returns a differentiable noisy version via STE.
        """
        # Eval mode: pure identity, no cost
        if not self.training:
            return sh_rest

        # Stochastic skip: with probability p do nothing
        if torch.rand(1).item() < self.p:
            return sh_rest

        N = sh_rest.shape[0]
        device = sh_rest.device
        dtype = sh_rest.dtype

        # --- Step 1: flatten to (N, d) ---
        x = sh_rest.reshape(N, self.d)

        # --- Step 2: compute per-vector norms and normalise to unit sphere ---
        norms = x.norm(dim=1, keepdim=True)          # (N, 1)
        # Avoid division by zero for zero-norm vectors
        safe_norms = norms.clamp(min=1e-12)
        unit = x / safe_norms                         # (N, d), each row on S^{d-1}

        # --- Step 3: fresh random rotation matrix on GPU ---
        # QR decomposition of a Gaussian matrix gives a Haar-distributed
        # orthogonal matrix (QR algorithm with sign correction is exact Haar,
        # but plain QR is sufficient for the noise-injection purpose here).
        gauss = torch.randn(self.d, self.d, device=device, dtype=dtype)
        Q, _ = torch.linalg.qr(gauss)                # (d, d) orthogonal

        # --- Step 4: rotate coordinates ---
        # rotated[i] = unit[i] @ Q^T   →  rotated shape (N, d)
        rotated = unit @ Q.t()

        # --- Step 5: quantize via straight-through estimator ---
        quantized = SQRFunction.apply(rotated, self.centroids.to(dtype),
                                      self.boundaries.to(dtype))

        # --- Step 6: inverse rotate ---
        # dequantized[i] = quantized[i] @ Q  (Q is orthogonal: Q^{-1} = Q^T)
        dequantized = quantized @ Q                   # (N, d)

        # --- Step 7: rescale by original norms ---
        out = dequantized * safe_norms                # broadcast (N, d) * (N, 1)

        # --- Step 8: reshape back to (N, n_coeffs, 3) ---
        return out.reshape(sh_rest.shape)
