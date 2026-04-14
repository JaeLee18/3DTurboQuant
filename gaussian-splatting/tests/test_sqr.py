"""Tests for SQR (Stochastic Quantization Regularization) module.

All tests run on CUDA.
"""

import pytest
import torch


@pytest.fixture
def sqr_module():
    """Return an SQRModule instance on CUDA in training mode."""
    from sqr.sqr_module import SQRModule
    module = SQRModule(d=45, b=3, p=0.5).cuda()
    module.train()
    return module


def test_sqr_output_shape(sqr_module):
    """Output shape must equal input shape."""
    N = 128
    sh_rest = torch.randn(N, 15, 3, device="cuda")
    out = sqr_module(sh_rest)
    assert out.shape == sh_rest.shape, (
        f"Expected shape {sh_rest.shape}, got {out.shape}"
    )


def test_sqr_adds_noise(sqr_module):
    """In train mode, output must differ from input (quantization noise injected).

    We run many forward passes to make the stochastic skip (prob p=0.5) unlikely
    to always trigger. With p=0.5 and 20 trials, probability all skip = 2^-20 < 1e-6.
    """
    N = 256
    sh_rest = torch.randn(N, 15, 3, device="cuda")
    any_different = False
    for _ in range(20):
        out = sqr_module(sh_rest)
        if not torch.allclose(out, sh_rest):
            any_different = True
            break
    assert any_different, (
        "After 20 train-mode forward passes, output always equalled input — "
        "quantization noise was never injected."
    )


def test_sqr_eval_passthrough(sqr_module):
    """In eval mode, output must equal input exactly (identity pass-through)."""
    sqr_module.eval()
    N = 64
    sh_rest = torch.randn(N, 15, 3, device="cuda")
    out = sqr_module(sh_rest)
    assert torch.equal(out, sh_rest), (
        "Eval mode must return input unchanged, but output differed."
    )


def test_sqr_gradient_flows():
    """Gradients must flow through SQRFunction via the straight-through estimator.

    We force quantization on every call by setting p=0.0 (never skip), then
    verify that loss.backward() produces a non-zero gradient on the input.
    """
    from sqr.sqr_module import SQRModule
    module = SQRModule(d=45, b=3, p=0.0).cuda()
    module.train()

    N = 32
    x = torch.randn(N, 15, 3, device="cuda", requires_grad=True)
    out = module(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "x.grad is None — gradient did not flow back."
    assert x.grad.abs().sum().item() > 0, (
        "x.grad is all zeros — straight-through estimator is broken."
    )
