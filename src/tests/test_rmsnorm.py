import torch

from src.model.rmsnorm import RMSNorm


def test_zero_input() -> None:
    norm = RMSNorm(256)
    x = torch.zeros(2, 16, 256)

    torch.testing.assert_close(norm(x), x)


def test_matches_formula() -> None:
    norm = RMSNorm(2)
    x = torch.tensor([[2.0, 4.0]])
    expected = x / torch.sqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)

    torch.testing.assert_close(norm(x), expected)


def test_backward() -> None:
    norm = RMSNorm(256)
    x = torch.randn(2, 16, 256, requires_grad=True)

    norm(x).square().mean().backward()

    assert x.grad is not None
    assert norm.weight.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(norm.weight.grad).all()


def test_scale_invariance() -> None:
    norm = RMSNorm(256, eps=1e-6)
    x = torch.randn(2, 16, 256)

    torch.testing.assert_close(norm(x), norm(x * 10), atol=1e-4, rtol=1e-4)
