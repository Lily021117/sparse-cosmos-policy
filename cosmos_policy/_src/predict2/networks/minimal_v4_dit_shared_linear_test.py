import torch

from cosmos_policy._src.predict2.networks.minimal_v4_dit import SharedTokenLinear


def test_shared_token_linear_identity_shape_dtype_and_device():
    linear = SharedTokenLinear(8)
    tokens = torch.randn(2, 3, 4, 5, 8)

    output = linear(tokens)

    assert output.shape == tokens.shape
    assert output.dtype == tokens.dtype
    assert output.device == tokens.device
    torch.testing.assert_close(output, tokens, rtol=0.0, atol=0.0)


def test_shared_token_linear_identity_after_dtype_transfer():
    linear = SharedTokenLinear(8).to(dtype=torch.float64)
    tokens = torch.randn(2, 7, 8, dtype=torch.float64)

    output = linear(tokens)

    assert output.dtype == torch.float64
    assert output.device == tokens.device
    torch.testing.assert_close(output, tokens, rtol=0.0, atol=0.0)


def test_shared_token_linear_receives_gradient():
    linear = SharedTokenLinear(8)
    tokens = torch.randn(2, 3, 8)

    linear(tokens).square().mean().backward()

    assert linear.weight.grad is not None
    assert torch.isfinite(linear.weight.grad).all()
    assert linear.weight.grad.abs().max() > 0
