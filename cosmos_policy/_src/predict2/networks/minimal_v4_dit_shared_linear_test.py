import torch

from cosmos_policy._src.predict2.networks.minimal_v4_dit import (
    CheckpointMode,
    MiniTrainDIT,
    SACConfig,
    SharedTokenLinear,
)


def _tiny_dit() -> MiniTrainDIT:
    """Construct the smallest real MiniTrainDIT path needed before block 0."""
    return MiniTrainDIT(
        max_img_h=4,
        max_img_w=4,
        max_frames=1,
        in_channels=2,
        out_channels=2,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=False,
        model_channels=8,
        num_blocks=1,
        num_heads=2,
        mlp_ratio=1.0,
        atten_backend="torch",
        crossattn_emb_channels=8,
        pos_emb_cls="rope3d",
        pos_emb_learnable=False,
        use_adaln_lora=False,
        sac_config=SACConfig(mode=CheckpointMode.NONE),
    )


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


def test_shared_token_linear_is_identity_on_real_dit_pre_block_path():
    """The real forward preparation path is unchanged when the linear is bypassed."""
    torch.manual_seed(0)
    model = _tiny_dit().eval()
    latent = torch.randn(1, 2, 1, 4, 4)

    with torch.no_grad():
        with_linear, rope_with_linear, extra_with_linear = model.prepare_embedded_sequence(latent)

        original_linear = model.shared_token_linear
        model.shared_token_linear = torch.nn.Identity()
        try:
            bypassed, rope_bypassed, extra_bypassed = model.prepare_embedded_sequence(latent)
        finally:
            model.shared_token_linear = original_linear

    assert with_linear.shape == (1, 1, 2, 2, model.model_channels)
    assert with_linear.dtype == latent.dtype
    assert with_linear.device == latent.device
    assert rope_with_linear is not None
    assert rope_bypassed is not None
    assert extra_with_linear is None
    assert extra_bypassed is None
    torch.testing.assert_close(with_linear, bypassed, rtol=0.0, atol=0.0)
    torch.testing.assert_close(rope_with_linear, rope_bypassed, rtol=0.0, atol=0.0)


def test_old_state_dict_loads_non_strictly_and_keeps_identity_linear():
    """An old checkpoint lacks only the new linear and leaves its identity init intact."""
    torch.manual_seed(0)
    source = _tiny_dit()
    old_state_dict = {
        key: value.clone()
        for key, value in source.state_dict().items()
        if key != "shared_token_linear.weight"
    }

    restored = _tiny_dit()
    incompatible = restored.load_state_dict(old_state_dict, strict=False)

    assert incompatible.missing_keys == ["shared_token_linear.weight"]
    assert incompatible.unexpected_keys == []
    torch.testing.assert_close(
        restored.shared_token_linear.weight,
        torch.eye(restored.model_channels, dtype=restored.shared_token_linear.weight.dtype),
        rtol=0.0,
        atol=0.0,
    )
