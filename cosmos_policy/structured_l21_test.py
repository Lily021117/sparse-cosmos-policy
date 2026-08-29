import pytest
import torch
from torch import nn
from torch.nn import functional as F

from cosmos_policy._src.predict2.networks.minimal_v4_dit import CheckpointMode, MiniTrainDIT, SACConfig
from cosmos_policy.structured_l21 import (
    CROSS_ATTENTION,
    MLP,
    SELF_ATTENTION,
    StructuredL21Regularizer,
    add_structured_l21_penalty,
    apply_structured_l21,
)


def _tiny_dit() -> MiniTrainDIT:
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


class _ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(3, 2, bias=False)
        self.self_attn.k_proj = nn.Linear(3, 2, bias=False)
        self.self_attn.v_proj = nn.Linear(3, 2, bias=False)
        self.cross_attn = nn.Module()
        self.cross_attn.q_proj = nn.Linear(3, 2, bias=False)
        self.mlp = nn.Module()
        self.mlp.layer1 = nn.Linear(3, 4, bias=False)


class _ToyDiT(nn.Module):
    def __init__(self, num_blocks: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_ToyBlock() for _ in range(num_blocks)])


def test_all_group_definitions_have_expected_counts_and_values():
    model = _ToyDiT(num_blocks=2)
    for parameter in model.parameters():
        nn.init.ones_(parameter)

    result = StructuredL21Regularizer(
        enable_self_attention=True,
        enable_cross_attention=True,
        enable_mlp=True,
    )(model)

    assert result.group_norms[SELF_ATTENTION].shape == (2 * 3,)
    assert result.group_norms[CROSS_ATTENTION].shape == (2 * 3,)
    assert result.group_norms[MLP].shape == (2 * 3,)
    torch.testing.assert_close(result.group_norms[SELF_ATTENTION], torch.full((6,), 6.0**0.5))
    torch.testing.assert_close(result.group_norms[CROSS_ATTENTION], torch.full((6,), 2.0**0.5))
    torch.testing.assert_close(result.group_norms[MLP], torch.full((6,), 2.0))
    torch.testing.assert_close(
        result.total,
        torch.tensor(6 * (6.0**0.5 + 2.0**0.5 + 2.0)),
    )


@pytest.mark.parametrize(
    ("enabled_name", "kwargs"),
    [
        (SELF_ATTENTION, {"enable_self_attention": True}),
        (CROSS_ATTENTION, {"enable_cross_attention": True}),
        (MLP, {"enable_mlp": True}),
    ],
)
def test_components_can_be_enabled_independently(enabled_name, kwargs):
    result = StructuredL21Regularizer(**kwargs)(_ToyDiT(num_blocks=1))

    assert set(result.group_norms) == {enabled_name}
    assert set(result.penalties) == {enabled_name}


def test_mlp_l21_has_finite_nonzero_gradient_only_for_enabled_weights():
    model = _ToyDiT(num_blocks=1)
    result = StructuredL21Regularizer(enable_mlp=True)(model)

    result.total.backward()

    mlp_grad = model.blocks[0].mlp.layer1.weight.grad
    assert mlp_grad is not None
    assert torch.isfinite(mlp_grad).all()
    assert torch.count_nonzero(mlp_grad) == mlp_grad.numel()
    assert model.blocks[0].self_attn.q_proj.weight.grad is None
    assert model.blocks[0].cross_attn.q_proj.weight.grad is None


def test_zero_lambda_returns_original_task_loss_without_regularizer_gradient():
    model = _ToyDiT(num_blocks=1)
    task_parameter = nn.Parameter(torch.tensor(2.0))
    task_loss = task_parameter.square()
    result = StructuredL21Regularizer(enable_mlp=True)(model)

    combined = add_structured_l21_penalty(task_loss, result, regularization_lambda=0.0)
    combined.backward()

    assert combined is task_loss
    torch.testing.assert_close(task_parameter.grad, torch.tensor(4.0))
    assert model.blocks[0].mlp.layer1.weight.grad is None


def test_zero_lambda_integration_skips_real_dit_scan_and_preserves_task_loss(monkeypatch):
    model = _tiny_dit()
    regularizer = StructuredL21Regularizer(enable_mlp=True)
    task_parameter = nn.Parameter(torch.tensor(2.0))
    task_loss = task_parameter.square()

    def fail_if_scanned(_regularizer, _model):
        raise AssertionError("lambda=0 must not scan DiT weights")

    monkeypatch.setattr(StructuredL21Regularizer, "__call__", fail_if_scanned)
    total_loss, metrics = apply_structured_l21(
        task_loss,
        model,
        regularizer,
        regularization_lambda=0.0,
    )
    total_loss.backward()

    assert total_loss is task_loss
    assert metrics == {}
    torch.testing.assert_close(task_parameter.grad, torch.tensor(4.0))
    assert model.blocks[0].mlp.layer1.weight.grad is None
    assert model.blocks[0].self_attn.q_proj.weight.grad is None
    assert model.blocks[0].cross_attn.q_proj.weight.grad is None


def test_positive_lambda_integrates_mlp_l21_on_real_dit():
    model = _tiny_dit()
    regularizer = StructuredL21Regularizer(enable_mlp=True)
    task_parameter = nn.Parameter(torch.tensor(2.0))
    task_loss = task_parameter.square()
    regularization_lambda = 0.25

    total_loss, metrics = apply_structured_l21(
        task_loss,
        model,
        regularizer,
        regularization_lambda=regularization_lambda,
    )
    expected = task_loss.detach() + regularization_lambda * metrics["structured_l21_mlp_penalty"]
    torch.testing.assert_close(total_loss.detach(), expected)
    total_loss.backward()

    mlp_grad = model.blocks[0].mlp.layer1.weight.grad
    assert mlp_grad is not None
    assert torch.isfinite(mlp_grad).all()
    assert torch.count_nonzero(mlp_grad) == mlp_grad.numel()
    assert model.blocks[0].self_attn.q_proj.weight.grad is None
    assert model.blocks[0].cross_attn.q_proj.weight.grad is None


def test_zero_lambda_explicit_diagnostics_collects_metrics_without_model_gradient():
    model = _tiny_dit()
    regularizer = StructuredL21Regularizer(enable_mlp=True)
    task_parameter = nn.Parameter(torch.tensor(2.0))
    task_loss = task_parameter.square()

    total_loss, metrics = apply_structured_l21(
        task_loss,
        model,
        regularizer,
        regularization_lambda=0.0,
        collect_diagnostics=True,
    )
    total_loss.backward()

    assert total_loss is task_loss
    assert "structured_l21_mlp_penalty" in metrics
    assert model.blocks[0].mlp.layer1.weight.grad is None


def test_predict2_2b_has_57344_mlp_input_groups():
    from cosmos_policy._src.predict2.configs.text2world.defaults.net import COSMOS_V1_2B_NET_MININET

    assert COSMOS_V1_2B_NET_MININET.num_blocks == 28
    assert COSMOS_V1_2B_NET_MININET.model_channels == 2048
    assert COSMOS_V1_2B_NET_MININET.num_blocks * COSMOS_V1_2B_NET_MININET.model_channels == 57_344


def test_mlp_l21_combines_with_toy_forward_and_backward():
    block = _ToyBlock()
    model = nn.Module()
    model.blocks = nn.ModuleList([block])
    inputs = torch.randn(2, 3, requires_grad=True)
    task_output = F.gelu(block.mlp.layer1(inputs))
    task_loss = task_output.square().mean()
    result = StructuredL21Regularizer(enable_mlp=True)(model)

    total_loss = add_structured_l21_penalty(task_loss, result, regularization_lambda=0.1)
    total_loss.backward()

    assert torch.isfinite(total_loss)
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert block.mlp.layer1.weight.grad is not None
    assert torch.isfinite(block.mlp.layer1.weight.grad).all()
