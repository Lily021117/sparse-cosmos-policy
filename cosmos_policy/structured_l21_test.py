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


class _CheckpointLikeWrapper(nn.Module):
    """Expose the same block attributes as SAC's checkpoint wrapper."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self._checkpoint_wrapped_module = module

    @property
    def self_attn(self) -> nn.Module:
        return self._checkpoint_wrapped_module.self_attn

    @property
    def cross_attn(self) -> nn.Module:
        return self._checkpoint_wrapped_module.cross_attn

    @property
    def mlp(self) -> nn.Module:
        return self._checkpoint_wrapped_module.mlp


def test_checkpoint_wrappers_do_not_duplicate_blocks_groups_or_penalty():
    unwrapped = _ToyDiT(num_blocks=2)
    for parameter in unwrapped.parameters():
        nn.init.ones_(parameter)

    wrapped = nn.Module()
    wrapped.blocks = nn.ModuleList([_CheckpointLikeWrapper(block) for block in unwrapped.blocks])
    # Also register a repeated reference to exercise module aliasing.
    wrapped.repeated_block_reference = unwrapped.blocks[0]

    regularizer = StructuredL21Regularizer(enable_mlp=True)
    unwrapped_result = regularizer(unwrapped)
    wrapped_result = regularizer(wrapped)

    assert len(list(regularizer._dit_blocks(wrapped))) == 2
    assert wrapped_result.group_norms[MLP].shape == (2 * 3,)
    torch.testing.assert_close(wrapped_result.group_norms[MLP], unwrapped_result.group_norms[MLP])
    torch.testing.assert_close(wrapped_result.penalties[MLP], unwrapped_result.penalties[MLP])


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


def test_block_subset_regularizes_only_selected_dit_blocks():
    model = _ToyDiT(num_blocks=4)
    for parameter in model.parameters():
        nn.init.ones_(parameter)

    regularizer = StructuredL21Regularizer(
        enable_self_attention=True,
        enable_cross_attention=True,
        enable_mlp=True,
    )
    selected = regularizer(model, block_indices=(2, 3))
    block_two = regularizer(model, block_indices=(2,))
    block_three = regularizer(model, block_indices=(3,))

    for component in (SELF_ATTENTION, CROSS_ATTENTION, MLP):
        # Three input-channel groups per selected toy block.
        assert selected.group_norms[component].numel() == 2 * 3
        torch.testing.assert_close(
            selected.penalties[component],
            block_two.penalties[component] + block_three.penalties[component],
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


@pytest.mark.parametrize(
    ("regularizer_kwargs", "enabled_components"),
    [
        ({"enable_self_attention": True}, {SELF_ATTENTION}),
        ({"enable_cross_attention": True}, {CROSS_ATTENTION}),
        ({"enable_mlp": True}, {MLP}),
        (
            {
                "enable_self_attention": True,
                "enable_cross_attention": True,
                "enable_mlp": True,
            },
            {SELF_ATTENTION, CROSS_ATTENTION, MLP},
        ),
    ],
)
def test_positive_lambda_integrates_enabled_components_on_real_dit(
    regularizer_kwargs,
    enabled_components,
):
    model = _tiny_dit()
    regularizer = StructuredL21Regularizer(**regularizer_kwargs)
    task_parameter = nn.Parameter(torch.tensor(2.0))
    task_loss = task_parameter.square()
    regularization_lambda = 0.25

    result = regularizer(model)
    assert set(result.group_norms) == enabled_components
    assert set(result.penalties) == enabled_components
    for component in enabled_components:
        assert result.group_norms[component].shape == (model.model_channels,)
        assert torch.isfinite(result.group_norms[component]).all()
        assert torch.isfinite(result.penalties[component])

    total_loss, metrics = apply_structured_l21(
        task_loss,
        model,
        regularizer,
        regularization_lambda=regularization_lambda,
    )
    expected_penalty = sum(metrics[f"structured_l21_{component}_penalty"] for component in enabled_components)
    expected = task_loss.detach() + regularization_lambda * expected_penalty
    torch.testing.assert_close(total_loss.detach(), expected)
    total_loss.backward()

    block = model.blocks[0]
    component_weights = {
        SELF_ATTENTION: (
            block.self_attn.q_proj.weight,
            block.self_attn.k_proj.weight,
            block.self_attn.v_proj.weight,
        ),
        CROSS_ATTENTION: (block.cross_attn.q_proj.weight,),
        MLP: (block.mlp.layer1.weight,),
    }
    for component, weights in component_weights.items():
        for weight in weights:
            if component in enabled_components:
                assert weight.grad is not None
                assert torch.isfinite(weight.grad).all()
                assert torch.count_nonzero(weight.grad) > 0
            else:
                assert weight.grad is None

    # These weights never belong to any of the configured input-channel groups.
    assert block.self_attn.output_proj.weight.grad is None
    assert block.cross_attn.k_proj.weight.grad is None
    assert block.cross_attn.v_proj.weight.grad is None
    assert block.cross_attn.output_proj.weight.grad is None
    assert block.mlp.layer2.weight.grad is None


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


@pytest.mark.parametrize(
    ("active_component", "component_lambdas", "expected_weight_names"),
    [
        (
            SELF_ATTENTION,
            {SELF_ATTENTION: 0.25, CROSS_ATTENTION: 0.0, MLP: 0.0},
            ("q_proj", "k_proj", "v_proj"),
        ),
        (
            CROSS_ATTENTION,
            {SELF_ATTENTION: 0.0, CROSS_ATTENTION: 0.25, MLP: 0.0},
            ("cross_q",),
        ),
        (
            MLP,
            {SELF_ATTENTION: 0.0, CROSS_ATTENTION: 0.0, MLP: 0.25},
            ("mlp",),
        ),
    ],
)
def test_component_lambdas_independently_select_regularizer_gradients(
    active_component,
    component_lambdas,
    expected_weight_names,
):
    model = _ToyDiT(num_blocks=1)
    block = model.blocks[0]
    regularizer = StructuredL21Regularizer(
        enable_self_attention=True,
        enable_cross_attention=True,
        enable_mlp=True,
    )
    task_parameter = nn.Parameter(torch.tensor(2.0))
    task_loss = task_parameter.square()

    total_loss, metrics = apply_structured_l21(
        task_loss,
        model,
        regularizer,
        component_lambdas=component_lambdas,
    )
    expected = task_loss.detach() + component_lambdas[active_component] * metrics[
        f"structured_l21_{active_component}_penalty"
    ]
    torch.testing.assert_close(total_loss.detach(), expected)
    total_loss.backward()

    weights = {
        "q_proj": block.self_attn.q_proj.weight,
        "k_proj": block.self_attn.k_proj.weight,
        "v_proj": block.self_attn.v_proj.weight,
        "cross_q": block.cross_attn.q_proj.weight,
        "mlp": block.mlp.layer1.weight,
    }
    for name, weight in weights.items():
        if name in expected_weight_names:
            assert weight.grad is not None
            assert torch.isfinite(weight.grad).all()
            assert torch.count_nonzero(weight.grad) > 0
        else:
            assert weight.grad is None


def test_component_lambdas_weighted_sum_is_exact_and_uses_one_regularizer_result():
    model = _ToyDiT(num_blocks=1)
    regularizer = StructuredL21Regularizer(
        enable_self_attention=True,
        enable_cross_attention=True,
        enable_mlp=True,
    )
    component_lambdas = {
        SELF_ATTENTION: 0.125,
        CROSS_ATTENTION: 0.25,
        MLP: 0.5,
    }
    task_loss = nn.Parameter(torch.tensor(2.0)).square()

    result = regularizer(model)
    total_loss = add_structured_l21_penalty(
        task_loss,
        result,
        component_lambdas=component_lambdas,
    )
    expected = task_loss.detach() + sum(
        component_lambdas[name] * result.penalties[name].detach() for name in component_lambdas
    )
    torch.testing.assert_close(total_loss.detach(), expected)
    total_loss.backward()

    block = model.blocks[0]
    for weight in (
        block.self_attn.q_proj.weight,
        block.self_attn.k_proj.weight,
        block.self_attn.v_proj.weight,
        block.cross_attn.q_proj.weight,
        block.mlp.layer1.weight,
    ):
        assert weight.grad is not None
        assert torch.isfinite(weight.grad).all()


def test_disabled_component_lambda_is_inert_and_does_not_scan_or_create_gradient(monkeypatch):
    model = _tiny_dit()
    regularizer = StructuredL21Regularizer(enable_mlp=True)
    task_parameter = nn.Parameter(torch.tensor(2.0))
    task_loss = task_parameter.square()

    def fail_if_scanned(_regularizer, _model):
        raise AssertionError("disabled component lambda must not scan DiT weights")

    monkeypatch.setattr(StructuredL21Regularizer, "__call__", fail_if_scanned)
    total_loss, metrics = apply_structured_l21(
        task_loss,
        model,
        regularizer,
        component_lambdas={SELF_ATTENTION: 1.0, CROSS_ATTENTION: 0.0, MLP: 0.0},
    )
    total_loss.backward()

    assert total_loss is task_loss
    assert metrics == {}
    assert model.blocks[0].mlp.layer1.weight.grad is None


def test_all_zero_component_lambdas_use_fast_path(monkeypatch):
    model = _tiny_dit()
    regularizer = StructuredL21Regularizer(
        enable_self_attention=True,
        enable_cross_attention=True,
        enable_mlp=True,
    )
    task_parameter = nn.Parameter(torch.tensor(2.0))
    task_loss = task_parameter.square()

    def fail_if_scanned(_regularizer, _model):
        raise AssertionError("all-zero component lambdas must not scan DiT weights")

    monkeypatch.setattr(StructuredL21Regularizer, "__call__", fail_if_scanned)
    total_loss, metrics = apply_structured_l21(
        task_loss,
        model,
        regularizer,
        component_lambdas={name: 0.0 for name in (SELF_ATTENTION, CROSS_ATTENTION, MLP)},
    )
    total_loss.backward()

    assert total_loss is task_loss
    assert metrics == {}
    torch.testing.assert_close(task_parameter.grad, torch.tensor(4.0))


def test_negative_component_lambda_is_rejected():
    task_loss = torch.tensor(1.0, requires_grad=True)
    with pytest.raises(ValueError, match="non-negative"):
        apply_structured_l21(
            task_loss,
            _ToyDiT(num_blocks=1),
            StructuredL21Regularizer(enable_mlp=True),
            component_lambdas={MLP: -0.1},
        )


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
