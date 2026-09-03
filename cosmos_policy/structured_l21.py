"""Structured input-channel L2,1 regularization for the Cosmos Policy DiT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import torch
from torch import nn
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard


SELF_ATTENTION = "self_attention"
CROSS_ATTENTION = "cross_attention"
MLP = "mlp"
COMPONENTS = (SELF_ATTENTION, CROSS_ATTENTION, MLP)


@dataclass(frozen=True)
class StructuredL21Result:
    """Differentiable penalties and their per-group norms."""

    penalties: Dict[str, torch.Tensor]
    group_norms: Dict[str, torch.Tensor]

    @property
    def total(self) -> torch.Tensor:
        penalties = list(self.penalties.values())
        if not penalties:
            raise RuntimeError("No structured L21 component is enabled")
        return torch.stack(penalties).sum()

    def detached_metrics(self) -> Dict[str, torch.Tensor]:
        metrics: Dict[str, torch.Tensor] = {}
        for name, group_norms in self.group_norms.items():
            detached = group_norms.detach().float()
            metrics[f"structured_l21_{name}_penalty"] = self.penalties[name].detach().float()
            metrics[f"structured_l21_{name}_group_norm_mean"] = detached.mean()
            metrics[f"structured_l21_{name}_group_norm_std"] = detached.std(unbiased=False)
            metrics[f"structured_l21_{name}_group_norm_min"] = detached.min()
            metrics[f"structured_l21_{name}_group_norm_max"] = detached.max()
        return metrics


class StructuredL21Regularizer:
    """Compute independently switchable DiT input-channel L2,1 penalties.

    Self-attention groups jointly contain the same input column from Q, K,
    and V. Cross-attention groups contain query-projection input columns only;
    K/V consume a separate text-condition feature space. MLP groups are input
    columns of the first feed-forward projection.
    """

    def __init__(
        self,
        *,
        enable_self_attention: bool = False,
        enable_cross_attention: bool = False,
        enable_mlp: bool = False,
    ) -> None:
        self.enable_self_attention = enable_self_attention
        self.enable_cross_attention = enable_cross_attention
        self.enable_mlp = enable_mlp

    @property
    def enabled(self) -> bool:
        return self.enable_self_attention or self.enable_cross_attention or self.enable_mlp

    @staticmethod
    def _dit_blocks(model: nn.Module) -> Iterable[nn.Module]:
        seen_block_components: set[tuple[int, int, int]] = set()
        for module in model.modules():
            if all(hasattr(module, name) for name in ("self_attn", "cross_attn", "mlp")):
                # Activation-checkpoint wrappers proxy these attributes from
                # their wrapped block, so model.modules() visits two distinct
                # module objects that describe the same logical DiT block.
                # Deduplicate by the actual submodules used by the penalty.
                block_components = (
                    id(module.self_attn),
                    id(module.cross_attn),
                    id(module.mlp),
                )
                if block_components in seen_block_components:
                    continue
                seen_block_components.add(block_components)
                yield module

    @staticmethod
    def _column_squared_norm(weight: torch.Tensor) -> torch.Tensor:
        if not isinstance(weight, DTensor):
            return weight.float().square().sum(dim=0)

        local_squared_norm = weight.to_local().float().square().sum(dim=0)
        partial_placements = []
        has_sharded_output = False
        for placement in weight.placements:
            if isinstance(placement, Shard):
                if placement.dim != 0:
                    raise NotImplementedError(
                        "Structured input-channel L21 requires FSDP weights to be sharded along output dimension 0"
                    )
                partial_placements.append(Partial())
                has_sharded_output = True
            elif isinstance(placement, Replicate):
                partial_placements.append(placement)
            else:
                raise NotImplementedError(f"Unsupported DTensor placement for structured L21: {placement}")

        if not has_sharded_output:
            return local_squared_norm

        partial_squared_norm = DTensor.from_local(
            local_squared_norm,
            device_mesh=weight.device_mesh,
            placements=partial_placements,
            run_check=False,
        )
        return partial_squared_norm.redistribute(
            placements=[Replicate() for _ in partial_placements]
        ).to_local()

    @staticmethod
    def _safe_sqrt(squared_norm: torch.Tensor) -> torch.Tensor:
        # Preserve the conventional zero subgradient of an L2 norm.
        positive_root = torch.sqrt(squared_norm.clamp_min(torch.finfo(squared_norm.dtype).tiny))
        return torch.where(squared_norm > 0, positive_root, torch.zeros_like(squared_norm))

    @classmethod
    def _column_norm(cls, weight: torch.Tensor) -> torch.Tensor:
        return cls._safe_sqrt(cls._column_squared_norm(weight))

    @classmethod
    def _self_attention_group_norms(cls, block: nn.Module) -> torch.Tensor:
        squared_norm = sum(
            cls._column_squared_norm(weight)
            for weight in (
                block.self_attn.q_proj.weight,
                block.self_attn.k_proj.weight,
                block.self_attn.v_proj.weight,
            )
        )
        return cls._safe_sqrt(squared_norm)

    @classmethod
    def _cross_attention_group_norms(cls, block: nn.Module) -> torch.Tensor:
        return cls._column_norm(block.cross_attn.q_proj.weight)

    @classmethod
    def _mlp_group_norms(cls, block: nn.Module) -> torch.Tensor:
        return cls._column_norm(block.mlp.layer1.weight)

    def __call__(self, model: nn.Module) -> StructuredL21Result:
        if not self.enabled:
            raise RuntimeError("At least one structured L21 component must be enabled")

        blocks = list(self._dit_blocks(model))
        if not blocks:
            raise ValueError("Could not find DiT blocks with self_attn, cross_attn, and mlp modules")

        # Traverse the logical DiT blocks once.  Besides avoiding repeated
        # wrapper traversal under SAC/FSDP, this lets callers independently
        # weight all three components without doing three model scans.
        per_component_norms: Dict[str, list[torch.Tensor]] = {}
        if self.enable_self_attention:
            per_component_norms[SELF_ATTENTION] = []
        if self.enable_cross_attention:
            per_component_norms[CROSS_ATTENTION] = []
        if self.enable_mlp:
            per_component_norms[MLP] = []

        for block in blocks:
            if self.enable_self_attention:
                per_component_norms[SELF_ATTENTION].append(self._self_attention_group_norms(block))
            if self.enable_cross_attention:
                per_component_norms[CROSS_ATTENTION].append(self._cross_attention_group_norms(block))
            if self.enable_mlp:
                per_component_norms[MLP].append(self._mlp_group_norms(block))

        group_norms = {name: torch.cat(norms) for name, norms in per_component_norms.items()}

        penalties = {name: norms.sum() for name, norms in group_norms.items()}
        return StructuredL21Result(penalties=penalties, group_norms=group_norms)


def add_structured_l21_penalty(
    task_loss: torch.Tensor,
    result: StructuredL21Result,
    regularization_lambda: float | None = None,
    *,
    component_lambdas: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Add independently weighted L2,1 components to ``task_loss``.

    ``regularization_lambda`` remains as a backwards-compatible uniform
    coefficient.  New callers should pass ``component_lambdas``.
    """
    lambdas = _resolve_component_lambdas(
        result.penalties,
        regularization_lambda=regularization_lambda,
        component_lambdas=component_lambdas,
    )
    # Exclude zero-coefficient terms entirely, rather than constructing
    # ``0 * penalty``.  The latter leaves a zero-valued autograd path and
    # materializes misleading zero gradients on otherwise disabled modules.
    active_penalties = (
        (lambdas[name] * penalty for name, penalty in result.penalties.items() if lambdas[name] != 0)
    )
    weighted_penalty = sum(
        active_penalties,
        start=torch.zeros_like(task_loss),
    )
    if all(value == 0 for value in lambdas.values()):
        return task_loss
    return task_loss + weighted_penalty


def _resolve_component_lambdas(
    enabled_penalties: Mapping[str, torch.Tensor],
    *,
    regularization_lambda: float | None,
    component_lambdas: Mapping[str, float] | None,
) -> Dict[str, float]:
    """Validate coefficients and retain only components present in ``result``."""
    if regularization_lambda is not None and component_lambdas is not None:
        raise ValueError("Specify either regularization_lambda or component_lambdas, not both")

    if component_lambdas is None:
        uniform_lambda = 0.0 if regularization_lambda is None else regularization_lambda
        component_lambdas = {name: uniform_lambda for name in COMPONENTS}

    unknown = set(component_lambdas) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown structured L21 component(s): {sorted(unknown)}")

    values = {name: float(component_lambdas.get(name, 0.0)) for name in COMPONENTS}
    if any(value < 0 for value in values.values()):
        raise ValueError("structured L21 lambdas must be non-negative")
    # A nonzero coefficient for a disabled component is intentionally inert:
    # it cannot create a penalty or a gradient without a corresponding group.
    return {name: values[name] for name in enabled_penalties}


def apply_structured_l21(
    task_loss: torch.Tensor,
    model: nn.Module,
    regularizer: StructuredL21Regularizer,
    regularization_lambda: float | None = None,
    *,
    component_lambdas: Mapping[str, float] | None = None,
    collect_diagnostics: bool = False,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Integrate L2,1 with a task loss, avoiding baseline weight scans.

    A zero lambda and disabled diagnostics return the original loss object
    without traversing model weights. Diagnostic-only collection is performed
    under ``no_grad`` so it cannot affect training gradients.
    """
    # Validate all supplied coefficients before the fast path, including a
    # coefficient for a disabled component.
    _resolve_component_lambdas(
        {},
        regularization_lambda=regularization_lambda,
        component_lambdas=component_lambdas,
    )
    configured_lambdas = component_lambdas
    if configured_lambdas is None:
        configured_lambdas = {name: 0.0 if regularization_lambda is None else regularization_lambda for name in COMPONENTS}
    enabled_lambdas = {
        SELF_ATTENTION: configured_lambdas.get(SELF_ATTENTION, 0.0) if regularizer.enable_self_attention else 0.0,
        CROSS_ATTENTION: configured_lambdas.get(CROSS_ATTENTION, 0.0) if regularizer.enable_cross_attention else 0.0,
        MLP: configured_lambdas.get(MLP, 0.0) if regularizer.enable_mlp else 0.0,
    }
    if not regularizer.enabled or (all(value == 0 for value in enabled_lambdas.values()) and not collect_diagnostics):
        return task_loss, {}

    if any(value > 0 for value in enabled_lambdas.values()):
        result = regularizer(model)
        total_loss = add_structured_l21_penalty(task_loss, result, component_lambdas=enabled_lambdas)
    else:
        with torch.no_grad():
            result = regularizer(model)
        total_loss = task_loss

    return total_loss, result.detached_metrics()
