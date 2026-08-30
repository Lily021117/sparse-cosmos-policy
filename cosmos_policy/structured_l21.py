"""Structured input-channel L2,1 regularization for the Cosmos Policy DiT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch
from torch import nn
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard


SELF_ATTENTION = "self_attention"
CROSS_ATTENTION = "cross_attention"
MLP = "mlp"


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

        group_norms: Dict[str, torch.Tensor] = {}
        if self.enable_self_attention:
            group_norms[SELF_ATTENTION] = torch.cat(
                [self._self_attention_group_norms(block) for block in blocks]
            )
        if self.enable_cross_attention:
            group_norms[CROSS_ATTENTION] = torch.cat(
                [self._cross_attention_group_norms(block) for block in blocks]
            )
        if self.enable_mlp:
            group_norms[MLP] = torch.cat([self._mlp_group_norms(block) for block in blocks])

        penalties = {name: norms.sum() for name, norms in group_norms.items()}
        return StructuredL21Result(penalties=penalties, group_norms=group_norms)


def add_structured_l21_penalty(
    task_loss: torch.Tensor,
    result: StructuredL21Result,
    regularization_lambda: float,
) -> torch.Tensor:
    """Add L2,1 without changing the task-loss tensor when lambda is zero."""
    if regularization_lambda < 0:
        raise ValueError("structured L21 lambda must be non-negative")
    if regularization_lambda == 0:
        return task_loss
    return task_loss + regularization_lambda * result.total


def apply_structured_l21(
    task_loss: torch.Tensor,
    model: nn.Module,
    regularizer: StructuredL21Regularizer,
    regularization_lambda: float,
    *,
    collect_diagnostics: bool = False,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Integrate L2,1 with a task loss, avoiding baseline weight scans.

    A zero lambda and disabled diagnostics return the original loss object
    without traversing model weights. Diagnostic-only collection is performed
    under ``no_grad`` so it cannot affect training gradients.
    """
    if regularization_lambda < 0:
        raise ValueError("structured L21 lambda must be non-negative")
    if not regularizer.enabled or (regularization_lambda == 0 and not collect_diagnostics):
        return task_loss, {}

    if regularization_lambda > 0:
        result = regularizer(model)
        total_loss = add_structured_l21_penalty(task_loss, result, regularization_lambda)
    else:
        with torch.no_grad():
            result = regularizer(model)
        total_loss = task_loss

    return total_loss, result.detached_metrics()
