"""Phase 3H: read-only SA/CA 32-micro structured-L21 gradient calibration.

Run from any location after setting ``POLICY_CHECKPOINT`` and
``BASE_DATASETS_DIR``, or pass ``--checkpoint`` explicitly.  The script does
not create an optimizer or update model parameters.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.distributed.tensor import DTensor
from torch.utils.data import DataLoader


SEED = 1234
MICRO_BATCHES = 32


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=os.environ.get("POLICY_CHECKPOINT"),
        required=os.environ.get("POLICY_CHECKPOINT") is None,
        help="Path to Cosmos-Policy-LIBERO-Predict2-2B.pt. Defaults to $POLICY_CHECKPOINT.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Cosmos Policy repository root; defaults relative to this script.",
    )
    return parser.parse_args()


args = parse_args()
repo_root = args.repo_root.resolve()
policy_checkpoint = Path(args.checkpoint).expanduser().resolve()
if not repo_root.is_dir():
    raise RuntimeError(f"repository root does not exist: {repo_root}")
if not policy_checkpoint.is_file():
    raise RuntimeError(f"Policy checkpoint does not exist: {policy_checkpoint}")
if "BASE_DATASETS_DIR" not in os.environ:
    raise RuntimeError("BASE_DATASETS_DIR must point to the LIBERO dataset root")
os.chdir(repo_root)
sys.path.insert(0, str(repo_root))

from cosmos_policy._src.imaginaire.config import load_config
from cosmos_policy._src.imaginaire.lazy_config import instantiate
from cosmos_policy._src.imaginaire.utils import distributed, misc
from cosmos_policy._src.imaginaire.utils.context_managers import data_loader_init, distributed_init, model_init
from cosmos_policy._src.predict2.utils.model_loader import create_model_from_consolidated_checkpoint_with_fsdp
from cosmos_policy.structured_l21 import CROSS_ATTENTION, SELF_ATTENTION


def local(value: torch.Tensor) -> torch.Tensor:
    return value.to_local() if isinstance(value, DTensor) else value


def unwrap(block):
    return getattr(block, "_checkpoint_wrapped_module", block)


def joined_norm(values: list[torch.Tensor]) -> float:
    return float(torch.sqrt(sum(local(value).detach().float().square().sum() for value in values)))


def target_weights(block, component: str) -> list[torch.Tensor]:
    block = unwrap(block)
    if component == SELF_ATTENTION:
        return [block.self_attn.q_proj.weight, block.self_attn.k_proj.weight, block.self_attn.v_proj.weight]
    if component == CROSS_ATTENTION:
        return [block.cross_attn.q_proj.weight]
    raise ValueError(component)


def gradient_values(block, component: str) -> list[torch.Tensor]:
    values = []
    for weight in target_weights(block, component):
        if weight.grad is None:
            raise RuntimeError(f"missing {component} gradient")
        values.append(local(weight.grad).detach().float().clone())
    return values


def component_statistics(norms: torch.Tensor) -> dict[str, float]:
    norms = norms.detach().float()
    return {
        "mean": float(norms.mean()),
        "std": float(norms.std(unbiased=False)),
        "cv": float(norms.std(unbiased=False) / norms.mean()),
        "p1": float(torch.quantile(norms, 0.01)),
        "p5": float(torch.quantile(norms, 0.05)),
        "p50": float(torch.quantile(norms, 0.50)),
    }


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
with distributed_init():
    distributed.init()

config = load_config(
    str(repo_root / "cosmos_policy/config/config.py"),
    ["--", "experiment=cosmos_predict2_2b_480p_libero"],
    enable_one_logger=True,
)
config.checkpoint.load_path = str(policy_checkpoint)
config.model.config.structured_l21_lambda = 0.0
config.model.config.structured_l21_diagnostic_metrics = False
config.model.config.enable_sa_input_channel_l21 = True
config.model.config.enable_ca_query_input_channel_l21 = True
config.model.config.enable_mlp_input_channel_l21 = False
config.optimizer.lr = 1.6667e-6
config.validate()

with model_init():
    model = create_model_from_consolidated_checkpoint_with_fsdp(config)
model = model.to("cuda", memory_format=config.trainer.memory_format)
model.on_train_start(config.trainer.memory_format)
model.train()
if model.net.shared_token_linear.weight.requires_grad:
    raise RuntimeError("SharedTokenLinear must be frozen for Phase3H")
if len(model.net.blocks) != 28:
    raise RuntimeError(f"expected 28 DiT blocks, got {len(model.net.blocks)}")

with data_loader_init():
    dataset = instantiate(config.dataloader_train.dataset)
generator = torch.Generator().manual_seed(SEED)
loader = DataLoader(dataset, batch_size=1, shuffle=True, generator=generator, num_workers=0, pin_memory=False, drop_last=True)
iterator = iter(loader)
calibration_batch = misc.to(next(iterator), device="cuda")
scaler = torch.amp.GradScaler("cuda", **config.trainer.grad_scaler_args)
scale = float(scaler.get_scale())

results = {}
for component in (SELF_ATTENTION, CROSS_ATTENTION):
    model.zero_grad(set_to_none=True)
    micro_losses = []
    for micro_index in range(MICRO_BATCHES):
        batch = calibration_batch if micro_index == 0 else misc.to(next(iterator), device="cuda")
        with torch.amp.autocast("cuda", dtype=model.precision):
            output, task_loss = model.training_step(batch, iteration=0)
        if any(key.startswith("structured_l21_") for key in output):
            raise RuntimeError("lambda=0 task path unexpectedly evaluated L21")
        if not torch.isfinite(task_loss):
            raise RuntimeError(f"non-finite task loss at micro {micro_index}")
        micro_losses.append(float(local(task_loss).detach().float()))
        scaler.scale(task_loss / MICRO_BATCHES).backward()
        model.on_after_backward()
    task_grads = [gradient_values(block, component) for block in model.net.blocks]
    task_grads = [[grad / scale for grad in grads] for grads in task_grads]

    model.zero_grad(set_to_none=True)
    regularizer = model.structured_l21_regularizer(model.net)
    penalty = regularizer.penalties[component]
    if not torch.isfinite(penalty):
        raise RuntimeError(f"non-finite {component} penalty")
    penalty.backward()
    regularizer_grads = [gradient_values(block, component) for block in model.net.blocks]

    rows = []
    for index, (task_grad, regularizer_grad) in enumerate(zip(task_grads, regularizer_grads)):
        task_norm = joined_norm(task_grad)
        regularizer_norm = joined_norm(regularizer_grad)
        if task_norm == 0:
            raise RuntimeError(f"zero task gradient in block {index} ({component})")
        rows.append({"block": index, "g_task_l2": task_norm, "g_R_l2": regularizer_norm, "ratio_per_unit_lambda": regularizer_norm / task_norm})
    ratios = sorted(row["ratio_per_unit_lambda"] for row in rows)
    median = float(torch.tensor(ratios).median())
    results[component] = {
        "R": float(local(penalty).detach().float()),
        "group_count": int(regularizer.group_norms[component].numel()),
        "group_norm": component_statistics(regularizer.group_norms[component]),
        "rows": rows,
        "ratio_per_unit_lambda": {"min": min(ratios), "median": median, "max": max(ratios)},
        "lambda_0.5pct": 0.005 / median,
        "lambda_1pct": 0.01 / median,
        "lambda_2pct": 0.02 / median,
        "micro_task_loss_mean": float(sum(micro_losses) / len(micro_losses)),
    }

payload = {
    "checkpoint": str(policy_checkpoint),
    "micro_batches": MICRO_BATCHES,
    "loss_divisor": MICRO_BATCHES,
    "grad_scaler_scale": scale,
    "components": results,
}
print("PHASE3H_RESULT " + json.dumps(payload, sort_keys=True), flush=True)
torch.cuda.synchronize()
