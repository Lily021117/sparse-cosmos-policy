# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Extended Trainer for Cosmos Policy with epoch tracking.

This trainer extends the base ImaginaireTrainer to add:
- Epoch tracking and sampler epoch setting for proper distributed sampling
"""

import signal

import torch
import torch.utils.data
import wandb

from cosmos_policy._src.imaginaire.model import ImaginaireModel
from cosmos_policy._src.imaginaire.trainer import ImaginaireTrainer
from cosmos_policy._src.imaginaire.utils import distributed, log, misc
from cosmos_policy._src.imaginaire.utils.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling


class CosmosPolicyTrainer(ImaginaireTrainer):
    """
    Extended Trainer for Cosmos Policy.

    Adds special handling for:
    - Epoch tracking to properly set dataloader sampler epochs (needed for distributed training)
    - Simplified initial validation check (removes run_validation_on_start requirement)
    """

    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def bilevel_phase_sequence(inner_steps: int, outer_steps: int) -> list[str]:
        if inner_steps < 1 or outer_steps < 1:
            raise ValueError("bilevel step counts must be positive")
        return ["inner"] * inner_steps + ["outer"] * outer_steps

    @staticmethod
    def advance_bilevel_state(state: dict, inner_steps: int = 5, outer_steps: int = 1) -> dict:
        """Advance counters after exactly one completed optimizer step."""
        sequence = CosmosPolicyTrainer.bilevel_phase_sequence(inner_steps, outer_steps)
        phase = sequence[state["phase_step_in_cycle"]]
        if phase != state["next_phase"]:
            raise ValueError("Bilevel phase state is inconsistent before advance")
        updated = dict(state)
        updated["global_optimizer_step"] += 1
        updated[f"{phase}_step"] += 1
        updated["phase_step_in_cycle"] += 1
        if updated["phase_step_in_cycle"] == len(sequence):
            updated["bilevel_cycle"] += 1
            updated["phase_step_in_cycle"] = 0
        updated["next_phase"] = sequence[updated["phase_step_in_cycle"]]
        return updated

    @staticmethod
    def bilevel_inner_block_indices(net, last_k_blocks: int | None) -> tuple[int, ...] | None:
        """Resolve the optional final-K DiT subset for an inner update."""
        if last_k_blocks is None:
            return None
        if isinstance(last_k_blocks, bool) or not isinstance(last_k_blocks, int):
            raise TypeError("bilevel_inner_last_k_blocks must be an integer or None")
        num_blocks = len(net.blocks)
        if not 1 <= last_k_blocks <= num_blocks:
            raise ValueError(
                f"bilevel_inner_last_k_blocks must be in [1, {num_blocks}], got {last_k_blocks}"
            )
        return tuple(range(num_blocks - last_k_blocks, num_blocks))

    @staticmethod
    def configure_bilevel_parameter_phase(net, phase: str, inner_block_indices: tuple[int, ...] | None) -> None:
        """Apply the optional single-GPU last-K phase freeze policy.

        ``None`` deliberately leaves the established full-DiT bilevel path
        unchanged.  A selected subset is configured before each full phase so
        autograd does not retain parameter gradients for frozen early blocks.
        """
        if inner_block_indices is None:
            if hasattr(net, "_bilevel_inner_block_indices"):
                delattr(net, "_bilevel_inner_block_indices")
            return

        for parameter in net.parameters():
            parameter.requires_grad_(False)
        if phase == "inner":
            for block_index in inner_block_indices:
                net.blocks[block_index].requires_grad_(True)
            net._bilevel_inner_block_indices = inner_block_indices
        elif phase == "outer":
            net.shared_token_linear.weight.requires_grad_(True)
            # Outer loss must not traverse structured L21 at all.
            net._bilevel_inner_block_indices = None
        else:
            raise ValueError(f"Unknown bilevel phase: {phase}")

    @staticmethod
    def configure_bilevel_last_k_ddp(trainer_config, model_config) -> bool:
        """Resolve DDP settings required by the dynamic last-K phase graph.

        This runs while the config is still mutable, before DDP construction.
        It intentionally leaves ordinary training, full-DiT bilevel, and FSDP
        configurations untouched.
        """
        is_last_k_no_fsdp_bilevel = (
            trainer_config.enable_bilevel_training
            and trainer_config.bilevel_inner_last_k_blocks is not None
            and getattr(model_config, "fsdp_shard_size", 1) == 1
        )
        if is_last_k_no_fsdp_bilevel:
            trainer_config.ddp.static_graph = False
            trainer_config.ddp.find_unused_parameters = True
        return is_last_k_no_fsdp_bilevel

    def _should_run_initial_validation(self, iteration: int) -> bool:
        return (
            self.config.trainer.run_validation
            and iteration == 0
            and self.config.trainer.run_validation_on_start
        )

    def train(
        self,
        model: ImaginaireModel,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_val: torch.utils.data.DataLoader,
    ) -> None:
        """The training function.

        Args:
            model (ImaginaireModel): The PyTorch model.
            dataloader_train (torch.utils.data.DataLoader): The training data loader.
            dataloader_val (torch.utils.data.DataLoader): The validation data loader.
        """
        # Leaving this for backward compability for now, but we can think about moving this to model.on_train_start for all models.
        model = model.to("cuda", memory_format=self.config.trainer.memory_format)  # type: ignore
        model.on_train_start(self.config.trainer.memory_format)

        # Initialize the optimizer, scheduler, and grad_scaler.
        self.callbacks.on_optimizer_init_start()
        if self.config.trainer.enable_bilevel_training:
            inner_block_indices = self.bilevel_inner_block_indices(
                model.net, self.config.trainer.bilevel_inner_last_k_blocks
            )
            if inner_block_indices is not None and getattr(model.config, "fsdp_shard_size", 1) > 1:
                raise RuntimeError(
                    "bilevel_inner_last_k_blocks currently requires fsdp_shard_size=1; "
                    "the phase-specific requires_grad policy is intentionally single-GPU/no-FSDP only."
                )
            inner_optimizer, outer_optimizer, inner_params, outer_params, inner_scheduler, outer_scheduler = model.init_bilevel_optimizers(
                self.config.optimizer,
                self.config.scheduler,
                outer_lr=1e-5,
                inner_block_indices=inner_block_indices,
            )
            optimizer, scheduler = inner_optimizer, None
        else:
            optimizer, scheduler = model.init_optimizer_scheduler(self.config.optimizer, self.config.scheduler)
        grad_scaler = torch.amp.GradScaler("cuda", **self.config.trainer.grad_scaler_args)
        self.callbacks.on_optimizer_init_end()
        if self.config.trainer.enable_bilevel_training:
            bilevel_state = self.checkpointer.load_bilevel(
                model, inner_optimizer, outer_optimizer, inner_scheduler, outer_scheduler, grad_scaler
            )
            if self.config.trainer.distributed_parallelism == "ddp":
                model_ddp = distributed.parallel_model_wrapper(self.config.trainer.ddp, model)
            elif self.config.trainer.distributed_parallelism == "fsdp":
                model_ddp = model
            else:
                raise ValueError(
                    f"Unknown distributed parallelism mode: {self.config.trainer.distributed_parallelism}"
                )
            self.callbacks.on_train_start(model, iteration=bilevel_state["global_optimizer_step"])
            if self._should_run_initial_validation(bilevel_state["global_optimizer_step"]):
                self.validate(model, dataloader_val, iteration=0)
            with (
                maybe_enable_profiling(self.config, global_step=bilevel_state["global_optimizer_step"]) as torch_profiler,
                maybe_enable_memory_snapshot(self.config, global_step=bilevel_state["global_optimizer_step"]) as memory_profiler,
            ):
                self._train_bilevel(
                    model, dataloader_train, dataloader_val, inner_optimizer, outer_optimizer,
                    inner_scheduler, outer_scheduler, grad_scaler, bilevel_state, torch_profiler,
                    memory_profiler, model_ddp=model_ddp, inner_params=inner_params,
                    outer_params=outer_params, inner_block_indices=inner_block_indices,
                )
            self.callbacks.on_train_end(model, iteration=self._bilevel_global_step)
            self.checkpointer.finalize()
            distributed.barrier()
            self.callbacks.on_app_end()
            return
        # Load the model checkpoint and get the starting iteration number.
        iteration = self.checkpointer.load(model, optimizer, scheduler, grad_scaler)
        grad_accum_iter = 0
        log.critical(f"Distributed parallelism mode: {self.config.trainer.distributed_parallelism}")
        if self.config.trainer.distributed_parallelism == "ddp":
            # Create a DDP model wrapper.
            model_ddp = distributed.parallel_model_wrapper(self.config.trainer.ddp, model)
        elif self.config.trainer.distributed_parallelism == "fsdp":
            model_ddp = model
        else:
            raise ValueError(f"Unknown distributed parallelism mode: {self.config.trainer.distributed_parallelism}")

        log.info("Starting training...")
        self.callbacks.on_train_start(model, iteration=iteration)
        # Initial validation.
        if self._should_run_initial_validation(iteration):
            self.validate(model, dataloader_val, iteration=iteration)
        _end_training = False
        with (
            maybe_enable_profiling(self.config, global_step=iteration) as torch_profiler,
            maybe_enable_memory_snapshot(self.config, global_step=iteration) as memory_profiler,
        ):
            epoch = 0
            while True:
                dataloader_train.sampler.set_epoch(epoch)
                dataloader_train_iter = iter(dataloader_train)
                while True:
                    self.callbacks.on_before_dataloading(iteration)
                    try:
                        with (
                            self.training_timer("dataloader_train"),
                            self.straggler_detector.profile_section(
                                "dataloading",
                                self.config.trainer.straggler_detection.analyze_dataloading,
                                profile_cuda=False,
                            ),
                        ):
                            data_batch = next(dataloader_train_iter)
                    except StopIteration:
                        break
                    finally:
                        self.callbacks.on_after_dataloading(iteration)
                    # If max_iter is reached, exit the training loop.
                    if iteration >= self.config.trainer.max_iter:
                        _end_training = True
                        break
                    # Move all tensors in the data batch to GPU device.
                    data_batch = misc.to(data_batch, device="cuda")
                    # The actual training step.
                    self.callbacks.on_training_step_start(model, data_batch, iteration=iteration)
                    self.callbacks.on_training_step_batch_start(model, data_batch, iteration=iteration)
                    if not model.training:
                        model_ddp.train()
                    assert model_ddp.training, "model_ddp is not in training mode."
                    assert model.training, "model is not in training mode."
                    output_batch, loss, grad_accum_iter = self.training_step(
                        model_ddp,
                        optimizer,
                        scheduler,
                        grad_scaler,
                        data_batch,
                        iteration=iteration,
                        grad_accum_iter=grad_accum_iter,
                    )
                    self.callbacks.on_training_step_batch_end(
                        model, data_batch, output_batch, loss, iteration=iteration
                    )
                    # If the gradients are still being accumulated, continue to load the next training batch.
                    if grad_accum_iter != 0:
                        continue
                    # Do the following when an actual optimizer (update) step has been made.
                    iteration += 1
                    # Save checkpoint.
                    if iteration % self.config.checkpoint.save_iter == 0:
                        self.checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=iteration)
                    self.callbacks.on_training_step_end(model, data_batch, output_batch, loss, iteration=iteration)
                    # Validation.
                    if self.config.trainer.run_validation and iteration % self.config.trainer.validation_iter == 0:
                        self.validate(model, dataloader_val, iteration=iteration)
                    # This iteration is successful; reset the timeout signal.
                    signal.alarm(self.config.trainer.timeout_period)
                    self.straggler_detector.generate_report(iteration)
                    if torch_profiler:
                        torch_profiler.step()
                    if memory_profiler:
                        memory_profiler.step()
                epoch += 1
                if _end_training:
                    break
        log.success("Done with training.")
        if iteration % self.config.checkpoint.save_iter != 0:
            self.checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=iteration)
        self.callbacks.on_train_end(model, iteration=iteration)
        self.checkpointer.finalize()
        distributed.barrier()
        self.callbacks.on_app_end()

    def _train_bilevel(self, model, dataloader_train, dataloader_val, inner_optimizer, outer_optimizer, inner_scheduler, outer_scheduler, grad_scaler, state, torch_profiler=None, memory_profiler=None, model_ddp=None, inner_params=None, outer_params=None, inner_block_indices=None):
        """First-order truncated alternating loop; each optimizer step accumulates N micro-batches."""
        if model_ddp is None:
            model_ddp = model
        self._bilevel_global_step = state["global_optimizer_step"]
        model.train()
        epoch = 0
        if hasattr(dataloader_train, "sampler") and hasattr(dataloader_train.sampler, "set_epoch"):
            dataloader_train.sampler.set_epoch(epoch)
        iterator = iter(dataloader_train)
        global_step = state["global_optimizer_step"]
        last_saved_global_optimizer_step = None
        accum = self.config.trainer.grad_accum_iter
        inner_param_list = [parameter for _, parameter in inner_params] if inner_params is not None else [
            parameter
            for name, parameter in model.net.named_parameters()
            if not name.endswith("shared_token_linear.weight") and parameter.requires_grad
        ]
        outer_param_list = [parameter for _, parameter in outer_params] if outer_params is not None else [
            parameter for name, parameter in model.net.named_parameters()
            if name.endswith("shared_token_linear.weight")
        ]
        while global_step < self.config.trainer.max_iter:
            sequence = self.bilevel_phase_sequence(self.config.trainer.bilevel_inner_steps, self.config.trainer.bilevel_outer_steps)
            for phase in sequence[state["phase_step_in_cycle"]:]:
                if global_step >= self.config.trainer.max_iter:
                    break
                self.configure_bilevel_parameter_phase(model.net, phase, inner_block_indices)
                optimizer, params = (inner_optimizer, inner_param_list) if phase == "inner" else (outer_optimizer, outer_param_list)
                inner_optimizer.zero_grad(set_to_none=True); outer_optimizer.zero_grad(set_to_none=True)
                model._bilevel_phase = phase
                model._bilevel_active_optimizer = optimizer
                for micro_step in range(accum):
                    self.callbacks.on_before_dataloading(global_step)
                    try:
                        with (
                            self.training_timer("dataloader_train"),
                            self.straggler_detector.profile_section(
                                "dataloading",
                                self.config.trainer.straggler_detection.analyze_dataloading,
                                profile_cuda=False,
                            ),
                        ):
                            batch = next(iterator)
                    except StopIteration:
                        epoch += 1
                        if hasattr(dataloader_train, "sampler") and hasattr(dataloader_train.sampler, "set_epoch"):
                            dataloader_train.sampler.set_epoch(epoch)
                        iterator = iter(dataloader_train)
                        with self.training_timer("dataloader_train"):
                            batch = next(iterator)
                    finally:
                        self.callbacks.on_after_dataloading(global_step)
                    batch = misc.to(batch, device="cuda")
                    self.callbacks.on_training_step_start(model, batch, iteration=global_step)
                    self.callbacks.on_training_step_batch_start(model, batch, iteration=global_step)
                    with distributed.ddp_sync_grad(model_ddp, micro_step == accum - 1):
                        self.callbacks.on_before_forward(iteration=global_step)
                        with self.training_timer("forward"):
                            with self.straggler_detector.profile_section(
                                "fwd", self.config.trainer.straggler_detection.analyze_forward
                            ):
                                output_batch, loss = model_ddp.training_step(batch, global_step, phase=phase)
                        self.callbacks.on_after_forward(iteration=global_step)
                        self.callbacks.on_before_backward(model_ddp, loss, iteration=global_step)
                        with self.training_timer("backward"):
                            with self.straggler_detector.profile_section(
                                "bwd", self.config.trainer.straggler_detection.analyze_backward
                            ):
                                grad_scaler.scale(loss / accum).backward()
                                if self.config.trainer.distributed_parallelism == "ddp":
                                    model_ddp.module.on_after_backward()
                                else:
                                    model_ddp.on_after_backward()
                        self.callbacks.on_after_backward(model_ddp, iteration=global_step)
                    self.callbacks.on_training_step_batch_end(model, batch, output_batch, loss, iteration=global_step)
                active_scheduler = inner_scheduler if phase == "inner" else outer_scheduler
                with self.training_timer("optimizer_step"):
                    with self.straggler_detector.profile_section(
                        "opt", self.config.trainer.straggler_detection.analyze_optimizer
                    ):
                        grad_scaler.unscale_(optimizer)
                        self._clear_inactive_gradients(model, params)
                        # Existing callbacks (including GradClip/W&B LR logging) now see
                        # only the active phase gradients and optimizer.
                        self.callbacks.on_before_optimizer_step(
                            model_ddp, optimizer, active_scheduler, grad_scaler, iteration=global_step
                        )
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                        active_scheduler.step()
                        self.callbacks.on_before_zero_grad(
                            model_ddp, optimizer, active_scheduler, iteration=global_step
                        )
                        if self.config.trainer.distributed_parallelism == "ddp":
                            model_ddp.module.on_before_zero_grad(
                                optimizer, active_scheduler, iteration=global_step
                            )
                        else:
                            model_ddp.on_before_zero_grad(
                                optimizer, active_scheduler, iteration=global_step
                            )
                        inner_optimizer.zero_grad(set_to_none=True)
                        outer_optimizer.zero_grad(set_to_none=True)
                state = self.advance_bilevel_state(state, self.config.trainer.bilevel_inner_steps, self.config.trainer.bilevel_outer_steps)
                global_step = state["global_optimizer_step"]
                self._bilevel_global_step = global_step
                raw_task_loss = output_batch["bilevel_raw_task_loss"]
                output_batch.update({
                    "bilevel/phase": phase,
                    "bilevel/global_optimizer_step": global_step,
                    "bilevel/inner_step": state["inner_step"],
                    "bilevel/outer_step": state["outer_step"],
                    "bilevel/cycle": state["bilevel_cycle"],
                    "bilevel/phase_step_in_cycle": state["phase_step_in_cycle"],
                    "bilevel/objective": loss.detach(),
                })
                if distributed.is_rank0() and wandb.run:
                    bilevel_log = {
                        "bilevel/phase": phase,
                        "bilevel/global_optimizer_step": global_step,
                        "bilevel/inner_step": state["inner_step"],
                        "bilevel/outer_step": state["outer_step"],
                        "bilevel/cycle": state["bilevel_cycle"],
                        "bilevel/phase_step_in_cycle": state["phase_step_in_cycle"],
                        "bilevel/task_loss": raw_task_loss.detach().float().item(),
                        f"bilevel/{phase}_objective": loss.detach().float().item(),
                    }
                    if phase == "inner":
                        bilevel_log.update({
                            f"bilevel/{name}": value.detach().float().item()
                            for name, value in output_batch.items()
                            if name.startswith("structured_l21_") and torch.is_tensor(value)
                        })
                    wandb.log(bilevel_log, step=global_step)
                # Match the ordinary lifecycle: checkpoint the completed
                # optimizer step, then emit step-end, then run validation.
                if global_step % self.config.checkpoint.save_iter == 0:
                    self.checkpointer.save_bilevel(
                        model, inner_optimizer, outer_optimizer, inner_scheduler,
                        outer_scheduler, grad_scaler, state,
                    )
                    last_saved_global_optimizer_step = global_step
                self.callbacks.on_training_step_end(model, batch, output_batch, loss, iteration=global_step)
                if dataloader_val is not None and self.config.trainer.run_validation and global_step % self.config.trainer.validation_iter == 0:
                    self.validate(model, dataloader_val, iteration=global_step)
                signal.alarm(self.config.trainer.timeout_period)
                self.straggler_detector.generate_report(global_step)
                if torch_profiler:
                    torch_profiler.step()
                if memory_profiler:
                    memory_profiler.step()
            if global_step >= self.config.trainer.max_iter:
                break
        if last_saved_global_optimizer_step != global_step:
            self.checkpointer.save_bilevel(
                model, inner_optimizer, outer_optimizer, inner_scheduler, outer_scheduler, grad_scaler, state
            )

    @staticmethod
    def _clear_inactive_gradients(model, active_params) -> None:
        active_ids = {id(parameter) for parameter in active_params}
        for parameter in model.parameters():
            if id(parameter) not in active_ids:
                parameter.grad = None
