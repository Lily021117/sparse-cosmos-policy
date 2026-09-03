import torch
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cosmos_policy.trainer import CosmosPolicyTrainer
from cosmos_policy._src.predict2.checkpointer.dcp import DistributedCheckpointer
from cosmos_policy._src.imaginaire.config import CheckpointConfig, JobConfig
from cosmos_policy._src.imaginaire.model import ImaginaireModel
from cosmos_policy.structured_l21 import apply_structured_l21_for_phase


class _TinyModel(ImaginaireModel):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_lora=False)
        self.inner = torch.nn.Linear(1, 1, bias=False)
        self.shared_token_linear = torch.nn.Linear(1, 1, bias=False)
        self.after_backward_calls = 0

    @property
    def net(self):
        return self

    def training_step(self, data, iteration, phase="inner"):
        del data, iteration
        task_loss = self.inner.weight.sum() + self.shared_token_linear.weight.sum()
        return {
            "edm_loss": self.inner.weight.sum().detach(),
            "bilevel_raw_task_loss": task_loss.detach(),
        }, task_loss

    def on_after_backward(self):
        self.after_backward_calls += 1


class _Recorder:
    def __init__(self):
        self.calls = []
        self.phases = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append(name)
            if name == "on_training_step_end" and args:
                self.phases.append(getattr(args[0], "_bilevel_phase", None))
        return record


class _BilevelCheckpointer:
    def __init__(self):
        self.states = []

    def save_bilevel(self, *args):
        self.states.append(args[-1].copy())


def test_bilevel_phase_sequence_and_accumulation_unit():
    assert CosmosPolicyTrainer.bilevel_phase_sequence(5, 1) == ["inner"] * 5 + ["outer"]


def test_initial_validation_predicate_does_not_advance_phase_state():
    trainer = CosmosPolicyTrainer.__new__(CosmosPolicyTrainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(run_validation=True, run_validation_on_start=True)
    )
    state = {"global_optimizer_step": 0, "inner_step": 0, "outer_step": 0,
             "bilevel_cycle": 0, "phase_step_in_cycle": 0, "next_phase": "inner"}
    before = state.copy()
    assert trainer._should_run_initial_validation(state["global_optimizer_step"])
    assert state == before
    assert not trainer._should_run_initial_validation(1)


def test_disjoint_parameter_sets_update_isolated():
    dit = torch.nn.Parameter(torch.tensor(1.0))
    shared = torch.nn.Parameter(torch.tensor(1.0))
    inner = torch.optim.SGD([dit], lr=0.1)
    outer = torch.optim.SGD([shared], lr=0.1)
    inner.zero_grad(); dit.grad = torch.ones_like(dit); inner.step()
    assert dit.item() != 1.0 and shared.item() == 1.0
    outer.zero_grad(); shared.grad = torch.ones_like(shared); outer.step()
    assert shared.item() != 1.0


def test_parameter_sets_do_not_overlap():
    dit = torch.nn.Parameter(torch.tensor(1.0))
    shared = torch.nn.Parameter(torch.tensor(1.0))
    assert {id(dit)}.isdisjoint({id(shared)})


def test_bilevel_checkpoint_state_continues_mid_cycle():
    state = {
        "global_optimizer_step": 4, "inner_step": 4, "outer_step": 0,
        "bilevel_cycle": 0, "phase_step_in_cycle": 4, "next_phase": "inner",
    }
    DistributedCheckpointer.validate_bilevel_state(state)
    resumed = []
    for _ in range(8):
        resumed.append(state["next_phase"])
        state = CosmosPolicyTrainer.advance_bilevel_state(state)
    assert resumed == ["inner", "outer", "inner", "inner", "inner", "inner", "inner", "outer"]
    assert state == {
        "global_optimizer_step": 12, "inner_step": 10, "outer_step": 2,
        "bilevel_cycle": 2, "phase_step_in_cycle": 0, "next_phase": "inner",
    }


def test_bilevel_checkpoint_state_after_outer_starts_next_cycle_inner():
    state = {
        "global_optimizer_step": 6, "inner_step": 5, "outer_step": 1,
        "bilevel_cycle": 1, "phase_step_in_cycle": 0, "next_phase": "inner",
    }
    DistributedCheckpointer.validate_bilevel_state(state)
    assert CosmosPolicyTrainer.advance_bilevel_state(state)["next_phase"] == "inner"


def test_dcp_bilevel_save_load_restores_two_optimizers_and_phase_state(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path))
    config = CheckpointConfig(dcp_async_mode_enabled=False, strict_resume=False)
    job = JobConfig(project="test", group="bilevel", name="run")
    model = _TinyModel()
    inner = torch.optim.AdamW(model.inner.parameters(), lr=0.1)
    outer = torch.optim.AdamW(model.shared_token_linear.parameters(), lr=0.2)
    inner_scheduler = torch.optim.lr_scheduler.LambdaLR(inner, lambda _: 1.0)
    outer_scheduler = torch.optim.lr_scheduler.LambdaLR(outer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    model.inner.weight.grad = torch.ones_like(model.inner.weight); inner.step(); inner_scheduler.step()
    model.shared_token_linear.weight.grad = torch.ones_like(model.shared_token_linear.weight); outer.step(); outer_scheduler.step()
    state = {"global_optimizer_step": 4, "inner_step": 4, "outer_step": 0,
             "bilevel_cycle": 0, "phase_step_in_cycle": 4, "next_phase": "inner"}
    checkpointer = DistributedCheckpointer(config, job, callbacks=None, disable_async=True)
    checkpointer.save_bilevel(model, inner, outer, inner_scheduler, outer_scheduler, scaler, state)
    saved_inner = model.inner.weight.detach().clone(); saved_outer = model.shared_token_linear.weight.detach().clone()
    restored = _TinyModel()
    restored_inner = torch.optim.AdamW(restored.inner.parameters(), lr=0.1)
    restored_outer = torch.optim.AdamW(restored.shared_token_linear.parameters(), lr=0.2)
    restored_inner_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_inner, lambda _: 1.0)
    restored_outer_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_outer, lambda _: 1.0)
    restored_scaler = torch.amp.GradScaler("cuda", enabled=False)
    load_callbacks = _Recorder()
    loaded = DistributedCheckpointer(config, job, callbacks=load_callbacks, disable_async=True).load_bilevel(
        restored, restored_inner, restored_outer, restored_inner_scheduler, restored_outer_scheduler, restored_scaler
    )
    torch.testing.assert_close(restored.inner.weight, saved_inner)
    torch.testing.assert_close(restored.shared_token_linear.weight, saved_outer)
    assert loaded == state
    assert restored_inner.state_dict()["state"] == inner.state_dict()["state"]
    assert restored_outer.state_dict()["state"] == outer.state_dict()["state"]
    assert restored_inner_scheduler.state_dict() == inner_scheduler.state_dict()
    assert restored_outer_scheduler.state_dict() == outer_scheduler.state_dict()
    assert load_callbacks.calls == [
        "on_load_checkpoint_start",
        "on_load_checkpoint",
        "on_load_checkpoint_end",
    ]
    # The next matching update proves moments were restored, not merely present.
    model.inner.weight.grad = torch.ones_like(model.inner.weight); inner.step()
    restored.inner.weight.grad = torch.ones_like(restored.inner.weight); restored_inner.step()
    torch.testing.assert_close(restored.inner.weight, model.inner.weight)


def test_legacy_dcp_checkpoint_schema_remains_loadable(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path))
    config = CheckpointConfig(dcp_async_mode_enabled=False, strict_resume=False, load_training_state=True)
    job = JobConfig(project="test", group="legacy", name="run")
    model = _TinyModel(); optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    model.inner.weight.grad = torch.ones_like(model.inner.weight); optimizer.step(); scheduler.step()
    DistributedCheckpointer(config, job, callbacks=None, disable_async=True).save(model, optimizer, scheduler, scaler, iteration=1)
    restored = _TinyModel(); restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.1)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda _: 1.0)
    restored_scaler = torch.amp.GradScaler("cuda", enabled=False)
    torch.distributed.init_process_group(
        backend="gloo", init_method=f"file://{tmp_path / 'legacy_dcp_pg'}", rank=0, world_size=1
    )
    try:
        iteration = DistributedCheckpointer(config, job, callbacks=None, disable_async=True).load(
            restored, restored_optimizer, restored_scheduler, restored_scaler
        )
        assert iteration == 1
        torch.testing.assert_close(restored.inner.weight, model.inner.weight)
    finally:
        torch.distributed.destroy_process_group()


def test_bilevel_lifecycle_callbacks_validation_and_boundary_checkpoint(monkeypatch):
    trainer = CosmosPolicyTrainer.__new__(CosmosPolicyTrainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(grad_accum_iter=2, bilevel_inner_steps=1, bilevel_outer_steps=1,
                                max_iter=2, run_validation=True, validation_iter=1, timeout_period=999999,
                                distributed_parallelism="fsdp",
                                straggler_detection=SimpleNamespace(
                                    analyze_dataloading=False, analyze_forward=False, analyze_backward=False,
                                    analyze_optimizer=False,
                                )),
        checkpoint=SimpleNamespace(save_iter=1),
    )
    trainer.callbacks = _Recorder()
    trainer.checkpointer = _BilevelCheckpointer()
    trainer.training_timer = lambda name: nullcontext()
    trainer.straggler_detector = SimpleNamespace(
        generate_report=lambda step: None,
        profile_section=lambda *args, **kwargs: nullcontext(),
    )
    trainer.validate = lambda model, loader, iteration: trainer.callbacks.calls.append("validate")
    model = _TinyModel()
    inner = torch.optim.SGD(model.inner.parameters(), lr=0.1)
    outer = torch.optim.SGD(model.shared_token_linear.parameters(), lr=0.1)
    inner_scheduler = torch.optim.lr_scheduler.LambdaLR(inner, lambda _: 1.0)
    outer_scheduler = torch.optim.lr_scheduler.LambdaLR(outer, lambda _: 1.0)
    state = {"global_optimizer_step": 0, "inner_step": 0, "outer_step": 0,
             "bilevel_cycle": 0, "phase_step_in_cycle": 0, "next_phase": "inner"}
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    with patch("cosmos_policy.trainer.misc.to", lambda batch, device: batch), patch("cosmos_policy.trainer.signal.alarm"):
        trainer._train_bilevel(model, [{"x": torch.tensor(1.0)}] * 4, [{}], inner, outer,
                               inner_scheduler, outer_scheduler, scaler, state)
    assert trainer.callbacks.calls.count("on_training_step_start") == 4
    assert trainer.callbacks.calls.count("on_before_optimizer_step") == 2
    assert trainer.callbacks.calls.count("on_training_step_end") == 2
    assert model.after_backward_calls == 4
    assert trainer.callbacks.calls.count("validate") == 2
    assert len(trainer.checkpointer.states) == 2


def _run_bilevel_phases(max_iter, save_iter=1000):
    trainer = CosmosPolicyTrainer.__new__(CosmosPolicyTrainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(
            grad_accum_iter=1, bilevel_inner_steps=5, bilevel_outer_steps=1,
            max_iter=max_iter, run_validation=False, validation_iter=100,
            timeout_period=999999, distributed_parallelism="fsdp",
            straggler_detection=SimpleNamespace(
                analyze_dataloading=False, analyze_forward=False, analyze_backward=False,
                analyze_optimizer=False,
            ),
        ),
        checkpoint=SimpleNamespace(save_iter=save_iter),
    )
    trainer.callbacks = _Recorder()
    trainer.checkpointer = _BilevelCheckpointer()
    trainer.training_timer = lambda name: nullcontext()
    trainer.straggler_detector = SimpleNamespace(
        generate_report=lambda step: None,
        profile_section=lambda *args, **kwargs: nullcontext(),
    )
    model = _TinyModel()
    inner = torch.optim.SGD(model.inner.parameters(), lr=0.1)
    outer = torch.optim.SGD(model.shared_token_linear.parameters(), lr=0.1)
    inner_scheduler = torch.optim.lr_scheduler.LambdaLR(inner, lambda _: 1.0)
    outer_scheduler = torch.optim.lr_scheduler.LambdaLR(outer, lambda _: 1.0)
    state = {"global_optimizer_step": 0, "inner_step": 0, "outer_step": 0,
             "bilevel_cycle": 0, "phase_step_in_cycle": 0, "next_phase": "inner"}
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    with patch("cosmos_policy.trainer.misc.to", lambda batch, device: batch), patch(
        "cosmos_policy.trainer.signal.alarm"
    ):
        trainer._train_bilevel(
            model, [{"x": torch.tensor(1.0)}], None, inner, outer,
            inner_scheduler, outer_scheduler, scaler, state,
        )
    return trainer.callbacks.phases, trainer.checkpointer.states, model


def test_bilevel_max_iter_stops_at_exact_optimizer_step():
    assert _run_bilevel_phases(4)[0] == ["inner"] * 4
    assert _run_bilevel_phases(6)[0] == ["inner"] * 5 + ["outer"]
    assert _run_bilevel_phases(7)[0] == ["inner"] * 5 + ["outer", "inner"]


def test_bilevel_final_checkpoint_is_not_duplicated():
    _, saves_on_boundary, _ = _run_bilevel_phases(2, save_iter=2)
    assert [state["global_optimizer_step"] for state in saves_on_boundary] == [2]
    _, saves_off_boundary, _ = _run_bilevel_phases(2, save_iter=3)
    assert [state["global_optimizer_step"] for state in saves_off_boundary] == [2]


def test_bilevel_inner_and_outer_objective_semantics():
    raw = torch.tensor(2.0, requires_grad=True)
    with patch(
        "cosmos_policy.structured_l21.apply_structured_l21",
        return_value=(raw + 0.75, {"structured_l21_mlp_penalty": torch.tensor(1.5)}),
    ) as regularize:
        inner, metrics = apply_structured_l21_for_phase(
            raw, torch.nn.Linear(1, 1), object(), phase="inner", component_lambdas={"mlp": 0.5}
        )
        outer, outer_metrics = apply_structured_l21_for_phase(
            raw, torch.nn.Linear(1, 1), object(), phase="outer", component_lambdas={"mlp": 0.5}
        )
    torch.testing.assert_close(inner - raw, torch.tensor(0.75))
    assert metrics["structured_l21_mlp_penalty"].item() == 1.5
    assert outer is raw
    assert outer_metrics == {}
    assert regularize.call_count == 1


def test_libero_ordinary_and_bilevel_freeze_configuration():
    config_source = (
        Path(__file__).parent / "config/experiment/cosmos_policy_experiment_configs.py"
    ).read_text()
    ordinary_start = config_source.index("cosmos_predict2_2b_480p_libero = LazyDict")
    variant_start = config_source.index("cosmos_predict2_2b_480p_libero__bilevel = LazyDict")
    ordinary = config_source[ordinary_start:variant_start]
    variant = config_source[variant_start:config_source.index("# Inference version", variant_start)]
    assert "freeze_shared_token_linear=True" in ordinary
    assert "enable_bilevel_training=True" in variant
    assert "freeze_shared_token_linear=False" in variant
