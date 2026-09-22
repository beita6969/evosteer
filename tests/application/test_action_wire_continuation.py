import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from skillev_private.experiments.bayesian_condition_transition import (
    resume_condition,
    save_condition_transition,
)
from skillev_private.experiments.bayesian_improve_training import _bind_run_directory
from skillev_private.experiments.bayesian_training_config import BayesianFormalConfig

from skillev.application import SKILLEVApplication
from skillev.application_continuation import ActionWireContinuation, ReasoningContinuation
from skillev.contracts.action_wire import NATIVE_TOOL_CARRIER_WIRE, NATIVE_TOOL_HANDOFF_WIRE
from skillev.training import PrivateCheckpointStorageBinding
from skillev.training.config import INTERFACE_POLICY_ROLLOUT_CONFIG_FORMAT
from tests.application.test_full_vertical_loop import build_application_fixture
from tests.application.test_reasoning_continuation import larger_reasoning, with_rollout


@pytest.fixture
def identity(tmp_path, training_backbone_config):
    fixture = build_application_fixture(tmp_path, training_backbone_config)
    return with_rollout(
        fixture.public_identity,
        format=INTERFACE_POLICY_ROLLOUT_CONFIG_FORMAT,
        phase_context=True,
        action_wire=NATIVE_TOOL_CARRIER_WIRE,
    )


def test_only_explicit_v2_to_v3_is_allowed_and_reasoning_contract_stays_strict(identity):
    target = with_rollout(identity, action_wire=NATIVE_TOOL_HANDOFF_WIRE)
    assert ActionWireContinuation(identity, 20).require_target(target) == identity.snapshot_identity
    with pytest.raises(ValueError):
        ReasoningContinuation(identity, 20).require_target(target)
    with pytest.raises(ValueError):
        ActionWireContinuation(identity, 20).require_target(larger_reasoning(target))


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (NATIVE_TOOL_CARRIER_WIRE, NATIVE_TOOL_CARRIER_WIRE),
        (NATIVE_TOOL_HANDOFF_WIRE, NATIVE_TOOL_CARRIER_WIRE),
        ("native-single-tool-call@1", NATIVE_TOOL_HANDOFF_WIRE),
        ("structured-action-json@3", NATIVE_TOOL_HANDOFF_WIRE),
    ],
)
def test_action_wire_transition_does_not_authorize_other_protocol_migrations(identity, old, new):
    with pytest.raises(ValueError):
        ActionWireContinuation(with_rollout(identity, action_wire=old), 20).require_target(
            with_rollout(identity, action_wire=new)
        )


@pytest.mark.parametrize("axis", ["seed", "catalog", "scorer", "sampling"])
def test_wire_transition_preserves_other_scientific_axes(identity, axis):
    target = with_rollout(identity, action_wire=NATIVE_TOOL_HANDOFF_WIRE)
    if axis in {"seed", "catalog"}:
        target = with_rollout(
            target, **({"base_seed": 1} if axis == "seed" else {"reasoning_tool_catalog": True})
        )
    else:
        changes = (
            {"terminal_evaluation_conditions_json": '{"scorer":"changed"}'}
            if axis == "scorer"
            else {"sampling_schedule_algorithm": "changed"}
        )
        target = replace(target, snapshot_identity=replace(target.snapshot_identity, **changes))
    with pytest.raises(ValueError):
        ActionWireContinuation(identity, 20).require_target(target)


def test_private_formal_transition_is_single_field_and_preserves_current_source_declaration(
    tmp_path,
):
    root = tmp_path / "branch"
    source = BayesianFormalConfig(
        format="skillev-bayesian-formal-training@6",
        phase_context=True,
        action_wire=NATIVE_TOOL_CARRIER_WIRE,
        reasoning_tool_catalog=True,
    )
    target = replace(source, action_wire=NATIVE_TOOL_HANDOFF_WIRE)
    _bind_run_directory(root, source, None)
    original = (root / "formal-config.json").read_bytes()
    snapshot = root / "checkpoints/paused-step-00000020"
    with pytest.raises(ValueError):
        _bind_run_directory(root, target, snapshot, allow_new_reasoning=True)
    assert _bind_run_directory(root, target, snapshot, allow_new_action_wire=True) == source
    assert source.condition != target.condition
    for bad in (
        replace(target, max_reasoning_tokens=2048),
        replace(target, max_turns=25),
        replace(target, adapter_learning_rate=0.002),
    ):
        with pytest.raises(ValueError):
            _bind_run_directory(root, bad, snapshot, allow_new_action_wire=True)
    with pytest.raises(ValueError):
        _bind_run_directory(
            root, target, snapshot, allow_new_action_wire=True, allow_new_reasoning=True
        )
    assert (root / "formal-config.json").read_bytes() == original
    # A branch of an earlier reasoning continuation must use its current config,
    # rather than reinterpreting the immutable original formal-config file.
    (root / "condition-current.json").write_text(json.dumps({"config": target.to_value()}))
    assert resume_condition(root, target, allow_new_horizons=False) is None


def test_action_wire_boundary_is_new_condition_without_relabeling_source(tmp_path):
    source = BayesianFormalConfig(
        format="skillev-bayesian-formal-training@6",
        phase_context=True,
        action_wire=NATIVE_TOOL_CARRIER_WIRE,
    )
    target = replace(source, action_wire=NATIVE_TOOL_HANDOFF_WIRE)
    saved_names = []
    app = SimpleNamespace(
        training_loop=SimpleNamespace(optimizer_step=20),
        snapshot_identity=SimpleNamespace(sampling_schedule_algorithm="original-curriculum"),
        evolution_loop=SimpleNamespace(
            save_condition_boundary=lambda name: saved_names.append(name)
        ),
    )
    boundary = save_condition_transition(
        app,
        root=tmp_path,
        source_snapshot=tmp_path / "source-step20",
        source=source,
        target=target,
        action_wire=True,
    )
    declaration = json.loads((tmp_path / "condition-current.json").read_text())
    assert declaration["format"] == "action-wire-condition-transition@1"
    assert declaration["source_config"] == source.to_value()
    assert declaration["config"] == target.to_value()
    assert declaration["saved_optimizer_step"] == 20
    assert declaration["effective_from_optimizer_step"] == 21
    assert declaration["sampling_condition"] == "original-curriculum"
    assert saved_names == [boundary.name]


def test_formal_wrapper_forwards_wire_declaration_to_complete_restore(tmp_path, monkeypatch):
    captures = {}
    app = object()

    def resume(**kwargs):
        captures.update(kwargs)
        return app

    monkeypatch.setattr(SKILLEVApplication, "resume", resume)
    runtime = SimpleNamespace(
        rollout_generator_factory=None,
        gradient_preparer=None,
        workflow_resources=SimpleNamespace(binding=None),
        step_adapter_publisher_factory=None,
        skill_author_factory=None,
        require_bound=lambda value: captures.update(bound=value),
    )
    continuation = ActionWireContinuation(SimpleNamespace(), 20)
    result = SKILLEVApplication.resume_formal(
        snapshot_directory=tmp_path / "step20",
        backbone_config=None,
        task_provider_factory=None,
        base_session_factory=None,
        terminal_components=None,
        checkpoint_storage=PrivateCheckpointStorageBinding(directory=str(tmp_path)),
        initial_checkpoint=None,
        public_identity=None,
        event_log=None,
        clock=lambda: "fixture",
        runtime=runtime,
        action_wire_continuation=continuation,
    )
    assert result is app
    assert captures["bound"] is app
    assert captures["action_wire_continuation"] is continuation
    assert captures["reasoning_continuation"] is None
    assert captures["horizon_continuation"] is None


def test_wire_cli_requires_a_checkpoint_before_loading_models(tmp_path, monkeypatch):
    from skillev_private.experiments.bayesian_improve_training import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "formal",
            "--config",
            "missing",
            "--bindings",
            "missing",
            "--run-root",
            str(tmp_path),
            "--allow-new-action-wire",
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
