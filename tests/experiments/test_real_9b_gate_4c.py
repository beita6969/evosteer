from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts import gate4c_runtime_support as runtime_support
from scripts import real_9b_gate_4c as gate
from scripts.gate4c_runtime_support import (
    CUBLAS_WORKSPACE_CONFIG,
    INITIALIZATION_SEED,
    production_config_hash,
)
from skillev.contracts import TrajectoryRecord, canonical_json, stable_hash
from skillev.experiments import (
    GATE_4C_CHILD_TERMINAL_FORMAT,
    GATE_4C_OPERATOR_TERMINAL_FORMAT,
    GATE_4C_STAGE_EVENT_FORMAT,
    DeterministicBackendIdentity,
    ExecutionHardwareIdentity,
    Gate4cSpec,
    checkpoint_tree_hash,
)
from skillev.experiments.gate4c_spec import (
    GATE_4C_OPERATOR_VERSION,
    GATE_4C_PROCEDURE_VERSION,
)
from skillev.policy import (
    BaseModelArtifactIdentity,
    TokenizerArtifactIdentity,
    TrainableStateIdentity,
)
from skillev.policy.artifact_identity import ArtifactFileIdentity
from skillev.policy.tokenizer_identity import PublicTokenizerKind
from skillev.scoring import render_forward_prefix, render_hindsight_prefix
from tests.v3_helpers import CharacterTokenizer


def _digest(label: str) -> str:
    return stable_hash({"fixture": label})


def _spec() -> Gate4cSpec:
    config_file = ArtifactFileIdentity(
        relative_path="config.json",
        size_bytes=1,
        sha256=_digest("config"),
    )
    weight_file = ArtifactFileIdentity(
        relative_path="model.safetensors",
        size_bytes=1,
        sha256=_digest("weights"),
    )
    base = BaseModelArtifactIdentity.create(
        backend_class="transformers.AutoModelForMultimodalLM",
        upstream_revision="revision",
        dtype_conversion_policy="from_pretrained:bfloat16;module_to:bfloat16",
        model_config=config_file,
        generation_config=None,
        weight_index=None,
        weight_shards=(weight_file,),
    )
    tokenizer = TokenizerArtifactIdentity.create(
        kind=PublicTokenizerKind.QWEN,
        tokenizer_id="Qwen/Qwen3.5-9B",
        revision="revision",
        backend_serialization_hash=_digest("tokenizer-backend"),
        tokenizer_config_hash=_digest("tokenizer-config"),
        chat_template_hash=_digest("chat-template"),
        special_tokens_hash=_digest("special-tokens"),
        added_tokens_hash=_digest("added-tokens"),
        transformers_version="1",
        tokenizers_version="1",
    )
    deployment = _digest("deployment")
    trainable = TrainableStateIdentity.create(
        backbone_deployment_hash=deployment,
        forward_adapter_hash=_digest("forward"),
        backward_adapter_hash=_digest("backward"),
        z_head_hash=_digest("z"),
    )
    hardware = ExecutionHardwareIdentity(
        accelerator_name="NVIDIA H800",
        compute_capability="9.0",
        visible_device_count=1,
        nvidia_driver_version="1",
        cuda_runtime_version="1",
        cudnn_version="1",
        nccl_version="1",
        kernel_release="1",
        safetensors_version="1",
    )
    backend = DeterministicBackendIdentity(
        torch_version="1",
        transformers_version="1",
        peft_version="1",
        flash_linear_attention_version="0.5.2",
        tokenizers_version="1",
        safetensors_version="1",
        attention_implementation="sdpa",
        cublas_workspace_config=CUBLAS_WORKSPACE_CONFIG,
        deterministic_algorithms=True,
        cudnn_deterministic=True,
        cudnn_benchmark=False,
        cuda_matmul_allow_tf32=False,
        cudnn_allow_tf32=False,
    )
    return Gate4cSpec(
        source_commit="a" * 40,
        source_tree_hash=_digest("source-tree"),
        source_archive_sha256=_digest("source-archive"),
        public_wheel_sha256=_digest("public-wheel"),
        private_wheel_sha256=_digest("private-wheel"),
        lockfile_sha256=_digest("lockfile"),
        base_model_artifact=base,
        tokenizer_artifact=tokenizer,
        qwen_deployment_hash=deployment,
        initial_trainable_state=trainable,
        initial_checkpoint_tree_hash=_digest("initial-checkpoint"),
        initial_checkpoint_inspection_hash=_digest("initial-checkpoint-inspection"),
        expected_hardware_identity=hardware,
        deterministic_backend_identity=backend,
        gate_procedure_version=GATE_4C_PROCEDURE_VERSION,
        operator_version=GATE_4C_OPERATOR_VERSION,
        stage_journal_version=GATE_4C_STAGE_EVENT_FORMAT,
        child_terminal_version=GATE_4C_CHILD_TERMINAL_FORMAT,
        operator_terminal_version=GATE_4C_OPERATOR_TERMINAL_FORMAT,
        gpu_sampling_interval_ms=250,
        expected_physical_gpu_uuid="GPU-11111111-2222-3333-4444-555555555555",
        production_config_hash=production_config_hash(),
        seed=gate.GATE_SEED,
        initialization_seed=INITIALIZATION_SEED,
        horizon=gate.HORIZON,
        maximum_h0_tokens=gate.MAXIMUM_H0_TOKENS,
        max_reasoning_tokens=gate.MAX_REASONING_TOKENS,
        max_action_tokens=gate.MAX_ACTION_TOKENS,
        model_input_cap=gate.MODEL_INPUT_CAP,
        peak_reserved_cap_bytes=gate.PEAK_RESERVED_LIMIT_BYTES,
        eos_token_id=1,
    )


def test_gate_4c_is_one_fixed_spec_bound_production_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "real_9b_gate_4c.py",
            "--spec",
            "gate-4c-spec.json",
            "--local-model-path",
            "model",
            "--work-directory",
            "work",
        ],
    )

    args = gate._args()

    assert args.spec == "gate-4c-spec.json"
    assert args.local_model_path == "model"
    assert gate.GATE_4C_FORMAT == "skillev-real-9b-gate-4c@6"
    assert gate.HORIZON == 15
    assert gate.MAX_REASONING_TOKENS == 768
    assert gate.MAX_ACTION_TOKENS == 768
    assert gate.MAXIMUM_H0_TOKENS == 32_768
    assert gate.MODEL_INPUT_CAP == 65_536
    assert gate.PEAK_RESERVED_LIMIT_BYTES == 64 * 1024**3
    assert not hasattr(args, "seed")
    assert not hasattr(args, "revision")
    assert not hasattr(args, "eos_token_id")
    assert not hasattr(args, "retry")


def test_gate_4c_spec_round_trips_all_execution_identities() -> None:
    spec = _spec()

    restored = Gate4cSpec.from_value(spec.to_value())

    assert restored == spec
    assert restored.content_hash == spec.content_hash
    assert restored.initial_trainable_state.backbone_deployment_hash == (
        restored.qwen_deployment_hash
    )
    assert restored.production_config_hash == production_config_hash()


def test_gate_4c_spec_accepts_the_real_qwen_zero_eos_token() -> None:
    value = _spec().to_value()
    value["eos_token_id"] = 0

    restored = Gate4cSpec.from_value(value)

    assert restored.eos_token_id == 0


def test_gate_4c_backend_identity_normalizes_torch_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TorchVersion(str):
        pass

    monkeypatch.setattr(runtime_support.torch, "__version__", TorchVersion("2.13.0+cu129"))
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
    real_version = runtime_support.importlib.metadata.version
    monkeypatch.setattr(
        runtime_support.importlib.metadata,
        "version",
        lambda package: (
            "0.5.2"
            if package == runtime_support.QWEN35_GATED_DELTA_KERNEL_PACKAGE
            else real_version(package)
        ),
    )

    identity = runtime_support.backend_identity()

    assert type(identity.torch_version) is str
    assert identity.torch_version == "2.13.0+cu129"
    assert identity.flash_linear_attention_version == "0.5.2"


def test_gate_4c_spec_rejects_a_different_fixed_shape(tmp_path: Path) -> None:
    value = _spec().to_value()
    value["horizon"] = 14
    path = tmp_path / "spec.json"
    path.write_text(canonical_json(value), encoding="utf-8")

    with pytest.raises(ValueError):
        gate._read_spec(path)


def test_gate_4c_checkpoint_tree_identity_covers_every_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter.safetensors").write_bytes(b"adapter")
    nested = checkpoint / "z-head"
    nested.mkdir()
    (nested / "weights.safetensors").write_bytes(b"z")

    before = checkpoint_tree_hash(checkpoint)
    (nested / "weights.safetensors").write_bytes(b"changed")

    assert checkpoint_tree_hash(checkpoint) != before


def test_gate_4c_synthetic_text_is_built_text_first_near_the_fixed_cap() -> None:
    tokenizer = CharacterTokenizer()

    text = gate._near_target_text(tokenizer, target=768, label="action")

    assert 768 - 64 <= len(tokenizer.encode(text)) <= 768


def test_gate_4c_synthetic_text_builder_never_decodes_candidate_token_ids() -> None:
    class EncodeOnlyTokenizer(CharacterTokenizer):
        def decode(self, token_ids: tuple[int, ...]) -> str:
            raise AssertionError(f"unexpected synthetic decode: {token_ids}")

    text = gate._near_target_text(EncodeOnlyTokenizer(), target=128, label="action")

    assert 64 <= len(EncodeOnlyTokenizer().encode(text)) <= 128


def test_gate_4c_synthetic_record_is_phase_one_admissible() -> None:
    tokenizer = CharacterTokenizer()

    initial_text, record, _ = gate.build_gate_4c_synthetic_record(tokenizer)

    assert record.environment_id == gate.GATE_4C_ENVIRONMENT_ID
    assert record.task_family == gate.GATE_4C_TASK_FAMILY
    assert record.initial_context.meta == {
        "environment_id": gate.GATE_4C_ENVIRONMENT_ID,
        "task_family": gate.GATE_4C_TASK_FAMILY,
    }
    assert record.horizon == gate.HORIZON
    assert len(record.steps) == gate.HORIZON
    assert record.initial_context.assembled_token_count == len(tokenizer.encode(initial_text))


def test_gate_4c_synthetic_record_round_trips_exactly() -> None:
    _, record, _ = gate.build_gate_4c_synthetic_record(CharacterTokenizer())

    restored = TrajectoryRecord.from_value(record.to_value())

    assert restored == record
    assert restored.content_hash == record.content_hash


def test_gate_4c_all_synthetic_scoring_inputs_fit_fixed_model_cap() -> None:
    tokenizer = CharacterTokenizer()
    initial_text, record, measured = gate.build_gate_4c_synthetic_record(tokenizer)

    for step in record.steps:
        forward_tokens = len(
            tokenizer.encode(render_forward_prefix(initial_text, record.steps, step.index).text)
        )
        hindsight_tokens = len(
            tokenizer.encode(render_hindsight_prefix(initial_text, record.steps, step.index).text)
        )
        assert forward_tokens + step.action_token_count <= gate.MODEL_INPUT_CAP
        assert hindsight_tokens + step.action_token_count <= gate.MODEL_INPUT_CAP
    assert measured["forward_prefix_plus_action"] <= gate.MODEL_INPUT_CAP
    assert measured["hindsight_prefix_plus_action"] <= gate.MODEL_INPUT_CAP
