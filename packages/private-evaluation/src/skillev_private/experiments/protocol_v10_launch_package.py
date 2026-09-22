"""Result-blind immutable launch package for the three Protocol 10 methods."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Self

from skillev.contracts import JsonValue, normalize_json, stable_hash
from skillev.experiments import FormalApplicationV10, FormalMethodV10
from skillev.experiments.protocol_v10 import load_active_protocol_v10
from skillev.experiments.protocol_v10_formal import load_protocol_v10_formal_experiment
from skillev.runtime.gpu_topology import ThreeGPURolePolicy
from skillev_private.benchmarks.protocol_v10_population import ProtocolV10TrainingSelection

from .protocol_v10_application_input import ProtocolV10ApplicationInput
from .protocol_v10_attempt_input import ProtocolV10AdmissionMode, ProtocolV10AttemptInput

PROTOCOL_V10_LAUNCH_PACKAGE_FORMAT: Final = "skillev-private-protocol-v10-launch-package@3"
PROTOCOL_V10_RESUME_POLICY: Final = "explicit-committed-snapshot-only-no-automatic-retry"


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be non-empty text")
    return value


def _absolute(value: object, *, label: str) -> Path:
    path = Path(_text(value, label=label))
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    return path


def _file_digest(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=32)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"blake2b:{digest.hexdigest()}"


def _digest(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    if not re.fullmatch(r"blake2b:[0-9a-f]{64}", text):
        raise ValueError(f"{label} must be a BLAKE2b-256 digest")
    return text


@dataclass(frozen=True, slots=True)
class ProtocolV10LaunchSlot:
    method: FormalMethodV10
    application: FormalApplicationV10
    attempt_id: str
    exact_input_path: Path
    exact_input_digest: str
    attempt_root: Path
    checkpoint_root: Path
    adapter_namespace: str
    method_contract: JsonValue
    resume_policy: str = PROTOCOL_V10_RESUME_POLICY

    def __post_init__(self) -> None:
        if not isinstance(self.method, FormalMethodV10):
            raise TypeError("Protocol 10 launch slot requires a formal method")
        if not isinstance(self.application, FormalApplicationV10):
            raise TypeError("Protocol 10 launch slot requires a formal application")
        _text(self.attempt_id, label="attempt_id")
        if any(not path.is_absolute() for path in (self.exact_input_path, self.attempt_root)):
            raise ValueError("Protocol 10 launch paths must be absolute")
        if not self.checkpoint_root.is_absolute() or not self.checkpoint_root.is_relative_to(
            self.attempt_root
        ):
            raise ValueError("Protocol 10 checkpoints must remain inside their attempt root")
        if self.attempt_id != self.attempt_root.name:
            raise ValueError("Protocol 10 attempt ID must equal its private root name")
        _digest(self.exact_input_digest, label="exact_input_digest")
        _text(self.adapter_namespace, label="adapter_namespace")
        if self.resume_policy != PROTOCOL_V10_RESUME_POLICY:
            raise ValueError("Protocol 10 resume policy cannot be weakened")
        object.__setattr__(self, "method_contract", normalize_json(self.method_contract))

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "adapter_namespace": self.adapter_namespace,
            "application": self.application.value,
            "attempt_id": self.attempt_id,
            "attempt_root": str(self.attempt_root),
            "checkpoint_root": str(self.checkpoint_root),
            "exact_input_digest": self.exact_input_digest,
            "exact_input_path": str(self.exact_input_path),
            "method": self.method.value,
            "method_contract": self.method_contract,
            "resume_policy": self.resume_policy,
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        normalized = normalize_json(value)
        fields = {
            "adapter_namespace",
            "application",
            "attempt_id",
            "attempt_root",
            "checkpoint_root",
            "exact_input_digest",
            "exact_input_path",
            "method",
            "method_contract",
            "resume_policy",
        }
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("Protocol 10 launch slot has incompatible fields")
        return cls(
            method=FormalMethodV10(_text(normalized["method"], label="method")),
            application=FormalApplicationV10(_text(normalized["application"], label="application")),
            attempt_id=_text(normalized["attempt_id"], label="attempt_id"),
            exact_input_path=_absolute(normalized["exact_input_path"], label="exact_input_path"),
            exact_input_digest=_digest(
                normalized["exact_input_digest"], label="exact_input_digest"
            ),
            attempt_root=_absolute(normalized["attempt_root"], label="attempt_root"),
            checkpoint_root=_absolute(normalized["checkpoint_root"], label="checkpoint_root"),
            adapter_namespace=_text(normalized["adapter_namespace"], label="adapter_namespace"),
            method_contract=normalized["method_contract"],
            resume_policy=_text(normalized["resume_policy"], label="resume_policy"),
        )


@dataclass(frozen=True, slots=True)
class ProtocolV10LaunchPackage:
    run_group_id: str
    source_commit: str
    protocol_identity: str
    formal_experiment_identity: str
    model_revision: str
    tokenizer_identity: str
    initial_checkpoint_identity: str
    initial_skill_library_identity: str
    training_catalog_identity: str
    ordered_episode_sequence_identity: str
    sglang_endpoint: str
    sglang_model_identity: str
    hardware: ThreeGPURolePolicy
    slots: tuple[ProtocolV10LaunchSlot, ...]
    seed: int = 0
    total_steps_per_method: int = 288
    total_episodes_per_method: int = 4_608
    format: str = PROTOCOL_V10_LAUNCH_PACKAGE_FORMAT

    def __post_init__(self) -> None:
        _text(self.run_group_id, label="run_group_id")
        if not re.fullmatch(r"[0-9a-f]{40}", self.source_commit):
            raise ValueError("source_commit must be a full lowercase Git commit")
        for field in (
            "protocol_identity",
            "formal_experiment_identity",
            "model_revision",
            "tokenizer_identity",
            "initial_checkpoint_identity",
            "initial_skill_library_identity",
            "training_catalog_identity",
            "ordered_episode_sequence_identity",
            "sglang_endpoint",
            "sglang_model_identity",
        ):
            _text(getattr(self, field), label=field)
        if self.seed != 0 or self.total_steps_per_method != 288:
            raise ValueError("Protocol 10 launch package has another training shape")
        if self.total_episodes_per_method != 4_608:
            raise ValueError("Protocol 10 launch package must consume 4,608 episodes per method")
        if not isinstance(self.hardware, ThreeGPURolePolicy):
            raise TypeError("Protocol 10 launch package requires a three-role GPU policy")
        if self.format != PROTOCOL_V10_LAUNCH_PACKAGE_FORMAT:
            raise ValueError("unsupported Protocol 10 launch package format")
        if tuple(slot.method for slot in self.slots) != tuple(FormalMethodV10):
            raise ValueError("Protocol 10 launch slots have another method order")
        expected_applications = (
            FormalApplicationV10.EXACT_SKILLFLOW_BASELINE,
            FormalApplicationV10.BAYESIAN_IMPROVE_FULL,
            FormalApplicationV10.BAYESIAN_IMPROVE_NO_CALIBRATION,
        )
        if tuple(slot.application for slot in self.slots) != expected_applications:
            raise ValueError("Protocol 10 launch slots have another application order")
        for field in ("attempt_id", "attempt_root", "checkpoint_root", "adapter_namespace"):
            values = tuple(getattr(slot, field) for slot in self.slots)
            if len(set(values)) != len(values):
                raise ValueError(f"Protocol 10 launch slots repeat {field}")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "formal_experiment_identity": self.formal_experiment_identity,
            "format": self.format,
            "hardware": normalize_json(self.hardware.to_value()),
            "initial_checkpoint_identity": self.initial_checkpoint_identity,
            "initial_skill_library_identity": self.initial_skill_library_identity,
            "model_revision": self.model_revision,
            "ordered_episode_sequence_identity": self.ordered_episode_sequence_identity,
            "protocol_identity": self.protocol_identity,
            "run_group_id": self.run_group_id,
            "seed": self.seed,
            "sglang_endpoint": self.sglang_endpoint,
            "sglang_model_identity": self.sglang_model_identity,
            "slots": [slot.to_value() for slot in self.slots],
            "source_commit": self.source_commit,
            "tokenizer_identity": self.tokenizer_identity,
            "total_episodes_per_method": self.total_episodes_per_method,
            "total_steps_per_method": self.total_steps_per_method,
            "training_catalog_identity": self.training_catalog_identity,
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        normalized = normalize_json(value)
        fields = {
            "formal_experiment_identity",
            "format",
            "hardware",
            "initial_checkpoint_identity",
            "initial_skill_library_identity",
            "model_revision",
            "ordered_episode_sequence_identity",
            "protocol_identity",
            "run_group_id",
            "seed",
            "sglang_endpoint",
            "sglang_model_identity",
            "slots",
            "source_commit",
            "tokenizer_identity",
            "total_episodes_per_method",
            "total_steps_per_method",
            "training_catalog_identity",
        }
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("Protocol 10 launch package has incompatible fields")
        slots = normalized["slots"]
        if not isinstance(slots, list):
            raise TypeError("Protocol 10 launch slots must be an array")
        integer_fields = ("seed", "total_episodes_per_method", "total_steps_per_method")
        if any(type(normalized[field]) is not int for field in integer_fields):
            raise TypeError("Protocol 10 launch quantities must be integers")
        text_fields = fields - {"hardware", "seed", "slots", *integer_fields[1:]}
        if any(type(normalized[field]) is not str for field in text_fields):
            raise TypeError("Protocol 10 launch identities must be text")
        return cls(
            run_group_id=normalized["run_group_id"],
            source_commit=normalized["source_commit"],
            protocol_identity=normalized["protocol_identity"],
            formal_experiment_identity=normalized["formal_experiment_identity"],
            model_revision=normalized["model_revision"],
            tokenizer_identity=normalized["tokenizer_identity"],
            initial_checkpoint_identity=normalized["initial_checkpoint_identity"],
            initial_skill_library_identity=normalized["initial_skill_library_identity"],
            training_catalog_identity=normalized["training_catalog_identity"],
            ordered_episode_sequence_identity=normalized["ordered_episode_sequence_identity"],
            sglang_endpoint=normalized["sglang_endpoint"],
            sglang_model_identity=normalized["sglang_model_identity"],
            hardware=ThreeGPURolePolicy.from_mapping(normalized["hardware"]),
            slots=tuple(ProtocolV10LaunchSlot.from_value(slot) for slot in slots),
            seed=normalized["seed"],
            total_steps_per_method=normalized["total_steps_per_method"],
            total_episodes_per_method=normalized["total_episodes_per_method"],
            format=normalized["format"],
        )

    @classmethod
    def read(cls, path: Path) -> Self:
        return cls.from_value(json.loads(path.read_text(encoding="utf-8")))

    def require_exact_inputs_unchanged(self) -> None:
        for slot in self.slots:
            if _file_digest(slot.exact_input_path) != slot.exact_input_digest:
                raise RuntimeError("Protocol 10 exact input changed after launch freeze")

    def write_once(self, path: Path) -> None:
        if not path.is_absolute():
            raise ValueError("Protocol 10 launch package output must be absolute")
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        encoded = (
            json.dumps(self.to_value(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            + "\n"
        ).encode()
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def build_protocol_v10_launch_package(
    *,
    exact_input_paths: tuple[Path, ...],
    run_root: Path,
    source_commit: str,
) -> ProtocolV10LaunchPackage:
    if len(exact_input_paths) != len(FormalMethodV10):
        raise ValueError("Protocol 10 launch freeze requires three exact inputs")
    if not run_root.is_absolute() or not run_root.is_dir():
        raise ValueError("Protocol 10 run root must be an existing absolute directory")
    exact_inputs = tuple(ProtocolV10AttemptInput.read(path) for path in exact_input_paths)
    if tuple(exact.method for exact in exact_inputs) != tuple(FormalMethodV10):
        raise ValueError("Protocol 10 exact inputs have another method order")
    if any(
        exact.admission_mode is not ProtocolV10AdmissionMode.FORMAL
        or exact.integration_smoke_steps is not None
        or exact.resume_snapshot is not None
        for exact in exact_inputs
    ):
        raise ValueError("Protocol 10 launch package accepts only fresh formal inputs")
    first = exact_inputs[0]
    shared_paths = ("protocol_config", "experiment_config", "catalog_root", "session_deployments")
    for field in shared_paths:
        expected = getattr(first, field).resolve()
        if any(getattr(exact, field).resolve() != expected for exact in exact_inputs[1:]):
            raise ValueError(f"Protocol 10 methods do not share {field}")
    if any(exact.hardware != first.hardware for exact in exact_inputs[1:]):
        raise ValueError("Protocol 10 methods do not share one GPU role policy")
    if any(exact.sglang.workflow != first.sglang.workflow for exact in exact_inputs[1:]):
        raise ValueError("Protocol 10 methods do not share one workflow budget")
    if any(
        exact.sglang.gateway.endpoint_base != first.sglang.gateway.endpoint_base
        or exact.sglang.gateway.base_model != first.sglang.gateway.base_model
        for exact in exact_inputs[1:]
    ):
        raise ValueError("Protocol 10 methods do not share one SGLang base deployment")
    if any(exact.attempt_root.parent.resolve() != run_root.resolve() for exact in exact_inputs):
        raise ValueError("Protocol 10 attempt roots must be direct children of the run root")
    if any(exact.attempt_root.exists() for exact in exact_inputs):
        raise FileExistsError("a Protocol 10 formal attempt root already exists")

    protocol = load_active_protocol_v10(first.protocol_config)
    experiment = load_protocol_v10_formal_experiment(first.experiment_config)
    experiment.require_execution_ready(protocol)
    selection_path = first.catalog_root / "training-selection.json"
    selection = ProtocolV10TrainingSelection.read(selection_path)
    applications = tuple(
        ProtocolV10ApplicationInput.read(exact.application_input) for exact in exact_inputs
    )
    for exact, application in zip(exact_inputs, applications, strict=True):
        application.require_method(exact.method)
        application.identity(method=exact.method, experiment=experiment)
        if application.application.trainer.rollout.base_seed != experiment.seed:
            raise ValueError("Protocol 10 application rollout seed differs from seed 0")
        checkpoint_root = Path(application.checkpoint_storage.directory).resolve()
        if not checkpoint_root.is_relative_to(exact.attempt_root.resolve()):
            raise ValueError("Protocol 10 checkpoint root escaped its method attempt")
    first_application = applications[0]
    backbone_identity = (
        first_application.backbone.revision,
        first_application.backbone.tokenizer_id,
        first_application.backbone.tokenizer_content_hash,
    )
    if any(
        (
            application.backbone.revision,
            application.backbone.tokenizer_id,
            application.backbone.tokenizer_content_hash,
        )
        != backbone_identity
        for application in applications[1:]
    ):
        raise ValueError("Protocol 10 methods do not share one model/tokenizer identity")
    initial_identity = first_application.initial_checkpoint.trainable_state.content_hash
    if any(
        application.initial_checkpoint.trainable_state.content_hash != initial_identity
        for application in applications[1:]
    ):
        raise ValueError("Protocol 10 methods do not share one initial checkpoint")
    sequence_identity = first_application.snapshot_identity.ordered_task_sequence_hash
    if any(
        application.snapshot_identity.ordered_task_sequence_hash != sequence_identity
        for application in applications[1:]
    ):
        raise ValueError("Protocol 10 methods do not share one ordered task sequence")
    initial_library_identity = first_application.snapshot_identity.initial_skill_library_state_hash
    if any(
        application.snapshot_identity.initial_skill_library_state_hash != initial_library_identity
        for application in applications[1:]
    ):
        raise ValueError("Protocol 10 methods do not share one initial skill library")
    if any(exact.storage.budget != first.storage.budget for exact in exact_inputs[1:]):
        raise ValueError("Protocol 10 methods do not share one measured storage budget")
    if len({exact.sglang.adapter_namespace for exact in exact_inputs}) != len(exact_inputs):
        raise ValueError("Protocol 10 methods require distinct adapter namespaces")

    slots = tuple(
        ProtocolV10LaunchSlot(
            method=exact.method,
            application=experiment.binding(exact.method).application,
            attempt_id=exact.attempt_root.name,
            exact_input_path=path,
            exact_input_digest=_file_digest(path),
            attempt_root=exact.attempt_root,
            checkpoint_root=Path(application.checkpoint_storage.directory).resolve(),
            adapter_namespace=exact.sglang.adapter_namespace,
            method_contract=application.method_contract.to_value(),
        )
        for path, exact, application in zip(
            exact_input_paths,
            exact_inputs,
            applications,
            strict=True,
        )
    )
    catalog_summary = json.loads((first.catalog_root / "catalog.json").read_text(encoding="utf-8"))
    return ProtocolV10LaunchPackage(
        run_group_id=run_root.name,
        source_commit=source_commit,
        protocol_identity=_file_digest(first.protocol_config),
        formal_experiment_identity=_file_digest(first.experiment_config),
        model_revision=first_application.backbone.revision,
        tokenizer_identity=stable_hash(
            {
                "content": first_application.backbone.tokenizer_content_hash,
                "id": first_application.backbone.tokenizer_id,
            }
        ),
        initial_checkpoint_identity=initial_identity,
        initial_skill_library_identity=initial_library_identity,
        training_catalog_identity=stable_hash(catalog_summary),
        ordered_episode_sequence_identity=stable_hash(selection.to_value()),
        sglang_endpoint=first.sglang.gateway.endpoint_base,
        sglang_model_identity=first.sglang.gateway.base_model,
        hardware=first.hardware,
        slots=slots,
        seed=experiment.seed,
        total_steps_per_method=experiment.total_steps,
        total_episodes_per_method=experiment.total_episodes,
    )


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(  # noqa: S603 - closed Git command and caller-supplied repository
        ["/usr/bin/git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--exact-input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    if _git(repository, "status", "--porcelain"):
        raise RuntimeError("Protocol 10 launch freeze requires a clean Git worktree")
    source_commit = _git(repository, "rev-parse", "HEAD")
    package = build_protocol_v10_launch_package(
        exact_input_paths=tuple(path.resolve() for path in arguments.exact_input),
        run_root=arguments.run_root.resolve(),
        source_commit=source_commit,
    )
    output = arguments.output.resolve()
    package.write_once(output)
    print(
        json.dumps(
            {
                "method_order": [slot.method.value for slot in package.slots],
                "output": str(output),
                "run_group_id": package.run_group_id,
                "status": "frozen",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "PROTOCOL_V10_LAUNCH_PACKAGE_FORMAT",
    "PROTOCOL_V10_RESUME_POLICY",
    "ProtocolV10LaunchPackage",
    "ProtocolV10LaunchSlot",
    "build_protocol_v10_launch_package",
]
