from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from skillev_private.experiments.protocol_v10_launch_package import (
    PROTOCOL_V10_RESUME_POLICY,
    ProtocolV10LaunchPackage,
    ProtocolV10LaunchSlot,
)

from skillev.experiments import FormalApplicationV10, FormalMethodV10
from skillev.runtime.gpu_topology import ThreeGPURolePolicy


def _digest(path: Path) -> str:
    return f"blake2b:{hashlib.blake2b(path.read_bytes(), digest_size=32).hexdigest()}"


def _package(tmp_path: Path) -> ProtocolV10LaunchPackage:
    methods = tuple(FormalMethodV10)
    applications = (
        FormalApplicationV10.EXACT_SKILLFLOW_BASELINE,
        FormalApplicationV10.BAYESIAN_IMPROVE_FULL,
        FormalApplicationV10.BAYESIAN_IMPROVE_NO_CALIBRATION,
    )
    slots: list[ProtocolV10LaunchSlot] = []
    for index, (method, application) in enumerate(zip(methods, applications, strict=True)):
        exact = (tmp_path / f"exact-{index}.json").resolve()
        exact.write_text(f'{{"method":"{method.value}"}}\n', encoding="utf-8")
        attempt = (tmp_path / f"attempt-{index}").resolve()
        slots.append(
            ProtocolV10LaunchSlot(
                method=method,
                application=application,
                attempt_id=attempt.name,
                exact_input_path=exact,
                exact_input_digest=_digest(exact),
                attempt_root=attempt,
                checkpoint_root=attempt / "checkpoints",
                adapter_namespace=f"protocol-v10-formal-{index}",
                method_contract={"application": application.value},
                resume_policy=PROTOCOL_V10_RESUME_POLICY,
            )
        )
    return ProtocolV10LaunchPackage(
        run_group_id="protocol-v10-formal-seed-0",
        source_commit="a" * 40,
        protocol_identity="protocol-v10",
        formal_experiment_identity="formal-experiment-v1",
        model_revision="qwen-revision",
        tokenizer_identity="tokenizer-revision",
        initial_checkpoint_identity="initial-trainable-state",
        initial_skill_library_identity="initial-skill-library",
        training_catalog_identity="catalog-v10",
        ordered_episode_sequence_identity="ordered-4608-seed-0",
        sglang_endpoint="http://127.0.0.1:30000",
        sglang_model_identity="qwen-base",
        hardware=ThreeGPURolePolicy(
            inference_physical_index=5,
            coordinator_physical_index=4,
            gradient_primary_physical_index=6,
            forbidden_physical_indices=(0, 1, 2, 3, 7),
            minimum_free_memory_mib=70_000,
        ),
        slots=tuple(slots),
    )


def test_protocol_v10_launch_package_round_trips_and_verifies_inputs(tmp_path: Path) -> None:
    package = _package(tmp_path)
    output = (tmp_path / "launch-package.json").resolve()

    package.require_exact_inputs_unchanged()
    package.write_once(output)

    assert ProtocolV10LaunchPackage.read(output) == package
    with pytest.raises(FileExistsError):
        package.write_once(output)


def test_protocol_v10_launch_package_detects_changed_exact_input(tmp_path: Path) -> None:
    package = _package(tmp_path)
    package.slots[0].exact_input_path.write_text("changed\n", encoding="utf-8")

    with pytest.raises(RuntimeError):
        package.require_exact_inputs_unchanged()


def test_protocol_v10_launch_package_rejects_reordered_or_shared_slots(tmp_path: Path) -> None:
    package = _package(tmp_path)

    with pytest.raises(ValueError):
        replace(package, slots=tuple(reversed(package.slots)))
    with pytest.raises(ValueError):
        replace(
            package,
            slots=(
                package.slots[0],
                replace(
                    package.slots[1],
                    adapter_namespace=package.slots[0].adapter_namespace,
                ),
                package.slots[2],
            ),
        )
