#!/usr/bin/env python3
"""Freeze the path-free Gate 4c @6 execution spec around one immutable bundle."""

from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
from pathlib import Path

from scripts.gate4c_runtime_support import (
    GATE_SEED,
    HORIZON,
    INITIAL_POLICY_DIRECTORY_NAME,
    INITIALIZATION_SEED,
    MAX_ACTION_TOKENS,
    MAX_REASONING_TOKENS,
    MAXIMUM_H0_TOKENS,
    MODEL_INPUT_CAP,
    PEAK_RESERVED_LIMIT_BYTES,
    PRIVATE_WHEEL_NAME,
    PUBLIC_WHEEL_NAME,
    SOURCE_ARCHIVE_NAME,
    production_config_hash,
)
from skillev.contracts import canonical_json
from skillev.experiments import (
    GATE_4C_CHILD_TERMINAL_FORMAT,
    GATE_4C_OPERATOR_TERMINAL_FORMAT,
    GATE_4C_STAGE_EVENT_FORMAT,
    DeterministicBackendIdentity,
    ExecutionHardwareIdentity,
    Gate4cSpec,
    checkpoint_tree_hash,
)
from skillev.experiments.build_identity import (
    read_source_package_provenance,
    sha256_file,
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
from skillev.policy.checkpoint import inspect_policy_checkpoint

GPU_SAMPLING_INTERVAL_MS = 250


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-directory", required=True)
    parser.add_argument("--base-model-artifact", required=True)
    parser.add_argument("--tokenizer-artifact", required=True)
    parser.add_argument("--initial-trainable-state", required=True)
    parser.add_argument("--hardware-identity", required=True)
    parser.add_argument("--deterministic-backend-identity", required=True)
    parser.add_argument("--eos-token-id", type=int, required=True)
    parser.add_argument("--expected-physical-gpu-uuid", required=True)
    return parser.parse_args()


def _json(path: str) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _source_provenance(archive_path: Path):
    with tempfile.TemporaryDirectory(prefix="skillev-gate-4c-spec-") as temporary:
        root = Path(temporary)
        with tarfile.open(archive_path, mode="r:*") as archive:
            archive.extractall(root, filter="data")
        return read_source_package_provenance(root)


def main() -> None:
    args = _args()
    bundle = Path(args.bundle_directory).resolve()
    output = bundle / "gate-4c-spec.json"
    if output.exists():
        raise FileExistsError(output)
    source_archive = bundle / SOURCE_ARCHIVE_NAME
    public_wheel = bundle / PUBLIC_WHEEL_NAME
    private_wheel = bundle / PRIVATE_WHEEL_NAME
    initial_policy = bundle / INITIAL_POLICY_DIRECTORY_NAME
    source = _source_provenance(source_archive)
    base_model = BaseModelArtifactIdentity.from_value(_json(args.base_model_artifact))
    tokenizer = TokenizerArtifactIdentity.from_value(_json(args.tokenizer_artifact))
    trainable = TrainableStateIdentity.from_value(_json(args.initial_trainable_state))
    spec = Gate4cSpec(
        source_commit=source.source_commit,
        source_tree_hash=source.source_tree_hash,
        source_archive_sha256=sha256_file(source_archive),
        public_wheel_sha256=sha256_file(public_wheel),
        private_wheel_sha256=sha256_file(private_wheel),
        lockfile_sha256=sha256_file(Path(__file__).resolve().parents[1] / "uv.lock"),
        base_model_artifact=base_model,
        tokenizer_artifact=tokenizer,
        qwen_deployment_hash=trainable.backbone_deployment_hash,
        initial_trainable_state=trainable,
        initial_checkpoint_tree_hash=checkpoint_tree_hash(initial_policy),
        initial_checkpoint_inspection_hash=inspect_policy_checkpoint(initial_policy).content_hash,
        expected_hardware_identity=ExecutionHardwareIdentity.from_value(
            _json(args.hardware_identity)
        ),
        deterministic_backend_identity=DeterministicBackendIdentity.from_value(
            _json(args.deterministic_backend_identity)
        ),
        gate_procedure_version=GATE_4C_PROCEDURE_VERSION,
        operator_version=GATE_4C_OPERATOR_VERSION,
        stage_journal_version=GATE_4C_STAGE_EVENT_FORMAT,
        child_terminal_version=GATE_4C_CHILD_TERMINAL_FORMAT,
        operator_terminal_version=GATE_4C_OPERATOR_TERMINAL_FORMAT,
        gpu_sampling_interval_ms=GPU_SAMPLING_INTERVAL_MS,
        expected_physical_gpu_uuid=args.expected_physical_gpu_uuid,
        production_config_hash=production_config_hash(),
        seed=GATE_SEED,
        initialization_seed=INITIALIZATION_SEED,
        horizon=HORIZON,
        maximum_h0_tokens=MAXIMUM_H0_TOKENS,
        max_reasoning_tokens=MAX_REASONING_TOKENS,
        max_action_tokens=MAX_ACTION_TOKENS,
        model_input_cap=MODEL_INPUT_CAP,
        peak_reserved_cap_bytes=PEAK_RESERVED_LIMIT_BYTES,
        eos_token_id=args.eos_token_id,
    )
    bundle.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(spec.to_value()) + "\n", encoding="utf-8")
    print(spec.content_hash)


if __name__ == "__main__":
    main()
