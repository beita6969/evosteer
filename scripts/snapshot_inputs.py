"""Resolve model/data inputs and record an answer-free, path-free snapshot manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import cast

import yaml

from scripts.validate_data import (
    _duplicate_policy,
    _load_records,
    _validated_path,
    reject_cross_split_overlap,
    validate_records,
)


def _load_config(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("configuration must be an object")
    baseline = value.get("baseline_config")
    if baseline is not None:
        if not isinstance(baseline, str) or not baseline:
            raise TypeError("baseline_config must be non-empty text")
        value = yaml.safe_load(Path(baseline).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("baseline configuration must be an object")
    return cast(dict[str, object], value)


def _required_environment(name: object) -> str:
    if not isinstance(name, str) or not name:
        raise TypeError("environment key must be non-empty text")
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise RuntimeError(f"required environment variable is unset: {name}")
    return value


def _model_weight_names(model_path: Path) -> tuple[str, ...]:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError("model.safetensors.index.json is required for target-module audit")
    value = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("weight_map"), dict):
        raise ValueError("model tensor index is malformed")
    weight_map = cast(dict[object, object], value["weight_map"])
    if any(not isinstance(name, str) for name in weight_map):
        raise ValueError("model tensor index contains a non-text tensor name")
    return tuple(sorted(cast(str, name) for name in weight_map))


def _audit_targets(training: dict[str, object], weight_names: tuple[str, ...]) -> dict[str, object]:
    result: dict[str, object] = {}
    for role in ("theta", "phi"):
        configured = training.get(f"{role}_target_modules")
        if (
            not isinstance(configured, list)
            or not configured
            or any(not isinstance(item, str) or not item for item in configured)
        ):
            raise ValueError(f"{role} target modules must be a non-empty text list")
        targets = cast(list[str], configured)
        missing = [
            target
            for target in targets
            if not any(name.endswith(f".{target}.weight") for name in weight_names)
        ]
        if missing:
            raise ValueError(f"{role} target modules are absent from the fixed model: {missing}")
        result[role] = targets
    return result


def build_snapshot(config_path: Path) -> dict[str, object]:
    config = _load_config(config_path)
    model = config.get("model")
    data = config.get("data")
    training = config.get("training")
    if not isinstance(model, dict) or not isinstance(data, dict) or not isinstance(training, dict):
        raise ValueError("baseline config requires model, data, and training objects")
    model_path = Path(_required_environment(model.get("path_env", "SKILLEV_MODEL_PATH"))).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError("resolved model path is not a directory")
    revision = _required_environment(model.get("revision_env"))
    model_config_path = model_path / "config.json"
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    if not isinstance(model_config, dict):
        raise TypeError("model config must be an object")
    architecture = model_config.get("architectures")
    if not isinstance(architecture, list) or any(
        not isinstance(item, str) for item in architecture
    ):
        raise ValueError("model architecture list is malformed")
    weights = _model_weight_names(model_path)

    train = _load_records(_validated_path(data, split="train"))
    validation = _load_records(_validated_path(data, split="validation"))
    expected_train = data.get("expected_train_rows")
    expected_validation = data.get("expected_validation_rows")
    if type(expected_train) is not int or type(expected_validation) is not int:
        raise TypeError("expected dataset counts must be integers")
    train_report = validate_records(
        train,
        split="train",
        expected_count=expected_train,
        allowed_duplicate_extra_rows=_duplicate_policy(data, "train"),
    )
    validation_report = validate_records(
        validation,
        split="validation",
        expected_count=expected_validation,
        allowed_duplicate_extra_rows=_duplicate_policy(data, "validation"),
    )
    reject_cross_split_overlap(train, validation)

    return {
        "data": {
            "train": {**train_report.to_value(), "blake2b": data["train_blake2b"]},
            "validation": {
                **validation_report.to_value(),
                "blake2b": data["validation_blake2b"],
            },
        },
        "format": "skillev-input-snapshot@1",
        "model": {
            "architectures": architecture,
            "model_type": model_config.get("model_type"),
            "repo_id": model.get("repo_id"),
            "revision": revision,
            "target_modules": _audit_targets(cast(dict[str, object], training), weights),
            "tensor_count": len(weights),
        },
        "source_config": config_path.name,
        "upstream_revision": config.get("upstream_revision"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline/paper_v1_250step.yaml")
    parser.add_argument("--output", default="manifests/input_snapshot.json")
    arguments = parser.parse_args()
    output = Path(arguments.output)
    if output.exists():
        raise FileExistsError(output)
    snapshot = build_snapshot(Path(arguments.config))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"format": snapshot["format"], "output": str(output)}))


if __name__ == "__main__":
    main()
