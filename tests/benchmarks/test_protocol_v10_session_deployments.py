from __future__ import annotations

import json
from pathlib import Path

import pytest
from skillev_private.benchmarks.protocol_v10_session_deployments import (
    PROTOCOL_V10_SESSION_DEPLOYMENTS_FORMAT,
    ProtocolV10SessionDeployments,
)
from skillev_private.benchmarks.protocol_v13_session_deployments import (
    Protocol13TrainingDeployments,
)


def _file(root: Path, name: str) -> str:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return str(path.resolve())


def _directory(root: Path, name: str) -> str:
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return str(path.resolve())


def _value(root: Path) -> dict[str, object]:
    python = _file(root, "bin/python")
    return {
        "format": PROTOCOL_V10_SESSION_DEPLOYMENTS_FORMAT,
        "webshop": {
            "goals": _file(root, "webshop/goals.jsonl"),
            "interpreter": python,
            "product_store": _file(root, "webshop/products.sqlite"),
            "search_index": _directory(root, "webshop/index"),
            "source_revision": "official-revision",
            "source_root": _directory(root, "webshop/source"),
            "timeout_seconds": 180,
        },
        "alfworld": {
            "config_path": _file(root, "alfworld/config.yaml"),
            "dataset_root": _directory(root, "alfworld/data"),
            "interpreter": python,
            "source_revision": "official-revision",
            "source_root": _directory(root, "alfworld/source"),
            "timeout_seconds": 180,
        },
        "appworld": {
            "interpreter": python,
            "source_root": _directory(root, "appworld/source"),
            "state_root": _directory(root, "appworld/state"),
            "timeout_seconds": 300,
        },
        "healthbench": {
            "interpreter": python,
            "model_revision": "Qwen/Qwen3.5-9B@frozen",
            "request_timeout_seconds": 90,
            "sglang_version": "0.5.3",
            "source_root": _directory(root, "healthbench/simple-evals"),
            "tokenizer_revision": "Qwen/Qwen3.5-9B@frozen",
            "worker_timeout_seconds": 900,
        },
        "spreadsheetbench": {
            "bubblewrap": _file(root, "tools/bwrap"),
            "edit_timeout_seconds": 60,
            "extraction_cache_root": _directory(root, "spreadsheet/extracted"),
            "interpreter": python,
            "libreoffice": _file(root, "tools/soffice"),
            "oj_timeout_seconds": 120,
            "prlimit": _file(root, "tools/prlimit"),
            "python_environment": _directory(root, "spreadsheet/python"),
            "python_package_roots": [_directory(root, "spreadsheet/site-packages")],
            "runtime_readonly_paths": [_directory(root, "spreadsheet/runtime")],
            "source_root": _directory(root, "spreadsheet/source"),
            "temporary_root": _directory(root, "spreadsheet/temporary"),
            "training_archive": _file(root, "spreadsheet/training.zip"),
            "verified_root": _directory(root, "spreadsheet/verified"),
            "workspace_root": _directory(root, "spreadsheet/workspaces"),
        },
    }


def test_protocol_v10_session_deployment_reads_all_nine_runtime_routes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session-deployments.json"
    path.write_text(json.dumps(_value(tmp_path)), encoding="utf-8")

    deployment = ProtocolV10SessionDeployments.read(path.resolve())

    assert deployment.alfworld.dataset_root.is_dir()
    assert deployment.appworld.timeout_seconds == 300
    assert deployment.healthbench.model_revision == "Qwen/Qwen3.5-9B@frozen"
    assert deployment.spreadsheetbench.python_package_roots[0].is_dir()


def test_protocol_v10_session_deployment_rejects_missing_runtime_route(
    tmp_path: Path,
) -> None:
    value = _value(tmp_path)
    del value["appworld"]

    with pytest.raises(ValueError):
        ProtocolV10SessionDeployments.from_value(value)


def test_protocol13_does_not_require_unused_legacy_deployments(tmp_path: Path) -> None:
    value = _value(tmp_path)
    value["appworld"] = {"source_root": str(tmp_path / "not-deployed")}
    del value["spreadsheetbench"]
    del value["webshop"]
    path = tmp_path / "training-deployments.json"
    path.write_text(json.dumps(value))

    deployment = Protocol13TrainingDeployments.read(path)

    assert deployment.alfworld.dataset_root.is_dir()
    assert deployment.healthbench.model_revision == "Qwen/Qwen3.5-9B@frozen"


@pytest.mark.parametrize("required", ["alfworld", "healthbench"])
def test_protocol13_still_requires_every_used_deployment(tmp_path: Path, required: str) -> None:
    value = _value(tmp_path)
    del value[required]
    path = tmp_path / "training-deployments.json"
    path.write_text(json.dumps(value))
    with pytest.raises((TypeError, ValueError)):
        Protocol13TrainingDeployments.read(path)


@pytest.mark.parametrize("budget", [1024, 4096])
def test_protocol13_declares_healthbench_repair_budget(tmp_path: Path, budget: int) -> None:
    value = _value(tmp_path)
    path = tmp_path / "deployments.json"
    path.write_text(json.dumps(value))
    assert Protocol13TrainingDeployments.read(path).healthbench_repair_max_output_tokens == 1024
    value["healthbench_repair_max_output_tokens"] = budget
    path.write_text(json.dumps(value))
    assert Protocol13TrainingDeployments.read(path).healthbench_repair_max_output_tokens == budget


def test_protocol13_still_validates_used_assets(tmp_path: Path) -> None:
    value = _value(tmp_path)
    Path(value["alfworld"]["config_path"]).unlink()
    path = tmp_path / "training-deployments.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        Protocol13TrainingDeployments.read(path)
