from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.snapshot_inputs import build_snapshot


def _row(index: int, *, split: str) -> dict[str, object]:
    return {
        "answer": f"private answer {index}",
        "context": [],
        "extra": {"metric": "accuracy", "source": f"source-{split}"},
        "question": f"private {split} question {index}",
        "task_type": "qa",
    }


def test_snapshot_is_path_free_and_answer_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"architectures": ["QwenForCausalLM"], "model_type": "qwen"}),
        encoding="utf-8",
    )
    (model / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.self_attn.q_proj.weight": "one.safetensors",
                    "model.layers.0.self_attn.v_proj.weight": "one.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    train = tmp_path / "train.json"
    validation = tmp_path / "validation.json"
    train.write_text(json.dumps([_row(0, split="train")]), encoding="utf-8")
    validation.write_text(json.dumps([_row(0, split="validation")]), encoding="utf-8")
    monkeypatch.setenv("MODEL_PATH", str(model))
    monkeypatch.setenv("MODEL_REVISION", "fixed-revision")
    monkeypatch.setenv("TRAIN_PATH", str(train))
    monkeypatch.setenv("VALIDATION_PATH", str(validation))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "expected_train_rows": 1,
                    "expected_validation_rows": 1,
                    "strict_data": True,
                    "train_path_env": "TRAIN_PATH",
                    "train_blake2b": hashlib.blake2b(
                        train.read_bytes(), digest_size=32
                    ).hexdigest(),
                    "validation_path_env": "VALIDATION_PATH",
                    "validation_blake2b": hashlib.blake2b(
                        validation.read_bytes(), digest_size=32
                    ).hexdigest(),
                },
                "model": {
                    "path_env": "MODEL_PATH",
                    "repo_id": "Qwen/test",
                    "revision_env": "MODEL_REVISION",
                },
                "training": {
                    "phi_target_modules": ["q_proj"],
                    "theta_target_modules": ["q_proj", "v_proj"],
                },
                "upstream_revision": "fixed-upstream",
            }
        ),
        encoding="utf-8",
    )

    snapshot = build_snapshot(config_path)
    encoded = json.dumps(snapshot)

    assert str(tmp_path) not in encoded
    assert "private answer" not in encoded
    assert snapshot["model"] == {
        "architectures": ["QwenForCausalLM"],
        "model_type": "qwen",
        "repo_id": "Qwen/test",
        "revision": "fixed-revision",
        "target_modules": {"phi": ["q_proj"], "theta": ["q_proj", "v_proj"]},
        "tensor_count": 2,
    }
