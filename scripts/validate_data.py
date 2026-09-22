"""Fail closed on missing or malformed official SkillFlow train/IID data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml


@dataclass(frozen=True, slots=True)
class DatasetValidationReport:
    split: str
    row_count: int
    task_types: Mapping[str, int]
    sources: Mapping[str, int]
    duplicate_extra_rows: Mapping[str, int]

    def to_value(self) -> dict[str, object]:
        return {
            "duplicate_extra_rows": dict(sorted(self.duplicate_extra_rows.items())),
            "row_count": self.row_count,
            "sources": dict(sorted(self.sources.items())),
            "split": self.split,
            "task_types": dict(sorted(self.task_types.items())),
        }


def _load_records(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"required official data file is missing: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise TypeError("dataset must be a JSON array or JSONL stream of objects")
    return cast(list[dict[str, object]], value)


def _blake2b(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=32)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_path(data: Mapping[str, object], *, split: str) -> Path:
    path = _resolve_environment_path(data.get(f"{split}_path_env"))
    expected = data.get(f"{split}_blake2b")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{split} data identity must be a 64-character BLAKE2b digest")
    if _blake2b(path) != expected:
        raise ValueError(f"{split} data identity differs from the formal configuration")
    return path


def _normalized_question(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def validate_records(
    records: Sequence[Mapping[str, object]],
    *,
    split: str,
    expected_count: int,
    allowed_duplicate_extra_rows: Mapping[str, int] | None = None,
) -> DatasetValidationReport:
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    if len(records) != expected_count:
        raise ValueError(
            f"{split} row count differs: expected {expected_count}, observed {len(records)}"
        )
    task_types: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    questions_by_normalized_text: defaultdict[str, list[str]] = defaultdict(list)
    for index, record in enumerate(records):
        required = {"question", "answer", "task_type", "context", "extra"}
        if not required.issubset(record):
            raise ValueError(f"{split} row {index} is missing required fields")
        question = record["question"]
        answer = record["answer"]
        task_type = record["task_type"]
        context = record["context"]
        extra = record["extra"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{split} row {index} has an empty question")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"{split} row {index} has an empty answer")
        if not isinstance(task_type, str) or not task_type.strip():
            raise ValueError(f"{split} row {index} has an invalid task_type")
        if not isinstance(context, list) or not isinstance(extra, dict):
            raise ValueError(f"{split} row {index} has invalid context or extra")
        normalized = _normalized_question(question)
        source = record.get("source", extra.get("source"))
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"{split} row {index} has no source provenance")
        metric = record.get("metric", extra.get("metric"))
        if not isinstance(metric, str) or not metric.strip():
            raise ValueError(f"{split} row {index} has no metric contract")
        if split == "train" and source.casefold().replace(" ", "") == "aime2026":
            raise ValueError("AIME 2026 must not enter training data")
        if source.casefold().replace("-", "") == "swebench":
            instance_id = record.get("instance_id", extra.get("instance_id"))
            base_commit = record.get("base_commit", extra.get("base_commit"))
            if not isinstance(instance_id, str) or not instance_id.strip():
                raise ValueError("SWE-bench rows require an instance ID")
            if not isinstance(base_commit, str) or not base_commit.strip():
                raise ValueError("SWE-bench rows require a base commit")
        task_types[task_type] += 1
        sources[source] += 1
        questions_by_normalized_text[normalized].append(source)
    observed_duplicates: Counter[str] = Counter()
    for duplicate_sources in questions_by_normalized_text.values():
        if len(duplicate_sources) == 1:
            continue
        if len(set(duplicate_sources)) != 1:
            raise ValueError(f"{split} duplicate question crosses source boundaries")
        observed_duplicates[duplicate_sources[0]] += len(duplicate_sources) - 1
    allowed = Counter(allowed_duplicate_extra_rows or {})
    if any(type(count) is not int or count <= 0 for count in allowed.values()):
        raise ValueError("allowed duplicate counts must be positive integers")
    if observed_duplicates != allowed:
        raise ValueError(
            f"{split} duplicate distribution differs from its declared oversampling policy"
        )
    return DatasetValidationReport(
        split,
        len(records),
        task_types,
        sources,
        observed_duplicates,
    )


def _duplicate_policy(data: Mapping[str, object], split: str) -> Mapping[str, int]:
    policy = data.get("allowed_duplicate_extra_rows", {})
    if not isinstance(policy, dict):
        raise TypeError("allowed_duplicate_extra_rows must be an object")
    value = policy.get(split, {})
    if not isinstance(value, dict) or any(
        not isinstance(source, str) or type(count) is not int for source, count in value.items()
    ):
        raise TypeError("split duplicate policy must map source names to integer counts")
    return cast(dict[str, int], value)


def reject_cross_split_overlap(
    train: Sequence[Mapping[str, object]],
    validation: Sequence[Mapping[str, object]],
) -> None:
    train_questions: defaultdict[str, list[str]] = defaultdict(list)
    for record in train:
        question = record.get("question")
        if isinstance(question, str):
            normalized = _normalized_question(question)
            prefix = " ".join(normalized.split()[:12])
            train_questions[prefix].append(normalized)
    for record in validation:
        question = record.get("question")
        if not isinstance(question, str):
            continue
        normalized = _normalized_question(question)
        prefix = " ".join(normalized.split()[:12])
        if normalized in train_questions[prefix]:
            raise ValueError("train and validation contain an exact normalized overlap")


def _resolve_environment_path(name: object) -> Path:
    if not isinstance(name, str) or not name:
        raise TypeError("dataset path environment key must be non-empty text")
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise RuntimeError(f"required data path environment variable is unset: {name}")
    return Path(value).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline/paper_v1_250step.yaml")
    arguments = parser.parse_args()
    config = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("data"), dict):
        raise ValueError("baseline config is missing the data contract")
    data = cast(dict[str, object], config["data"])
    if data.get("strict_data") is not True:
        raise ValueError("formal data validation requires strict_data=true")
    train = _load_records(_validated_path(data, split="train"))
    validation = _load_records(_validated_path(data, split="validation"))
    expected_train = data.get("expected_train_rows")
    expected_validation = data.get("expected_validation_rows")
    if type(expected_train) is not int or type(expected_validation) is not int:
        raise TypeError("expected row counts must be integers")
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
    print(
        json.dumps(
            {
                "train": train_report.to_value(),
                "validation": validation_report.to_value(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
