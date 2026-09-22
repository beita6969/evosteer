#!/usr/bin/env python3
"""Bridge the typed evaluator protocol to the official local SWE-bench harness."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import io
import json
import os
import sys
import tarfile
import traceback
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any, cast

PROTOCOL_VARIANT = "princeton-nlp/SWE-bench_Verified"
OFFICIAL_HARNESS_DATASET = "SWE-bench/SWE-bench_Verified"
DEFINITIVE_STATUSES = frozenset(
    {"resolved", "apply-failure", "test-failure", "timeout", "candidate-invalid"}
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--instance-timeout-seconds", type=int, default=1800)
    parser.add_argument("--prior-verdicts", type=Path)
    parser.add_argument("--partial-output", type=Path)
    parser.add_argument(
        "--maximum-pending",
        type=int,
        help="Evaluate at most this many still-pending predictions before returning progress.",
    )
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def _docker_client() -> object:
    import docker

    client = docker.from_env(timeout=30)
    client.ping()
    return client


def _rootless_safe_copy_to_container(container: Any, src: Path, dst: PurePosixPath) -> None:
    """Copy one file while keeping archive ownership representable in a user namespace."""

    if not str(dst.parent) or str(dst.parent) == ".":
        raise ValueError("container destination requires a parent directory")
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as stream:
        info = stream.gettarinfo(str(src), arcname=dst.name)
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        with src.open("rb") as source:
            stream.addfile(info, source)
    container.exec_run(["mkdir", "-p", str(dst.parent)])
    container.put_archive(str(dst.parent), archive.getvalue())


def _install_rootless_archive_copy() -> Callable[..., Path]:
    """Patch only transport metadata; official image, patch, tests, and scoring stay unchanged."""

    from swebench.harness import docker_utils
    from swebench.harness import run_evaluation as run_evaluation_module

    docker_utils.copy_to_container = _rootless_safe_copy_to_container
    run_evaluation_module.copy_to_container = _rootless_safe_copy_to_container
    return cast(Callable[..., Path], run_evaluation_module.main)


def _preflight() -> None:
    from datasets import load_dataset

    _docker_client()
    protocol_dataset = load_dataset(PROTOCOL_VARIANT, split="test")
    harness_dataset = load_dataset(OFFICIAL_HARNESS_DATASET, split="test")
    protocol_ids = set(protocol_dataset["instance_id"])
    harness_ids = set(harness_dataset["instance_id"])
    if len(protocol_ids) != 500 or protocol_ids != harness_ids:
        raise RuntimeError("official SWE-bench Verified population is unavailable")


def _read_payload() -> tuple[str, list[dict[str, str]]]:
    value = json.load(sys.stdin)
    if type(value) is not dict:
        raise ValueError("typed evaluator input must be an object")
    raw = cast(dict[str, object], value)
    variant = str(raw.get("dataset_variant"))
    if variant != PROTOCOL_VARIANT:
        raise ValueError("local harness only accepts SWE-bench Verified")
    predictions = raw.get("predictions")
    if type(predictions) is not list or len(predictions) != 128:
        raise ValueError("official SWE evaluation requires 128 predictions")
    rows: list[dict[str, str]] = []
    for item in cast(list[object], predictions):
        if type(item) is not dict:
            raise ValueError("prediction must be an object")
        row = cast(dict[str, object], item)
        instance_id = str(row.get("instance_id") or "")
        if not instance_id:
            raise ValueError("prediction instance_id is empty")
        rows.append(
            {
                "instance_id": instance_id,
                "model_name_or_path": "qwen35-direct-base",
                "model_patch": str(row.get("model_patch") or ""),
            }
        )
    if len({row["instance_id"] for row in rows}) != 128:
        raise ValueError("prediction instance IDs must be unique")
    return variant, rows


def _log_text(work_root: Path, run_id: str, instance_id: str) -> str:
    directory = work_root / "logs" / "run_evaluation" / run_id / "qwen35-direct-base" / instance_id
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (directory / "run_instance.log", directory / "test_output.txt")
        if path.is_file()
    )


def _stored_status(work_root: Path, run_id: str, row: dict[str, str]) -> str | None:
    if not row["model_patch"].strip():
        return "candidate-invalid"
    instance_id = row["instance_id"]
    directory = work_root / "logs" / "run_evaluation" / run_id / "qwen35-direct-base" / instance_id
    report_path = directory / "report.json"
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            resolved = cast(dict[str, object], report)[instance_id]
            if type(resolved) is not dict:
                raise ValueError
            return (
                "resolved"
                if bool(cast(dict[str, object], resolved)["resolved"])
                else "test-failure"
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("stored official SWE report is invalid") from exc
    text = _log_text(work_root, run_id, instance_id)
    if ">>>>> Patch Apply Failed" in text:
        return "apply-failure"
    if "Timeout error:" in text or "Test timed out after" in text:
        return "timeout"
    return None


def _load_prior_verdicts(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    verdicts: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if type(value) is not dict or set(value) != {"instance_id", "status"}:
                raise ValueError("prior SWE verdict rows require instance_id and status")
            row = cast(dict[str, object], value)
            instance_id = str(row["instance_id"])
            status = str(row["status"])
            if not instance_id or status not in DEFINITIVE_STATUSES:
                raise ValueError("prior SWE verdict row is invalid")
            if instance_id in verdicts:
                raise ValueError("prior SWE verdict IDs must be unique")
            verdicts[instance_id] = status
    return verdicts


def _combined_status(
    work_root: Path,
    run_id: str,
    row: dict[str, str],
    prior: dict[str, str],
) -> str | None:
    stored = _stored_status(work_root, run_id, row)
    imported = prior.get(row["instance_id"])
    if imported == "candidate-invalid" and row["model_patch"].strip():
        raise RuntimeError("prior candidate-invalid verdict conflicts with a submitted patch")
    if imported is not None and not row["model_patch"].strip():
        if imported != "candidate-invalid":
            raise RuntimeError("prior official verdict conflicts with an empty candidate")
    if stored is not None and imported is not None and stored != imported:
        raise RuntimeError("prior and work-root SWE verdicts conflict")
    return stored if stored is not None else imported


def _verdicts(
    rows: list[dict[str, str]],
    work_root: Path,
    run_id: str,
    prior: dict[str, str],
) -> list[dict[str, str]]:
    verdicts: list[dict[str, str]] = []
    unknown: list[str] = []
    for row in sorted(rows, key=lambda item: item["instance_id"]):
        status = _combined_status(work_root, run_id, row, prior)
        if status is None:
            unknown.append(row["instance_id"])
        else:
            verdicts.append({"instance_id": row["instance_id"], "status": status})
    if unknown or len(verdicts) != 128:
        raise RuntimeError(
            f"official harness left {len(unknown)} predictions without definitive verdicts"
        )
    return verdicts


def _partial_evidence(
    rows: list[dict[str, str]],
    work_root: Path,
    run_id: str,
    prior: dict[str, str],
    *,
    error: BaseException,
) -> dict[str, object]:
    verdicts: list[dict[str, str]] = []
    pending: list[str] = []
    for row in sorted(rows, key=lambda item: item["instance_id"]):
        status = _combined_status(work_root, run_id, row, prior)
        if status is None:
            pending.append(row["instance_id"])
        else:
            verdicts.append({"instance_id": row["instance_id"], "status": status})
    return {
        "status": "incomplete",
        "verdicts": verdicts,
        "pending_instance_ids": pending,
        "infrastructure_failure_count": len(pending),
        "error_type": type(error).__name__,
    }


def _progress_evidence(
    rows: list[dict[str, str]],
    work_root: Path,
    run_id: str,
    prior: dict[str, str],
) -> dict[str, object]:
    """Return restart-safe progress without classifying untouched rows as failures."""

    verdicts: list[dict[str, str]] = []
    pending: list[str] = []
    for row in sorted(rows, key=lambda item: item["instance_id"]):
        status = _combined_status(work_root, run_id, row, prior)
        if status is None:
            pending.append(row["instance_id"])
        else:
            verdicts.append({"instance_id": row["instance_id"], "status": status})
    return {
        "status": "partial" if pending else "complete",
        "verdicts": verdicts,
        "pending_instance_ids": pending,
        "infrastructure_failure_count": 0,
    }


def _evaluate(
    arguments: argparse.Namespace,
    prior: dict[str, str],
    *,
    variant: str,
    rows: list[dict[str, str]],
) -> dict[str, object]:
    run_evaluation = _install_rootless_archive_copy()

    work_root = arguments.work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    predictions_path = work_root / "pending-predictions.jsonl"
    panel_ids = {row["instance_id"] for row in rows}
    if not set(prior).issubset(panel_ids):
        raise ValueError("prior SWE verdicts contain IDs outside the current panel")
    pending = [
        row for row in rows if _combined_status(work_root, arguments.run_id, row, prior) is None
    ]
    if arguments.maximum_pending is not None:
        pending = pending[: arguments.maximum_pending]
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in pending),
        encoding="utf-8",
    )
    report_dir = work_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    harness_log = work_root / "official-harness.log"
    if pending:
        previous = Path.cwd()
        try:
            os.chdir(work_root)
            with harness_log.open("a", encoding="utf-8") as stream:
                with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                    report_path = run_evaluation(
                        dataset_name=OFFICIAL_HARNESS_DATASET,
                        split="test",
                        instance_ids=[row["instance_id"] for row in pending],
                        predictions_path=str(predictions_path),
                        max_workers=arguments.workers,
                        open_file_limit=4096,
                        run_id=arguments.run_id,
                        timeout=arguments.instance_timeout_seconds,
                        rewrite_reports=False,
                        modal=False,
                        report_dir=str(report_dir),
                        task_repo=None,
                    )
        finally:
            os.chdir(previous)
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        if type(report) is not dict:
            raise RuntimeError("official SWE report is not an object")
        submitted = set(cast(list[str], report.get("submitted_ids", [])))
        expected_pending = {row["instance_id"] for row in pending}
        if submitted != expected_pending:
            raise RuntimeError("official report submitted IDs differ from pending panel")
        if cast(list[str], report.get("infra_failure_ids", [])):
            raise RuntimeError("official harness reported environment infrastructure failures")
    if arguments.maximum_pending is not None:
        progress = _progress_evidence(rows, work_root, arguments.run_id, prior)
        if progress["status"] == "partial":
            return {
                "dataset_variant": variant,
                "evaluator_version": f"swebench@{importlib.metadata.version('swebench')}",
                **progress,
            }
    return {
        "dataset_variant": variant,
        "evaluator_version": f"swebench@{importlib.metadata.version('swebench')}",
        "verdicts": _verdicts(rows, work_root, arguments.run_id, prior),
    }


def main() -> None:
    arguments = _arguments()
    if arguments.workers < 1:
        raise ValueError("workers must be positive")
    if arguments.maximum_pending is not None and arguments.maximum_pending < 1:
        raise ValueError("maximum pending must be positive")
    if arguments.preflight:
        _preflight()
        return
    prior = _load_prior_verdicts(arguments.prior_verdicts)
    variant, rows = _read_payload()
    try:
        result = _evaluate(arguments, prior, variant=variant, rows=rows)
    except Exception as exc:
        arguments.work_root.mkdir(parents=True, exist_ok=True)
        (arguments.work_root / "bridge-error.log").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        if arguments.partial_output is not None:
            partial = _partial_evidence(
                rows,
                arguments.work_root.resolve(),
                arguments.run_id,
                prior,
                error=exc,
            )
            arguments.partial_output.parent.mkdir(parents=True, exist_ok=True)
            arguments.partial_output.write_text(
                json.dumps(partial, sort_keys=True) + "\n", encoding="utf-8"
            )
        raise
    if result.get("status") == "partial" and arguments.partial_output is not None:
        arguments.partial_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.partial_output.write_text(
            json.dumps(result, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
