import asyncio
import io
import json
import subprocess
import tarfile
from decimal import Decimal
from pathlib import Path, PurePosixPath

import pytest
from skillev_private.direct_reference.swe_evaluator import (
    LocalCommandSWEEvaluator,
    SWEDatasetVariant,
    SWEEvaluationResult,
    SWEInstanceVerdict,
    SWEVerdictKind,
)

from scripts import reparse_qwen35_direct_swe, score_qwen35_direct_swe
from scripts import run_local_swebench_official as local_bridge


def test_swe_result_requires_complete_unique_per_instance_verdicts() -> None:
    verdicts = tuple(
        SWEInstanceVerdict(f"instance-{index}", SWEVerdictKind.TEST_FAILURE) for index in range(128)
    )
    result = SWEEvaluationResult(SWEDatasetVariant.VERIFIED, verdicts, "fixture@1")
    assert len(result.verdicts) == 128
    with pytest.raises(ValueError):
        SWEEvaluationResult(SWEDatasetVariant.VERIFIED, verdicts[:-1], "fixture@1")


def test_local_swe_evaluator_executes_the_harness_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}

    def complete(command: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        observed["cwd"] = kwargs["cwd"]
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", complete)
    evaluator = LocalCommandSWEEvaluator(("official-swe", "--work-root", "run"), tmp_path)
    asyncio.run(evaluator.preflight())
    assert observed == {
        "command": ("official-swe", "--work-root", "run", "--preflight"),
        "cwd": tmp_path,
    }


def test_local_bridge_reuses_definitive_official_verdicts(tmp_path: Path) -> None:
    run_id = "resume"
    row = {
        "instance_id": "project__issue-1",
        "model_name_or_path": "qwen35-direct-base",
        "model_patch": "diff --git a/a.py b/a.py",
    }
    log_dir = (
        tmp_path / "logs" / "run_evaluation" / run_id / "qwen35-direct-base" / row["instance_id"]
    )
    log_dir.mkdir(parents=True)
    (log_dir / "run_instance.log").write_text(">>>>> Patch Apply Failed", encoding="utf-8")
    assert local_bridge._stored_status(tmp_path, run_id, row) == "apply-failure"

    (log_dir / "report.json").write_text(
        '{"project__issue-1": {"resolved": true}}', encoding="utf-8"
    )
    assert local_bridge._stored_status(tmp_path, run_id, row) == "resolved"


def test_swe_replay_reparses_one_frozen_panel_without_generation(tmp_path: Path) -> None:
    generations = tmp_path / "generations.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "replayed.jsonl"
    patch = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new"
    with generations.open("w", encoding="utf-8") as generation_stream:
        with predictions.open("w", encoding="utf-8") as prediction_stream:
            for index in range(128):
                task_id = f"task-{index}"
                generation_stream.write(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "raw_text": f"```python\n{patch}\n```",
                            "infrastructure_error": None,
                        }
                    )
                    + "\n"
                )
                prediction_stream.write(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "instance_id": f"project__issue-{index}",
                            "model_patch": None,
                            "generation_infrastructure_error": None,
                        }
                    )
                    + "\n"
                )

    summary = reparse_qwen35_direct_swe.reparse(
        generations=generations,
        prior_predictions=predictions,
        output=output,
        parser_profile="unified-diff@2",
    )

    assert summary["submission_count"] == 128
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert {row["model_patch"] for row in rows} == {patch}
    assert {row["parser_profile_id"] for row in rows} == {"unified-diff@2"}


def test_local_bridge_imports_prior_verdicts_and_rejects_conflicts(tmp_path: Path) -> None:
    prior_path = tmp_path / "prior.jsonl"
    prior_path.write_text(
        json.dumps({"instance_id": "project__issue-1", "status": "test-failure"}) + "\n",
        encoding="utf-8",
    )
    prior = local_bridge._load_prior_verdicts(prior_path)
    row = {
        "instance_id": "project__issue-1",
        "model_name_or_path": "qwen35-direct-base",
        "model_patch": "diff --git a/a.py b/a.py",
    }
    assert local_bridge._combined_status(tmp_path, "resume", row, prior) == "test-failure"

    log_dir = (
        tmp_path / "logs" / "run_evaluation" / "resume" / "qwen35-direct-base" / row["instance_id"]
    )
    log_dir.mkdir(parents=True)
    (log_dir / "report.json").write_text(
        json.dumps({row["instance_id"]: {"resolved": True}}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="conflict"):
        local_bridge._combined_status(tmp_path, "resume", row, prior)


def test_local_bridge_partial_evidence_keeps_pending_separate(tmp_path: Path) -> None:
    rows = [
        {
            "instance_id": f"project__issue-{index}",
            "model_name_or_path": "qwen35-direct-base",
            "model_patch": "" if index == 0 else "diff --git a/a.py b/a.py",
        }
        for index in range(128)
    ]
    partial = local_bridge._partial_evidence(
        rows,
        tmp_path,
        "resume",
        {},
        error=RuntimeError("fixture"),
    )
    assert partial["status"] == "incomplete"
    assert len(partial["verdicts"]) == 1
    assert len(partial["pending_instance_ids"]) == 127
    assert partial["infrastructure_failure_count"] == 127


def test_local_bridge_progress_evidence_does_not_fail_untouched_rows(tmp_path: Path) -> None:
    rows = [
        {
            "instance_id": f"project__issue-{index}",
            "model_name_or_path": "qwen35-direct-base",
            "model_patch": "" if index == 0 else "diff --git a/a.py b/a.py",
        }
        for index in range(128)
    ]
    report_dir = (
        tmp_path
        / "logs"
        / "run_evaluation"
        / "resume"
        / "qwen35-direct-base"
        / rows[1]["instance_id"]
    )
    report_dir.mkdir(parents=True)
    (report_dir / "report.json").write_text(
        json.dumps({rows[1]["instance_id"]: {"resolved": False}}), encoding="utf-8"
    )

    progress = local_bridge._progress_evidence(rows, tmp_path, "resume", {})

    assert progress["status"] == "partial"
    assert len(progress["verdicts"]) == 2
    assert len(progress["pending_instance_ids"]) == 126
    assert progress["infrastructure_failure_count"] == 0


def test_local_bridge_uses_root_owned_archive_metadata(tmp_path: Path) -> None:
    source = tmp_path / "patch.diff"
    source.write_text("public fixture patch", encoding="utf-8")

    class Container:
        def __init__(self) -> None:
            self.command: object = None
            self.destination = ""
            self.archive = b""

        def exec_run(self, command: object) -> None:
            self.command = command

        def put_archive(self, destination: str, archive: bytes) -> None:
            self.destination = destination
            self.archive = archive

    container = Container()
    local_bridge._rootless_safe_copy_to_container(
        container, source, PurePosixPath("/tmp/patch.diff")
    )

    with tarfile.open(fileobj=io.BytesIO(container.archive)) as stream:
        member = stream.getmembers()[0]
        payload = stream.extractfile(member)
        assert member.uid == 0
        assert member.gid == 0
        assert payload is not None
        assert payload.read() == b"public fixture patch"
    assert container.command == ["mkdir", "-p", "/tmp"]
    assert container.destination == "/tmp"


def test_swe_repo_agent_score_uses_workspace_diff_contract() -> None:
    binding = score_qwen35_direct_swe._generation_binding(
        score_qwen35_direct_swe.REPOSITORY_AGENT_CONTRACT
    )

    assert binding == (
        "skillflow-code-generation-repository-agent@1",
        "qwen35-repository-agent-supervisor@1",
        "workspace-git-diff@1",
    )


def test_swe_direct_repo_agent_score_uses_single_model_workspace_diff_contract() -> None:
    binding = score_qwen35_direct_swe._generation_binding(
        score_qwen35_direct_swe.DIRECT_REPOSITORY_AGENT_CONTRACT
    )

    assert binding == (
        "qwen-direct-repository-agent@1",
        "qwen35-repository-agent-supervisor@1",
        "workspace-git-diff@1",
    )


def test_swe_direct_raw_tools_score_uses_unorchestrated_workspace_diff_contract() -> None:
    binding = score_qwen35_direct_swe._generation_binding(
        score_qwen35_direct_swe.DIRECT_RAW_TOOLS_CONTRACT
    )

    assert binding == (
        "qwen-direct-repository-agent-raw-tools@1",
        "qwen35-repository-agent-supervisor@1",
        "workspace-git-diff@1",
    )


def test_swe_direct_readonly_score_uses_one_final_diff_contract() -> None:
    binding = score_qwen35_direct_swe._generation_binding(
        score_qwen35_direct_swe.DIRECT_READONLY_CONTRACT
    )

    assert binding == (
        "qwen-direct-readonly-repository-agent@1",
        "qwen35-repository-agent-supervisor@1",
        "unified-diff@2",
    )


def test_swe_resolved_metric_uses_exact_decimal_arithmetic() -> None:
    metric = score_qwen35_direct_swe._resolved_metric(
        35,
        reference=Decimal("17.19"),
        max_gap_pp_exclusive=Decimal("7.0"),
    )

    assert metric["formal_observed_percent"] == "27.34375"
    assert metric["absolute_gap_pp"] == "10.15375"
    assert metric["numeric_status"] == "FAIL"
