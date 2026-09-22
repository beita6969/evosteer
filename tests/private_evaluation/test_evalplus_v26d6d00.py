from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from skillev_private.benchmarks import evalplus_process
from skillev_private.benchmarks.evalplus_process import (
    EvalPlusExecutionContract,
    run_evalplus_26d6d00,
)
from skillev_private.benchmarks.evalplus_result_v26d6d00 import (
    parse_evalplus_26d6d00_task,
)

REVISION = "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e"


def _contract(root: Path) -> EvalPlusExecutionContract:
    (root / "evalplus").mkdir(parents=True)
    return EvalPlusExecutionContract(
        source_root=root,
        python_executable=Path("/usr/bin/python3"),
        code_revision=REVISION,
        dataset_name="mbpp",
        dataset_version="v0.2.0",
        result_schema_profile="evalplus-task-list-base-plus-status@26d6d00",
        min_time_limit=0.1,
        gt_time_limit_factor=2.0,
        parallelism=2,
        timeout_seconds=60,
    )


def test_pinned_evalplus_process_uses_explicit_version_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract(tmp_path / "source")
    carrier = tmp_path / "carrier.jsonl"
    carrier.write_text('{"task_id":"Mbpp/1","solution":"pass"}\n', encoding="utf-8")
    output = tmp_path / "results.json"
    private_dataset = tmp_path / "MbppPlus-v0.2.0.jsonl.gz"
    monkeypatch.setenv("MBPP_OVERRIDE_PATH", str(private_dataset))

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 0, stdout=REVISION + "\n", stderr="")
        assert command[command.index("--version") + 1] == "v0.2.0"
        assert command[command.index("--output-file") + 1] == str(output)
        assert kwargs["cwd"] == contract.source_root
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert environment["MBPP_OVERRIDE_PATH"] == str(private_dataset)
        output.write_text(
            json.dumps(
                {
                    "eval": {
                        "Mbpp/1": [
                            {
                                "task_id": "Mbpp/1",
                                "base_status": "pass",
                                "plus_status": "pass",
                            }
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(evalplus_process.subprocess, "run", fake_run)
    payload = run_evalplus_26d6d00(
        contract,
        carrier=carrier,
        output_file=output,
        stdout=tmp_path / "stdout.log",
        stderr=tmp_path / "stderr.log",
    )
    verdict = parse_evalplus_26d6d00_task("Mbpp/1", payload["eval"]["Mbpp/1"])
    assert verdict.base_plus_passed


def test_pinned_evalplus_parser_rejects_legacy_dictionary_schema() -> None:
    with pytest.raises(ValueError):
        parse_evalplus_26d6d00_task(
            "Mbpp/1",
            {"nfiles": 1, "base": [["success"]], "plus": [["success"]]},
        )


def test_pinned_evalplus_parser_uses_pass_fail_status_vocabulary() -> None:
    passed = parse_evalplus_26d6d00_task(
        "Mbpp/1",
        [{"task_id": "Mbpp/1", "base_status": "pass", "plus_status": "pass"}],
    )
    assert passed.base_plus_passed
    with pytest.raises(ValueError):
        parse_evalplus_26d6d00_task(
            "Mbpp/1",
            [
                {
                    "task_id": "Mbpp/1",
                    "base_status": "success",
                    "plus_status": "success",
                }
            ],
        )
