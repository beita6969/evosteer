"""Source export noninterference and syntax-only code boundary checks."""

from __future__ import annotations

import json

import pytest
from skillev_private.evaluation.integrity_grader_usage import MeteredCompletions
from skillev_private.evaluation.integrity_native_scoring import humaneval_source_parts
from skillev_private.evaluation.integrity_sources import (
    load_source_panel,
    public_source_view,
    select_source_panel,
)

from skillev.evaluation.input_metric_contracts import IID_BENCHMARKS
from skillev.evaluation.integrity_metric_schema import NATIVE_VERIFIER_VERSIONS
from skillev.evaluation.step0_completion import StepZeroTerminalMode, project_terminal_candidate


def test_mbpp_only_panel_does_not_load_unrequested_datasets_or_change_order(tmp_path):
    prompt = '"""Double the integer.\nassert twice(3) == 6\n"""\n'
    rows = [
        {
            "task_id": name,
            "prompt": prompt,
            "canonical_solution": "def twice(n): return n * 2",
            "entry_point": "twice",
            "base_input": [[3]],
            "plus_input": [[8]],
            "assertion": "private-synthetic-assertion",
            "contract": "",
            "atol": 0,
        }
        for name in ("case-first", "case-second")
    ]
    prompts, targets, manifest = (
        tmp_path / name for name in ("prompts.json", "targets.jsonl", "manifest.json")
    )
    prompts.write_text(json.dumps({row["task_id"]: prompt for row in rows}))
    targets.write_text("\n".join(json.dumps(row) for row in rows))
    manifest.write_text(
        json.dumps(
            {
                "entries": [
                    {"task_id": row["task_id"], "source_identity": row["task_id"]}
                    for row in reversed(rows)
                ]
            }
        )
    )
    config = {
        "source_files": {"mbpp_public_prompts": str(prompts), "mbpp_targets": str(targets)},
        "manifests": {"mbpp-plus": str(manifest)},
        "trivia_aliases": str(tmp_path / "unrequested-and-absent.json"),
        "evaluation_sample_counts": {"mbpp-plus": 1},
    }
    source = load_source_panel(config)
    selected = select_source_panel(source, config["evaluation_sample_counts"])
    assert [entry.task_id for entry in selected.panel.entries] == ["case-second"]
    assert dict(selected.panel.entries[0].fields)["prompt"] == prompt
    assert "private-synthetic-assertion" not in selected.panel.entries[0].render()
    assert not selected.interactive
    assert list(selected.targets) == ["case-second"]


@pytest.mark.parametrize("benchmark", IID_BENCHMARKS)
def test_mutating_all_private_labels_does_not_change_source_specific_actor_input(
    benchmark: str,
) -> None:
    public = {
        "question": "Question: A public question",
        "context": [f"Public passage {n}" for n in range(10)],
        "prompt": [{"role": "user", "content": "Public request", "hidden_metadata": "private"}]
        if benchmark == "healthbench"
        else "def public_function():\n    pass\n",
        "task": "Public task",
    }
    a = public_source_view(
        "synthetic",
        benchmark,
        {
            **public,
            "answer": "private-A",
            "rubrics": ["private-A"],
            "canonical_solution": "private-A",
            "test": "private-A",
        },
    )
    b = public_source_view(
        "synthetic",
        benchmark,
        {
            **public,
            "answer": "private-B",
            "rubrics": ["private-B"],
            "canonical_solution": "private-B",
            "test": "private-B",
        },
    )
    assert a == b
    assert "private-A" not in a.render()
    assert "hidden_metadata" not in a.render()


def test_complete_humaneval_function_is_not_prefixed_by_another_unfinished_stub() -> None:
    prompt = 'def twice(x):\n    """Return twice x."""\n'
    full = "def twice(x):\n    return 2 * x\n"
    assert humaneval_source_parts(prompt, full) == ("", full)
    body = "    return 2 * x\n"
    assert humaneval_source_parts(prompt, body) == (prompt, body)
    projected = project_terminal_candidate(
        StepZeroTerminalMode.PYTHON_SOURCE, f"```python\n{body}```"
    )
    assert projected == body.rstrip("\n")


@pytest.mark.parametrize(
    "candidate",
    ["-1", "1000", "42.0", "Final answer: 42\nFinal answer: 43", "Final answer: 42\n\\boxed{43}"],
)
def test_aime_never_chooses_between_conflicting_or_invalid_finals(candidate: str) -> None:
    assert project_terminal_candidate(StepZeroTerminalMode.AIME_INTEGER, candidate) is None


def test_grader_meter_keeps_retry_cost_separate_and_does_not_invent_unknown_tokens() -> None:
    from types import SimpleNamespace

    class Completions:
        calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise OSError("synthetic transport interruption")
            return SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=17, completion_tokens=3)
                if self.calls == 2
                else None
            )

    meter = MeteredCompletions(Completions())
    with pytest.raises(OSError):
        meter.create(messages=[{"content": "synthetic private rubric"}])
    meter.create()
    meter.create()
    cost = meter.snapshot()
    assert cost["model_request_attempts"] == 3
    assert cost["model_responses"] == 2
    assert cost["unknown_usage_calls"] == 2
    assert cost["known_input_tokens"] == 17
    assert cost["known_output_tokens"] == 3
    assert "private rubric" not in repr(cost)


def test_native_score_preserves_cost_but_reads_older_records_without_inventing_it() -> None:
    from dataclasses import asdict

    from skillev.evaluation.integrity_results import NativeScore

    score = NativeScore(
        "synthetic",
        "aime-2026",
        "accuracy",
        1.0,
        scorer_cost={"wall_seconds": 2},
        verifier_version=NATIVE_VERIFIER_VERSIONS["aime-2026"],
    )
    stored = asdict(score)
    assert NativeScore.from_value(stored) == score
    del stored["scorer_cost"]
    assert NativeScore.from_value(stored).scorer_cost == {}
