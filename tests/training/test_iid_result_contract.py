"""Native IID acceptance never hides missing cases or a changed live deployment."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from skillev_private.evaluation import iid_episode_runtime as runtime
from skillev_private.evaluation.iid_architecture import (
    IIDSnapshotSelection,
    freeze_iid_architecture,
)

from skillev.evaluation.healthbench_luna_profile import healthbench_condition
from skillev.rollout.readonly_collection import EvaluationEpisodeOutcome, ReadOnlyPanelResult
from skillev.runtime import BudgetVector
from skillev.training.rollout_workflow import RolloutWorkflowBinding
from tests.training.test_iid_architecture import (
    IID_BENCHMARKS,
    synthetic_current_config,
    synthetic_native_panel,
)


def result_for(architecture, *, missing=None):
    rows = []
    for i, record in enumerate(architecture["panel"]["records"]):
        reward = SimpleNamespace(
            native_payload={"public_metrics": {"synthetic-score": 0.8}},
            native_metric_name="synthetic-score",
            value=0.8,
            success=True,
        )
        artifact = None if i == missing else SimpleNamespace(record=SimpleNamespace(reward=reward))
        rows.append(EvaluationEpisodeOutcome(i, record["public_task"]["task_id"], artifact))
    return ReadOnlyPanelResult(tuple(rows), BudgetVector(), 2.0)


def test_all_planned_cases_and_strict_native_floor_are_preserved():
    architecture = {
        "architecture_id": "test",
        "panel": {
            "records": [
                {"source": {"benchmark": domain}, "public_task": {"task_id": domain}}
                for domain in IID_BENCHMARKS
            ]
        },
        "acceptance_rules": {
            domain: {"metric": "synthetic-score", "minimum": 0.8, "comparison": "at-least"}
            for domain in IID_BENCHMARKS
        },
    }
    assert runtime.iid_aggregate(result_for(architecture), architecture)["baseline_accepted"]
    architecture["acceptance_rules"]["triviaqa"]["comparison"] = "strictly-above"
    assert not runtime.iid_aggregate(result_for(architecture), architecture)["baseline_accepted"]
    summary = runtime.iid_aggregate(result_for(architecture, missing=0), architecture)
    absent = summary["domains"][IID_BENCHMARKS[0]]
    assert absent["planned"] == absent["infrastructure_failures_or_not_started"] == 1
    assert absent["reward_mean"] is absent["success_count"] is None
    assert not absent["accepted"]


@pytest.mark.parametrize("include_humaneval", [False, True])
def test_aggregate_uses_its_frozen_panel_not_the_current_global_domain_list(include_humaneval):
    domains = IID_BENCHMARKS + (("humaneval",) if include_humaneval else ())
    architecture = {
        "architecture_id": "declared-panel",
        "panel": {
            "records": [
                {"source": {"benchmark": d}, "public_task": {"task_id": d}} for d in domains
            ]
        },
        "acceptance_rules": {d: {"metric": "synthetic-score", "minimum": 0.7} for d in domains},
    }
    summary = runtime.iid_aggregate(result_for(architecture), architecture)
    assert tuple(summary["domains"]) == domains
    assert sum(v["planned"] for v in summary["domains"].values()) == len(domains)
    assert summary["baseline_accepted"]


@pytest.mark.parametrize("drift", [False, True])
def test_completed_collection_is_not_accepted_until_live_controls_are_rechecked(
    make_training_harness, tmp_path, monkeypatch, drift
):
    harness = make_training_harness()
    source, identities = synthetic_native_panel(tmp_path)
    config = synthetic_current_config()
    scorers = {d: {"scorer": "synthetic"} for d in IID_BENCHMARKS}
    scorers["healthbench"] = healthbench_condition(config.healthbench_judge)
    path = freeze_iid_architecture(
        destination=tmp_path / "frozen",
        architecture_id="test-evaluation",
        source=source,
        identities=identities,
        excluded_sources={name: frozenset() for name in ("training", "development", "quality")},
        formal=config,
        initial_library=harness.library.state,
        model_controls={"model": "synthetic"},
        scorer_controls=scorers,
        environment_controls={"bridge": "synthetic"},
        serving_controls={"server": "synthetic"},
        workflow=RolloutWorkflowBinding(),
        acceptance_rules={d: {"metric": "synthetic-score", "minimum": 0.8} for d in IID_BENCHMARKS},
    )
    architecture = json.loads(path.read_text())

    async def collected(**_):
        return result_for(architecture)

    monkeypatch.setattr(runtime, "collect_readonly_panel", collected)
    validations = []

    def check_live_execution():
        validations.append(True)
        if drift:
            raise RuntimeError("actual service changed")

    output = tmp_path / "result"
    task = runtime.run_iid_episodes(
        architecture=architecture,
        selection=IIDSnapshotSelection("skills-off", harness.generator.snapshot()),
        generator=harness.generator,
        base_sessions=object(),
        actual_controls=architecture["controls"],
        output=output,
        chunk_size=3,
        validate_execution=check_live_execution,
    )
    if drift:
        with pytest.raises(RuntimeError):
            asyncio.run(task)
        assert not (output / "summary.json").exists()
        assert (output / "execution-failure.json").exists()
    else:
        value = asyncio.run(task)
        assert value["baseline_accepted"]
        assert all(
            value[k] == 0
            for k in (
                "training_updates",
                "posterior_updates",
                "skill_evolution",
                "training_evidence_writes",
            )
        )
    assert validations == [True]
