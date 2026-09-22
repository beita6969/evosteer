import asyncio
import json
from dataclasses import replace

import pytest
from skillev_private.experiments.quality_panel import (
    FixedQualityPanel,
    PanelSlot,
    collection_probe,
)
from skillev_private.experiments.zero_update_bridge import collect_training_condition

from skillev.rollout import UnskilledRolloutSessionBundle
from skillev.training.quality_gate import (
    ProtocolProbe,
    QualityGatePolicy,
    QualityRule,
    evaluate_quality,
)
from skillev.training.rollout_workflow import RolloutWorkflowBinding
from skillev.training.run_condition import EffectiveRunCondition
from tests.training.fakes import OrderedSessionFactory


@pytest.fixture
def collected_panel(make_training_harness, tmp_path, request):
    harness = make_training_harness()
    count = getattr(request, "param", 2)
    tasks = harness.task_provider.tasks[:count]
    panel = FixedQualityPanel(
        "fixed-synthetic",
        "raw-panel",
        tuple(
            PanelSlot(t.task_id, "humaneval", f"synthetic-{index}", "same-source")
            for index, t in enumerate(tasks)
        ),
    )
    sessions = OrderedSessionFactory(tuple(float(i % 2) for i in range(count)))

    class BaseSessions:
        def create(self, task):
            bundle = sessions.create(task)

            class Evaluator:
                async def evaluate(self, request):
                    reward = await bundle.evaluator.evaluate(request)
                    return replace(
                        reward,
                        native_payload={
                            "benchmark_id": "humaneval",
                            "passed": reward.success,
                            "training_evidence_source": {
                                "benchmark_id": "humaneval",
                                "population_id": next(
                                    s.population_id
                                    for s in panel.slots
                                    if s.task_id == task.task_id
                                ),
                                "source_question_id": "same-source",
                            },
                        },
                    )

            return UnskilledRolloutSessionBundle(bundle.environment, Evaluator(), bundle.cleanup)

    root = tmp_path / "quality"
    result = asyncio.run(
        collect_training_condition(
            root=root,
            condition=EffectiveRunCondition.create(
                condition_id="raw-panel", scientific={"wire": "raw-json"}, execution={}
            ),
            tasks=tasks,
            generator=harness.generator,
            base_sessions=BaseSessions(),
            library_state=harness.library.state,
            trainer=replace(
                harness.config, execution=replace(harness.config.execution, batch_size=count)
            ),
            maximum_h0_tokens=2048,
            workflow=RolloutWorkflowBinding(),
            sampling_schedule_id="declared-quality-sampling-seed0@1",
            ordered_task_sequence_id="fixed-synthetic-source-order@1",
            quality_panel=panel,
            excluded_quality_sources=frozenset(),
        )
    )
    return harness, root, panel, result


def test_real_collect_only_panel_supplies_quality_gate(collected_panel):
    harness, root, panel, result = collected_panel
    probe = ProtocolProbe(**json.loads((root / "probe-00000000.json").read_text()))
    assert probe.source_question_count == 1  # two population aliases, one original source
    assert probe.metrics["panel/trajectory_count"] == 2
    assert probe.metrics["panel/success_fraction"] == 0.5
    assert probe.metrics["panel/valid_terminal_record_fraction"] == 1
    assert probe.metrics["panel/valid_terminal_record_count"] == 2
    assert probe.metrics["humaneval/success_fraction/source_group_mean"] == 0.5
    assert probe.metrics["humaneval/success_fraction/source_group_standard_error"] is None
    assert "Small fixed panels" in probe.metric_notes["source_groups"]
    assert probe.metrics["humaneval/passed"] == 0.5
    assert probe.metrics["panel/first_turn_structure_valid_fraction"] == 1
    assert probe.metrics["panel/admitted_fraction"] == 1
    assert probe.metrics["panel/execution_returned_success_fraction"] is None
    policy = QualityGatePolicy(
        "test-rule",
        panel.panel_id,
        panel.condition_id,
        1,
        1,
        1,
        (QualityRule("panel/first_turn_structure_valid_fraction", 0.95, 0.05),),
    )
    assert (
        evaluate_quality(
            policy,
            baseline=probe,
            probes=(),
            policy_step=0,
            policy_snapshot_id=result.policy_snapshot_id,
        ).status
        == "verified"
    )
    assert harness.loop.optimizer_step == 0
    assert harness.task_provider.cursor == 0
    assert not (root / "checkpoints").exists()


def test_missing_execution_evidence_does_not_become_admission(collected_panel):
    _, root, panel, result = collected_panel
    probe = collection_probe(
        result,
        panel=panel,
        policy_step=0,
        evidence_id="missing",
        events_path=root / "absent.jsonl",
        event_run_id=root.name,
    )
    assert probe.metrics["panel/action_structure_valid_fraction"] == 1
    assert probe.metrics["panel/admitted_fraction"] is None
    assert probe.metrics["panel/assessed_action_count"] == 0
    assert probe.metrics["panel/terminal_evidence_record_count"] == 0
    assert probe.metrics["panel/valid_terminal_record_fraction"] is None
    assert probe.metrics["panel/valid_terminal_record_fraction/source_group_mean"] is None


def test_panel_rejects_partial_population_wrong_source_and_excluded_question(collected_panel):
    _, root, panel, result = collected_panel
    with pytest.raises(ValueError):
        panel.require_disjoint(frozenset({panel.slots[0].source}))
    for candidate, specification in (
        (replace(result, diagnostic_artifacts=result.diagnostic_artifacts[:1]), panel),
        (
            result,
            replace(
                panel, slots=tuple(replace(s, source_question_id="another") for s in panel.slots)
            ),
        ),
        (result, replace(panel, slots=tuple(reversed(panel.slots)))),
    ):
        with pytest.raises(ValueError):
            collection_probe(
                candidate,
                panel=specification,
                policy_step=0,
                evidence_id="bad",
                events_path=root / "events.jsonl",
                event_run_id=root.name,
            )


@pytest.mark.parametrize("collected_panel", [4], indirect=True)
def test_four_population_alias_rollouts_have_one_source_group(collected_panel):
    _, root, _, _ = collected_panel
    probe = ProtocolProbe(**json.loads((root / "probe-00000000.json").read_text()))
    assert probe.metrics["panel/trajectory_count"] == 4
    assert probe.source_question_count == probe.metrics["panel/source_group_count"] == 1
    assert probe.metrics["humaneval/passed/source_group_mean"] == 0.5
    assert probe.metrics["humaneval/passed/source_group_sample_sd"] is None
    assert probe.metrics["humaneval/passed/source_group_standard_error"] is None


def test_distinct_source_changes_only_group_report_not_native_score(collected_panel):
    _, root, panel, result = collected_panel
    artifacts = list(result.diagnostic_artifacts)
    native = artifacts[1].record.reward.native_payload
    native = {
        **native,
        "training_evidence_source": {
            **native["training_evidence_source"],
            "source_question_id": "independent-source",
        },
    }
    artifacts[1] = replace(
        artifacts[1],
        record=replace(
            artifacts[1].record, reward=replace(artifacts[1].record.reward, native_payload=native)
        ),
    )
    panel = replace(
        panel,
        slots=(panel.slots[0], replace(panel.slots[1], source_question_id="independent-source")),
    )
    probe = collection_probe(
        replace(result, diagnostic_artifacts=tuple(artifacts)),
        panel=panel,
        policy_step=0,
        evidence_id="distinct-sources",
        events_path=root / "events.jsonl",
        event_run_id=root.name,
    )
    assert probe.metrics["humaneval/passed"] == 0.5
    assert probe.metrics["humaneval/source_group_count"] == 2
    assert probe.metrics["humaneval/passed/source_group_standard_error"] == pytest.approx(0.5)


def test_declared_composite_preserves_same_origin_arithmetic(collected_panel):
    from skillev_private.experiments.quality_composite import (
        DomainQualityEvidence,
        composite_baseline,
    )

    _, root, panel, result = collected_panel
    original = ProtocolProbe(**json.loads((root / "probe-00000000.json").read_text()))
    part = DomainQualityEvidence(
        "humaneval",
        panel.condition_id,
        result.diagnostic_artifacts,
        root / "events.jsonl",
        root.name,
    )
    derived = composite_baseline(
        (part,),
        panel=panel,
        policy_snapshot_id=result.policy_snapshot_id,
        declaration_id="owner-selected-baseline",
        declaration_reason="reuse the exact fixed panel",
    )
    assert derived.metrics == original.metrics
    assert derived.source_question_count == original.source_question_count
    assert derived.metric_notes["baseline_kind"] == "owner-declared-composite-not-joint-batch"
    for bad in ((replace(part, artifacts=part.artifacts[:1]),), (part, part)):
        with pytest.raises(ValueError):
            composite_baseline(
                bad,
                panel=panel,
                policy_snapshot_id=result.policy_snapshot_id,
                declaration_id="bad",
                declaration_reason="test",
            )
    with pytest.raises(ValueError):
        composite_baseline(
            (part,),
            panel=panel,
            policy_snapshot_id="another-policy",
            declaration_id="bad-policy",
            declaration_reason="test",
        )


def test_composite_origins_preserve_colliding_ids_and_missing_assessments(
    collected_panel, tmp_path
):
    from skillev_private.experiments.quality_composite import (
        DomainQualityEvidence,
        composite_baseline,
    )

    _, root, panel, result = collected_panel
    first, second = result.diagnostic_artifacts
    old_second_id = second.record.trajectory_id
    native = dict(second.record.reward.native_payload)
    native["benchmark_id"] = "alfworld"
    native["training_evidence_source"] = {
        **native["training_evidence_source"],
        "benchmark_id": "alfworld",
    }
    second = replace(
        second,
        manifest=replace(second.manifest, trajectory_id=first.record.trajectory_id),
        record=replace(
            second.record,
            trajectory_id=first.record.trajectory_id,
            reward=replace(second.record.reward, native_payload=native),
        ),
    )
    mixed = replace(
        panel,
        condition_id="explicit-new-condition",
        slots=(panel.slots[0], replace(panel.slots[1], benchmark_id="alfworld")),
    )
    other = tmp_path / "other-events.jsonl"
    with other.open("w") as f:
        for line in (root / "events.jsonl").open():
            event = json.loads(line)
            payload = event.get("payload", {})
            if payload.get("trajectory_id") == old_second_id and "assessment" in payload:
                payload["trajectory_id"] = second.record.trajectory_id
                f.write(json.dumps(event) + "\n")
    parts = (
        DomainQualityEvidence(
            "humaneval", "old-six-domain-condition", (first,), root / "events.jsonl", root.name
        ),
        DomainQualityEvidence("alfworld", "new-alf-condition", (second,), other, root.name),
    )
    originals = [a.to_value() for a in (first, second)]
    args = {
        "panel": mixed,
        "policy_snapshot_id": result.policy_snapshot_id,
        "declaration_id": "declared-composite",
        "declaration_reason": "owner selected whole domains",
    }
    probe = composite_baseline(parts, **args)
    assert probe.metrics["panel/trajectory_count"] == 2
    assert probe.metrics["panel/success_fraction"] == 0.5
    assert probe.metrics["panel/admitted_fraction"] == 1
    assert probe.source_question_count == 2
    assert [a.to_value() for a in (first, second)] == originals
    missing = composite_baseline(
        (parts[0], replace(parts[1], events_path=tmp_path / "absent")), **args
    )
    assert missing.metrics["panel/admitted_fraction"] is None
    assert missing.metrics["humaneval/admitted_fraction"] == 1
    assert missing.metrics["alfworld/admitted_fraction"] is None
