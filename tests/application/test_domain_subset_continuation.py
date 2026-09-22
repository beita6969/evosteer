import asyncio
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from skillev_private.benchmarks.protocol_v13_seven_training import (
    SEVEN_TRAINING_DOMAINS,
    SIX_TRAINING_DOMAINS,
    seven_domain_training_trajectories,
)
from skillev_private.benchmarks.protocol_v13_training import build_protocol13_training_records
from skillev_private.experiments.bayesian_condition_transition import (
    _bind_run_directory,
    observer_batch_size_starts,
    save_condition_transition,
)
from skillev_private.experiments.bayesian_training_config import BayesianFormalConfig
from skillev_private.experiments.training_domain_schedule import schedule_summary, training_schedule

from skillev.domain_subset_continuation import DomainSubsetContinuation
from tests.application.test_bayesian_recovery import restore_fixture
from tests.application.test_full_vertical_loop import build_application_fixture
from tests.application.test_reasoning_continuation import with_config
from tests.benchmarks.test_protocol_v13_training import _sources


def configs():
    return tuple(
        BayesianFormalConfig.load(Path("configs/training") / name)
        for name in (
            "bayesianimprove_autonomous_ttb_proactive_skills.yaml",
            "bayesianimprove_autonomous_ttb_six_domain.yaml",
        )
    )


def test_six_domain_schedule_preserves_consumed_prefix_and_next_sources(tmp_path):
    before, after = configs()
    assert before.batch_size == 28
    assert after.batch_size == 24
    assert set(after.domains) == {d.value for d in SIX_TRAINING_DOMAINS}
    assert "humaneval" not in after.reasoning_modes
    assert after.reasoning_modes["aime-2026"]
    assert after.reasoning_modes["healthbench"]
    records = build_protocol13_training_records(_sources(9))
    original = seven_domain_training_trajectories(records, steps=250)
    selected = seven_domain_training_trajectories(
        records,
        steps=250,
        domain_starts={1: SEVEN_TRAINING_DOMAINS, 5: SIX_TRAINING_DOMAINS},
    )
    assert selected[:112] == original[:112]
    assert selected[112:] == tuple(
        r for r in original[112:] if r.episode.benchmark in SIX_TRAINING_DOMAINS
    )
    assert len(selected) == len({r.input.task_id for r in selected}) == 6016
    assert Counter(r.episode.benchmark for r in selected[112:136]) == dict.fromkeys(
        SIX_TRAINING_DOMAINS, 4
    )
    summary = schedule_summary(after, selected)
    assert summary["question_occurrences"] == 1504
    assert summary["batch_size_segments"][-1]["first_step"] == 5
    six_sources = tuple(r for r in records if r.episode.benchmark in SIX_TRAINING_DOMAINS)
    assert len(training_schedule(after, six_sources, root=tmp_path)) == 6000


def test_domain_boundary_is_explicit_and_history_survives(tmp_path):
    before, after = configs()
    root = tmp_path / "run"
    _bind_run_directory(root, before, None)
    snapshot = root / "checkpoints/step-00000004"
    with pytest.raises(ValueError):
        _bind_run_directory(root, after, snapshot)
    assert _bind_run_directory(root, after, snapshot, allow_domain_subset=True) == before
    with pytest.raises(ValueError):
        _bind_run_directory(
            root, replace(after, adapter_learning_rate=0.002), snapshot, allow_domain_subset=True
        )
    app = SimpleNamespace(
        training_loop=SimpleNamespace(optimizer_step=4),
        snapshot_identity=SimpleNamespace(sampling_schedule_algorithm="original-curriculum"),
        evolution_loop=SimpleNamespace(save_condition_boundary=lambda name: None),
    )
    save_condition_transition(
        app, root=root, source_snapshot=snapshot, source=before, target=after, domain_subset=True
    )
    assert observer_batch_size_starts(root, after) == {1: 28, 5: 24}
    assert json.loads((root / "formal-config.json").read_text()) == before.to_value()
    records = build_protocol13_training_records(_sources(9))
    assert len(training_schedule(before, records, root=root)) == 7000
    assert len(training_schedule(after, records, root=root)) == 6016


def test_full_restore_keeps_optimizer_posterior_library_and_consumed_cursor(
    tmp_path, training_backbone_config, monkeypatch
):
    fixture = build_application_fixture(tmp_path, training_backbone_config, cycles=2, batch_size=4)
    asyncio.run(
        fixture.application.evolution_loop.run(fixture.run_plan, maximum_steps_this_attempt=1)
    )
    source = fixture.public_identity
    config = source.application_config
    target = with_config(
        source,
        replace(
            config,
            trainer=replace(
                config.trainer, execution=replace(config.trainer.execution, batch_size=2)
            ),
        ),
    )
    tasks = fixture.tasks[:4] + tuple(t for i, t in enumerate(fixture.tasks[4:]) if i % 4 < 2)
    rewards = fixture.rewards[:4] + tuple(r for i, r in enumerate(fixture.rewards[4:]) if i % 4 < 2)
    declaration = DomainSubsetContinuation(
        source, 1, 4, tuple(t.task_id for t in fixture.tasks), tuple(t.task_id for t in tasks)
    )
    assert declaration.require_target(target) == source.runtime_snapshot_identity()
    with pytest.raises(ValueError):
        replace(
            declaration, target_task_ids=tuple(reversed(declaration.target_task_ids))
        ).require_target(target)
    resumed_fixture = SimpleNamespace(**{**vars(fixture), "tasks": tasks, "rewards": rewards})
    original = tmp_path / "snapshots/step-00000001"
    restored, _ = restore_fixture(
        resumed_fixture,
        original,
        training_backbone_config,
        public_identity=target,
        domain_subset_continuation=declaration,
    )
    # The shared helper compares every optimizer tensor and full projection/library state.
    boundary = restored.evolution_loop.save_condition_boundary("smaller-batch-from-step-one")
    saved = restored.snapshot_store.load_exact(boundary)
    assert (
        saved.execution_state
        == fixture.application.snapshot_store.load_exact(original).execution_state
    )
    again, _ = restore_fixture(
        resumed_fixture, boundary, training_backbone_config, public_identity=target
    )
    again.training_loop.backbone.fixture_library = again.library
    collected = []
    collect = again.training_loop._collector.collect

    async def capture(**kwargs):
        result = await collect(**kwargs)
        collected.append(result)
        return result

    monkeypatch.setattr(again.training_loop._collector, "collect", capture)
    asyncio.run(again.evolution_loop.run(fixture.run_plan, maximum_steps_this_attempt=1))
    assert again.training_loop.optimizer_step == 2
    assert again.training_loop.task_provider.runtime_state.cursor == 6
    assert [a.manifest.sampling_coordinate.sequence_position for a in collected[0].artifacts] == [
        4,
        5,
    ]
    assert (
        again.projections.posterior_provenance.batches[:1]
        == saved.execution_state.projections.posterior_provenance.batches
    )
