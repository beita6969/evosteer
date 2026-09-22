"""Paper-level source balance and reproducible 12-step curriculum scheduling."""

from collections import Counter
from dataclasses import replace

import pytest

from skillev.training.task_schedule import CurriculumStage, SourceBalancedTaskSchedule


def records(sources=7, per_source=80):
    return {
        f"record-{source}-{index}": f"source-{source}"
        for source in range(sources)
        for index in range(per_source)
    }


def schedule(manifest=None, *, batch_size=56, seed=17, curriculum=None):
    manifest = records() if manifest is None else manifest
    stages = curriculum or (CurriculumStage(12, tuple(sorted(set(manifest.values())))),)
    return SourceBalancedTaskSchedule(manifest, curriculum=stages, batch_size=batch_size, seed=seed)


def test_default_fifty_six_records_are_balanced_over_explicit_sources():
    manifest = records()
    sampler = schedule(manifest)
    batch = sampler.plan_next()
    assert len(batch.task_ids) == len(set(batch.task_ids)) == 56
    assert Counter(manifest[task] for task in batch.task_ids) == dict(batch.source_counts)
    assert set(dict(batch.source_counts).values()) == {8}
    assert batch.step_index == 1
    assert sampler.plan_next() == batch
    assert sampler.committed_steps == 0
    sampler.commit(batch)
    assert sampler.committed_steps == 1


def test_unequal_pool_sizes_do_not_bias_source_balance_and_remainder_rotates():
    manifest = records(sources=5, per_source=15)
    manifest.update({f"large-extra-{i}": "source-0" for i in range(500)})
    sampler = schedule(manifest)
    totals = Counter()
    for _ in range(5):
        batch = sampler.plan_next()
        counts = dict(batch.source_counts)
        assert max(counts.values()) - min(counts.values()) == 1
        assert len(batch.task_ids) == len(set(batch.task_ids)) == 56
        totals.update(counts)
        sampler.commit(batch)
    assert set(totals.values()) == {56}


def test_explicit_curriculum_applies_only_through_step_twelve():
    stages = (
        CurriculumStage(4, ("source-0", "source-1")),
        CurriculumStage(12, ("source-1", "source-2", "source-3")),
    )
    sampler = schedule(records(sources=4), curriculum=stages)
    for step in range(1, 15):
        batch = sampler.plan_next()
        expected = (
            stages[0].source_ids
            if step <= 4
            else stages[1].source_ids
            if step <= 12
            else (
                "source-0",
                "source-1",
                "source-2",
                "source-3",
            )
        )
        assert batch.active_sources == expected
        assert batch.step_index == step
        sampler.commit(batch)


def test_restore_reproduces_future_batches_and_input_mapping_order_does_not_matter():
    manifest = records()
    first = schedule(manifest)
    for _ in range(13):
        first.commit(first.plan_next())
    restored = schedule(dict(reversed(tuple(manifest.items()))))
    restored.load_state_dict(first.state_dict())
    assert first.configuration_id == restored.configuration_id
    for _ in range(5):
        before = first.plan_next()
        assert restored.plan_next() == before
        first.commit(before)
        restored.commit(restored.plan_next())
    assert first.state_dict() == restored.state_dict()


def test_stale_or_altered_plan_cannot_advance_schedule():
    sampler = schedule()
    batch = sampler.plan_next()
    with pytest.raises(ValueError, match="stale"):
        sampler.commit(replace(batch, task_ids=batch.task_ids[:-1]))
    assert sampler.committed_steps == 0
    sampler.commit(batch)
    with pytest.raises(ValueError, match="stale"):
        sampler.commit(batch)
    assert sampler.committed_steps == 1


def test_capacity_failure_is_explicit_and_never_oversamples_a_small_source():
    with pytest.raises(ValueError, match="distinct records"):
        schedule(records(sources=7, per_source=7))


def test_no_curriculum_or_unknown_stage_is_not_silently_filled_in():
    with pytest.raises(ValueError, match="explicit static curriculum"):
        SourceBalancedTaskSchedule(records(), curriculum=())
    with pytest.raises(ValueError, match="finish at step 12"):
        schedule(curriculum=(CurriculumStage(11, ("source-0",)),))
    with pytest.raises(ValueError, match="unknown source"):
        schedule(curriculum=(CurriculumStage(12, ("invented",)),))
    with pytest.raises(ValueError, match="boundaries"):
        schedule(
            curriculum=(
                CurriculumStage(8, ("source-0",)),
                CurriculumStage(7, ("source-1",)),
                CurriculumStage(12, ("source-2",)),
            )
        )


def test_restore_rejects_changed_sources_seed_or_curriculum_before_mutation():
    original = schedule()
    original.commit(original.plan_next())
    state = original.state_dict()
    with pytest.raises(ValueError, match="differs"):
        schedule(seed=99).load_state_dict(state)
    manifest = records()
    manifest["record-0-0"] = "source-1"
    with pytest.raises(ValueError, match="differs"):
        schedule(manifest).load_state_dict(state)
    different = schedule(curriculum=(CurriculumStage(12, ("source-0", "source-1")),))
    with pytest.raises(ValueError, match="differs"):
        different.load_state_dict(state)
    assert different.committed_steps == 0
    with pytest.raises(RuntimeError, match="fresh"):
        original.load_state_dict(state)


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_invalid_restore_step_count_rejected(count):
    sampler = schedule()
    state = {**sampler.state_dict(), "committed_steps": count}
    with pytest.raises(ValueError, match="nonnegative integer"):
        sampler.load_state_dict(state)
    assert sampler.committed_steps == 0


def test_stage_json_contract_roundtrips_without_inventing_order_or_ratios():
    stage = CurriculumStage.from_value({"through_step": 12, "source_ids": ["a", "b"]})
    assert stage.to_value() == {"through_step": 12, "source_ids": ["a", "b"]}
