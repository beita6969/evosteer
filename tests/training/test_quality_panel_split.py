"""Synthetic-only fixtures for a proposed split; no real pool is read or changed."""

import json
import stat
from collections import Counter
from dataclasses import replace

import pytest
from skillev_private.benchmarks.protocol_v13_seven_training import (
    SEVEN_TRAINING_DOMAINS,
    load_seven_domain_training_sources,
    seven_domain_training_trajectories,
)
from skillev_private.experiments.quality_collection import QualityCollectionBinding
from skillev_private.experiments.quality_panel_split import (
    plan_quality_split,
    source_identity,
    write_quality_split,
)

from skillev.training.quality_gate import QualityGatePolicy, QualityRule
from tests.evaluation.test_native_source_bridge import record

VERIFIERS = (
    "hotpotqa-official-em-f1",
    "triviaqa-official-alias-em-f1",
    "integer-exact",
    "simple-evals-rubric",
    "alfworld-success",
    "evalplus-base-plus",
    "humaneval-native",
)


def source_pool(*, sources=7, repeats=2):
    records = []
    for occurrence in range(repeats):
        for index in range(sources):
            for domain, verifier in zip(SEVEN_TRAINING_DOMAINS, VERIFIERS, strict=True):
                original = record(domain.value, verifier, {"synthetic_private_target": index})
                task_id = f"synthetic/{domain.value}/{index}/{occurrence}"
                records.append(
                    replace(
                        original,
                        episode=replace(
                            original.episode,
                            episode_id=task_id,
                            source_id=f"source-{index}",
                            population_id=f"synthetic-population-{occurrence}",
                            repeat_ordinal=occurrence,
                        ),
                        input=replace(original.input, task_id=task_id),
                    )
                )
    return tuple(records)


def test_split_uses_canonical_sources_not_population_labels_or_targets():
    records = source_pool()
    split = plan_quality_split(records)
    selected = {source_identity(r) for r in split.panel}
    assert len(split.panel) == len(selected) == 28
    assert Counter(r.episode.benchmark for r in split.panel) == dict.fromkeys(
        SEVEN_TRAINING_DOMAINS, 4
    )
    assert len(split.training) == 1750
    assert Counter(r.episode.benchmark for r in split.training) == dict.fromkeys(
        SEVEN_TRAINING_DOMAINS, 250
    )
    assert len(split.heldout_occurrences) == 56
    assert {source_identity(r) for r in split.heldout_occurrences} == selected
    assert {r.episode.population_id for r in split.heldout_occurrences} == {
        "synthetic-population-0",
        "synthetic-population-1",
    }
    for r in records:
        side = (
            split.heldout_occurrences
            if source_identity(r) in selected
            else split.retained_occurrences
        )
        assert r in side
    changed = tuple(
        replace(r, output=replace(r.output, target={"synthetic_private_target": "changed"}))
        for r in records
    )
    assert tuple(source_identity(r) for r in plan_quality_split(changed).panel) == tuple(
        source_identity(r) for r in split.panel
    )
    reordered = plan_quality_split(tuple(reversed(records)))
    assert tuple(source_identity(r) for r in reordered.panel) == tuple(
        source_identity(r) for r in split.panel
    )
    assert all(r.episode.population_id == "synthetic-population-1" for r in reordered.panel)
    assert "synthetic_private_target" not in json.dumps(split.summary)
    for summary in split.summary["domains"].values():
        assert summary["original_sources"] == 7
        assert summary["retained_sources"] == 3
        assert summary["heldout_occurrences"] == 8
        assert summary["scheduled_training_occurrences"] == 250
        assert summary["changed_source_positions_vs_original_lane_cycle"] > 0


def test_training_filters_original_lane_then_cycles_without_rewriting_native_bindings():
    split = plan_quality_split(source_pool())
    for domain in SEVEN_TRAINING_DOMAINS:
        lane = tuple(r for r in split.retained_occurrences if r.episode.benchmark is domain)
        generated = tuple(r for r in split.training if r.episode.benchmark is domain)
        repeats = Counter()
        for step, r in enumerate(generated):
            original = lane[step % len(lane)]
            assert source_identity(r) == source_identity(original)
            assert r.episode.population_id == original.episode.population_id
            assert replace(r.input, task_id=original.input.task_id) == original.input
            assert r.output == original.output
            assert r.episode.repeat_ordinal == repeats[source_identity(r)]
            assert r.episode.optimizer_step == step + 1
            repeats[source_identity(r)] += 1
    assert len({r.input.task_id for r in split.training}) == 1750
    trajectories = seven_domain_training_trajectories(split.training, steps=250)
    assert len(trajectories) == 7000
    for step in (0, 124, 249):
        expected = tuple(source_identity(r) for r in split.training[step * 7 : (step + 1) * 7])
        batch = trajectories[step * 28 : (step + 1) * 28]
        assert tuple(source_identity(r) for r in batch) == expected * 4


@pytest.mark.parametrize("sources", [1, 4])
def test_repeated_population_aliases_cannot_fill_insufficient_independent_sources(sources):
    with pytest.raises(ValueError):
        plan_quality_split(source_pool(sources=sources, repeats=6))


def test_rejects_missing_domains_and_undeclared_selection_conditions():
    records = source_pool()
    with pytest.raises(ValueError):
        plan_quality_split(
            tuple(r for r in records if r.episode.benchmark is not SEVEN_TRAINING_DOMAINS[-1])
        )
    for changes in (
        {"seed": 1},
        {"seed": False},
        {"sources_per_domain": 3},
        {"training_occurrences_per_domain": 249},
    ):
        with pytest.raises(ValueError):
            plan_quality_split(records, **changes)


def test_final_sources_and_overlap_checks_ignore_population_relabels():
    records = source_pool()
    baseline = plan_quality_split(records)
    first = baseline.panel[0]
    excluded = frozenset(
        {(first.episode.benchmark.value, "another-population", first.episode.source_id)}
    )
    with pytest.raises(ValueError):
        plan_quality_split(records, final_evaluation_sources=excluded)
    training_source = baseline.training[0].episode
    training_excluded = frozenset(
        {(training_source.benchmark.value, "relabeled-final", training_source.source_id)}
    )
    with pytest.raises(ValueError):
        plan_quality_split(records, final_evaluation_sources=training_excluded)
    with pytest.raises(ValueError):
        replace(baseline, final_evaluation_sources=training_excluded).require_disjoint()
    nonoverlapping = frozenset({("hotpotqa", "final", "not-in-training-pool")})
    assert (
        plan_quality_split(records, final_evaluation_sources=nonoverlapping).panel == baseline.panel
    )
    with pytest.raises(ValueError):
        replace(baseline, final_evaluation_sources=excluded).require_disjoint()
    alias = replace(first, episode=replace(first.episode, population_id="new-training-label"))
    with pytest.raises(ValueError):
        replace(baseline, training=(*baseline.training, alias)).require_disjoint()


def test_private_writer_roundtrips_through_existing_collection_binding(tmp_path):
    records = source_pool()
    split = plan_quality_split(records)
    root = tmp_path / "private-synthetic-split"
    assert not root.exists()  # planning has no filesystem effects
    manifest = write_quality_split(
        split, root, panel_id="synthetic-panel", condition_id="synthetic-condition"
    )
    training = load_seven_domain_training_sources(root / "training.jsonl")
    policy = QualityGatePolicy(
        "synthetic-rule",
        "synthetic-panel",
        "synthetic-condition",
        5,
        28,
        2,
        (QualityRule("synthetic-metric", 0.5, 0.1),),
    )
    binding = QualityCollectionBinding.load(
        manifest, training=training, policy=policy, batch_size=28
    )
    assert binding.records == split.panel
    assert training == split.training
    assert (
        load_seven_domain_training_sources(root / "heldout-occurrences.jsonl")
        == split.heldout_occurrences
    )
    assert (
        load_seven_domain_training_sources(root / "retained-occurrences.jsonl")
        == split.retained_occurrences
    )
    assert "synthetic_private_target" not in manifest.read_text()
    assert "synthetic_private_target" not in (root / "summary.json").read_text()
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in root.iterdir())
    with pytest.raises(FileExistsError):
        write_quality_split(
            split, root, panel_id="synthetic-panel", condition_id="synthetic-condition"
        )
    # The complete original records remain on their side and the caller's pool is unchanged.
    assert source_pool() == records


def test_writer_rejects_cross_population_source_overlap_before_any_output(tmp_path):
    split = plan_quality_split(source_pool())
    alias = replace(split.panel[0], episode=replace(split.panel[0].episode, population_id="alias"))
    broken = replace(split, training=(*split.training, alias))
    root = tmp_path / "must-not-exist"
    with pytest.raises(ValueError):
        write_quality_split(
            broken, root, panel_id="synthetic-panel", condition_id="synthetic-condition"
        )
    assert not root.exists()
