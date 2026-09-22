import json
import math
from dataclasses import FrozenInstanceError, replace

import pytest

from skillev.training.reference_statistics import (
    ReferenceObservation,
    ReferenceStatistics,
    ReferenceStatisticsConfig,
    StructureReferenceObservation,
    StructureVisit,
)


def observation(identifier, task, family, reward, **kwargs):
    return ReferenceObservation(identifier, task, family, reward, "context-1", **kwargs)


def test_exact_hierarchy_leaves_only_task_level_out():
    store = ReferenceStatistics("context-1")
    snapshot = store.snapshot(
        "batch-1",
        (
            observation("a", "q", "math", 1),
            observation("b", "q", "math", 0),
            observation("c", "other", "code", 0),
        ),
    )
    category = (1 + 2 * (1 / 3)) / (2 + 2)
    expected = (0 + 0.5 * category) / (1 + 0.5)
    assert snapshot.task_rate("q", "math", exclude_observation_id="a") == pytest.approx(expected)
    assert snapshot.task_rate("q", "math") == pytest.approx((1 + 0.5 * category) / 2.5)
    with pytest.raises(ValueError):
        snapshot.task_rate("q", "math", exclude_observation_id="c")


def test_optional_prior_is_capped_and_cold_start_defaults_to_half():
    snapshot = ReferenceStatistics("context-1").snapshot("batch-1")
    assert snapshot.task_rate("new", "math") == 0.5
    assert snapshot.task_rate("new", "math", prior_rate=1, prior_count=200) == pytest.approx(
        22 / 24
    )
    assert snapshot.task_anchor("new", "math") == pytest.approx(math.log1p(math.expm1(1) / 2))


def test_reference_only_context_and_duplicate_boundaries():
    for kwargs in (
        {"source": "paired_reference"},
        {"source": "policy"},
        {"healthy": False},
        {"legal": False},
    ):
        with pytest.raises(ValueError):
            observation("a", "q", "math", 1, **kwargs)
    store = ReferenceStatistics("context-1")
    row = observation("a", "q", "math", 1)
    with pytest.raises(ValueError):
        store.snapshot("b", [row, row])
    with pytest.raises(ValueError):
        store.snapshot("b", [ReferenceObservation("x", "q", "math", 0, "other-context")])


def test_structure_uses_previous_batch_ratio_average_and_coarse_fallback():
    config = ReferenceStatisticsConfig(
        structure_minimum=2, structure_pseudocount=2, correction_bound=2
    )
    store = ReferenceStatistics("context-1", config)
    a = StructureVisit("graph-A", 2, 1)
    b = StructureVisit("graph-B", 2, 1)
    rows = (
        observation("a", "q", "math", 1, visits=(a,)),
        observation("b", "q", "math", 0, visits=(b,)),
    )
    current = store.snapshot("batch-1", rows)
    assert current.structure_correction(a) == 0.0
    anchors = [
        current.task_anchor("q", "math", exclude_observation_id=row.observation_id) for row in rows
    ]
    expected = math.log1p(
        sum(math.expm1(row.reward - anchor) for row, anchor in zip(rows, anchors, strict=True)) / 4
    )
    store.commit_batch(current)
    later = store.snapshot("batch-2")
    assert later.structure_correction(a) == pytest.approx(expected)
    assert later.structure_correction(StructureVisit("unseen", 2, 1)) == pytest.approx(expected)
    assert later.structure_correction(StructureVisit("graph-A", 2, 1, stopped=True)) == 0.0
    assert current.structure_correction(a) == 0.0
    with pytest.raises(FrozenInstanceError):
        current.batch_id = "changed"
    with pytest.raises(ValueError):
        store.commit_batch(current)


def test_serialization_roundtrip_and_context_restoration():
    store = ReferenceStatistics("context-1", ReferenceStatisticsConfig(structure_minimum=1))
    snapshot = store.snapshot(
        "batch-1", [observation("a", "q", "math", 0.6, visits=(StructureVisit("g", 1, 0),))]
    )
    store.commit_batch(snapshot)
    restored = ReferenceStatistics.from_value(json.loads(json.dumps(store.to_value())))
    assert restored.to_value() == store.to_value()
    assert restored.snapshot("batch-2") == store.snapshot("batch-2")
    with pytest.raises(ValueError):
        restored.snapshot("batch-1")


def test_conflicting_task_categories_or_priors_rejected_before_commit():
    store = ReferenceStatistics("context-1")
    with pytest.raises(ValueError):
        store.snapshot("b", [observation("a", "q", "math", 1), observation("b", "q", "code", 0)])
    with pytest.raises(ValueError):
        store.snapshot(
            "b",
            [
                observation("a", "q", "math", 1, prior_rate=1, prior_count=3),
                observation("b", "q", "math", 0, prior_rate=0, prior_count=3),
            ],
        )


def paired_structure(source, reward, visits, **kwargs):
    return StructureReferenceObservation(
        source,
        "q",
        "math",
        reward,
        "context-1",
        visits,
        source=source,
        pair_id="pair",
        candidate_id="candidate",
        forced_prefix_indices=(0,),
        **kwargs,
    )


def test_paired_forced_prefixes_enter_only_lagged_structural_ratio_average():
    config = ReferenceStatisticsConfig(
        structure_minimum=1, structure_pseudocount=2, correction_bound=2
    )
    store = ReferenceStatistics("context-1", config)
    visit = StructureVisit("same-graph", 1, 0)
    natural = (
        observation("a", "q", "math", 0.25, visits=(visit,)),
        observation("b", "q", "math", 0.75, visits=(visit,)),
    )
    pairs = (
        paired_structure("paired_treatment", 1, (visit, visit)),
        paired_structure("paired_control", 0, (visit,)),
    )
    baseline = store.snapshot("baseline-inspect", natural)
    current = store.snapshot("batch-1", natural, structure_observations=pairs)
    assert current.observations == baseline.observations
    assert current.task_anchor("q", "math") == baseline.task_anchor("q", "math")
    assert current.structure_correction(visit) == 0
    root = current.task_anchor("q", "math")
    terms = [
        math.expm1(
            math.log1p(math.expm1(1) * row.reward)
            - current.task_anchor("q", "math", exclude_observation_id=row.observation_id)
        )
        for row in natural
    ]
    terms += [math.expm1(1 - root)] * 2 + [math.expm1(-root)]
    store.commit_batch(current)
    later = store.snapshot("batch-2")
    assert later.task_anchor("q", "math") == baseline.task_anchor("q", "math")
    assert later.structure_correction(visit) == pytest.approx(math.log1p(sum(terms) / 7))
    assert all(bucket.visits == 5 for bucket in later.structure_buckets)
    assert current.structure_correction(visit) == 0
    saved = store.to_value()
    assert len(saved["observations"]) == 2
    assert len(saved["structure_observations"]) == 4
    assert len(saved["structure_measurements"]) == 4
    assert saved["structure_observations"][2]["forced_prefix_indices"] == (0,)
    restored = ReferenceStatistics.from_value(json.loads(json.dumps(saved)))
    assert restored.to_value() == saved
    assert restored.snapshot("batch-2") == later


def test_natural_only_structure_control_retains_but_does_not_pool_paired_provenance():
    store = ReferenceStatistics(
        "context-1",
        ReferenceStatisticsConfig(structure_source_mode="natural_only", structure_minimum=1),
    )
    visit = StructureVisit("g", 1, 0)
    row = observation("a", "q", "math", 0.4, visits=(visit,))
    paired = paired_structure("paired_treatment", 1, (visit, visit))
    without = ReferenceStatistics("context-1", store.config)
    without.commit_batch(without.snapshot("batch", (row,)))
    store.commit_batch(store.snapshot("batch", (row,), structure_observations=(paired,)))
    assert store.snapshot("next").structure_buckets == without.snapshot("next").structure_buckets
    assert store.snapshot("next").observations == without.snapshot("next").observations
    saved = store.to_value()
    assert len(saved["structure_observations"]) == 2
    assert len(saved["structure_measurements"]) == 1
    assert ReferenceStatistics.from_value(json.loads(json.dumps(saved))).to_value() == saved


@pytest.mark.parametrize("field", ["root_anchor", "excess_ratio", "bucket", "source"])
def test_structural_provenance_and_measurements_are_replayed_before_restore(field):
    store = ReferenceStatistics("context-1")
    paired = paired_structure("paired_treatment", 1, (StructureVisit("g", 1, 0),))
    store.commit_batch(store.snapshot("batch", structure_observations=(paired,)))
    saved = json.loads(json.dumps(store.to_value()))
    if field in {"root_anchor", "excess_ratio"}:
        saved["structure_measurements"][0][field] += 0.25
    elif field == "bucket":
        saved["structure_buckets"][0]["visits"] += 1
    else:
        saved["structure_observations"][0]["source"] = "current"
    with pytest.raises(ValueError):
        ReferenceStatistics.from_value(saved)


def test_structural_inputs_reject_duplicate_tasks_contexts_and_missing_forced_prefix():
    store = ReferenceStatistics("context-1")
    row = paired_structure("paired_treatment", 1, (StructureVisit("g", 1, 0),))
    with pytest.raises(ValueError, match="unique"):
        store.snapshot("batch", structure_observations=(row, row))
    with pytest.raises(ValueError, match="contexts"):
        store.snapshot("batch", structure_observations=(replace(row, context_version="other"),))
    with pytest.raises(ValueError, match="category"):
        store.snapshot(
            "batch", (observation("natural", "q", "code", 0),), structure_observations=(row,)
        )
    with pytest.raises(ValueError, match="forced first"):
        replace(row, forced_prefix_indices=())
    store.commit_batch(store.snapshot("batch", structure_observations=(row,)))
    with pytest.raises(ValueError, match="unique"):
        store.snapshot("later", structure_observations=(row,))


def test_paired_only_control_structure_uses_external_task_prior_without_creating_baseline_counts():
    config = ReferenceStatisticsConfig(
        structure_minimum=1,
        structure_pseudocount=2,
        correction_bound=2,
    )
    store = ReferenceStatistics("context-1", config)
    visit = StructureVisit("control-prefix", 1, 0)
    paired = paired_structure(
        "paired_control",
        0.25,
        (visit, visit),
        prior_rate=0.9,
        prior_count=100,
    )
    snapshot = store.snapshot("control-batch", structure_observations=(paired,))
    # No natural trajectory exists in this control-menu context. The category
    # mean remains 0.5, but the supplied prior contributes its capped 20 counts.
    expected_rate = (20 * 0.9 + 4 * 0.5) / 24
    expected_root = math.log1p(math.expm1(config.beta) * expected_rate)
    expected_reward = math.log1p(math.expm1(config.beta) * paired.reward)
    expected_excess = math.expm1(expected_reward - expected_root)
    assert snapshot.task_anchor(
        "q",
        "math",
        prior_rate=0.9,
        prior_count=100,
    ) == pytest.approx(expected_root)
    assert snapshot.observations == ()
    assert snapshot.task_rate("q", "math") == 0.5
    store.commit_batch(snapshot)
    saved = store.to_value()
    assert saved["observations"] == []
    assert saved["structure_measurements"][0]["root_anchor"] == pytest.approx(expected_root)
    assert saved["structure_measurements"][0]["excess_ratio"] == pytest.approx(expected_excess)
    assert saved["structure_observations"][0]["prior_rate"] == 0.9
    assert saved["structure_observations"][0]["prior_count"] == 100
    expected_correction = math.log1p(2 * expected_excess / (2 + config.structure_pseudocount))
    later = store.snapshot("next")
    assert later.structure_correction(visit) == pytest.approx(expected_correction)
    assert later.task_rate("q", "math") == 0.5
    restored = ReferenceStatistics.from_value(json.loads(json.dumps(saved)))
    assert restored.to_value() == saved
    assert restored.snapshot("next") == later


def test_natural_conversion_carries_prior_and_agrees_with_paired_task_metadata():
    store = ReferenceStatistics("context-1")
    visit = StructureVisit("g", 1, 0)
    natural = observation(
        "natural", "q", "math", 0.25, visits=(visit,), prior_rate=0.8, prior_count=3
    )
    paired = paired_structure("paired_treatment", 1, (visit,), prior_rate=0.8, prior_count=3)
    snapshot = store.snapshot("batch", (natural,), structure_observations=(paired,))
    converted = snapshot.structure_observations[0]
    assert converted.prior_rate == natural.prior_rate
    assert converted.prior_count == natural.prior_count
    natural_anchor = snapshot.task_anchor(
        "q", "math", exclude_observation_id="natural", prior_rate=0.8, prior_count=3
    )
    paired_anchor = snapshot.task_anchor("q", "math", prior_rate=0.8, prior_count=3)
    store.commit_batch(snapshot)
    measurements = store.to_value()["structure_measurements"]
    assert measurements[0]["root_anchor"] == pytest.approx(natural_anchor)
    assert measurements[1]["root_anchor"] == pytest.approx(paired_anchor)
    assert len(store.to_value()["observations"]) == 1


@pytest.mark.parametrize(
    "changed",
    [
        {"prior_rate": 0.7, "prior_count": 3},
        {"prior_rate": 0.8, "prior_count": 4},
        {"prior_rate": None, "prior_count": 0},
    ],
)
def test_natural_and_paired_prior_conflicts_rejected_before_any_statistics_mutation(changed):
    store = ReferenceStatistics("context-1")
    visit = StructureVisit("g", 1, 0)
    natural = observation("natural", "q", "math", 0.5, prior_rate=0.8, prior_count=3)
    paired = paired_structure("paired_control", 0, (visit,), **changed)
    before = store.to_value()
    with pytest.raises(ValueError, match="supplied prior"):
        store.snapshot("batch", (natural,), structure_observations=(paired,))
    assert store.to_value() == before
    store.commit_batch(store.snapshot("natural", (natural,)))
    before = store.to_value()
    with pytest.raises(ValueError, match="supplied prior"):
        store.snapshot("later", structure_observations=(paired,))
    assert store.to_value() == before


def test_prior_consistency_is_enforced_across_structural_only_batches_and_restore():
    store = ReferenceStatistics("context-1")
    visit = StructureVisit("g", 1, 0)
    first = paired_structure("paired_control", 0, (visit,), prior_rate=0.7, prior_count=5)
    store.commit_batch(store.snapshot("first", structure_observations=(first,)))
    restored = ReferenceStatistics.from_value(json.loads(json.dumps(store.to_value())))
    conflicting = paired_structure("paired_treatment", 1, (visit,), prior_rate=0.6, prior_count=5)
    with pytest.raises(ValueError, match="supplied prior"):
        restored.snapshot("second", structure_observations=(conflicting,))
    with pytest.raises(ValueError, match="supplied prior"):
        restored.snapshot("second", (observation("natural", "q", "math", 1),))


@pytest.mark.parametrize(
    "prior",
    [
        {"prior_rate": None, "prior_count": 1},
        {"prior_rate": 1.1, "prior_count": 1},
        {"prior_rate": 0.5, "prior_count": -1},
    ],
)
def test_structural_prior_fields_use_the_same_validation_as_natural_observations(prior):
    with pytest.raises(ValueError):
        paired_structure("paired_control", 0, (StructureVisit("g", 1, 0),), **prior)
