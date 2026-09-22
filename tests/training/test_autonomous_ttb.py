"""Pure-TTB input branch: no changes to sampled outcomes or learning targets."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from skillev_private.benchmarks.protocol_v13_seven_training import (
    SEVEN_TRAINING_DOMAINS,
    seven_domain_training_trajectories,
)
from skillev_private.benchmarks.protocol_v13_training import build_protocol13_training_records
from skillev_private.experiments.autonomous_ttb import (
    PROTOCOL,
    autonomous_training_sources,
    require_autonomous_initialization,
    source_coverage_report,
)
from skillev_private.experiments.bayesian_training_config import BayesianFormalConfig
from skillev_private.experiments.skill_practice import CONTEXT_KEY

from skillev.evolution.cold_start_config import ColdStartConfig
from skillev.experiments._evolution_preflight_seed import planned_seed_documents
from tests.benchmarks.test_protocol_v13_training import _sources

CONFIG = Path("configs/training/bayesianimprove_autonomous_ttb.yaml")


def test_new_protocol_is_explicit_and_legacy_runs_are_unchanged(tmp_path):
    candidate = BayesianFormalConfig.load(CONFIG)
    assert candidate.learning_protocol == PROTOCOL
    assert candidate.cold_start is None
    assert candidate.run_plan.phase_search_steps == 249
    assert candidate.batch_size == 28
    assert candidate.reasoning_modes["aime-2026"]
    application = candidate.application_config("pure-ttb")
    assert application.evolution.cold_start is None
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(candidate.to_value()))
    assert BayesianFormalConfig.load(path) == candidate
    legacy = BayesianFormalConfig.load(
        Path("configs/training/bayesianimprove_skill_learning_candidate.yaml")
    )
    assert legacy.cold_start is not None
    assert "learning_protocol" not in legacy.to_value()
    assert candidate.condition != replace(candidate, learning_protocol=None).condition
    for changes in (
        {"cold_start": ColdStartConfig()},
        {"skill_exposure": "catalog-then-read@2"},
        {"reasoning_tool_catalog": False},
        {"steps": 5},
        {"learning_protocol": "teacher-loss"},
    ):
        with pytest.raises(ValueError):
            replace(candidate, **changes)


def test_saved_specialized_warmup_is_not_a_pure_ttb_initialization(tmp_path):
    candidate = BayesianFormalConfig.load(CONFIG)
    preparation = tmp_path / "preparation.json"
    original = {"format": "skillev-private-protocol13-training-debug-preparation@1"}
    preparation.write_text(json.dumps(original))
    require_autonomous_initialization(candidate, preparation)
    for changed in (
        {**original, "initialization": {"kind": "skill-use-warmup"}},
        {"format": "skillev-private-warmup-ttb-preparation@2"},
    ):
        preparation.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            require_autonomous_initialization(candidate, preparation)
        require_autonomous_initialization(replace(candidate, learning_protocol=None), preparation)


def test_proactive_catalog_is_only_an_explicit_prompt_condition(tmp_path):
    from skillev_private.experiments.bayesian_condition_transition import resume_condition
    from skillev_private.experiments.fresh_restart import require_fresh_interface

    before = BayesianFormalConfig.load(CONFIG)
    after = BayesianFormalConfig.load(
        Path("configs/training/bayesianimprove_autonomous_ttb_proactive_skills.yaml")
    )
    assert replace(after, skill_exposure=before.skill_exposure) == before
    assert before.condition != after.condition
    require_fresh_interface(after)
    (tmp_path / "formal-config.json").write_text(json.dumps(before.to_value()))
    with pytest.raises(ValueError):
        resume_condition(tmp_path, after, allow_new_horizons=False)
    assert (
        resume_condition(tmp_path, after, allow_new_horizons=False, allow_catalog_read=True)
        == before
    )
    records, declaration = source_fixture()
    assert autonomous_training_sources(before, records, declaration) == autonomous_training_sources(
        after, records, declaration
    )


def source_fixture():
    records = tuple(
        {
            (r.episode.benchmark, r.episode.source_id): r
            for r in build_protocol13_training_records(_sources())
            if r.episode.benchmark in SEVEN_TRAINING_DOMAINS
        }.values()
    )
    rows = [
        {
            "benchmark": r.episode.benchmark.value,
            "source_id": r.episode.source_id,
            "role": "procedure-applicable" if index % 2 else "direct-control",
            "public_basis": "Synthetic public task contract, no result inspected.",
            "method_family": "boundary-checking",
        }
        for index, r in enumerate(records)
    ]
    return records, {
        "autonomous_ttb_sources": {
            "format": "public-task-needs@1",
            "ordered_sources": list(reversed(rows)),
            "source_aliases": {},
            "excluded_sources": {"iid": [], "development": [], "quality": []},
        }
    }


def test_public_schedule_preserves_tasks_budgets_and_full_batch():
    candidate = BayesianFormalConfig.load(CONFIG)
    records, declaration = source_fixture()
    ordered = autonomous_training_sources(candidate, records, declaration)
    assert ordered == tuple(reversed(records))
    assert all(CONTEXT_KEY not in r.input.public_context for r in ordered)
    expanded = seven_domain_training_trajectories(ordered, steps=250)
    assert len(expanded) == 7000
    assert all(len(expanded[s : s + 28]) == 28 for s in range(0, 7000, 28))
    report = source_coverage_report(declaration)
    assert sum(report["roles"].values()) == len(records)
    assert not report["skill_benefit_established"]
    assert not report["outcome_filtering"]
    assert autonomous_training_sources(replace(candidate, learning_protocol=None), records, None)


def test_acquisition_repeats_do_not_select_between_conflicting_source_inputs():
    candidate = BayesianFormalConfig.load(CONFIG)
    records, declaration = source_fixture()
    repeated = (*records, records[0])
    assert autonomous_training_sources(candidate, repeated, declaration) == tuple(reversed(records))
    changed = replace(records[0], input=replace(records[0].input, query="Changed public task"))
    with pytest.raises(ValueError):
        autonomous_training_sources(candidate, (*records, changed), declaration)


@pytest.mark.parametrize("fault", ["excluded", "teaching", "missing", "duplicate", "role"])
def test_source_plan_rejects_real_isolation_and_teaching_conflicts(fault):
    candidate = BayesianFormalConfig.load(CONFIG)
    records, declaration = source_fixture()
    plan = declaration["autonomous_ttb_sources"]
    first = plan["ordered_sources"][0]
    if fault == "excluded":
        plan["source_aliases"] = {first["benchmark"]: {"seen-alias": first["source_id"]}}
        plan["excluded_sources"]["development"] = [[first["benchmark"], "seen-alias"]]
    elif fault == "teaching":
        last = records[-1]
        records = (
            *records[:-1],
            replace(
                last,
                input=replace(
                    last.input,
                    public_context={**last.input.public_context, CONTEXT_KEY: "teaching"},
                ),
            ),
        )
    elif fault == "missing":
        first["source_id"] = "not-in-the-fixed-training-source-file"
    elif fault == "duplicate":
        plan["ordered_sources"].append(first)
    else:
        first["role"] = "only-if-successful-read"
    with pytest.raises(ValueError):
        autonomous_training_sources(candidate, records, declaration)


def test_new_method_library_retains_old_profiles_and_actual_read_boundary():
    old = planned_seed_documents("public-native-procedures@4")
    new = planned_seed_documents("public-method-cards@5")
    assert len(new) == 7
    assert not {s.manifest.skill_id for s in old} & {s.manifest.skill_id for s in new}
    assert old == planned_seed_documents("public-native-procedures@4")
    contexts = {context for s in new for context in s.applicability.contexts}
    assert contexts == {f"{d.value}:task" for d in SEVEN_TRAINING_DOMAINS}
    for skill in new:
        assert skill.summary
        assert len(skill.instructions) > len(skill.summary)
        assert skill.instructions not in skill.summary
