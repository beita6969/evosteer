"""Private declarations bind actual source order, not task bodies or new IDs."""

import json
from dataclasses import replace

import pytest
from skillev_private.experiments.bayesian_improve_training import FormalTrainingBindings
from skillev_private.experiments.training_data_condition import (
    data_condition_scientific,
    require_same_run_data_condition,
)

from skillev.training.run_condition import EffectiveRunCondition
from tests.benchmarks.test_protocol_v13_training_sessions import _record

DECLARATION = {"selection": "synthetic-fixed-holdout", "seed": 0, "sources_per_domain": 4}


def selected_records():
    first = _record()
    second = replace(
        first,
        episode=replace(first.episode, episode_id="second-occurrence", source_id="second-source"),
        input=replace(first.input, task_id="second-occurrence"),
    )
    return first, second


def effective(declaration=DECLARATION, records=None, *, horizon=20):
    records = selected_records() if records is None else records
    return EffectiveRunCondition.create(
        condition_id="synthetic-condition",
        scientific={"horizon": horizon, **data_condition_scientific(declaration, records)},
        execution={"placement": "synthetic"},
    )


def write_previous(root, condition, name="effective-condition-process-1.json"):
    path = root / name
    path.write_text(json.dumps(condition.to_value()))
    return path


def test_none_preserves_original_scientific_projection_exactly():
    original = EffectiveRunCondition.create(
        condition_id="synthetic-condition",
        scientific={"horizon": 20},
        execution={"placement": "synthetic"},
    )
    assert data_condition_scientific(None, selected_records()) == {}
    assert effective(None).to_value() == original.to_value()
    assert effective(None).scientific_json == original.scientific_json


def test_projection_uses_only_actual_ordered_coordinates_and_copies_declaration():
    selected = selected_records()
    declaration = {"selection": {"name": "synthetic-fixed"}, "seed": 0}
    value = data_condition_scientific(declaration, selected)
    rows = value["data_condition"]["ordered_selected_sources"]
    assert [r["occurrence_id"] for r in rows] == [r.episode.episode_id for r in selected]
    assert [r["source_question_id"] for r in rows] == [r.episode.source_id for r in selected]
    assert [r["population_id"] for r in rows] == [r.episode.population_id for r in selected]
    assert [r["benchmark_id"] for r in rows] == [r.episode.benchmark.value for r in selected]
    assert selected[0].input.query not in json.dumps(value)
    assert "private answer" not in json.dumps(value)
    assert "accepted_answers" not in json.dumps(value)
    declaration["selection"]["name"] = "changed later"
    assert value["data_condition"]["declaration"]["selection"]["name"] == "synthetic-fixed"


@pytest.mark.parametrize(
    "change", ["declaration", "order", "source", "population", "occurrence", "removed"]
)
def test_resume_rejects_changed_data_through_existing_science_comparison(tmp_path, change):
    previous = effective()
    path = write_previous(tmp_path, previous)
    before = path.read_text()
    records = selected_records()
    declaration = DECLARATION
    if change == "declaration":
        declaration = {**DECLARATION, "selection": "different"}
    elif change == "order":
        records = tuple(reversed(records))
    elif change == "removed":
        declaration = None
    else:
        field = {"source": "source_id", "population": "population_id", "occurrence": "episode_id"}[
            change
        ]
        first = replace(
            records[0],
            episode=replace(records[0].episode, **{field: "different"}),
            input=replace(records[0].input, task_id="different")
            if change == "occurrence"
            else records[0].input,
        )
        records = (first, records[1])
    current = effective(
        declaration, records, horizon=21
    )  # a horizon transition cannot hide data drift
    with pytest.raises(ValueError):
        require_same_run_data_condition(tmp_path, current)
    assert path.read_text() == before
    assert tuple(tmp_path.iterdir()) == (path,)


def test_resume_cannot_add_data_to_legacy_run_but_old_none_and_horizon_behavior_remain(tmp_path):
    require_same_run_data_condition(tmp_path, effective(None))
    with pytest.raises(ValueError):
        require_same_run_data_condition(tmp_path, effective())
    write_previous(tmp_path, effective(None))
    require_same_run_data_condition(tmp_path, effective(None, horizon=21))
    with pytest.raises(ValueError):
        require_same_run_data_condition(tmp_path, effective())


def test_resume_checks_only_current_root_process_pattern_and_all_existing_declarations(tmp_path):
    write_previous(tmp_path, effective())
    write_previous(tmp_path, effective(), "effective-condition-process-2.json")
    write_previous(tmp_path, effective({"different": True}), "unrelated.json")
    nested = tmp_path / "other-run"
    nested.mkdir()
    write_previous(nested, effective({"different": True}))
    require_same_run_data_condition(tmp_path, effective(horizon=21))
    write_previous(tmp_path, effective({"different": True}), "effective-condition-process-3.json")
    with pytest.raises(ValueError):
        require_same_run_data_condition(tmp_path, effective())


def test_binding_load_accepts_inline_object_and_none_but_never_dereferences_a_path(tmp_path):
    value = dict.fromkeys(
        (
            "preparation",
            "dataset",
            "deployments",
            "evalplus_python",
            "evalplus_source_root",
            "endpoint",
            "base_model",
            "adapter_namespace",
            "serving_gpu_uuid",
        ),
        "synthetic-private-placeholder",
    )
    value["training_gpu_uuids"] = ["synthetic-gpu"]
    path = tmp_path / "binding.json"
    path.write_text(json.dumps(value))
    assert FormalTrainingBindings.load(path).data_condition is None
    value["data_condition"] = DECLARATION
    path.write_text(json.dumps(value))
    assert FormalTrainingBindings.load(path).data_condition == DECLARATION
    value["data_condition"] = str(tmp_path / "must-not-read.json")
    path.write_text(json.dumps(value))
    with pytest.raises(TypeError):
        FormalTrainingBindings.load(path)


def test_explicit_domain_history_compares_each_original_source_schedule(tmp_path):
    before = effective()
    write_previous(tmp_path, before)
    after = EffectiveRunCondition.create(
        condition_id="six-domain",
        scientific=data_condition_scientific(DECLARATION, selected_records()[:1]),
        execution={},
    )
    declared = {
        before.condition_id: before.scientific["data_condition"],
        after.condition_id: after.scientific["data_condition"],
    }
    require_same_run_data_condition(tmp_path, after, declared_data_by_condition=declared)
    with pytest.raises(ValueError):
        require_same_run_data_condition(
            tmp_path,
            after,
            declared_data_by_condition={after.condition_id: after.scientific["data_condition"]},
        )
