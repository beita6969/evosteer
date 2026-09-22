"""Telemetry distinguishes exposure, calls, repeated questions and unknown history."""

import math
from dataclasses import replace

import pytest

from skillev.rollout.context import INITIAL_CONTEXT_FORMAT_VERSION
from skillev.training.calibration_reporting import prequential_calibration
from skillev.training.coverage_reporting import batch_coverage
from skillev.training.posterior_state import PosteriorEvidenceBatch
from tests.training.test_bayesian_chain import pipeline, restore
from tests.v3_helpers import make_artifact, make_source


def artifact(name, *, question="one", calls=True, success=True):
    item = make_artifact(
        name, skills_by_step=(("skill-alpha",) if calls else (),), reward_success=success
    )
    h0 = replace(
        item.record.initial_context,
        assembler_version=INITIAL_CONTEXT_FORMAT_VERSION,
        retrieved_skill_ids=("skill-alpha",),
        active_skill_ids=("skill-alpha",),
        meta={
            **item.record.initial_context.meta,
            "retrieval_inclusions": [
                {"skill_id": "skill-alpha", "inclusion_reason": "applicability-match"}
            ],
        },
    )
    reward = replace(
        item.record.reward,
        native_payload={
            "training_evidence_source": {
                "benchmark_id": "mbpp-plus",
                "population_id": "synthetic",
                "source_question_id": question,
            },
            "private_answer": "ANSWER_MUST_NOT_ENTER_REPORT",
        },
    )
    return replace(
        item,
        record=replace(item.record, reward=reward, initial_context=h0),
        initial_context=replace(item.initial_context, contract=h0),
        manifest=replace(item.manifest, assembler_version=INITIAL_CONTEXT_FORMAT_VERSION),
    )


def test_coverage_and_errors_do_not_confuse_rollouts_questions_exposure_or_mass():
    projection, _ = pipeline()
    source = make_source(
        artifacts=(artifact("a"), artifact("b", success=False), artifact("c", calls=False)),
        importances=((math.log(3),), (0.0,), (0.0,)),
    )
    projection.commit(projection.preview(source))
    batch = projection.posterior_provenance.batches[0]
    report = batch_coverage(batch)["benchmarks"]["mbpp-plus"]
    assert report["trajectory_count"] == 3
    assert report["distinct_source_question_count"] == 1
    assert report["body_visible_trajectory_count"] == 3
    assert report["invoking_trajectory_count"] == 2
    assert report["posterior_update_event_count"] == 2
    assert report["updated_cell_count"] == 1
    assert report["posterior_weight_mass"] == pytest.approx(2)
    predictions = prequential_calibration(projection.posterior_provenance)
    cell = predictions[0]["cells"][0]
    assert cell["distinct_trajectory_count"] == 2
    assert cell["distinct_source_question_count"] == 1
    assert cell["largest_event_weight_share"] == pytest.approx(0.75)
    assert cell["event_weight_concentration"] == pytest.approx(0.625)
    assert cell["source_question_grouped_error"]["group_count"] == 1
    assert cell["brier_score"] == 0.25
    assert "ANSWER_MUST_NOT_ENTER_REPORT" not in repr(batch.to_value())
    recovered = restore(projection)
    assert prequential_calibration(recovered.posterior_provenance) == predictions
    assert batch_coverage(recovered.posterior_provenance.batches[0]) == batch_coverage(batch)


def test_old_source_coordinates_remain_unknown_and_empty_updates_are_reported():
    projection, _ = pipeline()
    projection.commit(projection.preview(make_source(artifacts=(artifact("empty", calls=False),))))
    batch = projection.posterior_provenance.batches[0]
    coverage = batch_coverage(batch)["benchmarks"]["mbpp-plus"]
    assert coverage["body_visible_trajectory_count"] == 1
    assert coverage["posterior_update_event_count"] == 0
    assert projection.calibration_cells == ()
    raw = batch.to_value()
    del raw["trajectory_contexts"]
    old = PosteriorEvidenceBatch.from_value(raw)
    assert batch_coverage(old)["metadata_missing_trajectory_count"] == 1
    assert old.posterior == batch.posterior


def test_golden_full_batch_is_invariant_to_worker_order_and_restore():
    artifacts = (
        make_artifact(
            "success", skills_by_step=(("skill-alpha",), ("skill-alpha",)), reward_success=True
        ),
        make_artifact("failure", reward_success=False),
    )
    source = make_source(
        artifacts=artifacts, importances=((math.log(2), math.log(3)), (math.log(4),))
    )
    projection, _ = pipeline()
    transition = projection.preview(source)
    assert (
        projection.preview(replace(source, edge_records=source.edge_records[::-1])).posterior_batch
        == transition.posterior_batch
    )
    projection.commit(transition)
    other, _ = pipeline()
    other.commit(
        other.preview(
            make_source(
                artifacts=artifacts[::-1], importances=((math.log(4),), (math.log(2), math.log(3)))
            )
        )
    )
    left = projection.calibration_cells[0]
    right = other.calibration_cells[0]
    assert (left.alpha, left.beta_count) == pytest.approx((3, 2))
    assert (right.alpha, right.beta_count) == pytest.approx((3, 2))
    recovered = restore(projection)
    assert recovered.calibration_cells == projection.calibration_cells
    with pytest.raises(ValueError):
        recovered.preview(source)
    assert recovered.calibration_cells == projection.calibration_cells


def test_first_seen_source_predictions_exclude_all_repeated_source_labels():
    projection, _ = pipeline()
    projection.commit(projection.preview(make_source(artifacts=(artifact("first", question="q"),))))
    projection.commit(
        projection.preview(
            make_source(
                batch_id="next",
                optimizer_step=2,
                artifacts=(
                    artifact("repeat", question="q"),
                    artifact("new", question="r", success=False),
                ),
            )
        )
    )
    reports = prequential_calibration(projection.posterior_provenance)
    filtered = reports[-1]["source_disjoint_from_prediction_history"]
    assert filtered["excluded_event_count"] == 1
    assert filtered["summary"]["distinct_source_question_count"] == 1
    assert filtered["summary"]["failure_weight"] > 0
    assert filtered["reliability"][0]["success_fraction"] == 0


def test_canonical_aliases_do_not_become_first_seen_independent_prediction_sources():
    projection, _ = pipeline()
    first = artifact("first-alias", question="same")
    projection.commit(projection.preview(make_source(artifacts=(first,))))
    second = artifact("second-alias", question="same", success=False)
    reward = replace(
        second.record.reward,
        native_payload={
            "training_evidence_source": {
                "benchmark_id": "mbpp-plus",
                "population_id": "renamed-population",
                "source_question_id": "same",
            }
        },
    )
    second = replace(second, record=replace(second.record, reward=reward))
    projection.commit(
        projection.preview(make_source(artifacts=(second,), batch_id="second", optimizer_step=2))
    )
    reports = prequential_calibration(projection.posterior_provenance)
    assert (
        reports[-1]["source_disjoint_from_prediction_history"]["summary"]["invocation_event_count"]
        == 0
    )
    from skillev.training.calibration_reporting import posterior_evidence_composition

    composed = posterior_evidence_composition(projection.posterior_provenance)[0]
    assert composed["trajectory_count"] == 2
    assert composed["canonical_source_question_count"] == 1
    assert composed["success_weight"] == composed["failure_weight"] == 1
    assert all(row["terminal_verifier_version"] is not None for row in composed["versions"])
    assert posterior_evidence_composition(restore(projection).posterior_provenance) == [composed]


def test_one_and_nine_flows_share_one_batch_denominator_and_never_reapply_after_restore():
    projection, _ = pipeline()
    source = make_source(
        artifacts=(artifact("one", success=True), artifact("nine", success=False)),
        importances=((0.0,), (math.log(9),)),
    )
    transition = projection.preview(source)
    by_trajectory = {e.trajectory_id: e.flow_weight for e in transition.posterior_batch.updates}
    assert by_trajectory == pytest.approx({"one": 0.2, "nine": 1.8})
    assert sum(by_trajectory.values()) == pytest.approx(2)
    assert projection.calibration_cells == ()
    projection.commit(transition)
    cell = projection.calibration_cells[0]
    assert (cell.alpha, cell.beta_count) == pytest.approx((1.2, 2.8))
    recovered = restore(projection)
    with pytest.raises(ValueError):
        recovered.preview(source)
    assert recovered.calibration_cells == projection.calibration_cells


def test_missing_catalog_window_evidence_is_null_not_zero_read_or_visibility():
    from tests.training.test_skill_discovery_chain import catalog_record

    projection, _ = pipeline()
    projection.commit(projection.preview(make_source(artifacts=(catalog_record(calls=False),))))
    report = batch_coverage(projection.posterior_provenance.batches[-1])["benchmarks"]["mbpp-plus"]
    assert report["body_visible_skill_ids"] is None
    assert report["body_visible_trajectory_count"] is None
    assert report["catalog_or_inline_visible_trajectory_count"] is None
    assert report["body_not_fully_visible_input_count"] is None
    assert report["body_visibility_unknown_trajectory_count"] == 1
    assert report["body_read_count"] == 0  # original empty invocation list is positive evidence
    assert report["posterior_update_event_count"] == 0
