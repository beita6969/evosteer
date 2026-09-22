import json
from dataclasses import replace

import pytest
from skillev_private.experiments.quality_collection import QualityCollectionBinding

from skillev.training.quality_gate import QualityGatePolicy, QualityRule
from tests.evaluation.test_native_source_bridge import record


@pytest.fixture
def panel_binding(tmp_path):
    domains = (
        ("hotpotqa", "hotpotqa-official-em-f1"),
        ("triviaqa", "triviaqa-official-alias-em-f1"),
        ("aime-2026", "integer-exact"),
        ("healthbench", "simple-evals-rubric"),
        ("alfworld", "alfworld-success"),
        ("mbpp-plus", "evalplus-base-plus"),
        ("humaneval", "humaneval-native"),
    )
    records = tuple(
        replace(
            r, input=replace(r.input, task_id=domain), episode=replace(r.episode, episode_id=domain)
        )
        for domain, verifier in domains
        for r in (record(domain, verifier, {"private_target": "kept-private"}),)
    )
    data = tmp_path / "records.jsonl"
    data.write_text("".join(json.dumps(r.to_value()) + "\n" for r in records))
    path = tmp_path / "binding.json"
    value = {
        "records": "records.jsonl",
        "panel_id": "reserved",
        "condition_id": "native-candidate",
        "final_evaluation_sources": [],
        "sampling_schedule_id": "quality-seed-zero",
        "ordered_task_sequence_id": "declared-seven-sources",
    }
    path.write_text(json.dumps(value))
    policy = QualityGatePolicy(
        "rule1",
        "reserved",
        "native-candidate",
        5,
        7,
        2,
        (QualityRule("panel/first_turn_structure_valid_fraction", 0.5, 0.1),),
    )
    return path, policy, records


def test_panel_binding_keeps_exact_sources_and_freezes_private_targets(panel_binding, tmp_path):
    path, policy, records = panel_binding
    binding = QualityCollectionBinding.load(path, training=(), policy=policy, batch_size=7)
    assert binding.records == records
    root = tmp_path / "quality"
    binding.freeze(root)
    binding.freeze(root)  # resume has no generation or new source selection
    altered = replace(
        binding,
        records=(
            replace(
                records[0],
                output=replace(records[0].output, target={"private_target": "different"}),
            ),
            *records[1:],
        ),
    )
    with pytest.raises(ValueError):
        altered.freeze(root)


@pytest.mark.parametrize("population", ["training", "final"])
def test_panel_rejects_training_and_final_evaluation_overlap(panel_binding, population):
    path, policy, records = panel_binding
    training = ()
    if population == "training":
        training = (
            replace(records[0], episode=replace(records[0].episode, population_id="alias")),
        )
    else:
        value = json.loads(path.read_text())
        value["final_evaluation_sources"] = [
            {
                "benchmark_id": "hotpotqa",
                "population_id": "different-final-population",
                "source_question_id": "source-1",
            }
        ]
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        QualityCollectionBinding.load(path, training=training, policy=policy, batch_size=7)


def test_repeated_rollouts_are_not_independent_panel_questions(panel_binding):
    path, policy, _ = panel_binding
    with pytest.raises(ValueError):
        QualityCollectionBinding.load(
            path, training=(), policy=replace(policy, minimum_source_questions=8), batch_size=7
        )
