"""Synthetic-only GPQA selection, answer isolation and native choice scoring."""

import asyncio
import csv
import json
from collections import Counter
from types import SimpleNamespace

import pytest
from skillev_private.evaluation.ood_export import export_sources
from skillev_private.evaluation.ood_gpqa import BENCHMARK, project_gpqa, select_bioorganic
from skillev_private.evaluation.ood_scoring import score_ood
from skillev_private.evaluation.ood_sources import load_ood_panel

from skillev.evaluation.input_metric_contracts import HISTORICAL_OOD_BENCHMARKS, PublicTaskView
from skillev.evaluation.step0_integrity import InferenceArm, ToolCallMode
from skillev.evaluation.thinking_policy import ThinkingPolicy
from tests.evaluation.test_integrity_broker_boundary import runtime


@pytest.fixture
def population():
    rows = []
    for i in range(198):
        rows.append(
            {
                "Record ID": f"synthetic-{i:03}",
                "High-level domain": "Biology" if i < 19 else "Chemistry" if i < 91 else "Physics",
                "Subdomain": "Genetics" if i < 19 else "Organic Chemistry" if i < 91 else "Other",
                "Question": f"Synthetic multiple-choice question {i}",
                "Correct Answer": f"Public option {i}-0",
                **{f"Incorrect Answer {n}": f"Public option {i}-{n}" for n in range(1, 4)},
                "Explanation": "PRIVATE EXPLANATION",
            }
        )
    return rows, [r["Record ID"] for r in rows[:19]], [r["Record ID"] for r in rows[19:91]]


def test_bioorganic_keeps_all_biology_and_samples_45_organic_without_answers(population):
    rows, biology, organic = population
    selected = select_bioorganic(rows, biology, organic, count=64, seed=0)
    ids = [row["Record ID"] for _, row in selected]
    assert len(ids) == len(set(ids)) == 64
    assert set(biology) <= set(ids)
    assert len(set(ids) & set(organic)) == 45
    changed = [{**r, "Correct Answer": "CHANGED"} for r in reversed(rows)]
    repeated = select_bioorganic(changed, biology[::-1], organic[::-1], count=64, seed=0)
    assert [row["Record ID"] for _, row in repeated] == ids


@pytest.mark.parametrize("failure", ["missing-id", "duplicate-id", "wrong-domain", "main-size"])
def test_bioorganic_never_silently_drops_unmatched_population(population, failure):
    rows, biology, organic = population
    if failure == "missing-id":
        organic[-1] = "not-in-diamond"
    elif failure == "duplicate-id":
        rows[-1]["Record ID"] = rows[0]["Record ID"]
    elif failure == "wrong-domain":
        rows[0]["High-level domain"] = "Physics"
    else:
        rows.pop()
    with pytest.raises(ValueError):
        select_bioorganic(rows, biology, organic, count=64, seed=0)


def test_gpqa_export_and_public_projection_keep_labels_and_explanations_private(
    tmp_path, population
):
    rows, biology, organic = population
    csv_path = tmp_path / "gpqa_diamond.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(json.dumps({"biology_record_ids": biology, "organic_record_ids": organic}))
    exported = export_sources(
        {
            "sample_count": 64,
            "seed": 0,
            "sources": {BENCHMARK: {"paths": [str(csv_path)], "allowlist_path": str(allowlist)}},
        },
        tmp_path / "frozen",
    )
    assert len(load_ood_panel(exported).panel.entries) == 64
    projected = [json.loads(line) for line in open(exported["ood_sources"][BENCHMARK])]
    assert len(projected) == 64
    assert exported["ood_provenance"][BENCHMARK]["health_subset"] is False
    assert exported["ood_provenance"][BENCHMARK]["sample_composition"]["Biology"] == 19
    labels = Counter()
    original = {r["Record ID"]: r for r in rows}
    for record in projected:
        entry = PublicTaskView.from_record(record["task_id"], BENCHMARK, record["public"])
        rendered = entry.render()
        assert "PRIVATE EXPLANATION" not in rendered
        assert "correct_label" not in rendered
        assert "Correct Answer" not in rendered
        answer = original[entry.task_id.split(":", 1)[1]]["Correct Answer"]
        label = record["target"]["correct_label"]
        assert f"{label}. {answer}" in dict(entry.fields)["options"].splitlines()
        labels[label] += 1
    assert len(labels) == 4


def test_gpqa_preserves_repeated_official_distractor_slots(population):
    rows, _, _ = population
    row = {**rows[0], "Incorrect Answer 3": rows[0]["Incorrect Answer 2"]}
    projected = project_gpqa(0, row, seed=0)
    options = projected["public"]["options"].splitlines()
    assert len(options) == 4
    assert sum(line.endswith(row["Incorrect Answer 2"]) for line in options) == 2
    assert f"{projected['target']['correct_label']}. {row['Correct Answer']}" in options


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("C", 1.0),
        ("Explanation.\nFinal answer: C", 1.0),
        ("**Final answer:**\nOption C correctly identifies:\nFinal answer: C", 1.0),
        ("C. Synthetic option\nExplanation: Public rationale.", 1.0),
        ("C. Synthetic option\nExplanation: Public rationale.\nFinal answer: B", 0.0),
        ("Final answer: B", 0.0),
        ("Final answer: A\nFinal answer: C", 0.0),
        ("An unfinished discussion without a choice.", 0.0),
        ("", 0.0),
    ],
)
def test_gpqa_exact_choice_score_never_uses_reference_to_select_from_prose(text, value):
    reader = SimpleNamespace(get=lambda *args: SimpleNamespace(text=text, episode_id="fixture"))
    score = asyncio.run(
        score_ood(
            reader,
            ("run", "arm", "fixture"),
            BENCHMARK,
            {"correct_label": "C"},
            settings={},
            sandbox=None,
            diagnostics=lambda payload: None,
        )
    )
    assert score.metric == "accuracy"
    assert score.value == value


@pytest.mark.parametrize("native_tools", [False, True])
def test_gpqa_uses_real_single_owner_actor_broker_and_native_scorer(tmp_path, native_tools):
    entry = PublicTaskView.from_record(
        "gpqa-fixture",
        BENCHMARK,
        {"question": "Synthetic question", "options": "A. one\nB. two\nC. three\nD. four"},
    )
    instance = runtime(tmp_path, entry, ["Final answer: C"])
    arm = (
        InferenceArm("bioorganic", tool_call_mode=ToolCallMode.QWEN_XML)
        if native_tools
        else InferenceArm("bioorganic")
    )
    try:
        final = asyncio.run(instance.generate(entry, arm, "synthetic"))
        instance.validate_candidate(instance.journal, entry, arm, "synthetic")
        assert final.intervention_counts["model_calls"] == 1
        scope = ("synthetic", arm.arm_id, entry.task_id)
        assert all(o.participant == "owner" for o in instance.journal.model_outputs(scope))
        score = asyncio.run(
            score_ood(
                instance.journal,
                scope,
                BENCHMARK,
                {"correct_label": "C"},
                settings={},
                sandbox=instance.sandbox,
                diagnostics=lambda payload: None,
            )
        )
        assert score.value == 1.0
    finally:
        instance.journal.close()


def test_historical_six_ood_thinking_map_is_readable_without_relabeling():
    legacy = tuple(
        "gpqa-diamond-health" if b == "livemedbench" else b for b in HISTORICAL_OOD_BENCHMARKS
    )
    policy = ThinkingPolicy("historical-ood", tuple((b, False) for b in legacy))
    assert not policy.resolve("musique", InferenceArm("historical")).native_thinking
    assert "gpqa-diamond-health" in dict(policy.rows)
    assert BENCHMARK not in dict(policy.rows)
