"""Public task export is complete for public inputs and independent of labels."""

from __future__ import annotations

from copy import deepcopy

import pytest

from skillev.evaluation.input_metric_contracts import (
    CONTRACTS,
    PublicTaskView,
    require_public_fields,
    require_runtime_public_fields,
)


def public_rows() -> dict[str, dict[str, object]]:
    return {
        "hotpotqa": {
            "question": "Which public entity connects these passages?",
            "context": [f"Public passage {index}" for index in range(10)],
        },
        "triviaqa": {"question": "What is the public answer?", "public_context": "Public RC."},
        "aime-2026": {"problem": "Compute 2 + 2."},
        "healthbench": {"prompt": [{"role": "user", "content": "Provide public guidance."}]},
        "webshop": {"task": "Find a publicly described fictional item."},
        "alfworld": {"task": "Complete this public synthetic task."},
        "mbpp-plus": {"prompt": "def f(x):\n    return x\n"},
        "humaneval": {"prompt": "def f(x):\n    return x\n"},
        "musique": {"question": "Synthetic multihop question?", "context": "Public passages."},
        "nq-open": {"question": "Synthetic factual question?"},
        "omni-math": {"problem": "Synthetic mathematical problem."},
        "gpqa-diamond-bioorganic": {
            "question": "Synthetic scientific multiple-choice question.",
            "options": "A. first\nB. second\nC. third\nD. fourth",
        },
        "livemedbench": {
            "narrative": "A synthetic patient describes their situation.",
            "core_request": "What should I do next?",
        },
        "scienceworld": {"task": "Synthetic scientific goal."},
        "math-hard": {"problem": "Synthetic Level 5 mathematical problem."},
        "livecodebench": {"prompt": "Synthetic Python programming task."},
        "apps-introductory": {"prompt": "Synthetic introductory programming task."},
    }


@pytest.mark.parametrize("benchmark", tuple(CONTRACTS))
def test_label_changes_do_not_change_public_actor_export(benchmark: str) -> None:
    source = public_rows()[benchmark]
    changed = deepcopy(source)
    changed.update(
        {
            "gold": "counterfactual label one",
            "aliases": ["counterfactual alias"],
            "rubrics": ["counterfactual rubric"],
            "hidden_tests": ["counterfactual test"],
            "target_product": "counterfactual product",
            "doctor_advice": "counterfactual clinician reference",
            "rubric_items": [{"criterion": "counterfactual criterion", "points": 9}],
        }
    )
    if benchmark == "healthbench":
        changed["prompt"] = [
            {"role": "user", "content": "Provide public guidance.", "rubric": "counterfactual"}
        ]

    assert PublicTaskView.from_record("synthetic", benchmark, source) == PublicTaskView.from_record(
        "synthetic", benchmark, changed
    )


@pytest.mark.parametrize("benchmark", tuple(CONTRACTS))
def test_every_required_public_field_must_be_present_and_nonempty(benchmark: str) -> None:
    source = public_rows()[benchmark]
    for name in CONTRACTS[benchmark].required_source_fields:
        missing = dict(source)
        del missing[name]
        with pytest.raises(ValueError):
            require_public_fields(benchmark, missing)
        empty = dict(source)
        empty[name] = [] if name in {"context", "prompt"} else ""
        with pytest.raises(ValueError):
            require_public_fields(benchmark, empty)


def test_runtime_state_is_not_exported_from_source_records_and_is_structurally_checked() -> None:
    source = public_rows()["webshop"]
    assert PublicTaskView.from_record("synthetic", "webshop", source).fields == (
        ("task", source["task"]),
    )
    require_runtime_public_fields(
        "webshop", {"observation": "Public page", "available_actions": ("search[item]",)}
    )
    with pytest.raises(ValueError):
        require_runtime_public_fields(
            "webshop", {"observation": "Public page", "available_actions": ()}
        )


def test_serialized_public_view_cannot_bypass_required_structures() -> None:
    with pytest.raises(ValueError):
        PublicTaskView("synthetic", "hotpotqa", (("question", "Question"), ("context", "[]")))
    with pytest.raises(ValueError):
        PublicTaskView("synthetic", "healthbench", (("prompt", "[]"),))


def test_trivia_released_passage_list_is_preserved_without_a_generated_dossier() -> None:
    passages = ["Title one\nFirst complete public passage.", "Title two\n第二段公共原文。"]
    view = PublicTaskView.from_record(
        "synthetic", "triviaqa", {"question": "Public question", "public_context": passages}
    )
    assert dict(view.fields)["public_context"] == "\n\n".join(passages)
    assert all(passage in view.render() for passage in passages)
    with pytest.raises(ValueError):
        PublicTaskView.from_record(
            "synthetic", "triviaqa", {"question": "Public question", "public_context": []}
        )


@pytest.mark.parametrize("role", ["scorer", "environment", "unknown"])
def test_health_source_roles_are_checked_before_chat_template_encoding(role) -> None:
    with pytest.raises(ValueError):
        PublicTaskView.from_record(
            "synthetic", "healthbench", {"prompt": [{"role": role, "content": "public"}]}
        )
