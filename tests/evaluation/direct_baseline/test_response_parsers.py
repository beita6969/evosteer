import pytest

from skillev.evaluation.direct_baseline.parsing import (
    ParseReason,
    ParseStatus,
    parse_aime_integer,
    parse_boxed_math,
    parse_multiple_choice,
    parse_multiple_choice_json,
    parse_multiple_choice_label_v2,
    parse_multiple_choice_label_v3,
    parse_native_action,
    parse_python_source,
    parse_short_answer,
    parse_unified_diff,
)


def test_short_answer_removes_thinking_and_public_marker() -> None:
    parsed = parse_short_answer("<think>private reasoning</think>\nFinal answer: Wellington")
    assert parsed.value == "Wellington"
    assert parsed.status is ParseStatus.EXTRACTED


def test_short_answer_rejects_contradictory_markers() -> None:
    parsed = parse_short_answer("Answer: one\nFinal answer: two")
    assert parsed.value is None
    assert parsed.status is ParseStatus.AMBIGUOUS


@pytest.mark.parametrize("text", ["Final answer: C", "answer = (C)", "C"])
def test_multiple_choice_accepts_one_label(text: str) -> None:
    assert parse_multiple_choice(text).value == "C"


def test_multiple_choice_uses_explicit_final_marker_instead_of_prose_letters() -> None:
    parsed = parse_multiple_choice(
        "Option A misses the diagnosis, while B is closer.\nFinal answer: C"
    )
    assert parsed.value == "C"


@pytest.mark.parametrize(
    "text",
    [
        "**Final answer:** C",
        "**Final answer: C**",
        "Reasoning mentions A and D.\n## __Final answer__: **C**",
        "C. The selected synthetic option",
        "Final answer: Option C",
        "final answer c",
        "Final answer: C\n**Final answer:** C",
        "```text\nFinal answer: A\n```\nFinal answer: C",
    ],
)
def test_multiple_choice_v2_presentation_preserves_explicit_choice(text):
    assert parse_multiple_choice_label_v2(text).value == "C"


@pytest.mark.parametrize(
    "text",
    [
        "**Final answer:** A\nFinal answer: C",
        "Final answer: C or D",
        "C. or D. are possible",
        "A. First option\nC. Another option",
        "Reasoning discusses C but never submits it.",
        "```text\nFinal answer: C\n```",
        "Final answer: C\nFinal answer: neither option",
    ],
)
def test_multiple_choice_v2_never_selects_among_alternatives(text):
    assert parse_multiple_choice_label_v2(text).value is None


@pytest.mark.parametrize(
    "text",
    [
        "**Final Answer:**\n\nOption C correctly identifies:\n- x = synthetic\nFinal answer: C",
        "C. Synthetic option\n\nExplanation: This is the owner's explanation.",
        "Final answer:\nC\nFinal answer: C",
        "**Final Answer:**\nA discussion without a choice.\nFinal answer: C",
        "C. Synthetic option\nExplanation: Reasoning.\nFinal answer: C",
    ],
)
def test_multiple_choice_v3_accepts_structured_choice_without_reference(text):
    assert parse_multiple_choice_label_v3(text).value == "C"


@pytest.mark.parametrize(
    "text",
    [
        "Final answer:\nA\nFinal answer: C",
        "Final answer:\nOption A correctly identifies:\nFinal answer: C",
        "Final answer:\nOption C or D\nFinal answer: C",
        "C. Synthetic option\nExplanation: Reasoning.\nFinal answer: D",
        "A. One option\nC. Another option\nExplanation: Both are possible.",
        "Explanation: I considered C.\nNo final choice.",
        "Final answer:\nNo choice submitted.",
    ],
)
def test_multiple_choice_v3_never_selects_among_competing_or_missing_choices(text):
    assert parse_multiple_choice_label_v3(text).value is None


def test_multiple_choice_v2_keeps_historical_heading_and_explanation_failures():
    assert parse_multiple_choice_label_v2("Final answer:\nC").value is None
    assert parse_multiple_choice_label_v2("C. Synthetic\nExplanation: Detail.").value is None


def test_aime_rejects_conflicting_final_markers_and_checks_range() -> None:
    assert (
        parse_aime_integer("Answer: 12\nFinal answer: 007").reason is ParseReason.CONFLICTING_FINALS
    )
    assert parse_aime_integer("Final answer: 007").value == "7"
    assert parse_aime_integer("Final answer: 1000").value is None


def test_boxed_math_balances_nested_braces() -> None:
    assert parse_boxed_math(r"Reasoning. \boxed{\frac{1}{2}}").value == r"\frac{1}{2}"


def test_python_source_removes_only_code_fence() -> None:
    assert parse_python_source("```python\ndef f():\n    return 1\n```").value == (
        "def f():\n    return 1"
    )


def test_native_action_never_parses_json() -> None:
    assert parse_native_action("Action: click[item]").value == "click[item]"
    assert parse_native_action("click[item]").value == "click[item]"
    assert parse_native_action('{"action":"click[item]"}').value is None


def test_native_action_rejects_conflicting_action_lines() -> None:
    parsed = parse_native_action("Action: search[a]\nAction: click[b]")
    assert parsed.reason is ParseReason.CONFLICTING_FINALS


def test_json_mc_and_unified_diff_have_separate_contracts() -> None:
    assert parse_multiple_choice_json('{"answer":"B"}').value == "B"
    assert parse_multiple_choice_json("Final answer: B").value is None
    patch = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-a\n+b"
    assert parse_unified_diff(patch).value == patch
    assert parse_unified_diff("```python\nprint(1)\n```").value is None


def test_unified_diff_accepts_structure_despite_a_wrong_fence_label() -> None:
    patch = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new"
    assert parse_unified_diff(f"```python\n{patch}\n```").value == patch


def test_unified_diff_does_not_pair_closing_fence_with_a_later_block() -> None:
    patch = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-a\n+b"
    response = f"```python\nprint(1)\n```\n\n```diff\n{patch}\n```"
    assert parse_unified_diff(response).value == patch


def test_unified_diff_rejects_distinct_patch_candidates() -> None:
    first = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-a\n+b"
    second = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-a\n+c"
    parsed = parse_unified_diff(f"```diff\n{first}\n```\n```diff\n{second}\n```")
    assert parsed.status is ParseStatus.AMBIGUOUS
    assert parsed.value is None
    assert parsed.candidate_count == 2
