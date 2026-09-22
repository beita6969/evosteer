"""Selected upstream LiveMedBench MIT functions, no model loading or CLI.

Source: https://github.com/ZhilingYan/LiveMedBench/tree/main/evaluate
Retrieved 2026-09-12; LICENSE.livemedbench.txt accompanies this module.
Only type spelling/formatting is adapted; prompt and arithmetic are unchanged.
"""

from __future__ import annotations

from typing import Any

EVALUATION_PROMPT = (
    "\nRole: You are an Objective Grader.\nTask: Evaluate the Model Response (M"
    "_out) against the provided Rubric (R).\n\nInstructions:\n- Objective Verifi"
    "cation: For each criterion in the Rubric, determine if the Model Respons"
    "e satisfies it.\n- Binary Judgment: Return true (Met) or false (Not Met)."
    "\n- Positive Criteria Logic: true if the model includes the required info"
    "rmation.\n- Negative Criteria Logic: true if the model commits the error "
    '(e.g., if the rubric asks "Does model suggest antibiotics?" and the mode'
    "l suggests them, return true). Note: The scoring formula handles the neg"
    "ative sign; you simply detect presence.\n- Evidence: Quote the specific s"
    "entence from the model output that supports your decision.\n\nInput:\n- Use"
    "r Query (Q): This is the original question from the patient, built as:\n "
    '   Q = """{user_query}"""\n- Model Response (M_out):\n{model_response}\n\n- '
    "Rubric (R): JSON list of criteria from Phase 1. In this call you will re"
    'ceive exactly one criterion:\n[\n  {{"question": "{criterion}"}}\n]\n\nOutput'
    ' Format (JSON):\n[\n  {{\n    "question": "Does the model identify the like'
    'ly cause as Norovirus?",\n    "met": true,\n    "reasoning": "Model explic'
    'itly states \'symptoms suggest Norovirus\'."\n  }},\n  {{\n    "question": "D'
    'oes the model recommend antibiotics?",\n    "met": false,\n    "reasoning"'
    ": \"Model correctly states 'antibiotics are not effective'.\"\n  }}\n]\n\nNow,"
    " given the User Query (Q), the Model Response (M_out) and the Rubric (R)"
    " with one criterion, output a JSON list with a single object in the exac"
    't format above, where:\n- "question" is the criterion string you evaluate'
    'd,\n- "met" is true or false,\n- "reasoning" briefly quotes or summarizes '
    "the evidence from the model response (and, if relevant, the user query) "
    "that supports your decision.\n"
)


def create_evaluation_prompt(criterion: str, model_response: str, user_query: str) -> str:
    """Fill the evaluation prompt template."""
    return EVALUATION_PROMPT.format(
        criterion=criterion.strip(),
        model_response=(model_response or "").strip(),
        user_query=(user_query or "").strip(),
    )


def calculate_max_possible_score(rubric_items: list[dict[str, Any]]) -> float:
    """
    Compute the maximum possible positive score for a case.

    This follows the MedOnline logic:
      - Sum only positive `points` values in the rubric (negative ones are
        handled in weighted_score already).
    """
    if not rubric_items:
        return 0.0
    max_score = 0.0
    for item in rubric_items:
        points = item.get("points", 0)
        if isinstance(points, int | float) and points > 0:
            max_score += float(points)
    return max_score


def calculate_case_total_score(
    evaluations: dict[str, Any], rubric_items: list[dict[str, Any]]
) -> float:
    """
    Compute the total weighted_score for a single case.

    We:
      - Build a set of valid criteria from the rubric file.
      - Sum weighted_score for evaluation entries whose criterion appears in
        the rubric (to stay aligned with the reference rubric).
    """
    if not evaluations or not rubric_items:
        return 0.0
    rubric_criteria = {}
    for item in rubric_items:
        criterion = item.get("criterion", "")
        if criterion:
            rubric_criteria[criterion] = item.get("points", 0)
    total_score = 0.0
    for _, rubric_data in evaluations.items():
        if not isinstance(rubric_data, dict):
            continue
        criterion = rubric_data.get("criterion", "")
        weighted_score = rubric_data.get("weighted_score", 0)
        if not criterion or criterion not in rubric_criteria:
            continue
        if isinstance(weighted_score, int | float):
            total_score += float(weighted_score)
    return total_score
