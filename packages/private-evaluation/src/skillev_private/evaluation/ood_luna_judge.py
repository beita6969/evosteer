"""Post-submission Omni-MATH exports for the owner-requested Luna medium API.

This is a terminal evaluator, never an episode participant. Exports contain
private references and must remain outside Git and outside evaluated actors.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from skillev.evaluation.external_judge_policy import (
    DIRECT_JUDGE_MODEL,
    DIRECT_OMNI_JUDGE_PROFILE,
    DIRECT_OMNI_JUDGE_VERIFIER,
    EXTERNAL_JUDGE_API_BASE,
    EXTERNAL_JUDGE_EFFORT,
    EXTERNAL_JUDGE_MAX_TOKENS,
    EXTERNAL_JUDGE_MODEL,
    EXTERNAL_JUDGE_PROVIDER,
    EXTERNAL_JUDGE_ROUTING,
    GATEWAY_OMNI_JUDGE_PROFILE,
    GATEWAY_OMNI_JUDGE_VERIFIER,
    LEGACY_OMNI_JUDGE_METRIC,
    LEGACY_OMNI_JUDGE_PROFILE,
    LEGACY_OMNI_JUDGE_VERIFIER,
    OMNI_JUDGE_METRIC,
    OMNI_JUDGE_PROFILE,
    OMNI_JUDGE_VERIFIER,
)
from skillev.evaluation.sealed_candidates import CandidateReader, FinalCandidate

from .integrity_sources import SourcePanel
from .ood_sources import load_ood_panel

JUDGE_MODEL = EXTERNAL_JUDGE_MODEL
JUDGE_EFFORT = EXTERNAL_JUDGE_EFFORT
JUDGE_PROFILE = OMNI_JUDGE_PROFILE
TEMPLATE_SOURCE = (
    "https://github.com/KbsdJames/Omni-MATH/blob/main/GPT_eval/gpt_evaluation_template.txt"
)


def render_judge_prompt(template: str, *, problem: str, answer: str, candidate: str) -> str:
    """Fill the official template once; inserted text is never templated again."""
    fields = {"Problem": problem, "Reference Answer": answer, "Solution": candidate}
    if any("{{" + name + "}}" not in template for name in fields):
        raise ValueError("Omni-MATH template must contain all three official fields")
    return re.sub(
        r"\{\{(Problem|Reference Answer|Solution)\}\}",
        lambda match: fields[match[1]],
        template,
    )


def export_judge_batch(
    reader: CandidateReader,
    source: SourcePanel,
    run_id: str,
    arm_id: str,
    *,
    template: str,
) -> dict[str, Any]:
    records = []
    for entry in source.panel.entries:
        if entry.benchmark != "omni-math":
            continue
        candidate = reader.get(run_id, arm_id, entry.task_id)
        target = source.targets[entry.task_id]
        records.append(judge_record((run_id, arm_id, entry.task_id), candidate, target, template))
    return finalized_batch(records, template)


def judge_record(
    scope: tuple[str, str, str],
    candidate: FinalCandidate,
    target: dict[str, Any],
    template: str,
) -> dict[str, Any]:
    return {
        "run_id": scope[0],
        "arm_id": scope[1],
        "task_id": scope[2],
        "owner_call_id": candidate.owner_call_id,
        "problem": target["problem"],
        "reference_answer": target["answer"],
        "candidate": candidate.text,
        "judge_prompt": render_judge_prompt(
            template, problem=target["problem"], answer=target["answer"], candidate=candidate.text
        ),
    }


def finalized_batch(records: list[dict[str, Any]], template: str) -> dict[str, Any]:
    """Same terminal export for both the CLI batch and the native scoring phase."""
    if not records:
        raise ValueError("no finalized Omni-MATH candidates are available")
    return {
        "profile": JUDGE_PROFILE,
        "judge_model": JUDGE_MODEL,
        "reasoning_effort": JUDGE_EFFORT,
        "backend": "openai-chat-completions",
        "max_completion_tokens": EXTERNAL_JUDGE_MAX_TOKENS,
        "endpoint": EXTERNAL_JUDGE_API_BASE,
        "provider": EXTERNAL_JUDGE_PROVIDER,
        "routing_policy": EXTERNAL_JUDGE_ROUTING,
        "upstream_api": "responses",
        "template_source": TEMPLATE_SOURCE,
        "template_text": template,
        "planned_count": len(records),
        "records": records,
    }


def judge_identity(settings: dict[str, Any]) -> tuple[str, str, str, str]:
    """Legacy imports require an explicit profile; future runs default to medium."""
    profile = settings.get("judge_profile", JUDGE_PROFILE)
    if profile == JUDGE_PROFILE:
        return JUDGE_PROFILE, JUDGE_EFFORT, OMNI_JUDGE_METRIC, OMNI_JUDGE_VERIFIER
    if profile == DIRECT_OMNI_JUDGE_PROFILE:
        return profile, "medium", OMNI_JUDGE_METRIC, DIRECT_OMNI_JUDGE_VERIFIER
    if profile == GATEWAY_OMNI_JUDGE_PROFILE:
        return profile, "medium", OMNI_JUDGE_METRIC, GATEWAY_OMNI_JUDGE_VERIFIER
    if profile == LEGACY_OMNI_JUDGE_PROFILE:
        return profile, "high", LEGACY_OMNI_JUDGE_METRIC, LEGACY_OMNI_JUDGE_VERIFIER
    raise ValueError("unknown Omni-MATH judge profile")


def read_judgement(candidate: FinalCandidate, settings: dict[str, Any]) -> dict[str, Any]:
    # A missing/incomplete judge result stops grading; it is never a zero score.
    result = json.loads(Path(settings["judgements_path"]).read_text())
    profile, effort, metric, verifier = judge_identity(settings)
    if (result.get("profile"), result.get("judge_model"), result.get("reasoning_effort")) != (
        profile,
        JUDGE_MODEL
        if profile in {JUDGE_PROFILE, GATEWAY_OMNI_JUDGE_PROFILE}
        else DIRECT_JUDGE_MODEL,
        effort,
    ):
        raise ValueError("Omni-MATH results use a different evaluator profile")
    scope = (candidate.run_id, candidate.arm_id, candidate.episode_id, candidate.owner_call_id)
    matched = [
        row
        for row in result["records"]
        if tuple(row.get(key) for key in ("run_id", "arm_id", "task_id", "owner_call_id")) == scope
    ]
    if len(matched) != 1 or type(matched[0].get("equivalent")) is not bool:
        raise ValueError("Omni-MATH requires one resolved verdict for this finalized candidate")
    row: dict[str, Any] = matched[0]
    if not isinstance(row.get("reason"), str) or not row["reason"].strip():
        raise ValueError("Omni-MATH judge must retain its equivalence explanation")
    return {
        "passed": row["equivalent"],
        "judgement": row,
        "profile": profile,
        "metric": metric,
        "verifier_version": verifier,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.runtime_config.read_text())
    if judge_identity(config["scorers"]["omni-math"])[0] != JUDGE_PROFILE:
        raise ValueError("new judge exports require the current medium API condition")
    source = load_ood_panel(config)
    template = Path(config["scorers"]["omni-math"]["template_path"]).read_text()
    reader = CandidateReader(args.candidates)
    try:
        result = export_judge_batch(reader, source, args.run_id, args.arm_id, template=template)
    finally:
        reader.close()
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(f"Exported {len(result['records'])} finalized Omni-MATH candidates for Luna medium API")


if __name__ == "__main__":
    main()
