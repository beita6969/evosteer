"""Private development-only skill qualification, never an IID selection tool.

Compare all predeclared sources under the same base/controls with library off/on.
Reading plus success is insufficient: reviewed application evidence is required.
Reports can guide a NEW training data condition; no training store is written.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from skillev.rollout import RolloutArtifact
from skillev.rollout.generator import RolloutTokenizerProtocol
from skillev.training.invocation_evidence import invocation_execution_links

from .development_collection import read, save
from .development_comparison import inspect


@dataclass(frozen=True)
class SkillUtilityPolicy:
    minimum_reviewed_sources: int
    minimum_success_rate_gain: float
    minimum_output_token_saving_fraction: float

    def __post_init__(self) -> None:
        if type(self.minimum_reviewed_sources) is not int or self.minimum_reviewed_sources < 2:
            raise ValueError("utility qualification needs multiple distinct reviewed sources")
        for value in (self.minimum_success_rate_gain, self.minimum_output_token_saving_fraction):
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("positive utility thresholds must be frozen before collection")

    def to_value(self) -> dict[str, Any]:
        return asdict(self)


def _artifact(root: Path, position: int, tokenizer: RolloutTokenizerProtocol) -> RolloutArtifact:
    value = read(root / "collection" / "episodes" / f"episode-{position:06d}-private.json")
    return RolloutArtifact.from_value(value["artifact"], tokenizer=tokenizer)


def application_evidence(
    artifact: RolloutArtifact,
    reviews: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Review references must point to real visible reads and subsequent actions.

    The reviewer supplies the public rationale, not a generated score or answer.
    This is an explicit reviewed demonstration, not automated causal credit.
    """
    links = invocation_execution_links(artifact.record, artifact.skill_input_evidence)
    result = []
    seen: set[tuple[str, int]] = set()
    for review in reviews:
        skill_id, read_step = review["skill_id"], review["read_step"]
        if (skill_id, read_step) in seen:
            raise ValueError("a repeated review does not create another application witness")
        seen.add((skill_id, read_step))
        if review["decision"] not in {"applied-procedure", "disregarded-inapplicable"}:
            raise ValueError("review must describe application or justified nonapplication")
        if not all(
            isinstance(review.get(k), str) and review[k].strip()
            for k in ("reviewer", "public_rationale", "evidence_reference")
        ):
            raise ValueError("application requires an attributable public-behavior review")
        match = [
            link
            for link in links
            if link.declared_skill_id == skill_id and link.step_index == read_step
        ]
        steps = review["application_steps"]
        if (
            len(match) != 1
            or match[0].body_returned is not True
            or not isinstance(steps, list)
            or not steps
            or any(type(s) is not int for s in steps)
            or len(steps) != len(set(steps))
            or not set(steps) <= set(match[0].body_visible_execution_steps or ())
        ):
            raise ValueError(
                "review is not backed by returned body and actual visible task actions"
            )
        result.append(
            {
                **review,
                "terminal_success": artifact.record.reward.success,
                "skill_version": match[0].returned_skill_version,
                "library_version": match[0].returned_library_version,
            }
        )
    return result


def compare_skill_utility(
    off: Path,
    on: Path,
    reviews: list[dict[str, Any]],
    *,
    tokenizer: RolloutTokenizerProtocol | None = None,
) -> dict[str, Any]:
    a, b = read(off / "selection-private.json"), read(on / "selection-private.json")
    if a.get("arm") != "skills-off" or b.get("arm") != "initial-library":
        raise ValueError("utility contrast requires skills-off and the candidate initial library")
    if {k: v for k, v in a.items() if k != "arm"} != {k: v for k, v in b.items() if k != "arm"}:
        raise ValueError("utility contrast must preserve the full fixed development selection")
    policy = SkillUtilityPolicy(**a["skill_utility_policy"])
    left, right = inspect(off, a), inspect(on, b)
    for key in ("formal", "backbone", "policy", "service", "scorers", "deployments", "workflow"):
        if (
            left["controls"] is None
            or right["controls"] is None
            or left["controls"][key] != right["controls"][key]
        ):
            raise ValueError("only the library axis may change in a skill utility contrast")
    if left["controls"].get("development_practice") != right["controls"].get(
        "development_practice"
    ):
        raise ValueError("guided practice cannot be mixed with autonomous controls")
    if not left["complete"] or not right["complete"]:
        raise ValueError(
            "incomplete native outcomes cannot qualify a skill or produce warmup labels"
        )
    if tokenizer is None:
        from skillev.policy import QwenMultimodalBackboneConfig, QwenTokenizerAdapter

        tokenizer = QwenTokenizerAdapter.from_config(
            QwenMultimodalBackboneConfig.from_value(right["controls"]["backbone"])
        )
    positions = {row["position"] for row in right["rows"]}
    if any(r.get("position") not in positions for r in reviews):
        raise ValueError("review points outside the frozen source selection")
    rows = []
    for x, y in zip(left["rows"], right["rows"], strict=True):
        for key in ("sampling_coordinate", "policy_snapshot"):
            if x["sampling_identity"][key] != y["sampling_identity"][key]:
                raise ValueError("skill contrast changed actual policy or sampling coordinates")
        artifact = _artifact(on, y["position"], tokenizer)
        evidence = application_evidence(
            artifact, [r for r in reviews if r["position"] == y["position"]]
        )

        def output(row: dict[str, Any]) -> int:
            values = [row["actions"][f"{phase}_output_tokens"] for phase in ("reasoning", "action")]
            if any(type(value) is not int for value in values):
                raise ValueError("utility cost comparison requires measured output tokens")
            return sum(values)

        off_tokens, on_tokens = output(x), output(y)
        succeeded = artifact.record.reward.success
        used = any(r["decision"] == "applied-procedure" for r in evidence)
        # A per-source development label, not a reclassification of native reward.
        useful = (
            succeeded
            and used
            and (
                not x["success"]
                or (
                    y["reward"] >= x["reward"]
                    and off_tokens > 0
                    and 1 - on_tokens / off_tokens >= policy.minimum_output_token_saving_fraction
                )
            )
        )
        category = (
            "skill-benefit-candidate"
            if useful
            else "direct-completion"
            if x["success"]
            else "unresolved-exploration"
        )
        rows.append(
            {
                "position": y["position"],
                "source": y["source"],
                "off_success": x["success"],
                "on_success": y["success"],
                "off_reward": x["reward"],
                "on_reward": y["reward"],
                "off_output_tokens": off_tokens,
                "on_output_tokens": on_tokens,
                "reviewed_application": evidence,
                "proposed_training_role": category,
            }
        )
    n = len(rows)
    success_gain = sum(int(r["on_success"]) - int(r["off_success"]) for r in rows) / n
    reward_gain = math.fsum(r["on_reward"] - r["off_reward"] for r in rows) / n
    off_tokens = sum(r["off_output_tokens"] for r in rows)
    saved = 1 - sum(r["on_output_tokens"] for r in rows) / off_tokens if off_tokens else None
    applied_sources = sum(
        any(
            e["decision"] == "applied-procedure" and e["terminal_success"]
            for e in r["reviewed_application"]
        )
        for r in rows
    )
    passed = (
        applied_sources >= policy.minimum_reviewed_sources
        and reward_gain >= 0
        and (
            success_gain >= policy.minimum_success_rate_gain
            or (
                success_gain >= 0
                and saved is not None
                and saved >= policy.minimum_output_token_saving_fraction
            )
        )
    )
    return {
        "format": "training-development-skill-utility@1",
        "comparison_id": a["comparison_id"],
        "policy": policy.to_value(),
        "candidate_library": a["initial_library"],
        "development_practice": a.get("development_practice"),
        "autonomous_skill_choice_measurement": a.get("development_practice") is None,
        "sources": rows,
        "planned_source_count": n,
        "reviewed_successful_sources": applied_sources,
        "all_source_success_rate_gain": success_gain,
        "all_source_reward_gain": reward_gain,
        "all_source_output_token_saving_fraction": saved,
        "observed_library_utility_qualified": passed,
        "individual_skills_qualified": False,
        "scope": "fixed training development sources; not IID/general utility certification",
        "training_updates": 0,
        "posterior_updates": 0,
        "skill_evolution": 0,
        "training_evidence_writes": 0,
        "new_data_condition_required": True,
        "interpretation": (
            "all planned sources retained; reviewed association plus library-axis "
            "contrast, not per-read causal credit"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skills-off", required=True, type=Path)
    parser.add_argument("--skills-on", required=True, type=Path)
    parser.add_argument("--application-reviews", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("do not overwrite a prior skill qualification")
    save(
        args.output,
        compare_skill_utility(args.skills_off, args.skills_on, read(args.application_reviews)),
    )


if __name__ == "__main__":
    main()
