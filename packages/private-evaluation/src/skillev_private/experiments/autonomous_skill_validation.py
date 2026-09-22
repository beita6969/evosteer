"""No-teaching four-arm validation after real TTB or historical skill warmup.

This is a read-only behavioral check, not IID admission or training evidence.
Freeze source roles and thresholds in collection requests before any response;
retain every native outcome, including failed and unreviewed skill branches.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from skillev.contracts.skill_invocation import parse_action_invocation
from skillev.rollout import RolloutArtifact
from skillev.rollout.generator import RolloutTokenizerProtocol
from skillev.training.invocation_evidence import invocation_execution_links

FORMAT = "autonomous-skill-validation@1"
TTB_FORMAT = "autonomous-skill-validation@2"
ARMS = ("before-off", "before-on", "after-off", "after-on")
ROLES = ("procedure-applicable", "direct-control")
ARCHITECTURE_FIELDS = ("formal", "backbone", "service", "scorers", "deployments", "workflow")


@dataclass(frozen=True)
class AutonomousThresholds:
    minimum_successful_application_sources: int
    minimum_applicable_application_rate: float
    minimum_direct_no_read_rate: float
    minimum_direct_success_rate: float
    maximum_repeat_source_rate: float
    maximum_unrelated_source_rate: float
    minimum_success_rate_gain: float
    minimum_output_token_saving_fraction: float
    maximum_output_token_ratio: float

    def __post_init__(self) -> None:
        if (
            type(self.minimum_successful_application_sources) is not int
            or self.minimum_successful_application_sources < 2
        ):
            raise ValueError("multiple independent successful applications are required")
        for key, value in asdict(self).items():
            if key == "minimum_successful_application_sources":
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("finite predeclared behavioral thresholds are required")
            if key == "maximum_output_token_ratio":
                valid = value >= 1
            elif key.startswith("maximum_"):
                valid = 0 <= value < 1
            else:
                valid = 0 < value <= 1
            if not valid:
                raise ValueError("invalid predeclared behavioral threshold")


def freeze_validation(value: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    """Called before the collector constructs services, environments or journals."""
    from skillev_private.evaluation.iid_episode_sources import canonical_source_key

    from .skill_practice import CONTEXT_KEY

    plan = value["plan"]
    if (
        plan.get("format") not in {FORMAT, TTB_FORMAT}
        or plan.get("comparison_id") != frozen["comparison_id"]
    ):
        raise ValueError("autonomous validation needs its frozen development plan")
    if not all(key in plan.get("architecture", {}) for key in ARCHITECTURE_FIELDS):
        raise ValueError("expanded phase/tool/budget/scorer/runtime controls must be frozen")
    if plan.get("candidate_library") != frozen["initial_library"]:
        raise ValueError("candidate skill content changed from the precollection plan")
    if plan["policies"]["before"] == plan["policies"]["after"]:
        raise ValueError("before and after must identify their distinct initialization snapshots")
    arm = value["arm"]
    if arm not in ARMS or frozen["arm"] != (
        "skills-off" if arm.endswith("-off") else "initial-library"
    ):
        raise ValueError("validation role differs from the actual library axis")
    if frozen.get("development_practice") is not None:
        raise ValueError("teaching examples and requested reading are forbidden in this check")
    for record in frozen["records"]:
        # This reserved metadata is how the real practice wrapper changes H0.
        context = record["public_task"].get("public_context", {})
        if isinstance(context, dict) and CONTEXT_KEY in context:
            raise ValueError("original validation task still carries a teaching instruction")
    thresholds = AutonomousThresholds(**plan["thresholds"])
    roles = plan["source_roles"]
    sources = frozen["source_manifest"]["ordered_sources"]
    if len(roles) != len(sources) or len(frozen["records"]) != len(sources) or not sources:
        raise ValueError("every fixed source needs one pre-outcome public task role")
    aliases = frozen["source_aliases"]
    excluded = {
        canonical_source_key((row["benchmark"], row["source_id"]), aliases)
        for row in plan["excluded_development_sources"]
    }
    if not excluded:
        raise ValueError("the original teaching/warmup/development exclusion list is required")
    counts = dict.fromkeys(ROLES, 0)
    for role, source in zip(roles, sources, strict=True):
        key = source["benchmark"], source["source_id"]
        if (
            (role["benchmark"], role["source_id"]) != key
            or canonical_source_key(key, aliases) in excluded
            or role.get("role") not in ROLES
            or not isinstance(role.get("public_basis"), str)
            or not role["public_basis"].strip()
        ):
            raise ValueError("validation role/source overlaps practice or lacks a public basis")
        counts[role["role"]] += 1
    if (
        counts["procedure-applicable"] < thresholds.minimum_successful_application_sources
        or counts["direct-control"] < 2
    ):
        raise ValueError("both applicable tasks and independent direct controls are required")
    stage = arm.split("-")[0]
    if frozen["policy"] != plan["policies"][stage]:
        raise ValueError("actual policy differs from the frozen before/after snapshot")
    initialization = frozen.get("forward_initialization")
    if plan["format"] == TTB_FORMAT:
        _require_ttb_axis(plan, frozen, excluded)
        return {"plan": plan, "arm": arm}
    if stage == "after" and (
        not initialization or initialization.get("kind") != "skill-use-warmup"
    ):
        raise ValueError("the after arm must bind the saved supervised warmup initialization")
    if stage == "after":
        assert isinstance(initialization, dict)
        trained_sources = {
            canonical_source_key(tuple(row["source"]), aliases)
            for row in initialization["initialization"]["application_annotations"]
        }
        if not trained_sources or not trained_sources <= excluded:
            raise ValueError("the actual saved warmup sources must be excluded")
    return {"plan": plan, "arm": arm}


def _require_ttb_axis(
    plan: dict[str, Any], frozen: dict[str, Any], excluded: set[tuple[str, str]]
) -> None:
    from skillev_private.evaluation.iid_episode_sources import canonical_source_key

    from .development_ttb import resolve_ttb

    if plan.get("comparison_kind") != "ttb-checkpoint":
        raise ValueError("version 2 names a real TTB checkpoint comparison, not a warmup alias")
    after, reference = resolve_ttb(plan["ttb_binding"])
    if plan["policies"] != {"before": reference["before_policy"], "after": after.to_value()}:
        raise ValueError("both policy axes must belong to the original complete TTB run")
    if plan["candidate_library"] not in (
        reference["initial_library"],
        reference["checkpoint_library"],
    ):
        raise ValueError("both on arms must use one real frozen library axis")
    aliases = frozen["source_aliases"]
    # Resolve original canonical identities again in the panel's alias namespace.
    required = {
        canonical_source_key((row[0], row[1]), aliases)
        for row in reference["training_sources"] + reference["skill_construction_sources"]
    }
    if not required or not required <= excluded:
        raise ValueError("exclude ALL training and skill-construction sources before collection")
    initialization = frozen.get("forward_initialization", {})
    if frozen["policy"] == after.to_value():
        if initialization != reference:
            raise ValueError("actual after actor is not bound to the selected complete checkpoint")
    elif (
        initialization.get("kind") != "fresh-forward"
        or initialization.get("original_step_zero") is not True
        or Path(initialization["preparation"]).resolve() != Path(reference["preparation"]).resolve()
    ):
        raise ValueError("before actor must use this run's saved pre-SFT initialization")


def _output(row: dict[str, Any]) -> int:
    counts = [row["actions"][f"{phase}_output_tokens"] for phase in ("reasoning", "action")]
    if any(type(n) is not int or n < 0 for n in counts):
        raise ValueError("missing phase tokens cannot be treated as zero validation cost")
    return sum(counts)


def _effect(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(before)
    previous = sum(_output(row) for row in before)
    current = sum(_output(row) for row in after)
    return {
        "source_count": n,
        "success_rate_gain": sum(
            int(b["success"]) - int(a["success"]) for a, b in zip(before, after, strict=True)
        )
        / n,
        "reward_gain": math.fsum(
            b["reward"] - a["reward"] for a, b in zip(before, after, strict=True)
        )
        / n,
        "output_token_saving_fraction": 1 - current / previous if previous else None,
        "output_token_ratio": current / previous if previous else None,
    }


def _useful(effect: dict[str, Any], policy: AutonomousThresholds) -> bool:
    saving = effect["output_token_saving_fraction"]
    return effect["reward_gain"] >= 0 and (
        effect["success_rate_gain"] >= policy.minimum_success_rate_gain
        or (
            effect["success_rate_gain"] >= 0
            and saving is not None
            and saving >= policy.minimum_output_token_saving_fraction
        )
    )


def _review_usage(
    root: Path,
    rows: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    tokenizer: RolloutTokenizerProtocol | None,
) -> tuple[list[dict[str, Any]], bool]:
    from skillev.training.warmup_supervision import validate_application_annotation

    from .development_collection import read

    by_read: dict[tuple[int, int], dict[str, Any]] = {}
    for review in reviews:
        key = review["position"], review["read_step"]
        if key in by_read or key[0] not in range(len(rows)):
            raise ValueError("duplicate or out-of-panel application review")
        if review.get("selection_relevance") not in {
            "applicable",
            "reasonable-exploration",
            "unrelated",
        }:
            raise ValueError("every actual read needs an attributable relevance assessment")
        by_read[key] = review
    used: set[tuple[int, int]] = set()
    result = []
    complete = True
    for row in rows:
        position = row["position"]
        artifact_value = read(root / row["artifact_path"])["artifact"]
        if tokenizer is None:
            # A no-read result can fail the gate without constructing a model/tokenizer.
            raw_record = artifact_value["record"]
            declarations = (
                parse_action_invocation(
                    step["action_text"], initial_meta=raw_record["initial_context"]["meta"]
                )
                for step in raw_record["steps"]
            )
            # Rejected read attempts have no admitted invocation count. They
            # still must not masquerade as a deliberate no-read control.
            if any(value is not None and value.skill_id is not None for value in declarations):
                from skillev.policy import QwenMultimodalBackboneConfig, QwenTokenizerAdapter

                controls = read(root / "controls-private.json")
                tokenizer = QwenTokenizerAdapter.from_config(
                    QwenMultimodalBackboneConfig.from_value(controls["backbone"])
                )
            else:
                result.append({"position": position, "reads": [], "successful_application": False})
                continue
        artifact = RolloutArtifact.from_value(artifact_value, tokenizer=tokenizer)
        links = invocation_execution_links(artifact.record, artifact.skill_input_evidence)
        reads = []
        for link in links:
            note = by_read.get((position, link.step_index))
            if note is None:
                complete = False
            else:
                used.add((position, link.step_index))
                validate_application_annotation(note, artifact)
                if note["skill_id"] != link.declared_skill_id:
                    raise ValueError("review names another actual read")
            applied = bool(
                note
                and note["decision"] == "applied"
                and not note["behavior_evidence"]["ceremonial_read"]
                and note["selection_relevance"] == "applicable"
            )
            reads.append(
                {
                    "step_index": link.step_index,
                    "skill_id": link.declared_skill_id,
                    "body_returned": link.body_returned,
                    "body_visible_steps": link.body_visible_execution_steps,
                    "repeat_same_version": link.repeat_same_version,
                    "review": note,
                    "reviewed_application": applied,
                }
            )
        result.append(
            {
                "position": position,
                "reads": reads,
                "successful_application": row["success"]
                and any(read["reviewed_application"] for read in reads),
            }
        )
    if used != set(by_read):
        raise ValueError("review invents a read outside the original execution")
    return result, complete


def compare_autonomous_use(
    roots: dict[str, Path],
    reviews: dict[str, list[dict[str, Any]]],
    *,
    tokenizer: RolloutTokenizerProtocol | None = None,
) -> dict[str, Any]:
    """Full denominator and factor axes; no positive-only branch selection."""
    from .development_collection import read
    from .development_comparison import inspect

    if set(roots) != set(ARMS) or set(reviews) != {"before-on", "after-on"}:
        raise ValueError("all four arms and both on-arm review sets are required")
    selections = {name: read(root / "selection-private.json") for name, root in roots.items()}
    plan = selections["after-on"]["autonomous_validation"]["plan"]
    ttb = plan["format"] == TTB_FORMAT
    datasets = {}
    invariant = None
    for name, frozen in selections.items():
        declaration = freeze_validation(frozen["autonomous_validation"], frozen)
        if declaration != {"plan": plan, "arm": name}:
            raise ValueError("validation rules changed after collecting an arm")
        fixed = {
            k: v
            for k, v in frozen.items()
            if k not in {"arm", "policy", "forward_initialization", "autonomous_validation"}
        }
        if invariant is not None and fixed != invariant:
            raise ValueError("validation populations, library or sampling plan changed")
        invariant = fixed
        datasets[name] = inspect(roots[name], frozen)
        if not datasets[name]["complete"]:
            raise ValueError("incomplete native outcomes cannot pass autonomous validation")
        summary = read(roots[name] / "summary.json")
        if any(
            summary.get(key) != 0
            for key in (
                "training_updates",
                "posterior_updates",
                "skill_evolution",
                "training_evidence_writes",
            )
        ):
            raise ValueError("autonomous validation must remain genuinely read-only")
        controls = datasets[name]["controls"]
        if controls.get("development_practice") is not None:
            raise ValueError("actual runtime still used teaching prompts")
        actual = {key: controls[key] for key in plan["architecture"]}
        if actual != plan["architecture"]:
            raise ValueError("expanded runtime changed from the precollection architecture")
        if controls["policy"] != frozen["policy"]:
            raise ValueError("actual runtime policy differs from the arm snapshot")
        if name.endswith("-on"):
            if controls["library"] != plan["candidate_library"]:
                raise ValueError("actual skills-on runtime used another library")
        elif controls["library"]["active_skill_ids"]:
            raise ValueError("skills-off runtime retained an active library")
        for row in datasets[name]["rows"]:
            if row["sampling_identity"]["policy_snapshot"] != frozen["policy"]:
                raise ValueError("collected trajectory used another policy snapshot")
            if name.endswith("-off") and row["actions"]["skill_invocation_count"]:
                raise ValueError("skills-off produced a skill invocation")
    reference_rows = datasets["before-off"]["rows"]
    for dataset in datasets.values():
        for a, b in zip(reference_rows, dataset["rows"], strict=True):
            if (
                a["sampling_identity"]["sampling_coordinate"]
                != b["sampling_identity"]["sampling_coordinate"]
            ):
                raise ValueError("actual per-request sampling coordinates changed")
    policy = AutonomousThresholds(**plan["thresholds"])
    usage = {}
    reviews_complete = True
    for name in ("before-on", "after-on"):
        usage[name], complete = _review_usage(
            roots[name], datasets[name]["rows"], reviews[name], tokenizer
        )
        reviews_complete &= complete
    positions = {
        role: [i for i, row in enumerate(plan["source_roles"]) if row["role"] == role]
        for role in ROLES
    }

    def effect(left: str, right: str, role: str | None = None) -> dict[str, Any]:
        indices = range(len(reference_rows)) if role is None else positions[role]
        return _effect(
            [datasets[left]["rows"][i] for i in indices],
            [datasets[right]["rows"][i] for i in indices],
        )

    effects = {
        "before_library": effect("before-off", "before-on"),
        "after_library": effect("after-off", "after-on"),
        "ttb_with_library" if ttb else "warmup_with_library": effect("before-on", "after-on"),
        "ttb_without_library" if ttb else "warmup_without_library": effect(
            "before-off", "after-off"
        ),
        "after_library_applicable": effect("after-off", "after-on", "procedure-applicable"),
    }
    post = usage["after-on"]
    applicable, direct = (positions[role] for role in ROLES)
    applied = sum(post[i]["successful_application"] for i in applicable)
    direct_rate = sum(not post[i]["reads"] for i in direct) / len(direct)
    direct_success = sum(datasets["after-on"]["rows"][i]["success"] for i in direct) / len(direct)
    repeats = sum(any(r["repeat_same_version"] for r in row["reads"]) for row in post) / len(post)
    unrelated = (
        sum(
            any(
                r["review"] and r["review"]["selection_relevance"] == "unrelated"
                for r in row["reads"]
            )
            for row in post
        )
        / len(post)
        if reviews_complete
        else None
    )
    after, warm = (
        effects["after_library"],
        effects["ttb_with_library" if ttb else "warmup_with_library"],
    )
    checks = {
        "every_read_reviewed": reviews_complete,
        "multiple_successful_application_sources": applied
        >= policy.minimum_successful_application_sources,
        "applicable_application_coverage": applied / len(applicable)
        >= policy.minimum_applicable_application_rate,
        "appropriate_direct_choices": direct_rate >= policy.minimum_direct_no_read_rate,
        "direct_task_success_preserved": direct_success >= policy.minimum_direct_success_rate,
        "bounded_repeat_reads": repeats <= policy.maximum_repeat_source_rate,
        "bounded_unrelated_reads": unrelated is not None
        and unrelated <= policy.maximum_unrelated_source_rate,
        "library_benefit_on_applicable_tasks": _useful(effects["after_library_applicable"], policy),
        "all_source_library_quality_nonregression": after["reward_gain"] >= 0
        and after["success_rate_gain"] >= 0,
        "ttb_quality_nonregression" if ttb else "warmup_quality_nonregression": (
            warm["reward_gain"] >= 0 and warm["success_rate_gain"] >= 0
        ),
        "bounded_total_output_cost": after["output_token_ratio"] is not None
        and after["output_token_ratio"] <= policy.maximum_output_token_ratio,
    }
    return {
        "format": plan["format"],
        "plan": plan,
        "checks": checks,
        "autonomous_skill_use_qualified": all(checks.values()),
        **(
            {"five_step_diagnostic_eligible": all(checks.values())}
            if not ttb
            else {
                "ttb_checkpoint_assessment": selections["after-on"]["forward_initialization"],
                "policy_axis_library": plan["candidate_library"],
                "evaluation_does_not_authorize_or_reset_training": True,
            }
        ),
        "formal_iid_admission": False,
        "training_updates": 0,
        "posterior_updates": 0,
        "skill_evolution": 0,
        "training_evidence_writes": 0,
        "effects": effects,
        "successful_application_sources": applied,
        "applicable_application_rate": applied / len(applicable),
        "direct_no_read_rate": direct_rate,
        "direct_success_rate": direct_success,
        "repeat_source_rate": repeats,
        "unrelated_source_rate": unrelated,
        "usage": usage,
        "arms": datasets,
        "interpretation": (
            "Predeclared small development check, not statistical/causal proof or IID admission. "
            "All sources and failed branches retained; model-assisted review is not a human label. "
            "Requested demonstration success and supervised NLL alone cannot pass this check."
        ),
    }


def main() -> None:
    from .development_collection import read, save

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ARMS:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve the original validation report")
    result = compare_autonomous_use(
        {name: getattr(args, name.replace("-", "_")) for name in ARMS}, read(args.reviews)
    )
    save(args.output, result)
    if not result["autonomous_skill_use_qualified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
