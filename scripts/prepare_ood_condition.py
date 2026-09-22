"""Prepare one independently named OOD development condition; never launch generation."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import quote

from skillev.evaluation.step0_integrity import SkillMode, decode_integrity_arm
from skillev.task_semantic_guidance import LEGACY_TASK_SEMANTICS, PUBLIC_TASK_SEMANTICS_V5

CONDITIONS = (
    "native-public-step0",
    "training-shared-semantics-step0",
    "code-finalization-reserve",
    "scienceworld-public-state",
)


def prepare_condition(base: dict[str, Any], condition: str, arm_id: str) -> dict[str, Any]:
    if condition not in CONDITIONS or len(base["arms"]) != 1:
        raise ValueError("choose one named single-owner condition")
    arm = decode_integrity_arm(base["arms"][0])
    arm.validate_live_topology()
    if arm.optimizer_steps or arm.skill_mode is not SkillMode.OFF or arm.arm_id == arm_id:
        raise ValueError("this comparison requires a newly named, skills-off Step-0 arm")
    config = copy.deepcopy(base)
    if condition in {"native-public-step0", "training-shared-semantics-step0"}:
        arm = replace(
            arm,
            task_semantic_guidance=(
                LEGACY_TASK_SEMANTICS
                if condition == "native-public-step0"
                else PUBLIC_TASK_SEMANTICS_V5
            ),
        )
    elif condition == "code-finalization-reserve":
        if not {"livecodebench", "apps-introductory"} & set(config["evaluation_sample_counts"]):
            raise ValueError("code finalization reserve requires a code benchmark population")
        for benchmark in ("livecodebench", "apps-introductory"):
            if benchmark not in config["evaluation_sample_counts"]:
                continue
            limits = {
                **config["budgets"][benchmark],
                **config.get("budget_overrides", {}).get(benchmark, {}),
            }
            decoding = {
                **config["decoding"][benchmark],
                **config.get("decoding_overrides", {}).get(benchmark, {}),
            }
            if (
                limits["total_output_tokens"] != 12000
                or decoding["max_new_tokens"] != 12000
                or "native_chunk_tokens" in limits
            ):
                raise ValueError(
                    "compare reserve against the unchanged single-response 12000-token condition"
                )
            config.setdefault("budget_overrides", {}).setdefault(benchmark, {})[
                "finalization_reserve_tokens"
            ] = 1000
    else:
        if "scienceworld" not in config["evaluation_sample_counts"]:
            raise ValueError("public-state observations require ScienceWorld episodes")
        config["scienceworld_observation_profile"] = "public-state@1"
    config["arms"] = [replace(arm, arm_id=arm_id).to_value()]
    # A different owner submission cannot reuse the previous condition's judge
    # ledger. Keep those files untouched, even though the judge protocol is fixed.
    for settings in config.get("scorers", {}).values():
        if "judgements_path" in settings:
            raise ValueError(
                "new generation requires fresh judging, not fixed historical judgements"
            )
        if "cache_directory" in settings:
            settings["cache_directory"] = str(
                Path(settings["cache_directory"]) / "conditions" / quote(arm_id, safe="")
            )
    config["declared_condition"] = condition
    config["ood_panel_exposure"] = "inspected-development; " + base.get(
        "ood_panel_exposure", "source exposure unspecified"
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_condition(json.loads(args.base.read_text()), args.condition, args.arm_id)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {"condition": args.condition, "arm_id": args.arm_id, "generation_started": False}
        )
    )


if __name__ == "__main__":
    main()
