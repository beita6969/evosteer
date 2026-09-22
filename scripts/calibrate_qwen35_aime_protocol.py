#!/usr/bin/env python3
"""Validate and freeze an AIME profile using only pre-final populations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skillev.evaluation.current_iid.protocol13.aime_profiles import (
    load_aime_profile_registry,
    validate_calibration_year,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--calibration-year", type=int, required=True)
    parser.add_argument("--selected-profile", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate_calibration_year(args.calibration_year)
    registry = load_aime_profile_registry(args.config)
    profile = registry.require(args.selected_profile, calibration_year=args.calibration_year)
    receipt = {
        "format": "skillev-aime-protocol-freeze@2",
        "profile_id": profile.profile_id,
        "profile": {
            "prompt_profile": profile.prompt_profile,
            "parser_profile": profile.parser_profile,
            "decoding_profile": profile.decoding_profile,
            "formal_reference_eligible": profile.formal_reference_eligible,
            "evidence_status": profile.evidence.status.value,
        },
        "calibration_population": registry.calibration_population,
        "final_population": registry.final_population,
        "calibration_year": args.calibration_year,
        "selection_rule": "caller-selected-declared-profile-on-pre-final-population",
        "final_population_not_used": True,
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
