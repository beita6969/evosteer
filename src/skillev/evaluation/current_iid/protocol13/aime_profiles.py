"""Pre-final AIME profile evidence and calibration contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .aime_evidence import AIMEProfileEvidence, AIMEProfileEvidenceStatus


@dataclass(frozen=True, slots=True)
class AIMEProfileCandidate:
    profile_id: str
    prompt_profile: str
    parser_profile: str
    decoding_profile: str
    enable_thinking: bool
    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    max_new_tokens: int
    seed: int
    calibration_max_year: int
    evidence: AIMEProfileEvidence

    @property
    def formal_reference_eligible(self) -> bool:
        return self.evidence.formal_reference_eligible


@dataclass(frozen=True, slots=True)
class AIMEProfileRegistry:
    calibration_population: str
    final_population: str
    profiles: tuple[AIMEProfileCandidate, ...]

    def require(self, profile_id: str, *, calibration_year: int) -> AIMEProfileCandidate:
        validate_calibration_year(calibration_year)
        matches = tuple(item for item in self.profiles if item.profile_id == profile_id)
        if len(matches) != 1:
            raise ValueError("selected AIME profile is not uniquely declared")
        profile = matches[0]
        if calibration_year > profile.calibration_max_year:
            raise ValueError("calibration year exceeds profile contract")
        return profile


def validate_calibration_year(year: int) -> None:
    if year >= 2026:
        raise ValueError("AIME 2026 cannot select the evaluation profile")


def load_aime_profile_registry(path: Path) -> AIMEProfileRegistry:
    root = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(root, dict) or root.get("format") != "skillev-aime-protocol-candidates@3":
        raise ValueError("invalid AIME candidate config")
    if set(root) != {"format", "calibration_population", "final_population", "profiles"}:
        raise ValueError("AIME candidate config fields differ")
    if root["final_population"] != "aime-2026-all-30-v13":
        raise ValueError("AIME final population differs from Protocol 13")
    profiles = root["profiles"]
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("AIME candidate config requires profiles")
    parsed: list[AIMEProfileCandidate] = []
    for raw in profiles:
        if not isinstance(raw, dict):
            raise ValueError("AIME profile must be a mapping")
        expected = {
            "id",
            "prompt_profile",
            "parser_profile",
            "decoding_profile",
            "thinking",
            "sampling",
            "max_new_tokens",
            "seed",
            "calibration_max_year",
            "formal_reference_eligible",
            "evidence",
        }
        if set(raw) != expected or not isinstance(raw["sampling"], dict):
            raise ValueError("AIME profile fields differ")
        if set(raw["sampling"]) != {"temperature", "top_p", "top_k", "presence_penalty"}:
            raise ValueError("AIME sampling fields differ")
        evidence = raw["evidence"]
        if not isinstance(evidence, dict) or set(evidence) != {
            "status",
            "repository_or_paper",
            "revision_or_version",
            "path_or_section",
            "supported_fields",
        }:
            raise ValueError("AIME evidence fields differ")
        supported = evidence["supported_fields"]
        if not isinstance(supported, list) or any(type(item) is not str for item in supported):
            raise ValueError("AIME supported evidence fields are invalid")
        if type(raw["thinking"]) is not bool:
            raise ValueError("AIME thinking flag is invalid")
        candidate = AIMEProfileCandidate(
            profile_id=str(raw["id"]),
            prompt_profile=str(raw["prompt_profile"]),
            parser_profile=str(raw["parser_profile"]),
            decoding_profile=str(raw["decoding_profile"]),
            enable_thinking=raw["thinking"],
            temperature=float(raw["sampling"]["temperature"]),
            top_p=float(raw["sampling"]["top_p"]),
            top_k=int(raw["sampling"]["top_k"]),
            presence_penalty=float(raw["sampling"]["presence_penalty"]),
            max_new_tokens=int(raw["max_new_tokens"]),
            seed=int(raw["seed"]),
            calibration_max_year=int(raw["calibration_max_year"]),
            evidence=AIMEProfileEvidence(
                status=AIMEProfileEvidenceStatus(str(evidence["status"])),
                repository_or_paper=str(evidence["repository_or_paper"]),
                revision_or_version=str(evidence["revision_or_version"]),
                path_or_section=str(evidence["path_or_section"]),
                supported_fields=tuple(supported),
            ),
        )
        if type(raw["formal_reference_eligible"]) is not bool:
            raise ValueError("declared AIME eligibility must be boolean")
        if raw["formal_reference_eligible"] is not candidate.formal_reference_eligible:
            raise ValueError("declared AIME eligibility differs from evidence")
        parsed.append(candidate)
    ids = tuple(item.profile_id for item in parsed)
    if len(ids) != len(set(ids)):
        raise ValueError("AIME profile IDs must be unique")
    return AIMEProfileRegistry(
        calibration_population=str(root["calibration_population"]),
        final_population=str(root["final_population"]),
        profiles=tuple(parsed),
    )


__all__ = [
    "AIMEProfileCandidate",
    "AIMEProfileEvidence",
    "AIMEProfileEvidenceStatus",
    "AIMEProfileRegistry",
    "load_aime_profile_registry",
    "validate_calibration_year",
]
