"""Single typed generation-profile registry for Protocol 13 runners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from skillev.evaluation.direct_baseline.config import DirectDecodingProfile


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric")
    return float(value)


@dataclass(frozen=True, slots=True)
class Protocol13RunnerProfiles:
    served_model_name: str
    context_length: int
    adapter_policy: str
    profiles: dict[str, DirectDecodingProfile]

    def require(self, profile_id: str) -> DirectDecodingProfile:
        try:
            return self.profiles[profile_id]
        except KeyError as exc:
            raise ValueError(f"Protocol 13 profile is undefined: {profile_id}") from exc


def load_protocol13_runner_profiles(path: Path) -> Protocol13RunnerProfiles:
    root = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(root, dict) or root.get("format") != "skillev-qwen35-protocol13-runner@2":
        raise ValueError("invalid Protocol 13 runner registry")
    if set(root) != {"format", "model", "profiles"}:
        raise ValueError("Protocol 13 runner registry fields differ")
    model = root["model"]
    profiles = root["profiles"]
    if not isinstance(model, dict) or set(model) != {
        "served_model_name",
        "context_length",
        "adapter_policy",
    }:
        raise ValueError("Protocol 13 model profile fields differ")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("Protocol 13 runner profiles are missing")
    parsed: dict[str, DirectDecodingProfile] = {}
    required = {
        "sampling_mode",
        "enable_thinking",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "repetition_penalty",
        "max_new_tokens",
        "seed",
        "stop",
    }
    for profile_id, raw in profiles.items():
        if type(profile_id) is not str or not profile_id.strip() or not isinstance(raw, dict):
            raise ValueError("Protocol 13 generation profile is invalid")
        if set(raw) != required:
            raise ValueError(f"Protocol 13 profile fields differ: {profile_id}")
        stop = raw["stop"]
        if not isinstance(stop, list) or any(type(item) is not str for item in stop):
            raise ValueError(f"Protocol 13 stop sequences are invalid: {profile_id}")
        if type(raw["enable_thinking"]) is not bool:
            raise ValueError(f"Protocol 13 thinking flag is invalid: {profile_id}")
        if type(raw["top_k"]) is not int or type(raw["max_new_tokens"]) is not int:
            raise ValueError(f"Protocol 13 integer controls are invalid: {profile_id}")
        if type(raw["seed"]) is not int:
            raise ValueError(f"Protocol 13 seed is invalid: {profile_id}")
        parsed[profile_id] = DirectDecodingProfile(
            profile_id=profile_id,
            sampling_mode=str(raw["sampling_mode"]),
            enable_thinking=raw["enable_thinking"],
            temperature=_number(raw["temperature"], "temperature"),
            top_p=_number(raw["top_p"], "top p"),
            top_k=raw["top_k"],
            min_p=_number(raw["min_p"], "min p"),
            presence_penalty=_number(raw["presence_penalty"], "presence penalty"),
            repetition_penalty=_number(raw["repetition_penalty"], "repetition penalty"),
            max_new_tokens=raw["max_new_tokens"],
            seed=raw["seed"],
            stop=tuple(stop),
        )
    served_model_name = model["served_model_name"]
    context_length = model["context_length"]
    adapter_policy = model["adapter_policy"]
    if type(served_model_name) is not str or not served_model_name.strip():
        raise ValueError("Protocol 13 served model name is invalid")
    if type(context_length) is not int or context_length <= 0:
        raise ValueError("Protocol 13 context length is invalid")
    if adapter_policy != "forbidden":
        raise ValueError("Protocol 13 runner must forbid adapters")
    return Protocol13RunnerProfiles(
        served_model_name=served_model_name,
        context_length=context_length,
        adapter_policy=adapter_policy,
        profiles=parsed,
    )


__all__ = ["Protocol13RunnerProfiles", "load_protocol13_runner_profiles"]
