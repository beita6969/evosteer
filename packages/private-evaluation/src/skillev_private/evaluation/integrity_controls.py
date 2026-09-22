"""Effective frozen evaluation controls, separate from live broker execution."""

from __future__ import annotations

import importlib.metadata
import os
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from skillev.evaluation.input_metric_contracts import CONTRACTS, PublicTaskView
from skillev.evaluation.integrity_metric_schema import NATIVE_VERIFIER_VERSIONS
from skillev.evaluation.integrity_results import ExecutionControls
from skillev.evaluation.native_continuation import (
    BOUNDED_THINKING_POLICY,
    NATIVE_CONTINUATION_POLICY,
)
from skillev.evaluation.sampling_stream import SAMPLING_SEED_SCHEDULE
from skillev.evaluation.scienceworld_commands import (
    LEGACY_COMMAND_PROFILE,
    REFERENCE_PROFILE,
    command_profile,
    native_functions,
)
from skillev.evaluation.step0_integrity import InferenceArm
from skillev.evaluation.submission_outcome import SubmissionBudget
from skillev.evolution.task_features import configured_public_task_features

from .architecture_matched import evaluation_isolation
from .interactive_lifecycle import termination_controls

if TYPE_CHECKING:
    from .integrity_runtime import PrivateIntegrityRuntime


def episode_budgets(self: PrivateIntegrityRuntime, entry: PublicTaskView) -> dict[str, Any]:
    limits = {
        **self.config["budgets"][entry.benchmark],
        **self.config.get("budget_overrides", {}).get(entry.benchmark, {}),
    }
    limits["context_length"] = int(self.config["context_length"])
    submission = SubmissionBudget(
        limits["total_output_tokens"], limits.get("finalization_reserve_tokens", 0)
    )
    if submission.finalization_reserve_tokens and (
        "native_chunk_tokens" in limits
        or min(limits["total_model_calls"], limits["calls_per_turn"]) < 2
    ):
        raise ValueError(
            "finalization reserve needs another owner call and a separate non-chunked condition"
        )
    if "native_chunk_tokens" in limits:
        chunk, reserve = limits["native_chunk_tokens"], limits["native_final_reserve_tokens"]
        if (
            type(chunk) is not int
            or type(reserve) is not int
            or not 0 < reserve < limits["total_output_tokens"]
            or chunk < reserve
        ):
            raise ValueError(
                "native continuation requires positive chunk and final reserve budgets"
            )
    if limits.get("native_close_at_reserve", 0) not in (0, 1):
        raise ValueError("native_close_at_reserve must be zero or one")
    if limits.get("native_close_at_reserve", 0) and (
        "native_chunk_tokens" not in limits
        or limits["native_final_reserve_tokens"] < 4
        or min(limits["total_model_calls"], limits["calls_per_turn"]) < 5
    ):
        raise ValueError("bounded thinking needs a delimiter call and final-answer allowance")
    if entry.task_id in self.source.interactive:
        step_limit = self.config.get("environment_horizons", {}).get(
            entry.benchmark,
            int(self.source.interactive[entry.task_id]["case"]["max_steps"]),
        )
        if type(step_limit) is not int or step_limit < 1:
            raise ValueError("native environment action budget must be a positive integer")
        limits["environment_steps"] = step_limit
    return limits


def execution_controls(
    self: PrivateIntegrityRuntime, arm: InferenceArm, entries: tuple[PublicTaskView, ...]
) -> ExecutionControls:
    binding = self._binding(arm)
    skills = self._skill_binding(arm)
    policy = self.policy if binding is None else binding.policy
    benchmarks = sorted({entry.benchmark for entry in entries})
    verifiers = dict(NATIVE_VERIFIER_VERSIONS)
    if "healthbench" in benchmarks:
        from skillev.evaluation.healthbench_luna_profile import PROFILE_ID, VERIFIER

        from .integrity_native_scoring import resolve_healthbench_profile

        if (
            resolve_healthbench_profile(self.config["scorers"]["healthbench"]).profile_id
            == PROFILE_ID
        ):
            verifiers["healthbench"] = VERIFIER
    server_fields: tuple[str, ...] = (
        "context_length",
        "max_running_requests",
        "max_total_tokens",
        "attention_backend",
        "sampling_backend",
        "grammar_backend",
        "enable_deterministic_inference",
        "enable_lora",
        "served_model_name",
        "disable_radix_cache",
        "disable_cuda_graph",
        "cuda_graph_config",
        "linear_attn_backend",
        "mamba_radix_cache_strategy",
        "page_size",
        "mem_fraction_static",
        "chunked_prefill_size",
        "max_prefill_tokens",
        "reasoning_parser",
        "random_seed",
        "dtype",
        "quantization",
        "weight_version",
        "lora_paths",
        "lora_target_modules",
        "max_lora_rank",
        "lora_backend",
        "version",
    )
    serving = []
    seed_authorities = []
    for item in self.observed:
        raw = item["server_info"].get("server_args", item["server_info"])
        arguments = {name: raw.get(name) for name in server_fields}
        # SGLang reports its version beside server_args, not necessarily in it.
        arguments["version"] = raw.get("version") or item["server_info"].get("version")
        serving.append(arguments)
        deterministic = raw.get("enable_deterministic_inference")
        # In deployed SGLang 0.5.9 the disabled branch discards request seeds
        # before sampling. This describes the RNG authority, not a guarantee
        # that the model's kernels produce batch-invariant logits.
        seed_authorities.append(
            "request-seed-enabled"
            if deterministic is True
            else "server-global-rng"
            if deterministic is False
            else "unverified"
        )
    return ExecutionControls(
        model={
            "observed": self.observed[0]["model_info"],
            "adapter_route_sent": None if binding is None else binding.adapter_name,
            "policy": asdict(policy),
            **({"checkpoint": binding.controls()} if binding else {}),
        },
        tokenizer={
            "class": type(self.tokenizer.inner).__name__,
            "vocabulary_size": len(self.tokenizer.inner),
            "chat_template": self.tokenizer.inner.chat_template,
            "transformers_version": importlib.metadata.version("transformers"),
        },
        service={
            "endpoints": self.config["endpoints"],
            "actual_server_arguments": serving,
            "request_routing": self.replicas.policy_id,
            "sampling_seed_schedule": SAMPLING_SEED_SCHEDULE,
            "sampling_seed_authority_by_replica": seed_authorities,
            "interactive_context_policy": "lossless-unexecuted-repair-archive@1",
            "coordinator_concurrency": self.config["concurrency"],
            "request_timeout_seconds": self.config["request_timeout_seconds"],
            "episode_timeout_seconds": self.config["episode_timeout_seconds"],
            "transport_attempts": self.config["transport_attempts"],
            **(
                {
                    "registered_models": [
                        sorted(
                            (
                                {key: card.get(key) for key in ("id", "root", "parent")}
                                for card in item["models"]["data"]
                            ),
                            key=lambda card: str(card["id"]),
                        )
                        for item in self.observed
                    ]
                }
                if self.trained_policies
                else {}
            ),
        },
        public_inputs=tuple((entry.task_id, entry.render()) for entry in entries),
        tools={
            name: (
                ["corpus_search"]
                if name in self.config.get("corpus_retrieval", {})
                else list(native_functions(command_profile(arm.task_semantic_guidance)))
                if name == "scienceworld"
                else list(CONTRACTS[name].tools)
                if name != "triviaqa"
                else []
            )
            for name in benchmarks
        },
        environment={
            "corpus_retrieval": self.config.get("corpus_retrieval", {}),
            "webshop_observation_mode": self.config.get("webshop_observation_mode", "text"),
            "scienceworld_observation_profile": self.config.get(
                "scienceworld_observation_profile", "text-only@1"
            ),
            "scienceworld_termination": termination_controls(self.config),
            **(
                {
                    "scienceworld_commands": command_profile(arm.task_semantic_guidance),
                    "scienceworld_object_references": REFERENCE_PROFILE,
                }
                if command_profile(arm.task_semantic_guidance) != LEGACY_COMMAND_PROFILE
                else {}
            ),
            "native_runtime_environment": {
                name: os.environ.get(name) for name in ("JAVA_HOME", "JVM_PATH", "SDL_VIDEODRIVER")
            },
            "cases": {
                entry.task_id: self.source.interactive[entry.task_id]
                for entry in entries
                if entry.task_id in self.source.interactive
            },
        },
        evaluator={
            "settings": self.config["scorers"],
            "verifier_versions": verifiers,
            "private_targets": {
                entry.task_id: self.source.targets.get(entry.task_id, {}) for entry in entries
            },
            "source_provenance": self.source.provenance,
        },
        parser={
            "architecture_id": self.config.get("architecture_id"),
            "evaluation_isolation": evaluation_isolation(),
            "task_semantic_guidance": arm.task_semantic_guidance,
            "declared_condition": self.config.get("declared_condition"),
            "public_input_profiles": {entry.task_id: entry.input_profile for entry in entries},
            "implementation_revision": self.config["implementation_revision"],
            "benchmarks": {name: CONTRACTS[name].parser for name in benchmarks},
            "tool_call_mode": arm.tool_call_mode.value,
            "thinking_policy": self.thinking_policy.to_value() if self.thinking_policy else None,
            "effective_benchmark_configuration": {
                name: {
                    "native_thinking": self._resolved_arm(
                        next(entry for entry in entries if entry.benchmark == name), arm
                    ).native_thinking,
                    "agent_topology": arm.agent_topology.value,
                    "decoding": asdict(
                        self._profile(
                            next(entry for entry in entries if entry.benchmark == name), arm
                        )
                    ),
                    "budgets": self._budgets(
                        next(entry for entry in entries if entry.benchmark == name)
                    ),
                    "submission_budget_profile": SubmissionBudget(
                        self._budgets(next(entry for entry in entries if entry.benchmark == name))[
                            "total_output_tokens"
                        ],
                        self._budgets(
                            next(entry for entry in entries if entry.benchmark == name)
                        ).get("finalization_reserve_tokens", 0),
                    ).profile_id,
                    "native_continuation": BOUNDED_THINKING_POLICY
                    if self._budgets(
                        next(entry for entry in entries if entry.benchmark == name)
                    ).get("native_close_at_reserve", 0)
                    else NATIVE_CONTINUATION_POLICY
                    if "native_chunk_tokens"
                    in self._budgets(next(entry for entry in entries if entry.benchmark == name))
                    else None,
                    "reasoning_state_context": "verbatim-required-within-declared-context",
                    "python_carrier": "single-python-module@2"
                    if name in {"mbpp-plus", "humaneval"}
                    else None,
                }
                for name in benchmarks
            },
        },
        sampling={
            name: asdict(
                self._profile(next(entry for entry in entries if entry.benchmark == name), arm)
            )
            for name in benchmarks
        },
        budgets={entry.task_id: self._budgets(entry) for entry in entries},
        skills={
            **skills.to_value(),
            "training_provenance": skills.training_provenance(),
            "context_entry_mode": "automatic-retrieval-plus-owner-discovery",
            "task_features": {
                name: configured_public_task_features(
                    name, self.config.get("skill_task_features", {}).get(name, {})
                ).to_value()
                for name in benchmarks
            },
        }
        if skills is not None
        else {},
    )
