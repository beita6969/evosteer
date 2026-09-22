"""Executable seven-IID runtime with isolated owners and a public I/O broker."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
import urllib.request
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

from transformers import AutoTokenizer

from skillev.evaluation.actor_sandbox import ActorSandbox
from skillev.evaluation.capability_registry import native_tool_definitions
from skillev.evaluation.decision_transport import (
    ExplicitDecision,
    PublicSurface,
    normalize_decision,
)
from skillev.evaluation.direct_baseline.config import DirectDecodingProfile
from skillev.evaluation.input_metric_contracts import (
    CONTRACTS,
    OOD_BENCHMARKS,
    PublicTaskView,
    require_runtime_public_fields,
)
from skillev.evaluation.integrity_actor import decode_arm
from skillev.evaluation.integrity_final_validation import (
    project_completed_owner_final,
    terminal_mode,
    validate_final_identity,
    validate_owner_projection,
    validate_persisted_owner_source,
)
from skillev.evaluation.integrity_pipeline import FrozenPanel
from skillev.evaluation.integrity_results import ExecutionControls, NativeScore
from skillev.evaluation.journaled_environment import JournaledEnvironment
from skillev.evaluation.model_output_provenance import (
    CandidateStatus,
    StoredModelOutput,
    served_token_stream,
)
from skillev.evaluation.native_channels import ChannelStatus, split_native_channels
from skillev.evaluation.native_continuation import (
    native_call_allowance,
    thinking_boundary_at_reserve,
)
from skillev.evaluation.native_tool_calls import NativeTools
from skillev.evaluation.sampling_stream import evaluation_call_seed
from skillev.evaluation.scienceworld_commands import command_profile
from skillev.evaluation.sealed_candidates import (
    CandidateJournal,
    CandidateReader,
    EventOrigin,
    FinalCandidate,
    authorize_actor_trace,
)
from skillev.evaluation.skill_library_config import FrozenSkillLibrary, initial_skill_library
from skillev.evaluation.step0_completion import StepZeroTerminalMode
from skillev.evaluation.step0_integrity import (
    InferenceArm,
    InterventionCounts,
    SkillMode,
    ToolCallMode,
)
from skillev.evaluation.step0_types import ArchitectureInferenceState
from skillev.evaluation.thinking_policy import ThinkingPolicy
from skillev.evolution.task_features import configured_public_task_features
from skillev.rollout import GenerationPhase, RolloutGenerationRequest
from skillev.rollout.evaluation_sglang import (
    EvaluationGenerationConstraint,
    EvaluationGenerationProfile,
    EvaluationPolicyDescriptor,
    EvaluationSGLangRolloutGenerator,
)
from skillev.rollout.external_sglang import ExternalSGLangRolloutConfig
from skillev_private.evaluation.integrity_controls import execution_controls
from skillev_private.evaluation.integrity_environments import create_native_environment
from skillev_private.evaluation.integrity_grader_usage import IncompleteNativeGradingError
from skillev_private.evaluation.integrity_mbpp import resolve_mbpp_profile
from skillev_private.evaluation.integrity_native_scoring import (
    resolve_healthbench_profile,
    score_native,
)
from skillev_private.evaluation.integrity_policies import TrainedPolicyBinding
from skillev_private.evaluation.integrity_replicas import ReplicaLoadBalancer
from skillev_private.evaluation.integrity_skills import read_skill_library
from skillev_private.evaluation.integrity_sources import (
    SourcePanel,
    load_source_panel,
    order_source_panel,
    select_source_panel,
)
from skillev_private.evaluation.interactive_lifecycle import InteractiveLifecycle
from skillev_private.evaluation.ood_retrieval import corpus_session


class PublicServiceTokenizer:
    tokenizer_id = "qwen35-service-tokenizer"

    def __init__(self, path: str) -> None:
        self.inner = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )

    def encode(self, text: str) -> list[int]:
        return cast(list[int], self.inner.encode(text, add_special_tokens=False))

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return cast(
            str,
            self.inner.decode(
                list(token_ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
            ),
        )

    def encode_rollout_prompt(self, text: str) -> list[int]:
        return self.encode_integrity_messages(
            ({"role": "user", "content": text},), enable_thinking=False
        )

    def encode_integrity_messages(
        self,
        messages: tuple[dict[str, str], ...],
        *,
        enable_thinking: bool,
        tools: NativeTools = (),
    ) -> list[int]:
        return cast(
            list[int],
            self.inner.apply_chat_template(
                list(messages),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                return_dict=False,
                tools=list(tools) if tools else None,
            ),
        )


def _server(endpoint: str, *, include_models: bool = False) -> dict[str, Any]:
    result = {}
    for name in ("model_info", "server_info"):
        with urllib.request.urlopen(endpoint.rstrip("/") + "/" + name, timeout=15) as response:  # noqa: S310 -- owner-configured local SGLang endpoint
            result[name] = json.load(response)
    if include_models:
        with urllib.request.urlopen(endpoint.rstrip("/") + "/v1/models", timeout=15) as response:  # noqa: S310 -- same owner-configured SGLang service
            result["models"] = json.load(response)
    return result


def _interpreter_sandbox(
    python: Path,
    source: Path,
    bubblewrap: Path,
    *,
    dependencies: tuple[Path, ...] = (),
) -> ActorSandbox:
    completed = subprocess.run(  # noqa: S603 -- owner-configured Python, CPU metadata only
        [
            str(python),
            "-c",
            "import sys,json,os; "
            "print(json.dumps([sys.executable,sys.prefix,sys.base_prefix,"
            "os.environ.get('PYTHONPATH','')]))",
        ],
        # Do not inherit the coordinator's private source/data search path.
        # Only paths explicitly established by the owner-selected wrapper are
        # dependencies of this trusted scorer (not of the evaluated actor).
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": ""},
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    executable, prefix, base, pythonpath = json.loads(completed.stdout)
    wrapper_dependencies = tuple(Path(path) for path in pythonpath.split(os.pathsep) if path)
    return ActorSandbox(
        source,
        Path(executable),
        Path(prefix),
        Path(base),
        bubblewrap,
        dependencies,
        wrapper_dependencies,
    )


class PrivateIntegrityRuntime:
    def __init__(self, config: dict[str, Any], source: SourcePanel, private_output: Path) -> None:
        if config["transport_attempts"] != 1:
            raise ValueError(
                "clean evaluation cannot automatically replay a request with unknown usage"
            )
        for value in config["arms"]:
            decode_arm(value).validate_live_topology()
        self.config, self.source, self.directory = config, source, private_output
        if "thinking_policy" not in config:
            raise ValueError(
                "new owner-source evaluation requires its explicit benchmark thinking map"
            )
        self.thinking_policy: ThinkingPolicy | None = ThinkingPolicy.from_value(
            config["thinking_policy"]
        )
        private_output.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.observed = [
            _server(endpoint, include_models=bool(config.get("trained_policies")))
            for endpoint in config["endpoints"]
        ]
        first = self.observed[0]["model_info"]
        if first.get("model_type") != "qwen3_5" or "Qwen3.5-9B" not in first["model_path"]:
            raise ValueError("the actual service is not the declared Qwen3.5-9B backbone")
        if any(item["model_info"] != first for item in self.observed[1:]):
            raise ValueError("paired replicas serve different model/tokenizer identities")
        self.tokenizer = PublicServiceTokenizer(first["tokenizer_path"])
        self.policy = EvaluationPolicyDescriptor(
            "qwen35-9b:step0:adapter-free@integrity1", "Qwen3.5-9B", self.tokenizer.tokenizer_id
        )
        self.trained_policies = {
            policy_id: TrainedPolicyBinding.read(
                policy_id,
                settings,
                observed=self.observed,
                tokenizer_id=self.tokenizer.tokenizer_id,
            )
            for policy_id, settings in config.get("trained_policies", {}).items()
        }
        if self.policy.snapshot_id in self.trained_policies:
            raise ValueError("a trained policy cannot reuse the adapter-free identity")
        routes = [binding.adapter_name for binding in self.trained_policies.values()]
        if len(set(routes)) != len(routes):
            raise ValueError("each configured trained policy needs its own named route")
        arms = tuple(decode_arm(value) for value in config["arms"])
        self.skill_libraries = {
            library_id: read_skill_library(
                library_id, config.get("skill_libraries", {})[library_id]
            )
            for library_id in {
                arm.skill_library_id for arm in arms if arm.skill_library_id is not None
            }
        }
        self.initial_skills = (
            initial_skill_library()
            if any(
                arm.skill_mode in {SkillMode.GENERIC_TEXT, SkillMode.CAPABILITY_RETRIEVED_TEXT}
                for arm in arms
            )
            else None
        )
        for arm in arms:
            self._binding(arm)
            self._skill_binding(arm)
        self.generators = self._make_generators(self.policy, None)
        self.trained_generators = {
            policy_id: self._make_generators(binding.policy, binding.adapter_name)
            for policy_id, binding in self.trained_policies.items()
        }
        self.replicas = ReplicaLoadBalancer(
            len(self.generators),
            context_lengths=tuple(
                item["server_info"].get("server_args", item["server_info"])["context_length"]
                for item in self.observed
            ),
        )
        public_source = Path(__file__).resolve().parents[5] / "src"
        if not (public_source / "skillev").is_dir():
            public_source = Path(config["public_source"])
        self.sandbox = replace(
            ActorSandbox.current(public_source), bubblewrap=Path(config["bubblewrap"])
        )
        self.mbpp_sandbox = self.sandbox
        self.mbpp_limiter = asyncio.Semaphore(1)
        if any(entry.benchmark == "mbpp-plus" for entry in source.panel.entries):
            self.mbpp_sandbox = _interpreter_sandbox(
                Path(config["scorers"]["mbpp-plus"]["python"]),
                public_source,
                self.sandbox.bubblewrap,
                # The deployed evaluator's .pth refers to this companion runtime.
                # Mount only its dependency prefix, not the surrounding data tree.
                dependencies=(self.sandbox.python_prefix,),
            )
            mbpp_profile = resolve_mbpp_profile(config["scorers"]["mbpp-plus"])
            config["scorers"]["mbpp-plus"]["profile"] = mbpp_profile.to_value()
            self.mbpp_limiter = asyncio.Semaphore(mbpp_profile.maximum_concurrency)
        if any(entry.benchmark == "healthbench" for entry in source.panel.entries):
            health_settings = config["scorers"]["healthbench"]
            health_settings["effective_profile"] = asdict(
                resolve_healthbench_profile(health_settings)
            )
        self.indices = {entry.task_id: index for index, entry in enumerate(source.panel.entries)}
        self.journal = CandidateJournal(private_output / "candidates-private.sqlite")

    async def refresh(self) -> None:
        self.observed = list(
            await asyncio.gather(
                *(
                    asyncio.to_thread(
                        _server, url, include_models=bool(self.config.get("trained_policies"))
                    )
                    for url in self.config["endpoints"]
                )
            )
        )
        for binding in self.trained_policies.values():
            binding.validate(self.observed)

    def _make_generators(
        self, policy: EvaluationPolicyDescriptor, adapter_name: str | None
    ) -> list[EvaluationSGLangRolloutGenerator]:
        # Separate immutable route pools; concurrent arms never mutate a shared adapter selector.
        return [
            EvaluationSGLangRolloutGenerator(
                ExternalSGLangRolloutConfig(
                    endpoint,
                    request_timeout_seconds=float(self.config["request_timeout_seconds"]),
                    transport_worker_threads=int(self.config["concurrency"]),
                ),
                self.tokenizer,
                lambda: policy,
                adapter_name=adapter_name,
                transport_maximum_attempts=1,
            )
            for endpoint in self.config["endpoints"]
        ]

    def _binding(self, arm: InferenceArm) -> TrainedPolicyBinding | None:
        if arm.policy_id is None:
            return None
        binding = self.trained_policies.get(arm.policy_id)
        if binding is None:
            raise ValueError("the arm's explicitly selected trained policy is not configured")
        binding.require_arm(arm)
        return binding

    def _budgets(self, entry: PublicTaskView) -> dict[str, Any]:
        from .integrity_controls import episode_budgets

        return episode_budgets(self, entry)

    def _skill_binding(self, arm: InferenceArm) -> FrozenSkillLibrary | None:
        if arm.skill_mode is SkillMode.OFF:
            return None
        if arm.skill_mode is not SkillMode.LIBRARY:
            if self.initial_skills is None:
                raise ValueError("fixed initial skills were not configured for this run")
            return self.initial_skills
        snapshot = self.skill_libraries.get(arm.skill_library_id or "")
        if snapshot is None or snapshot.retrieval_rule != arm.skill_retrieval_rule:
            raise ValueError("the selected skill library and retrieval rule are not configured")
        return snapshot

    def _profile(self, entry: PublicTaskView, arm: InferenceArm) -> DirectDecodingProfile:
        arm = self._resolved_arm(entry, arm)
        values = {
            **self.config["decoding"][entry.benchmark],
            **self.config.get("decoding_overrides", {}).get(entry.benchmark, {}),
        }
        return DirectDecodingProfile(
            **{
                **values,
                "enable_thinking": arm.native_thinking,
                "stop": tuple(values.get("stop", ())),
            }
        )

    def _resolved_arm(self, entry: PublicTaskView, arm: InferenceArm) -> InferenceArm:
        arm.validate_live_topology()
        return self.thinking_policy.resolve(entry.benchmark, arm) if self.thinking_policy else arm

    def controls(self, arm: InferenceArm, entries: tuple[PublicTaskView, ...]) -> ExecutionControls:
        return execution_controls(self, arm, entries)

    def interventions(self, arm: InferenceArm) -> InterventionCounts:
        total = InterventionCounts()
        rows = self.journal.connection.execute(
            "SELECT payload FROM candidates WHERE arm_id=?", (arm.arm_id,)
        ).fetchall()
        for row in rows:
            for name, value in json.loads(row[0])["intervention_counts"].items():
                setattr(total, name, getattr(total, name) + value)
        return total

    async def generate(
        self, entry: PublicTaskView, arm: InferenceArm, run_id: str
    ) -> FinalCandidate:
        arm = self._resolved_arm(entry, arm)
        scope = (run_id, arm.arm_id, entry.task_id)
        try:
            self.journal.get(*scope)
        except KeyError:
            pass
        else:
            raise ValueError("a submitted final cannot be regenerated")
        if self.journal.traces(scope, "environment-reset") or self.journal.traces(
            scope, "rendered-request"
        ):
            raise RuntimeError(
                "interrupted episode needs reconciliation, not a fresh generation budget or reset"
            )
        binding = self._binding(arm)
        policy = self.policy if binding is None else binding.policy
        adapter_name = None if binding is None else binding.adapter_name
        skills = self._skill_binding(arm)
        generators = (
            self.generators
            if binding is None
            else self.trained_generators[binding.policy.snapshot_id]
        )
        profile, budgets = self._profile(entry, arm), self._budgets(entry)
        attempt_id = self.journal.start_attempt(scope, policy_id=policy.snapshot_id)
        corpus = corpus_session(self.config, entry, self.journal, scope)
        episode_started = time.monotonic()
        lifecycle = InteractiveLifecycle.for_episode(
            entry.benchmark, self.config, episode_started, self.journal, scope
        )
        environment: JournaledEnvironment | None = None
        model_calls = input_tokens = output_tokens = 0
        rendered_request: dict[str, Any] | None = None
        last_response = ""
        last_participant = ""
        last_response_revision = 0
        last_call_id = ""
        last_channel_status = ChannelStatus.FINAL_EMPTY
        last_finish_reason = ""
        last_thinking_boundary = None
        last_input_ids: tuple[int, ...] = ()
        stream_prompt_ids: tuple[int, ...] = ()
        stream_tokens: tuple[int, ...] = ()
        owner_final_ready = False
        public_actions: tuple[str, ...] = ()
        executed_actions: list[str] = []
        owner_tools = (
            native_tool_definitions(
                entry.benchmark
                if entry.benchmark in {"webshop", "alfworld", "scienceworld"}
                else "completion",
                skills=skills is not None,
                natural_language=terminal_mode(entry.benchmark)
                is StepZeroTerminalMode.NATURAL_LANGUAGE,
                owner_finish=lifecycle is not None,
                scienceworld_profile=command_profile(arm.task_semantic_guidance),
                corpus_search=corpus is not None,
            )
            if arm.tool_call_mode is ToolCallMode.QWEN_XML
            else ()
        )
        from skillev.evaluation.input_identity import SourceIdentity
        from skillev.evaluation.public_input_receipt import describe_public_input
        from skillev.evaluation.submission_outcome import SubmissionBudget, SubmissionOutcome

        source_identity = SourceIdentity.for_entry(
            entry,
            self.source.provenance,
            panel_id=self.config.get("panel_id", f"{run_id}:panel"),
            exposure_status=self.source.panel.exposure,
        )
        self.journal.record(
            scope, "source-identity", asdict(source_identity), origin=EventOrigin.MODEL_TRANSPORT
        )
        initial = {
            "public_task": asdict(entry),
            "corpus_search": asdict(corpus.profile) if corpus is not None else None,
            "arm": arm.to_value(),
            "run_id": run_id,
            "decoding": asdict(profile),
            "budgets": budgets,
            "owner_finish_allowed": lifecycle is not None,
            "deadline_monotonic": lifecycle.deadline if lifecycle else None,
            "policy": asdict(policy),
            "inference_state": asdict(
                ArchitectureInferenceState() if binding is None else binding.inference_state
            ),
            "population_id": source_identity.population_id
            or f"unresolved-source:{entry.benchmark}",
            "source_identity": asdict(source_identity),
            "panel_position": self.indices[entry.task_id],
            "attempt_id": attempt_id,
            "skill_library": skills.to_value() if skills is not None else None,
            "skill_task_features": configured_public_task_features(
                entry.benchmark, self.config.get("skill_task_features", {}).get(entry.benchmark, {})
            ).to_value()
            if skills is not None
            else {},
        }

        async def handle(message: dict[str, Any]) -> object:
            nonlocal environment, model_calls, input_tokens, output_tokens
            nonlocal rendered_request, last_response, last_participant, last_response_revision
            nonlocal public_actions
            nonlocal last_call_id, last_channel_status, owner_final_ready
            nonlocal last_input_ids, stream_prompt_ids, stream_tokens, last_finish_reason
            nonlocal last_thinking_boundary
            operation = message["operation"]
            if operation == "corpus-search":
                if corpus is None:
                    raise ValueError("no corpus tool is enabled in this condition")
                return await corpus.execute(
                    message["query"],
                    response=last_response,
                    call_id=last_call_id,
                    participant=last_participant,
                    final_ready=owner_final_ready,
                )
            if operation == "trace":
                origin = authorize_actor_trace(message["stage"])
                self.journal.record(scope, message["stage"], message["payload"], origin=origin)
                if message["stage"] == "rendered-request":
                    rendered_request = message["payload"]
                return None
            if operation == "encode":
                return self.tokenizer.encode(message["text"])
            if operation == "decode":
                return self.tokenizer.decode(tuple(message["token_ids"]))
            if operation == "encode-messages":
                template_options = (
                    {"tools": tuple(message["tools"])} if message.get("tools") else {}
                )
                return self.tokenizer.encode_integrity_messages(
                    tuple(message["messages"]),
                    enable_thinking=arm.native_thinking,
                    **template_options,
                )
            if operation == "generate":
                if lifecycle is not None and not lifecycle.admit(operation):
                    return {"episode_deadline_reached": True}
                if owner_final_ready:
                    raise ValueError(
                        "the owner already produced a valid final; no replacement generation"
                    )
                if rendered_request is None:
                    raise ValueError("model request has no observed public prompt boundary")
                raw_profile = message["profile"]
                for name in (
                    "enable_thinking",
                    "temperature",
                    "top_p",
                    "top_k",
                    "min_p",
                    "presence_penalty",
                    "repetition_penalty",
                    "sampling_mode",
                    "stop",
                ):
                    if json.dumps(raw_profile[name]) != json.dumps(asdict(profile)[name]):
                        raise ValueError(
                            "actual actor sampling differs from the frozen paired profile"
                        )
                if raw_profile["seed"] != evaluation_call_seed(
                    profile.seed, model_calls, sampling_mode=profile.sampling_mode
                ):
                    raise ValueError("actual actor seed differs from its declared call substream")
                allowance = SubmissionBudget(
                    budgets["total_output_tokens"], budgets.get("finalization_reserve_tokens", 0)
                ).allowance(output_tokens, model_calls)
                expected_boundary = thinking_boundary_at_reserve(
                    enabled=arm.native_thinking and budgets.get("native_close_at_reserve", 0) == 1,
                    continuing=rendered_request.get("continuation_of_call_id") is not None,
                    remaining=budgets["total_output_tokens"] - output_tokens,
                    reserve=budgets.get("native_final_reserve_tokens", 0),
                    channel_status=last_channel_status,
                    previous_boundary=last_thinking_boundary,
                )
                if raw_profile.get("thinking_boundary") != expected_boundary:
                    raise ValueError("channel closure differs from the frozen budget policy")
                if expected_boundary is not None and raw_profile["max_new_tokens"] != 1:
                    raise ValueError("only one channel delimiter may be forced")
                if arm.native_thinking and "native_chunk_tokens" in budgets:
                    allowance = native_call_allowance(
                        allowance,
                        budgets["native_chunk_tokens"],
                        budgets["native_final_reserve_tokens"],
                    )
                if model_calls >= budgets["total_model_calls"] or raw_profile[
                    "max_new_tokens"
                ] > min(profile.max_new_tokens, allowance):
                    raise ValueError("actual actor request exceeds its declared budget")
                raw_request = message["request"]
                request = RolloutGenerationRequest(
                    **{
                        **raw_request,
                        "input_ids": tuple(raw_request["input_ids"]),
                        "phase": GenerationPhase(raw_request["phase"]),
                    }
                )
                evaluation_profile = EvaluationGenerationProfile(
                    **{**raw_profile, "stop": tuple(raw_profile["stop"])}
                )
                participant = rendered_request.get("participant")
                expected_tools = (
                    owner_tools
                    if participant == "owner"
                    and rendered_request.get("purpose") in {"decision", "interface-repair"}
                    else ()
                )
                if tuple(rendered_request.get("tools", ())) != expected_tools:
                    raise ValueError("actual tool definitions differ from the public capabilities")
                encoding_options = {"tools": expected_tools} if expected_tools else {}
                expected_input = self.tokenizer.encode_integrity_messages(
                    tuple(rendered_request["messages"]),
                    enable_thinking=arm.native_thinking,
                    **encoding_options,
                )
                view = rendered_request.get("public_view")
                revision = environment.revision if environment is not None else 0
                continuation_id = rendered_request.get("continuation_of_call_id")
                if continuation_id is not None:
                    if (
                        not arm.native_thinking
                        or "native_chunk_tokens" not in budgets
                        or continuation_id != last_call_id
                        or last_participant != "owner"
                        or last_response_revision != revision
                        or last_finish_reason != "length"
                        or last_channel_status
                        not in {
                            ChannelStatus.REASONING_UNFINISHED,
                            ChannelStatus.FINAL_EMPTY,
                            ChannelStatus.FINAL_UNFINISHED,
                        }
                        or tuple(expected_input) != stream_prompt_ids
                    ):
                        raise ValueError(
                            "request is not the preceding same-owner native continuation"
                        )
                    expected_input = list(last_input_ids)
                else:
                    stream_tokens = ()
                    stream_prompt_ids = tuple(expected_input)
                if (
                    participant != "owner"
                    or tuple(expected_input) != request.input_ids
                    or request.expected_policy_snapshot_id != policy.snapshot_id
                    or request.max_new_tokens != evaluation_profile.max_new_tokens
                    or request.seed != evaluation_profile.seed
                    or len(request.input_ids) + request.max_new_tokens > budgets["context_length"]
                    or (
                        participant == "owner"
                        and (not isinstance(view, dict) or view.get("revision") != revision)
                    )
                ):
                    raise ValueError(
                        "actual model request differs from its public execution boundary"
                    )
                constraint = (
                    EvaluationGenerationConstraint(**message["constraint"])
                    if message["constraint"]
                    else None
                )
                call_id = f"model-call-{model_calls + 1}"
                public_receipt = describe_public_input(
                    entry,
                    self.tokenizer.decode(request.input_ids),
                    input_tokens=len(request.input_ids),
                    omitted_history_messages=rendered_request.get("archived_message_count", 0),
                    source_paragraphs=getattr(self.source, "public_input_receipts", {}).get(
                        entry.task_id
                    ),
                )
                self.journal.record(
                    scope,
                    "public-input-receipt",
                    {"call_id": call_id, **asdict(public_receipt)},
                    origin=EventOrigin.MODEL_TRANSPORT,
                )
                model_calls += 1
                with self.replicas.acquire(
                    required_context_tokens=len(request.input_ids) + request.max_new_tokens
                ) as replica:
                    self.journal.record(
                        scope,
                        "model-transport-start",
                        {
                            "call_id": call_id,
                            "attempt_id": attempt_id,
                            "replica": replica,
                            "maximum_output_tokens": request.max_new_tokens,
                            "transport_attempts": 1,
                            "participant": participant,
                            "purpose": rendered_request.get("purpose"),
                            "public_revision": revision,
                            "input_tokens": len(request.input_ids),
                            "input_token_ids": request.input_ids,
                            "tool_call_mode": arm.tool_call_mode.value,
                            "tools": expected_tools,
                        },
                        origin=EventOrigin.MODEL_TRANSPORT,
                    )
                    try:
                        result = await generators[replica].generate_evaluation(
                            request, profile=evaluation_profile, constraint=constraint
                        )
                    except (Exception, asyncio.CancelledError) as error:
                        self.journal.record(
                            scope,
                            "model-transport-failure",
                            {
                                "call_id": call_id,
                                "failure_type": type(error).__name__,
                                "usage": "unknown; no automatic replay",
                            },
                            origin=EventOrigin.MODEL_TRANSPORT,
                        )
                        raise
                    if result.policy_snapshot_id != policy.snapshot_id:
                        raise ValueError("the service response belongs to another evaluated policy")
                    stream_tokens += result.content_token_ids
                    channels = split_native_channels(
                        stream_tokens,
                        enabled=arm.native_thinking,
                        tokenizer=self.tokenizer,
                        finish_reason=result.finish_reason,
                    )
                    last_response = self.tokenizer.decode(channels.final_token_ids)
                    last_channel_status = channels.status
                    output_outcome = SubmissionOutcome.after_response(
                        has_final=(
                            entry.benchmark not in {"webshop", "alfworld", "scienceworld"}
                            and rendered_request.get("purpose") in {"decision", "interface-repair"}
                            and channels.status is ChannelStatus.COMPLETE
                            and project_completed_owner_final(
                                terminal_mode(entry.benchmark),
                                last_response,
                                message_id=f"{entry.task_id}:owner-final",
                            )
                            is not None
                        ),
                        finish_reason=result.finish_reason,
                        remaining_tokens=budgets["total_output_tokens"]
                        - output_tokens
                        - result.usage.output_tokens,
                        remaining_calls=budgets["total_model_calls"] - model_calls,
                    )
                    self.journal.record_model_output(
                        StoredModelOutput(
                            run_id=run_id,
                            arm_id=arm.arm_id,
                            episode_id=entry.task_id,
                            attempt_id=attempt_id,
                            call_id=call_id,
                            policy_id=policy.snapshot_id,
                            benchmark=entry.benchmark,
                            participant=str(participant),
                            purpose=str(rendered_request.get("purpose")),
                            public_revision=revision,
                            native_thinking=arm.native_thinking,
                            adapter_name=adapter_name,
                            budgets=budgets,
                            result=result.to_value(),
                            final_token_ids=channels.final_token_ids,
                            final_text=last_response,
                            channel_status=channels.status,
                            continuation_of_call_id=continuation_id,
                            submission_outcome=asdict(output_outcome),
                        )
                    )
                last_call_id = call_id
                last_participant = str(participant)
                last_response_revision = revision
                last_input_ids = request.input_ids + result.content_token_ids
                last_finish_reason = result.finish_reason
                last_thinking_boundary = evaluation_profile.thinking_boundary
                owner_final_ready = output_outcome.final_carrier_present
                rendered_request = None
                input_tokens += result.usage.input_tokens
                output_tokens += result.usage.output_tokens
                return result.to_value()
            if operation == "environment-reset":
                if environment is not None or entry.task_id not in self.source.interactive:
                    raise ValueError("task has no uninitialized native environment")
                delegate = await asyncio.to_thread(
                    create_native_environment,
                    self.source.interactive[entry.task_id],
                    seed=profile.seed,
                    maximum_steps=budgets["environment_steps"],
                    webshop_observation_mode=self.config.get("webshop_observation_mode", "text"),
                    scienceworld_observation_profile=self.config.get(
                        "scienceworld_observation_profile", "text-only@1"
                    ),
                    scienceworld_command_profile=command_profile(arm.task_semantic_guidance),
                    stderr_path=self.directory
                    / "environment-logs"
                    / (f"{arm.arm_id}-{self.indices[entry.task_id]}.log"),
                )
                environment = JournaledEnvironment(
                    delegate,
                    self.journal,
                    scope,
                    owner="owner",
                    maximum_steps=budgets["environment_steps"],
                )
                state = await environment.reset()
                if isinstance(state, str):
                    raise TypeError("native environment must supply its complete public surface")
                require_runtime_public_fields(
                    entry.benchmark,
                    {
                        "observation": state.observation_text,
                        "available_actions": state.available_actions,
                    },
                )
                public_actions = state.available_actions
                return asdict(state)
            if operation == "owner-finish":
                if lifecycle is None or environment is None:
                    raise ValueError("owner finish is not enabled for this environment")
                lifecycle.finish(last_participant, last_response, last_call_id)
                owner_final_ready = True
                return None
            if operation == "environment-step":
                if lifecycle is not None and not lifecycle.admit(operation):
                    return {"episode_deadline_reached": True}
                if environment is None or environment.revision > budgets["environment_steps"]:
                    raise ValueError("native action has no active environment budget")
                source_revision = message.get("source_revision")
                if (
                    type(source_revision) is not int
                    or source_revision != last_response_revision
                    or last_participant != "owner"
                ):
                    raise ValueError("native action is not based on the current owner's decision")
                transported = normalize_decision(
                    ExplicitDecision(last_response, source_revision),
                    PublicSurface(entry.benchmark, environment.revision, public_actions),
                )
                if transported.action is None or transported.action != message["action"]:
                    raise ValueError("native action differs from the actual explicit owner intent")
                result_step = await environment.execute(
                    "owner", message["decision_id"], message["action"], source_revision
                )
                executed_actions.append(message["action"])
                self.journal.record(
                    scope,
                    "owner-action-source",
                    {
                        "attempt_id": attempt_id,
                        "call_id": last_call_id,
                        "decision_id": message["decision_id"],
                        "action": message["action"],
                        "source_revision": source_revision,
                    },
                    origin=EventOrigin.ENVIRONMENT,
                )
                public_actions = result_step.available_actions
                # Rewards and task success never cross into the actor/peer process.
                return {
                    "observation": result_step.observation,
                    "terminal": result_step.terminal,
                    "action_valid": result_step.action_valid,
                    "available_actions": result_step.available_actions,
                }
            raise ValueError("operation is outside the public actor capability space")

        try:
            raw = await self.sandbox.run(
                initial,
                handle,
                stderr_path=self.directory
                / "actor-logs"
                / f"{arm.arm_id}-{self.indices[entry.task_id]}.log",
                timeout_seconds=lifecycle.hard_timeout
                if lifecycle
                else float(self.config["episode_timeout_seconds"]),
            )
            candidate = FinalCandidate(**raw)
            if candidate.attempt_id not in (None, attempt_id) or candidate.owner_call_id not in (
                None,
                last_call_id,
            ):
                raise ValueError("actor candidate claims a different source attempt or call")
            validate_final_identity(
                candidate,
                expected_scope=scope,
                expected_policy=policy.snapshot_id,
                expected_parser=CONTRACTS[entry.benchmark].parser,
                expected_message_id=f"{entry.task_id}:owner-final",
            )
            if environment is not None:
                if json.loads(candidate.text) != executed_actions:
                    raise ValueError("native final differs from the acknowledged owner action log")
            else:
                validate_owner_projection(
                    candidate,
                    raw_response=last_response
                    if last_channel_status is ChannelStatus.COMPLETE
                    else "",
                    participant=last_participant,
                    mode=terminal_mode(entry.benchmark),
                )
            if (
                candidate.prompt_tokens,
                candidate.completion_tokens,
                candidate.intervention_counts["model_calls"],
            ) != (input_tokens, output_tokens, model_calls):
                raise ValueError("actor-reported cost differs from actual serving calls")
            candidate_counts = InterventionCounts(**candidate.intervention_counts)
            candidate_counts.require_arm(arm)
            terminal_status = (
                CandidateStatus.SUBMITTED
                if environment is not None or candidate.text
                else CandidateStatus.BUDGET_EXHAUSTED
                if model_calls >= min(budgets["total_model_calls"], budgets["calls_per_turn"])
                or output_tokens >= budgets["total_output_tokens"]
                else CandidateStatus.MODEL_NO_FINAL
                if last_channel_status is not ChannelStatus.COMPLETE or not last_response.strip()
                else CandidateStatus.MODEL_FORMAT_INVALID
            )
            candidate = replace(
                candidate,
                attempt_id=attempt_id,
                owner_call_id=last_call_id,
                terminal_status=terminal_status,
            )
            # Preserve the model's unique final even if native outcome/cleanup
            # later fails. Resume may repair scoring infrastructure, not regenerate.
            self.journal.seal(candidate)
            submission_outcome = SubmissionOutcome.after_response(
                has_final=candidate.submission is not None or environment is not None,
                finish_reason=last_finish_reason,
                remaining_tokens=budgets["total_output_tokens"] - output_tokens,
                remaining_calls=budgets["total_model_calls"] - model_calls,
            ).seal()
            self.journal.record(
                scope,
                "submission-outcome",
                asdict(submission_outcome),
                origin=EventOrigin.MODEL_TRANSPORT,
            )
            if environment is not None:
                if lifecycle is None:
                    await environment.outcome()
                else:
                    await lifecycle.capture_before_close(environment)
            return candidate
        finally:
            self.journal.record(
                scope,
                "episode-timing",
                {"wall_seconds": time.monotonic() - episode_started},
                origin=EventOrigin.MODEL_TRANSPORT,
            )
            if environment is not None:
                try:
                    if lifecycle is not None:
                        await lifecycle.capture_before_close(environment)
                finally:
                    await environment.close()

    def validate_candidate(
        self, reader: CandidateReader, entry: PublicTaskView, arm: InferenceArm, run_id: str
    ) -> None:
        arm = self._resolved_arm(entry, arm)
        binding = self._binding(arm)
        policy = self.policy if binding is None else binding.policy
        final = reader.get(run_id, arm.arm_id, entry.task_id)
        validate_final_identity(
            final,
            expected_scope=(run_id, arm.arm_id, entry.task_id),
            expected_policy=policy.snapshot_id,
            expected_parser=CONTRACTS[entry.benchmark].parser,
            expected_message_id=f"{entry.task_id}:owner-final",
        )
        source = validate_persisted_owner_source(
            reader,
            final,
            benchmark=entry.benchmark,
            native_thinking=arm.native_thinking,
            adapter_name=None if binding is None else binding.adapter_name,
        )
        channels = split_native_channels(
            served_token_stream(reader.model_outputs(source.scope)),
            enabled=arm.native_thinking,
            tokenizer=self.tokenizer,
            finish_reason=source.result["finish_reason"],
        )
        if (
            channels.status != source.channel_status
            or channels.final_token_ids != source.final_token_ids
            or self.tokenizer.decode(channels.final_token_ids) != source.final_text
        ):
            raise ValueError("persisted final channel disagrees with the original served tokens")

    async def score(
        self, reader: CandidateReader, scope: tuple[str, str, str], benchmark: str
    ) -> NativeScore:
        if benchmark in OOD_BENCHMARKS:
            from .ood_scoring import score_ood

            return await score_ood(
                reader,
                scope,
                benchmark,
                self.source.targets.get(scope[2], {}),
                settings=self.config["scorers"],
                sandbox=self.sandbox,
                diagnostics=lambda value: self.journal.record(
                    scope, "native-ood-verdict", value, origin=EventOrigin.SCORER
                ),
            )
        settings = self.config["scorers"]
        grader = settings.get("healthbench", {})
        shared_grader = (
            benchmark == "healthbench"
            and grader.get("effective_profile", {}).get("backend") != "openai-chat-completions"
            and grader.get("endpoint_base") in self.config["endpoints"]
        )
        # Reuse only the already observed identical serving replicas. A separate
        # judge endpoint remains separate; this never routes grading to actors.
        with self.replicas.acquire() if shared_grader else nullcontext(None) as replica:
            if replica is not None:
                original = self.config["endpoints"].index(grader["endpoint_base"])
                observed = self.observed[replica]
                raw = observed["server_info"].get("server_args", observed["server_info"])
                if (
                    observed["model_info"] != self.observed[original]["model_info"]
                    or raw.get("served_model_name") != grader["model_route"]
                ):
                    raise ValueError("grader replicas must serve the unchanged judge model")
                endpoint = self.config["endpoints"][replica]
                settings = {**settings, "healthbench": {**grader, "endpoint_base": endpoint}}
                self.journal.record(
                    scope,
                    "scorer-routing",
                    {"replica": replica, "endpoint": endpoint},
                    origin=EventOrigin.SCORER,
                )
            try:
                async with self.mbpp_limiter if benchmark == "mbpp-plus" else nullcontext():
                    return await score_native(
                        reader,
                        scope,
                        benchmark,
                        self.source.targets.get(scope[2], {}),
                        settings=settings,
                        sandbox=self.sandbox,
                        mbpp_sandbox=self.mbpp_sandbox,
                        record_diagnostics=lambda value: self.journal.record(
                            scope,
                            "native-healthbench-verdict"
                            if benchmark == "healthbench"
                            else "native-mbpp-verdict",
                            value,
                            origin=EventOrigin.SCORER,
                        ),
                    )
            except IncompleteNativeGradingError as error:
                # Persist on the coordinator thread; no evaluator material or
                # failure-as-zero verdict is sent back to an actor.
                self.journal.record(
                    scope,
                    "scorer-failure",
                    {
                        "benchmark": benchmark,
                        "failure_type": type(error.__cause__).__name__,
                        "scorer_cost": error.scorer_cost,
                    },
                    origin=EventOrigin.SCORER,
                )
                raise

    async def close(self) -> None:
        for generator in self.generators:
            generator.close()
        for generators in self.trained_generators.values():
            for generator in generators:
                generator.close()
        self.journal.close()


def create_runtime(
    path: Path, *, private_output: Path
) -> tuple[PrivateIntegrityRuntime, FrozenPanel, tuple[InferenceArm, ...]]:
    config = json.loads(path.read_text())
    if config.get("catalog") == "ood":
        from .ood_sources import load_ood_panel

        source = load_ood_panel(config)
    else:
        source = load_source_panel(config)
        if config.get("evaluation_sample_counts"):
            source = select_source_panel(source, config["evaluation_sample_counts"])
    if "benchmark_order" in config:
        source = order_source_panel(source, tuple(config["benchmark_order"]))
    arms = tuple(decode_arm(value) for value in config["arms"])
    if len(arms) not in {1, 2}:
        raise ValueError("evaluation requires one selected arm or two paired arms")
    runtime = PrivateIntegrityRuntime(config, source, private_output)
    return runtime, source.panel, arms
