#!/usr/bin/env python3
"""Bounded non-benchmark real-backbone evidence for Gate 4a.

This gate deliberately does not try to force a particular rollout action and
does not search seeds.  The tiny-backbone Gate 4b owns the complete
detector-to-Phi mutation cycle; this real-backbone gate owns model, tokenizer,
gradient, checkpoint, hardware, and frozen-base authoring evidence.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Final, cast

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

import skillev
from skillev.contracts import (
    FailureMode,
    GenerateEvidence,
    HorizonBucket,
    JsonValue,
    PosteriorTaskFamilyMode,
    SplitBranch,
    SplitModalityEvidence,
    SplitTaskFamilyAssignment,
    TokenBucket,
    canonical_json,
    stable_hash,
)
from skillev.evolution import (
    RETAIN_AUTHORING_SAMPLING,
    AuthoringActionKind,
    AuthoringCallMaximum,
    AuthoringEdgeEvidence,
    AuthoringFailedError,
    BaseModelSkillAuthor,
    EvolutionConfig,
    GenerateAuthoringRequest,
    GenerateAuthoritySelection,
    RefineAuthoringRequest,
    RetainAuthoringRequest,
    SplitAuthoringRequest,
)
from skillev.experiments.build_identity import (
    require_source_archive_matches_execution,
    sha256_file,
)
from skillev.experiments.execution_hardware import ExecutionHardwareIdentity
from skillev.policy import (
    AdapterRole,
    AuthoringGenerationRequest,
    AuthoringTokenizerProtocol,
    BaseModelArtifactIdentity,
    GenerationResult,
    PolicyBackbone,
    PolicyGenerationRequest,
    QwenMultimodalBackboneConfig,
    QwenMultimodalPolicyBackbone,
    QwenTokenizerAdapter,
    qwen_backend_class,
    qwen_dtype_conversion_policy,
    qwen_tokenizer_artifact_identity,
)
from skillev.policy.config import DEFAULT_LORA_TARGET_MODULES, public_qwen_deployment_hash
from skillev.rollout import DecodedSegment, StructuredJsonActionCodec, decode_action_segment
from skillev.runtime import (
    BudgetLedger,
    LiveAttemptEventLog,
    ReservationState,
    RuntimeEventEmitter,
    SkillApplicability,
    SkillDocument,
    SkillManifest,
    SkillRequirement,
)

GATE_4A_FORMAT: Final = "skillev-real-9b-gate-4a@7"
GATE_4A_ATTESTATION_FORMAT: Final = "skillev-real-9b-gate-4a-attestation@7"
GATE_4A_PRIVATE_AUTHORING_FORMAT: Final = "skillev-gate-4a-private-authoring@1"
GATE_4A_SEED: Final = 20_260_730
GATE_4A_SYNTHETIC_REWARD_LOG_TERM: Final = 0.5


def admit_action_generation(
    tokenizer: AuthoringTokenizerProtocol,
    generated: GenerationResult,
) -> tuple[DecodedSegment, dict[str, JsonValue]]:
    """Bind Gate 4a evidence to the exact sampled content span."""

    segment = decode_action_segment(tokenizer, generated.content_token_ids)
    if segment.token_ids is not generated.content_token_ids:
        raise RuntimeError("Gate 4a replaced the sampled action token span")
    evidence: dict[str, JsonValue] = {
        "classification": classify_generated_action(segment.text),
        "content_hash": stable_hash(segment.text),
        "decoded_text_nonempty": bool(segment.text),
        "finish_reason": generated.finish_reason,
        "sampled_span_preserved": True,
        "token_count": len(segment.token_ids),
    }
    return segment, evidence


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--eos-token-id", type=int, required=True)
    parser.add_argument("--source-package", required=True)
    parser.add_argument("--work-directory", required=True)
    return parser.parse_args()


def _driver_version() -> str:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise FileNotFoundError("nvidia-smi")
    completed = subprocess.run(  # noqa: S603 -- resolved executable, fixed arguments
        [executable, "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    versions = {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    if len(versions) != 1:
        raise RuntimeError("Gate 4a requires one visible NVIDIA driver version")
    return versions.pop()


def _hardware_identity() -> ExecutionHardwareIdentity:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Gate 4a requires exactly one explicitly visible CUDA device")
    properties = torch.cuda.get_device_properties(0)
    cuda_version = torch.version.cuda
    cudnn_version = torch.backends.cudnn.version()  # type: ignore[no-untyped-call]
    nccl = getattr(torch.cuda, "nccl", None)
    nccl_version = nccl.version() if nccl is not None else None
    if not cuda_version or not cudnn_version or not nccl_version:
        raise RuntimeError("Gate 4a requires complete CUDA runtime identity")
    return ExecutionHardwareIdentity(
        accelerator_name=properties.name,
        compute_capability=f"{properties.major}.{properties.minor}",
        visible_device_count=1,
        nvidia_driver_version=_driver_version(),
        cuda_runtime_version=cuda_version,
        cudnn_version=str(cudnn_version),
        nccl_version=".".join(str(item) for item in nccl_version),
        kernel_release=platform.release(),
        safetensors_version=importlib.metadata.version("safetensors"),
    )


def _backbone_config(args: argparse.Namespace) -> QwenMultimodalBackboneConfig:
    model_path = Path(args.model_path)
    raw = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    text_config = raw.get("text_config")
    if not isinstance(text_config, dict) or type(text_config.get("hidden_size")) is not int:
        raise ValueError("pinned model config lacks text hidden_size")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        revision=args.revision,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise TypeError("Gate 4a requires the pinned fast tokenizer")
    tokenizer_identity = qwen_tokenizer_artifact_identity(
        tokenizer=tokenizer,
        tokenizer_id="Qwen/Qwen3.5-9B",
        revision=args.revision,
    )
    unbound = QwenMultimodalBackboneConfig(
        base_model_path=args.model_path,
        revision=args.revision,
        tokenizer_id=tokenizer_identity.tokenizer_id,
        tokenizer_content_hash=tokenizer_identity.content_hash,
        hidden_size=text_config["hidden_size"],
        device="cuda",
        torch_dtype="bfloat16",
        lora_rank=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_target_modules=DEFAULT_LORA_TARGET_MODULES,
        z_hidden_width=32,
        eos_token_ids=(args.eos_token_id,),
    )
    artifact = BaseModelArtifactIdentity.from_directory(
        directory=model_path,
        backend_class=qwen_backend_class(unbound),
        upstream_revision=args.revision,
        dtype_conversion_policy=qwen_dtype_conversion_policy(unbound),
    )
    return replace(unbound, base_model_artifact=artifact)


def classify_generated_action(text: str) -> str:
    parsed = StructuredJsonActionCodec().parse(text)
    if parsed.action is None:
        return parsed.status.value.replace("-", "_")
    return parsed.action.kind.value


def _source_skill(
    *,
    name: str,
    task_families: tuple[str, ...],
    context_id: str,
    title: str,
    summary: str,
    instructions: str,
    requirements: tuple[SkillRequirement, ...],
) -> SkillDocument:
    applicability = SkillApplicability(
        task_families=task_families,
        contexts=(context_id,),
        required_tools=("debug.tool",),
        excluded_contexts=(),
    )
    content = {
        "applicability": applicability.to_value(),
        "instructions": instructions,
        "requirements": [item.to_value() for item in requirements],
        "summary": summary,
        "title": title,
    }
    return SkillDocument(
        manifest=SkillManifest(
            skill_id=f"skill-gate-4a-{name}",
            version="1",
            content_hash=stable_hash(content),
            input_schema_id="gate-4a-input@1",
            output_schema_id="gate-4a-output@1",
            license_id="project-owned-debug",
            provenance_hash=stable_hash({"fixture": "gate-4a", "name": name}),
        ),
        title=title,
        summary=summary,
        instructions=instructions,
        applicability=applicability,
        requirements=requirements,
    )


def _edge(
    *,
    name: str,
    task_family: str,
    context_id: str,
) -> AuthoringEdgeEvidence:
    return AuthoringEdgeEvidence(
        edge_id=f"gate-4a-{name}:1",
        task_family=task_family,
        context_id=context_id,
        action_kind=AuthoringActionKind.TOOL,
        tool_or_skill_name="debug.tool",
        argument_schema_id=stable_hash({"query": "text"}),
        observation_status=FailureMode.SUCCESS,
        token_bucket=TokenBucket.LE_1K,
        horizon_bucket=HorizonBucket.LE_3,
        absolute_log_importance=1.0,
        log_importance_quantile=1.0,
        invoked_skill_ids=(),
        available_tools=("debug.tool",),
    )


def _mode(name: str, *, mean: float) -> PosteriorTaskFamilyMode:
    return PosteriorTaskFamilyMode(
        task_family=name,
        cell_keys=(f"cell-{name}",),
        posterior_event_ids=(f"posterior-{name}",),
        evidence_mass=8.0,
        mean=mean,
        sigma=0.05,
        lcb=mean - 0.05,
        ucb=mean + 0.05,
        within_cell_mean_span=0.01,
    )


def authoring_requests() -> tuple[
    RetainAuthoringRequest,
    RefineAuthoringRequest,
    SplitAuthoringRequest,
    GenerateAuthoringRequest,
]:
    retain_instructions = "\n\n".join(
        (
            (
                "Establish the public execution boundary before taking any action. Read the "
                "task family, public context, available-tool declaration, and the declared "
                "debug.tool argument schema. List the public facts that are actually present, "
                "distinguish them from assumptions, and reject any temptation to infer an "
                "answer, reward, verifier state, native payload, or evaluator-only field. "
                "Confirm that the requested operation can be expressed with the available "
                "public tool and that no private information is required."
            ),
            (
                "Prepare the tool request by inspecting every declared argument. Copy values "
                "only from public task material, preserve their documented types, and omit "
                "fields that the schema does not define. Build one canonical JSON object, check "
                "that required fields are present, and check that no convenience aliases, "
                "comments, guessed defaults, or hidden identifiers were introduced. Repeat the "
                "schema check immediately before execution so that a reasoning note cannot "
                "silently drift away from the action that is actually submitted."
            ),
            (
                "Invoke debug.tool exactly once for the current public operation. Keep the "
                "resource name, action kind, and argument object aligned with the declared "
                "interface. Do not issue a speculative probe, do not send the same request a "
                "second time after a valid response, and do not substitute a differently named "
                "resource. Treat malformed JSON or a schema-invalid action as the agent's own "
                "observable failure rather than hiding it with a retry. Preserve the emitted "
                "action and its public execution result as the next-step evidence."
            ),
            (
                "Classify the returned public observation before deciding what follows. A "
                "successful tool response may contribute facts to the final response; a public "
                "tool error may justify one different, schema-valid next action when the task "
                "still permits it; and a parse or schema failure must remain visible. Never "
                "reinterpret an infrastructure exception as a task failure, and never transform "
                "missing evaluator data into a zero reward. Record only the status and content "
                "that the model is allowed to observe."
            ),
            (
                "Carry forward the minimum public state needed for continuation: the canonical "
                "request, the observation status, useful returned fields, and whether the public "
                "goal is now satisfied. Keep this state separate from private evaluation state. "
                "Before another action, compare the proposed request with the already executed "
                "one so the procedure does not repeat work. If the public observation is enough "
                "to finish, stop using tools. If it is not enough, explain which public fact is "
                "missing and choose only an action that can obtain that fact."
            ),
            (
                "Complete the task with a concise response grounded in the public task and the "
                "public observation. State the useful result, distinguish an observed failure "
                "from a successful result, and avoid claims that the tool did not support. Do "
                "not include internal hashes, budget reservations, checkpoint identities, "
                "evaluator payloads, answer keys, or hidden scoring information. Completion is "
                "a terminal submission rather than a reusable tool edge, so it must not be "
                "treated as an uncovered skill opportunity."
            ),
            (
                "Apply the same safety boundary at every stage even when the surrounding text "
                "is verbose: public inputs determine the request, the declared schema determines "
                "its shape, the public observation determines the next step, and only public "
                "facts determine the final response. This restates the earlier constraints on "
                "purpose so that operators can audit a long-running workflow without consulting "
                "private state. It does not authorize additional calls, hidden lookups, retries, "
                "or evaluator access."
            ),
            (
                "For handoff, leave a compact public trace that another agent can follow: which "
                "schema was used, whether the single call was valid, which public status was "
                "observed, and whether completion followed. The handoff must not copy the full "
                "reasoning transcript or any private diagnostics. Recheck the four core rules—"
                "declared arguments only, one canonical call, public observation only, concise "
                "completion—and resolve any discrepancy by reporting it instead of fabricating "
                "a successful outcome."
            ),
        )
    )
    retain_source = _source_skill(
        name="retain-source",
        task_families=("gate-4a-retain",),
        context_id="gate-4a-retain",
        title="Reliable public debug-tool orchestration with explicit completion discipline",
        summary=(
            "A deliberately verbose, production-shaped procedure that repeats the public-input "
            "boundary, canonical debug-tool invocation, observation handling, handoff, and "
            "completion rules so a Retain action has genuine material to consolidate."
        ),
        instructions=retain_instructions,
        requirements=(
            SkillRequirement(
                requirement_id="gate-4a-retain-tool-call",
                text=(
                    "Validate the declared debug.tool schema, construct one canonical public "
                    "argument object, call the declared resource exactly once, and never hide "
                    "an invalid action by retrying or substituting a different resource."
                ),
            ),
            SkillRequirement(
                requirement_id="gate-4a-retain-public-observation",
                text=(
                    "Use only the public observation status and payload to decide the next "
                    "action, keeping evaluator state, rewards, answers, native payloads, and "
                    "other private information out of reasoning and completion."
                ),
            ),
            SkillRequirement(
                requirement_id="gate-4a-retain-no-repeat",
                text=(
                    "Preserve enough public state to prevent duplicate tool calls, distinguish "
                    "agent action failures from infrastructure failures, and stop invoking "
                    "tools once the public task can be completed."
                ),
            ),
            SkillRequirement(
                requirement_id="gate-4a-retain-completion",
                text=(
                    "Return a concise public completion grounded in observed facts, with no "
                    "internal identity, checkpoint, budget, evaluator, or answer-key material."
                ),
            ),
        ),
    )
    retain_edge = _edge(
        name="retain-edge",
        task_family="gate-4a-retain",
        context_id="gate-4a-retain",
    )
    refine_source = _source_skill(
        name="refine-source",
        task_families=("gate-4a-refine",),
        context_id="gate-4a-refine",
        title="Public debug tool procedure",
        summary="A reusable public tool-call procedure that needs one context patch.",
        instructions=(
            "Call debug.tool with its declared object arguments and use only the public "
            "observation in the next step."
        ),
        requirements=(
            SkillRequirement(
                requirement_id="gate-4a-refine-base",
                text="Preserve the canonical public debug-tool call.",
            ),
        ),
    )
    refine_edge = _edge(
        name="refine-edge",
        task_family="gate-4a-refine",
        context_id="gate-4a-refine",
    )
    split_source = _source_skill(
        name="split-source",
        task_families=("gate-4a-high", "gate-4a-low"),
        context_id="gate-4a-split",
        title="Two-mode public debug tool procedure",
        summary="A shared procedure whose two public task families need separate variants.",
        instructions=(
            "Use debug.tool once, inspect the public response, and follow the task-family "
            "specific procedure without using hidden evaluator information."
        ),
        requirements=(
            SkillRequirement(
                requirement_id="gate-4a-split-base",
                text="Keep the shared canonical public debug-tool behavior.",
            ),
        ),
    )
    split_low_edge = _edge(
        name="split-low-edge",
        task_family="gate-4a-low",
        context_id="gate-4a-split",
    )
    split_high_edge = _edge(
        name="split-high-edge",
        task_family="gate-4a-high",
        context_id="gate-4a-split",
    )
    generate_edge = _edge(
        name="generate-edge",
        task_family="gate-4a-generate",
        context_id="gate-4a-generate",
    )
    low_mode = _mode("gate-4a-low", mean=0.2)
    high_mode = _mode("gate-4a-high", mean=0.8)
    modality = SplitModalityEvidence(
        low_mode=low_mode,
        high_mode=high_mode,
        between_mean_gap=0.6,
        intervals_disjoint=True,
        source_task_families=("gate-4a-high", "gate-4a-low"),
        assignments=(
            SplitTaskFamilyAssignment(high_mode, SplitBranch.HIGH),
            SplitTaskFamilyAssignment(low_mode, SplitBranch.LOW),
        ),
        separation_cutpoint=0.5,
    )
    return (
        RetainAuthoringRequest(
            retain_source,
            (retain_edge,),
            "high public flow and high calibrated reliability",
            GATE_4A_SEED + 1,
        ),
        RefineAuthoringRequest(
            refine_source,
            (refine_edge,),
            "low public context posterior",
            ("gate-4a-target-cell",),
            GATE_4A_SEED + 2,
        ),
        SplitAuthoringRequest(
            split_source,
            (split_low_edge, split_high_edge),
            "two separated public task-family modes",
            modality,
            GATE_4A_SEED + 3,
        ),
        GenerateAuthoringRequest(
            (generate_edge,),
            "uncovered high-asymmetry public tool edge",
            GenerateAuthoritySelection(
                task_family=generate_edge.task_family,
                context=generate_edge.context_id,
                required_tools=generate_edge.available_tools,
                input_schema_id="gate-4a-input@1",
                output_schema_id="gate-4a-output@1",
                license_id="project-owned-debug",
            ),
            GenerateEvidence((generate_edge.edge_id,), 0.1, 0.9, "absolute-log-density-ratio@1"),
            GATE_4A_SEED + 4,
        ),
    )


class RecordingAuthoringBackbone:
    """Record each real frozen-base call without changing its result."""

    def __init__(self, delegate: QwenMultimodalPolicyBackbone) -> None:
        self._delegate = delegate
        self.requests: list[AuthoringGenerationRequest] = []
        self.results: list[GenerationResult] = []

    @property
    def tokenizer(self) -> AuthoringTokenizerProtocol:
        return self._delegate.tokenizer

    def generate_base(self, request: AuthoringGenerationRequest) -> GenerationResult:
        self.requests.append(request)
        result = self._delegate.generate_base(request)
        self.results.append(result)
        return result


def _failure_chain(error: BaseException) -> list[JsonValue]:
    output: list[JsonValue] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        output.append(
            {
                "message": str(current),
                "type": type(current).__name__,
            }
        )
        current = current.__cause__
    return output


def _generation_evidence(
    *,
    tokenizer: AuthoringTokenizerProtocol,
    result: GenerationResult,
) -> dict[str, JsonValue]:
    decoded = tokenizer.decode(result.content_token_ids)
    return {
        "content_token_count": len(result.content_token_ids),
        "content_token_ids_hash": stable_hash(list(result.content_token_ids)),
        "decoded_output_hash": stable_hash(decoded),
        "finish_reason": result.finish_reason,
        "stop_token_count": len(result.stop_token_ids),
        "stop_token_ids_hash": stable_hash(list(result.stop_token_ids)),
    }


def _write_private_authoring(
    path: Path,
    records: list[JsonValue],
) -> None:
    payload: dict[str, JsonValue] = {
        "format": GATE_4A_PRIVATE_AUTHORING_FORMAT,
        "records": records,
    }
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    path.chmod(0o600)


def _require_authoring_accounting(
    *,
    ledger: BudgetLedger,
    recording: RecordingAuthoringBackbone,
    reservation_ids_by_action: dict[str, str],
) -> None:
    ledger.assert_fully_settled()
    entries = ledger.entries
    if (
        ledger.settled.model_calls != 4
        or len(recording.requests) != 4
        or len(recording.results) != 4
        or len(entries) != 4
        or any(entry.state is not ReservationState.SETTLED for entry in entries)
    ):
        raise RuntimeError("Gate 4a requires exactly four settled frozen-base calls")
    reservation_ids = tuple(reservation_ids_by_action.values())
    if len(reservation_ids) != 4 or len(set(reservation_ids)) != 4:
        raise RuntimeError("Gate 4a requires one unique reservation per action type")


def _gradient_summary(groups: object) -> dict[str, JsonValue]:
    output: dict[str, JsonValue] = {}
    for name in ("forward", "backward", "z_head"):
        parameters = getattr(groups, name)
        gradients = tuple(parameter.grad for parameter in parameters if parameter.grad is not None)
        finite = bool(gradients) and all(bool(torch.isfinite(item).all()) for item in gradients)
        nonzero_count = sum(bool(torch.count_nonzero(item)) for item in gradients)
        if not finite or not nonzero_count:
            raise RuntimeError(
                f"Gate 4a {name} gradients failed: parameters={len(parameters)}, "
                f"present={len(gradients)}, nonzero={nonzero_count}, finite={finite}"
            )
        output[name] = {
            "parameter_count": len(parameters),
            "gradient_tensor_count": len(gradients),
            "nonzero_gradient_tensor_count": nonzero_count,
        }
    return output


def _checkpoint_roundtrip(
    backbone: QwenMultimodalPolicyBackbone,
    checkpoint: Path,
) -> str:
    groups = backbone.parameter_groups()
    parameters = (*groups.forward, *groups.backward, *groups.z_head)
    expected = tuple(parameter.detach().cpu().clone() for parameter in parameters)
    identity = backbone.trainable_state_identity
    backbone.save_checkpoint(str(checkpoint))
    with torch.no_grad():
        parameters[0].view(-1)[0].add_(1.0)
    backbone.load_checkpoint(str(checkpoint))
    restored = (
        *backbone.parameter_groups().forward,
        *backbone.parameter_groups().backward,
        *backbone.parameter_groups().z_head,
    )
    if backbone.trainable_state_identity != identity or any(
        not torch.equal(before, after.detach().cpu())
        for before, after in zip(expected, restored, strict=True)
    ):
        raise RuntimeError("Gate 4a checkpoint save/load is not bit-identical")
    return identity.content_hash


def main() -> None:
    args = _args()
    started = time.monotonic()
    work = Path(args.work_directory)
    work.mkdir(parents=True, exist_ok=False)
    source_package = Path(args.source_package)
    if not source_package.is_file():
        raise FileNotFoundError(source_package)
    executing_root = Path(__file__).resolve().parents[1]
    imported_package_root = Path(skillev.__file__).resolve().parent
    if imported_package_root != (executing_root / "src" / "skillev").resolve():
        raise RuntimeError("Gate 4a imported SKILLEV from another source tree")
    source_provenance = require_source_archive_matches_execution(
        source_archive=source_package,
        executing_root=executing_root,
        temporary_parent=work,
    )
    source_package_sha256 = sha256_file(source_package)
    hardware = _hardware_identity()
    config = _backbone_config(args)
    backbone = QwenMultimodalPolicyBackbone(config)
    tokenizer = cast(QwenTokenizerAdapter, backbone.tokenizer)
    initial_trainable_state = backbone.trainable_state_identity

    generation_prompt = (
        "Return one public structured action JSON object for a synthetic debug task. "
        "Available tool: debug.tool. Do not use Markdown.\nAction:\n"
    )
    prompt_ids = tuple(tokenizer.encode(generation_prompt))
    generated = backbone.generate_policy(
        PolicyGenerationRequest(
            input_ids=prompt_ids,
            max_new_tokens=96,
            seed=GATE_4A_SEED,
            decoding_snapshot_id=stable_hash({"gate": "4a", "seed": GATE_4A_SEED}),
        )
    )
    if not generated.content_token_ids:
        raise RuntimeError("Gate 4a real generation returned no content")
    segment, action_generation = admit_action_generation(tokenizer, generated)

    backward_prompt = "Public observation: the synthetic debug action was recorded.\nAction:\n"
    backward_ids = tuple(tokenizer.encode(backward_prompt))
    query_ids = tuple(tokenizer.encode("Synthetic public Gate 4a query."))
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone.parameter_groups().forward, "lr": 1e-6},
            {"params": backbone.parameter_groups().backward, "lr": 1e-6},
            {"params": backbone.parameter_groups().z_head, "lr": 1e-6},
        ],
        weight_decay=0.0,
    )
    optimizer.zero_grad(set_to_none=True)
    # Match production score_trajectory ordering: adapter-free Z first, then
    # forward and backward policy graphs.  Calling z_value after the policy
    # graphs would restore only the last active PEFT adapter on context exit
    # and freeze the earlier graph's adapter before backward.
    z_value = backbone.z_value(query_ids)
    forward = backbone.score(prompt_ids, generated.content_token_ids, AdapterRole.FORWARD_POLICY)
    backward = backbone.score(
        backward_ids, generated.content_token_ids, AdapterRole.BACKWARD_POLICY
    )
    if not all(bool(torch.isfinite(value).all()) for value in (forward, backward, z_value)):
        raise RuntimeError("Gate 4a teacher-forced scores are non-finite")
    # The frozen base and both LoRA adapters start from the same policy while
    # the Z head starts at zero.  A reward-free residual would therefore be
    # identically zero before the first update and could not exercise any of
    # the three gradient paths.  Use a fixed synthetic beta*log(R-tilde) term,
    # as the real TTB objective does, without depending on rollout content.
    loss = (z_value + forward.mean() - GATE_4A_SYNTHETIC_REWARD_LOG_TERM - backward.mean()).square()
    if not bool(torch.isfinite(loss).all()):
        raise RuntimeError("Gate 4a joint loss is non-finite")
    loss.backward()  # type: ignore[no-untyped-call]
    gradient_summary = _gradient_summary(backbone.parameter_groups())
    optimizer.step()
    backbone.mark_policy_update(1)
    checkpoint_identity = _checkpoint_roundtrip(backbone, work / "checkpoint")

    authoring_config = EvolutionConfig(generate_min_absolute_log_importance=0.1)
    maximum = AuthoringCallMaximum(
        input_tokens=authoring_config.max_authoring_prompt_tokens,
        output_tokens=authoring_config.max_authoring_completion_tokens,
    )
    ledger = BudgetLedger(
        run_id="gate-4a",
        attempt_id="real-9b",
        cap=maximum.to_budget_vector().scale(4),
    )
    recording_backbone = RecordingAuthoringBackbone(backbone)
    author = BaseModelSkillAuthor(
        backbone=cast(PolicyBackbone, recording_backbone),
        config=authoring_config,
        sampling=RETAIN_AUTHORING_SAMPLING,
        ledger=ledger,
        emitter=RuntimeEventEmitter(
            LiveAttemptEventLog(
                work / "authoring-events.jsonl",
                run_id=ledger.run_id,
                attempt_id=ledger.attempt_id,
            ),
            producer_id="gate-4a-author",
        ),
        maximum=maximum,
    )
    authoring: dict[str, JsonValue] = {}
    private_authoring: list[JsonValue] = []
    reservation_ids_by_action: dict[str, str] = {}
    private_authoring_path = work / "gate-4a-private-authoring.json"
    for request in authoring_requests():
        action_type = type(request).__name__.removesuffix("AuthoringRequest").lower()
        calls_before = len(recording_backbone.requests)
        results_before = len(recording_backbone.results)
        entry_ids_before = {entry.reservation.reservation_id for entry in ledger.entries}
        try:
            authored = author.author(request)
        except AuthoringFailedError as error:
            # Gate 4a records the fixed-seed frozen-base authoring outcome; it
            # never retries or turns a rejected draft into a library mutation.
            authoring[action_type] = {"classification": "rejected"}
            calls_after = len(recording_backbone.requests)
            results_after = len(recording_backbone.results)
            entries_after = ledger.entries
            new_entries = tuple(
                entry
                for entry in entries_after
                if entry.reservation.reservation_id not in entry_ids_before
            )
            stage = (
                "pre-generation" if calls_after == calls_before else "post-generation-validation"
            )
            generation = (
                _generation_evidence(
                    tokenizer=recording_backbone.tokenizer,
                    result=recording_backbone.results[-1],
                )
                if results_after == results_before + 1
                else None
            )
            private_authoring.append(
                {
                    "action_type": action_type,
                    "classification": "rejected",
                    "failure_chain": _failure_chain(error),
                    "generation": generation,
                    "reservation_id": (
                        new_entries[0].reservation.reservation_id if len(new_entries) == 1 else None
                    ),
                    "stage": stage,
                }
            )
            _write_private_authoring(private_authoring_path, private_authoring)
            if stage == "pre-generation":
                raise RuntimeError(
                    f"Gate 4a {action_type} fixture failed before generate_base"
                ) from error
        else:
            authoring[action_type] = {
                "classification": "validated",
                "draft_count": len(authored.drafts),
                "result_hash": stable_hash(authored.to_value()),
            }
            calls_after = len(recording_backbone.requests)
            results_after = len(recording_backbone.results)
            new_entries = tuple(
                entry
                for entry in ledger.entries
                if entry.reservation.reservation_id not in entry_ids_before
            )
            if calls_after != calls_before + 1 or results_after != results_before + 1:
                raise RuntimeError(f"Gate 4a {action_type} did not execute one generate_base call")
            if len(new_entries) != 1 or new_entries[0].state is not ReservationState.SETTLED:
                raise RuntimeError(f"Gate 4a {action_type} did not settle one reservation")
            reservation_id = new_entries[0].reservation.reservation_id
            reservation_ids_by_action[action_type] = reservation_id
            private_authoring.append(
                {
                    "action_type": action_type,
                    "classification": "validated",
                    "failure_chain": [],
                    "generation": _generation_evidence(
                        tokenizer=recording_backbone.tokenizer,
                        result=recording_backbone.results[-1],
                    ),
                    "reservation_id": reservation_id,
                    "stage": "post-generation-validation",
                }
            )
            _write_private_authoring(private_authoring_path, private_authoring)
            continue
        calls_after = len(recording_backbone.requests)
        results_after = len(recording_backbone.results)
        new_entries = tuple(
            entry
            for entry in ledger.entries
            if entry.reservation.reservation_id not in entry_ids_before
        )
        if calls_after != calls_before + 1 or results_after != results_before + 1:
            raise RuntimeError(
                f"Gate 4a {action_type} rejection did not follow one completed generation"
            )
        if len(new_entries) != 1 or new_entries[0].state is not ReservationState.SETTLED:
            raise RuntimeError(f"Gate 4a {action_type} rejection did not settle one reservation")
        reservation_ids_by_action[action_type] = new_entries[0].reservation.reservation_id
    _require_authoring_accounting(
        ledger=ledger,
        recording=recording_backbone,
        reservation_ids_by_action=reservation_ids_by_action,
    )
    _write_private_authoring(private_authoring_path, private_authoring)

    peak_memory = int(torch.cuda.max_memory_allocated())
    runtime_seconds = time.monotonic() - started
    if not math.isfinite(runtime_seconds):
        raise RuntimeError("Gate 4a runtime is non-finite")
    deployment_hash = public_qwen_deployment_hash(config, backend_kind="qwen-multimodal")
    attestation_value: dict[str, JsonValue] = {
        "backbone_deployment_hash": deployment_hash,
        "final_trainable_state_hash": backbone.trainable_state_identity.content_hash,
        "base_model_artifact_hash": cast(
            BaseModelArtifactIdentity,
            config.base_model_artifact,
        ).content_hash,
        "format": GATE_4A_ATTESTATION_FORMAT,
        "hardware_identity_hash": hardware.content_hash,
        "initial_trainable_state_hash": initial_trainable_state.content_hash,
        "source_package_sha256": source_package_sha256,
        "source_commit": source_provenance.source_commit,
        "source_tree_hash": source_provenance.source_tree_hash,
        "tokenizer_identity_hash": tokenizer.public_identity.content_hash,
    }
    gate_result: dict[str, JsonValue] = {
        "action_generation": action_generation,
        "authoring": authoring,
        "authoring_model_calls": ledger.settled.model_calls,
        "attestation": {
            "content_hash": stable_hash(attestation_value),
            "value": attestation_value,
        },
        "checkpoint_trainable_state_hash": checkpoint_identity,
        "format": GATE_4A_FORMAT,
        "gate_passed": all(
            isinstance(value, dict) and value.get("classification") == "validated"
            for value in authoring.values()
        ),
        "gradients": gradient_summary,
        "hardware": hardware.to_value(),
        "identity": {
            "base_model_artifact": cast(
                BaseModelArtifactIdentity,
                config.base_model_artifact,
            ).to_value(),
            "backbone_deployment_hash": deployment_hash,
            "source_package_sha256": source_package_sha256,
            "source_provenance": source_provenance.to_value(),
            "tokenizer": tokenizer.public_identity.to_value(),
            "trainable_state": backbone.trainable_state_identity.to_value(),
        },
        "joint_loss": float(loss.detach().item()),
        "optimizer_steps": 1,
        "peak_cuda_memory_bytes": peak_memory,
        "runtime_seconds": runtime_seconds,
        "seed": GATE_4A_SEED,
        "synthetic_reward_log_term": GATE_4A_SYNTHETIC_REWARD_LOG_TERM,
    }
    output = work / "gate-4a-result.json"
    output.write_text(canonical_json(gate_result) + "\n", encoding="utf-8")
    print(canonical_json(gate_result))
    if not gate_result["gate_passed"]:
        raise RuntimeError("Gate 4a requires all four structured authoring results to validate")


if __name__ == "__main__":
    main()
