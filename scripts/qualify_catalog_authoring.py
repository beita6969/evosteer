"""Controlled non-IID Generate -> mutation -> live catalog/read qualification.

No optimizer, posterior, benchmark target or natural Phi detector is constructed.
Modes run in separate processes. Resume forbids author HTTP and reuses its exact
journal response; task-policy calls are real unless injected by a CPU test.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from skillev.contracts import (
    EntropyObservation,
    FailureMode,
    GenerateEvidence,
    HorizonBucket,
    PhaseTransitionEvent,
    PhaseTriggerRule,
    ScientificSamplingCoordinate,
    SuccessRule,
    TerminalReward,
    TokenBucket,
    WindowStats,
)
from skillev.evolution import (
    AuthoringActionKind,
    AuthoringCallMaximum,
    AuthoringEdgeEvidence,
    AuthoringSamplingConfig,
    EvolutionConfig,
    EvolutionDecision,
    GenerateProposal,
    SkillAuthoringAuthority,
    TaskConditionedSkillRetriever,
)
from skillev.evolution.authoring import authoring_reservation_id, render_authoring_prompt
from skillev.evolution.execution import build_evolution_mutation
from skillev.evolution.external_sglang_authoring import ExternalSGLangSkillAuthor
from skillev.policy import QwenMultimodalBackboneConfig, QwenTokenizerAdapter
from skillev.rollout import PolicySnapshot, RolloutSessionBundle, RolloutTask
from skillev.rollout.action_surface import (
    ACTION_SURFACE_FORMAT_V3,
    ActionSurface,
    CompletionSpec,
    PublicActionInstructions,
    TerminalMode,
)
from skillev.rollout.environment import SubmittedTerminalValue
from skillev.rollout.episode_executor import episode_decoding, execute_episode
from skillev.rollout.external_sglang import (
    ExternalSGLangRolloutConfig,
    ExternalSGLangRolloutGenerator,
)
from skillev.rollout.prompt_profiles import InitialContextProfile
from skillev.runtime import (
    BudgetLedger,
    LiveAttemptEventLog,
    RuntimeEventEmitter,
    SkillLibrary,
    SkillLibraryState,
)
from skillev.runtime.request_journal import DurableRequestJournal
from skillev.runtime.sglang_gateway import (
    SGLangGateway,
    SGLangGatewayConfig,
    UrllibSGLangControlTransport,
)
from skillev.training.config import conservative_rollout_maximum
from skillev.training.inflight import durable_json
from skillev.training.rollout_workflow import RolloutWorkflowBinding, RolloutWorkflowResources

PURPOSE = "controlled-non-IID-catalog-authoring@1"
FAMILY, CONTEXT = "development/catalog-control", "public-catalog-integration"


def read(path):
    return json.loads(Path(path).read_text())


def preserve(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError("cannot replace saved development evidence")
    else:
        durable_json(path, value)


def clock():
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def declared_control(library):
    # These typed numbers solely drive an explicitly scripted integration
    # proposal. They are NOT measured training residuals/entropy/posteriors.
    phase = PhaseTransitionEvent(
        event_id="controlled-catalog-phase-not-measured",
        triggered_at_step=2,
        library_version=library.current_version,
        previous_window=WindowStats(1, 1, 1, 1.0, ("control-before",)),
        current_window=WindowStats(2, 2, 1, 0.99, ("control-after",)),
        relative_improvement=0.01,
        rho=0.05,
        residual_condition_met=True,
        entropy_series=(EntropyObservation(1, 0.5, 2, 2), EntropyObservation(2, 0.0, 1, 1)),
        trigger_rule=PhaseTriggerRule.RESIDUAL_AND_ENTROPY,
        required_consecutive_drops=1,
        entropy_condition_met=True,
        triggered=True,
    )
    edge = AuthoringEdgeEvidence(
        edge_id="controlled-public-example:1",
        task_family=FAMILY,
        context_id=CONTEXT,
        action_kind=AuthoringActionKind.COMPLETE,
        tool_or_skill_name="complete",
        argument_schema_id="public-development@1",
        observation_status=FailureMode.SUCCESS,
        token_bucket=TokenBucket.LE_1K,
        horizon_bucket=HorizonBucket.LE_3,
        absolute_log_importance=1.0,
        log_importance_quantile=1.0,
        invoked_skill_ids=(),
        available_tools=(),
    )
    proposal = GenerateProposal(
        evidence=GenerateEvidence((edge.edge_id,), 0.1, 0.9, "absolute-log-density-ratio@1"),
        rationale_text=(
            "CONTROLLED DEVELOPMENT INTEGRATION, not natural Phi or measured high importance. "
            "Create a reusable public procedure for reading available documentation, separating "
            "observations from unverified claims, and reporting an honest short status. "
            "There is no benchmark question, private target, or task answer to solve."
        ),
        edge_exemplars=(edge,),
    )
    authority = SkillAuthoringAuthority(
        "public-development-input@1",
        "public-status@1",
        "synthetic-public-development",
        (FAMILY,),
        (),
    )
    return phase, EvolutionDecision(phase.event_id, (proposal,)), authority


class CountedTransport:
    def __init__(self, root, role, delegate=None, *, forbid=False):
        self.root, self.role = root, role
        self.delegate = delegate or UrllibSGLangControlTransport()
        self.forbid, self.calls = forbid, 0

    def request(self, **kwargs):
        if self.forbid:
            raise RuntimeError("resume must not dispatch a new author request")
        self.calls += 1
        started = time.perf_counter()
        result = None
        try:
            result = self.delegate.request(**kwargs)
            return result
        finally:
            preserve(
                self.root / f"{self.role}-http-{self.calls:03d}.json",
                {
                    "role": self.role,
                    "elapsed_seconds": time.perf_counter() - started,
                    "received_response": result is not None,
                    "request": kwargs,
                    "response": result,
                },
            )


class RecordedAuthor:
    def __init__(self, author, root):
        self.delegate, self.root, self.tokenizer = author, root, author.tokenizer

    def author(self, request):
        preserve(
            self.root / "author-request.json",
            {
                "purpose": PURPOSE,
                "reservation_id": authoring_reservation_id(request),
                "journal_identity": [
                    "author",
                    self.delegate.ledger.run_id,
                    authoring_reservation_id(request),
                ],
                "prompt": render_authoring_prompt(request, tokenizer=self.tokenizer),
            },
        )
        result = self.delegate.author(request)
        preserve(self.root / "accepted-author-result.json", result.to_value())
        return result


def runtime(root, maximum, mode):
    ledger = BudgetLedger(run_id=root.name, attempt_id=mode, cap=maximum)
    events = LiveAttemptEventLog(root / f"{mode}-events.jsonl", run_id=root.name, attempt_id=mode)
    return ledger, RuntimeEventEmitter(events, mode)


def recover_mutation(root, plan, tokenizer, *, resumed=False, transport=None):
    before = SkillLibrary(SkillLibraryState.from_value(read(root / "initial-library.json")))
    phase, decision, authority = declared_control(before)
    preserve(
        root / "controlled-proposal.json",
        {
            "purpose": PURPOSE,
            "measured_training_evidence": False,
            "phase": phase.to_value(),
            "rationale": decision.proposals[0].rationale_text,
        },
    )
    config = EvolutionConfig.from_value(plan["evolution"])
    maximum = AuthoringCallMaximum(
        config.max_authoring_prompt_tokens, config.max_authoring_completion_tokens
    )
    ledger, emitter = runtime(
        root, maximum.to_budget_vector(), "resume-author" if resumed else "author"
    )
    counted = CountedTransport(root, "author", transport, forbid=resumed)
    journal = DurableRequestJournal(root / "author-requests.sqlite3")
    journal.require_resolved_prefix(("author", root.name))
    author = ExternalSGLangSkillAuthor(
        tokenizer=tokenizer,
        config=config,
        sampling=AuthoringSamplingConfig.from_value(plan["authoring_sampling"]),
        ledger=ledger,
        emitter=emitter,
        maximum=maximum,
        rollout_config=ExternalSGLangRolloutConfig(plan["endpoint"]),
        gateway=SGLangGateway(
            SGLangGatewayConfig(
                endpoint_base=plan["endpoint"],
                base_model=plan["base_model"],
                supervisor_adapter="unused-frozen-base-author",
                control_retries=0,
            )
        ),
        transport=counted,
        request_journal=journal,
    )
    mutation = build_evolution_mutation(
        decision,
        author=RecordedAuthor(author, root),
        library=before,
        phase_event=phase,
        config=config,
        authority=authority,
        base_seed=0,
        cycle_ordinal=1,
    )
    ledger.assert_fully_settled()
    before.apply(before.preview(mutation))
    preserve(root / "mutated-library.json", before.state.to_value())
    preserve(
        root / "mutation.json",
        {
            "library_version_before": mutation.library_version_before,
            "library_version_after": mutation.library_version_after,
            "actions": [action.to_value() for action in mutation.actions],
            "new_skill_ids": [doc.manifest.skill_id for doc in mutation.new_documents],
        },
    )
    summary = {
        "mode": "resume-author" if resumed else "author",
        "driver_pid": os.getpid(),
        "physical_author_calls": counted.calls,
        "logical_recovered_usage": ledger.settled.to_value(),
        "library_version": before.current_version,
        "training_posterior_updates": 0,
    }
    summary_name = "resume-author-summary.json" if resumed else "author-summary.json"
    if resumed and (root / summary_name).exists():
        summary_name = f"resume-author-{os.getpid()}-summary.json"
    preserve(root / summary_name, summary)
    return before, mutation.new_documents[0].manifest.skill_id


def development_task():
    return RolloutTask(
        task_id="non-iid-public-catalog-development",
        environment_id="controlled-public-environment@1",
        task_family=FAMILY,
        context_id=CONTEXT,
        available_tools=(),
        query=(
            "This is a controlled documentation integration task, not a benchmark. "
            "Inspect the visible catalog and read its applicable newly available procedure. "
            "Then give a short status distinguishing what you actually read from any unverified "
            "usefulness claim. Do not invent a task result. No external facts are needed."
        ),
        public_context={"purpose": PURPOSE},
        action_surface=ActionSurface(
            format=ACTION_SURFACE_FORMAT_V3,
            terminal_mode=TerminalMode.EXPLICIT_COMPLETION,
            completion=CompletionSpec({"answer": "string"}, {"answer": "public status"}),
            instructions=PublicActionInstructions(
                semantic=("Submit a brief public integration status.",)
            ),
        ),
    )


class DevelopmentSession:
    def __init__(self, task, skills):
        self.environment_id, self.task_family = task.environment_id, task.task_family
        self.skills = skills

    async def execute(self, action, *, step_index):
        raise ValueError("synthetic environment exposes no tool other than the catalog wrapper")

    def validate_completion(self, submission):
        return isinstance(submission, dict) and isinstance(submission.get("answer"), str)

    async def evaluate(self, request):
        submitted = isinstance(request.evaluation_input, SubmittedTerminalValue)
        return TerminalReward(
            value=float(submitted),
            success=submitted,
            success_rule=SuccessRule.R_AT_THRESHOLD,
            success_threshold=0.5,
            native_metric_name="controlled-submission-presence-only",
            native_payload={"purpose": PURPOSE, "not_task_correctness_or_skill_usefulness": True},
            environment_id=self.environment_id,
            verifier_version=PURPOSE,
        )


class DevelopmentSessions:
    def __init__(self, library):
        self.retriever = TaskConditionedSkillRetriever(library=library)

    def create(self, task):
        skills = self.retriever.retrieve(task)
        session = DevelopmentSession(task, skills)
        return RolloutSessionBundle(session, session, skills)


class RecordedGenerator:
    def __init__(self, generator, root):
        self.generator, self.root, self.calls = generator, root, []
        self.tokenizer = generator.tokenizer

    def snapshot(self):
        return self.generator.snapshot()

    def begin_episode(self, *args):
        self.generator.begin_episode(*args)

    def end_episode(self, *args):
        self.generator.end_episode(*args)

    async def generate(self, request):
        index = len(self.calls) + 1
        self.calls.append(request)
        preserve(
            self.root / f"actor-input-{index:03d}.json",
            {
                "phase": request.phase.value,
                "episode_id": request.episode_id,
                "input_ids": list(request.input_ids),
                "seed": request.seed,
                "max_new_tokens": request.max_new_tokens,
                "policy_snapshot_id": request.expected_policy_snapshot_id,
                "journal_identity": [
                    request.episode_id,
                    str(request.turn_index),
                    request.phase.value,
                    request.expected_policy_snapshot_id,
                    request.library_version,
                    request.decoding_snapshot_id,
                ],
            },
        )
        started = time.perf_counter()
        result = await self.generator.generate(request)
        preserve(
            self.root / f"actor-output-{index:03d}.json",
            {
                "elapsed_seconds": time.perf_counter() - started,
                "content_token_ids": list(result.content_token_ids),
                "stop_token_ids": list(result.stop_token_ids),
                "finish_reason": result.finish_reason,
                "usage": result.usage.to_value(),
            },
        )
        return result


async def read_episode(root, plan, library, skill_id, generator):
    from skillev.training.config import PolicyRolloutConfig

    rollout = PolicyRolloutConfig.from_value(plan["rollout"])
    task = development_task()
    ledger, emitter = runtime(root, rollout.per_rollout_maximum, "read")
    recorded = RecordedGenerator(generator, root)
    artifact = await execute_episode(
        generator=recorded,
        sessions=DevelopmentSessions(library),
        assembler=rollout.context_assembler(maximum_h0_tokens=plan["max_input_tokens"]),
        rollout=rollout,
        task=task,
        trajectory_id=root.name + "-read",
        library=library.state,
        coordinate=ScientificSamplingCoordinate(
            sampling_schedule_hash=PURPOSE,
            schedule_purpose="controlled-development",
            ordered_sequence_hash="one-public-documentation-task",
            sequence_position=0,
            task_id=task.task_id,
            optimizer_step_or_anchor_ordinal=0,
        ),
        decoding=episode_decoding(rollout, task),
        epsilon_min=0.1,
        condition_id=PURPOSE,
        initial_context_profile=InitialContextProfile.TRAINED_SKILLEV,
        ledger=ledger,
        emitter=emitter,
        clock=clock,
        resources=RolloutWorkflowResources(RolloutWorkflowBinding()),
    )
    preserve(root / "read-artifact.json", artifact.to_value())
    ledger.assert_fully_settled()
    evidence = artifact.skill_input_evidence or ()
    returned = [
        s.index
        for s in artifact.record.steps
        if skill_id in s.invoked_skill_ids
        and json.loads(s.observation_text).get("status") == "skill-read"
    ]
    visible = [
        row
        for row in evidence
        if any(ref["skill_id"] == skill_id for ref in (row.get("visible_skill_body_refs") or []))
    ]
    # The engine sidecar is computed from the exact admitted token sequence;
    # it is not a check against an untruncated rendered prompt.
    after = [s for s in artifact.record.steps if returned and s.index > min(returned)]
    catalog_before_read = any(
        returned
        and r["step_index"] <= min(returned)
        and skill_id in (r.get("catalog_visible_skill_ids") or [])
        for r in evidence
    )
    result = {
        "purpose": PURPOSE,
        "driver_pid": os.getpid(),
        "library_version": library.current_version,
        "new_skill_id": skill_id,
        "returned_read_step_indices": returned,
        "body_visible_actual_input_evidence": visible,
        "catalog_visible": any(
            skill_id in (r.get("catalog_visible_skill_ids") or []) for r in evidence
        ),
        "catalog_visible_before_read": catalog_before_read,
        "followed_actions_basis": "chronological after read; not causal instruction following",
        "followed_actions": [
            {
                "index": s.index,
                "action_text": s.action_text,
                "observation_status": s.observation_status,
            }
            for s in after
        ],
        "followed_instructions_or_usefulness": None,
        "actor_logical_calls": len(recorded.calls),
        "actor_usage": ledger.settled.to_value(),
        "training_updates": 0,
        "posterior_updates": 0,
        "natural_phi_trigger": False,
        "autonomous_unprompted_read": False,
        "qualified_catalog_read_visibility": bool(
            catalog_before_read and returned and visible and after
        ),
    }
    preserve(root / "read-result.json", result)
    if not result["qualified_catalog_read_visibility"]:
        raise RuntimeError("actual read/body visibility chain not demonstrated; do not resample")
    return result


def initialize(root, plan, initial_library):
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    preserve(root / "plan.json", plan)
    preserve(root / "initial-library.json", initial_library.to_value())


def resume_before_dispatch(root):
    """Recover a failed CLI preflight, never a sampled or ambiguous episode.

    The first real qualification failed in generator construction, after writing
    read-started but before any actor request. Preserve that failed attempt;
    only an empty request history with a stopped original driver can continue.
    """
    marker = root / "read-started.json"
    previous = read(marker)
    try:
        os.kill(previous["driver_pid"], 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError("the original read driver has not stopped")
    if any(root.glob("actor-input-*.json")) or any(root.glob("actor-http-*.json")):
        raise RuntimeError("a prepared or dispatched actor request cannot be replaced")
    if (root / "read-artifact.json").exists() or (root / "read-result.json").exists():
        raise RuntimeError("an existing read result cannot be replaced")
    journal = root / "actor-requests.sqlite3"
    if journal.exists():
        with sqlite3.connect(journal.as_uri() + "?mode=ro", uri=True) as db:
            if db.execute("SELECT COUNT(*) FROM requests").fetchone()[0]:
                raise RuntimeError("an actor journal operation cannot be replaced")
    preserved = root / f"read-preflight-failed-{previous['driver_pid']}.json"
    preserve(preserved, previous)
    marker.unlink()


def read_generator(root, plan, tokenizer):
    counted = CountedTransport(root, "actor")
    generator = ExternalSGLangRolloutGenerator(
        config=ExternalSGLangRolloutConfig(plan["endpoint"]),
        tokenizer=tokenizer,
        gateway=None,  # Explicit adapter-free Step-0; never an actor LoRA fallback.
        snapshot_provider=lambda: PolicySnapshot.from_value(plan["policy"]),
        transport=counted,
        request_journal=DurableRequestJournal(root / "actor-requests.sqlite3"),
    )
    return generator, counted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("author-mutate", "resume-read"))
    parser.add_argument("--root", type=Path, required=True)
    for name in ("backbone", "formal-config", "step0-policy", "initial-library"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--base-model")
    parser.add_argument("--resume-before-dispatch", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "author-mutate":
        if args.resume_before_dispatch:
            parser.error("preflight recovery applies only to resume-read")
        from skillev_private.experiments.fresh_restart import load_fresh_config

        if any(
            getattr(args, name) is None
            for name in (
                "backbone",
                "formal_config",
                "step0_policy",
                "initial_library",
                "endpoint",
                "base_model",
            )
        ):
            parser.error("author-mutate requires all explicit runtime bindings")
        formal = load_fresh_config(args.formal_config)
        backbone = QwenMultimodalBackboneConfig.from_value(read(args.backbone))
        formal.require_backbone(backbone)
        policy = PolicySnapshot.from_value(read(args.step0_policy))
        if (
            policy.forward_adapter_version != "adapter-free"
            or policy.tokenizer_id != backbone.tokenizer_id
        ):
            raise ValueError(
                "controlled actor/author must use the actual frozen-base Step-0 identity"
            )
        rollout = replace(
            formal.sampling_config,
            max_turns=4,
            max_reasoning_tokens=1024,
            max_action_tokens=2048,
            reasoning_by_domain=(),
            reasoning_native_thinking=True,
            skill_exposure="catalog-then-read@1",
            task_semantic_guidance="legacy",
            hotpot_deliberation=False,
            per_rollout_maximum=conservative_rollout_maximum(
                max_turns=4,
                max_reasoning_tokens=1024,
                max_action_tokens=2048,
                max_model_input_tokens=formal.max_input_tokens,
                max_tool_wall_time_milliseconds=1000,
            ),
        )
        author_config = formal.application_config(PURPOSE)
        evolution = author_config.evolution
        plan = {
            "purpose": PURPOSE,
            "endpoint": args.endpoint,
            "base_model": args.base_model,
            "backbone": backbone.to_value(),
            "policy": policy.to_value(),
            "evolution": evolution.to_value(),
            "rollout": rollout.to_value(),
            "authoring_sampling": author_config.authoring_sampling.to_value(),
            "max_input_tokens": formal.max_input_tokens,
            "condition_scope": "controlled public development; not IID nor formal training",
            "task_semantics": (
                "explicit synthetic public action instructions; no fabricated benchmark domain"
            ),
            "seed": 0,
            "no_trainables_loaded": True,
        }
        initialize(root, plan, SkillLibraryState.from_value(read(args.initial_library)))
    else:
        if any(
            getattr(args, name) is not None
            for name in (
                "backbone",
                "formal_config",
                "step0_policy",
                "initial_library",
                "endpoint",
                "base_model",
            )
        ):
            parser.error("resume-read takes only --root; saved conditions cannot be overridden")
        plan = read(root / "plan.json")
        if (root / "read-started.json").exists():
            if not args.resume_before_dispatch:
                raise RuntimeError(
                    "read episode was already attempted; no replacement driver is permitted"
                )
            resume_before_dispatch(root)
        elif args.resume_before_dispatch:
            parser.error("no failed read preflight exists to recover")
    plan = read(root / "plan.json")
    tokenizer = QwenTokenizerAdapter.from_config(
        QwenMultimodalBackboneConfig.from_value(plan["backbone"])
    )
    resumed = args.mode == "resume-read"
    library, skill_id = recover_mutation(root, plan, tokenizer, resumed=resumed)
    if resumed:
        if read(root / "author-summary.json")["driver_pid"] == os.getpid():
            raise RuntimeError("resume-read must be a new driver process")
        generator, counted = read_generator(root, plan, tokenizer)
        try:
            preserve(root / "read-started.json", {"driver_pid": os.getpid(), "started_at": clock()})
            asyncio.run(read_episode(root, plan, library, skill_id, generator))
        finally:
            generator.close()
            preserve(root / "actor-physical-summary.json", {"physical_actor_calls": counted.calls})
    print(json.dumps({"root": str(root), "mode": args.mode, "completed": True}))


if __name__ == "__main__":
    main()
