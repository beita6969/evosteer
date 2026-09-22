"""One isolated evaluated-policy episode; the only I/O is a scoped public broker.

No private-evaluation import, evaluator path, model endpoint, or credential is
provided to this process. One owner makes all task decisions through the broker.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict
from typing import Any, cast

from skillev.benchmarks.reasoning_guidance import HOTPOT_DELIBERATION
from skillev.evolution.task_features import configured_public_task_features
from skillev.rollout import RolloutGenerationRequest, RolloutGenerationResult
from skillev.rollout.evaluation_sglang import (
    EvaluationGenerationConstraint,
    EvaluationGenerationProfile,
    EvaluationPolicyDescriptor,
)
from skillev.task_semantic_guidance import (
    LEGACY_TASK_SEMANTICS,
    evaluation_submission_instruction,
)

from .corpus_search import CorpusSearchProfile
from .direct_baseline import DirectGenerationRequest
from .direct_baseline.config import DirectBenchmark, DirectDecodingProfile
from .direct_baseline.interactive_tasks import (
    NativeEnvironmentStep,
    NativeInteractiveTask,
    NativePublicState,
    task_with_authoritative_reset_instruction,
)
from .input_metric_contracts import CONTRACTS, PublicTaskView
from .integrity_controller import IntegrityStepZeroClient
from .integrity_generation import EvaluationBudgetExhausted, EvaluationDeadlineExceeded
from .public_context import PublicPrompt
from .sealed_candidates import FinalCandidate
from .skill_library_config import FrozenSkillLibrary
from .step0_integrity import InferenceArm, SkillMode, decode_integrity_arm
from .step0_types import (
    ArchitectureInferenceState,
    StepZeroActionMode,
    StepZeroArchitectureConfig,
    StepZeroCompletionMode,
    StepZeroTaskBinding,
)


def rpc(operation: str, **arguments: object) -> Any:
    print(json.dumps({"operation": operation, **arguments}, ensure_ascii=False), flush=True)
    raw = sys.stdin.readline()
    if not raw:
        raise RuntimeError("public broker disconnected")
    return json.loads(raw)


class BrokerTokenizer:
    tokenizer_id = "qwen35-service-tokenizer"

    def encode(self, text: str) -> list[int]:
        return cast(list[int], rpc("encode", text=text))

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return cast(str, rpc("decode", token_ids=token_ids))

    def encode_rollout_prompt(self, text: str) -> list[int]:
        return self.encode_integrity_messages(
            ({"role": "user", "content": text},), enable_thinking=False
        )

    def encode_integrity_messages(
        self,
        messages: tuple[dict[str, str], ...],
        *,
        enable_thinking: bool,
        tools: tuple[dict[str, Any], ...] = (),
    ) -> list[int]:
        return cast(
            list[int],
            rpc("encode-messages", messages=messages, enable_thinking=enable_thinking, tools=tools),
        )


class BrokerGenerator:
    tokenizer = BrokerTokenizer()

    def __init__(self, descriptor: EvaluationPolicyDescriptor) -> None:
        self.descriptor = descriptor
        self.episodes: set[str] = set()

    def snapshot(self) -> EvaluationPolicyDescriptor:
        return self.descriptor

    def begin_episode(self, episode_id: str, expected_policy_snapshot_id: str) -> None:
        if (
            expected_policy_snapshot_id != self.descriptor.snapshot_id
            or episode_id in self.episodes
        ):
            raise ValueError("invalid actor episode identity")
        self.episodes.add(episode_id)

    def end_episode(self, episode_id: str) -> None:
        self.episodes.remove(episode_id)

    async def generate_evaluation(
        self,
        request: RolloutGenerationRequest,
        *,
        profile: EvaluationGenerationProfile,
        constraint: EvaluationGenerationConstraint | None = None,
    ) -> RolloutGenerationResult:
        result = rpc(
            "generate",
            request=asdict(request),
            profile=asdict(profile),
            constraint=asdict(constraint) if constraint else None,
        )
        if result.get("episode_deadline_reached"):
            raise EvaluationDeadlineExceeded("serving request was not admitted")
        return RolloutGenerationResult.from_value(result)


class BrokerTrace:
    def record(self, scope: tuple[str, str, str], stage: str, payload: object) -> None:
        del scope  # The parent fills scope; the actor cannot write another episode.
        rpc("trace", stage=stage, payload=payload)


def decode_arm(value: dict[str, Any]) -> InferenceArm:
    return decode_integrity_arm(value)


def shared_task_instruction(entry: PublicTaskView, arm: InferenceArm) -> str:
    """Project a selected shared semantic catalog into evaluation's text wire."""
    if arm.task_semantic_guidance == LEGACY_TASK_SEMANTICS:
        raise ValueError("shared instruction requested for a legacy evaluation arm")
    return " ".join(
        part
        for part in (
            CONTRACTS[entry.benchmark].task_semantics(
                entry.input_profile,
                hotpot_deliberation=arm.hotpot_deliberation,
                version=arm.task_semantic_guidance,
            ),
            evaluation_submission_instruction(entry.benchmark),
        )
        if part
    )


async def run_episode(initial: dict[str, Any]) -> FinalCandidate:
    value = initial["public_task"]
    entry = PublicTaskView(
        value["task_id"],
        value["benchmark"],
        tuple(tuple(item) for item in value["fields"]),
        value.get("input_profile", "released-iid-source@1"),
    )
    arm, run_id = decode_arm(initial["arm"]), str(initial["run_id"])
    profile = DirectDecodingProfile(
        **{**initial["decoding"], "stop": tuple(initial["decoding"].get("stop", ()))}
    )
    settings = initial["budgets"]
    generator = BrokerGenerator(EvaluationPolicyDescriptor(**initial["policy"]))
    skills = (
        FrozenSkillLibrary.from_value(initial["skill_library"])
        if arm.skill_mode is not SkillMode.OFF and initial.get("skill_library") is not None
        else None
    )
    mode = (
        StepZeroActionMode(entry.benchmark)
        if entry.benchmark in {"webshop", "alfworld", "scienceworld"}
        else StepZeroActionMode.COMPLETION
    )
    completion = {
        "hotpotqa": StepZeroCompletionMode.SHORT_ANSWER,
        "triviaqa": StepZeroCompletionMode.SHORT_ANSWER,
        "musique": StepZeroCompletionMode.SHORT_ANSWER,
        "nq-open": StepZeroCompletionMode.SHORT_ANSWER,
        "aime-2026": StepZeroCompletionMode.AIME_BOXED_INTEGER,
        "mbpp-plus": StepZeroCompletionMode.PYTHON_SOURCE,
        "humaneval": StepZeroCompletionMode.PYTHON_SOURCE,
        "livecodebench": StepZeroCompletionMode.PYTHON_SOURCE,
        "apps-introductory": StepZeroCompletionMode.PYTHON_SOURCE,
    }.get(entry.benchmark, StepZeroCompletionMode.NATURAL_LANGUAGE)
    features = configured_public_task_features(
        entry.benchmark, initial.get("skill_task_features", {})
    )
    client = IntegrityStepZeroClient(
        generator=generator,
        library_state=skills.state if skills is not None else None,
        skill_library=skills,
        bindings=(
            StepZeroTaskBinding(
                task_id=entry.task_id,
                benchmark_id=entry.benchmark,
                task_family=features.task_family,
                context_id=features.context_id,
                panel_index=initial["panel_position"],
                action_mode=mode,
                completion_mode=completion,
                skill_instruction_token_budget=settings["skill_instruction_tokens"],
                owner_finish_allowed=initial.get("owner_finish_allowed", False),
            ),
        ),
        config=StepZeroArchitectureConfig(
            settings["calls_per_turn"],
            profile.max_new_tokens,
            profile.max_new_tokens,
            settings["history_input_tokens"],
            profile.seed,
            "Qwen3.5-9B",
            initial["policy"]["snapshot_id"],
            inference_state=ArchitectureInferenceState(**initial.get("inference_state", {})),
            arm=arm,
        ),
        journal=BrokerTrace(),
        run_id=run_id,
        corpus_search=(
            lambda query: json.dumps(rpc("corpus-search", query=query), ensure_ascii=False)
        )
        if initial.get("corpus_search")
        else None,
    )
    client.generation.call_limit = settings["total_model_calls"]
    client.generation.output_token_limit = settings["total_output_tokens"]
    client.generation.context_length = settings["context_length"]
    client.generation.native_chunk_tokens = settings.get("native_chunk_tokens")
    client.generation.native_final_reserve_tokens = settings.get("native_final_reserve_tokens", 0)
    client.generation.native_close_at_reserve = settings.get("native_close_at_reserve", 0) == 1
    client.generation.finalization_reserve_tokens = settings.get("finalization_reserve_tokens", 0)
    client.generation.deadline_monotonic = initial.get("deadline_monotonic")
    final = ""
    actions: list[str] = []
    stop_reason = "infrastructure"
    submission = None
    try:
        if mode in {
            StepZeroActionMode.WEB_SHOP,
            StepZeroActionMode.ALF_WORLD,
            StepZeroActionMode.SCIENCE_WORLD,
        }:
            state = rpc("environment-reset")
            task = NativeInteractiveTask(
                entry.task_id,
                DirectBenchmark(entry.benchmark),
                entry.render()
                if entry.input_profile == "training-public-source-bridge@1"
                else dict(entry.fields)["task"],
                profile,
                settings["environment_steps"],
                "public-native-tools@1",
                CONTRACTS[entry.benchmark].parser,
                initial["population_id"],
                profile.seed,
                history_window_steps=None,
                history_maximum_characters=None,
            )
            public_state = NativePublicState(
                state["observation_text"], tuple(state["available_actions"])
            )
            task = task_with_authoritative_reset_instruction(task, public_state)
            await client.begin_interactive_episode(
                task,
                public_state,
                semantic_instruction=shared_task_instruction(entry, arm)
                if arm.task_semantic_guidance != LEGACY_TASK_SEMANTICS
                else "",
            )
            stop_reason = "action-budget"
            for step in range(task.max_steps):
                client.generation.require_time_remaining()
                response = await client.generate(
                    DirectGenerationRequest(
                        f"{entry.task_id}:step:{step + 1}",
                        ({"role": "user", "content": task.task},),
                        profile,
                    )
                )
                client.generation.require_time_remaining()
                if response.finish_reason == "owner-finish":
                    rpc("owner-finish")
                    client.counts.tool_calls += 1
                    stop_reason = "owner-finish"
                    break
                if not response.text:
                    stop_reason = (
                        "model-budget"
                        if client.counts.model_calls >= settings["total_model_calls"]
                        else "turn-call-budget"
                        if response.finish_reason == "budget-exhausted"
                        else "unsubmitted"
                    )
                    break
                action = response.text.removeprefix("Action: ")
                decided_on = client.current_decision(entry.task_id)
                state = rpc(
                    "environment-step",
                    action=action,
                    source_revision=decided_on.state_revision,
                    decision_id=f"{entry.task_id}:action:{step + 1}",
                )
                if state.get("episode_deadline_reached"):
                    raise EvaluationDeadlineExceeded("native action admission deadline reached")
                actions.append(action)
                await client.observe_interactive_episode(
                    entry.task_id,
                    action,
                    NativeEnvironmentStep(
                        state["observation"],
                        state["terminal"],
                        0.0,
                        False,
                        state["action_valid"],
                        available_actions=tuple(state["available_actions"]),
                    ),
                )
                if state["terminal"]:
                    stop_reason = "terminal"
                    break
            final = json.dumps(actions, ensure_ascii=False)
        else:
            if (
                entry.benchmark == "livemedbench"
                and entry.input_profile == "training-public-source-bridge@1"
            ):
                messages: tuple[dict[str, str], ...] = (
                    {"role": "system", "content": shared_task_instruction(entry, arm)},
                    {"role": "user", "content": entry.render()},
                )
            elif entry.benchmark == "livemedbench":
                from .livemedbench import generation_instruction

                public = dict(entry.fields)
                instruction = generation_instruction(public["narrative"], public["core_request"])
                if arm.task_semantic_guidance != LEGACY_TASK_SEMANTICS:
                    instruction = shared_task_instruction(entry, arm) + "\n\n" + instruction
                messages = (
                    {
                        "role": "user",
                        "content": instruction
                        + "\n\n"
                        + public["narrative"]
                        + "\n\n"
                        + public["core_request"],
                    },
                )
            elif entry.benchmark == "healthbench":
                conversation = PublicPrompt.required(entry.conversation()).with_instruction(
                    shared_task_instruction(entry, arm)
                    if arm.task_semantic_guidance != LEGACY_TASK_SEMANTICS
                    else "You are a helpful assistant. Write the assistant's next reply to the "
                    "user's latest message using the preceding dialogue as context. "
                    "This is a conversation, not a short-answer exam: your reply can "
                    "provide information, explain uncertainty, or ask for clarification "
                    "as appropriate."
                )
                messages = tuple(block.message() for block in conversation.blocks)
            else:
                instructions = {
                    "hotpotqa": (
                        "Answer the question using the supplied public passages. "
                        "For a factoid question, return the answer span using the passage's "
                        "wording. For a yes/no question, return yes or no. "
                        "If you include an explanation, put your short answer on a separate "
                        "Final answer: line."
                    ),
                    "triviaqa": (
                        "Answer the question; the supplied reading context is available as "
                        "evidence. Return the short answer itself, rather than a sentence "
                        "restating the question. If you include an explanation, put the "
                        "short answer on a separate Final answer: line."
                    ),
                    "aime-2026": (
                        "Solve the problem and state your final integer from 0 through 999. "
                        "A boxed integer is also accepted."
                    ),
                    "humaneval": (
                        "Implement the requested function. Return executable Python source, "
                        "preserving the supplied function signature."
                    ),
                    "mbpp-plus": (
                        "Implement the requested Python function using the function name and "
                        "calling convention shown in the public examples. The examples are "
                        "part of the specification, including the expected return values. "
                        "Return executable Python source."
                    ),
                }
                qa_submission = (
                    "The submitted answer is scored as a short phrase, not an explanation. "
                    "You can give the answer alone or use a Final answer: field; "
                    "Short answer: and Answer: are also accepted. The first nonempty line "
                    "of that field is submitted. You may explain your reasoning outside "
                    "the answer field."
                )
                code_submission = (
                    "Bare Python source or a fenced program is accepted. If you include "
                    "several fenced blocks or explicitly labeled final code sections, "
                    "the last is your submitted program."
                )
                instructions.update(
                    {
                        "musique": (
                            "Answer the question using the supplied passages as evidence. "
                            + qa_submission
                        ),
                        "nq-open": (
                            (
                                "Answer using your knowledge and the frozen corpus search. "
                                if initial.get("corpus_search")
                                else "Answer the question from your own knowledge. "
                            )
                            + qa_submission
                        ),
                        "omni-math": (
                            "Solve the mathematical problem. Give your complete final answer, "
                            "including a proof if the problem asks for one. Put the final "
                            "mathematical answer in \\boxed{} when applicable. Keep the solution "
                            "concise enough to finish within the available output budget; "
                            "do not repeat the derivation after giving your final answer."
                        ),
                        "math-hard": (
                            "Solve the mathematical problem and state a single final answer, "
                            "preferably in \\boxed{...}. Explanation may precede it."
                        ),
                        "gpqa-diamond-bioorganic": (
                            "Answer the multiple-choice question using the supplied options. "
                            "Choose exactly one of A, B, C, or D. Put your final choice on a "
                            "separate Final answer: line."
                        ),
                        "livecodebench": (
                            "Write a complete Python solution following the supplied problem "
                            "and calling convention. You may explain your reasoning and use "
                            "the available capabilities as needed. Provide your final "
                            "executable program as the solution. " + code_submission
                        ),
                        "apps-introductory": (
                            "Write a complete Python solution following the supplied problem "
                            "and calling convention. You may explain your reasoning and use "
                            "the available capabilities as needed. Provide your final "
                            "executable program as the solution. " + code_submission
                        ),
                    }
                )
                if (
                    entry.input_profile == "training-public-source-bridge@1"
                    and entry.benchmark == "triviaqa"
                ):
                    instructions["triviaqa"] = (
                        "Answer the question from the supplied public task. Return the short "
                        "answer itself, rather than a sentence restating the question. "
                        "If you include an explanation, put the short answer on a separate "
                        "Final answer: line."
                    )
                if entry.benchmark == "hotpotqa" and arm.native_thinking:
                    # Owner-requested HotpotQA-only deliberation, not an effort
                    # API label or a forced minimum generation length. The same
                    # public guidance applies to every task, without score input.
                    instructions["hotpotqa"] = HOTPOT_DELIBERATION + instructions["hotpotqa"]
                if arm.task_semantic_guidance != LEGACY_TASK_SEMANTICS:
                    instructions[entry.benchmark] = shared_task_instruction(entry, arm)
                if initial.get("corpus_search"):
                    instructions[entry.benchmark] += (
                        "\n" + CorpusSearchProfile(**initial["corpus_search"]).instruction()
                    )
                messages = (
                    {"role": "system", "content": instructions[entry.benchmark]},
                    {"role": "user", "content": entry.render()},
                )
            response = await client.generate(
                DirectGenerationRequest(entry.task_id, messages, profile)
            )
            final = response.text
            stop_reason = "terminal" if final else "unsubmitted"
    except EvaluationBudgetExhausted as exc:
        # An unfinished answer is a candidate failure, never a scorer-driven retry.
        rpc(
            "trace",
            stage="candidate-budget-exhausted",
            payload={"model_calls": client.counts.model_calls},
        )
        stop_reason = (
            "wall-clock" if isinstance(exc, EvaluationDeadlineExceeded) else "model-budget"
        )
    finally:
        if mode in {
            StepZeroActionMode.WEB_SHOP,
            StepZeroActionMode.ALF_WORLD,
            StepZeroActionMode.SCIENCE_WORLD,
        }:
            # Already acknowledged actions survive a later exhausted model budget.
            final = json.dumps(actions, ensure_ascii=False)
        submission = client.final_submission(entry.task_id)
        await client.close_public_episode(entry.task_id, reason=stop_reason)
    return FinalCandidate(
        run_id,
        arm.arm_id,
        entry.task_id,
        generator.descriptor.snapshot_id,
        f"{entry.task_id}:owner-final",
        final,
        CONTRACTS[entry.benchmark].parser,
        client.generation.input_tokens,
        client.generation.output_tokens,
        asdict(client.counts),
        submission=asdict(submission) if submission is not None else None,
    )


def main() -> None:
    initial = json.loads(sys.stdin.readline())
    final = asyncio.run(run_episode(initial))
    print(
        json.dumps({"operation": "final", "value": asdict(final)}, ensure_ascii=False), flush=True
    )


if __name__ == "__main__":
    main()
