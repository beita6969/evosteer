"""Explicit development practice, not the autonomous formal rollout condition.

The same owner still chooses and generates every action. The public instruction
encourages trying a relevant procedure, but supplies neither an action nor an
answer. No read quota, mask, reward bonus, replacement sample or extra agent.
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

from skillev.contracts import JsonValue
from skillev.evolution import BaseRolloutSessionFactory
from skillev.rollout import RolloutTask, UnskilledRolloutSessionBundle

PROFILE = "training-development-skill-practice@1"
PROFILE_V2 = "training-development-skill-practice@2"
PROFILE_V3 = "training-development-skill-practice@3"
CONTEXT_KEY = "development_skill_practice"
INSTRUCTION = (
    "This is a training-development skill-use practice episode, not an evaluation. "
    "Before committing to a long solution, compare the visible skill summaries with "
    "the task's needs. When a document offers a relevant procedure you have not read, "
    "try reading it early, then apply the parts that actually help the task. "
    "You choose the document and all subsequent actions. If no listed procedure "
    "would help, solve directly. If a read turns out to be inapplicable, disregard it. "
    "Do not reread an already visible document as though it performed another check. "
    "Reading is not execution, verification or task success. There is no call-count "
    "target or extra reward; the original token, turn and tool-call budgets still apply."
)


INSTRUCTION_V2 = INSTRUCTION + (
    " The examples below are fictional teaching material, not this episode's history, "
    "tool responses, skill bodies, or evidence of any executed action. "
    "A catalog summary is not the document body. If a concrete procedure would help, "
    "you can end the current reasoning phase at that decision, before lengthy reasoning, "
    "then use the real native read_skill tool in the action phase. Its skill_id must "
    "come from the current visible catalog; these examples select no current skill. "
    "Only an actual read response supplies the body. If the catalog is empty or no "
    "procedure fits, solve directly. The examples are not a mandatory action sequence."
)


INSTRUCTION_V3 = (
    "This is a requested skill-use teaching demonstration on training-development material, "
    "not autonomous adoption, IID evaluation, or formal training. Before solving the complete "
    "task or committing to lengthy reasoning, inspect the actual visible skill catalog and "
    "yourself select one relevant procedure, if one genuinely applies. Demonstrate its use: "
    "end the current reasoning phase at this selection decision, then issue a real native "
    "read_skill action using the skill_id you selected from that catalog. Do not finish the "
    "whole solution in reasoning before reading. A catalog summary is not the full body; "
    "wait for the actual read response, examine the body, and apply its relevant procedure "
    "in subsequent real task actions. You, the same owner, choose the skill and generate "
    "every action; no current skill ID or task answer is supplied by this instruction. "
    "If the catalog is empty or no skill genuinely applies, state that truthfully and solve "
    "directly rather than making a cosmetic read. If the returned body proves inapplicable, "
    "explain the mismatch and abandon it. If that body is already actually visible, use it "
    "without repeating a read. Reading alone is not application, verification or task success. "
    "This is an explicit teaching request, not an enforced tool-call quota or an extra reward. "
    "The original token, turn and tool-call budgets, native action interface and scoring remain "
    "unchanged. The examples below are fictional teaching material, not this episode's history, "
    "tool responses, skill bodies or evidence of executed actions."
)


def _fictional_examples() -> list[JsonValue]:
    return [
        {
            "case": "procedure-needed",
            "fictional_situation": "A made-up toy filing job needs an order-preserving procedure.",
            "illustration": [
                "A catalog summary suggests a relevant procedure, but is not its full body.",
                "The owner ends reasoning before a long plan; "
                "its next action uses native read_skill.",
                "The skill_id argument is selected from the actual catalog; "
                "no example ID is supplied.",
                "After the real body returns, the owner checks applicability and uses it "
                "in a later task action. Reading alone performs no filing and proves no success.",
            ],
        },
        {
            "case": "reasonable-direct-solution",
            "fictional_situation": "A trivial toy filing job needs no unfamiliar procedure.",
            "illustration": "The owner solves directly without reading; a read would add no value.",
        },
        {
            "case": "read-then-abandon",
            "fictional_situation": "A read procedure assumes a capability the toy workplace lacks.",
            "illustration": "The owner notices the mismatch, discards that advice, and chooses "
            "a suitable task action. It does not force the procedure to fit.",
        },
        {
            "case": "already-visible-do-not-repeat",
            "fictional_situation": "The same body is already visible from an actual read.",
            "illustration": "The owner consults that visible body and continues task work; "
            "rereading it is not another verification or environmental action.",
        },
    ]


def practice_controls(profile: object) -> dict[str, JsonValue] | None:
    if profile is None:
        return None
    if profile not in (PROFILE, PROFILE_V2, PROFILE_V3):
        raise ValueError("unsupported explicitly declared development practice")
    controls: dict[str, JsonValue] = {
        "profile": cast(str, profile),
        "instruction": {
            PROFILE: INSTRUCTION,
            PROFILE_V2: INSTRUCTION_V2,
            PROFILE_V3: INSTRUCTION_V3,
        }[cast(str, profile)],
        "autonomous_use_evidence": False,
        "formal_training_condition": False,
        "read_quota": None,
        "reward_bonus": 0,
    }
    if profile in (PROFILE_V2, PROFILE_V3):
        controls["teaching_material"] = {
            "kind": "fictional-not-current-execution",
            "counts_as_invocation_or_application_evidence": False,
            "examples": _fictional_examples(),
        }
    if profile == PROFILE_V3:
        controls["demonstration"] = {
            "kind": "requested-teaching-demonstration",
            "timing": "select-before-long-reasoning-and-complete-solution",
            "skill_choice": "same-owner-from-actual-catalog-if-applicable",
            "actions": "owner-generated-native-read-then-task-application",
            "framework_inserted_actions": False,
            "iid_condition": False,
            "success_or_application_inferred_from_request": False,
        }
    return controls


class PracticeSessionFactory:
    """Annotate hydrated public tasks; the native session and verdict stay intact."""

    def __init__(self, base: BaseRolloutSessionFactory, profile: str) -> None:
        controls = practice_controls(profile)
        assert controls is not None
        self.base = base
        self.controls = controls
        self.prepared: dict[str, tuple[RolloutTask, RolloutTask]] = {}

    async def prepare_tasks(self, tasks: tuple[RolloutTask, ...]) -> tuple[RolloutTask, ...]:
        prepare = getattr(self.base, "prepare_tasks", None)
        native = tasks if prepare is None else cast(tuple[RolloutTask, ...], await prepare(tasks))
        if tuple(t.task_id for t in native) != tuple(t.task_id for t in tasks):
            raise ValueError("practice preparation changed the fixed task order")
        result = []
        for task in native:
            if not isinstance(task.public_context, dict) or CONTEXT_KEY in task.public_context:
                raise ValueError("practice needs unmodified structured public task context")
            guided = replace(
                task, public_context={**task.public_context, CONTEXT_KEY: self.controls}
            )
            pair = task, guided
            if task.task_id in self.prepared and self.prepared[task.task_id] != pair:
                raise ValueError("practice task changed after preparation")
            self.prepared[task.task_id] = pair
            result.append(guided)
        return tuple(result)

    def create(self, task: RolloutTask) -> UnskilledRolloutSessionBundle:
        native, guided = self.prepared[task.task_id]
        if task != guided:
            raise ValueError("episode differs from the actual prepared practice task")
        return self.base.create(native)
