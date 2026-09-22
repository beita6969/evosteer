"""Current six-IID/six-OOD catalog and public/native contracts (not rewards).

The WebShop contract remains readable for historical records; it is not a
member of the active catalog and cannot enter a new evaluation panel.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .external_judge_policy import OMNI_JUDGE_METRIC
from .livemedbench import METRIC as LIVEMEDBENCH_METRIC

if TYPE_CHECKING:
    from skillev.rollout import RolloutTask

IID_BENCHMARKS = (
    "hotpotqa",
    "triviaqa",
    "aime-2026",
    "healthbench",
    "alfworld",
    "mbpp-plus",
)

OOD_BENCHMARKS = (
    "musique",
    "nq-open",
    "math-hard",
    "gpqa-diamond-bioorganic",
    "scienceworld",
    "apps-introductory",
)

# Retain source identities for reading old artifacts, not for new panel loading.
HISTORICAL_IID_BENCHMARKS = (*IID_BENCHMARKS, "humaneval")
HISTORICAL_OOD_BENCHMARKS = (
    "musique",
    "nq-open",
    "omni-math",
    "livemedbench",
    "scienceworld",
    "livecodebench",
    "apps-introductory",
)
OOD_SOURCE_BENCHMARKS = frozenset(
    (*OOD_BENCHMARKS, *HISTORICAL_OOD_BENCHMARKS, "gpqa-diamond-health")
)


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """Public task interface shared across native evaluation and training bridges.

    References, hidden tests, rubrics and observed scores are not spec inputs.
    Episode-specific numeric budgets are frozen separately by the caller.
    """

    benchmark: str
    required_source_fields: tuple[str, ...]
    optional_source_fields: tuple[str, ...]
    runtime_public_fields: tuple[str, ...]
    tools: tuple[str, ...]
    payload: str
    parser: str
    metric: str
    formula: str
    examples: str = "none unless explicitly released in the public input"
    extra_help: str = "none; frozen skill-library access is a separate declared arm"
    denominator: str = "every frozen panel ID, including candidate failures"
    retry: str = (
        "transport retry is distinct from budgeted public action repair; no score retry generation"
    )
    selection: str = (
        "one evaluated-policy final; no vote, baseline fallback, or targeted regeneration"
    )
    budget_semantics: str = (
        "one owner; all generation and interface repairs share the episode ledger"
    )
    termination: str = (
        "unique owner submission or declared budget stop; native environment for actions"
    )

    @property
    def input_profiles(self) -> tuple[str, ...]:
        from .corpus_search import INPUT_PROFILE

        released = (
            "released-ood-source@1"
            if self.benchmark in OOD_SOURCE_BENCHMARKS
            else "released-iid-source@1"
        )
        return (released, "training-public-source-bridge@1") + (
            (INPUT_PROFILE,) if self.benchmark == "nq-open" else ()
        )

    def task_semantics(
        self, input_profile: str, *, version: str, hotpot_deliberation: bool = False
    ) -> str:
        from skillev.task_semantic_guidance import public_task_semantics

        return public_task_semantics(
            self.benchmark,
            input_profile=input_profile,
            version=version,
            hotpot_deliberation=hotpot_deliberation,
        )

    @property
    def source_fields(self) -> tuple[str, ...]:
        """Fields that the source exporter, rather than a live reset, may project."""
        return self.required_source_fields + self.optional_source_fields

    @property
    def public_fields(self) -> tuple[str, ...]:
        """All public fields, including reset-only runtime state."""
        return self.source_fields + self.runtime_public_fields


# Compatibility for older source adapters; there is one spec registry, not two.
InputMetricContract = BenchmarkSpec


CONTRACTS = {
    "hotpotqa": InputMetricContract(
        benchmark="hotpotqa",
        required_source_fields=("question", "context"),
        optional_source_fields=(),
        runtime_public_fields=(),
        tools=(),
        payload="passage answer span or yes/no",
        parser="single-short-answer-explicit-owner-v3",
        metric="answer-f1",
        formula="official distractor answer F1/EM over aliases; all ten public passages",
    ),
    "triviaqa": InputMetricContract(
        benchmark="triviaqa",
        required_source_fields=("question", "public_context"),
        optional_source_fields=(),
        runtime_public_fields=(),
        tools=(),  # Released reading-comprehension lane, not the old generated-dossier lane.
        payload="answer text",
        parser="single-short-answer-explicit-owner-v3",
        metric="answer-f1",
        formula="official TriviaQA punctuation-to-spaces EM/F1 over aliases",
    ),
    "aime-2026": InputMetricContract(
        benchmark="aime-2026",
        required_source_fields=("problem",),
        optional_source_fields=(),
        runtime_public_fields=(),
        tools=(),
        payload="integer 0..999",
        parser="single-aime-final-explicit-owner-v5",
        metric="accuracy",
        formula="exact integer equality over all 30 problems",
    ),
    "healthbench": InputMetricContract(
        benchmark="healthbench",
        required_source_fields=("prompt",),
        optional_source_fields=(),
        runtime_public_fields=(),
        tools=(),
        payload="natural-language response",
        parser="natural-language-explicit-owner-v2",
        metric="qwen-local-rubric-score",
        formula=(
            "clip(mean(raw earned points / sum positive points), 0, 1); "
            "negative item scores retained"
        ),
    ),
    "webshop": InputMetricContract(
        benchmark="webshop",
        required_source_fields=("task",),
        optional_source_fields=(),
        runtime_public_fields=("observation", "available_actions"),
        tools=("search", "click"),
        payload="environment trajectory",
        parser="explicit-native-action-explicit-owner-v2",
        metric="native-reward",
        formula="official environment reward mean; official success also reported",
    ),
    "alfworld": InputMetricContract(
        benchmark="alfworld",
        required_source_fields=("task",),
        optional_source_fields=(),
        runtime_public_fields=("observation", "available_actions"),
        tools=("act",),
        payload="environment trajectory",
        parser="explicit-native-action-explicit-owner-v2",
        metric="success",
        formula="official task success fraction, not inferred action completion",
    ),
    "mbpp-plus": InputMetricContract(
        benchmark="mbpp-plus",
        required_source_fields=("prompt",),
        optional_source_fields=(),
        runtime_public_fields=(),
        tools=(),
        payload="Python source",
        parser="single-python-source-explicit-owner-v3",
        metric="base-plus-pass-at-1",
        formula=(
            "EvalPlus MBPP v0.2 Base AND Plus tests; owner display MBPP+ hard, "
            "no separate official hard split"
        ),
    ),
    "humaneval": InputMetricContract(
        benchmark="humaneval",
        required_source_fields=("prompt",),
        optional_source_fields=(),
        runtime_public_fields=(),
        tools=(),
        payload="Python source",
        parser="single-python-source-explicit-owner-v3",
        metric="pass-at-1",
        formula="original HumanEval tests on one candidate; no hidden-test feedback",
    ),
}


# OOD shares the owner/broker boundary, not the IID population or scorer labels.
for _name in ("musique", "nq-open"):
    CONTRACTS[_name] = InputMetricContract(
        _name,
        ("question", "context") if _name == "musique" else ("question",),
        (),
        (),
        (),
        "short answer",
        "single-short-answer-explicit-owner-v3",
        "answer-f1" if _name == "musique" else "answer-exact-match",
        "official normalized answer EM/F1 over released aliases",
    )
CONTRACTS["omni-math"] = InputMetricContract(
    "omni-math",
    ("problem",),
    (),
    (),
    (),
    "complete mathematical answer",
    "natural-language-explicit-owner-v2",
    OMNI_JUDGE_METRIC,
    "official Omni-MATH equivalence prompt with declared lab-gpt-5.6-luna medium API judge",
)
CONTRACTS["math-hard"] = InputMetricContract(
    "math-hard",
    ("problem",),
    (),
    (),
    (),
    "mathematical final answer, preferably boxed",
    "natural-language-explicit-owner-v2",
    "accuracy",
    "MATH Level 5 test; unique submitted answer, Math-Verify symbolic equivalence",
)
CONTRACTS["gpqa-diamond-bioorganic"] = InputMetricContract(
    "gpqa-diamond-bioorganic",
    ("question", "options"),
    (),
    (),
    (),
    "one A-D choice label, optionally with an explanation",
    "natural-language-explicit-owner-v2",
    "accuracy",
    "one unambiguous final choice equals the private label in the frozen option order",
)
CONTRACTS["livemedbench"] = InputMetricContract(
    "livemedbench",
    ("narrative", "core_request"),
    (),
    (),
    (),
    "complete natural-language response to the patient",
    "natural-language-explicit-owner-v2",
    LIVEMEDBENCH_METRIC,
    "clip(mean(sum(points * met) / sum(positive points)), 0, 1); raw negatives retained",
)
for _name in ("livecodebench", "apps-introductory"):
    CONTRACTS[_name] = InputMetricContract(
        _name,
        ("prompt",),
        (),
        (),
        (),
        "Python source",
        "single-python-source-explicit-owner-v3",
        "pass-at-1",
        "one candidate passes all official code-generation tests",
    )
CONTRACTS["scienceworld"] = InputMetricContract(
    "scienceworld",
    ("task",),
    (),
    ("observation", "available_actions"),
    ("act",),
    "environment trajectory",
    "explicit-native-action-explicit-owner-v2",
    "native-final-score",
    "official final environment score / 100, including negatives; separate clipped learning reward",
)


REQUIRED_SOURCE_FIELDS = {
    benchmark: contract.required_source_fields for benchmark, contract in CONTRACTS.items()
}


def _require_nonempty_text(value: object, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be non-empty public text")
    return value


def _validate_hotpot_context(value: object) -> list[object]:
    if not isinstance(value, list) or len(value) != 10:
        raise ValueError("HotpotQA context requires exactly ten public passages")
    for position, passage in enumerate(value):
        if type(passage) is str:
            _require_nonempty_text(passage, f"context[{position}]")
            continue
        if not isinstance(passage, list | tuple) or len(passage) != 2:
            raise TypeError("each HotpotQA passage must be text or a title/sentences pair")
        title, sentences = passage
        _require_nonempty_text(title, f"context[{position}].title")
        if not isinstance(sentences, list) or not sentences:
            raise ValueError("each HotpotQA title/sentences pair needs public sentences")
        for sentence_position, sentence in enumerate(sentences):
            _require_nonempty_text(sentence, f"context[{position}].sentences[{sentence_position}]")
    return value


def _validate_health_prompt(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("HealthBench prompt must be a non-empty public conversation")
    public_messages: list[dict[str, str]] = []
    for position, message in enumerate(value):
        if not isinstance(message, Mapping):
            raise TypeError("each HealthBench message must be a mapping")
        role = _require_nonempty_text(message.get("role"), f"prompt[{position}].role")
        if role not in {"system", "user", "assistant"} or (role == "system" and position != 0):
            raise ValueError("HealthBench source conversation has an unsupported role layout")
        content = _require_nonempty_text(message.get("content"), f"prompt[{position}].content")
        public_messages.append({"role": role, "content": content})
    return public_messages


def require_public_fields(benchmark: str, row: Mapping[str, object]) -> None:
    """Require every source-owned public field before actor projection."""
    try:
        required_fields = REQUIRED_SOURCE_FIELDS[benchmark]
    except KeyError as error:
        raise ValueError(f"unsupported benchmark: {benchmark}") from error
    missing = tuple(name for name in required_fields if name not in row)
    if missing:
        raise ValueError(f"missing required public fields: {', '.join(missing)}")
    contract = CONTRACTS[benchmark]
    for name in contract.required_source_fields:
        _validate_public_source_field(benchmark, name, row[name])
    for name in contract.optional_source_fields:
        if name in row:
            _validate_public_source_field(benchmark, name, row[name])


def require_runtime_public_fields(benchmark: str, row: Mapping[str, object]) -> None:
    """Validate live-reset state without allowing it into source-record export."""
    try:
        runtime_fields = CONTRACTS[benchmark].runtime_public_fields
    except KeyError as error:
        raise ValueError(f"unsupported benchmark: {benchmark}") from error
    missing = tuple(name for name in runtime_fields if name not in row)
    if missing:
        raise ValueError(f"missing runtime public fields: {', '.join(missing)}")
    for name in runtime_fields:
        value = row[name]
        if name == "observation":
            _require_nonempty_text(value, name)
        elif name == "available_actions":
            if not isinstance(value, list | tuple) or not value:
                raise ValueError("available_actions must be a non-empty public action sequence")
            for position, action in enumerate(value):
                _require_nonempty_text(action, f"available_actions[{position}]")
        else:
            raise ValueError(f"unsupported runtime public field: {name}")


def _validate_public_source_field(benchmark: str, name: str, value: object) -> object:
    if benchmark == "hotpotqa" and name == "context":
        return _validate_hotpot_context(value)
    if benchmark == "healthbench" and name == "prompt":
        return _validate_health_prompt(value)
    if benchmark == "triviaqa" and name == "public_context" and isinstance(value, list):
        # The released RC source stores a list of public passages. Preserve
        # every passage, in order, rather than rejecting it as a non-string
        # field or exposing a Python/JSON wrapper as part of the question.
        if not value:
            raise ValueError("TriviaQA requires non-empty released reading context")
        return "\n\n".join(
            _require_nonempty_text(passage, f"public_context[{index}]")
            for index, passage in enumerate(value)
        )
    return _require_nonempty_text(value, name)


@dataclass(frozen=True, slots=True)
class PublicTaskView:
    """An allowlist projection, never a blacklist/redaction of evaluator labels.

    Only the exporter sees a private source row. Actor workers receive this
    serialized projection, not the source record, rubric, aliases, or tests.
    """

    task_id: str
    benchmark: str
    fields: tuple[tuple[str, str], ...]
    input_profile: str = "released-iid-source@1"

    def __post_init__(self) -> None:
        _require_nonempty_text(self.task_id, "task_id")
        if self.benchmark not in CONTRACTS:
            raise ValueError(f"unsupported benchmark: {self.benchmark}")
        if self.input_profile == "training-public-source-bridge@1":
            if self.benchmark not in HISTORICAL_IID_BENCHMARKS + OOD_BENCHMARKS or tuple(
                name for name, _ in self.fields
            ) != (
                "query",
                "public_context",
                "source_messages",
            ):
                raise ValueError("training bridge requires its explicit public task fields")
            for name, value in self.fields:
                _require_nonempty_text(value, name)
            context = json.loads(dict(self.fields)["public_context"])
            if not isinstance(context, dict) or context.get("benchmark_id") != self.benchmark:
                raise ValueError("training public context belongs to another benchmark")
            messages = json.loads(dict(self.fields)["source_messages"])
            if messages:
                _validate_health_prompt(messages)
            elif messages != []:
                raise ValueError("source messages must be an explicit message list")
            return
        contract = CONTRACTS[self.benchmark]
        if self.input_profile not in contract.input_profiles:
            raise ValueError("public input profile belongs to another benchmark catalog")
        names = tuple(name for name, _ in self.fields)
        expected_fields = contract.required_source_fields + tuple(
            name for name in contract.optional_source_fields if name in names
        )
        if names != expected_fields:
            raise ValueError("public task fields do not match the source-field contract")
        for name, value in self.fields:
            if type(name) is not str or type(value) is not str or not value:
                raise TypeError("public task fields must be non-empty text pairs")
            public_value = (
                json.loads(value)
                if (self.benchmark, name) in {("hotpotqa", "context"), ("healthbench", "prompt")}
                else value
            )
            _validate_public_source_field(self.benchmark, name, public_value)

    @classmethod
    def from_record(
        cls, task_id: str, benchmark: str, record: Mapping[str, object]
    ) -> PublicTaskView:
        require_public_fields(benchmark, record)
        contract = CONTRACTS[benchmark]
        fields: list[tuple[str, str]] = []
        for name in contract.source_fields:
            if name not in record:
                continue  # Optional fields remain absent rather than being synthesized.
            value = _validate_public_source_field(benchmark, name, record[name])
            fields.append(
                (name, value if type(value) is str else json.dumps(value, ensure_ascii=False))
            )
        return cls(
            task_id,
            benchmark,
            tuple(fields),
            "released-ood-source@1"
            if benchmark in OOD_SOURCE_BENCHMARKS
            else "released-iid-source@1",
        )

    @classmethod
    def from_training_task(cls, task: RolloutTask) -> PublicTaskView:
        """Project only an already answer-free RolloutTask, never its evaluator.

        This deliberately does not relabel closed-book training as released RC.
        No paragraph splitting, scaffold rebuilding or reference lookup occurs.
        """
        context = task.public_context
        if (
            not isinstance(context, dict)
            or context.get("benchmark_id") not in HISTORICAL_IID_BENCHMARKS + OOD_BENCHMARKS
        ):
            raise ValueError("training public task requires a supported public-input contract")
        return cls(
            task.task_id,
            str(context["benchmark_id"]),
            (
                ("query", task.query),
                ("public_context", json.dumps(context, ensure_ascii=False)),
                (
                    "source_messages",
                    json.dumps(
                        [{"role": m.role, "content": m.content} for m in task.source_messages],
                        ensure_ascii=False,
                    ),
                ),
            ),
            "training-public-source-bridge@1",
        )

    def conversation(self) -> tuple[dict[str, str], ...]:
        field = (
            "source_messages"
            if self.input_profile == "training-public-source-bridge@1"
            else "prompt"
        )
        messages = _validate_health_prompt(json.loads(dict(self.fields)[field]))
        return tuple(messages)

    def render(self) -> str:
        if self.input_profile == "training-public-source-bridge@1":
            fields = dict(self.fields)
            text = fields["query"] + "\n\nPublic context:\n" + fields["public_context"]
            if json.loads(fields["source_messages"]):
                text += "\n\nSource conversation:\n" + fields["source_messages"]
            return text
        if self.benchmark == "hotpotqa" and "context" in dict(self.fields):
            fields = dict(self.fields)
            # Present every released passage as text, not a JSON string array,
            # with the question after the evidence. No passage is shortened.
            passages = json.loads(fields["context"])
            blocks = [
                passage if isinstance(passage, str) else f"{passage[0]}\n" + "\n".join(passage[1])
                for passage in passages
            ]
            return "Passages:\n" + "\n\n".join(blocks) + f"\n\nQuestion:\n{fields['question']}"
        return "\n\n".join(f"{name}:\n{value}" for name, value in self.fields)
