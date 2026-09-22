"""Answer-blind interactive retrieval for adapter-free direct QA evaluation."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from .client import DirectGenerationClient, DirectGenerationError, DirectGenerationRequest
from .config import DirectBenchmark, DirectDecodingProfile
from .parsing import ParsedResponse, ParseReason, ParseStatus, parse_short_answer_last_line

_SEARCH = re.compile(r"(?im)^\s*search\s*:\s*([^\r\n]+?)\s*$")
_FINAL = re.compile(r"(?im)^\s*final\s+answer\s*:\s*([^\r\n]+?)\s*$")


class TriviaSearchCommandKind(StrEnum):
    SEARCH = "search"
    FINAL = "final"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class TriviaSearchCommand:
    kind: TriviaSearchCommandKind
    value: str | None


def parse_trivia_search_command(
    text: str, *, allow_unlabeled_final: bool = False
) -> TriviaSearchCommand:
    """Parse one answer-blind search or final command without repairing it."""

    visible = text.rsplit("</think>", 1)[-1].strip()
    searches = _SEARCH.findall(visible)
    finals = _FINAL.findall(visible)
    if len(searches) == 1 and not finals:
        query = searches[0].strip()
        if query and len(query) <= 256 and "\n" not in query:
            return TriviaSearchCommand(TriviaSearchCommandKind.SEARCH, query)
    if len(finals) == 1 and not searches:
        answer = finals[0].strip()
        if answer and "\n" not in answer:
            return TriviaSearchCommand(TriviaSearchCommandKind.FINAL, answer)
    if allow_unlabeled_final and not searches and not finals:
        parsed = parse_short_answer_last_line(visible)
        if parsed.value is not None:
            return TriviaSearchCommand(TriviaSearchCommandKind.FINAL, parsed.value)
    return TriviaSearchCommand(TriviaSearchCommandKind.INVALID, None)


@dataclass(frozen=True, slots=True)
class TriviaSearchHit:
    passage_id: str
    document_id: str
    title: str
    snippet: str
    rank: int

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (self.passage_id, self.document_id, self.title, self.snippet)
        ):
            raise ValueError("search hit text fields must be non-empty")
        if self.rank < 1:
            raise ValueError("search hit rank must be positive")


class TriviaSearchBackend(Protocol):
    def search(
        self,
        task_id: str,
        query: str,
        *,
        limit: int,
        max_per_document: int,
    ) -> tuple[TriviaSearchHit, ...]: ...


@dataclass(frozen=True, slots=True)
class TriviaSearchTask:
    task_id: str
    question: str
    profile: DirectDecodingProfile
    population_id: str
    run_seed: int
    max_search_calls: int = 3
    hits_per_search: int = 5
    max_hits_per_document: int = 2
    infrastructure_maximum_attempts: int = 3
    prompt_profile_id: str = "triviaqa-local-search-react@1"
    external_action_surface: bool = False
    deduplicate_search_hits: bool = False

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.question.strip() or not self.population_id.strip():
            raise ValueError("search task identity, question, and population are required")
        if self.run_seed != self.profile.seed:
            raise ValueError("search task seed must equal decoding seed")
        if type(self.external_action_surface) is not bool:
            raise TypeError("external_action_surface must be boolean")
        if type(self.deduplicate_search_hits) is not bool:
            raise TypeError("deduplicate_search_hits must be boolean")
        if self.prompt_profile_id not in {
            "triviaqa-local-search-react@1",
            "triviaqa-local-search-react-detailed@2",
        }:
            raise ValueError("search task prompt profile is unsupported")
        if (
            min(
                self.max_search_calls,
                self.hits_per_search,
                self.max_hits_per_document,
                self.infrastructure_maximum_attempts,
            )
            < 1
        ):
            raise ValueError("search budgets must be positive")


@dataclass(frozen=True, slots=True)
class TriviaSearchStep:
    turn_index: int
    raw_text: str
    reasoning_text: str | None
    command_kind: TriviaSearchCommandKind
    command_value: str
    observation: str | None
    finish_reason: str
    prompt_tokens: int | None
    completion_tokens: int | None


@dataclass(frozen=True, slots=True)
class TriviaSearchAttempt:
    task_id: str
    benchmark: DirectBenchmark
    final_response: ParsedResponse
    search_calls: int
    infrastructure_error: str | None
    trace: tuple[TriviaSearchStep, ...]

    @property
    def candidate_valid(self) -> bool:
        return self.infrastructure_error is None and self.final_response.value is not None


TriviaSearchStepObserver = Callable[[str, TriviaSearchStep], Awaitable[None]]


_SYSTEM_V1 = (
    "Answer the TriviaQA question using the private local search tool. Do not assume any "
    "passage was supplied with the question. Follow the requested wire format exactly: output "
    "only `Search: <query>` when searching, or only `Final answer: <short answer>` when the "
    "instructions permit a final answer. Never output JSON."
)

_SYSTEM_V2 = (
    "Answer the TriviaQA question using the private local search tool. Your whole response on "
    "every turn must be exactly one line and contain no explanation, preamble, Markdown, or JSON. "
    "Use only `Search: <concise query>` to search. Use only `Final answer: <short answer>` to "
    "answer. Any other text makes the response invalid. Search before answering."
)


def render_trivia_search_messages(
    task: TriviaSearchTask,
    trace: tuple[TriviaSearchStep, ...],
    *,
    final_only: bool,
) -> tuple[dict[str, str], ...]:
    transcript = [f"Question:\n{task.question}"]
    for step in trace:
        if step.command_kind is TriviaSearchCommandKind.SEARCH:
            transcript.append(f"Search: {step.command_value}\nResults:\n{step.observation}")
    if task.external_action_surface:
        searches = sum(step.command_kind is TriviaSearchCommandKind.SEARCH for step in trace)
        transcript.append(
            "Public retrieval state: "
            f"{searches} search(es) completed; "
            f"{max(task.max_search_calls - searches, 0)} search slot(s) remain."
        )
        return ({"role": "user", "content": "\n\n".join(transcript)},)
    if final_only:
        instruction = "Search budget exhausted. Output only `Final answer: <short answer>`."
    elif not trace:
        instruction = "You must search before answering. Output only `Search: <query>`."
    else:
        remaining = task.max_search_calls - sum(
            step.command_kind is TriviaSearchCommandKind.SEARCH for step in trace
        )
        instruction = (
            f"You may make {remaining} more search call(s). Output either one `Search: <query>` "
            "or one `Final answer: <short answer>`."
        )
    transcript.append(instruction)
    system = _SYSTEM_V2 if task.prompt_profile_id.endswith("@2") else _SYSTEM_V1
    return (
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(transcript)},
    )


def _invalid(task_id: str, trace: list[TriviaSearchStep]) -> TriviaSearchAttempt:
    return TriviaSearchAttempt(
        task_id,
        DirectBenchmark.TRIVIA_QA,
        ParsedResponse(None, ParseStatus.EMPTY, ParseReason.INVALID_FORMAT, 0),
        sum(step.command_kind is TriviaSearchCommandKind.SEARCH for step in trace),
        None,
        tuple(trace),
    )


async def run_trivia_search_task(
    client: DirectGenerationClient,
    task: TriviaSearchTask,
    backend: TriviaSearchBackend,
    *,
    resume_trace: tuple[TriviaSearchStep, ...] = (),
    step_observer: TriviaSearchStepObserver | None = None,
) -> TriviaSearchAttempt:
    """Run one immutable search trajectory, resuming only after infrastructure failure."""

    trace = list(resume_trace)
    if any(step.command_kind is not TriviaSearchCommandKind.SEARCH for step in trace):
        raise ValueError("resume trace may contain only completed search steps")
    searches = len(trace)
    if searches > task.max_search_calls:
        raise ValueError("resume trace exceeds the search budget")
    turn_index = len(trace) + 1
    seen_passage_ids = (
        {passage_id for step in trace for passage_id in _rendered_passage_ids(step.observation)}
        if task.deduplicate_search_hits
        else set()
    )
    while True:
        final_only = searches == task.max_search_calls
        generation = None
        for infrastructure_attempt in range(1, task.infrastructure_maximum_attempts + 1):
            try:
                generation = await client.generate(
                    DirectGenerationRequest(
                        request_id=f"{task.task_id}:search-turn:{turn_index}",
                        messages=render_trivia_search_messages(
                            task, tuple(trace), final_only=final_only
                        ),
                        profile=task.profile,
                    )
                )
                break
            except DirectGenerationError as exc:
                if infrastructure_attempt == task.infrastructure_maximum_attempts:
                    return TriviaSearchAttempt(
                        task.task_id,
                        DirectBenchmark.TRIVIA_QA,
                        ParsedResponse(None, ParseStatus.EMPTY, ParseReason.EMPTY_RESPONSE, 0),
                        searches,
                        type(exc).__name__,
                        tuple(trace),
                    )
                await asyncio.sleep(float(infrastructure_attempt))
        assert generation is not None
        command = parse_trivia_search_command(
            generation.text,
            allow_unlabeled_final=final_only,
        )
        if command.kind is TriviaSearchCommandKind.INVALID or command.value is None:
            return _invalid(task.task_id, trace)
        if command.kind is TriviaSearchCommandKind.FINAL:
            if searches == 0:
                return _invalid(task.task_id, trace)
            step = TriviaSearchStep(
                turn_index,
                generation.text,
                generation.reasoning_text,
                command.kind,
                command.value,
                None,
                generation.finish_reason,
                generation.prompt_tokens,
                generation.completion_tokens,
            )
            trace.append(step)
            if step_observer is not None:
                await step_observer(task.task_id, step)
            return TriviaSearchAttempt(
                task.task_id,
                DirectBenchmark.TRIVIA_QA,
                ParsedResponse(
                    command.value,
                    ParseStatus.EXTRACTED,
                    ParseReason.EXPLICIT_FINAL,
                    1,
                ),
                searches,
                None,
                tuple(trace),
            )
        if final_only:
            return _invalid(task.task_id, trace)
        hits = None
        for infrastructure_attempt in range(1, task.infrastructure_maximum_attempts + 1):
            try:
                requested_hits = task.hits_per_search
                if task.deduplicate_search_hits:
                    requested_hits += len(seen_passage_ids)
                hits = backend.search(
                    task.task_id,
                    command.value,
                    limit=requested_hits,
                    max_per_document=task.max_hits_per_document,
                )
                if task.deduplicate_search_hits:
                    hits = _without_seen_hits(
                        hits,
                        seen_passage_ids,
                        limit=task.hits_per_search,
                    )
                break
            except (OSError, RuntimeError) as exc:
                if infrastructure_attempt == task.infrastructure_maximum_attempts:
                    return TriviaSearchAttempt(
                        task.task_id,
                        DirectBenchmark.TRIVIA_QA,
                        ParsedResponse(None, ParseStatus.EMPTY, ParseReason.EMPTY_RESPONSE, 0),
                        searches,
                        type(exc).__name__,
                        tuple(trace),
                    )
                await asyncio.sleep(float(infrastructure_attempt))
        assert hits is not None
        observation = _render_hits(hits)
        step = TriviaSearchStep(
            turn_index,
            generation.text,
            generation.reasoning_text,
            command.kind,
            command.value,
            observation,
            generation.finish_reason,
            generation.prompt_tokens,
            generation.completion_tokens,
        )
        trace.append(step)
        if task.deduplicate_search_hits:
            seen_passage_ids.update(hit.passage_id for hit in hits)
        if step_observer is not None:
            await step_observer(task.task_id, step)
        searches += 1
        turn_index += 1


def _render_hits(hits: tuple[TriviaSearchHit, ...]) -> str:
    if not hits:
        return "No matching passages."
    return "\n\n".join(
        f"[{hit.rank}] {hit.passage_id} | {hit.title}\n{hit.snippet}" for hit in hits
    )


def _rendered_passage_ids(observation: str | None) -> tuple[str, ...]:
    """Recover stable passage identities from a completed private search observation."""

    if observation is None:
        return ()
    return tuple(
        match.group(1).strip()
        for match in re.finditer(r"(?m)^\[\d+\]\s+([^|\r\n]+?)\s+\|", observation)
    )


def _without_seen_hits(
    hits: tuple[TriviaSearchHit, ...],
    seen_passage_ids: set[str],
    *,
    limit: int,
) -> tuple[TriviaSearchHit, ...]:
    """Keep the highest-ranked unseen passages and renumber their display ranks."""

    selected = tuple(hit for hit in hits if hit.passage_id not in seen_passage_ids)[:limit]
    return tuple(
        TriviaSearchHit(
            hit.passage_id,
            hit.document_id,
            hit.title,
            hit.snippet,
            rank,
        )
        for rank, hit in enumerate(selected, start=1)
    )
