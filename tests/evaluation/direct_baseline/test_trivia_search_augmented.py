from __future__ import annotations

import asyncio

import pytest

from skillev.evaluation.direct_baseline.client import (
    DirectGenerationError,
    DirectGenerationRequest,
    DirectGenerationResult,
)
from skillev.evaluation.direct_baseline.config import DirectDecodingProfile
from skillev.evaluation.direct_baseline.search_augmented import (
    TriviaSearchCommandKind,
    TriviaSearchHit,
    TriviaSearchStep,
    TriviaSearchTask,
    parse_trivia_search_command,
    render_trivia_search_messages,
    run_trivia_search_task,
)


class _Client:
    def __init__(self, responses: list[str | DirectGenerationError]) -> None:
        self.responses = responses
        self.requests: list[DirectGenerationRequest] = []

    async def generate(self, request: DirectGenerationRequest) -> DirectGenerationResult:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, DirectGenerationError):
            raise response
        return DirectGenerationResult(
            request.request_id,
            response,
            "stop",
            10,
            3,
            response_model="qwen35-direct-base",
        )


class _Backend:
    def __init__(self) -> None:
        self.queries: list[tuple[str, str]] = []

    def search(
        self,
        task_id: str,
        query: str,
        *,
        limit: int,
        max_per_document: int,
    ) -> tuple[TriviaSearchHit, ...]:
        self.queries.append((task_id, query))
        assert (limit, max_per_document) == (5, 2)
        return (TriviaSearchHit("p1", "d1", "Title", "Public evidence.", 1),)


class _OverlappingBackend:
    def __init__(self) -> None:
        self.limits: list[int] = []

    def search(
        self,
        task_id: str,
        query: str,
        *,
        limit: int,
        max_per_document: int,
    ) -> tuple[TriviaSearchHit, ...]:
        del task_id, query, max_per_document
        self.limits.append(limit)
        available = tuple(
            TriviaSearchHit(
                f"p{index}",
                f"d{index}",
                f"Title {index}",
                f"Public evidence {index}.",
                index,
            )
            for index in range(1, 7)
        )
        return available[:limit]


def _task(max_search_calls: int = 3) -> TriviaSearchTask:
    return TriviaSearchTask(
        "trivia-001",
        "Which public fact is requested?",
        DirectDecodingProfile(
            "qwen35-nonthinking-general@1",
            False,
            0.7,
            0.8,
            20,
            0.0,
            1.5,
            1.0,
            8192,
            seed=42,
        ),
        "skillflow-released-iid-v3",
        42,
        max_search_calls=max_search_calls,
    )


def test_parser_accepts_one_unambiguous_labeled_command_line() -> None:
    assert parse_trivia_search_command("Search: public entity").kind is (
        TriviaSearchCommandKind.SEARCH
    )
    assert parse_trivia_search_command("Final answer: Public value").kind is (
        TriviaSearchCommandKind.FINAL
    )
    extracted = parse_trivia_search_command(
        "I used the retrieved evidence.\nFinal answer: Public value\nDone."
    )
    assert extracted == parse_trivia_search_command("Final answer: Public value")
    assert parse_trivia_search_command("Search: x\nFinal answer: y").kind is (
        TriviaSearchCommandKind.INVALID
    )
    assert parse_trivia_search_command("Search: x\nSearch: y").kind is (
        TriviaSearchCommandKind.INVALID
    )
    assert parse_trivia_search_command("Reasoning line.\nPublic value").kind is (
        TriviaSearchCommandKind.INVALID
    )
    fallback = parse_trivia_search_command(
        "Reasoning line.\nPublic value", allow_unlabeled_final=True
    )
    assert fallback.kind is TriviaSearchCommandKind.FINAL
    assert fallback.value == "Public value"


def test_search_runner_requires_search_then_scores_one_final_answer() -> None:
    client = _Client(["Search: public entity", "Final answer: Public value"])
    backend = _Backend()

    attempt = asyncio.run(run_trivia_search_task(client, _task(), backend))

    assert attempt.final_response.value == "Public value"
    assert attempt.search_calls == 1
    assert backend.queries == [("trivia-001", "public entity")]
    assert "You must search before answering" in client.requests[0].messages[-1]["content"]
    assert "Public evidence" in client.requests[1].messages[-1]["content"]


def test_search_runner_resumes_without_regenerating_completed_search() -> None:
    completed = TriviaSearchStep(
        1,
        "Search: prior query",
        None,
        TriviaSearchCommandKind.SEARCH,
        "prior query",
        "[1] p1 | Title\nPrior evidence.",
        "stop",
        10,
        3,
    )
    client = _Client(["Final answer: resumed"])
    backend = _Backend()

    attempt = asyncio.run(
        run_trivia_search_task(client, _task(), backend, resume_trace=(completed,))
    )

    assert attempt.final_response.value == "resumed"
    assert attempt.search_calls == 1
    assert backend.queries == []
    assert len(client.requests) == 1


def test_search_runner_treats_malformed_candidate_as_definitive_failure() -> None:
    attempt = asyncio.run(
        run_trivia_search_task(_Client(["I would search now."]), _task(), _Backend())
    )

    assert attempt.infrastructure_error is None
    assert attempt.final_response.value is None
    assert attempt.search_calls == 0


def test_search_runner_uses_last_line_only_after_search_budget_is_exhausted() -> None:
    client = _Client(["Search: public entity", "Reasoning line.\nPublic value"])

    attempt = asyncio.run(run_trivia_search_task(client, _task(max_search_calls=1), _Backend()))

    assert attempt.final_response.value == "Public value"
    assert attempt.search_calls == 1


def test_detailed_prompt_repeats_strict_one_line_wire_contract() -> None:
    base = _task()
    task = TriviaSearchTask(
        base.task_id,
        base.question,
        base.profile,
        base.population_id,
        base.run_seed,
        prompt_profile_id="triviaqa-local-search-react-detailed@2",
    )
    client = _Client(["Search: public entity", "Final answer: concise"])

    attempt = asyncio.run(run_trivia_search_task(client, task, _Backend()))

    assert attempt.final_response.value == "concise"
    assert "exactly one line" in client.requests[0].messages[0]["content"]


def test_external_action_surface_receives_only_public_task_and_retrieval_state() -> None:
    base = _task(max_search_calls=2)
    task = TriviaSearchTask(
        base.task_id,
        base.question,
        base.profile,
        base.population_id,
        base.run_seed,
        max_search_calls=2,
        external_action_surface=True,
    )
    trace = (
        TriviaSearchStep(
            1,
            "Search: first public relation",
            None,
            TriviaSearchCommandKind.SEARCH,
            "first public relation",
            "[1] p1 | Title\nPublic evidence.",
            "stop",
            10,
            3,
        ),
    )

    messages = render_trivia_search_messages(task, trace, final_only=False)

    assert tuple(item["role"] for item in messages) == ("user",)
    assert "Which public fact is requested?" in messages[0]["content"]
    assert "Public evidence" in messages[0]["content"]
    assert "1 search(es) completed; 1 search slot(s) remain" in messages[0]["content"]
    assert "must search" not in messages[0]["content"].casefold()
    assert "final answer" not in messages[0]["content"].casefold()


def test_search_runner_can_remove_passages_seen_on_prior_searches() -> None:
    base = _task(max_search_calls=2)
    task = TriviaSearchTask(
        base.task_id,
        base.question,
        base.profile,
        base.population_id,
        base.run_seed,
        max_search_calls=2,
        hits_per_search=2,
        deduplicate_search_hits=True,
    )
    backend = _OverlappingBackend()
    client = _Client(["Search: first relation", "Search: second relation", "Final answer: concise"])

    attempt = asyncio.run(run_trivia_search_task(client, task, backend))

    assert attempt.final_response.value == "concise"
    assert backend.limits == [2, 4]
    observations = tuple(step.observation for step in attempt.trace if step.observation is not None)
    assert "p1" in observations[0]
    assert "p2" in observations[0]
    assert "p1" not in observations[1]
    assert "p2" not in observations[1]
    assert "p3" in observations[1]
    assert "p4" in observations[1]


def test_search_runner_retries_only_generation_infrastructure_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    client = _Client(
        [
            DirectGenerationError("temporary"),
            "Search: public entity",
            "Final answer: recovered",
        ]
    )

    attempt = asyncio.run(run_trivia_search_task(client, _task(), _Backend()))

    assert attempt.final_response.value == "recovered"
    assert len(client.requests) == 3
