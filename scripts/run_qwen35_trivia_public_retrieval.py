#!/usr/bin/env python3
"""Run the fixed-public-corpus TriviaQA lane with one adapter-free Qwen route."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from skillev_private.benchmarks.qa_metrics import normalize_triviaqa_answer, score_qa_answers

from skillev.evaluation.direct_baseline import (
    OpenAICompatibleDirectClient,
    QwenChatTokenCounter,
)
from skillev.evaluation.direct_baseline.client import DirectGenerationRequest
from skillev.evaluation.direct_baseline.config import DirectDecodingProfile
from skillev.evaluation.trivia_retrieval.session import (
    PublicPassage,
    SearchHit,
    TriviaQAPublicRetrievalSession,
)


@dataclass(frozen=True, slots=True)
class _Task:
    task_id: str
    question: str
    accepted_answers: tuple[str, ...]


def _load_tasks(public_path: Path, target_path: Path) -> tuple[_Task, ...]:
    questions: dict[str, str] = {}
    with public_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            questions[str(row["task_id"])] = str(row["question"])
    targets: dict[str, tuple[str, ...]] = {}
    with target_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            answers = row.get("accepted_answers")
            if not isinstance(answers, list) or not answers:
                raise ValueError("Trivia target requires accepted answers")
            targets[str(row["task_id"])] = tuple(str(item) for item in answers)
    if set(questions) != set(targets) or len(questions) != 128:
        raise ValueError("Trivia public and private panels must contain the same 128 task IDs")
    return tuple(_Task(task_id, questions[task_id], targets[task_id]) for task_id in questions)


def _messages(task: _Task, transcript: list[str]) -> tuple[dict[str, str], ...]:
    contract = (
        'Return exactly one JSON object: {"action":"search","value":"query"}, '
        '{"action":"read","value":"passage_id"}, or '
        '{"action":"complete","value":"short answer"}. '
        "You may search at most twice and read at most twice. Use only public search evidence."
    )
    return (
        {"role": "system", "content": contract},
        {
            "role": "user",
            "content": f"Question: {task.question}\n\n" + "\n\n".join(transcript),
        },
    )


def _parse_action(text: str) -> tuple[str, str]:
    visible = text.rsplit("</think>", 1)[-1].strip()
    try:
        row = json.loads(visible)
    except json.JSONDecodeError as exc:
        raise ValueError("Trivia action is not JSON") from exc
    if not isinstance(row, dict) or set(row) != {"action", "value"}:
        raise ValueError("Trivia action has an incompatible shape")
    action, value = row["action"], row["value"]
    if (
        action not in {"search", "read", "complete"}
        or not isinstance(value, str)
        or not value.strip()
    ):
        raise ValueError("Trivia action is invalid")
    return str(action), value.strip()


def _render_result(value: tuple[SearchHit, ...] | PublicPassage) -> str:
    if isinstance(value, PublicPassage):
        return f"Read {value.passage_id} | {value.title}\n{value.text}"
    if not value:
        return "No public passages matched."
    return "\n".join(f"{item.passage_id} | {item.title} | {item.snippet}" for item in value)


async def _run_one(
    task: _Task,
    *,
    client: OpenAICompatibleDirectClient,
    database: Path,
    profile: DirectDecodingProfile,
) -> dict[str, object]:
    transcript: list[str] = []
    answer: str | None = None
    failure: str | None = None
    with TriviaQAPublicRetrievalSession(database) as session:
        for turn in range(1, 6):
            generation = await client.generate(
                DirectGenerationRequest(
                    request_id=f"{task.task_id}:public-retrieval:{turn}",
                    messages=_messages(task, transcript),
                    profile=profile,
                )
            )
            try:
                action, value = _parse_action(generation.text)
                if action == "complete":
                    answer = value
                    break
                observation = session.search(value) if action == "search" else session.read(value)
                failure = None
                transcript.append(
                    f"Assistant action: {action}({value})\nObservation: "
                    f"{_render_result(observation.results)}\n"
                    f"Remaining searches: {observation.remaining_searches}; "
                    f"remaining reads: {observation.remaining_reads}"
                )
            except (KeyError, RuntimeError, ValueError) as exc:
                failure = type(exc).__name__
                transcript.append(
                    "Protocol observation: the previous action was rejected. Return exactly one "
                    "valid JSON search, read, or complete action within the remaining budget."
                )
    metrics = (
        score_qa_answers(answer, task.accepted_answers, normalizer=normalize_triviaqa_answer)
        if answer is not None
        else None
    )
    return {
        "task_id": task.task_id,
        "answer": answer,
        "candidate_failure": failure,
        "em": 0.0 if metrics is None else metrics.em,
        "f1": 0.0 if metrics is None else metrics.f1,
    }


async def _run(args: argparse.Namespace) -> None:
    tasks = _load_tasks(args.public_tasks, args.private_targets)
    if args.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    client = OpenAICompatibleDirectClient(
        endpoint_base=args.endpoint_base,
        served_model_name=args.served_model_name,
        api_key="EMPTY",
        timeout_seconds=args.request_timeout_seconds,
        context_length=args.context_length,
        token_counter=QwenChatTokenCounter.from_pretrained(args.tokenizer_path),
    )
    if await client.model_routes() != (args.served_model_name,):
        raise RuntimeError("SGLang model route is incompatible")
    profile = DirectDecodingProfile(
        "qwen35-trivia-public-retrieval@1",
        False,
        0.7,
        0.8,
        20,
        0.0,
        0.0,
        1.0,
        1024,
        seed=0,
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    async def limited(task: _Task) -> dict[str, object]:
        async with semaphore:
            return await _run_one(task, client=client, database=args.database, profile=profile)

    rows = tuple(await asyncio.gather(*(limited(task) for task in tasks)))
    args.private_output_dir.mkdir(parents=True, exist_ok=False)
    with (args.private_output_dir / "per-task-results.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "format": "skillev-trivia-public-retrieval-summary@1",
        "condition_id": "triviaqa-public-retrieval-no-skill@2",
        "sample_count": 128,
        "definitive_count": 128,
        "candidate_failure_count": sum(row["answer"] is None for row in rows),
        "infrastructure_failure_count": 0,
        "em": sum(float(row["em"]) for row in rows) / 128,
        "f1": sum(float(row["f1"]) for row in rows) / 128,
        "auxiliary_models": [],
        "target_dependency": False,
    }
    (args.private_output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-tasks", type=Path, required=True)
    parser.add_argument("--private-targets", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--endpoint-base", required=True)
    parser.add_argument("--served-model-name", default="qwen35-direct-base")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--context-length", type=int, default=98_304)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--request-timeout-seconds", type=float, default=600.0)
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
