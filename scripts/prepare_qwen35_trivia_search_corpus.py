#!/usr/bin/env python3
"""Build the private GPT-knowledge plus Wikipedia TriviaQA search database."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from skillev_private.direct_reference.journal import require_fresh_output_directory
from skillev_private.direct_reference.trivia_search import (
    CODEX_PLAN_PROFILE,
    DATABASE_FORMAT,
    CodexKnowledgePlan,
    CorpusPassage,
    WikipediaClient,
    build_trivia_search_database,
    generate_codex_knowledge_plan,
    load_frozen_trivia_questions,
    passages_for_task,
    select_wikipedia_pages,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iid-population", required=True, type=Path)
    parser.add_argument("--private-output-dir", required=True, type=Path)
    parser.add_argument("--codex-model", default="gpt-5.6-sol")
    parser.add_argument("--codex-reasoning-effort", default="xhigh")
    parser.add_argument("--codex-concurrency", type=int, default=2)
    parser.add_argument("--codex-maximum-attempts", type=int, default=3)
    parser.add_argument("--codex-plans-input", type=Path)
    parser.add_argument("--maximum-wikipedia-pages", type=int, default=4)
    return parser.parse_args()


def _generate_one(
    *,
    task_id: str,
    question: str,
    root: Path,
    ordinal: int,
    model: str,
    reasoning_effort: str,
    maximum_attempts: int,
) -> CodexKnowledgePlan:
    for attempt in range(1, maximum_attempts + 1):
        try:
            return generate_codex_knowledge_plan(
                task_id=task_id,
                question=question,
                work_directory=root / f"item-{ordinal:03d}" / f"attempt-{attempt}",
                model=model,
                reasoning_effort=reasoning_effort,
            )
        except RuntimeError:
            if attempt == maximum_attempts:
                raise
    raise AssertionError("unreachable")


def _load_plans(path: Path) -> dict[str, CodexKnowledgePlan]:
    plans: dict[str, CodexKnowledgePlan] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if type(value) is not dict or set(value) != {
                "task_id",
                "background_note",
                "queries",
            }:
                raise ValueError("Codex plan input has an incompatible row")
            raw_queries = value["queries"]
            if type(raw_queries) is not list or len(raw_queries) != 3:
                raise ValueError("Codex plan input has incompatible queries")
            plan = CodexKnowledgePlan(
                str(value["task_id"]),
                str(value["background_note"]),
                (str(raw_queries[0]), str(raw_queries[1]), str(raw_queries[2])),
            )
            if plan.task_id in plans:
                raise ValueError("Codex plan input repeats a task ID")
            plans[plan.task_id] = plan
    return plans


def main() -> None:
    args = _arguments()
    if (
        min(
            args.codex_concurrency,
            args.codex_maximum_attempts,
            args.maximum_wikipedia_pages,
        )
        < 1
    ):
        raise ValueError("corpus preparation budgets must be positive")
    questions = load_frozen_trivia_questions(args.iid_population)
    require_fresh_output_directory(args.private_output_dir)
    started = time.monotonic()
    plans: dict[str, CodexKnowledgePlan] = {}
    if args.codex_plans_input is not None:
        plans = _load_plans(args.codex_plans_input)
        print(
            json.dumps(
                {"phase": "codex", "status": "reused", "completed": len(plans)},
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        with ThreadPoolExecutor(max_workers=args.codex_concurrency) as executor:
            futures = {
                executor.submit(
                    _generate_one,
                    task_id=task_id,
                    question=question,
                    root=args.private_output_dir / "codex",
                    ordinal=ordinal,
                    model=args.codex_model,
                    reasoning_effort=args.codex_reasoning_effort,
                    maximum_attempts=args.codex_maximum_attempts,
                ): task_id
                for ordinal, (task_id, question) in enumerate(questions)
            }
            completed = 0
            for future in as_completed(futures):
                plan = future.result()
                plans[plan.task_id] = plan
                completed += 1
                elapsed = max(time.monotonic() - started, 0.001)
                rate = completed / elapsed
                eta = (len(questions) - completed) / rate if rate else None
                print(
                    json.dumps(
                        {
                            "phase": "codex",
                            "completed": completed,
                            "planned": len(questions),
                            "items_per_minute": 60 * rate,
                            "eta_seconds": eta,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    if set(plans) != {task_id for task_id, _ in questions}:
        raise RuntimeError("Codex planning did not cover the frozen population")
    plan_path = args.private_output_dir / "codex-plans.jsonl"
    with plan_path.open("w", encoding="utf-8") as stream:
        for task_id, _question in questions:
            plan = plans[task_id]
            stream.write(
                json.dumps(
                    {
                        "task_id": task_id,
                        "background_note": plan.background_note,
                        "queries": list(plan.queries),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    wikipedia = WikipediaClient(
        user_agent="SKILLEV-new-TriviaQA-research/1.0 (https://github.com/YJLi-new/SKILLEV-new)"
    )
    passages: list[CorpusPassage] = []
    provenance: list[dict[str, object]] = []
    wikipedia_page_association_count = 0
    wiki_started = time.monotonic()
    for ordinal, (task_id, _question) in enumerate(questions, start=1):
        plan = plans[task_id]
        pages = select_wikipedia_pages(
            wikipedia,
            plan,
            maximum_pages=args.maximum_wikipedia_pages,
        )
        task_passages = passages_for_task(plan, pages)
        passages.extend(task_passages)
        wikipedia_page_association_count += len(pages)
        provenance.append(
            {
                "task_id": task_id,
                "pages": [
                    {
                        "page_id": page.page_id,
                        "revision_id": page.revision_id,
                        "revision_timestamp": page.revision_timestamp,
                        "title": page.title,
                        "canonical_url": page.canonical_url,
                    }
                    for page in pages
                ],
                "passage_count": len(task_passages),
            }
        )
        elapsed = max(time.monotonic() - wiki_started, 0.001)
        rate = ordinal / elapsed
        print(
            json.dumps(
                {
                    "phase": "wikipedia",
                    "completed": ordinal,
                    "planned": len(questions),
                    "items_per_minute": 60 * rate,
                    "eta_seconds": (len(questions) - ordinal) / rate,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    database_path = args.private_output_dir / "trivia-search.sqlite3"
    build_trivia_search_database(database_path, tuple(passages))
    manifest = {
        "format": DATABASE_FORMAT,
        "codex_profile": CODEX_PLAN_PROFILE,
        "codex_model": args.codex_model,
        "codex_reasoning_effort": args.codex_reasoning_effort,
        "task_count": len(questions),
        "passage_count": len(passages),
        "codex_note_count": len(plans),
        "wikipedia_page_association_count": wikipedia_page_association_count,
        "maximum_wikipedia_pages_per_task": args.maximum_wikipedia_pages,
        "tasks": provenance,
    }
    (args.private_output_dir / "corpus-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "task_count": len(questions),
                "passage_count": len(passages),
                "wikipedia_page_association_count": manifest["wikipedia_page_association_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
