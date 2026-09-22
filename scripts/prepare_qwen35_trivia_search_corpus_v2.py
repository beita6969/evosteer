#!/usr/bin/env python3
"""Build the private detailed Codex plus Wikipedia TriviaQA search database."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from skillev_private.direct_reference.journal import require_fresh_output_directory
from skillev_private.direct_reference.trivia_search import (
    CODEX_DETAILED_PLAN_PROFILE,
    DATABASE_FORMAT_V2,
    CodexDetailedKnowledgePlan,
    CorpusPassage,
    WikipediaClient,
    WikipediaPage,
    build_trivia_search_database,
    detailed_passages_for_task,
    generate_codex_detailed_knowledge_plan,
    load_frozen_trivia_labels,
    load_frozen_trivia_questions,
    remove_evaluator_labels,
    select_wikipedia_pages_for_queries,
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
    parser.add_argument("--maximum-wikipedia-pages", type=int, default=12)
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
) -> CodexDetailedKnowledgePlan:
    for attempt in range(1, maximum_attempts + 1):
        try:
            return generate_codex_detailed_knowledge_plan(
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


def _load_plans(path: Path) -> dict[str, CodexDetailedKnowledgePlan]:
    plans: dict[str, CodexDetailedKnowledgePlan] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if type(value) is not dict or set(value) != {
                "task_id",
                "research_dossier",
                "queries",
                "key_entities",
            }:
                raise ValueError("detailed Codex plan input has an incompatible row")
            raw_queries = value["queries"]
            raw_entities = value["key_entities"]
            if type(raw_queries) is not list or type(raw_entities) is not list:
                raise ValueError("detailed Codex plan input has incompatible lists")
            plan = CodexDetailedKnowledgePlan(
                str(value["task_id"]),
                str(value["research_dossier"]),
                tuple(str(item) for item in raw_queries),
                tuple(str(item) for item in raw_entities),
            )
            if plan.task_id in plans:
                raise ValueError("detailed Codex plan input repeats a task ID")
            plans[plan.task_id] = plan
    return plans


def _write_plans(
    path: Path, questions: tuple[tuple[str, str], ...], plans: dict[str, CodexDetailedKnowledgePlan]
) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for task_id, _question in questions:
            plan = plans[task_id]
            stream.write(
                json.dumps(
                    {
                        "task_id": task_id,
                        "research_dossier": plan.research_dossier,
                        "queries": list(plan.queries),
                        "key_entities": list(plan.key_entities),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


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
        raise ValueError("detailed corpus preparation budgets must be positive")
    questions = load_frozen_trivia_questions(args.iid_population)
    expected_ids = {task_id for task_id, _question in questions}
    require_fresh_output_directory(args.private_output_dir)
    started = time.monotonic()
    if args.codex_plans_input is not None:
        plans = _load_plans(args.codex_plans_input)
        print(
            json.dumps(
                {"phase": "codex-detailed", "status": "reused", "completed": len(plans)},
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        plans: dict[str, CodexDetailedKnowledgePlan] = {}
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
                print(
                    json.dumps(
                        {
                            "phase": "codex-detailed",
                            "completed": completed,
                            "planned": len(questions),
                            "items_per_minute": 60 * rate,
                            "eta_seconds": (len(questions) - completed) / rate,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    if set(plans) != expected_ids:
        raise RuntimeError("detailed Codex planning does not exactly cover the frozen population")
    _write_plans(args.private_output_dir / "codex-detailed-plans.jsonl", questions, plans)

    wikipedia = WikipediaClient(
        user_agent="SKILLEV-new-TriviaQA-detailed-research/2.0 "
        "(https://github.com/YJLi-new/SKILLEV-new)",
        maximum_attempts=6,
    )
    page_cache: dict[int, WikipediaPage] = {}
    passages: list[CorpusPassage] = []
    provenance: list[dict[str, object]] = []
    page_associations = 0
    wiki_started = time.monotonic()
    for ordinal, (task_id, _question) in enumerate(questions, start=1):
        plan = plans[task_id]
        pages = select_wikipedia_pages_for_queries(
            wikipedia,
            plan.queries,
            maximum_pages=args.maximum_wikipedia_pages,
            page_cache=page_cache,
        )
        task_passages = detailed_passages_for_task(plan, pages)
        passages.extend(task_passages)
        page_associations += len(pages)
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
                    "phase": "wikipedia-detailed",
                    "completed": ordinal,
                    "planned": len(questions),
                    "items_per_minute": 60 * rate,
                    "eta_seconds": (len(questions) - ordinal) / rate,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    database_path = args.private_output_dir / "trivia-search-detailed.sqlite3"
    sanitized_passages = remove_evaluator_labels(
        tuple(passages), load_frozen_trivia_labels(args.iid_population)
    )
    build_trivia_search_database(
        database_path,
        sanitized_passages,
        database_format=DATABASE_FORMAT_V2,
    )
    manifest = {
        "format": DATABASE_FORMAT_V2,
        "codex_profile": CODEX_DETAILED_PLAN_PROFILE,
        "codex_model": args.codex_model,
        "codex_reasoning_effort": args.codex_reasoning_effort,
        "task_count": len(questions),
        "passage_count": len(sanitized_passages),
        "codex_dossier_count": len(plans),
        "wikipedia_page_association_count": page_associations,
        "unique_wikipedia_page_count": len(page_cache),
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
                "passage_count": len(sanitized_passages),
                "wikipedia_page_association_count": page_associations,
                "unique_wikipedia_page_count": len(page_cache),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
