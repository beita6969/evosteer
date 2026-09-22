"""Owner-selected GPQA Diamond biology/organic population, not a health subset.

Only private artifacts contain record IDs, choices, references or source rows.
Selection uses IDs and domain metadata; answer fields cannot influence sampling.
Option shuffling and exact final-choice scoring follow the existing GPQA lane.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

BENCHMARK = "gpqa-diamond-bioorganic"


def select_bioorganic(
    rows: list[dict[str, Any]],
    biology_ids: list[str],
    organic_ids: list[str],
    *,
    count: int,
    seed: int,
) -> list[tuple[int, dict[str, Any]]]:
    """Keep all 19 Biology records, then sample Organic by sorted Record ID."""
    records = [row["Record ID"] for row in rows]
    if len(rows) != 198 or len(set(records)) != 198:
        raise ValueError("GPQA Diamond requires 198 distinct original Record IDs")
    biology, organic = set(biology_ids), set(organic_ids)
    if len(biology_ids) != 19 or len(biology) != 19 or len(organic_ids) != 72 or len(organic) != 72:
        raise ValueError("the owner population requires 19 Biology and 72 Organic IDs")
    if biology & organic or not 19 <= count <= 91:
        raise ValueError("BioOrganic strata overlap or the requested count is unsupported")
    actual_biology = {row["Record ID"] for row in rows if row["High-level domain"] == "Biology"}
    actual_organic = {
        row["Record ID"]
        for row in rows
        if row["High-level domain"] == "Chemistry" and row["Subdomain"] == "Organic Chemistry"
    }
    if biology != actual_biology or organic != actual_organic:
        raise ValueError("the complete owner allowlist does not match Diamond domain metadata")
    rng = random.Random(seed)  # noqa: S311 -- result-blind frozen evaluation sample
    chosen = biology | set(rng.sample(sorted(organic), count - len(biology)))
    return sorted(
        ((index, row) for index, row in enumerate(rows) if row["Record ID"] in chosen),
        key=lambda pair: pair[1]["Record ID"],
    )


def project_gpqa(index: int, row: dict[str, Any], *, seed: int) -> dict[str, Any]:
    record_id = row["Record ID"]
    question = row["Question"]
    choices = [row["Correct Answer"], *(row[f"Incorrect Answer {n}"] for n in range(1, 4))]
    if not all(isinstance(text, str) and text.strip() for text in [record_id, question, *choices]):
        raise ValueError("GPQA question, identity and all four options must be nonempty text")
    # Released records may repeat a distractor. Preserve all four source slots;
    # deduplicating or dropping the record would change the authorized panel.
    order = list(range(4))
    random.Random(f"gpqa:{record_id}:{seed}").shuffle(order)  # noqa: S311 -- frozen option order
    labels = "ABCD"
    return {
        "task_id": f"{BENCHMARK}:{record_id}",
        "public": {
            "question": question,
            "options": "\n".join(
                f"{labels[position]}. {choices[source]}" for position, source in enumerate(order)
            ),
        },
        "target": {"correct_label": labels[order.index(0)]},
        "source_metadata": {
            "record_id": record_id,
            "source_index": index,
            "high_level_domain": row["High-level domain"],
            "subdomain": row["Subdomain"],
            "option_source_order": order,
        },
    }


def export_bioorganic(
    source: dict[str, Any], directory: Path, *, count: int, seed: int
) -> dict[str, Any]:
    paths = source["paths"]
    if len(paths) != 1 or Path(paths[0]).name != "gpqa_diamond.csv":
        raise ValueError("BioOrganic must read gpqa_diamond.csv, not main or extended")
    with Path(paths[0]).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    allowlist = json.loads(Path(source["allowlist_path"]).read_text())
    selected = select_bioorganic(
        rows,
        allowlist["biology_record_ids"],
        allowlist["organic_record_ids"],
        count=count,
        seed=seed,
    )
    path = directory / f"{BENCHMARK}-private.jsonl"
    with path.open("x") as out:
        for index, row in selected:
            out.write(json.dumps(project_gpqa(index, row, seed=seed), ensure_ascii=False) + "\n")
    return {
        "ood_sources": {BENCHMARK: str(path.resolve())},
        "evaluation_sample_counts": {BENCHMARK: len(selected)},
        "ood_provenance": {
            BENCHMARK: {
                **source.get("provenance", {}),
                "source_subset": "gpqa_diamond",
                "source_count": 198,
                "population_name": "GPQA-Diamond-BioOrganic-91",
                "panel_name": f"GPQA-Diamond-BioOrganic-{count}",
                "population": 91,
                "sample_count": len(selected),
                "sample_composition": {"Biology": 19, "Organic Chemistry": count - 19},
                "seed": seed,
                "selection": "all-biology-plus-random-sample-of-sorted-organic-record-ids",
                "option_order": "per-record-id-seeded-shuffle-correct-and-three-incorrect",
                "health_subset": False,
            }
        },
    }
