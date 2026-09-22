"""Freeze the paper's five static IID benchmarks into one answer-private curve manifest.

Sources (pinned files, downloaded once): HotpotQA (distractor train rows already in
data/curve_benchmarks), NQ-Open train, MedQA-USMLE 4-option train, AIME 2026 (all 30),
MBPP+ (EvalPlus). Items are chosen by stable SHA-256 rank of "<kind>:<source id>", so
the selection is deterministic. Prompts never contain answers, hidden tests or code;
verifier-only fields stay in the row for the grader.

Usage: python build_paper_benchmarks.py <raw_dir> <curve_manifest> <out_jsonl> [--per-kind 64]
Needs pyarrow for the parquet sources.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def _rank(kind: str, source_id: str) -> str:
    return hashlib.sha256(f"{kind}:{source_id}".encode()).hexdigest()


def _take(kind: str, items: list[tuple[str, dict]], count: int) -> list[dict]:
    chosen = sorted(items, key=lambda item: _rank(kind, item[0]))[:count]
    return [{**row, "kind": kind, "source_id": sid, "task_id": f"{kind}/{sid}", "split": "train"} for sid, row in chosen]


def hotpotqa(manifest: Path, count: int) -> list[dict]:
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["kind"] == "hotpotqa" and r["split"] == "train"]
    return [{**r, "task_id": f"hotpotqa/{r['source_id']}"} for r in rows[:count]]


def nq_open(raw: Path, count: int) -> list[dict]:
    table = pq.read_table(raw / "nq_open_train.parquet").to_pylist()
    items = []
    for index, row in enumerate(table):
        answers = row["answer"] if isinstance(row["answer"], list) else ast.literal_eval(row["answer"])
        prompt = f"Answer the question. Give a short answer only.\nQuestion: {row['question']}"
        items.append((str(index), {"prompt": prompt, "answers": list(answers)}))
    return _take("nq_open", items, count)


def medqa(raw: Path, count: int) -> list[dict]:
    items = []
    for index, line in enumerate((raw / "medqa_train.jsonl").read_text().splitlines()):
        row = json.loads(line)
        options = row["options"] if isinstance(row["options"], dict) else ast.literal_eval(row["options"])
        listed = "\n".join(f"{letter}. {text}" for letter, text in sorted(options.items()))
        prompt = (
            "Answer the following medical question by choosing one option.\n"
            f"Question: {row['question']}\nOptions:\n{listed}\n"
            "End your reply with 'Answer: <letter>'."
        )
        items.append((str(index), {"prompt": prompt, "answer": row["answer_idx"], "options": sorted(options)}))
    return _take("medqa", items, count)


def aime_2026(raw: Path, count: int) -> list[dict]:
    items = []
    for row in pq.read_table(raw / "aime_2026.parquet").to_pylist():
        prompt = (
            "Solve the following competition math problem. The answer is an integer from 0 to 999. "
            f"Put the final answer in \\boxed{{}}.\n{row['problem']}"
        )
        items.append((str(row["problem_idx"]), {"prompt": prompt, "answer": str(int(row["answer"]))}))
    return _take("aime_2026", items, count)


def mbpp_plus(raw: Path, count: int) -> list[dict]:
    items = []
    for row in pq.read_table(raw / "mbppplus.parquet").to_pylist():
        tests = ast.literal_eval(row["test_list"]) if isinstance(row["test_list"], str) else list(row["test_list"])
        imports = ast.literal_eval(row["test_imports"]) if isinstance(row["test_imports"], str) else list(row["test_imports"])
        # Standard MBPP(+) protocol: the first base assertion names the function.
        prompt = f"{row['prompt']}\nYour code should pass this test:\n{tests[0]}"
        items.append(
            (
                str(row["task_id"]),
                {"prompt": prompt, "test_list": tests, "test_imports": imports, "plus_test": row["test"]},
            )
        )
    return _take("mbpp_plus", items, count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("curve_manifest", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--per-kind", type=int, default=64)
    args = parser.parse_args()
    rows = (
        hotpotqa(args.curve_manifest, args.per_kind)
        + nq_open(args.raw, args.per_kind)
        + medqa(args.raw, args.per_kind)
        + aime_2026(args.raw, args.per_kind)
        + mbpp_plus(args.raw, args.per_kind)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    (args.out.parent / "metadata.json").write_text(
        json.dumps({"format": "evosteer-paper-benchmarks@1", "counts": counts, "manifest_sha256": digest}, indent=1)
    )
    print(counts, digest[:16])


if __name__ == "__main__":
    main()
