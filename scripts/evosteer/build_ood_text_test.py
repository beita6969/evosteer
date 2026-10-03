from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

COUNT = 128
LETTERS = ("A", "B", "C", "D")
GPQA_SOURCE = {
    "dataset": "Idavidrein/gpqa",
    "config": "gpqa_diamond",
    "file": "gpqa_diamond.csv",
    "revision": "83022cefff930aea54f654c0b282e74b9eeda5c6",
    "git_blob_sha1": "7589e3e467d69a1dceb126a60c4108d6d4f1d166",
    "url": "https://huggingface.co/datasets/Idavidrein/gpqa/blob/83022cefff930aea54f654c0b282e74b9eeda5c6/gpqa_diamond.csv",
}
RAW_SHA256 = {
    "triviaqa_rc_nocontext_validation.parquet": "48a5005c0eb4f8a4ae5b9868644297fd5bf1e694aeb3dc9c8ab958cac0d5b201",
    "musique_validation.parquet": "2d671592a7674a0920d2a595df2e13acd74c8a80997f763ca81484a273f8adfe",
    "math_hard_test.parquet": "90ba4ce7f108cd511c8eb716ea7ca906d5dc1621f5e804565765a368ca02ab01",
    "gpqa_diamond.csv": "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _check(path: Path, expected: str) -> None:
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    found = _sha256(path)
    if found != expected:
        raise SystemExit(f"sha256 mismatch for {path}: {found} != {expected}")


def _rank(kind: str, source_id: str) -> str:
    return hashlib.sha256(f"test:{kind}:{source_id}".encode()).hexdigest()


def _select(kind: str, task_type: str, origin: str, items: list[tuple[str, dict]]) -> list[dict]:
    ids = [sid for sid, _ in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{kind}: duplicate source ids")
    chosen = sorted(items, key=lambda item: _rank(kind, item[0]))[:COUNT]
    if len(chosen) != COUNT:
        raise ValueError(f"{kind}: only {len(chosen)} candidates")
    return [
        {
            **row,
            "kind": kind,
            "task_type": task_type,
            "source_id": sid,
            "task_id": f"{kind}/test-{sid}",
            "split": "test",
            "origin": origin,
        }
        for sid, row in chosen
    ]


def _dedupe(values: list[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return seen


def triviaqa(raw: Path) -> list[dict]:
    import pyarrow.parquet as pq

    by_id: dict[str, dict] = {}
    conflicting: set[str] = set()
    for row in pq.read_table(raw / "triviaqa_rc_nocontext_validation.parquet").to_pylist():
        answer = row["answer"]
        aliases = _dedupe([answer["normalized_value"], *answer["normalized_aliases"]])
        item = {
            "prompt": f"Answer the question. Give a short answer only.\nQuestion: {row['question']}",
            "answer": aliases,
            "answers": aliases,
        }
        if by_id.setdefault(row["question_id"], item) != item:
            conflicting.add(row["question_id"])
    items = sorted((qid, item) for qid, item in by_id.items() if qid not in conflicting)
    return _select("triviaqa", "nq_open", "mandarjoshi/trivia_qa/rc.nocontext/validation", items)


def musique(raw: Path) -> list[dict]:
    import pyarrow.parquet as pq

    items = []
    for row in pq.read_table(raw / "musique_validation.parquet").to_pylist():
        if not row["answerable"]:
            continue
        paragraphs = sorted(row["paragraphs"], key=lambda p: p["idx"])
        context = "\n".join(f"[{p['title']}] {p['paragraph_text']}" for p in paragraphs)
        prompt = f"Answer the question using the passages. Give a short answer only.\nQuestion: {row['question']}\nPassages:\n{context}"
        answers = _dedupe([row["answer"], *(row["answer_aliases"] or [])])
        items.append((row["id"], {"prompt": prompt, "answer": row["answer"], "answers": answers}))
    return _select("musique", "hotpotqa", "dgslibisey/MuSiQue/default/validation (musique_ans_v1.0_dev)", items)


def last_boxed(text: str) -> str | None:
    start = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if start < 0:
        return None
    rest = text[start:]
    if rest.startswith("\\boxed "):
        return rest[len("\\boxed ") :].split("$")[0].strip()
    open_at = rest.find("{")
    if open_at < 0:
        return None
    depth = 0
    for index in range(open_at, len(rest)):
        if rest[index] == "{":
            depth += 1
        elif rest[index] == "}":
            depth -= 1
            if depth == 0:
                return rest[open_at + 1 : index].strip()
    return None


MATH_SUBJECTS = {
    "Algebra": "algebra",
    "Counting & Probability": "counting_and_probability",
    "Geometry": "geometry",
    "Intermediate Algebra": "intermediate_algebra",
    "Number Theory": "number_theory",
    "Prealgebra": "prealgebra",
    "Precalculus": "precalculus",
}


def math_hard(raw: Path) -> list[dict]:
    import pyarrow.parquet as pq

    items = []
    counters: dict[str, int] = {}
    for row in pq.read_table(raw / "math_hard_test.parquet").to_pylist():
        if row["level"] != "Level 5":
            raise ValueError("math_hard: row outside Level 5")
        subject = MATH_SUBJECTS[row["type"]]
        index = counters.get(subject, 0)
        counters[subject] = index + 1
        answer = last_boxed(row["solution"])
        if not answer:
            raise ValueError(f"math_hard: no boxed answer in {subject}/{index}")
        prompt = f"Solve the following competition math problem. Put the final answer in \\boxed{{}}.\n{row['problem']}"
        items.append((f"{subject}_{index:03d}", {"prompt": prompt, "answer": answer}))
    return _select("math_hard", "aime_2026", "lighteval/MATH-Hard/default/test", items)


def gpqa_seed(record_id: str) -> int:
    return int(hashlib.sha256(f"options:gpqa_diamond:{record_id}".encode()).hexdigest()[:8], 16)


def gpqa_order(seed: int) -> list[int]:
    return sorted(range(4), key=lambda k: hashlib.sha256(f"{seed}:{k}".encode()).hexdigest())


def gpqa_ids(record_ids: list[str], raw_sha256: str | None) -> dict:
    chosen = sorted(record_ids, key=lambda rid: _rank("gpqa_diamond", rid))[:COUNT]
    items = []
    for rid in chosen:
        items.append({"record_id": rid, "option_seed": gpqa_seed(rid)})
    return {
        "kind": "gpqa_diamond",
        "task_type": "medqa",
        "source": {**GPQA_SOURCE, "rows": len(record_ids), "sha256": raw_sha256},
        "selection": "first 128 Record IDs by ascending sha256('test:gpqa_diamond:<Record ID>')",
        "option_seed": "int(sha256('options:gpqa_diamond:<Record ID>').hexdigest()[:8], 16)",
        "option_order": "options [Correct Answer, Incorrect Answer 1, Incorrect Answer 2, Incorrect Answer 3] sorted by sha256('<seed>:<k>') for k=0..3 give letters A-D",
        "items": items,
    }


def _gpqa_csv(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = {row["Record ID"].strip(): row for row in csv.DictReader(stream)}
    if len(rows) != 198:
        raise ValueError(f"gpqa_diamond: expected 198 records, got {len(rows)}")
    return rows


def gpqa(path: Path, ids_path: Path) -> list[dict]:
    spec = json.loads(ids_path.read_text())
    _check(path, spec["source"]["sha256"])
    records = _gpqa_csv(path)
    expected = gpqa_ids(sorted(records), spec["source"]["sha256"])
    if expected["items"] != spec["items"]:
        raise ValueError("gpqa_diamond: Record IDs or option seeds differ from ids.json")
    rows = []
    for item in spec["items"]:
        record = records[item["record_id"]]
        texts = [
            record["Correct Answer"],
            record["Incorrect Answer 1"],
            record["Incorrect Answer 2"],
            record["Incorrect Answer 3"],
        ]
        order = gpqa_order(item["option_seed"])
        listed = "\n".join(f"{letter}. {texts[k].strip()}" for letter, k in zip(LETTERS, order))
        prompt = (
            "Answer the following question by choosing one option.\n"
            f"Question: {record['Question'].strip()}\nOptions:\n{listed}\n"
            "End your reply with 'Answer: <letter>'."
        )
        rid = item["record_id"]
        rows.append(
            {
                "answer": LETTERS[order.index(0)],
                "kind": "gpqa_diamond",
                "options": list(LETTERS),
                "origin": "Idavidrein/gpqa/gpqa_diamond/train",
                "prompt": prompt,
                "source_id": rid,
                "split": "test",
                "task_id": f"gpqa_diamond/test-{rid}",
                "task_type": "medqa",
            }
        )
    return rows


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(path.name, len(rows), hashlib.sha256(path.read_bytes()).hexdigest())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the TriviaQA, MuSiQue, GPQA Diamond and MATH-Hard OOD test sets.")
    sub = parser.add_subparsers(dest="part", required=True)
    text = sub.add_parser("text")
    text.add_argument("raw", type=Path)
    text.add_argument("out", type=Path)
    gp = sub.add_parser("gpqa")
    gp.add_argument("csv", type=Path)
    gp.add_argument("ids", type=Path)
    gp.add_argument("out", type=Path)
    gi = sub.add_parser("gpqa-ids")
    gi.add_argument("csv", type=Path)
    gi.add_argument("ids", type=Path)
    args = parser.parse_args()
    if args.part == "text":
        for name in ("triviaqa_rc_nocontext_validation.parquet", "musique_validation.parquet", "math_hard_test.parquet"):
            _check(args.raw / name, RAW_SHA256[name])
        _write(args.out / "triviaqa.jsonl", triviaqa(args.raw))
        _write(args.out / "musique.jsonl", musique(args.raw))
        _write(args.out / "math_hard.jsonl", math_hard(args.raw))
    elif args.part == "gpqa":
        _write(args.out, gpqa(args.csv, args.ids))
    else:
        _check(args.csv, RAW_SHA256["gpqa_diamond.csv"])
        digest = _sha256(args.csv)
        spec = gpqa_ids(sorted(_gpqa_csv(args.csv)), digest)
        args.ids.write_text(json.dumps(spec, indent=1) + "\n")
        print(args.ids.name, len(spec["items"]))


if __name__ == "__main__":
    main()
