from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from pathlib import Path

PER_KIND = 128

RAW_FILES = {
    "hotpotqa_distractor_validation.parquet": (
        "hotpotqa/hotpot_qa@1908d6afbbead072334abe2965f91bd2709910ab:distractor/validation-00000-of-00001.parquet",
        "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6",
    ),
    "nq_open_validation.parquet": (
        "google-research-datasets/nq_open@5dd9790a83002ad084ddeb7c420dc716852c6f28:nq_open/validation-00000-of-00001.parquet",
        "b074bed0bccb56fa1551a8ac1c9c51ce89bc11c7fbb6a9c713b2c33a98531e12",
    ),
    "medqa_test.jsonl": (
        "GBaker/MedQA-USMLE-4-options@0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18:phrases_no_exclude_test.jsonl",
        "c3b905ccfa66152dc25afbcb2c10e86c0bdb208824f0658fcdb2c040f60a2beb",
    ),
    "mbppplus.parquet": (
        "evalplus/mbppplus@b2d74c91837c3f2a20c1299ae98133cbe7cfa077:data/test-00000-of-00001-d5781c9c51e02795.parquet",
        "dc20030b3788fccf617444edcb34138ef13d7e4fafd17bfcb8c1279dbb12399b",
    ),
    "aime_2026.parquet": (
        "MathArena/aime_2026@d2de22f3c656b4f56cf8981212186377d1e23bc3:data/train-00000-of-00001.parquet",
        "d91db799651b4cc1f0734f52792a695c9cc60dac342524b3d8e5b2ff31c3e957",
    ),
}

ORIGIN = {
    "hotpotqa": "hotpotqa/hotpot_qa/distractor/validation",
    "nq_open": "google-research-datasets/nq_open/nq_open/validation",
    "medqa": "GBaker/MedQA-USMLE-4-options/phrases_no_exclude/test",
    "mbpp_plus": "evalplus/mbppplus/default/test",
    "aime_2026": "MathArena/aime_2026/default/train",
    "alfworld": "alfworld/json_2.1.1/valid_unseen",
}

MBPP_PLUS_TRAINING_IDS = frozenset(
    "57 67 71 84 95 101 109 111 123 138 233 244 260 392 406 410 446 451 572 579 "
    "608 611 614 626 631 633 741 765 777 786 787 791 801".split()
)

ALFWORLD_TASK_TYPES = frozenset(
    {
        "pick_and_place_simple",
        "look_at_obj_in_light",
        "pick_clean_then_place_in_recep",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_two_obj_and_place",
    }
)

VACUOUS_FIX = "vacuous-assertion@1"
ASSERT_LINE = 'assert exact_match, f"out: {out}, exp: {exp}"'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_raw(raw: Path, names: list[str]) -> None:
    for name in names:
        path = raw / name
        if not path.is_file():
            source, _ = RAW_FILES[name]
            raise SystemExit(f"missing {path}; download {source}")
        found = sha256_file(path)
        if found != RAW_FILES[name][1]:
            raise SystemExit(f"{path}: sha256 {found} != {RAW_FILES[name][1]}")


def read_parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def rank(kind: str, source_id: str) -> str:
    return hashlib.sha256(f"test:{kind}:{source_id}".encode()).hexdigest()


def take(kind: str, items: list[tuple[str, dict]], count: int) -> list[dict]:
    ranked = sorted(items, key=lambda item: rank(kind, item[0]))[:count]
    rows = []
    for source_id, fields in ranked:
        rows.append(
            {
                **fields,
                "kind": kind,
                "task_type": kind,
                "source_id": source_id,
                "task_id": f"{kind}/test-{source_id}",
                "split": "test",
                "origin": ORIGIN[kind],
            }
        )
    return rows


def hotpotqa(raw: Path, count: int) -> list[dict]:
    items = []
    for row in read_parquet(raw / "hotpotqa_distractor_validation.parquet"):
        context = "\n".join(
            f"[{title}] " + " ".join(sentences)
            for title, sentences in zip(row["context"]["title"], row["context"]["sentences"], strict=True)
        )
        prompt = f"Answer the question using the passages. Give a short answer only.\nQuestion: {row['question']}\nPassages:\n{context}"
        items.append((row["id"], {"prompt": prompt, "answer": row["answer"]}))
    return take("hotpotqa", items, count)


def nq_open(raw: Path, count: int) -> list[dict]:
    items = []
    for index, row in enumerate(read_parquet(raw / "nq_open_validation.parquet")):
        answers = row["answer"] if isinstance(row["answer"], list) else ast.literal_eval(row["answer"])
        prompt = f"Answer the question. Give a short answer only.\nQuestion: {row['question']}"
        items.append((str(index), {"prompt": prompt, "answer": list(answers), "answers": list(answers)}))
    return take("nq_open", items, count)


def medqa(raw: Path, count: int) -> list[dict]:
    items = []
    lines = (raw / "medqa_test.jsonl").read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        row = json.loads(line)
        options = row["options"] if isinstance(row["options"], dict) else ast.literal_eval(row["options"])
        listed = "\n".join(f"{letter}. {text}" for letter, text in sorted(options.items()))
        prompt = (
            "Answer the following medical question by choosing one option.\n"
            f"Question: {row['question']}\nOptions:\n{listed}\n"
            "End your reply with 'Answer: <letter>'."
        )
        items.append((str(index), {"prompt": prompt, "answer": row["answer_idx"], "options": sorted(options)}))
    return take("medqa", items, count)


def checks(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, (ast.Assert, ast.Raise)):
            return True
        if isinstance(child, ast.Call):
            func = child.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name.startswith("assert"):
                return True
    return False


def patch_vacuous_assertion(plus_test: str) -> str | None:
    tree = ast.parse(plus_test)
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef) and node.name == "assertion") or checks(node):
            continue
        stored = {
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
        }
        if "exact_match" not in stored:
            raise ValueError("assertion() checks nothing and computes no exact_match")
        lines = plus_test.split("\n")
        lines.insert(node.end_lineno, " " * node.body[0].col_offset + ASSERT_LINE)
        return "\n".join(lines)
    return None


def patch_mbpp_row(row: dict) -> dict:
    patched = patch_vacuous_assertion(row["plus_test"])
    if patched is None:
        return row
    record = {
        "field": "plus_test",
        "fix": VACUOUS_FIX,
        "detail": f"assertion() computed exact_match without asserting it; appended '{ASSERT_LINE}'",
        "sha256_before": hashlib.sha256(row["plus_test"].encode("utf-8")).hexdigest(),
    }
    return {**row, "plus_test": patched, "patched": [*row.get("patched", []), record]}


def mbpp_plus(raw: Path, count: int) -> list[dict]:
    items = []
    for row in read_parquet(raw / "mbppplus.parquet"):
        source_id = str(row["task_id"])
        if source_id in MBPP_PLUS_TRAINING_IDS:
            continue
        tests = ast.literal_eval(row["test_list"]) if isinstance(row["test_list"], str) else list(row["test_list"])
        imports = ast.literal_eval(row["test_imports"]) if isinstance(row["test_imports"], str) else list(row["test_imports"])
        prompt = f"{row['prompt']}\nYour code should pass this test:\n{tests[0]}"
        item = {"prompt": prompt, "answer": None, "test_list": tests, "test_imports": imports, "plus_test": row["test"]}
        items.append((source_id, patch_mbpp_row(item)))
    return take("mbpp_plus", items, count)


def aime_2026(raw: Path) -> list[dict]:
    rows = read_parquet(raw / "aime_2026.parquet")
    if sorted(int(row["problem_idx"]) for row in rows) != list(range(1, 31)):
        raise SystemExit("expected the 30 AIME 2026 problems (problem_idx 1..30)")
    out = []
    for row in sorted(rows, key=lambda row: int(row["problem_idx"])):
        source_id = f"2026_{int(row['problem_idx'])}"
        prompt = (
            "Solve the following competition math problem. The answer is an integer from 0 to 999. "
            f"Put the final answer in \\boxed{{}}.\n{row['problem']}"
        )
        out.append(
            {
                "kind": "aime_2026",
                "task_type": "aime_2026",
                "source_id": source_id,
                "task_id": f"aime_2026/test-{source_id}",
                "split": "test",
                "prompt": prompt,
                "answer": str(int(row["answer"])),
                "origin": ORIGIN["aime_2026"],
            }
        )
    return out


def alfworld_catalog(data: Path) -> list[str]:
    root = data / "json_2.1.1" / "valid_unseen"
    if not root.is_dir():
        raise SystemExit(f"missing {root}; set ALFWORLD_DATA to the extracted ALFWorld data")
    files = []
    for directory, _, names in os.walk(root):
        if "traj_data.json" not in names or "movable" in directory or "Sliced" in directory:
            continue
        traj = json.loads((Path(directory) / "traj_data.json").read_text(encoding="utf-8"))
        if traj["task_type"] not in ALFWORLD_TASK_TYPES:
            continue
        game = Path(directory) / "game.tw-pddl"
        if not game.is_file():
            continue
        if not json.loads(game.read_text(encoding="utf-8")).get("solvable", False):
            continue
        files.append(str(game))
    files.sort()
    return [path[path.index("json_2.1.1/"):] for path in files]


def alfworld(data: Path, count: int) -> list[dict]:
    files = alfworld_catalog(data)
    order = sorted(range(len(files)), key=lambda i: hashlib.sha256(f"alfworld:valid_unseen:{files[i]}".encode()).hexdigest())
    return [
        {
            "kind": "alfworld",
            "task_type": "alfworld",
            "game_type": files[i].split("/")[2].split("-")[0],
            "source_id": files[i],
            "task_id": f"alfworld/valid_unseen/{i:06d}",
            "split": "test",
            "alfworld_split": "valid_unseen",
            "index": i,
            "prompt": None,
            "answer": None,
            "origin": ORIGIN["alfworld"],
        }
        for i in order[:count]
    ]


def write(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return sha256_file(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild the EvoSteer IID test sets (data/evosteer/test/iid/<kind>.jsonl).")
    parser.add_argument("--raw", type=Path, help="directory holding the raw files listed in RAW_FILES")
    parser.add_argument("--alfworld-data", type=Path, default=os.environ.get("ALFWORLD_DATA"))
    parser.add_argument("--out", type=Path, default=Path("data/evosteer/test/iid"))
    parser.add_argument("--only", nargs="+", choices=sorted(ORIGIN), default=sorted(ORIGIN))
    parser.add_argument("--per-kind", type=int, default=PER_KIND)
    parser.add_argument("--no-sha-check", action="store_true")
    args = parser.parse_args()
    needed = {
        "hotpotqa": "hotpotqa_distractor_validation.parquet",
        "nq_open": "nq_open_validation.parquet",
        "medqa": "medqa_test.jsonl",
        "mbpp_plus": "mbppplus.parquet",
        "aime_2026": "aime_2026.parquet",
    }
    text_kinds = [kind for kind in args.only if kind in needed]
    if text_kinds:
        if args.raw is None:
            parser.error("--raw is required for " + ", ".join(text_kinds))
        if not args.no_sha_check:
            check_raw(args.raw, [needed[kind] for kind in text_kinds])
    if "alfworld" in args.only and args.alfworld_data is None:
        parser.error("--alfworld-data (or ALFWORLD_DATA) is required for alfworld")
    builders = {
        "hotpotqa": lambda: hotpotqa(args.raw, args.per_kind),
        "nq_open": lambda: nq_open(args.raw, args.per_kind),
        "medqa": lambda: medqa(args.raw, args.per_kind),
        "mbpp_plus": lambda: mbpp_plus(args.raw, args.per_kind),
        "aime_2026": lambda: aime_2026(args.raw),
        "alfworld": lambda: alfworld(Path(args.alfworld_data), args.per_kind),
    }
    for kind in args.only:
        rows = builders[kind]()
        digest = write(args.out / f"{kind}.jsonl", rows)
        print(f"{kind}\t{len(rows)}\t{digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
