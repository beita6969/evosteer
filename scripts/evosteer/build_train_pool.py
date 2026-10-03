from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

PER_KIND = 1280
KINDS = ("hotpotqa", "nq_open", "medqa", "aime_2026", "mbpp_plus", "alfworld")

RAW_FILES = {
    "hotpotqa_train_0.parquet": (
        "hotpotqa/hotpot_qa@1908d6afbbead072334abe2965f91bd2709910ab:distractor/train-00000-of-00002.parquet",
        "76d3bb3048a7cc73c1958107c0c5872a00d7e7d00c105b81e92f6769e7822e68",
    ),
    "hotpotqa_train_1.parquet": (
        "hotpotqa/hotpot_qa@1908d6afbbead072334abe2965f91bd2709910ab:distractor/train-00001-of-00002.parquet",
        "713661628434fbb19fff7392e2e321e4ed107e3c7c7784d0690946e5f722763f",
    ),
    "nq_open_train.parquet": (
        "google-research-datasets/nq_open@5dd9790a83002ad084ddeb7c420dc716852c6f28:nq_open/train-00000-of-00001.parquet",
        "25d3a544324f900b31ebc05a3a5686bdd5b9b42738133500675c3f20eee7d83a",
    ),
    "medqa_train.jsonl": (
        "GBaker/MedQA-USMLE-4-options@0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18:phrases_no_exclude_train.jsonl",
        "3a65baf760f17c395058d6699315a7ce8aa767c50cb193fea87ceb94dd790324",
    ),
    "mbppplus.parquet": (
        "evalplus/mbppplus@b2d74c91837c3f2a20c1299ae98133cbe7cfa077:data/test-00000-of-00001-d5781c9c51e02795.parquet",
        "dc20030b3788fccf617444edcb34138ef13d7e4fafd17bfcb8c1279dbb12399b",
    ),
    "aime_1983_2024.csv": (
        "di-zhang-fdu/AIME_1983_2024@3e2cc86390666c5c756622afc0eeb9e6194496bc:AIME_Dataset_1983_2024.csv",
        "959358884aa93b30b8d85ecb80fd8b577014498c08f7569faf88f675cc3fb9f4",
    ),
    "aimo_validation_aime.parquet": (
        "AI-MO/aimo-validation-aime@13f9e12f613e720c2a2b2f345dd04b998a29494d:data/train-00000-of-00001.parquet",
        "025484a99fea498e7d0c3b0ee42afcbec0176405c19c5dbf557b9f6ca6445675",
    ),
    "aime_2025.parquet": (
        "MathArena/aime_2025@c94da77eb22bbd6439e62a323bec18493a421302:data/train-00000-of-00001.parquet",
        "9f9066ff48ad2e31f9bf1b1ac6d5e80693195f987985f2859f89dd25ffa51c2d",
    ),
}

ORIGIN = {
    "hotpotqa": "hotpotqa/hotpot_qa/distractor/train",
    "nq_open": "google-research-datasets/nq_open/nq_open/train",
    "medqa": "GBaker/MedQA-USMLE-4-options/phrases_no_exclude/train",
    "mbpp_plus": "evalplus/mbppplus/default/test",
    "alfworld": "alfworld/json_2.1.1/train",
}

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
AIME_PROMPT = "Solve the following competition math problem. The answer is an integer from 0 to 999. Put the final answer in \\boxed{{}}.\n{problem}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_raw(raw: Path, names: list[str], verify: bool) -> None:
    for name in names:
        path = raw / name
        source, expected = RAW_FILES[name]
        if not path.is_file():
            raise SystemExit(f"missing {path}; download {source}")
        if verify and expected is not None and sha256_file(path) != expected:
            raise SystemExit(f"{path}: sha256 differs from {expected}")


def read_parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^0-9a-z]+", " ", text.lower()).split())


def question_of(prompt: str) -> str:
    if "\nQuestion: " in prompt:
        body = prompt.split("\nQuestion: ", 1)[1]
        return re.split(r"\n(?:Passages|Options):\n", body, maxsplit=1)[0]
    if prompt.startswith("Solve the following"):
        return prompt.split("\n", 1)[1]
    return prompt.split("\nYour code should pass this test:", 1)[0]


def task_statement(prompt: str) -> str:
    first = re.split(r"(?<=[.?!])\s", question_of(prompt).strip(), maxsplit=1)[0]
    return re.sub(r"\d+", "", norm(first)).strip()


def grams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    words = norm(text).split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def load_tests(test_dir: Path) -> dict[str, list[dict]]:
    found: dict[str, list[dict]] = {}
    for path in sorted(p for p in test_dir.glob("*/*.jsonl") if not p.name.startswith(".")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                found.setdefault(row.get("task_type") or row["kind"], []).append(row)
    return found


def hotpotqa(raw: Path) -> list[tuple[str, dict]]:
    items = []
    for name in ("hotpotqa_train_0.parquet", "hotpotqa_train_1.parquet"):
        for row in read_parquet(raw / name):
            context = "\n".join(
                f"[{title}] " + " ".join(sentences)
                for title, sentences in zip(row["context"]["title"], row["context"]["sentences"], strict=True)
            )
            prompt = f"Answer the question using the passages. Give a short answer only.\nQuestion: {row['question']}\nPassages:\n{context}"
            if "\n" in row["answer"] or not row["answer"].strip():
                continue
            items.append((row["id"], {"prompt": prompt, "answer": row["answer"]}))
    return items


def nq_open(raw: Path) -> list[tuple[str, dict]]:
    items = []
    for index, row in enumerate(read_parquet(raw / "nq_open_train.parquet")):
        answers = row["answer"] if isinstance(row["answer"], list) else ast.literal_eval(row["answer"])
        if not answers or any("\n" in answer for answer in answers):
            continue
        prompt = f"Answer the question. Give a short answer only.\nQuestion: {row['question']}"
        items.append((str(index), {"prompt": prompt, "answer": list(answers), "answers": list(answers)}))
    return items


def medqa(raw: Path) -> list[tuple[str, dict]]:
    items = []
    for index, line in enumerate((raw / "medqa_train.jsonl").read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        options = row["options"] if isinstance(row["options"], dict) else ast.literal_eval(row["options"])
        listed = "\n".join(f"{letter}. {text}" for letter, text in sorted(options.items()))
        prompt = (
            "Answer the following medical question by choosing one option.\n"
            f"Question: {row['question']}\nOptions:\n{listed}\n"
            "End your reply with 'Answer: <letter>'."
        )
        items.append((str(index), {"prompt": prompt, "answer": row["answer_idx"], "options": sorted(options)}))
    return items


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


def patch_mbpp_row(row: dict) -> dict:
    tree = ast.parse(row["plus_test"])
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef) and node.name == "assertion") or checks(node):
            continue
        stored = {c.id for c in ast.walk(node) if isinstance(c, ast.Name) and isinstance(c.ctx, ast.Store)}
        if "exact_match" not in stored:
            raise ValueError("assertion() checks nothing and computes no exact_match")
        lines = row["plus_test"].split("\n")
        lines.insert(node.end_lineno, " " * node.body[0].col_offset + ASSERT_LINE)
        record = {
            "field": "plus_test",
            "fix": VACUOUS_FIX,
            "detail": f"assertion() computed exact_match without asserting it; appended '{ASSERT_LINE}'",
            "sha256_before": hashlib.sha256(row["plus_test"].encode("utf-8")).hexdigest(),
        }
        return {**row, "plus_test": "\n".join(lines), "patched": [record]}
    return row


def mbpp_plus(raw: Path, test_ids: set[str]) -> list[tuple[str, dict]]:
    items = []
    for row in read_parquet(raw / "mbppplus.parquet"):
        source_id = str(row["task_id"])
        if source_id in test_ids:
            continue
        tests = ast.literal_eval(row["test_list"]) if isinstance(row["test_list"], str) else list(row["test_list"])
        imports = ast.literal_eval(row["test_imports"]) if isinstance(row["test_imports"], str) else list(row["test_imports"])
        prompt = f"{row['prompt']}\nYour code should pass this test:\n{tests[0]}"
        item = {"prompt": prompt, "answer": None, "test_list": tests, "test_imports": imports, "plus_test": row["test"]}
        items.append((source_id, patch_mbpp_row(item)))
    return items


def aime(raw: Path) -> list[tuple[str, dict]]:
    found: dict[tuple[int, str, int], tuple[str, str, str]] = {}
    for row in csv.DictReader((raw / "aime_1983_2024.csv").open(encoding="utf-8")):
        year, number = int(row["Year"]), int(row["Problem Number"])
        part = (row.get("Part") or "").strip()
        found[(year, part, number)] = (row["Question"], row["Answer"].strip(), "di-zhang-fdu/AIME_1983_2024")
    for row in read_parquet(raw / "aimo_validation_aime.parquet"):
        match = re.search(r"/(\d{4})_AIME_(I{1,2})_Problems/Problem_(\d+)", row["url"])
        year, part, number = int(match[1]), match[2], int(match[3])
        found[(year, part, number)] = (row["problem"], str(row["answer"]).strip(), "AI-MO/aimo-validation-aime")
    for row in read_parquet(raw / "aime_2025.parquet"):
        index = int(row["problem_idx"])
        part, number = ("I", index) if index <= 15 else ("II", index - 15)
        found[(2025, part, number)] = (row["problem"], str(row["answer"]).strip(), "MathArena/aime_2025")
    items = []
    for (year, part, number), (problem, answer, source) in sorted(found.items()):
        if year >= 2026 or not re.fullmatch(r"\d{1,3}", answer):
            continue
        source_id = f"y{year}_{part}_{number:02d}" if part else f"y{year}_{number:02d}"
        origin = f"AIME {year} {part} #{number}" if part else f"AIME {year} #{number}"
        items.append(
            (
                source_id,
                {
                    "prompt": AIME_PROMPT.format(problem=problem.strip()),
                    "answer": str(int(answer)),
                    "origin": f"{origin} ({source})",
                },
            )
        )
    return items


def alfworld_catalog(data: Path) -> list[str]:
    root = data / "json_2.1.1" / "train"
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
        if not game.is_file() or not json.loads(game.read_text(encoding="utf-8")).get("solvable", False):
            continue
        files.append(str(game))
    files.sort()
    return [path[path.index("json_2.1.1/") :] for path in files]


def alfworld(data: Path) -> list[tuple[str, dict]]:
    return [
        (
            path,
            {
                "game_type": path.split("/")[2].split("-")[0],
                "alfworld_split": "train",
                "index": index,
                "prompt": None,
                "answer": None,
                "fixed_task_id": f"alfworld/train/{index:06d}",
            },
        )
        for index, path in enumerate(alfworld_catalog(data))
    ]


def exclusions(kind: str, items: list[tuple[str, dict]], tests: dict[str, list[dict]]) -> set[str]:
    held = tests.get(kind, [])
    if kind == "alfworld":
        return {sid for sid, _ in items if sid in {row["source_id"] for row in held}}
    if kind == "mbpp_plus":
        held = [row for row in held if row["kind"] == "mbpp_plus"]
        names = {task_statement(row["prompt"]) for row in held}
        return {sid for sid, f in items if task_statement(f["prompt"]) in names}
    texts = {norm(question_of(row["prompt"])) for row in held if row.get("prompt")}
    dropped = {sid for sid, f in items if norm(question_of(f["prompt"])) in texts}
    if kind == "aime_2026":
        held_grams = [grams(question_of(row["prompt"])) for row in held if row.get("prompt")]
        for sid, f in items:
            mine = grams(question_of(f["prompt"]))
            if any(len(mine & other) >= 0.5 * max(1, min(len(mine), len(other))) for other in held_grams):
                dropped.add(sid)
    return dropped


def rank(kind: str, source_id: str) -> str:
    return hashlib.sha256(f"train:{kind}:{source_id}".encode()).hexdigest()


def select(kind: str, items: list[tuple[str, dict]], count: int) -> list[dict]:
    ranked = sorted(items, key=lambda item: rank(kind, item[0]))
    if not ranked:
        raise SystemExit(f"no training candidates for {kind}")
    rows = []
    for position in range(count):
        source_id, fields = ranked[position % len(ranked)]
        repeat = position // len(ranked)
        fields = dict(fields)
        task_id = fields.pop("fixed_task_id", None) or f"{kind}/{source_id}"
        if repeat:
            task_id = f"{task_id}~r{repeat}"
        rows.append(
            {
                **fields,
                "kind": kind,
                "task_type": kind,
                "source_id": source_id,
                "task_id": task_id,
                "split": "train",
                "repeat": repeat,
                "origin": fields.get("origin", ORIGIN.get(kind)),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the EvoSteer IID training pool (data/evosteer/train/train_pool.jsonl).")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--alfworld-data", type=Path, default=os.environ.get("ALFWORLD_DATA"))
    parser.add_argument("--tests", type=Path, default=Path("data/evosteer/test"))
    parser.add_argument("--out", type=Path, default=Path("data/evosteer/train/train_pool.jsonl"))
    parser.add_argument("--config", type=Path, help="rewrite sampling.task_sources of this training config")
    parser.add_argument("--per-kind", type=int, default=PER_KIND)
    parser.add_argument("--no-sha-check", action="store_true")
    args = parser.parse_args()
    if args.alfworld_data is None:
        parser.error("--alfworld-data (or ALFWORLD_DATA) is required")
    check_raw(args.raw, sorted(RAW_FILES), not args.no_sha_check)
    tests = load_tests(args.tests)
    mbpp_test_ids = {row["source_id"] for row in tests.get("mbpp_plus", []) if row["kind"] == "mbpp_plus"}
    builders = {
        "hotpotqa": lambda: hotpotqa(args.raw),
        "nq_open": lambda: nq_open(args.raw),
        "medqa": lambda: medqa(args.raw),
        "aime_2026": lambda: aime(args.raw),
        "mbpp_plus": lambda: mbpp_plus(args.raw, mbpp_test_ids),
        "alfworld": lambda: alfworld(Path(args.alfworld_data)),
    }
    pool = []
    for kind in KINDS:
        items = builders[kind]()
        seen: dict[str, None] = {}
        unique = []
        for source_id, fields in items:
            key = norm(question_of(fields["prompt"])) if fields.get("prompt") else source_id
            if key not in seen:
                seen[key] = None
                unique.append((source_id, fields))
        dropped = exclusions(kind, unique, tests)
        kept = [(sid, f) for sid, f in unique if sid not in dropped]
        rows = select(kind, kept, args.per_kind)
        distinct = len({row["source_id"] for row in rows})
        print(f"{kind}\tcandidates={len(items)}\tunique={len(unique)}\theld_out_overlap={len(dropped)}\tdistinct={distinct}\trows={len(rows)}")
        pool.extend(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in pool:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"pool\t{len(pool)}\t{sha256_file(args.out)}")
    if args.config is not None:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        config["sampling"]["task_sources"] = {row["task_id"]: row["kind"] for row in pool}
        args.config.write_text(json.dumps(config, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
