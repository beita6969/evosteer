from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import urllib.request
from pathlib import Path

PER_KIND = 128

SWE_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
WEBSHOP_REPOSITORY_REVISION = "64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd"
WEBSHOP_MIRROR_REVISION = "ce990fff5aee388db2706f07820c578ab68e0453"
WEBSHOP_GOAL_SHUFFLE_SEED = 233
WEBSHOP_TEST_GOALS = range(0, 500)

RAW_FILES = {
    "swe_bench_verified_test.parquet": (
        f"https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/resolve/{SWE_REVISION}/data/test-00000-of-00001.parquet",
        "a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd",
    ),
    "webshop_items_human_ins.json": (
        f"https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/{WEBSHOP_MIRROR_REVISION}/items_human_ins.json",
        "cf78667548a71786e1d9049c24b802e48e1084ad4bb021cae56ce1f6d96954a3",
    ),
    "webshop_items_shuffle.json": (
        f"https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/{WEBSHOP_MIRROR_REVISION}/items_shuffle.json",
        "2ef591d65df3af89e972ab72468eb82cbf124d876552d9f3678667edd620a6c8",
    ),
}

NEEDS = {
    "swe_bench_verified": ["swe_bench_verified_test.parquet"],
    "webshop": ["webshop_items_human_ins.json", "webshop_items_shuffle.json"],
}

ORIGIN = {
    "swe_bench_verified": "princeton-nlp/SWE-bench_Verified/default/test",
    "webshop": "princeton-nlp/WebShop/human_goals/test",
}

TASK_TYPE = {"swe_bench_verified": "mbpp_plus", "webshop": "alfworld"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(raw: Path, name: str) -> None:
    url, _ = RAW_FILES[name]
    target = raw / name
    partial = target.with_suffix(target.suffix + ".part")
    print(f"downloading {url}", file=sys.stderr)
    with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, 1 << 22)
    partial.rename(target)


def check_raw(raw: Path, names: list[str], download: bool, verify: bool) -> None:
    for name in names:
        path = raw / name
        if not path.is_file():
            if not download:
                raise SystemExit(f"missing {path}; download {RAW_FILES[name][0]} or pass --download")
            raw.mkdir(parents=True, exist_ok=True)
            fetch(raw, name)
        if verify:
            found = sha256_file(path)
            if found != RAW_FILES[name][1]:
                raise SystemExit(f"sha256 mismatch for {path}: {found} != {RAW_FILES[name][1]}")


def rank(kind: str, source_id: str) -> str:
    return hashlib.sha256(f"test:{kind}:{source_id}".encode()).hexdigest()


def select(kind: str, candidates: list[dict], count: int) -> list[dict]:
    ids = [row["source_id"] for row in candidates]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{kind}: duplicate source ids")
    ranked = sorted(candidates, key=lambda row: rank(kind, row["source_id"]))
    return [
        {
            "prompt": None,
            "answer": None,
            **row,
            "task_id": f"{kind}/test-{row['source_id']}",
            "kind": kind,
            "task_type": TASK_TYPE[kind],
            "split": "test",
            "origin": ORIGIN[kind],
        }
        for row in ranked[:count]
    ]


def swe_bench_verified(raw: Path, count: int) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(raw / "swe_bench_verified_test.parquet", columns=["instance_id", "repo", "base_commit"])
    rows = table.to_pylist()
    if len(rows) != 500:
        raise SystemExit(f"swe_bench_verified: expected 500 instances, found {len(rows)}")
    candidates = [
        {"source_id": row["instance_id"], "repo": row["repo"], "base_commit": row["base_commit"]}
        for row in rows
    ]
    return select("swe_bench_verified", candidates, count)


def iter_product_asins(path: Path):
    try:
        import ijson
    except ImportError:
        with path.open() as handle:
            for product in json.load(handle):
                yield product["asin"]
        return
    with path.open("rb") as handle:
        yield from ijson.items(handle, "item.asin")


def webshop_goals(raw: Path) -> list[dict]:
    with (raw / "webshop_items_human_ins.json").open() as handle:
        human = json.load(handle)
    seen: set[str] = set()
    goals: list[dict] = []
    for asin in iter_product_asins(raw / "webshop_items_shuffle.json"):
        if asin == "nan" or len(asin) > 10 or asin in seen:
            continue
        seen.add(asin)
        for position, instruction in enumerate(human.get(asin, [])):
            if len(instruction["instruction_attributes"]) == 0:
                continue
            goals.append(
                {
                    "asin": asin,
                    "instruction_position": position,
                    "instruction": instruction["instruction"].strip("."),
                }
            )
    random.seed(WEBSHOP_GOAL_SHUFFLE_SEED)
    random.shuffle(goals)
    return goals


def webshop(raw: Path, count: int) -> list[dict]:
    goals = webshop_goals(raw)
    if len(goals) != 12087:
        raise SystemExit(f"webshop: expected 12087 human goals, found {len(goals)}")
    candidates = [
        {
            "source_id": str(index),
            "goal_index": index,
            "goal_shuffle_seed": WEBSHOP_GOAL_SHUFFLE_SEED,
            "asin": goals[index]["asin"],
            "instruction_position": goals[index]["instruction_position"],
            "instruction": goals[index]["instruction"],
        }
        for index in WEBSHOP_TEST_GOALS
    ]
    return select("webshop", candidates, count)


def write(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return sha256_file(path)


BUILDERS = {"swe_bench_verified": swe_bench_verified, "webshop": webshop}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild the EvoSteer OOD environment test sets (SWE-Bench Verified, WebShop) in data/evosteer/test/ood/."
    )
    parser.add_argument("--raw", type=Path, required=True, help="directory holding the raw files listed in RAW_FILES")
    parser.add_argument("--out", type=Path, default=Path("data/evosteer/test/ood"))
    parser.add_argument("--only", nargs="+", choices=sorted(BUILDERS), default=sorted(BUILDERS))
    parser.add_argument("--per-kind", type=int, default=PER_KIND)
    parser.add_argument("--download", action="store_true", help="fetch missing raw files from the pinned URLs")
    parser.add_argument("--no-sha-check", action="store_true")
    args = parser.parse_args()
    for kind in args.only:
        check_raw(args.raw, NEEDS[kind], args.download, not args.no_sha_check)
        rows = BUILDERS[kind](args.raw, args.per_kind)
        digest = write(args.out / f"{kind}.jsonl", rows)
        print(f"{kind}: {len(rows)} rows sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
