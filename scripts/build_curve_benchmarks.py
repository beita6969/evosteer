"""Download full public TRAIN splits and freeze a balanced train/validation mix.

The output contains public prompts plus verifier-side fields in one local file;
the task factory strips verifier fields before constructing ``EvoTask``.  The
split is deterministic (SHA-256 ranked IDs), and no official test split is
downloaded or used.  Sources are recorded with URL, row counts and hashes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import time
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

HOT = "https://datasets-server.huggingface.co/rows?dataset=hotpotqa%2Fhotpot_qa&config=distractor&split=train&offset={offset}&length={length}"
HOT_PARQUET = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
    "distractor/{shard}?download=true"
)
GSM = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
MBPP = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"


def _download(url: str, path: Path) -> bytes:
    if path.is_file():
        return path.read_bytes()
    request = urllib.request.Request(url, headers={"User-Agent": "EvoSteer-curve-benchmarks/1"})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        payload = response.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _hotpot_rows(cache: Path, *, page_limit: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = cache / "hotpotqa_train_rows.jsonl"
    cache.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if raw.is_file():
        rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line]
    else:
        # Prefer the two official parquet shards: unlike the rows API this is
        # the complete raw TRAIN artifact and is resumable by urllib's cache.
        try:
            try:
                import pyarrow.parquet as pq
                parquet_reader = "pyarrow"
            except ImportError:
                import duckdb
                parquet_reader = "duckdb"
            parquet_paths = [cache / f"hotpot_train_{index:05d}.parquet" for index in (0, 1)]
            for index, parquet_path in enumerate(parquet_paths):
                _download(HOT_PARQUET.format(shard=f"train-{index:05d}-of-00002.parquet"), parquet_path)
            for parquet_path in parquet_paths:
                if parquet_reader == "pyarrow":
                    rows.extend(pq.read_table(parquet_path).to_pylist())
                else:
                    connection = duckdb.connect()
                    result = connection.execute("SELECT * FROM read_parquet(?)", [str(parquet_path)])
                    columns = [item[0] for item in result.description]
                    rows.extend(dict(zip(columns, values, strict=True)) for values in result.fetchall())
                    connection.close()
            raw.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
            return rows, {"url": HOT_PARQUET.format(shard="train-00000-of-00002.parquet"), "raw_path": str(raw), "rows": len(rows), "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(), "raw_shards": [str(path) for path in parquet_paths]}
        except ImportError:
            pass
        page = 100
        def fetch_page(offset: int) -> tuple[int, list[dict[str, Any]]]:
            for attempt in range(7):
                try:
                    request = urllib.request.Request(HOT.format(offset=offset, length=page), headers={"User-Agent": "EvoSteer-curve-benchmarks/1"})
                    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
                        body = json.load(response)
                    return offset, [item["row"] for item in body.get("rows", [])]
                except Exception:
                    if attempt == 6:
                        raise
                    time.sleep(2 ** min(attempt, 4))
        # The server advertises the total in every response. Discover it with
        # one request, then fetch all pages concurrently while preserving order.
        first_offset, first_batch = fetch_page(0)
        total = 0
        request = urllib.request.Request(HOT.format(offset=0, length=page), headers={"User-Agent": "EvoSteer-curve-benchmarks/1"})
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            total = int(json.load(response).get("num_rows_total", 0))
        pages: dict[int, list[dict[str, Any]]] = {first_offset: first_batch}
        offsets = list(range(page, total, page))
        if page_limit is not None:
            offsets = offsets[: max(0, page_limit - 1)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for offset, batch in pool.map(fetch_page, offsets):
                pages[offset] = batch
                if offset % 1000 == 0:
                    print(f"hotpotqa train rows fetched: {offset}/{total}", flush=True)
        with raw.open("w", encoding="utf-8") as out:
            for offset in sorted(pages):
                for row in pages[offset]:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows.append(row)
    return rows, {"url": HOT.replace("{offset}", "0").replace("{length}", "100"), "raw_path": str(raw), "rows": len(rows), "sha256": hashlib.sha256(raw.read_bytes()).hexdigest()}


def _jsonl_rows(url: str, cache: Path, name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = _download(url, cache / name)
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    return rows, {"url": url, "raw_path": str(cache / name), "rows": len(rows), "sha256": hashlib.sha256(payload).hexdigest()}


def _rank(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record(kind: str, item: dict[str, Any]) -> dict[str, Any]:
    if kind == "hotpotqa":
        context = "\n".join(f"[{title}] " + " ".join(sentences) for title, sentences in zip(item["context"]["title"], item["context"]["sentences"], strict=True))
        return {"kind": kind, "source_id": item["id"], "task_id": f"hotpotqa/{item['id']}", "prompt": f"Answer the question using the passages. Give a short answer only.\nQuestion: {item['question']}\nPassages:\n{context}", "answer": item["answer"]}
    if kind == "gsm8k":
        source_id = hashlib.sha256(item["question"].encode()).hexdigest()[:24]
        return {"kind": kind, "source_id": source_id, "task_id": f"gsm8k/{source_id}", "prompt": f"Solve the following grade-school math problem. Return the final answer clearly.\n{item['question']}", "answer": item["answer"]}
    source_id = str(item["task_id"])
    return {"kind": kind, "source_id": source_id, "task_id": f"mbpp/{source_id}", "prompt": item["text"], "answer": item.get("code", ""), "test_setup_code": item.get("test_setup_code", ""), "test_list": item.get("test_list", [])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/curve_benchmarks"))
    parser.add_argument("--train-per-kind", type=int, default=64)
    parser.add_argument("--validation-per-kind", type=int, default=16)
    parser.add_argument("--hotpot-pages", type=int, default=None, help="debug-only cap; default downloads all TRAIN pages")
    args = parser.parse_args()
    if args.train_per_kind < 1 or args.validation_per_kind < 1:
        raise SystemExit("split sizes must be positive")
    cache = args.output / "raw"
    hot, hot_meta = _hotpot_rows(cache, page_limit=args.hotpot_pages)
    gsm, gsm_meta = _jsonl_rows(GSM, cache, "gsm8k_train.jsonl")
    mbpp, mbpp_meta = _jsonl_rows(MBPP, cache, "mbpp_train.jsonl")
    source_rows = {"hotpotqa": hot, "gsm8k": gsm, "mbpp": mbpp}
    selected: list[dict[str, Any]] = []
    split_ids: dict[str, dict[str, list[str]]] = {}
    for kind, rows in source_rows.items():
        records = [_record(kind, row) for row in rows]
        records.sort(key=lambda item: _rank(item["source_id"]))
        required = args.train_per_kind + args.validation_per_kind
        if len(records) < required:
            raise ValueError(f"{kind} has only {len(records)} rows, need {required}")
        validation = records[: args.validation_per_kind]
        train = records[args.validation_per_kind : required]
        split_ids[kind] = {"train": [item["task_id"] for item in train], "validation": [item["task_id"] for item in validation]}
        selected.extend({**item, "split": "train"} for item in train)
        selected.extend({**item, "split": "validation"} for item in validation)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "curve_benchmarks.jsonl"
    with manifest.open("w", encoding="utf-8") as stream:
        for item in selected:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    metadata = {"format": "evosteer-curve-data@1", "sources": [hot_meta, gsm_meta, mbpp_meta], "split_ids": split_ids, "counts": {kind: {"train": args.train_per_kind, "validation": args.validation_per_kind} for kind in source_rows}, "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    (args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
