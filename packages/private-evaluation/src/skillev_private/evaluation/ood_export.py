"""Result-blind OOD sampling and explicit public/private source projections.

LiveCodeBench's test decoding and evaluation-sample layout follow its MIT
licensed ``lcb_runner/benchmarks/code_generation.py``. APPS uses its released
input_output object. Dataset contents and exported records are private artifacts.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import pickle
import random
import zipfile
import zlib
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from skillev_private.benchmarks.converters import _last_boxed_answer

from .ood_gpqa import BENCHMARK as GPQA_BIOORGANIC
from .ood_gpqa import export_bioorganic


class _StringUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        raise ValueError("LiveCodeBench compressed tests must contain data, not Python classes")


def decode_lcb_tests(value: str) -> list[dict[str, Any]]:
    try:
        result = json.loads(value)
    except json.JSONDecodeError:
        serialized = zlib.decompress(base64.b64decode(value))
        result = json.loads(_StringUnpickler(io.BytesIO(serialized)).load())
    if not isinstance(result, list):
        raise TypeError("LiveCodeBench tests are not a list")
    return result


def sample_rows(
    rows: Iterable[dict[str, Any]], count: int, seed: int
) -> tuple[list[tuple[int, dict[str, Any]]], int]:
    """Uniform reservoir sample, then restore source order; never inspect outcomes."""
    rng = random.Random(seed)  # noqa: S311 -- reproducible benchmark sampling
    selected: list[tuple[int, dict[str, Any]]] = []
    population = 0
    for index, row in enumerate(rows):
        population += 1
        if len(selected) < count:
            selected.append((index, row))
        else:
            slot = rng.randrange(population)
            if slot < count:
                selected[slot] = (index, row)
    if len(selected) != count:
        raise ValueError("source population is smaller than the requested panel")
    return sorted(selected), population


def _code_prompt(question: str, starter: str, function: str | None) -> str:
    if starter:
        interface = "Use this starter code and its calling convention:\n```python\n" + starter
        interface += "\n```"
    elif function:
        interface = f"Implement the callable named {function}, as specified in the problem."
    else:
        interface = "Read from standard input and write to standard output."
    return question + "\n\n" + interface + "\nReturn one complete Python solution."


def project_row(
    benchmark: str,
    index: int,
    row: dict[str, Any],
    *,
    science_manifest: dict[str, Any] | None = None,
    maximum_steps: int = 200,
) -> dict[str, Any]:
    target: dict[str, Any]
    extra: dict[str, Any] = {}
    task_id = f"{benchmark}:{row.get('id', row.get('question_id', index))}"
    if benchmark == "musique":
        paragraphs = [
            {"id": str(p.get("idx", position)), "title": p["title"], "text": p["paragraph_text"]}
            for position, p in enumerate(row["paragraphs"])
        ]
        public = {
            "question": row["question"],
            "context": "\n\n".join(
                f"[Paragraph {p['id']} | {p['title']}]\n{p['text']}" for p in paragraphs
            ),
        }
        extra["public_input_receipt"] = {
            "profile": "musique-all-public-paragraphs-stable-ids@2",
            "paragraphs": paragraphs,
        }
        target = {"accepted_answers": list(dict.fromkeys([row["answer"], *row["answer_aliases"]]))}
    elif benchmark == "nq-open":
        public = {"question": row["question"]}
        answers = row["answer"]
        target = {"accepted_answers": [answers] if isinstance(answers, str) else answers}
    elif benchmark == "omni-math":
        public = {"problem": row["problem"]}
        target = {"problem": row["problem"], "answer": row["answer"]}
        target["source_metadata"] = {
            key: row[key] for key in ("domain", "difficulty", "source") if key in row
        }
    elif benchmark == "math-hard":
        if row["level"] != "Level 5":
            raise ValueError("MATH-Hard requires the released Level 5 population")
        public = {"problem": row["problem"]}
        target = {"answer": _last_boxed_answer(row["solution"])}
        extra["source_metadata"] = {"level": row["level"], "subject": row["type"]}
    elif benchmark == "livecodebench":
        metadata = json.loads(row["metadata"])
        public_tests = json.loads(row["public_test_cases"])
        private_tests = decode_lcb_tests(row["private_test_cases"])
        tests = public_tests + private_tests
        function = metadata.get("func_name")
        public = {"prompt": _code_prompt(row["question_content"], row["starter_code"], function)}
        target = {
            "public_test_count": len(public_tests),
            "private_test_count": len(private_tests),
            "input_output": {
                "inputs": [t["input"] for t in tests],
                "outputs": [t["output"] for t in tests],
                "fn_name": function,
            },
        }
        task_id = f"{benchmark}:{row['platform']}:{row['question_id']}"
    elif benchmark == "apps-introductory":
        if row["difficulty"] != "introductory":
            raise ValueError("APPS task is not in the requested introductory subset")
        in_out = json.loads(row["input_output"])
        public = {
            "prompt": _code_prompt(
                row["question"], row.get("starter_code") or "", in_out.get("fn_name")
            )
        }
        target = {"input_output": in_out}
    elif benchmark == "scienceworld":
        if science_manifest is None:
            raise ValueError("ScienceWorld runtime configuration is required")
        task_id = f"scienceworld:{row['task_name']}:{row['variation_index']}"
        public = {"task": "Complete the ScienceWorld goal supplied by the environment at reset."}
        target = {}
        case = {
            "benchmark": benchmark,
            "deployment": "scienceworld",
            "task_id": task_id,
            "max_steps": maximum_steps,
            "task": public["task"],
            "payload": {
                "environment_id": task_id,
                "task_name": row["task_name"],
                "variation_index": row["variation_index"],
            },
        }
        extra = {"interactive": {"case": case, "manifest": science_manifest}}
    else:
        raise ValueError("OOD benchmark is not enabled")
    return {"task_id": task_id, "public": public, "target": target, **extra}


def source_rows(paths: list[str]) -> Iterator[dict[str, Any]]:
    for source in paths:
        if "#" in source:
            archive, member = source.split("#", 1)
            with zipfile.ZipFile(archive) as zipped, zipped.open(member) as stream:
                for raw_line in stream:
                    if raw_line.strip():
                        yield json.loads(raw_line)
        else:
            path = Path(source)
            if path.suffix == ".parquet":
                import pyarrow.parquet as pq  # type: ignore[import-untyped]

                for batch in pq.ParquetFile(path).iter_batches():
                    yield from batch.to_pylist()
            else:
                with path.open() as stream:
                    # The upstream MATH-Hard *.jsonl files contain JSON arrays.
                    # Parse their actual container, not the misleading extension.
                    prefix = stream.read(128).lstrip()
                    stream.seek(0)
                    if prefix.startswith("["):
                        yield from json.load(stream)
                    else:
                        for line in stream:
                            if line.strip():
                                yield json.loads(line)


def export_sources(config: dict[str, Any], directory: Path) -> dict[str, Any]:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "ood_sources": {},
        "ood_provenance": {},
        "evaluation_sample_counts": {},
    }
    for benchmark, source in config["sources"].items():
        if benchmark == "livemedbench":
            from .ood_livemedbench import export_livemedbench

            exported = export_livemedbench(
                source, directory, count=config["sample_count"], seed=config["seed"]
            )
            for key, value in exported.items():
                result[key].update(value)
            continue
        if benchmark == GPQA_BIOORGANIC:
            exported = export_bioorganic(
                source, directory, count=config["sample_count"], seed=config["seed"]
            )
            for key, value in exported.items():
                result[key].update(value)
            continue
        rows: Iterable[dict[str, Any]] = source_rows(source["paths"])
        if benchmark == "apps-introductory":
            rows = (row for row in rows if row["difficulty"] == "introductory")
        selected, population = sample_rows(rows, config["sample_count"], config["seed"])
        path = directory / f"{benchmark}-private.jsonl"
        with path.open("x") as out:
            for index, row in selected:
                projected = project_row(
                    benchmark,
                    index,
                    row,
                    science_manifest=config.get("science_manifest"),
                    maximum_steps=config.get("scienceworld_maximum_steps", 200),
                )
                out.write(json.dumps(projected, ensure_ascii=False) + "\n")
        result["ood_sources"][benchmark] = str(path.resolve())
        result["evaluation_sample_counts"][benchmark] = len(selected)
        result["ood_provenance"][benchmark] = {
            **source["provenance"],
            "population": population,
            "sample_count": len(selected),
            "seed": config["seed"],
            "selection": "uniform-reservoir-result-blind-source-order",
            **(
                {"input_profile": "musique-all-public-paragraphs-stable-ids@2"}
                if benchmark == "musique"
                else {}
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    exported = export_sources(json.loads(args.config.read_text()), args.output)
    (args.output / "source-config-private.json").write_text(json.dumps(exported, indent=2))
    print(json.dumps(exported["ood_provenance"], indent=2))


if __name__ == "__main__":
    main()
