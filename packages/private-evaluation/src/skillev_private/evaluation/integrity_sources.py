"""Source-specific allowlist exports for declared frozen IID populations.

Selection uses existing population manifests, never evaluator correctness.
Explicit small panels load only their requested domains, not the historical
eight-domain catalog's unrelated assets (including retired WebShop inputs).
TriviaQA is explicitly the released RC input, not a generated search dossier.
HealthBench exports only role/content; rubric objects stay in scorer targets.
"""

from __future__ import annotations

import gzip
import json
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from skillev.evaluation.input_metric_contracts import IID_BENCHMARKS, PublicTaskView
from skillev.evaluation.integrity_pipeline import FrozenPanel
from skillev_private.benchmarks.protocol_v13_training_sources import _nonfinite_constant


@dataclass(frozen=True, slots=True)
class SourcePanel:
    panel: FrozenPanel
    targets: dict[str, dict[str, Any]]
    interactive: dict[str, dict[str, Any]]
    provenance: dict[str, object]
    public_input_receipts: dict[str, list[dict[str, object]]] = field(default_factory=dict)


def select_source_panel(source: SourcePanel, counts: dict[str, int]) -> SourcePanel:
    """Frozen source-order prefixes only; no results or target values consulted."""
    if not counts or any(
        name not in IID_BENCHMARKS or type(count) is not int or count < 1
        for name, count in counts.items()
    ):
        raise ValueError("sample counts require named IID populations and positive sizes")
    entries = []
    for name, count in counts.items():
        available = [entry for entry in source.panel.entries if entry.benchmark == name]
        if count > len(available):
            raise ValueError("requested sample exceeds the available frozen population")
        entries.extend(available[:count])
    selected = {entry.task_id for entry in entries}
    provenance = {
        **source.provenance,
        "sample_selection": "fixed-source-order-prefix@1",
        "sample_counts": counts,
    }
    return SourcePanel(
        replace(
            source.panel,
            entries=tuple(entries),
            sample_counts=tuple(counts.items()),
            source_provenance=json.dumps(provenance, sort_keys=True),
        ),
        {key: value for key, value in source.targets.items() if key in selected},
        {key: value for key, value in source.interactive.items() if key in selected},
        provenance,
        {key: value for key, value in source.public_input_receipts.items() if key in selected},
    )


def order_source_panel(source: SourcePanel, benchmark_order: tuple[str, ...]) -> SourcePanel:
    """Schedule whole domains first without selecting tasks or inspecting labels."""
    catalog = source.panel.catalog
    if len(benchmark_order) != len(catalog) or set(benchmark_order) != set(catalog):
        raise ValueError(
            "execution order must contain every benchmark in the selected catalog once"
        )
    rank = {benchmark: index for index, benchmark in enumerate(benchmark_order)}
    entries = tuple(sorted(source.panel.entries, key=lambda entry: rank[entry.benchmark]))
    return replace(source, panel=replace(source.panel, entries=entries))


def public_source_view(task_id: str, benchmark: str, row: dict[str, Any]) -> PublicTaskView:
    fields: dict[str, Any]
    if benchmark == "hotpotqa":
        question = str(row["question"]).rpartition("Question:")[2] or row["question"]
        fields = {"question": question, "context": row["context"]}
    elif benchmark == "triviaqa":
        question = str(row["question"]).rpartition("Question:")[2].strip() or row["question"]
        fields = {"question": question}
        if row.get("context"):
            fields["public_context"] = row["context"]
    elif benchmark == "aime-2026":
        fields = {"problem": row["question"]}
    elif benchmark == "healthbench":
        fields = {"prompt": [{"role": m["role"], "content": m["content"]} for m in row["prompt"]]}
    elif benchmark in {"humaneval", "mbpp-plus"}:
        fields = {"prompt": row["prompt"]}
    elif benchmark in {"webshop", "alfworld"}:
        fields = {"task": row["task"]}
    else:
        raise ValueError("not an owner-approved IID source")
    return PublicTaskView.from_record(task_id, benchmark, fields)


def _json_rows(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        # EvalPlus uses non-finite test inputs. Reuse the formal-training wire
        # encoding already restored by the isolated native MBPP worker.
        return [
            json.loads(line, parse_constant=_nonfinite_constant) for line in stream if line.strip()
        ]


def mbpp_public_prompt(public: str | dict[str, Any], reference: dict[str, Any]) -> str:
    """Verify the standard public carrier, not canonical code or private tests.

    Its original interface and examples are preserved byte-for-byte. Additional
    target fields are never copied to the actor, regardless of skills mode.
    """
    prompt = public if isinstance(public, str) else public["prompt"]
    if not isinstance(prompt, str) or prompt != reference["prompt"]:
        raise ValueError("MBPP public prompt differs from its declared standard public carrier")
    return prompt


def load_source_panel(config: dict[str, Any]) -> SourcePanel:
    files = config["source_files"]
    requested = config.get("evaluation_sample_counts")
    benchmarks = tuple(name for name in IID_BENCHMARKS if requested is None or name in requested)
    skillflow = (
        json.loads(Path(files["skillflow_iid"]).read_text())
        if any(name in benchmarks for name in ("hotpotqa", "triviaqa", "aime-2026"))
        else []
    )
    human = (
        {row["task_id"]: row for row in _json_rows(Path(files["humaneval"]))}
        if "humaneval" in benchmarks
        else {}
    )
    mbpp_prompts = (
        json.loads(Path(files["mbpp_public_prompts"]).read_text())
        if "mbpp-plus" in benchmarks
        else {}
    )
    mbpp_targets = (
        {row["task_id"]: row for row in _json_rows(Path(files["mbpp_targets"]))}
        if "mbpp-plus" in benchmarks
        else {}
    )
    entries: list[PublicTaskView] = []
    targets: dict[str, dict[str, Any]] = {}
    interactive: dict[str, dict[str, Any]] = {}
    provenance: dict[str, object] = {
        "source_files": files,
        "triviaqa_lane": "released-skillflow-iid-v3-reading-comprehension",
        "triviaqa_generated_dossiers": False,
        "mbpp_display_name": "MBPP+",
        "mbpp_owner_requested_name": "MBPP+",
        "mbpp_hard_subset_verified": False,
        "mbpp_population": "EvalPlus MBPP v0.2.0; no distinct official hard split",
        "mbpp_public_carrier": "standard-public-prompt-verbatim@1",
    }
    for benchmark in benchmarks:
        if benchmark == "healthbench":
            rows = _json_rows(Path(files["healthbench"]))
            chosen = random.Random(0).sample(rows, 128)  # noqa: S311 -- official population rule
            if config.get("canary"):
                frozen_ids = {row["prompt_id"] for row in chosen}
                chosen = [row for row in rows if row["prompt_id"] not in frozen_ids][:1]
            for row in chosen:
                task_id = str(row["prompt_id"])
                # The complete case is never passed to PublicTaskView.
                health_prompt = [
                    {"role": m["role"], "content": m["content"]} for m in row["prompt"]
                ]
                entries.append(public_source_view(task_id, benchmark, {"prompt": health_prompt}))
                targets[task_id] = {"prompt": health_prompt, "rubrics": row["rubrics"]}
            continue
        manifest_path = Path(config["manifests"][benchmark])
        manifest = json.loads(manifest_path.read_text())
        provenance[benchmark] = {
            "manifest": str(manifest_path),
            "population_id": manifest.get("population_id"),
            "selection_rule": manifest.get("selection_rule"),
            "dataset_revision": manifest.get("dataset_revision"),
        }
        if benchmark in {"webshop", "alfworld"}:
            for row in manifest["cases"]:
                task_id = str(row["task_id"])
                entries.append(public_source_view(task_id, benchmark, {"task": row["task"]}))
                interactive[task_id] = {"case": row, "manifest": manifest}
            continue
        for item in manifest["entries"]:
            task_id, source_id = str(item["task_id"]), str(item["source_identity"])
            if benchmark == "humaneval":
                row = human[source_id]
                target = {name: row[name] for name in ("prompt", "test", "entry_point")}
            elif benchmark == "mbpp-plus":
                public = mbpp_prompts[source_id]
                reference = mbpp_targets[source_id]
                prompt = mbpp_public_prompt(public, reference)
                row = {"prompt": prompt}
                target = {
                    "source_task_id": source_id,
                    "prompt": prompt,
                    "reference_prompt": reference["prompt"],
                    "private_target": {
                        name: reference[name]
                        for name in (
                            "assertion",
                            "atol",
                            "base_input",
                            "canonical_solution",
                            "contract",
                            "entry_point",
                            "plus_input",
                        )
                    },
                }
            else:
                row = skillflow[int(item["source_position"])]
                if benchmark == "aime-2026":
                    target = {"answer": str(int(row["answer"]))}
                else:
                    aliases = (
                        row["answer"].split("|") if benchmark == "triviaqa" else [row["answer"]]
                    )
                    target = {"accepted_answers": aliases}
            entries.append(public_source_view(task_id, benchmark, row))
            targets[task_id] = target
    if "triviaqa" in benchmarks and config.get("trivia_aliases"):
        # Scorer-only aliases never participate in public input construction.
        aliases = json.loads(Path(config["trivia_aliases"]).read_text())
        for entry in entries:
            if entry.benchmark == "triviaqa":
                targets[entry.task_id] = {"accepted_answers": aliases[entry.task_id]}
        provenance["trivia_aliases"] = "official scorer-only alias population"
    return SourcePanel(
        FrozenPanel(
            tuple(entries),
            "development-exposed" if not config.get("canary") else "nonfinal-development-canary",
            json.dumps(provenance, sort_keys=True),
        ),
        targets,
        interactive,
        provenance,
    )
