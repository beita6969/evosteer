"""Source-question duplicate exclusion for a future holdout, without scores or hashes."""

import json
import unicodedata
from collections.abc import Iterable, Mapping

from .input_metric_contracts import PublicTaskView


def public_problem_key(
    entry: PublicTaskView, source_identities: Mapping[str, tuple[str, ...]] | None = None
) -> tuple[str, str]:
    if entry.benchmark in {"scienceworld", "alfworld"}:
        if not source_identities or entry.task_id not in source_identities:
            raise ValueError(
                "environment isolation needs source task/variation/configuration identity"
            )
        return entry.benchmark, json.dumps(source_identities[entry.task_id])
    fields = dict(entry.fields)
    if entry.benchmark == "livemedbench" and "narrative" in fields:
        text = fields["narrative"] + "\n" + fields["core_request"]
    else:
        text = next(
            (
                fields[name]
                for name in ("question", "problem", "prompt", "query", "task")
                if name in fields
            ),
            entry.render(),
        )
    return entry.benchmark, " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def exclude_seen_public_problems(
    candidates: Iterable[PublicTaskView],
    seen: Iterable[PublicTaskView],
    *,
    source_identities: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[PublicTaskView, ...]:
    excluded = {public_problem_key(entry, source_identities) for entry in seen}
    selected = []
    for entry in candidates:
        key = public_problem_key(entry, source_identities)
        if key not in excluded:
            selected.append(entry)
            excluded.add(key)
    return tuple(selected)
