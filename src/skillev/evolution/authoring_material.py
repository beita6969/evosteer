"""Deterministic bounded projection, separate from complete decision evidence.

Never trims applicability, immutable constraints or the output contract. If
those mandatory inputs cannot fit, fail before submitting an author request.
The full request remains in the existing private transaction journal.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

from skillev.contracts import canonical_json

if TYPE_CHECKING:
    from skillev.policy import AuthoringTokenizerProtocol

    from .authoring import AuthoringRequest

AUTHORING_MATERIAL_VERSION = "bounded-public-execution-diversity@1"


def _compact(value: Any, *, depth: int = 0) -> Any:
    """Bound source-identity arrays AND summaries, not merely exemplars."""
    if isinstance(value, str):
        return value if len(value) <= 1200 else value[:1200] + " [excerpt]"
    if isinstance(value, list):
        return {"count": len(value), "examples": [_compact(v, depth=depth + 1) for v in value[:4]]}
    if isinstance(value, dict):
        if depth >= 5:
            return {"field_count": len(value), "omitted": True}
        return {key: _compact(item, depth=depth + 1) for key, item in sorted(value.items())[:32]}
    return value


def render_bounded_authoring_prompt(
    request: AuthoringRequest,
    *,
    tokenizer: AuthoringTokenizerProtocol,
    maximum_tokens: int,
) -> str:
    from .authoring import AuthoringFailedError, render_authoring_prompt

    full = render_authoring_prompt(request, tokenizer=tokenizer)
    header, encoded, footer = full.split("\n", 2)
    payload = json.loads(encoded)
    material = payload["material"]
    exemplars = material.pop("edge_exemplars")
    summary = material.get("evidence_summary", "")
    try:
        summary = json.loads(summary)
    except (ValueError, TypeError):
        pass
    material["evidence_summary"] = _compact(summary)
    if "related_skills" in material:
        material["related_skills"] = _compact(material["related_skills"])
    if "modality" in material:
        material["modality"] = _compact(material["modality"])
    material["projection"] = {
        "version": AUTHORING_MATERIAL_VERSION,
        "full_evidence_reference": "transaction-request-and-phase-decision",
        "full_edge_count": len(exemplars),
        "selected_edge_count": 0,
        "selection": "round-robin-public-context-status-then-edge-id",
        "not_all_evidence_shown": True,
    }
    material["edge_exemplars"] = []

    def render() -> str:
        return header + "\n" + canonical_json(payload) + "\n" + footer

    def fits() -> bool:
        return len(tokenizer.encode_authoring_prompt(render())) <= maximum_tokens

    if not fits():
        raise AuthoringFailedError(
            "mandatory author constraints exceed input budget before submission"
        )
    # One exemplar per observed public failure/context first, independent of
    # author outcomes. Clip wire text only in the prompt, never the stored edge.
    groups: dict[tuple[str, str, str], deque[dict[str, Any]]] = defaultdict(deque)
    for item in sorted(exemplars, key=lambda edge: edge["edge_id"]):
        groups[(item["task_family"], item["context_id"], item["observation_status"])].append(item)
    ordered = [groups[key] for key in sorted(groups)]
    while any(ordered):
        for group in ordered:
            if not group:
                continue
            item = group.popleft()
            if item.get("public_execution") is not None:
                item["public_execution"] = _compact(item["public_execution"])
            material["edge_exemplars"].append(item)
            material["projection"]["selected_edge_count"] += 1
            if not fits():
                material["edge_exemplars"].pop()
                material["projection"]["selected_edge_count"] -= 1
    if exemplars and not material["edge_exemplars"]:
        raise AuthoringFailedError("no public execution exemplar fits the author input budget")
    return render()
