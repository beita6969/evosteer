"""Judge a private finalized Omni export using Luna medium, never an answer agent.

One request per unresolved record, with no automatic POST replay. An explicitly
declared provider failover may route subsequent unsent records to official Luna. A
failure stops the batch and preserves every planned row. Explicit --resume
retries only unresolved judging, retaining all attempts and never regenerating
or selecting the owner's answer. No live API call happens on import.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote

from skillev.evaluation.external_judge_policy import EXTERNAL_JUDGE_MAX_TOKENS
from skillev.evaluation.sealed_candidates import FinalCandidate

from .external_judge_api import make_external_judge_client
from .ood_luna_judge import (
    JUDGE_EFFORT,
    JUDGE_MODEL,
    JUDGE_PROFILE,
    finalized_batch,
    judge_identity,
    judge_record,
    read_judgement,
)

_SCOPE = ("run_id", "arm_id", "task_id", "owner_call_id")


def grade_candidate(
    candidate: FinalCandidate,
    target: dict[str, Any],
    settings: dict[str, Any],
    *,
    client: Any = None,
) -> dict[str, Any]:
    """Terminal-only integration: no artificial missing-file failure before judging.

    Each finalized answer owns one ledger. Both resolved verdicts are final;
    failed attempts require an explicit judge-CLI resume, never an automatic retry.
    """
    if judge_identity(settings)[0] != JUDGE_PROFILE:
        raise ValueError("live API judging requires the declared Luna medium profile")
    template = Path(settings["template_path"]).read_text()
    scope = (candidate.run_id, candidate.arm_id, candidate.episode_id)
    batch = finalized_batch([judge_record(scope, candidate, target, template)], template)
    directory = Path(settings["cache_directory"])
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / (quote(candidate.episode_id, safe="") + "-private.json")
    if path.exists():
        state = json.loads(path.read_text())
        if state.get("export") != batch:
            raise ValueError("saved judge belongs to a different submission or condition")
    else:
        owned_client = client is None
        if owned_client:
            client = make_external_judge_client()
        try:
            state = judge_batch(batch, path, client=client)
        finally:
            if owned_client:
                client.close()
    if not state["complete"]:
        raise RuntimeError("Omni-MATH judge unresolved; see private attempt ledger")
    result = read_judgement(candidate, {**settings, "judgements_path": str(path)})
    attempts = state["records"][0]["attempts"]
    result["requested_judge_model"] = JUDGE_MODEL
    result["returned_judge_models"] = [row.get("returned_model") for row in attempts]
    result["judge_providers"] = [row.get("provider") for row in attempts]
    result["judge_endpoints"] = [row.get("endpoint") for row in attempts]
    result["source_metadata"] = target.get("source_metadata", {})
    result["cost"] = {
        "model_calls": float(len(attempts)),
        "input_tokens": sum((row["usage"] or {}).get("prompt_tokens", 0) for row in attempts),
        "output_tokens": sum((row["usage"] or {}).get("completion_tokens", 0) for row in attempts),
        "unknown_usage_calls": float(sum(row["usage"] is None for row in attempts)),
        "api_wall_seconds": sum(row["wall_seconds"] for row in attempts),
    }
    return result


def parse_report(report: str) -> tuple[bool, str]:
    """Read the official Markdown sections without treating unknown text as FALSE."""
    parts = re.split(r"(?m)^##[ \t]+", report)
    sections: dict[str, str] = {}
    for part in parts[1:]:
        title, separator, body = part.partition("\n")
        if not separator or title.strip() in sections:
            raise ValueError("judge report has missing or duplicate sections")
        sections[title.strip()] = body.strip()
    verdict = (
        sections.get("Equivalence Judgement", "").strip().removeprefix("**").removesuffix("**")
    )
    reason = sections.get("Justification", "")
    marker = "=== report over ==="
    if verdict not in {"TRUE", "FALSE"} or not reason.endswith(marker):
        raise ValueError("judge report has no complete, unambiguous equivalence verdict")
    reason = reason.removesuffix(marker).strip()
    if not reason:
        raise ValueError("judge report has no justification")
    return verdict == "TRUE", reason


def _validate_export(batch: dict[str, Any]) -> None:
    if (batch.get("profile"), batch.get("judge_model"), batch.get("reasoning_effort")) != (
        JUDGE_PROFILE,
        JUDGE_MODEL,
        JUDGE_EFFORT,
    ):
        raise ValueError("new API judging requires the Luna medium export profile")
    if (
        batch.get("backend") != "openai-chat-completions"
        or batch.get("max_completion_tokens") != EXTERNAL_JUDGE_MAX_TOKENS
    ):
        raise ValueError("judge transport settings differ from the declared condition")
    rows = batch["records"]
    if not rows or batch.get("planned_count") != len(rows):
        raise ValueError("judge export is missing planned candidates")
    scopes = [tuple(row[key] for key in _SCOPE) for row in rows]
    if len(scopes) != len(set(scopes)) or any(not row["judge_prompt"].strip() for row in rows):
        raise ValueError("judge export has duplicate candidates or empty prompts")


def _write_private(path: Path, state: dict[str, Any]) -> None:
    """Atomically update only the output already claimed by this invocation."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(state, stream, ensure_ascii=False, indent=2)
            stream.flush()
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _request(client: Any, prompt: str) -> dict[str, Any]:
    from openai import APIError

    started = time.monotonic()
    attempt: dict[str, Any] = {"usage": None, "status": "transport-error"}
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            reasoning_effort=JUDGE_EFFORT,
            max_completion_tokens=EXTERNAL_JUDGE_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            timeout=120.0,
            stream=False,
            store=False,
        )
    except (APIError, TimeoutError, ConnectionError) as error:
        # No exception body/headers: these may echo secrets or private prompts.
        attempt.update(
            error_type=type(error).__name__,
            http_status=getattr(error, "status_code", None),
            request_id=getattr(error, "request_id", None),
            provider=getattr(error, "judge_provider", None),
            endpoint=getattr(error, "judge_endpoint", None),
        )
    else:
        attempt.update(
            response_id=response.id,
            request_id=getattr(response, "_request_id", None),
            returned_model=response.model,
            provider=getattr(response, "judge_provider", None),
            endpoint=getattr(response, "judge_endpoint", None),
            provider_requested_model=getattr(response, "judge_requested_model", None),
            failover_reason=getattr(response, "judge_failover_reason", None),
            usage=None if response.usage is None else response.usage.model_dump(),
        )
        if len(response.choices) != 1:
            attempt["status"] = "invalid-response"
        else:
            choice = response.choices[0]
            attempt.update(raw_output=choice.message.content, finish_reason=choice.finish_reason)
            if choice.finish_reason != "stop":
                attempt["status"] = "incomplete-response"
            elif choice.message.refusal:
                attempt["status"] = "judge-refusal"
            else:
                try:
                    equivalent, reason = parse_report(choice.message.content or "")
                except ValueError:
                    attempt["status"] = "parse-error"
                else:
                    attempt.update(status="resolved", equivalent=equivalent, reason=reason)
    attempt["wall_seconds"] = time.monotonic() - started
    return attempt


def judge_batch(
    batch: dict[str, Any],
    output: Path,
    *,
    client: Any,
    resume: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    _validate_export(batch)
    state: dict[str, Any]
    if resume:
        state = json.loads(output.read_text())
        if state.get("export") != batch:
            raise ValueError("resume must use exactly the original finalized judge export")
        if len(state["records"]) != batch["planned_count"]:
            raise ValueError("saved judge output is missing planned rows")
        for original, row in zip(batch["records"], state["records"], strict=True):
            if any(original[key] != row[key] for key in _SCOPE):
                raise ValueError("saved judge row differs from its finalized candidate")
    else:
        state = {
            "profile": JUDGE_PROFILE,
            "judge_model": JUDGE_MODEL,
            "reasoning_effort": JUDGE_EFFORT,
            "backend": batch["backend"],
            "max_completion_tokens": EXTERNAL_JUDGE_MAX_TOKENS,
            "planned_count": batch["planned_count"],
            "export": batch,
            "records": [
                {**{key: row[key] for key in _SCOPE}, "status": "not-requested", "attempts": []}
                for row in batch["records"]
            ],
        }
        # Refuse accidental replacement, including old high-judge results.
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)

    def save() -> None:
        state["resolved_count"] = sum(row["status"] == "resolved" for row in state["records"])
        state["attempt_count"] = sum(len(row["attempts"]) for row in state["records"])
        state["complete"] = state["resolved_count"] == state["planned_count"]
        _write_private(output, state)

    save()
    for original, row in zip(batch["records"], state["records"], strict=True):
        if row["status"] == "resolved":
            continue  # A FALSE verdict is final too; never retry until positive.
        row["status"] = "in-flight"
        row["attempts"].append({"status": "in-flight", "usage": None})
        save()
        attempt = _request(client, original["judge_prompt"])
        row["attempts"][-1] = attempt
        row["status"] = attempt["status"]
        if row["status"] == "resolved":
            row.update(equivalent=attempt["equivalent"], reason=attempt["reason"])
        save()
        if progress is not None:
            progress(state)
        if row["status"] != "resolved":
            break  # Explicit recovery, no hidden resampling or denominator reduction.
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    batch = json.loads(args.batch.read_text())
    _validate_export(batch)
    started = time.monotonic()
    initial_resolved = json.loads(args.output.read_text())["resolved_count"] if args.resume else 0

    def progress(state: dict[str, Any]) -> None:
        seconds = time.monotonic() - started
        count = state["resolved_count"]
        rate = (count - initial_resolved) / max(seconds, 0.001)
        eta = f"{(state['planned_count'] - count) / rate:.0f}s" if rate else "unknown"
        print(
            f"Resolved {count}/{state['planned_count']}; elapsed {seconds:.1f}s; "
            f"rate {rate * 60:.2f} records/min; ETA {eta}",
            flush=True,
        )

    with make_external_judge_client() as client:
        result = judge_batch(
            batch, args.output, client=client, resume=args.resume, progress=progress
        )
    if not result["complete"]:
        raise SystemExit(
            "Judge batch incomplete; inspect private attempts before explicit --resume"
        )


if __name__ == "__main__":
    main()
