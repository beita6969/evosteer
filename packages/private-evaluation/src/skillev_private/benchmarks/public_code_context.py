"""Answer-free HumanEval assembly: public syntax plus unchanged submitted code.

A target definition replaces the public stub, never its helpers/imports. A
standalone module is an explicit input contract, not a regex or test-result guess.
No code is executed and no reference solution or test is accepted by this API.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Literal

ASSEMBLY_VERSION = "humaneval-public-context@2"
HUMANEVAL_VERIFIER = "humaneval-public-context@2-original-isolated@2"
LEGACY_TRAINING_ASSEMBLY = "humaneval-training-regex@1"
SubmissionForm = Literal["auto-public-context", "body", "target-definition", "module"]


@dataclass(frozen=True, slots=True)
class PublicCodeContext:
    public_source: str
    entry_point: str

    def __post_init__(self) -> None:
        if not isinstance(self.public_source, str) or not self.public_source.strip():
            raise ValueError("HumanEval requires its actual public source")
        if not isinstance(self.entry_point, str) or not self.entry_point.isidentifier():
            raise ValueError("HumanEval requires its public entry-point identifier")


@dataclass(frozen=True, slots=True)
class CodeSubmission:
    original_payload: str
    form: SubmissionForm = "auto-public-context"


@dataclass(frozen=True, slots=True)
class AssembledCodeCandidate:
    prefix: str
    completion: str
    original_payload: str
    form: str
    assembly_version: str
    public_context_retained: bool

    @property
    def executable_source(self) -> str:
        return self.prefix + self.completion

    def diagnostics(self) -> dict[str, str | bool]:
        return {
            "assembly_version": self.assembly_version,
            "submission_form": self.form,
            "public_context_retained": self.public_context_retained,
        }


def public_module(source: str) -> ast.Module:
    """Native prompts can end at a header or at a docstring-only function body."""
    try:
        return ast.parse(source)
    except IndentationError:
        return ast.parse(source + "\n    pass\n")


def public_entry_point(source: str) -> str:
    functions = [
        node
        for node in public_module(source).body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    if not functions:
        raise ValueError("public HumanEval context has no top-level entry point")
    return functions[-1].name


def _start(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    return (
        min(node.lineno, *(d.lineno for d in node.decorator_list))
        if node.decorator_list
        else node.lineno
    )


def _future(node: ast.stmt) -> bool:
    return isinstance(node, ast.ImportFrom) and node.module == "__future__"


def compose_code_candidate(
    context: PublicCodeContext,
    submission: CodeSubmission,
    *,
    assembly_version: str = ASSEMBLY_VERSION,
) -> AssembledCodeCandidate:
    prompt, candidate = context.public_source, submission.original_payload
    if not isinstance(prompt, str) or not isinstance(candidate, str):
        raise TypeError("public context and original submission must be text")
    if assembly_version == LEGACY_TRAINING_ASSEMBLY:
        # Read-only historical comparison only. Never choose this branch by verdict.
        prefix, body, retained = prompt, candidate, True
        if candidate.startswith(prompt):
            body = candidate[len(prompt) :]
        elif re.search(
            rf"(?m)^\s*(?:async\s+)?def\s+({re.escape(context.entry_point)})\s*\(", candidate
        ):
            prefix, retained = "# complete candidate source\n", False
        return AssembledCodeCandidate(prefix, body, candidate, "legacy", assembly_version, retained)
    if assembly_version != ASSEMBLY_VERSION:
        raise ValueError("unknown HumanEval assembly version")
    if submission.form not in {"auto-public-context", "body", "target-definition", "module"}:
        raise ValueError("unknown declared code submission form")
    if submission.form == "module":
        return AssembledCodeCandidate("", candidate, candidate, "module", assembly_version, False)
    if submission.form == "body":
        return AssembledCodeCandidate(prompt, candidate, candidate, "body", assembly_version, True)
    if candidate.startswith(prompt):
        remaining = candidate[len(prompt) :]
        return AssembledCodeCandidate(
            prompt if remaining.strip() else "",
            remaining if remaining.strip() else candidate,
            candidate,
            "public-prompt-and-body",
            assembly_version,
            True,
        )
    try:
        module = ast.parse(candidate)
    except SyntaxError:
        module = None
    target_definition = submission.form == "target-definition" or (
        module is not None
        and any(
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name == context.entry_point
            for node in module.body
        )
    )
    if not target_definition:
        # Strings, nested defs, methods and prose never masquerade as top-level
        # definitions. A malformed body still reaches the real compiler unchanged.
        return AssembledCodeCandidate(prompt, candidate, candidate, "body", assembly_version, True)
    public = public_module(prompt)
    targets = [
        node
        for node in public.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name == context.entry_point
    ]
    if len(targets) != 1:
        raise ValueError("public target insertion region must be unambiguous")
    target = targets[0]
    lines = prompt.splitlines(keepends=True)
    scaffold = "".join(lines[: _start(target) - 1] + lines[target.end_lineno or target.lineno :])
    if scaffold and not scaffold.endswith("\n"):
        scaffold += "\n"
    if not scaffold.strip() or candidate.startswith(scaffold):
        prefix, body = "", candidate
    else:
        futures = [] if module is None else [n for n in module.body if _future(n)]
        if not futures:
            prefix, body = scaffold, candidate
        else:
            # Keep candidate bytes and their order. Public helpers belong AFTER
            # its module docstring/future header, not before a future import.
            end = futures[-1].end_lineno
            assert end is not None
            candidate_lines = candidate.splitlines(keepends=True)
            prefix = "".join(candidate_lines[:end])
            if not prefix.endswith("\n"):
                prefix += "\n"
            public_futures = {
                i
                for node in public.body
                if _future(node)
                for i in range(node.lineno - 1, node.end_lineno or node.lineno)
            }
            future_source = "".join(line for i, line in enumerate(lines) if i in public_futures)
            remaining_source = "".join(
                line
                for i, line in enumerate(lines)
                if i not in public_futures
                and not _start(target) - 1 <= i < (target.end_lineno or target.lineno)
            )
            prefix += future_source + remaining_source
            if not prefix.endswith("\n"):
                prefix += "\n"
            body = "".join(candidate_lines[end:])
    return AssembledCodeCandidate(
        prefix, body, candidate, "target-definition", assembly_version, True
    )
