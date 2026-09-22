"""Compose the public HumanEval scaffold with one unchanged owner submission."""

from __future__ import annotations

import ast


def _function_start(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    return min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)]) - 1


def _future_import(node: ast.stmt) -> bool:
    return isinstance(node, ast.ImportFrom) and node.module == "__future__"


def legacy_humaneval_source_parts(prompt: str, candidate: str) -> tuple[str, str]:
    """Accept native body completions and complete functions, without tests.

    The last public function is the unfinished entry point. A complete owner
    implementation replaces that stub, not the earlier public helpers/imports.
    This is harness assembly only: the stored candidate is never changed.
    """
    if candidate.startswith(prompt):
        return prompt, candidate[len(prompt) :]
    try:
        module = ast.parse(candidate)
    except SyntaxError:
        return prompt, candidate
    # A native prompt can end at the function header rather than a docstring.
    public = ast.parse(prompt + "\n    pass\n")
    functions = [
        node for node in public.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    if not functions:
        raise ValueError("HumanEval public prompt has no entry function")
    target = functions[-1]
    if not any(
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == target.name
        for node in module.body
    ):
        return prompt, candidate
    prefix = "".join(prompt.splitlines(keepends=True)[: _function_start(target)])
    if not prefix.strip() or candidate.startswith(prefix):
        return "", candidate
    futures = [node for node in module.body if _future_import(node)]
    if not futures:
        return prefix, candidate

    # Inserting helpers before an owner's future import would make valid Python
    # fail compilation. Insert after the original module docstring/future header
    # instead. Keep every candidate byte in its original order.
    header_end = futures[-1].end_lineno
    assert header_end is not None
    lines = candidate.splitlines(keepends=True)
    header, body = "".join(lines[:header_end]), "".join(lines[header_end:])
    # Public future imports must also precede ordinary statements. Relocate only
    # this public scaffold syntax; no candidate or private-test text is rewritten.
    prefix_lines = prefix.splitlines(keepends=True)
    public_future_lines = {
        line
        for node in public.body
        if _future_import(node)
        for line in range(node.lineno - 1, node.end_lineno or node.lineno)
    }
    imports = "".join(line for i, line in enumerate(prefix_lines) if i in public_future_lines)
    rest = "".join(line for i, line in enumerate(prefix_lines) if i not in public_future_lines)
    return header + imports + rest, body


def humaneval_source_parts(
    prompt: str, candidate: str, *, entry_point: str | None = None
) -> tuple[str, str]:
    """Current shared public-context contract; legacy replay is explicitly named above."""
    from skillev_private.benchmarks.public_code_context import (
        CodeSubmission,
        PublicCodeContext,
        compose_code_candidate,
        public_entry_point,
    )

    assembled = compose_code_candidate(
        PublicCodeContext(prompt, entry_point or public_entry_point(prompt)),
        CodeSubmission(candidate),
    )
    return assembled.prefix, assembled.completion
