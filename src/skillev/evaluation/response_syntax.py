"""Locate response framing without interpreting Python string contents as controls."""

from __future__ import annotations

import io
import token
import tokenize
from collections.abc import Iterator


def python_literal_lines(text: str) -> set[int]:
    """Return one-based lines whose beginning is inside a Python string literal.

    Tokenization does not require a syntactically valid program: a response may
    also contain prose or markdown. This only recognizes literal boundaries;
    it does not check correctness, repair source, or select a program.
    """
    lines: set[int] = set()
    formatted_starts: list[int] = []
    try:
        for item in tokenize.generate_tokens(io.StringIO(text).readline):
            if item.type == token.STRING:
                lines.update(range(item.start[0] + 1, item.end[0] + 1))
            # Python 3.12+ tokenizes f-strings separately. Treat the whole
            # literal as data, including any interpolation expression.
            name = token.tok_name[item.type]
            if name in {"FSTRING_START", "TSTRING_START"}:
                formatted_starts.append(item.start[0])
            elif name in {"FSTRING_END", "TSTRING_END"} and formatted_starts:
                lines.update(range(formatted_starts.pop() + 1, item.end[0] + 1))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Non-code prose is allowed; keep any complete literals already read.
        pass
    return lines


def protocol_lines(text: str) -> Iterator[tuple[int, str]]:
    """Yield offsets and whole lines outside fenced code and Python literals."""
    literals = python_literal_lines(text)
    offset, fenced = 0, False
    for number, line in enumerate(text.splitlines(keepends=True), start=1):
        if number not in literals:
            if line.lstrip().startswith("```"):
                fenced = not fenced
            elif not fenced:
                yield offset, line
        offset += len(line)
