"""Durable, write-once filesystem primitives for one-shot Gate processes."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_bytes_once_atomic(path: Path, payload: bytes) -> None:
    """Publish bytes atomically while refusing to replace an existing record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.staging-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def write_text_once_atomic(path: Path, text: str) -> None:
    write_bytes_once_atomic(path, text.encode("utf-8"))


def append_line_durable(path: Path, line: str) -> None:
    """Append exactly one newline-terminated record and fsync it before return."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        payload = f"{line}\n".encode()
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError("short write to Gate 4c stage journal")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "append_line_durable",
    "fsync_directory",
    "write_bytes_once_atomic",
    "write_text_once_atomic",
]
