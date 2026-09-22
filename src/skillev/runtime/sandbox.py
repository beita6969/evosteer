"""Small filesystem primitives for offline, per-invocation workspaces."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class SandboxViolationError(PermissionError):
    """A path or operation falls outside the configured sandbox boundary."""


def validate_relative_path(value: str) -> PurePosixPath:
    """Validate a portable relative path without touching the filesystem."""

    if not isinstance(value, str):
        raise TypeError("Sandbox paths must be strings")
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or "\x00" in value
    ):
        raise SandboxViolationError("Path is outside the sandbox namespace")
    return path


@dataclass(frozen=True, slots=True)
class SandboxPolicy:
    """Method-neutral defaults for a local tool workspace."""

    network_enabled: bool = False
    package_install_enabled: bool = False
    inherited_environment: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.network_enabled or self.package_install_enabled:
            raise ValueError("The local execution policy must remain offline and immutable")
        if len(set(self.inherited_environment)) != len(self.inherited_environment):
            raise ValueError("Inherited environment names must be unique")
        for name in self.inherited_environment:
            if not name or "=" in name or "\x00" in name:
                raise ValueError("Inherited environment names must be valid")


class SandboxWorkspace:
    """Byte-bounded storage rooted below one private directory.

    Paths use POSIX separators even when the host platform differs. Writes are
    staged in the destination directory and atomically replaced, so readers
    never observe a partially written value.
    """

    def __init__(
        self,
        root: Path,
        *,
        max_file_bytes: int,
        max_path_bytes: int = 1024,
    ) -> None:
        if max_file_bytes < 1 or max_path_bytes < 1:
            raise ValueError("Workspace byte limits must be positive")
        self.root = root
        self.max_file_bytes = max_file_bytes
        self.max_path_bytes = max_path_bytes
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        self._resolved_root = self.root.resolve(strict=True)

    def write_bytes(self, relative_path: str, payload: bytes) -> Path:
        if not isinstance(payload, bytes):
            raise TypeError("Workspace payloads must be bytes")
        if len(payload) > self.max_file_bytes:
            raise ValueError("Workspace payload exceeds the file byte limit")
        target = self._safe_path(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._ensure_within_root(target.parent.resolve(strict=True))

        descriptor, staging_name = tempfile.mkstemp(prefix=".write-", dir=target.parent)
        staging = Path(staging_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(staging, 0o600)
            os.replace(staging, target)
        finally:
            staging.unlink(missing_ok=True)
        return target

    def read_bytes(self, relative_path: str) -> bytes:
        target = self._safe_path(relative_path)
        if not target.is_file() or target.is_symlink():
            raise SandboxViolationError("Workspace reads require a regular file")
        with target.open("rb") as stream:
            payload = stream.read(self.max_file_bytes + 1)
        if len(payload) > self.max_file_bytes:
            raise ValueError("Workspace file exceeds the read byte limit")
        return payload

    def exists(self, relative_path: str) -> bool:
        target = self._safe_path(relative_path)
        return target.is_file() and not target.is_symlink()

    def _safe_path(self, relative_path: str) -> Path:
        if len(relative_path.encode("utf-8")) > self.max_path_bytes:
            raise SandboxViolationError("Workspace path exceeds the byte limit")
        logical = validate_relative_path(relative_path)
        candidate = self.root.joinpath(*logical.parts)
        self._ensure_within_root(candidate.resolve(strict=False))
        return candidate

    def _ensure_within_root(self, resolved: Path) -> None:
        if resolved != self._resolved_root and self._resolved_root not in resolved.parents:
            raise SandboxViolationError("Path resolves outside the workspace")
