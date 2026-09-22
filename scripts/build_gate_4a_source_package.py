#!/usr/bin/env python3
"""Build the immutable source archive consumed by real Gate 4a."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from skillev.experiments.build_identity import write_source_package_provenance


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _git(root: Path, *arguments: str) -> str:
    executable = shutil.which("git")
    if executable is None:
        raise FileNotFoundError("git")
    completed = subprocess.run(  # noqa: S603 -- resolved executable, fixed call sites
        [executable, *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip()


def build_source_package(*, repository: Path, output: Path) -> None:
    """Archive the exact committed tree plus its self-verifying provenance."""

    repository = repository.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if _git(repository, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("source package requires a clean tracked worktree")
    source_commit = _git(repository, "rev-parse", "HEAD")
    executable = shutil.which("git")
    if executable is None:
        raise FileNotFoundError("git")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="skillev-gate-4a-source-") as temporary:
        temporary_root = Path(temporary)
        git_archive = temporary_root / "source.tar"
        subprocess.run(  # noqa: S603 -- resolved executable, immutable commit argument
            [
                executable,
                "archive",
                "--format=tar",
                f"--output={git_archive}",
                source_commit,
            ],
            cwd=repository,
            check=True,
            timeout=60,
        )
        source_root = temporary_root / "source"
        source_root.mkdir()
        with tarfile.open(git_archive, mode="r:") as archive:
            archive.extractall(source_root, filter="data")
        write_source_package_provenance(source_root, source_commit=source_commit)
        with tarfile.open(output, mode="w:gz", compresslevel=6) as archive:
            for path in sorted(source_root.rglob("*")):
                archive.add(path, arcname=path.relative_to(source_root), recursive=False)
        shutil.copystat(repository, output, follow_symlinks=False)


def main() -> None:
    args = _args()
    build_source_package(
        repository=Path(__file__).resolve().parents[1],
        output=Path(args.output),
    )


if __name__ == "__main__":
    main()
