from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from skillev.experiments.build_identity import (
    SOURCE_PACKAGE_PROVENANCE_FILE,
    read_source_package_provenance,
    require_source_archive_matches_execution,
    write_source_package_provenance,
)


def test_source_archive_is_bound_to_executing_tree_and_embedded_commit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "src").mkdir()
    (source / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / ".private").mkdir()
    (source / ".private" / "runtime.bin").write_bytes(b"not source")
    provenance = write_source_package_provenance(
        source,
        source_commit="a" * 40,
    )
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, mode="w:gz") as stream:
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            if ".private" in relative.parts:
                continue
            stream.add(path, arcname=relative, recursive=False)

    assert read_source_package_provenance(source) == provenance
    assert (
        require_source_archive_matches_execution(
            source_archive=archive,
            executing_root=source,
            temporary_parent=tmp_path,
        )
        == provenance
    )
    assert (source / SOURCE_PACKAGE_PROVENANCE_FILE).is_file()


def test_source_identity_rejects_executing_bytes_not_in_archive(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    write_source_package_provenance(source, source_commit="b" * 40)
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, mode="w:gz") as stream:
        for path in sorted(source.iterdir()):
            stream.add(path, arcname=path.name, recursive=False)

    (source / "module.py").write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(ValueError):
        require_source_archive_matches_execution(
            source_archive=archive,
            executing_root=source,
            temporary_parent=tmp_path,
        )
