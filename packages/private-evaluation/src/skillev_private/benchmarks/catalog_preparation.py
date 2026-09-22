"""Bind prepared benchmark receipts and lexical retrieval to one catalog.

This is the final result-blind join in dataset preparation.  It accepts only
the two already-published preparation receipts and the retrieval index named by
the committed acquisition lock.  No source discovery, conversion, download,
or substitution happens here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from skillev.contracts import normalize_json, validate_sha256

from .acquisition import BenchmarkAcquisitionLock
from .catalog_freeze import (
    FrozenProductionCatalogBundle,
    freeze_production_catalog_bundle,
)
from .non_process_preparation import LockedNonProcessPreparationReceipt
from .process_preparation import LockedProcessPreparationReceipt
from .retrieval_preparation import verify_retrieval_preparation_output

ATLAS_RETRIEVAL_INFRASTRUCTURE_KEY = "atlas_wikipedia_fts5"


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"{field} must be non-empty text without NUL")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class LockedRetrievalInfrastructure:
    """Exact lexical retrieval identity committed in the acquisition lock."""

    retrieval_backend: str
    corpus_source: str
    source_locator: str
    source_sha256: str
    source_size: int
    passage_count: int
    index_id: str

    def __post_init__(self) -> None:
        for field in ("retrieval_backend", "corpus_source", "source_locator"):
            _text(getattr(self, field), field=field)
        validate_sha256(self.source_sha256)
        _positive_int(self.source_size, field="source_size")
        _positive_int(self.passage_count, field="passage_count")
        validate_sha256(self.index_id)

    @classmethod
    def from_lock(
        cls,
        lock: BenchmarkAcquisitionLock,
    ) -> LockedRetrievalInfrastructure:
        if not isinstance(lock, BenchmarkAcquisitionLock):
            raise TypeError("lock must be BenchmarkAcquisitionLock")
        raw = normalize_json(lock.shared_infrastructure.get(ATLAS_RETRIEVAL_INFRASTRUCTURE_KEY))
        expected = {
            "corpus_source",
            "index_id",
            "passage_count",
            "retrieval_backend",
            "source_locator",
            "source_sha256",
            "source_size",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("Atlas retrieval infrastructure has an invalid field set")
        raw_sha256 = _text(raw["source_sha256"], field="source_sha256")
        if (
            len(raw_sha256) != 64
            or raw_sha256 != raw_sha256.lower()
            or any(character not in "0123456789abcdef" for character in raw_sha256)
        ):
            raise ValueError("source_sha256 must be a lowercase raw SHA-256 digest")
        return cls(
            retrieval_backend=_text(
                raw["retrieval_backend"],
                field="retrieval_backend",
            ),
            corpus_source=_text(raw["corpus_source"], field="corpus_source"),
            source_locator=_text(raw["source_locator"], field="source_locator"),
            source_sha256=f"sha256:{raw_sha256}",
            source_size=_positive_int(raw["source_size"], field="source_size"),
            passage_count=_positive_int(
                raw["passage_count"],
                field="passage_count",
            ),
            index_id=_text(raw["index_id"], field="index_id"),
        )


def _relative_file(path: Path, *, root: Path, field: str) -> str:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError(f"{field} must be an absolute Path")
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{field} must be below the prepared target root") from error
    return relative.as_posix()


def freeze_locked_production_catalog(
    *,
    lock: BenchmarkAcquisitionLock,
    non_process_receipt: LockedNonProcessPreparationReceipt,
    process_receipt: LockedProcessPreparationReceipt,
    retrieval_index_path: Path,
    retrieval_manifest_path: Path,
) -> FrozenProductionCatalogBundle:
    """Verify all prepared identities and freeze the active source catalog."""

    if not isinstance(lock, BenchmarkAcquisitionLock):
        raise TypeError("lock must be BenchmarkAcquisitionLock")
    if not isinstance(non_process_receipt, LockedNonProcessPreparationReceipt):
        raise TypeError("non_process_receipt must be LockedNonProcessPreparationReceipt")
    if not isinstance(process_receipt, LockedProcessPreparationReceipt):
        raise TypeError("process_receipt must be LockedProcessPreparationReceipt")
    lock_hash = lock.content_hash
    if (
        non_process_receipt.acquisition_lock_hash != lock_hash
        or process_receipt.acquisition_lock_hash != lock_hash
    ):
        raise ValueError("preparation receipt belongs to a different acquisition lock")
    if non_process_receipt.target_root != process_receipt.target_root:
        raise ValueError("preparation receipts use different target roots")
    if (
        non_process_receipt.acquisition_receipt_hash != process_receipt.acquisition_receipt_hash
        or non_process_receipt.archive_batch_receipt_hash
        != process_receipt.archive_batch_receipt_hash
    ):
        raise ValueError("preparation receipts do not share one acquisition chain")

    target_root = non_process_receipt.target_root
    if not target_root.is_dir():
        raise NotADirectoryError(target_root)
    retrieval_relative_path = _relative_file(
        retrieval_index_path,
        root=target_root,
        field="retrieval_index_path",
    )
    _relative_file(
        retrieval_manifest_path,
        root=target_root,
        field="retrieval_manifest_path",
    )
    prepared = verify_retrieval_preparation_output(
        index_path=retrieval_index_path,
        manifest_path=retrieval_manifest_path,
    )
    locked = LockedRetrievalInfrastructure.from_lock(lock)
    actual = prepared.index_manifest
    if (
        prepared.input_raw_sha256 != locked.source_sha256
        or prepared.input_size_bytes != locked.source_size
        or actual.corpus_version != locked.source_sha256
        or actual.passage_count != locked.passage_count
        or actual.retrieval_backend != locked.retrieval_backend
        or actual.index_id != locked.index_id
    ):
        raise ValueError("published retrieval index differs from the acquisition lock")

    return freeze_production_catalog_bundle(
        dataset_root=target_root,
        non_process_specs=non_process_receipt.freeze_specs(),
        process_specs=process_receipt.freeze_specs(),
        retrieval_index_relative_path=retrieval_relative_path,
        retrieval_index_id=actual.index_id,
    )


__all__ = [
    "ATLAS_RETRIEVAL_INFRASTRUCTURE_KEY",
    "LockedRetrievalInfrastructure",
    "freeze_locked_production_catalog",
]
