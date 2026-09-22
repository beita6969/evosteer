"""Committed posterior evidence, including empty batches and recovery cursors.

Only invocation labels/weights and source coordinates are retained here, never
native evaluator payloads. Cells are the query snapshot; this journal permits
their arithmetic reconstruction at restore, without re-running calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

from skillev.contracts import (
    JsonValue,
    PosteriorBatchUpdate,
    PosteriorCellState,
    PosteriorUpdateEvent,
)

from .evidence_context import TrajectoryEvidenceContext

POSTERIOR_PROVENANCE_FORMAT = "skillev-posterior-event-provenance@2"


@dataclass(frozen=True, slots=True)
class PosteriorEvidenceBatch:
    optimizer_step: int
    policy_snapshot_id: str
    library_version: str
    trajectory_ids: tuple[str, ...]
    posterior: PosteriorBatchUpdate
    trajectory_contexts: tuple[TrajectoryEvidenceContext, ...] = ()

    def __post_init__(self) -> None:
        if type(self.optimizer_step) is not int or self.optimizer_step < 1:
            raise ValueError("posterior evidence step must be positive")
        if not self.policy_snapshot_id or not self.library_version:
            raise ValueError("posterior evidence requires policy and library identities")
        if not self.trajectory_ids or len(set(self.trajectory_ids)) != len(self.trajectory_ids):
            raise ValueError("posterior evidence requires unique trajectory identities")
        if (
            self.trajectory_contexts
            and tuple(item.trajectory_id for item in self.trajectory_contexts)
            != self.trajectory_ids
        ):
            raise ValueError("trajectory metadata differs from the committed batch order")
        if any(
            update.trajectory_id not in self.trajectory_ids for update in self.posterior.updates
        ):
            raise ValueError("posterior event targets a trajectory outside its batch")

        contexts = {item.trajectory_id: item for item in self.trajectory_contexts}
        for update in self.posterior.updates:
            context = contexts.get(update.trajectory_id)
            if context is None or context.invocation_links is None:
                continue  # Historical evidence remains explicitly unknown.
            matches = [
                link
                for link in context.invocation_links
                if link.step_index == update.step_index
                and link.declared_skill_id == update.skill_id
                and link.admitted
            ]
            if len(matches) != 1 or matches[0].terminal_success != update.outcome:
                raise ValueError(
                    "posterior event lacks its admitted declaration and terminal label"
                )

    def to_value(self) -> dict[str, JsonValue]:
        value: dict[str, JsonValue] = {
            "optimizer_step": self.optimizer_step,
            "policy_snapshot_id": self.policy_snapshot_id,
            "library_version": self.library_version,
            "trajectory_ids": list(self.trajectory_ids),
            "posterior": self.posterior.to_value(),
        }
        if self.trajectory_contexts:
            value["trajectory_contexts"] = [item.to_value() for item in self.trajectory_contexts]
        return value

    @classmethod
    def from_value(cls, value: object) -> PosteriorEvidenceBatch:
        if not isinstance(value, dict):
            raise TypeError("posterior evidence batch must be an object")
        return cls(
            optimizer_step=value["optimizer_step"],
            policy_snapshot_id=value["policy_snapshot_id"],
            library_version=value["library_version"],
            trajectory_ids=tuple(value["trajectory_ids"]),
            posterior=PosteriorBatchUpdate.from_value(value["posterior"]),
            trajectory_contexts=tuple(
                TrajectoryEvidenceContext.from_value(item)
                for item in value.get("trajectory_contexts", [])
            ),
        )


@dataclass(frozen=True, slots=True)
class PosteriorEventProvenance:
    batches: tuple[PosteriorEvidenceBatch, ...]
    format: str = POSTERIOR_PROVENANCE_FORMAT

    def __post_init__(self) -> None:
        if self.format != POSTERIOR_PROVENANCE_FORMAT:
            raise ValueError("posterior recovery requires complete event evidence")
        batch_ids: set[str] = set()
        trajectory_ids: set[str] = set()
        event_ids: set[str] = set()
        last_step = 0
        for batch in self.batches:
            if batch.optimizer_step <= last_step or batch.posterior.batch_id in batch_ids:
                raise ValueError("posterior evidence repeats or reorders a batch")
            if trajectory_ids.intersection(batch.trajectory_ids):
                raise ValueError("posterior trajectory evidence is already committed")
            for update in batch.posterior.updates:
                if update.event_id in event_ids:
                    raise ValueError("posterior update event is already committed")
                event_ids.add(update.event_id)
            last_step = batch.optimizer_step
            batch_ids.add(batch.posterior.batch_id)
            trajectory_ids.update(batch.trajectory_ids)

    @classmethod
    def empty(cls) -> PosteriorEventProvenance:
        return cls(batches=())

    def require_new_batch(self, batch_id: str, step: int, trajectory_ids: tuple[str, ...]) -> None:
        if self.batches and step <= self.batches[-1].optimizer_step:
            raise ValueError("posterior evidence step is already committed")
        for batch in self.batches:
            if batch.posterior.batch_id == batch_id or set(batch.trajectory_ids).intersection(
                trajectory_ids
            ):
                # Reject the entire retry, rather than silently renormalizing a subset.
                raise ValueError("posterior batch or trajectory evidence is already committed")

    def append_batch(self, batch: PosteriorEvidenceBatch) -> PosteriorEventProvenance:
        self.require_new_batch(batch.posterior.batch_id, batch.optimizer_step, batch.trajectory_ids)
        return PosteriorEventProvenance(batches=(*self.batches, batch))

    @property
    def updates(self) -> tuple[PosteriorUpdateEvent, ...]:
        return tuple(update for batch in self.batches for update in batch.posterior.updates)

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(update.event_id for update in self.updates)

    @property
    def cell_keys(self) -> tuple[str, ...]:
        return tuple(update.z.cell_key(update.skill_id) for update in self.updates)

    @property
    def skill_ids(self) -> tuple[str, ...]:
        return tuple(update.skill_id for update in self.updates)

    def event_ids_for_cell(self, cell_key: str) -> tuple[str, ...]:
        return tuple(
            update.event_id
            for update in self.updates
            if update.z.cell_key(update.skill_id) == cell_key
        )

    def event_ids_for_skill(self, skill_id: str) -> tuple[str, ...]:
        return tuple(update.event_id for update in self.updates if update.skill_id == skill_id)

    def require_reconstructed_cells(self, cells: tuple[PosteriorCellState, ...]) -> None:
        """Restore-time arithmetic check of labels, weights and rolling cell state."""

        expected = {cell.z.cell_key(cell.skill_id): cell for cell in cells}
        rolling: dict[str, PosteriorCellState] = {}
        for update in self.updates:
            key = update.z.cell_key(update.skill_id)
            if key not in expected:
                raise ValueError("posterior evidence targets an absent cell")
            if key not in rolling:
                cell = expected[key]
                rolling[key] = PosteriorCellState(
                    skill_id=cell.skill_id,
                    z=cell.z,
                    alpha=cell.alpha_0,
                    beta_count=cell.beta_0,
                    alpha_0=cell.alpha_0,
                    beta_0=cell.beta_0,
                )
            rolling[key] = rolling[key].apply(update)
        if rolling != expected:
            raise ValueError("posterior cells differ from committed weighted evidence")

    def to_value(self) -> dict[str, JsonValue]:
        return {"batches": [batch.to_value() for batch in self.batches], "format": self.format}

    @classmethod
    def from_value(cls, value: object) -> PosteriorEventProvenance:
        if not isinstance(value, dict) or value.get("format") != POSTERIOR_PROVENANCE_FORMAT:
            raise ValueError(
                "old ID-only posterior snapshots need their original event journal; "
                "automatic migration is unsupported"
            )
        return cls(
            batches=tuple(PosteriorEvidenceBatch.from_value(batch) for batch in value["batches"])
        )


def validate_posterior_provenance(
    *, cells: tuple[PosteriorCellState, ...], provenance: PosteriorEventProvenance
) -> None:
    """Cheap live consistency check; full arithmetic reconstruction is restore-only."""

    cells_by_key = {cell.z.cell_key(cell.skill_id): cell for cell in cells}
    if len(cells_by_key) != len(cells):
        raise ValueError("projection repeats a posterior cell")
    by_cell: dict[str, list[PosteriorUpdateEvent]] = {}
    for update in provenance.updates:
        by_cell.setdefault(update.z.cell_key(update.skill_id), []).append(update)
    if set(by_cell) != set(cells_by_key):
        raise ValueError("posterior provenance cells differ")
    for key, cell in cells_by_key.items():
        updates = by_cell[key]
        if len(updates) != cell.update_count or updates[-1].event_id != cell.last_event_id:
            raise ValueError("posterior cell event cursor differs")
