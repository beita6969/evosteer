"""Training-source views accepted only from the four-source reducer."""

from __future__ import annotations

from collections.abc import Iterator

from skillev.contracts import TrainingStepCommit

from .source_reducer import CommittedLibrarySegment


def iter_training_step_commits(
    segments: tuple[CommittedLibrarySegment, ...],
) -> Iterator[TrainingStepCommit]:
    if not isinstance(segments, tuple) or any(
        not isinstance(segment, CommittedLibrarySegment) for segment in segments
    ):
        raise TypeError("training audit requires committed library segments")
    for segment in segments:
        yield from segment.training_steps


__all__ = ["iter_training_step_commits"]
