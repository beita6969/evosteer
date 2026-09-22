"""Keep consumed source coordinates while removing future domain occurrences."""

from collections import Counter
from pathlib import Path
from typing import Any

from skillev.training.checkpoint import FilesystemTrainingCheckpointStore
from skillev_private.benchmarks.protocol_v13_seven_training import (
    seven_domain_training_trajectories,
)
from skillev_private.benchmarks.protocol_v13_training import Protocol13TrainingRecord

from .bayesian_condition_transition import condition_configs
from .bayesian_training_config import BayesianFormalConfig


def training_schedule(
    config: BayesianFormalConfig,
    records: tuple[Protocol13TrainingRecord, ...],
    *,
    root: Path,
    resume: Path | None = None,
) -> tuple[Protocol13TrainingRecord, ...]:
    starts = {1: config.scheduled_domains}
    if (root / "formal-config.json").exists():
        history = condition_configs(root)
        matching = [step for step, value in history.items() if value == config]
        if matching:
            history = {step: value for step, value in history.items() if step <= max(matching)}
        starts = {step: value.scheduled_domains for step, value in history.items()}
        if starts[max(starts)] != config.scheduled_domains:
            if resume is None:
                raise ValueError("domain continuation needs a complete checkpoint")
            saved = FilesystemTrainingCheckpointStore(root=root / "checkpoints").load_metadata(
                resume
            )
            starts[saved.optimizer_step + 1] = config.scheduled_domains
    return seven_domain_training_trajectories(records, steps=config.steps, domain_starts=starts)


def schedule_summary(
    config: BayesianFormalConfig, selected: tuple[Protocol13TrainingRecord, ...]
) -> dict[str, Any]:
    """Account for original B28 prefixes and new B24 suffixes without rewriting either."""
    sizes = Counter(r.episode.optimizer_step for r in selected)
    segments: list[dict[str, int]] = []
    for step, size in sorted(sizes.items()):
        if not segments or segments[-1]["batch_size"] != size:
            segments.append({"first_step": step, "last_step": step, "batch_size": size})
        else:
            segments[-1]["last_step"] = step
    return {
        **config.schedule_summary(),
        "trajectories": len(selected),
        "question_occurrences": len(selected) // 4,
        "batch_size_segments": segments,
    }
