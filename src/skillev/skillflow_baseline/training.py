"""Ordered Protocol 10 specialization of the official-derived trainer."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol


class SkillFlowEpisodeRunner(Protocol):
    def run(self, question: dict[str, Any], trainer: Any) -> Any: ...


def build_protocol_v10_skillflow_trainer(
    config: dict[str, object],
    *,
    episode_runner: SkillFlowEpisodeRunner,
    checkpoint_callback: Callable[[int], None] | None = None,
) -> Any:
    """Construct the oracle trainer with only boundary-method overrides."""

    from training.gflownet_trainer import GFlowNetTrainer

    runtime_config = dict(config)
    runtime_config["formal_step_committed_callback"] = checkpoint_callback

    class ProtocolV10GFlowNetTrainer(GFlowNetTrainer):
        def _sample_batch_for_step(self, step: int) -> list[dict[str, Any]]:
            start = step * self.batch_size
            end = start + self.batch_size
            batch = self._train_data[start:end]
            if len(batch) != self.batch_size:
                raise RuntimeError("Protocol 10 ordered SkillFlow curriculum is exhausted")
            expected = tuple(range(start, end))
            observed = tuple(int(item["extra"]["protocol_v10_sequence_position"]) for item in batch)
            if observed != expected:
                raise RuntimeError("Protocol 10 SkillFlow batch order changed")
            return [dict(item) for item in batch]

        def _run_episode(self, question: dict[str, Any]) -> Any:
            return episode_runner.run(question, self)

    return ProtocolV10GFlowNetTrainer(runtime_config)


__all__ = ["SkillFlowEpisodeRunner", "build_protocol_v10_skillflow_trainer"]
