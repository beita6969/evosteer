"""Trusted overlap identities derived from private Protocol 10 task records.

Population files may declare identities for convenient inspection, but those
declarations are never authoritative.  This materializer rebuilds identities
from the source record identifier and the answer-free task projection each
time a population is opened.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from skillev.experiments import BenchmarkV10
from skillev.experiments.protocol_v10 import BenchmarkPopulation
from skillev.rollout import RolloutTask

from .protocol_v10_population import PopulationOverlapKey, PopulationOverlapKind

IDENTITY_MATERIALIZER_VERSION = "protocol-v10-task-overlap@4"


def normalize_public_content(value: str) -> str:
    """Normalize public task text without consulting private evaluator truth."""

    if type(value) is not str or not value.strip():
        raise ValueError("public task content must be non-empty text")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def _context_text(task: RolloutTask, field: str) -> str:
    context = task.public_context
    if not isinstance(context, dict):
        raise ValueError("Protocol 10 public context must be an object")
    value = context.get(field)
    if type(value) is not str or not value.strip():
        raise ValueError(f"Protocol 10 public context lacks {field}")
    return normalize_public_content(value)


@dataclass(frozen=True, slots=True)
class ProtocolV10TaskIdentityMaterializer:
    """Reconstruct item and group identities for one frozen population."""

    version: str = IDENTITY_MATERIALIZER_VERSION

    def materialize(
        self,
        *,
        spec: BenchmarkPopulation,
        source_id: str,
        task: RolloutTask,
    ) -> tuple[PopulationOverlapKey, ...]:
        if task.task_id != source_id:
            raise ValueError("source ID and task ID must match before episode repetition")
        context = task.public_context
        if not isinstance(context, dict) or context.get("benchmark_id") != spec.benchmark.value:
            raise ValueError("task benchmark differs from its population")

        public_content = task.query
        if spec.benchmark in {BenchmarkV10.ALFWORLD, BenchmarkV10.APPWORLD}:
            # Interactive tasks may intentionally expose the same instruction
            # while starting from different public environment states.  The
            # source task identity selects that state, and the scenario key
            # below prevents a variation group from crossing populations.
            public_content = f"{task.query}\ninteractive-task-variation:{source_id}"
        elif spec.benchmark is BenchmarkV10.MBPP_PLUS_FIXED_100:
            # Code tasks can share a natural-language instruction while
            # exposing different required function signatures.  Both fields
            # are model-visible and together define the item content.
            public_content = f"{task.query}\ncode-signature:{_context_text(task, 'code_signature')}"

        keys = [
            PopulationOverlapKey(
                PopulationOverlapKind.SOURCE_RECORD,
                normalize_public_content(source_id),
            ),
            PopulationOverlapKey(
                PopulationOverlapKind.NORMALIZED_PUBLIC_CONTENT,
                normalize_public_content(public_content),
            ),
        ]
        if spec.benchmark is BenchmarkV10.MBPP_PLUS_FIXED_100:
            keys.append(
                PopulationOverlapKey(
                    PopulationOverlapKind.CODE_SIGNATURE,
                    _context_text(task, "code_signature"),
                )
            )
        if spec.benchmark is BenchmarkV10.SPREADSHEETBENCH:
            keys.append(
                PopulationOverlapKey(
                    PopulationOverlapKind.WORKBOOK_STRUCTURE,
                    _context_text(task, "workbook_structure_id"),
                )
            )
        if spec.benchmark in {
            BenchmarkV10.WEBSHOP,
            BenchmarkV10.ALFWORLD,
            BenchmarkV10.APPWORLD,
        }:
            keys.append(
                PopulationOverlapKey(
                    PopulationOverlapKind.INTERACTIVE_SCENARIO,
                    _context_text(task, "scenario_id"),
                )
            )
        return tuple(keys)


def protocol_v10_identity_materializers(
    protocol: object,
) -> dict[str, ProtocolV10TaskIdentityMaterializer]:
    """Return the same trusted algorithm for every declared population."""

    from skillev.experiments.protocol_v10 import ActiveBenchmarkProtocolV10

    if not isinstance(protocol, ActiveBenchmarkProtocolV10):
        raise TypeError("identity routes require the active Protocol 10")
    return {
        population.population_id: ProtocolV10TaskIdentityMaterializer()
        for benchmark in protocol.benchmarks
        for population in benchmark.populations
    }


__all__ = [
    "IDENTITY_MATERIALIZER_VERSION",
    "ProtocolV10TaskIdentityMaterializer",
    "normalize_public_content",
    "protocol_v10_identity_materializers",
]
