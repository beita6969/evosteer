"""Adapter-free direct evaluation primitives.

This package intentionally has no dependency on rollout, skill, adapter, or
training modules.  Answer-bearing populations and scorers live in the private
evaluation distribution.
"""

from .client import (
    DirectGenerationClient,
    DirectGenerationError,
    DirectGenerationRequest,
    DirectGenerationResult,
    OpenAICompatibleDirectClient,
    QwenChatTokenCounter,
)
from .config import (
    AggregateComponent,
    AggregateMetricSpec,
    BenchmarkComparability,
    ComparabilityEvidence,
    DirectBenchmark,
    DirectDecodingProfile,
    DirectParityPolicy,
    DirectReferenceProtocol,
    EvidenceStatus,
    MetricContract,
    PaperBenchmarkSpec,
    ParityMetric,
    SeedAggregationMode,
    SeedAggregationSpec,
)
from .interactive_tasks import (
    InteractiveProtocol,
    InteractiveStepRecordV2,
    InvalidCandidatePolicy,
    InvalidEnvironmentActionPolicy,
    NativeEnvironmentOutcome,
    NativeEnvironmentStep,
    NativeInteractiveAttempt,
    NativeInteractiveEnvironment,
    NativeInteractiveTask,
    NativePublicState,
    run_native_interactive_task,
)
from .prompts import WebShopPublicCatalogCandidate
from .reporting import (
    BenchmarkResult,
    DirectParityGateResult,
    EvaluationScope,
    GateStatus,
    MetricResult,
)
from .runner import (
    DirectAttempt,
    DirectTask,
    RawGenerationRecord,
    profile_for_seed,
    run_direct_tasks,
)

__all__ = [
    "AggregateComponent",
    "AggregateMetricSpec",
    "BenchmarkComparability",
    "BenchmarkResult",
    "ComparabilityEvidence",
    "DirectAttempt",
    "DirectBenchmark",
    "DirectDecodingProfile",
    "DirectGenerationClient",
    "DirectGenerationError",
    "DirectGenerationRequest",
    "DirectGenerationResult",
    "DirectParityGateResult",
    "DirectParityPolicy",
    "DirectReferenceProtocol",
    "DirectTask",
    "EvaluationScope",
    "EvidenceStatus",
    "GateStatus",
    "InteractiveProtocol",
    "InteractiveStepRecordV2",
    "InvalidCandidatePolicy",
    "InvalidEnvironmentActionPolicy",
    "MetricContract",
    "MetricResult",
    "NativeEnvironmentOutcome",
    "NativeEnvironmentStep",
    "NativeInteractiveAttempt",
    "NativeInteractiveEnvironment",
    "NativeInteractiveTask",
    "NativePublicState",
    "OpenAICompatibleDirectClient",
    "PaperBenchmarkSpec",
    "ParityMetric",
    "QwenChatTokenCounter",
    "RawGenerationRecord",
    "SeedAggregationMode",
    "SeedAggregationSpec",
    "WebShopPublicCatalogCandidate",
    "profile_for_seed",
    "run_direct_tasks",
    "run_native_interactive_task",
]
