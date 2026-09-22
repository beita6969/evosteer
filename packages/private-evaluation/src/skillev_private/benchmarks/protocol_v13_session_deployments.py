"""Load only deployments used by the authoritative Protocol 13 IID suite."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .alfworld_public_goal import LEGACY_GOAL_BINDING, require_goal_binding
from .protocol_v10_session_deployments import (
    PROTOCOL_V10_SESSION_DEPLOYMENTS_FORMAT,
    ProtocolV10ALFWorldDeployment,
    ProtocolV10HealthDeployment,
)


@dataclass(frozen=True, slots=True)
class Protocol13TrainingDeployments:
    alfworld: ProtocolV10ALFWorldDeployment
    healthbench: ProtocolV10HealthDeployment
    healthbench_repair_max_output_tokens: int = 1024
    alfworld_goal_binding: str = LEGACY_GOAL_BINDING

    def __post_init__(self) -> None:
        require_goal_binding(self.alfworld_goal_binding)
        if type(self.healthbench_repair_max_output_tokens) is not int or (
            self.healthbench_repair_max_output_tokens not in (1024, 4096)
        ):
            raise ValueError("unsupported HealthBench repair output budget")

    @classmethod
    def read(cls, path: Path) -> Protocol13TrainingDeployments:
        if not path.is_absolute() or not path.is_file():
            raise ValueError("training deployment input must be an absolute file")
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(data, dict)
            or data.get("format") != PROTOCOL_V10_SESSION_DEPLOYMENTS_FORMAT
        ):
            raise ValueError("training deployment format is unsupported")
        # Reuse the existing document and validators, not its obsolete suite.
        # WebShop/AppWorld/SpreadsheetBench are not current IID requirements. Missing
        # or stale sections for them must not demand unrelated environment copies.
        return cls(
            alfworld=ProtocolV10ALFWorldDeployment.from_value(data.get("alfworld")),
            healthbench=ProtocolV10HealthDeployment.from_value(data.get("healthbench")),
            alfworld_goal_binding=data.get("alfworld_goal_binding", LEGACY_GOAL_BINDING),
            healthbench_repair_max_output_tokens=data.get(
                "healthbench_repair_max_output_tokens", 1024
            ),
        )
