from pathlib import Path

import pytest
from skillev_private.experiments.protocol_v11_application_input import (
    ProtocolV11ApplicationInput,
)
from skillev_private.experiments.protocol_v11_launch_package import ProtocolV11LaunchPackage

from skillev.experiments.protocol_v11 import ProtocolV11Error, load_protocol_v11

ROOT = Path(__file__).parents[2]


def test_launch_package_is_inspectable_but_gate_is_closed() -> None:
    protocol = load_protocol_v11(
        ROOT / "configs/evaluation/protocol_v11.yaml",
        ROOT / "configs/evaluation/protocol_v11_sources.yaml",
    )
    launch = ProtocolV11LaunchPackage(
        ProtocolV11ApplicationInput(protocol, "bayesian-improve-full")
    )
    assert launch.total_steps == 320
    with pytest.raises(ProtocolV11Error):
        launch.require_executable()
