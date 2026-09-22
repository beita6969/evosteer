"""Persisted action representations shared by collection, scoring and recovery."""

STRUCTURED_ACTION_WIRE = "structured-action-json@3"
STRICT_NATIVE_TOOL_WIRE = "native-single-tool-call@1"
NATIVE_TOOL_CARRIER_WIRE = "native-single-tool-call@2"
NATIVE_TOOL_HANDOFF_WIRE = "native-single-tool-call@3"
NATIVE_CARRIER_WIRES = frozenset({NATIVE_TOOL_CARRIER_WIRE, NATIVE_TOOL_HANDOFF_WIRE})
NATIVE_TOOL_WIRES = NATIVE_CARRIER_WIRES | {STRICT_NATIVE_TOOL_WIRE}
ACTION_WIRES = NATIVE_TOOL_WIRES | {STRUCTURED_ACTION_WIRE}
