"""Public instrument semantics, never a task solution or hidden-state lookup.

Electrical facts follow allenai/ScienceWorld issue 50 and the released
Generator.tick / PolarizedElectricalComponent.tick implementations. The
deployed 1.2.3 JVM bytecode was inspected after an observed polarity mismatch.
This catalog has no episode, object, score, rubric, or reference-answer input.
"""

EQUIPMENT_PROFILE = "scienceworld-public-instruments@1"


def equipment_reference() -> str:
    return (
        f" Public instrument reference ({EQUIPMENT_PROFILE}): "
        "Electrical terminal names in this simulator distinguish sources from loads. "
        "An operating source supplies voltage at its anode and ground at its cathode. "
        "A polarized load becomes powered when its cathode receives voltage and its "
        "anode has a conducting path to ground. Thus matching the same terminal names "
        "on source and load is not the polarity convention used here. "
        "This describes the device interface, not whether any particular material "
        "conducts or whether your circuit is complete. The native connection and device "
        "observations remain the evidence of the circuit you actually constructed. "
        "Measurements are snapshots of state at the reported action. Temperature and "
        "other physical properties can change on subsequent simulator ticks; measuring "
        "an object does not freeze its state. Native ticking, device activation and "
        "all task decisions remain unchanged."
    )
