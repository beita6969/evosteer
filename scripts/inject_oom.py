"""Validate an OOM injection selector without allocating CUDA memory."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OOMInjection:
    worker: str
    step: int
    microbatch: int

    def __post_init__(self) -> None:
        if self.worker not in {"gpu3", "gradient_primary"}:
            raise ValueError("OOM injection is allowed only on the steady gradient worker")
        if type(self.step) is not int or self.step <= 0:
            raise ValueError("injected OOM step must be positive")
        if type(self.microbatch) is not int or self.microbatch < 0:
            raise ValueError("injected OOM microbatch must be non-negative")


def parse_injection(value: str) -> OOMInjection:
    fields: dict[str, str] = {}
    for part in value.split(","):
        key, separator, item = part.partition("=")
        if not separator or key in fields:
            raise ValueError("OOM injection must contain unique key=value fields")
        fields[key] = item
    if set(fields) != {"worker", "step", "microbatch"}:
        raise ValueError("OOM injection requires worker, step, and microbatch")
    return OOMInjection(
        worker=fields["worker"],
        step=int(fields["step"]),
        microbatch=int(fields["microbatch"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selector")
    arguments = parser.parse_args()
    print(parse_injection(arguments.selector))


if __name__ == "__main__":
    main()
