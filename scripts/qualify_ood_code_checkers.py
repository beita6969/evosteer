"""Execute fixed synthetic programs through deployed official checkers, CPU sandbox only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skillev_private.evaluation.ood_scoring import grade_code

from skillev.evaluation.actor_sandbox import ActorSandbox


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.runtime_config.read_text())["scorers"]
    sandbox = ActorSandbox.current(Path(__file__).resolve().parents[1] / "src")
    rows = []
    for benchmark in ("apps-introductory", "livecodebench"):
        stdin = {"inputs": ["4\n", "6\n"], "outputs": ["8\n", "12\n"], "fn_name": None}
        callable_tests = {
            "inputs": [[4], [6]] if benchmark == "apps-introductory" else ["4", "6"],
            "outputs": [8, 12] if benchmark == "apps-introductory" else ["8", "12"],
            "fn_name": "double",
        }
        cases = [
            ("stdin", "print(int(input()) * 2)", stdin, True),
            ("main-guard", "if __name__ == '__main__':\n    print(int(input()) * 2)", stdin, True),
            (
                "callable",
                "class Solution:\n    def double(self, x):\n        return 2*x",
                callable_tests,
                True,
            ),
            ("wrong-answer", "print(0)", stdin, False),
            ("syntax-error", "def broken(", stdin, False),
            ("runtime-error", "raise RuntimeError('synthetic failure')", stdin, False),
            ("timeout", "while True:\n    pass", stdin, False),
        ]
        for name, candidate, in_out, expected in cases:
            row = {"benchmark": benchmark, "case": name, "expected_pass": expected}
            try:
                result = grade_code(
                    benchmark, candidate, {"input_output": in_out}, settings[benchmark], sandbox
                )
            except Exception as error:
                row.update(
                    matched=False, infrastructure_error=type(error).__name__, detail=str(error)
                )
            else:
                row.update(matched=result["passed"] is expected, native_result=result)
            rows.append(row)
    output = {
        "model_calls": 0,
        "planned": len(rows),
        "matched": sum(row["matched"] for row in rows),
        "rows": rows,
    }
    with args.output.open("x") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: value for key, value in output.items() if key != "rows"}))
    if not all(row["matched"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
