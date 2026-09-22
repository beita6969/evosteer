from evalplus.eval import FAIL, PASS, untrusted_check


def _check(source: str) -> str:
    status, _details = untrusted_check(
        "humaneval",
        source,
        [()],
        "candidate",
        [1],
        0,
        [0.01],
        fast_check=True,
        min_time_limit=0.1,
        gt_time_limit_factor=2.0,
    )
    return status


def test_evalplus_process_sandbox_classifies_handwritten_canaries() -> None:
    assert _check("def candidate():\n    return 1") == PASS
    assert _check("def candidate():\n    return 2") == FAIL
    assert _check("def candidate(:\n    pass") == FAIL
    # EvalPlus catches its per-input alarm inside unsafe_execute and reports a
    # definitive candidate failure; only a stuck sandbox process is TIMEOUT.
    assert _check("def candidate():\n    while True:\n        pass") == FAIL
    assert _check("def candidate():\n    return bytearray(5 * 1024**3)") == FAIL
