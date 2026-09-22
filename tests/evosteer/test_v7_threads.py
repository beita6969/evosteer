"""Thread safety of shared kernel dispatch in the pipelined loop."""

import threading
import time

import pytest


def test_autotuner_dispatch_is_serialized_idempotent_and_reentrant(monkeypatch):
    autotuner = pytest.importorskip("triton.runtime.autotuner")
    from skillev.policy.evosteer import _install_triton_autotuner_lock

    active, overlaps = [0], []

    class Base:
        # Stands in for triton's Autotuner: per-call state on the shared object.
        def run(self, *args, **kwargs):
            active[0] += 1
            overlaps.append(active[0])
            self.nargs = args
            time.sleep(0.01)
            seen = self.nargs
            self.nargs = None
            active[0] -= 1
            return seen

    class Cached(Base):
        # Like fla's cached autotuner: overrides run and calls super().run.
        def run(self, *args, **kwargs):
            return super().run(*args, **kwargs)

    monkeypatch.setattr(autotuner, "Autotuner", Base)
    _install_triton_autotuner_lock()
    wrapped = Cached.__dict__["run"]
    _install_triton_autotuner_lock()
    assert Cached.__dict__["run"] is wrapped  # Not wrapped twice.

    kernel = Cached()
    results = []
    threads = [
        threading.Thread(target=lambda i=i: results.append((i, kernel.run(i)))) for i in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    # Each call saw its own arguments, never another thread's cleared state.
    assert sorted(results) == [(i, (i,)) for i in range(8)]
    assert max(overlaps) == 1


def test_gil_switch_interval_is_opt_in_and_validated(monkeypatch):
    import sys

    from skillev.evosteer_cli import _gil_switch_interval

    before = sys.getswitchinterval()
    try:
        monkeypatch.delenv("EVOSTEER_GIL_SWITCH_INTERVAL", raising=False)
        _gil_switch_interval()
        assert sys.getswitchinterval() == before
        monkeypatch.setenv("EVOSTEER_GIL_SWITCH_INTERVAL", "0.0005")
        _gil_switch_interval()
        assert abs(sys.getswitchinterval() - 0.0005) < 1e-9
        for bad in ("0", "-1", "0.5"):
            monkeypatch.setenv("EVOSTEER_GIL_SWITCH_INTERVAL", bad)
            with pytest.raises(ValueError, match="EVOSTEER_GIL_SWITCH_INTERVAL"):
                _gil_switch_interval()
    finally:
        sys.setswitchinterval(before)
