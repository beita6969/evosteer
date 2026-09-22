import sys
from types import ModuleType

from skillev.runtime.sglang_timing import install_request_timing_transport


def test_measured_timing_survives_two_ipc_hops_without_reenabling_collectors(monkeypatch):
    module = ModuleType("sglang.srt.observability.req_time_stats")
    module.global_diff_realtime_monotonic = 1000.0

    class Stats:
        enable_metrics = False
        wait_queue_entry_time = forward_entry_time = prefill_finished_time = 0.0

        def __getstate__(self):
            if not self.enable_metrics:
                return {}
            return {
                "wait_queue_entry_time": self.wait_queue_entry_time,
                "forward_entry_time": self.forward_entry_time,
                "prefill_finished_time": self.prefill_finished_time,
                "diff_realtime_monotonic": module.global_diff_realtime_monotonic,
            }

        def __setstate__(self, state):
            for key in state:
                if key.endswith("time"):
                    state[key] += (
                        state["diff_realtime_monotonic"] - module.global_diff_realtime_monotonic
                    )
            self.__dict__.update(state)

    module.SchedulerReqTimeStats = Stats
    monkeypatch.setitem(sys.modules, module.__name__, module)
    source = Stats()
    source.enable_metrics = True
    source.wait_queue_entry_time = 10.0
    source.forward_entry_time = 10.25
    source.prefill_finished_time = 10.75
    install_request_timing_transport()
    installed = Stats.__getstate__
    install_request_timing_transport()
    assert Stats.__getstate__ is installed
    for _ in range(3):
        value = source.__getstate__()
        module.global_diff_realtime_monotonic += 0.125
        received = Stats()
        received.__setstate__(value)
        assert not received.enable_metrics
        assert received.forward_entry_time - received.wait_queue_entry_time == 0.25
        assert received.prefill_finished_time - received.forward_entry_time == 0.5
        assert received.forward_entry_time + module.global_diff_realtime_monotonic == 1010.25
        source = received
    assert Stats().__getstate__() == {}
    missing = Stats()
    missing.enable_metrics = True
    missing.forward_entry_time = 20.0
    value = missing.__getstate__()
    module.global_diff_realtime_monotonic += 10
    received = Stats()
    received.__setstate__(value)
    assert received.prefill_finished_time == 0.0
    assert received.wait_queue_entry_time == 0.0
