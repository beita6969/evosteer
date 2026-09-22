import os

from skillev.training.run_clock import RunWallClock


def test_restart_includes_failed_work_and_downtime_without_a_committed_step(tmp_path):
    wall, mono = [100.0], [10.0]
    first = RunWallClock(tmp_path, wall_clock=lambda: wall[0], monotonic=lambda: mono[0])
    mono[0] = 30
    assert first.elapsed() == 20
    # The process vanished without a final write or an optimizer commit.
    wall[0], mono[0] = 150, 1
    resumed = RunWallClock(tmp_path, wall_clock=lambda: wall[0], monotonic=lambda: mono[0])
    assert resumed.elapsed() == 50
    mono[0] = 6
    assert resumed.elapsed() == 55


def test_legacy_origin_is_frozen_before_configuration_metadata_changes(tmp_path):
    config = tmp_path / "formal-config.json"
    config.write_text("{}")
    os.utime(config, (100, 100))
    first = RunWallClock(tmp_path, wall_clock=lambda: 150, monotonic=lambda: 0)
    assert first.elapsed() == 50
    assert first.origin == "existing-formal-config-mtime"
    os.utime(config, (200, 200))
    resumed = RunWallClock(tmp_path, wall_clock=lambda: 250, monotonic=lambda: 0)
    assert resumed.elapsed() == 150
