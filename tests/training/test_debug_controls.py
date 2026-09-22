import pytest

from skillev.diagnostics.debug_controls import NoUpdateControlReceipt, OneStepDebugReceipt


def test_no_update_and_one_step_receipts() -> None:
    NoUpdateControlReceipt("structured-no-skill", 16, 16, 0, 0, 0, False, 0).validate()
    OneStepDebugReceipt(16, 16, 1, 1, 1, 1, 1, 0).validate()
    with pytest.raises(ValueError):
        OneStepDebugReceipt(16, 16, 2, 1, 1, 1, 1, 0).validate()
