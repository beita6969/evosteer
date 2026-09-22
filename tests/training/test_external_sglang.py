from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from training.external_sglang import publish_external_adapter


class _Response:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


class _SGLangControl:
    def __init__(self, adapters: set[str], *, capacity: int = 2) -> None:
        self.adapters = set(adapters)
        self.capacity = capacity
        self.calls: list[tuple[str, str]] = []
        self.fail_validation = False

    def get(self, url: str, **_: object) -> _Response:
        self.calls.append(("get", url.rsplit("/", 1)[-1]))
        return _Response(
            {"data": [{"id": "skillflow-qwen35"}, *({"id": name} for name in self.adapters)]}
        )

    def post(self, url: str, **kwargs: object) -> _Response:
        operation = url.rsplit("/", 1)[-1]
        self.calls.append(("post", operation))
        payload = kwargs.get("json")
        assert isinstance(payload, dict)
        if operation == "unload_lora_adapter":
            self.adapters.remove(str(payload["lora_name"]))
            return _Response({"success": True})
        if operation == "load_lora_adapter":
            if len(self.adapters) >= self.capacity:
                return _Response({}, status_code=400)
            self.adapters.add(str(payload["lora_name"]))
            return _Response({"success": True})
        assert operation == "completions"
        return _Response({"choices": [] if self.fail_validation else [{"text": "OK"}]})


def _publish(
    control: _SGLangControl,
    *,
    adapter_name: str,
    previous_adapter: str = "theta_uninitialized",
) -> None:
    publish_external_adapter(
        api_base="http://127.0.0.1:8005",
        headers={"Authorization": "Bearer test"},
        adapter_name=adapter_name,
        adapter_path=Path("/private/adapter"),
        previous_adapter=previous_adapter,
        managed_prefix="theta_step_",
        request_timeout_seconds=5,
        max_retries=0,
        http=control,
    )


def test_fresh_process_removes_stale_managed_adapters_before_loading() -> None:
    control = _SGLangControl({"theta_step_old_1", "theta_step_old_2"})

    _publish(control, adapter_name="theta_step_new_run_000000")

    assert control.adapters == {"theta_step_new_run_000000"}
    load_index = control.calls.index(("post", "load_lora_adapter"))
    assert control.calls[:load_index].count(("post", "unload_lora_adapter")) == 2


def test_replacement_keeps_current_adapter_until_new_version_is_validated() -> None:
    previous = "theta_step_run_000000"
    control = _SGLangControl({previous, "theta_step_old_run_000004"})

    _publish(
        control,
        adapter_name="theta_step_run_000001",
        previous_adapter=previous,
    )

    assert control.adapters == {"theta_step_run_000001"}
    load_index = control.calls.index(("post", "load_lora_adapter"))
    validation_index = control.calls.index(("post", "completions"))
    assert load_index < validation_index < len(control.calls) - 1
    assert control.calls[-1] == ("post", "unload_lora_adapter")


def test_retry_of_same_version_validates_without_reloading() -> None:
    target = "theta_step_run_000001"
    control = _SGLangControl({target})

    _publish(control, adapter_name=target)

    assert control.adapters == {target}
    assert ("post", "completions") in control.calls
    assert ("post", "unload_lora_adapter") not in control.calls
    assert ("post", "load_lora_adapter") not in control.calls


def test_failed_validation_removes_new_version_and_preserves_previous() -> None:
    previous = "theta_step_run_000000"
    control = _SGLangControl({previous})
    control.fail_validation = True

    try:
        _publish(
            control,
            adapter_name="theta_step_run_000001",
            previous_adapter=previous,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("validation failure must abort publication")

    assert control.adapters == {previous}
