"""Frozen F-reference logits on the same base model, never another task agent.

A reference is captured explicitly at an accepted initial-policy boundary and
must be saved/reloaded separately from the evolving checkpoint. Functional
parameter substitution does not overwrite live adapter parameters or versions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from safetensors.torch import load_file, save_file

from skillev.scoring.edge_plan import PreparedEdge

from .interface import AdapterRole

if TYPE_CHECKING:
    from .hf_backbone import QwenPolicyBackbone


class QwenFrozenForwardReference:
    def __init__(
        self,
        backbone: QwenPolicyBackbone,
        *,
        reference_id: str,
        forward_state: dict[str, torch.Tensor],
        maximum_context_tokens: int,
    ) -> None:
        if (
            not reference_id.strip()
            or type(maximum_context_tokens) is not int
            or maximum_context_tokens < 2
        ):
            raise ValueError("reference requires an explicit identity and capacity")
        self.backbone = backbone
        self.reference_id = reference_id
        self.maximum_context_tokens = maximum_context_tokens
        forward_ids = {id(parameter) for parameter in backbone.parameter_groups().forward}
        named = {
            name: parameter
            for name, parameter in backbone._model.named_parameters()
            if id(parameter) in forward_ids
        }
        if set(named) != set(forward_state) or any(
            named[name].shape != value.shape or named[name].dtype != value.dtype
            for name, value in forward_state.items()
        ):
            raise ValueError(
                "frozen reference state differs from the current forward adapter layout"
            )
        self._state = {name: value.detach().cpu().clone() for name, value in forward_state.items()}

    @classmethod
    def capture(
        cls, backbone: QwenPolicyBackbone, *, reference_id: str, maximum_context_tokens: int
    ) -> QwenFrozenForwardReference:
        backbone._require_idle_scoring()
        forward_ids = {id(parameter) for parameter in backbone.parameter_groups().forward}
        state: dict[str, torch.Tensor] = {
            name: parameter
            for name, parameter in backbone._model.named_parameters()
            if id(parameter) in forward_ids
        }
        return cls(
            backbone,
            reference_id=reference_id,
            forward_state=state,
            maximum_context_tokens=maximum_context_tokens,
        )

    def logits(self, edge: PreparedEdge) -> tuple[torch.Tensor, torch.Tensor]:
        if edge.role is not AdapterRole.FORWARD_POLICY:
            raise ValueError("F reference must not constrain the backward policy")
        backbone = self.backbone
        backbone._require_idle_scoring()
        if len(edge.prefix_ids) + len(edge.action_ids) > self.maximum_context_tokens:
            raise ValueError("reference context exceeds its declared capacity; no truncation")
        inputs = backbone._ids_tensor(edge.prefix_ids + edge.action_ids, field="reference context")
        kwargs = {
            "input_ids": inputs,
            "attention_mask": torch.ones_like(inputs),
            "use_cache": False,
            "return_dict": True,
        }
        start = len(edge.prefix_ids) - 1
        stop = start + len(edge.action_ids)
        with backbone.scoring_session(AdapterRole.FORWARD_POLICY):
            state = {name: value.to(backbone._input_device) for name, value in self._state.items()}
            with torch.no_grad():
                output = torch.func.functional_call(
                    backbone._model, state, (), kwargs, strict=False
                )
                reference = output.logits[0, start:stop].detach().float().cpu()
                del output
            current = backbone._model(**kwargs).logits[0, start:stop].float()
        return cast(torch.Tensor, current), reference

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        save_file(self._state, str(directory / "forward_reference.safetensors"))
        (directory / "reference.json").write_text(
            json.dumps(
                {
                    "format": "frozen-forward-reference@1",
                    "reference_id": self.reference_id,
                    "maximum_context_tokens": self.maximum_context_tokens,
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls, backbone: QwenPolicyBackbone, directory: Path, *, expected_reference_id: str
    ) -> QwenFrozenForwardReference:
        meta = json.loads((directory / "reference.json").read_text(encoding="utf-8"))
        if (
            meta.get("format") != "frozen-forward-reference@1"
            or meta.get("reference_id") != expected_reference_id
        ):
            raise ValueError(
                "reference artifact identity differs from the declared optimizer condition"
            )
        return cls(
            backbone,
            reference_id=expected_reference_id,
            forward_state=load_file(str(directory / "forward_reference.safetensors")),
            maximum_context_tokens=meta["maximum_context_tokens"],
        )
