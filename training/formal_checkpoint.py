"""Atomic full-state checkpoints for the corrected SkillFlow baseline."""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch

CHECKPOINT_FORMAT = "skillflow-formal-checkpoint@2"
logger = logging.getLogger(__name__)


class FormalCheckpointStore:
    def __init__(
        self,
        run_directory: Path,
        *,
        keep_recent: int = 3,
        initial_checkpoint_estimate_bytes: int = 1024**3,
        minimum_free_headroom_bytes: int = 5 * 1024**3,
    ) -> None:
        if keep_recent < 1:
            raise ValueError("keep_recent must be positive")
        if initial_checkpoint_estimate_bytes < 0 or minimum_free_headroom_bytes < 0:
            raise ValueError("checkpoint storage reserves must be non-negative")
        self.run_directory = run_directory
        self.root = run_directory / "checkpoints"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.keep_recent = keep_recent
        self.initial_checkpoint_estimate_bytes = initial_checkpoint_estimate_bytes
        self.minimum_free_headroom_bytes = minimum_free_headroom_bytes

    def save(self, trainer: Any, *, committed_step: int, final: bool = False) -> Path:
        if committed_step < 0:
            raise ValueError("committed_step must be non-negative")
        tag = "final" if final else f"step_{committed_step:06d}"
        target = self.root / f"checkpoint_{tag}"
        if target.exists():
            raise FileExistsError("checkpoint target already exists")
        required = self._estimated_peak_bytes()
        if shutil.disk_usage(self.root).free < required:
            raise OSError("checkpoint filesystem is below its dynamic safety floor")
        staging = self.root / f"checkpoint_tmp_{uuid.uuid4().hex}"
        staging.mkdir(mode=0o700)
        try:
            state = self._capture_state(trainer, committed_step=committed_step)
            torch.save(state, staging / "training_state.pt")
            skills = staging / "skills"
            skills.mkdir(mode=0o700)
            trainer.workspace.save_all()
            for source in sorted(trainer.workspace.skills_dir.glob("*.md")):
                shutil.copy2(source, skills / source.name)
            (staging / "COMPLETE").write_text("complete\n", encoding="utf-8")
            manifest = {
                "committed_step": committed_step,
                "format": CHECKPOINT_FORMAT,
                "method_identity": self._method_identity(trainer),
            }
            (staging / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            self.verify(staging, trainer=trainer)
            os.replace(staging, target)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise
        self._apply_retention()
        return target

    def restore(
        self,
        trainer: Any,
        checkpoint: Path,
        *,
        allow_git_commit_mismatch: bool = False,
    ) -> int:
        if allow_git_commit_mismatch:
            logger.warning("obsolete git-identity override ignored by checkpoint format 2")
        manifest = self.verify(checkpoint, trainer=trainer)
        state = torch.load(
            checkpoint / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        if state.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("checkpoint state format differs")
        parameters = dict(trainer.shared_model.named_parameters())
        if set(state["lora_parameters"]) != {name for name in parameters if "lora_" in name}:
            raise ValueError("checkpoint LoRA parameter set differs")
        for name, tensor in state["lora_parameters"].items():
            parameters[name].data.copy_(tensor.to(parameters[name].device))
        trainer.partition_fn.load_state_dict(state["partition_function"])
        trainer._supervisor_optimizer.load_state_dict(state["optimizers"]["theta"])
        trainer._phi_optimizer.load_state_dict(state["optimizers"]["phi"])
        trainer._partition_optimizer.load_state_dict(state["optimizers"]["partition"])
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch_cpu"])
        torch.cuda.set_rng_state(state["rng"]["coordinator_cuda"], trainer.device)
        client = trainer.distributed_gradient_client
        client.restore_worker_rng(state["rng"]["worker_cuda"])
        client.current_micro_batch = int(state["micro_batch"])
        for name, value in state["trainer_state"].items():
            setattr(trainer, name, value)
        self._restore_skills(trainer, checkpoint / "skills")
        trainer._current_step = int(manifest["committed_step"])
        return trainer._current_step

    def verify(
        self,
        checkpoint: Path,
        *,
        trainer: Any | None = None,
    ) -> dict[str, Any]:
        if (
            not checkpoint.is_dir()
            or not (checkpoint / "COMPLETE").is_file()
            or not (checkpoint / "training_state.pt").is_file()
            or not (checkpoint / "skills").is_dir()
        ):
            raise ValueError("checkpoint is incomplete")
        manifest_path = checkpoint / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("checkpoint manifest is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("checkpoint manifest format differs")
        committed = manifest.get("committed_step")
        identity = manifest.get("method_identity")
        if type(committed) is not int or committed < 0 or not isinstance(identity, dict):
            raise ValueError("checkpoint manifest is malformed")
        if trainer is not None and identity != self._method_identity(trainer):
            raise ValueError("checkpoint method identity differs")
        return manifest

    def _capture_state(self, trainer: Any, *, committed_step: int) -> dict[str, Any]:
        client = trainer.distributed_gradient_client
        tracked_names = (
            "_experience_buffer",
            "_observation_buffer",
            "_per_type_acc_history",
            "_per_type_balance_history",
            "_per_type_evolution_helped",
            "_per_type_last_evolution",
            "_phase_trajectories",
            "_plateau_detector",
            "_skill_negative_counter",
            "_skills_just_updated",
            "accuracy_tracker",
        )
        return {
            "adapter_version": getattr(trainer, "_current_adapter_version", "theta_uninitialized"),
            "committed_step": committed_step,
            "format": CHECKPOINT_FORMAT,
            "lora_parameters": {
                name: parameter.detach().cpu()
                for name, parameter in trainer.shared_model.named_parameters()
                if "lora_" in name
            },
            "micro_batch": client.current_micro_batch,
            "optimizers": {
                "partition": trainer._partition_optimizer.state_dict(),
                "phi": trainer._phi_optimizer.state_dict(),
                "theta": trainer._supervisor_optimizer.state_dict(),
            },
            "partition_function": trainer.partition_fn.state_dict(),
            "rng": {
                "coordinator_cuda": torch.cuda.get_rng_state(trainer.device),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
                "torch_cpu": torch.get_rng_state(),
                "worker_cuda": client.capture_worker_rng(),
            },
            "sampler": {"next_step": committed_step},
            "trainer_state": {name: getattr(trainer, name) for name in tracked_names},
            "world_size": torch.distributed.get_world_size(),
        }

    def _restore_skills(self, trainer: Any, skills: Path) -> None:
        from src.skills.format import SkillEntry

        trainer.workspace._skills.clear()
        for path in sorted(skills.glob("*.md")):
            skill = SkillEntry.from_file(path)
            trainer.workspace._skills[skill.meta.skill_id] = skill
        trainer.workspace.save_all()

    @staticmethod
    def _method_identity(trainer: Any) -> dict[str, Any]:
        config = trainer.config
        return {
            "base_model": config["base_model"],
            "batch_size": config["batch_size"],
            "max_steps": config["max_steps"],
            "parity_contract": config["parity_contract"],
            "upstream_revision": config["upstream_revision"],
        }

    def _estimated_peak_bytes(self) -> int:
        complete = sorted(self.root.glob("checkpoint_*"))
        sizes = [
            sum(path.stat().st_size for path in checkpoint.rglob("*") if path.is_file())
            for checkpoint in complete
        ]
        checkpoint_size = max(sizes, default=self.initial_checkpoint_estimate_bytes)
        return 2 * checkpoint_size + self.minimum_free_headroom_bytes

    def _apply_retention(self) -> None:
        numbered = sorted(self.root.glob("checkpoint_step_*"))
        protected = {path for path in numbered if (path / "BEST").exists()}
        removable = [path for path in numbered[: -self.keep_recent] if path not in protected]
        for path in removable:
            shutil.rmtree(path)
