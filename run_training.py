"""Single fail-closed entrypoint for corrected SkillFlow baseline training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import yaml

from scripts.validate_data import (
    _duplicate_policy,
    _load_records,
    _validated_path,
    reject_cross_split_overlap,
    validate_records,
)
from skillev.runtime.formal_preflight import (
    FormalBaselineConfig,
    build_run_manifest,
    collect_gpu_observations,
    resolve_gpu_assignment,
    resolve_training_cuda_visibility,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline/paper_v1_250step.yaml")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", metavar="CHECKPOINT")
    parser.add_argument("--allow-resume-git-mismatch", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--prepared-run-dir", help=argparse.SUPPRESS)
    return parser.parse_args()


def _git(*arguments: str) -> str:
    return subprocess.run(  # noqa: S603 - fixed Git subcommands from trusted callers
        ["/usr/bin/git", *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout.strip()


def _safe_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("run ID contains unsupported characters")
    return value


def _validate_official_data(config: FormalBaselineConfig) -> dict[str, object]:
    data = config.data
    train = _load_records(_validated_path(data, split="train"))
    validation = _load_records(_validated_path(data, split="validation"))
    train_report = validate_records(
        train,
        split="train",
        expected_count=cast(int, data["expected_train_rows"]),
        allowed_duplicate_extra_rows=_duplicate_policy(data, "train"),
    )
    validation_report = validate_records(
        validation,
        split="validation",
        expected_count=cast(int, data["expected_validation_rows"]),
        allowed_duplicate_extra_rows=_duplicate_policy(data, "validation"),
    )
    reject_cross_split_overlap(train, validation)
    return {"train": train_report.to_value(), "validation": validation_report.to_value()}


def _prepare_run(
    config: FormalBaselineConfig,
    *,
    requested_run_id: str | None,
    resume: str | None,
) -> Path:
    if _git("status", "--porcelain"):
        raise RuntimeError("formal training requires a clean Git worktree")
    commit = _git("rev-parse", "HEAD")
    manifest = build_run_manifest(
        config,
        git_commit=commit,
        observations=collect_gpu_observations(),
    )
    manifest["validated_data"] = cast(object, _validate_official_data(config))
    config_id = str(manifest["config_blake2b"])[:8]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = _safe_run_id(
        requested_run_id
        or f"skillflow__{config_id}__seed{config.training['seed']}__{timestamp}__{commit[:8]}"
    )
    root_key = cast(str, config.runtime["output_root_env"])
    root = Path(os.environ[root_key]).expanduser().resolve()
    run_directory = root / run_id
    if run_directory.exists():
        raise FileExistsError("formal run output already exists")
    run_directory.mkdir(parents=True, mode=0o700)
    (run_directory / "checkpoints").mkdir(mode=0o700)
    (run_directory / "recovery").mkdir(mode=0o700)
    manifest["run_id"] = run_id
    if resume is not None:
        checkpoint = Path(resume).expanduser().resolve()
        parent_manifest_path = checkpoint / "manifest.json"
        if not parent_manifest_path.is_file():
            raise RuntimeError("resume checkpoint manifest is unavailable")
        match = re.fullmatch(r"checkpoint_step_(\d+)", checkpoint.name)
        if match is None:
            raise RuntimeError("resume checkpoint name has no committed step")
        manifest["continuation"] = {
            "parent_run_id": checkpoint.parent.parent.name,
            "parent_committed_step": int(match.group(1)),
            "parent_checkpoint_manifest_blake2b": hashlib.blake2b(
                parent_manifest_path.read_bytes(), digest_size=32
            ).hexdigest(),
            "reason": "swe_harness_and_context_budget_fix",
        }
    (run_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_directory / "normalized_config.json").write_text(
        json.dumps(config.normalized_value(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_directory


def _upstream_training_config(
    config: FormalBaselineConfig,
    *,
    run_directory: Path,
    max_steps: int | None,
) -> dict[str, object]:
    upstream = yaml.safe_load(Path("configs/skillflow.yaml").read_text(encoding="utf-8"))
    if not isinstance(upstream, dict) or not isinstance(upstream.get("training"), dict):
        raise RuntimeError("fixed upstream training configuration is malformed")
    result = dict(upstream["training"])
    environment = config.resolved_environment()
    training = config.training
    service = config.supervisor
    result.update(
        {
            "backward_lora_rank": training["phi_lora_rank"],
            "base_model": environment["model_path"],
            "batch_size": training["effective_batch_size"],
            "beta": training["beta"],
            "device": f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}",
            "dump_trajectories": False,
            "epsilon_min": training["epsilon_min"],
            "executor_api_base": environment["supervisor_api_base"].rstrip("/") + "/v1",
            "executor_model": service["served_model"],
            # Executor calls may reserve 8K output tokens.  The real BR-2
            # workload exhausted the 60-second supervisor timeout, while the
            # 180-second bound still keeps three attempts inside the existing
            # 600-second episode boundary.
            "executor_request_timeout_seconds": max(180, int(service["request_timeout_seconds"])),
            "extra_device": None,
            "formal_checkpoint": True,
            "formal_runtime": True,
            "kl_coeff": training["kl_coefficient"],
            "logprob_context_length": training["logprob_context_length"],
            "learning_rate": training["learning_rate"],
            "lora_alpha": training["theta_lora_alpha"],
            "lora_rank": training["theta_lora_rank"],
            "lora_target_modules": training["theta_target_modules"],
            "max_episode_steps": training["max_episode_steps"],
            "max_grad_norm": training["max_grad_norm"],
            "max_steps": max_steps or training["max_training_steps"],
            "n_trajectories_per_question": training["trajectories_per_question"],
            "output_dir": str(run_directory),
            "save_every": training["checkpoint_every_steps"],
            "seed_offset": training["seed"],
            "supervisor_api_base": environment["supervisor_api_base"].rstrip("/") + "/v1",
            "supervisor_adapter_prefix": service["supervisor_adapter_prefix"],
            "supervisor_context_length": service["expected_context_length"],
            # The pinned Qwen/SGLang server template used 285 more prompt
            # tokens than the local tokenizer for the formal tool schema.
            "supervisor_context_safety_tokens": 512,
            "supervisor_max_retries": service["max_retries"],
            "supervisor_mode": "external",
            "supervisor_model": "supervisor_theta",
            "supervisor_request_timeout_seconds": service["request_timeout_seconds"],
            "sync_lora_every": 1,
            "tracking_mode": "disabled",
            "tokenizer_path": environment["tokenizer_path"],
        }
    )
    return result


def _run_training(
    config: FormalBaselineConfig,
    *,
    run_directory: Path,
    max_steps: int | None,
    resume: str | None,
    allow_resume_git_mismatch: bool,
) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        raise RuntimeError("formal training must run under the process-based gradient launcher")
    import torch
    import torch.distributed as dist

    from training.distributed_gradient import (
        DistributedGradientClient,
        StandbyGradientRequiredError,
        coordinator_barrier,
        initialize_process_group,
        worker_loop,
    )
    from training.gflownet_trainer import GFlowNetTrainer

    rank, _, local_rank = initialize_process_group()
    torch.manual_seed(cast(int, config.training["seed"]))
    translated = _upstream_training_config(config, run_directory=run_directory, max_steps=max_steps)
    if rank > 0:
        try:
            worker_loop(translated)
        finally:
            dist.destroy_process_group()
        return
    data = config.data
    trainer = GFlowNetTrainer(config=translated)
    client = None
    try:
        trainer.setup(
            train_data=_load_records(_validated_path(data, split="train")),
            val_data=_load_records(_validated_path(data, split="validation")),
        )
        coordinator_barrier()
        runtime = config.runtime
        client = DistributedGradientClient.create(
            trainer.shared_model,
            initial_micro_batch=cast(int, runtime["initial_micro_batch"]),
            minimum_micro_batch=cast(int, runtime["minimum_micro_batch"]),
        )
        trainer.distributed_gradient_client = client
        if resume is not None:
            trainer.resume(
                resume,
                allow_git_commit_mismatch=allow_resume_git_mismatch,
            )
        trainer.train()
    except StandbyGradientRequiredError:
        checkpoint = run_directory / "checkpoints" / f"checkpoint_step_{trainer._current_step:06d}"
        if not checkpoint.is_dir():
            raise RuntimeError("no committed checkpoint is available for GPU4 expansion") from None
        (run_directory / "EXPAND_GRADIENT_WORKERS").write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "last_committed_step": trainer._current_step,
                    "reason": "primary_oom",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        if client is not None:
            client.close()
        dist.destroy_process_group()


def _torchrun_command(
    arguments: argparse.Namespace,
    *,
    run_directory: Path,
    processes: int,
    resume_override: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={processes}",
        "--master_port=29635",
        "run_training.py",
        "--config",
        arguments.config,
        "--prepared-run-dir",
        str(run_directory),
    ]
    resume = resume_override or arguments.resume
    command.extend(["--resume", resume] if resume else ["--fresh"])
    if arguments.max_steps is not None:
        command.extend(["--max-steps", str(arguments.max_steps)])
    if arguments.allow_resume_git_mismatch:
        command.append("--allow-resume-git-mismatch")
    return command


def _training_visible_devices(config: FormalBaselineConfig, *, expanded: bool) -> str:
    return resolve_training_cuda_visibility(config, expanded=expanded)


def _launch_process_group(arguments: argparse.Namespace, run_directory: Path) -> None:
    config = FormalBaselineConfig.read(arguments.config)
    assignment = resolve_gpu_assignment(config)
    environment = dict(os.environ)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = _training_visible_devices(config, expanded=False)
    environment["SKILLEV_FORMAL_RUNTIME"] = "1"
    environment["WEBSHOP_USE_SMALL"] = "1" if bool(config.resources["webshop_use_small"]) else "0"
    environment["WEBSHOP_GOAL_SPLIT"] = cast(str, config.resources["webshop_goal_split"])
    environment["SWE_BENCH_DOCKER_NAMESPACE"] = cast(str, config.resources["swe_docker_namespace"])
    completed_returncode = _run_torchrun(
        _torchrun_command(arguments, run_directory=run_directory, processes=2),
        environment=environment,
        run_directory=run_directory,
    )
    expansion = run_directory / "EXPAND_GRADIENT_WORKERS"
    if completed_returncode == 0:
        return
    if not expansion.is_file():
        raise RuntimeError("steady distributed training failed")
    if assignment.gradient_standby is None:
        raise RuntimeError("primary-gradient OOM requires a standby GPU, but standby is disabled")
    observations = {item.physical_index: item for item in collect_gpu_observations()}
    standby = observations[assignment.gradient_standby]
    if standby.compute_pids or standby.memory_free_mib < 70_000:
        raise RuntimeError("standby gradient GPU is unavailable after primary-gradient OOM")
    expansion_state = json.loads(expansion.read_text(encoding="utf-8"))
    resume_checkpoint = expansion_state.get("checkpoint")
    if not isinstance(resume_checkpoint, str):
        raise RuntimeError("GPU4 expansion marker has no committed checkpoint")
    expansion.unlink()
    environment["CUDA_VISIBLE_DEVICES"] = _training_visible_devices(config, expanded=True)
    expanded_returncode = _run_torchrun(
        _torchrun_command(
            arguments,
            run_directory=run_directory,
            processes=3,
            resume_override=resume_checkpoint,
        ),
        environment=environment,
        run_directory=run_directory,
    )
    if expanded_returncode != 0:
        raise RuntimeError("expanded distributed training failed")


def _run_torchrun(command: list[str], *, environment: dict[str, str], run_directory: Path) -> int:
    """Run torchrun in its own terminal group and convert host signals to safe-stop files."""

    stop_request = run_directory / "STOP_REQUESTED"
    previous: dict[signal.Signals, object] = {}

    def request_stop(signum, frame) -> None:
        del frame
        stop_request.write_text(
            json.dumps({"signal": int(signum), "status": "requested"}) + "\n",
            encoding="utf-8",
        )
        stop_request.chmod(0o600)

    for name in (signal.SIGINT, signal.SIGTERM):
        previous[name] = signal.getsignal(name)
        signal.signal(name, request_stop)
    try:
        process = subprocess.Popen(  # noqa: S603 - exact argv, never a shell command
            command,
            env=environment,
            start_new_session=True,
        )
        return process.wait()
    finally:
        for name, handler in previous.items():
            signal.signal(name, handler)


def main() -> None:
    arguments = _arguments()
    if arguments.max_steps is not None and arguments.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if arguments.allow_resume_git_mismatch and arguments.resume is None:
        raise ValueError("--allow-resume-git-mismatch requires --resume")
    config = FormalBaselineConfig.read(arguments.config)
    if arguments.prepared_run_dir:
        run_directory = Path(arguments.prepared_run_dir)
    else:
        run_directory = _prepare_run(
            config,
            requested_run_id=arguments.run_id,
            resume=arguments.resume,
        )
    if arguments.preflight_only:
        print(json.dumps({"run_id": run_directory.name, "status": "preflight_passed"}))
        return
    if "RANK" not in os.environ:
        _launch_process_group(arguments, run_directory)
        return
    _run_training(
        config,
        run_directory=run_directory,
        max_steps=arguments.max_steps,
        resume=arguments.resume,
        allow_resume_git_mismatch=arguments.allow_resume_git_mismatch,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            json.dumps({"error_class": type(error).__name__, "status": "failed"}, sort_keys=True),
            file=sys.stderr,
        )
        raise
