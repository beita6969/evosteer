"""New fresh-F0 development read, reusing an accepted frozen-base author response.

This is a new declared read condition, not recovery/relabeling of adapter-free
sampling. No model weights are loaded here; the owner must first publish F0.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sqlite3
from pathlib import Path

from scripts.qualify_catalog_authoring import (
    CountedTransport,
    clock,
    preserve,
    read,
    read_episode,
    recover_mutation,
)
from skillev.policy import QwenTokenizerAdapter
from skillev.policy.checkpoint import read_policy_checkpoint_state
from skillev.rollout import PolicySnapshot
from skillev.rollout.external_sglang import (
    ExternalSGLangRolloutConfig,
    ExternalSGLangRolloutGenerator,
)
from skillev.runtime.request_journal import DurableRequestJournal
from skillev.runtime.sglang_gateway import SGLangGateway, SGLangGatewayConfig


def recover_author_copy(author_root, output, tokenizer):
    # Keep the original run basename because it is part of the author request ID.
    # SQLite backup includes committed WAL data without touching original evidence.
    recovery = output / "author-recovery" / author_root.name
    recovery.mkdir(parents=True, mode=0o700, exist_ok=False)
    for name in (
        "plan.json",
        "initial-library.json",
        "author-request.json",
        "accepted-author-result.json",
    ):
        shutil.copyfile(author_root / name, recovery / name)
    with sqlite3.connect(
        (author_root / "author-requests.sqlite3").as_uri() + "?mode=ro", uri=True
    ) as source:
        with sqlite3.connect(recovery / "author-requests.sqlite3") as destination:
            source.backup(destination)
    library, skill = recover_mutation(
        recovery, read(recovery / "plan.json"), tokenizer, resumed=True
    )
    if library.state.to_value() != read(author_root / "mutated-library.json"):
        raise ValueError("recovered author mutation differs from original accepted library")
    if read(recovery / "resume-author-summary.json")["physical_author_calls"] != 0:
        raise ValueError("new author generation is forbidden")
    preserve(
        output / "author-origin.json",
        {
            "author_root": str(author_root),
            "recovery_copy": str(recovery),
            "request": read(recovery / "author-request.json"),
            "new_author_generations": 0,
            "original_author_policy": read(author_root / "plan.json")["policy"],
        },
    )
    for name in ("mutated-library.json", "mutation.json"):
        shutil.copyfile(recovery / name, output / name)
    return library, skill


def fresh_generator(output, plan, tokenizer, *, gateway=None):
    gateway = gateway or SGLangGateway(
        SGLangGatewayConfig(
            endpoint_base=plan["endpoint"],
            base_model=plan["base_model"],
            supervisor_adapter=plan["published_forward_adapter"],
            control_retries=0,
        )
    )
    policy = PolicySnapshot.from_value(plan["policy"])
    if policy.forward_adapter_version == "adapter-free":
        raise ValueError("fresh read cannot fall back to adapter-free sampling")
    generation = gateway.bind_existing_supervisor_adapter(
        adapter_revision=policy.forward_adapter_version
    )
    if generation.adapter_name != plan["published_forward_adapter"]:
        raise ValueError("gateway bound another published adapter")
    counted = CountedTransport(output, "actor")
    generator = ExternalSGLangRolloutGenerator(
        config=ExternalSGLangRolloutConfig(plan["endpoint"]),
        tokenizer=tokenizer,
        gateway=gateway,
        snapshot_provider=lambda: policy,
        transport=counted,
        request_journal=DurableRequestJournal(output / "actor-requests.sqlite3"),
    )
    return generator, counted


def collect(author_root, output, preparation, adapter_name):
    from skillev_private.experiments.bayesian_training_setup import _read_preparation

    if output == author_root or author_root in output.parents or output in author_root.parents:
        raise ValueError("fresh read requires a separate development directory")
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    preserve(output / "read-started.json", {"driver_pid": os.getpid(), "started_at": clock()})
    try:
        config, binding = _read_preparation(preparation)
        state = read_policy_checkpoint_state(Path(binding.directory))
        if state.optimizer_step != 0 or not adapter_name.strip():
            raise ValueError("fresh preparation and explicit published adapter are required")
        original = read(author_root / "plan.json")
        expected = dict(original["backbone"])
        expected["device"] = config.device
        if config.to_value() != expected:
            raise ValueError("fresh scorer and original author backbone declarations differ")
        tokenizer = QwenTokenizerAdapter.from_config(config)
        library, skill = recover_author_copy(author_root, output, tokenizer)
        policy = PolicySnapshot.create(
            backbone_id=state.backbone_id,
            forward_adapter_version=state.forward_version,
            tokenizer_id=config.tokenizer_id,
            backend_id="sglang-native-exact-token",
            initial_trainable_state_hash=state.trainable_state.content_hash,
        )
        plan = {
            **original,
            "policy": policy.to_value(),
            "backbone": config.to_value(),
            "published_forward_adapter": adapter_name,
            "no_trainables_loaded": True,
            "read_condition": "fresh-F0-controlled-development@1",
            "read_condition_scope": "new F0 sampling; not adapter-free recovery",
            "preparation": str(preparation),
        }
        preserve(output / "plan.json", plan)
        generator, counted = fresh_generator(output, plan, tokenizer)
        try:
            result = asyncio.run(read_episode(output, plan, library, skill, generator))
        finally:
            generator.close()
            preserve(
                output / "actor-physical-summary.json", {"physical_actor_calls": counted.calls}
            )
        return result
    except BaseException as error:
        preserve(
            output / "failure.json", {"error_type": type(error).__name__, "message": str(error)}
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--published-forward-adapter", required=True)
    args = parser.parse_args(argv)
    collect(
        args.author_root.resolve(),
        args.output.resolve(),
        args.preparation.resolve(),
        args.published_forward_adapter,
    )


if __name__ == "__main__":
    main()
