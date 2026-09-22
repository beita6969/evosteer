"""Private supervised development initialization; no sampling or native grading.

Commands: run (new exclusive root), resume (saved corpus/Adam), export (new TTB
preparation). Attributable annotations must already exist; this CLI never creates them.
The output is not original Step-0, IID acceptance, or a formal training result.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from skillev.contracts import normalize_json
from skillev.policy import (
    PolicyBackbone,
    QwenMultimodalBackboneConfig,
    QwenTokenizerAdapter,
    build_qwen_policy_backbone,
)
from skillev.policy.checkpoint import read_policy_checkpoint_state
from skillev.rollout import RolloutArtifact
from skillev.runtime import SkillLibraryState
from skillev.scoring.edge_plan import PreparedEdgePlan
from skillev.training.inflight import durable_json
from skillev.training.skill_use_warmup import (
    SkillUseWarmup,
    WarmupConfig,
    WarmupCorpus,
    WarmupDemonstration,
)

from .bayesian_training_setup import _PREPARATION_FORMAT, _read_preparation
from .warmup_initialization import (
    _save_tensors,
    export_initialization,
    load_named,
    require_parameters,
)

CORPUS_FORMAT = "skill-use-warmup-corpus@1"
STRUCTURED_CORPUS_FORMAT = "skill-use-warmup-corpus@2"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("warmup input must be a JSON object")
    return value


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_corpus(path: Path, tokenizer: Any) -> tuple[WarmupCorpus, dict[str, Any], list[Path]]:
    """Admit original artifacts/plans and independently attributed behavior review records."""
    manifest = _read(path)
    if manifest.get("format") not in (CORPUS_FORMAT, STRUCTURED_CORPUS_FORMAT):
        raise ValueError("explicit warmup corpus format is required")
    candidate = manifest["candidate"]
    library = SkillLibraryState.from_value(candidate["library"])
    if not isinstance(candidate.get("sampling"), dict) or not candidate["sampling"]:
        raise ValueError("warmup requires its frozen candidate sampling/phase condition")
    if (
        not isinstance(candidate.get("collection_condition"), dict)
        or not candidate["collection_condition"]
    ):
        raise ValueError(
            "warmup requires explicit original collection condition, including guided practice"
        )
    examples, originals, annotations = [], [path], []
    for row in manifest["demonstrations"]:
        artifact_file = _path(path.parent, row["artifact"])
        plan_file = _path(path.parent, row["prepared_plan"])
        annotation_file = _path(path.parent, row["annotation"])
        raw = _read(artifact_file)
        artifact = RolloutArtifact.from_value(raw.get("artifact", raw), tokenizer=tokenizer)
        plan_value, annotation = _read(plan_file), _read(annotation_file)
        if (
            plan_value["trajectory_id"] != artifact.record.trajectory_id
            or plan_value["tokenizer_id"] != artifact.record.tokenizer_id
            or plan_value["initial_text"] != artifact.initial_context.text
        ):
            raise ValueError("prepared token plan belongs to another original artifact")
        plan = PreparedEdgePlan.from_wire_edges(
            plan_value["edges"],
            record=artifact.record,
            initial_text=artifact.initial_context.text,
            tokenizer_id=artifact.record.tokenizer_id,
        )
        if (
            annotation.get("format")
            not in ("skill-application-annotation@1", "skill-application-annotation@2")
            or annotation.get("review_kind") not in {"human", "model-assisted-public-behavior"}
            or annotation.get("approved_for_warmup") is not True
            or annotation.get("trajectory_id") != artifact.record.trajectory_id
            or any(
                not isinstance(annotation.get(key), str) or not annotation[key].strip()
                for key in ("reviewer", "reviewed_at", "rationale")
            )
        ):
            raise ValueError("missing attributable application review; no automatic approval")
        if artifact.manifest.library_version != library.current_version:
            raise ValueError("demonstration was collected under another candidate library")
        examples.append(
            WarmupDemonstration(
                artifact,
                plan,
                tuple(row["source"]),
                str(annotation_file),
                annotation["decision"],
                annotation.get("skill_id"),
                annotation.get("read_step"),
                annotation.get("decision_step"),
                annotation if annotation["format"] == "skill-application-annotation@2" else None,
            )
        )
        originals.extend((artifact_file, plan_file, annotation_file))
        annotations.append({"source": str(annotation_file), "original": annotation})
    corpus = WarmupCorpus(
        tuple(examples),
        tuple(map(tuple, manifest["development_sources"])),
        tuple(map(tuple, manifest["excluded_sources"])),
        manifest["isolation_ref"],
        tuple(map(tuple, manifest["task_sources"])),
        tuple(map(tuple, manifest.get("aliases", []))),
    )
    corpus.require()
    return (
        corpus,
        {"candidate": candidate, "annotations": annotations, "original_manifest": manifest},
        originals,
    )


def saved_corpus(value: dict[str, Any], tokenizer: Any) -> WarmupCorpus:
    examples = []
    for row in value["demonstrations"]:
        artifact = RolloutArtifact.from_value(row["artifact"], tokenizer=tokenizer)
        plan = PreparedEdgePlan.from_wire_edges(
            row["edges"],
            record=artifact.record,
            initial_text=artifact.initial_context.text,
            tokenizer_id=artifact.record.tokenizer_id,
        )
        examples.append(
            WarmupDemonstration(
                artifact,
                plan,
                tuple(row["source"]),
                row["annotation_ref"],
                row["decision"],
                row["skill_id"],
                row["read_step"],
                row["decision_step"],
                row.get("annotation"),
            )
        )
    return WarmupCorpus(
        tuple(examples),
        tuple(map(tuple, value["development_sources"])),
        tuple(map(tuple, value["excluded_sources"])),
        value["isolation_ref"],
        tuple(map(tuple, value["task_sources"])),
        tuple(map(tuple, value["aliases"])),
    )


def _archive(paths: list[Path], root: Path) -> dict[str, str]:
    directory = root / "original-inputs"
    directory.mkdir(mode=0o700)
    mapping = {}
    for position, path in enumerate(dict.fromkeys(paths), 1):
        target = directory / f"input-{position:05d}{path.suffix}"
        with path.open("rb") as source, target.open("xb") as destination:
            shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
        target.chmod(0o600)
        mapping[str(path)] = str(target.relative_to(root))
    return mapping


def _build(config: QwenMultimodalBackboneConfig) -> PolicyBackbone:
    if config.device.startswith("cuda") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("warmup requires explicitly selected CUDA device visibility")
    return build_qwen_policy_backbone(config)


def run(
    *, preparation: Path, corpus_file: Path, output_root: Path, config: WarmupConfig, run_id: str
) -> Path:
    manifest = _read(corpus_file)
    if manifest.get("format") != STRUCTURED_CORPUS_FORMAT or not manifest.get("demonstrations"):
        raise ValueError("no explicitly reviewed demonstrations supplied")
    for row in manifest["demonstrations"]:
        note = _read(_path(corpus_file.parent, row["annotation"]))
        if (
            note.get("format") != "skill-application-annotation@2"
            or note.get("review_kind") not in {"human", "model-assisted-public-behavior"}
            or note.get("approved_for_warmup") is not True
        ):
            raise ValueError("missing attributable approved application review")
    original = _read(preparation)
    if original["format"] != _PREPARATION_FORMAT:
        raise ValueError(
            "warmup must start from a separate original preparation, not old training or warmup"
        )
    backbone_config, binding = _read_preparation(preparation)
    source_state = read_policy_checkpoint_state(Path(binding.directory))
    if source_state.optimizer_step != 0:
        raise ValueError("warmup cannot start from an old trained checkpoint")
    initial = load_named(preparation.parent / "initial_named_parameters.pt")
    if output_root.exists():
        raise FileExistsError(output_root)
    corpus, evidence, originals = read_corpus(
        corpus_file, QwenTokenizerAdapter.from_config(backbone_config)
    )
    corpus = corpus.for_training()
    backbone = _build(backbone_config)
    backbone.load_checkpoint(binding.directory)
    backbone.bind_initial_trainable_state(binding.trainable_state)
    require_parameters(backbone, initial, forward=True)
    f_ids = {id(p) for p in backbone.parameter_groups().forward}
    b_tensors = [
        p
        for name, p in backbone.named_trainable_parameters().items()
        if id(p) in f_ids and ".lora_B." in name
    ]
    if not b_tensors or any(torch.count_nonzero(p) for p in b_tensors):
        raise ValueError("original preparation must have base-equivalent zero forward LoRA B")
    trainer = SkillUseWarmup(backbone=backbone, corpus=corpus, config=config, run_id=run_id)
    output_root.mkdir(parents=True, mode=0o700, exist_ok=False)
    mapping = _archive([preparation, *originals], output_root)
    _save_tensors(output_root / "base_initial_named_parameters.pt", initial)
    durable_json(
        output_root / "run.json",
        {
            "format": "private-skill-use-warmup-run@1",
            "source_preparation": original,
            "config": asdict(config),
            "run_id": run_id,
            "corpus": trainer.corpus.to_value(),
            "supervision_report": trainer.corpus.supervision_report(),
            "application_review": evidence,
            "input_relocations": normalize_json(mapping),
            "formal_training": False,
            "iid_acceptance": None,
        },
    )
    trainer.save(output_root / "update-00000000")
    return finish(trainer, output_root)


def finish(trainer: SkillUseWarmup, root: Path) -> Path:
    last = root / f"update-{trainer.updates:08d}"
    while trainer.updates < trainer.config.maximum_updates:
        report = trainer.step()
        last = root / f"update-{trainer.updates:08d}"
        trainer.save(last)
        print(json.dumps(report, allow_nan=False), flush=True)
    return last


def resume(*, root: Path, checkpoint: Path) -> Path:
    # Reject older boundaries if later updates were already completed. Never
    # overwrite, merge histories, or call any generation API on recovery.
    state = _read(root / "run.json")
    saved = _read(checkpoint / "warmup.json")
    if checkpoint.parent.resolve() != root.resolve():
        raise ValueError("warmup checkpoint belongs to another private run")
    if any(
        (root / f"update-{step:08d}").exists()
        for step in range(saved["updates"] + 1, state["config"]["maximum_updates"] + 1)
    ):
        raise ValueError("a later warmup boundary/partial update exists; do not overwrite it")
    backbone = _build(
        QwenMultimodalBackboneConfig.from_value(state["source_preparation"]["backbone"])
    )
    trainer = SkillUseWarmup.restore(
        checkpoint,
        backbone=backbone,
        corpus=saved_corpus(state["corpus"], backbone.tokenizer),
        config=WarmupConfig(**state["config"]),
        run_id=state["run_id"],
    )
    return finish(trainer, root)


def export(*, root: Path, checkpoint: Path, output_root: Path) -> Path:
    state = _read(root / "run.json")
    if checkpoint.parent.resolve() != root.resolve():
        raise ValueError("warmup checkpoint belongs to another private run")
    saved = _read(checkpoint / "warmup.json")
    if saved["corpus"] != state["corpus"] or saved["run_id"] != state["run_id"]:
        raise ValueError("export checkpoint does not match approved corpus/run")
    backbone = _build(
        QwenMultimodalBackboneConfig.from_value(state["source_preparation"]["backbone"])
    )
    return export_initialization(
        backbone=backbone,
        warmup_checkpoint=checkpoint,
        source_preparation=state["source_preparation"],
        base_parameters=load_named(root / "base_initial_named_parameters.pt"),
        candidate={
            **state["application_review"]["candidate"],
            "application_review": state["application_review"]["annotations"],
        },
        output_root=output_root,
    )


def report_inputs(
    *,
    backbone_config: Path,
    output: Path,
    corpus_file: Path | None = None,
    checkpoint: Path | None = None,
) -> Path:
    """Tokenizer-only diagnostic; legacy material is not re-admitted or re-trained."""
    if (corpus_file is None) == (checkpoint is None):
        raise ValueError("select one existing corpus or saved checkpoint for reporting")
    tokenizer = QwenTokenizerAdapter.from_config(
        QwenMultimodalBackboneConfig.from_value(_read(backbone_config))
    )
    if corpus_file is not None:
        corpus, _, _ = read_corpus(corpus_file, tokenizer)
    else:
        assert checkpoint is not None
        corpus = saved_corpus(_read(checkpoint / "warmup.json")["corpus"], tokenizer)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(
            corpus.supervision_report(), stream, ensure_ascii=False, indent=2, allow_nan=False
        )
        stream.write("\n")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    new = commands.add_parser("run")
    for name in ("preparation", "corpus", "output-root"):
        new.add_argument("--" + name, required=True, type=Path)
    new.add_argument("--run-id", required=True)
    new.add_argument("--learning-rate", required=True, type=float)
    new.add_argument("--updates", required=True, type=int)
    new.add_argument("--weight-decay", type=float, default=0)
    for name in ("resume", "export"):
        command = commands.add_parser(name)
        command.add_argument("--run-root", required=True, type=Path)
        command.add_argument("--checkpoint", required=True, type=Path)
        if name == "export":
            command.add_argument("--output-root", required=True, type=Path)
    diagnostic = commands.add_parser("report")
    diagnostic.add_argument("--backbone-config", type=Path, required=True)
    diagnostic.add_argument("--output", type=Path, required=True)
    source = diagnostic.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus", type=Path)
    source.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run(
            preparation=args.preparation,
            corpus_file=args.corpus,
            output_root=args.output_root,
            config=WarmupConfig(args.learning_rate, args.updates, args.weight_decay),
            run_id=args.run_id,
        )
    elif args.command == "report":
        result = report_inputs(
            backbone_config=args.backbone_config,
            output=args.output,
            corpus_file=args.corpus,
            checkpoint=args.checkpoint,
        )
    elif args.command == "resume":
        result = resume(root=args.run_root, checkpoint=args.checkpoint)
    else:
        result = export(
            root=args.run_root, checkpoint=args.checkpoint, output_root=args.output_root
        )
    print(json.dumps({"output": str(result), "original_step_zero": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
