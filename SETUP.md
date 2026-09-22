# Corrected SkillFlow baseline setup

The fixed upstream source is retained under `training/`, `src/executor/`, and
`src/skills/`. The only supported formal entrypoint is `run_training.py`; do
not invoke the upstream trainer directly.

## Environments

The CPU development environment uses `uv`. The server has separate serving
and training environments frozen by:

- `requirements/serve.lock`
- `requirements/train.lock`
- `requirements/eval.lock`

The training environment must include the pinned
`causal-conv1d==1.6.2.post1` fast path.

## Required private environment

Set the model, independent tokenizer, official train/IID files, fixed model
revision, private output root, and external supervisor URL through the
environment keys named in `configs/baseline/paper_v1_250step.yaml`. Never put
their resolved paths or credentials in Git or public logs.

## External SGLang

SGLang is a separately managed service on physical GPU1. The trainer verifies
health, the OpenAI-compatible request path, 32K context, metrics, and LoRA
control routes before importing the CUDA training stack. It never starts,
stops, restarts, or signals that service.

## Training

Before launch, confirm GPU0 belongs only to its existing owner, GPU1 contains
the external SGLang service, and physical GPUs 2, 3, and 4 are idle. Normal
training exposes only GPUs 2 and 3:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 \
  python -u run_training.py \
  --config configs/baseline/paper_v1_250step.yaml \
  --fresh
```

The entrypoint starts one coordinator process on logical `cuda:0` (physical
GPU2) and one gradient worker on logical `cuda:1` (physical GPU3). GPU4 is not
visible and creates no CUDA context in the steady profile. Only a confirmed
primary-worker CUDA OOM at the minimum micro-batch causes a process-level
restart with `CUDA_VISIBLE_DEVICES=2,3,4`.

Use `--preflight-only` for a CUDA-free validation and `--resume <exact
checkpoint directory>` for an identity-checked recovery. `--fresh` and
`--resume` are mutually exclusive.
