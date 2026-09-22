# Direct-H800 runbook

## Invariants

Use only the configured direct SSH endpoint. Do not use Slurm, Delta, or Duo. At every launch,
inspect all eight physical GPUs. The inference card must host the verified external SGLang service;
choose any other two owner-authorized cards only when they are driver-healthy, process-idle, and
meet the memory reserve. Never stop, signal, alter, or reuse another user's process. Every CUDA
process must receive an explicit `CUDA_VISIBLE_DEVICES` value; the Protocol 10 torchrun launcher
binds exactly the two training cards by GPU UUID.

Roles:

1. **inference** — one Qwen3.5-9B SGLang service; base model serves executor and skill creator,
   live LoRA serves supervisor;
2. **coordinator** — training state, optimizer/step transaction, and non-worker gradient work;
3. **primary-gradient** — steady-state gradient worker;
Physical indices are assignments, not role names. Record the selected mapping in the private run
manifest. `cuda:0` inside a process is the logical device after remapping.

## Preflight

Confirm all of the following immediately before each method launch:

- clean frozen Git worktree and open Protocol 10 execution gates;
- all eight GPU identities and compute PIDs;
- SGLang health, frozen base-model identity, and empty expected adapter namespace;
- the 4,608-entry seed-0 training selection and all nine private session routes;
- output and temporary roots on the same filesystem with atomic rename and `fsync` support;
- free bytes and inodes above the measured 22-step-derived storage budget.

The formal storage floor is the estimated 288-step peak plus one rollback checkpoint, atomic
publication space, and 25% safety margin. `FormalStorageBinding` is part of every formal exact input;
the supervisor rejects the launch before creating CUDA processes if either storage or GPU admission
fails. If the declared role mapping is no longer valid, do not evict anyone or choose an undeclared
card; freeze a new result-blind launch package before observing any method result.

## Service and training

Before the first method, freeze exactly three formal exact inputs in this order:

```text
skillflow-baseline
bayesian-improve-full
bayesian-improve-no-calibration
```

Create the write-once private package from a clean worktree:

```bash
python -m skillev_private.experiments.protocol_v10_launch_package \
  --repository <clean-repository> \
  --run-root <new-private-run-root> \
  --exact-input <exact-skillflow-input> \
  --exact-input <full-input> \
  --exact-input <no-calibration-input> \
  --output <new-private-launch-package>
```

Launch one declared method at a time from that package (normally under remote `tmux` or `nohup`):

```bash
python -m skillev_private.experiments.protocol_v10_attempt_supervisor \
  --launch-package <private-launch-package> \
  --method <declared-method>
```

Direct formal launch from a standalone exact input is rejected. A CUDA OOM aborts the whole
uncommitted step and fails the attempt. Resume is an explicit later attempt from the last completed
checkpoint; the supervisor never retries, silently shrinks the scientific batch, or adds another
GPU.

## Monitoring and cleanup

Record only PIDs started by this project. At launch confirm PID, log movement, and GPU memory. For
long tasks report completed work, throughput, throughput change, and a rough ETA at least every 20
minutes. If throughput falls below the configured threshold, label the run “需要人工决策”.

On failure or completion, terminate only recorded project PIDs, then verify both the process table
and `nvidia-smi`. Never use a broad process-name kill.
