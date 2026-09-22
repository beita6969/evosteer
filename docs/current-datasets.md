# Current benchmark catalog — 2026-09-15

Owner authority: **six IID, six OOD, native thinking off by default**.
One Qwen3.5-9B owner per episode; no consultants. Explicit historical thinking
conditions and results retain their identities. No evaluation/training is
launched by downloading data or loading these configurations.

| Role | Dataset | Download / native source | Evaluation population |
|---|---|---|---|
| IID | HotpotQA | [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa) | distractor validation; 7,405 |
| IID | TriviaQA | [TriviaQA](https://huggingface.co/datasets/mandarjoshi/trivia_qa) | RC validation, with public context; 17,944 |
| IID | AIME2026 | [MathArena](https://huggingface.co/datasets/MathArena/aime_2026) | all 30; HF calls its only split `train`, but these 2026 exam records remain evaluation-only |
| IID | HealthBench | [OpenAI](https://huggingface.co/datasets/openai/healthbench) | full 5,000 release, not Hard or Consensus |
| IID | ALFWorld | [official release](https://github.com/alfworld/alfworld) | text environment, separate seen/unseen evaluation |
| IID | MBPP+ | [EvalPlus](https://github.com/evalplus/mbppplus_release/releases/tag/v0.2.0) | 378 tasks in the currently downloaded official v0.2.0 artifact; Base **and** Plus tests |
| OOD | MuSiQue-Ans | [official data](https://github.com/StonyBrookNLP/musique#data) | answerable dev; 2,417; all public candidate paragraphs |
| OOD | NQ-Open | [Google Research](https://github.com/google-research-datasets/natural-questions/tree/master/nq_open) | dev; 3,610; closed book |
| OOD | Math-Hard | [lighteval/MATH-Hard](https://huggingface.co/datasets/lighteval/MATH-Hard) | test; 1,324 Level-5 tasks, **not MATH-500** |
| OOD | GPQA Biology + Organic Chemistry | [GPQA](https://huggingface.co/datasets/Idavidrein/gpqa) | **Diamond BioOrganic-91**: 19 Biology + 72 Organic Chemistry |
| OOD | ScienceWorld | [official environment](https://github.com/allenai/ScienceWorld) | official task/variation split; fixed environment/simplifications per run |
| OOD | APPS Introductory | [APPS](https://huggingface.co/datasets/codeparrot/apps) | 1,000 Introductory tasks within the 5,000-task test source |

The older MBPP+ release announcement says 399 tasks, but both the downloaded
official v0.2.0 release artifact and its HF copy contain 378 distinct task IDs.
The acquisition inventory records the actual 378, without padding or relabelling
old frozen panels.

HumanEval, Omni-MATH, LiveMedBench and LiveCodeBench are **not current members**.
GPQA main/extended are not used. BioOrganic is not a strict health subset.
The six active IID/OOD constants drive new panel loading; old private panels
with removed domains are rejected rather than silently filtered and relabelled.
Historical source adapters and scorer implementations remain readable.

## Acquisition and installation

The machine-readable source catalog is
[`current_datasets.json`](../configs/evaluation/current_datasets.json).
Data, answer keys, rubrics, IDs and inventories stay outside Git. Download into
a private native-Linux filesystem directory, not a model-visible mount.

```bash
uv venv --python 3.11 "$PRIVATE_ENV"
uv pip install --python "$PRIVATE_ENV/bin/python" \
  scienceworld==1.2.3 alfworld==0.4.2 math-verify==0.9.0 \
  gdown==5.2.0 huggingface-hub pyarrow requests tqdm
CUDA_VISIBLE_DEVICES="" "$PRIVATE_ENV/bin/python" \
  scripts/download_current_datasets.py --destination "$PRIVATE_DATA"
```

Use `--only math-hard gpqa-diamond-bioorganic` for just the two added datasets.
GPQA requires authorized Hugging Face access: use the standard credential
store and accept its dataset terms yourself if access is unavailable. No mirror
is used to bypass a gate. The downloader records source revisions and actual
row counts, retains completed files, and leaves failed downloads incomplete.
It does not compute hashes, draw a scored panel, invoke a model, write training
evidence or perform optimizer/posterior/library updates.

ALFWorld downloads the three official text-game archives, not a pretrained
BUTLER agent or vision detector. Set `ALFWORLD_DATA` to its private extracted
directory. ScienceWorld requires Java; its installed package supplies the jar.
Environment package versions are acquisition versions, **not a claim of parity
with older frozen environment builds**. Freeze exact runtime profiles before a run.

## Evaluation integration

- IID default overlay: [`current_iid_step0.yaml`](../configs/evaluation/current_iid_step0.yaml).
- OOD default overlay: [`current_ood_step0.json`](../configs/evaluation/current_ood_step0.json).
- Native thinking defaults are false, with current explicit **thinking-on**
  exceptions: **AIME2026, HealthBench, Math-Hard and GPQA BioOrganic**.
  Historical frozen maps retain their original settings.
- The authorized backbone-only run uses all **30 unique AIME2026** questions
  and **64 per other benchmark** (734 episodes total). The generic IID overlay
  is not a substitute for freezing the actual formal-training-matched runtime.
- OOD keeps batch 32, seed 0, 8,000 output tokens per episode, APPS 12,000;
  these are preparation defaults, not permission to run an evaluation.
- Math-Hard publishes only `problem` to the owner. The final boxed answer is
  isolated from private solutions and judged through
  [Math-Verify 0.9.0](https://github.com/huggingface/Math-Verify) in a bounded
  sandbox. This is a declared native scoring condition, not a Luna judgment
  and not a retroactive reuse of Omni-MATH scores. Only the last explicit box
  is used; no reference-driven selection or fall-back to an earlier draft.
- GPQA keeps shuffled public choices separate from the correct-label key.
  Existing metadata/Record-ID selection preserves all 19 Biology questions;
  if a later authorized panel asks for 64, sample 45 Organic questions with
  seed 0 from sorted IDs. Installing the complete 91 does not run that panel.
- ScienceWorld retains its **raw terminal score**, including -100. Native
  scoring, full denominators, actor isolation and submission boundaries remain.
- HealthBench's external judge remains the separately declared Luna-medium
  scorer. Native owner thinking is independent of the external
  judge's reasoning-effort protocol; historical Qwen-judge scores are not relabelled.

The downloaded source populations are not a new training/final split. Prior
inspected panels remain development evidence. No old seven-domain training
configuration, B28 schedule or checkpoint is silently rewritten as a six-domain
run; new training population/schedule construction is a separate operation.
OOD sources remain outside all training evidence and update paths.
