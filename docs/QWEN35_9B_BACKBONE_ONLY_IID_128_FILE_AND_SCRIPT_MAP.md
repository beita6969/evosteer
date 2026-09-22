# Qwen3.5-9B Backbone-Only IID 文件与脚本地图

本文梳理当前仓库中 **Qwen3.5-9B backbone-only、exact-eight IID** 评测所涉及的
配置、运行脚本、公共运行库、私有评测适配器、产物和测试。路径均相对于仓库根目录。

## 1. 范围与计数先说明

当前 IID 清单以项目规则和 Protocol 13 exact-eight catalog 为准：

1. HotpotQA
2. TriviaQA
3. AIME 2026
4. HealthBench
5. WebShop
6. ALFWorld
7. MBPP+ hard
8. HumanEval

“128-sample”是这套评测的简称，不表示八项都强行取 128 条：

| Benchmark | 最终 panel 数量 | 说明 |
|---|---:|---|
| HotpotQA | 128 | 冻结 final panel |
| TriviaQA | 128 | 冻结 final panel |
| AIME 2026 | 30 | 使用已发布的全部 30 题，不补样、不重复采样 |
| HealthBench | 128 | 冻结的 `Random(0)` panel；由 Qwen 本地 judge 评分 |
| WebShop | 128 | 冻结 native-environment panel |
| ALFWorld | 128 | 97 seen + 31 unseen |
| MBPP+ hard | 128 | 机器 ID 为 `mbpp-plus`；成功要求 Base 与 Plus tests 同时通过 |
| HumanEval | 128 | 原始 HumanEval 164 题中的冻结 128-ID panel |

因此一次完整运行是 **7 × 128 + 30 = 926 条**。

这里的 backbone-only 具体表示：

- actor route 为 `qwen35-direct-base`；
- `adapter_policy: forbidden`；
- 不加载 LoRA、技能 adapter 或训练中间态；
- HealthBench 的 grader 也是 Qwen3.5-9B SGLang 本地 grader，不调用 OpenAI/GPT；
- 只运行一组冻结 seed。

> `docs/benchmark-evaluation-guide.md` 的 TerminalReward、答案隔离和私有产物边界仍是
> benchmark 层工程约束；但其 §1 尚保留 SpreadsheetBench 的旧九项清单。当前 exact-eight
> 运行必须排除 SpreadsheetBench，并以上述八项为准。

## 2. 当前主链：Protocol 13

当前最新的 returned-state ReAct memory v5 和 fresh exact-eight 结果位于 Protocol 13
链路。端到端数据流如下：

```text
Protocol/source/condition/runner configs
        +
Qwen SGLang service + runtime-owned service receipt
        +
private final manifests / sources / official environments
        |
        v
benchmark runner（生成 + trusted scoring）
        |
        v
private per-task artifacts + execution receipt
        |
        v
build_qwen35_protocol13_receipt.py
        |
        v
8 个 answer-free receipt JSON
        |
        v
compose_qwen35_protocol13_iid.py
        |
        v
public aggregate JSON + Markdown results
```

Targets 只在最后的 receipt admission/composition 阶段读取，不能进入模型 prompt，也不能
用于选择 parser、修改 action 或修正单题答案。

## 3. 科学与工程约束文件

| 路径 | 作用 | 是否直接被 runner 读取 |
|---|---|---|
| `idea.tex` | 项目科学方法与架构最高依据；禁止修改 | 否 |
| `docs/benchmark-evaluation-guide.md` | benchmark 层 TerminalReward、答案隔离、基础指标约束 | 否 |
| `evaluation.tex` | 历史 Protocol 10 benchmark science；用于理解指标来源，不能覆盖当前 exact-eight catalog | 否 |
| `docs/QWEN35_9B_PROTOCOL13_EXACT_EIGHT_IMPLEMENTATION.md` | Protocol 13 exact-eight 实现说明、执行边界与各 benchmark 修复概览 | 否 |
| `docs/QWEN35_9B_PROTOCOL13_EXACT_EIGHT_BACKBONE_RESULTS.md` | 最新 Protocol 13 answer-free 汇总结果与解释 | 否 |
| `docs/machine-results/qwen35_protocol13_exact_eight_128_20260830.json` | 与上述结果对应的公开机器可读 aggregate；不含逐题内容 | 否 |

## 4. Protocol 13 权威配置

| 路径 | 作用 | 主要消费者 |
|---|---|---|
| `configs/evaluation/protocol_v13.yaml` | 固定 exact-eight catalog、训练形状与执行 gate 总体身份 | 全部 Protocol 13 runner/composer |
| `configs/evaluation/protocol_v13_sources.yaml` | 固定 training、validation-IID、final-evaluation 三种 population；声明 final 数量与数据/环境类型 | runner、manifest validator、composer |
| `configs/evaluation/protocol_v13_conditions.yaml` | 八项逐 benchmark execution contract：模型、panel、prompt、decoding、parser、scorer、grader、环境、horizon、sandbox 等 | 全部 runner、receipt builder |
| `configs/evaluation/qwen35_protocol13_iid.yaml` | Qwen3.5-9B 的 decoding profiles、thinking 开关、context length、base route 和 adapter 禁用策略 | static、interactive、MBPP+ runner |
| `configs/evaluation/qwen35_protocol13_iid_targets.yaml` | target evidence、可比性角色、指标与 admission 阈值；只用于结果比较 | composer，不供模型生成使用 |
| `configs/evaluation/qwen35_service_contract.yaml` | 冻结 SGLang base service 的模型、tokenizer、route、context 和 adapter policy | 所有生成 runner、service receipt validator |
| `configs/evaluation/aime_protocol_candidates.yaml` | AIME 预-final generation profile evidence 与候选 profile | AIME static runner、校准脚本 |
| `configs/serving/qwen35_9b_direct_reference.yaml` | adapter-free Qwen3.5-9B SGLang 启动配置 | `scripts/launch_direct_sglang.py` |
| `benchmark-acquisition-lock.json` | 上游 benchmark 来源与获取方式总表；用于重建私有 source，不是 final runtime condition 的事实源 | 数据获取/准备工具 |
| `pyproject.toml` | Python workspace 与 EvalPlus、OpenAI-compatible client、Transformers 等依赖声明 | `uv` |
| `uv.lock` | 可复现 Python 依赖版本锁 | `uv sync` / `uv run` |

当前 `protocol_v13_conditions.yaml` 中的重要 final profiles：

| Benchmark | Condition / prompt 要点 |
|---|---|
| HotpotQA | raw ten-passage context，non-thinking，官方式 EM/F1 |
| TriviaQA | context short answer，保留完整 aliases，官方式 max-over-alias EM/F1 |
| AIME 2026 | thinking + boxed integer，完整 30 题 |
| HealthBench | official direct candidate + Qwen local rubric grader |
| WebShop | `webshop-native-react-memory-v5@1`，10-step native horizon |
| ALFWorld | `alfworld-native-react-memory-v5@1`，20-step native horizon |
| MBPP+ hard | deterministic code generation + EvalPlus Base/Plus |
| HumanEval | deterministic code generation + original HumanEval tests |

## 5. 顶层运行与编排脚本

### 5.1 模型服务

| 路径 | 作用 |
|---|---|
| `scripts/launch_direct_sglang.py` | 从 serving config 构造并启动 adapter-free SGLang server；要求显式 GPU 映射 |
| `scripts/write_qwen35_service_receipt.py` | 为已启动的 SGLang 实例写 runtime-owned service receipt，供 runner 校验实际服务身份 |

SGLang 进程本身不包含 benchmark 数据。Runner 通过 OpenAI-compatible endpoint 调用它，
并通过 service receipt 证明实际 route 是冻结底座而不是 adapter route。
这里的 OpenAI-compatible 仅指本地 HTTP wire/API client；HealthBench 不会因此调用 OpenAI
模型，candidate 与 grader 都绑定到本地 Qwen service。

### 5.2 Panel 与环境准备

| 路径 | 作用 | 备注 |
|---|---|---|
| `scripts/prepare_qwen35_direct_population_manifests.py` | 生成静态 benchmark 的私有 population manifests | 上游/兼容准备工具；最终仍以 Protocol 13 condition 校验为准 |
| `scripts/prepare_qwen35_direct_interactive_manifest.py` | 把 WebShop/ALFWorld 私有环境清单整理成 Protocol 13 v3 manifest，并固定 panel/horizon 身份 | 只写私有目录 |
| `scripts/prepare_mbpp_plus_protocol13_manifest.py` | 对 canonical task ID 排序后，用冻结 `Random(0)` 规则选 128 个 MBPP+ task | 结果 manifest 不进 Git |

WebShop/ALFWorld 的 v5 demonstration asset 由 train split 的成功环境 replay 构造。相关
构造代码在 `packages/private-evaluation/src/skillev_private/direct_reference/prompt_assets.py`；
运行时预期在私有 `--prompt-asset-dir` 下找到：

- `webshop_native_react_v5.yaml`
- `alfworld_native_react_v5.yaml`

这两个文件包含 train replay 信息，必须保持在私有存储，且要与 final 128 panel 做隔离。

### 5.3 八项 final-panel 生成/评分入口

| 路径 | 负责的 benchmark | 作用 |
|---|---|---|
| `scripts/run_qwen35_protocol13_static.py` | HotpotQA、TriviaQA、AIME 2026、HumanEval | 直接消费 `ExecutionContractV3`；加载私有 panel，调用 Qwen，解析，并交给 trusted scorer |
| `scripts/run_qwen35_protocol13_interactive.py` | WebShop、ALFWorld | 运行 native environment 多轮 ReAct memory v5；按冻结 record ordinal 确定 endpoint；输出环境 native reward/success |
| `scripts/run_qwen35_healthbench_official.py` | HealthBench | 分离 candidate journal 与 grader journal；candidate 和 rubric grade 都走 Qwen SGLang；支持分阶段续跑 |
| `scripts/run_qwen35_mbpp_plus.py` | MBPP+ hard | 分离 generate/score；生成 full carrier，调用 EvalPlus Base+Plus evaluator，保留失败 taxonomy |

`scripts/run_qwen35_protocol13_interactive.py` 会导入
`scripts/run_qwen35_direct_interactive.py` 的私有 case/environment loader。后者是兼容适配层，
不是当前 final run 的外层权威入口。

### 5.4 Receipt、汇总与离线诊断

| 路径 | 作用 |
|---|---|
| `scripts/build_qwen35_protocol13_receipt.py` | 将一个 benchmark 的私有 artifacts 与完成态 execution receipt 转换成 answer-free public receipt |
| `scripts/compose_qwen35_protocol13_iid.py` | 读取八个 receipt、conditions 和 targets，构造 926 条 aggregate 并渲染 Markdown |
| `scripts/replay_qwen35_protocol13_artifacts.py` | 不调用模型，离线检查既有 artifacts 的覆盖与失败分类 |
| `scripts/analyze_qwen35_direct_qa_failures.py` | HotpotQA/TriviaQA parse 与错误类型诊断 |
| `scripts/analyze_qwen35_aime_outputs.py` | AIME 严格 parser 的离线诊断 |
| `scripts/analyze_qwen35_webshop_traces.py` | WebShop 提前购买、错误商品/option、循环、horizon 等失败分类 |
| `scripts/analyze_qwen35_alfworld_traces.py` | ALFWorld task type、subgoal、循环、位置与对象状态失败分类 |
| `scripts/analyze_qwen35_direct_interactive_traces.py` | WebShop/ALFWorld 通用 trace funnel 与 no-progress 统计 |
| `scripts/analyze_qwen35_mbpp_plus_failures.py` | MBPP+ candidate-invalid、Base failure、Plus-only failure、scorer failure 分类 |
| `scripts/calibrate_qwen35_aime_protocol.py` | 只在 pre-2026 calibration population 上比较 AIME profiles；不能用 final 30 题选 profile |
| `scripts/run_qwen35_react_memory_ablation.py` | final-disjoint train/validation 小面板上的 WebShop/ALFWorld memory ablation；不是冻结 final 128 评分入口 |

## 6. 公共运行库

### 6.1 Protocol 13 类型与结果层

目录：`src/skillev/evaluation/current_iid/protocol13/`

| 文件 | 作用 |
|---|---|
| `catalog.py` | exact-eight 枚举与每项预期 final 数量 |
| `contracts.py` | `ExecutionContractV3`、interactive extension、lane 与 reference eligibility |
| `config.py` | 严格加载 conditions、targets、receipts，并拒绝 catalog/condition 不一致 |
| `runner_profiles.py` | 加载 Qwen decoding profile registry |
| `service_receipts.py` | model service contract/receipt DTO 与 replica 校验；拒绝 adapter/Lora route |
| `execution_receipts.py` | execution start/complete receipt、runtime generation/environment identity |
| `evaluator_contracts.py` | HumanEval/EvalPlus 等 code executor runtime identity |
| `receipts.py` | answer-free outcome counts、metric observations、provenance 和 run receipt |
| `adapters.py` | 将私有 per-task outcome 与 frozen panel 做 exact-ID join 的公共边界 |
| `targets.py` | typed target evidence、reference population/aggregation、metric projection |
| `admission.py` | 按 execution-equal 条件比较 receipt 与 target，fail closed |
| `gate.py` | 从 admitted receipts 计算 Protocol 13 gate |
| `aggregation.py` | 守恒地合成八项结果；不接受缺项或额外 benchmark |
| `rendering.py` | 把 aggregate 确定性渲染为不含逐题内容的 Markdown |
| `aime_evidence.py` | AIME profile 的字段级 evidence 类型 |
| `aime_profiles.py` | AIME profile registry 与 pre-final year 校准约束 |
| `aime_extraction.py` | AIME 多种严格、target-independent extraction 的离线诊断实现 |
| `healthbench_grading.py` | rubric criterion/verdict 类型和 HealthBench 聚合公式 |
| `interactive_demo_assets.py` | source-proven WebShop/ALFWorld train demonstration 类型与覆盖检查 |
| `alfworld_state.py` | 仅基于公开反馈的 ALFWorld state/task-type helper |
| `native_agent_metrics.py` | WebShop native Avg/SR 与 ALFWorld SR 的小型指标适配器 |
| `__init__.py` | Protocol 13 公共导出入口 |

另外：

| 路径 | 作用 |
|---|---|
| `src/skillev/experiments/protocol_v13.py` | 加载 Protocol 13 catalog 与三类 population role，形成科学身份对象 |

### 6.2 Adapter-free 生成与交互层

目录：`src/skillev/evaluation/direct_baseline/`

| 文件 | 作用 |
|---|---|
| `client.py` | OpenAI-compatible Qwen client、thinking-aware token preflight、确定性 replica routing |
| `config.py` | benchmark、decoding、prompt/scorer identity 等共享 immutable DTO |
| `runner.py` | 有界异步静态生成；生成 raw record、parsed attempt 与 infra/model-invalid 分类 |
| `prompts.py` | 静态 prompt registry；WebShop/ALFWorld v5 structured memory prompt 与 demonstration 渲染 |
| `parsing.py` | short answer、AIME integer、Python source、`Memory/Thought/Action` 等 answer-blind parser |
| `context_budget.py` | 在 context 足够时保留完整 episode；不足时保留结构化 memory 与最新连续 raw suffix |
| `interactive_tasks.py` | native episode loop；保存 state-before/action/state-after/action-surface；不自动替换模型 action |
| `static_tasks.py` | decoding profile 的兼容加载 helper |
| `aggregation.py`、`reporting.py` | 较早 direct-reference 结果 DTO/聚合；Protocol 13 final aggregate 由专用模块接管 |
| `protocol.py` | 旧 direct-reference YAML 兼容 loader；不能成为 Protocol 13 的第二事实源 |

其它共享文件：

| 路径 | 作用 |
|---|---|
| `src/skillev/evaluation/interactive_prompt_assets.py` | 载入 source-proven train replay asset、选择 ALFWorld task-specific demo，并验证 final 隔离 |
| `src/skillev/evaluation/healthbench_official.py` | HealthBench candidate journal、Qwen grader profile、重试与官方式 aggregation helper |
| `src/skillev/policy/token_counting.py` | 加载 Qwen tokenizer，供精确 chat-template token 计数 |

## 7. 私有评测包

代码位于 `packages/private-evaluation/`，可以提交代码；但它处理的题面、答案、rubric、
tests、task IDs、轨迹和逐题结果必须保留在私有存储。

### 7.1 通用 private direct-reference 层

目录：`packages/private-evaluation/src/skillev_private/direct_reference/`

| 文件 | 作用 |
|---|---|
| `protocol13_runner.py` | 从 Protocol 13 condition 加载/校验 static 或 interactive panel；保证 manifest 顺序和 source identity 不变 |
| `manifests.py` | population manifest DTO、读写、exact task-ID 选择 |
| `populations.py` | 题面与 target 分离的静态 population loaders |
| `skillflow_iid.py` | released SkillFlow IID wire-format 解析；Hotpot full-context 与 Trivia alias provenance |
| `evaluators.py` | HotpotQA、TriviaQA、AIME、HumanEval 的 trusted scorer registry |
| `environments.py` | WebShop/ALFWorld private native environment adapter |
| `journal.py` | 静态 generation journal、fresh output directory 与 public projection |
| `interactive_journal.py` | task-keyed interactive journal；只允许 infrastructure outcome 重试 |
| `prompt_assets.py` | 在 pinned train environments 中 replay 成功 demonstration 并写私有 asset |

### 7.2 Benchmark-specific trusted code

目录：`packages/private-evaluation/src/skillev_private/benchmarks/`

| 文件 | 负责内容 |
|---|---|
| `qa_metrics.py` | HotpotQA/TriviaQA normalization、EM、token F1、max-over-alias |
| `hotpot_context_audit.py` | HotpotQA 十段 context、supporting title/sentence 的 answer-free 完整性检查 |
| `trivia_context_audit.py` | TriviaQA model-visible wire 与 alias provenance 检查 |
| `healthbench_qwen_sglang.py` | 冻结 Qwen SGLang rubric grader；不是 GPT grader |
| `webshop_official.py` | WebShop official text environment 到公共 observation/action/reward 的严格 bridge |
| `alfworld_official.py` | ALFWorld official episode 到 observation/admissible action/success 的严格 bridge |
| `alfworld_taxonomy.py` | 六类 ALFWorld task type 与 97/31 panel 校验 |
| `official_process.py` | 启动和管理 pinned official environment 子进程 |
| `official_environment_worker.py` | WebShop/ALFWorld 环境 worker 协议与 reset/step/outcome 实现 |
| `evalplus_adapter.py` | EvalPlus candidate verdict 与 full-carrier receipt |
| `evalplus_process.py` | 隔离的 EvalPlus CLI process boundary、timeout 与输出捕获 |
| `evalplus_result_v26d6d00.py` | Protocol 13 所需 EvalPlus result schema adapter |
| `humaneval_identity.py` | 原始 HumanEval 128-ID panel、test-suite identity 与 candidate diagnostics |
| `humaneval_official.py` | HumanEval 独立进程执行、timeout/CPU/内存/进程限制 |
| `humaneval_executor.py` | 原始 HumanEval scorer receipt |
| `code_math.py` | HumanEval trusted terminal evaluator 与 code execution outcome 类型 |

## 8. 每个 benchmark 的文件链

| Benchmark | 主 runner | 私有输入/环境 | parser/scorer 核心 | 私有 runner 输出 | 公共指标 |
|---|---|---|---|---|---|
| HotpotQA | `scripts/run_qwen35_protocol13_static.py` | SkillFlow IID source + frozen manifest | `prompts.py`、`parsing.py`、`skillflow_iid.py`、`hotpot_context_audit.py`、`qa_metrics.py`、`evaluators.py` | static artifact dir | EM、F1 |
| TriviaQA | 同上 | SkillFlow IID source + frozen manifest | `prompts.py`、`parsing.py`、`skillflow_iid.py`、`trivia_context_audit.py`、`qa_metrics.py`、`evaluators.py` | static artifact dir | EM、F1 |
| AIME 2026 | 同上 | SkillFlow IID source + all-30 manifest | `aime_protocol_candidates.yaml`、`parsing.py`、`aime_profiles.py`、`evaluators.py` | static artifact dir | accuracy |
| HealthBench | `scripts/run_qwen35_healthbench_official.py` | official simple-evals source + private 128 input | `healthbench_official.py`、`healthbench_grading.py`、`healthbench_qwen_sglang.py` | candidate + grader artifact dirs | Qwen-local native rubric mean |
| WebShop | `scripts/run_qwen35_protocol13_interactive.py` | v3 private environment manifest + v5 train replay asset + official WebShop process | `prompts.py`、`parsing.py`、`context_budget.py`、`interactive_tasks.py`、`webshop_official.py` | interactive artifact dir | native average score、SR |
| ALFWorld | 同上 | v3 private environment manifest + task-specific v5 train assets + official ALFWorld process | `prompts.py`、`parsing.py`、`context_budget.py`、`interactive_tasks.py`、`alfworld_official.py`、`alfworld_taxonomy.py` | interactive artifact dir | overall SR；seen/unseen 以 diagnostics counts 提供 |
| MBPP+ hard | `scripts/run_qwen35_mbpp_plus.py` | 128 manifest + MBPP+ source + pinned EvalPlus checkout | `parsing.py`、`evalplus_adapter.py`、`evalplus_process.py`、`evalplus_result_v26d6d00.py` | MBPP+ artifact dir | Base+Plus pass@1 |
| HumanEval | `scripts/run_qwen35_protocol13_static.py` | original HumanEval source + frozen 128 manifest | `parsing.py`、`humaneval_identity.py`、`humaneval_official.py`、`code_math.py` | static artifact dir | original-tests pass@1 |

## 9. 运行时输入与生成产物

### 9.1 必须留在私有存储的输入

以下内容由 CLI 传入，仓库只保存类型与使用逻辑，不保存实际内容：

- 每个 benchmark 的 final population manifest；
- HotpotQA、TriviaQA、AIME、HumanEval 的 source files；
- HealthBench 题目、rubrics 和官方 source checkout；
- WebShop 商品/goal/index 数据与 environment manifest；
- ALFWorld game data、config、instruction 与 environment manifest；
- WebShop/ALFWorld train replay prompt assets；
- MBPP+ tasks、tests 与 EvalPlus checkout；
- service receipts；
- 所有 responses、traces、逐题 verdicts 和 grader outputs。

### 9.2 各 runner 的标准私有产物

| Runner | 主要文件 |
|---|---|
| static | `execution-start.json`、`generations.jsonl`、`per-task-results.jsonl`、`execution-complete.json`、`summary.json` |
| interactive | `execution-start.json`、`per-task-results.jsonl`、`aggregate.json`、`execution-complete.json` |
| HealthBench | 根目录 execution receipts；`candidate/candidates.jsonl`；`grades/<grader-profile>/scores.jsonl`；各自 start/complete markers 与 infra-failure journals |
| MBPP+ | `execution-start.json`、`generations.jsonl`、`generation-summary.json`、`official-samples-full-carrier.jsonl`、`evalplus-results.json`、`verdicts.jsonl`、`report.json`、`execution-complete.json` |

这些文件可能包含许可题库内容或逐题输出，不能放入 `docs/` 或 Git。

### 9.3 Answer-free receipt 与 aggregate

`build_qwen35_protocol13_receipt.py` 生成的 receipt directory 必须恰好包含：

```text
hotpotqa.json
triviaqa.json
aime-2026.json
healthbench.json
webshop.json
alfworld.json
mbpp-plus.json
humaneval.json
```

`compose_qwen35_protocol13_iid.py` 再生成：

- 一个公开 answer-free aggregate JSON；
- 一个公开 Markdown summary。

公开文件只能包含 counts、aggregate metrics、condition/provenance 和 diagnostics，不能包含
task ID、题面、答案、rubric、tests、完整模型输出或轨迹。

## 10. 主要定向测试

### 10.1 Protocol 13 contract/runtime

- `tests/evaluation/current_iid/test_protocol13_catalog.py`
- `tests/evaluation/current_iid/test_conditions.py`
- `tests/evaluation/current_iid/test_protocol13_runtime_v4.py`
- `tests/evaluation/current_iid/test_protocol13_service_receipts.py`
- `tests/evaluation/current_iid/test_evaluator_contracts.py`
- `tests/evaluation/current_iid/test_receipt_admission_v3.py`
- `tests/evaluation/current_iid/test_receipts.py`
- `tests/evaluation/current_iid/test_renderer.py`
- `tests/evaluation/current_iid/test_protocol13_repairs_v4.py`

### 10.2 WebShop/ALFWorld memory v5

- `tests/evaluation/current_iid/test_protocol13_interactive_manifest.py`
- `tests/evaluation/current_iid/test_interactive_history.py`
- `tests/evaluation/current_iid/test_interactive_memory_v5.py`
- `tests/evaluation/current_iid/test_interactive_failure_taxonomy.py`
- `tests/evaluation/current_iid/test_prompt_example_isolation.py`
- `tests/evaluation/current_iid/test_react_memory_ablation.py`
- `tests/evaluation/direct_baseline/test_client.py`
- `tests/evaluation/direct_baseline/test_context_budget.py`
- `tests/evaluation/direct_baseline/test_protocol13_parser_repairs.py`
- `tests/evaluation/test_interactive_source_demonstrations.py`

### 10.3 Benchmark-specific trusted evaluator

- `tests/evaluation/test_healthbench_official.py`
- `tests/private_evaluation/test_hotpot_context_audit.py`
- `tests/private_evaluation/test_trivia_alias_provenance.py`
- `tests/private_evaluation/test_protocol13_qa_scorer_registry.py`
- `tests/private_evaluation/test_evalplus_v26d6d00.py`
- `tests/private_evaluation/test_humaneval_identity.py`
- `tests/private_evaluation/test_generation_journal.py`

`Makefile` 提供 `make test`、`make lint`、`make format-check`、`make typecheck` 和 `make check`
等验证入口。文档修改本身不需要触发 GPU benchmark；完整 `make check` 应按项目规则在
22049 的 CPU/Linux 本地盘上运行，而不是在 WSL/E 盘反复执行。

## 11. Protocol 14 文件为何仍然存在

仓库还保留一套独立的 Protocol 14 corrected/reference-publication 链：

- 配置：`configs/evaluation/protocol_v14*.yaml`、
  `configs/evaluation/qwen35_protocol14_iid*.yaml`；
- 运行库：`src/skillev/experiments/protocol_v14.py`、
  `src/skillev/evaluation/current_iid/protocol14/`；
- runner：`scripts/run_qwen35_protocol14_static.py`、
  `scripts/run_qwen35_protocol14_interactive.py`、
  `scripts/run_qwen35_protocol14_healthbench.py`、
  `scripts/run_qwen35_protocol14_evalplus.py`；
- publication：`scripts/build_qwen35_protocol14_receipt.py`；
- 既有结果：`docs/QWEN35_9B_PROTOCOL14_EXACT_EIGHT_BACKBONE_RESULTS.md` 和
  `docs/machine-results/qwen35_protocol14_exact_eight_128_20260830.json`。

Protocol 14 是不同的 execution/target/receipt schema。不得把 Protocol 13 的 conditions、
private run directory 或 receipts 混入 Protocol 14 publication，反之亦然。当前 memory-v5
交互链和最新 fresh run 以本文前述 Protocol 13 文件为准。

## 12. 不属于当前 final backbone-only 运行的文件

以下文件可能仍有历史或诊断价值，但不是当前 926-record final 主入口：

- `configs/evaluation/qwen35_current_iid.yaml`
- `configs/evaluation/qwen35_current_iid_targets.yaml`
- `scripts/run_qwen35_current_iid.py`
- `scripts/compose_qwen35_current_iid.py`
- `configs/evaluation/qwen35_skillflow_direct_reference.yaml`
- `scripts/run_qwen35_direct_reference.py`
- Protocol 10/11/12 configs 与旧结果文件
- WebShop/ALFWorld v3/v4 prompt configs（当前 final 使用 v5）
- training、LoRA、skill evolution、BayesianImprove optimizer 相关文件
- SWE-bench、AppWorld、SpreadsheetBench 及其它 OOD/diagnostic benchmark runner

特别注意：`scripts/run_qwen35_react_memory_ablation.py` 用于 final-disjoint 的小面板实验，
不能代替 `scripts/run_qwen35_protocol13_interactive.py` 运行最终 WebShop/ALFWorld 128 panel。

## 13. 最短操作索引

如果只想定位一次 fresh exact-eight run 的关键入口，按下面顺序看即可：

1. `configs/evaluation/protocol_v13.yaml`
2. `configs/evaluation/protocol_v13_sources.yaml`
3. `configs/evaluation/protocol_v13_conditions.yaml`
4. `configs/evaluation/qwen35_protocol13_iid.yaml`
5. `configs/evaluation/qwen35_service_contract.yaml`
6. `scripts/launch_direct_sglang.py`
7. `scripts/write_qwen35_service_receipt.py`
8. 四个正式 runner：static、interactive、HealthBench、MBPP+
9. `scripts/build_qwen35_protocol13_receipt.py`
10. `scripts/compose_qwen35_protocol13_iid.py`
11. `docs/QWEN35_9B_PROTOCOL13_EXACT_EIGHT_BACKBONE_RESULTS.md`
12. `docs/machine-results/qwen35_protocol13_exact_eight_128_20260830.json`
