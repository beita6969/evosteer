# 2025--2026 LLM Agent 论文中的 IID benchmark 评测代码调研

调研日期：2026-08-28

## 0. 范围与结论

本轮以用户最新指定的 **8 项** IID 清单为准：

1. HotpotQA
2. TriviaQA
3. AIME 2026
4. HealthBench
5. WebShop
6. ALFWorld
7. MBPP+
8. HumanEval

SpreadsheetBench 已被用户删除，不应继续出现在新的 IID catalog、aggregate 或训练人口中。
当前仓库的 `docs/benchmark-evaluation-guide.md`、`docs/EVALUATION_SPEC.md`、
`configs/evaluation/protocol_v12.yaml` 和 `src/skillev/evaluation/current_iid/` 仍描述旧的
exact-nine Protocol 12，因此在下一次实现修复时必须改为 exact-eight。按每项最多 128 条、
AIME 2026 使用完整 30 条计算，一轮 backbone-only panel 应为 **926 条**，而不是旧的 1,054
条。

本调研只把以下证据称为“2025 年至今已发表在会议上的工作”：论文有会议 proceedings、
ACL Anthology 或 OpenReview 的正式 conference-paper 页面，且 GitHub 仓库由论文作者或项目
组织维护。对于论文仓库没有公开 scorer、或 2025--2026 论文不可能覆盖的新数据集，另列
“benchmark 官方/事实标准实现”，不冒充会议论文代码。

最重要的总原则是：

- **Direct backbone-only 与 search/tool-assisted agent 必须分轨**。Search-R1、Search-o1 的
  QA 数值不能作为 direct-Qwen target，但其 alias、EM/F1 和完整分母逻辑可以参考。
- **交互环境的 native score 不能被训练 reward 覆盖**。WebShop headline 要保留连续
  `task_score`，SR 才是 `task_score == 1`；ALFWorld 使用环境的 `won`。
- **代码 benchmark 必须运行官方测试器**。不要用模型 judge，也不要只跑公开示例测试。
- **HealthBench 的 scorer identity 是实验条件的一部分**。Qwen-local judge 结果必须与
  GPT-4.1 judge 的官方结果分栏，不能拿后者做 `<7 pp` 同条件门禁。
- 下文仅复述行为并给出短小的自有伪代码；未把第三方长代码拷入本仓库。实际实现时应按
  对方许可证重写接口，并保留来源注释。

## 1. 已核验的会议论文与官方仓库

| 工作 | 正式发表 | 覆盖本项目 benchmark | 官方仓库 | 本轮用途 |
|---|---|---|---|---|
| Search-R1 | [COLM 2025](https://openreview.net/forum?id=Rwhi91ideu) | HotpotQA、TriviaQA | [PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1) | QA normalization、alias EM、检索轨道边界 |
| Search-o1 | [EMNLP 2025 Main](https://aclanthology.org/2025.emnlp-main.276/) | HotpotQA、TriviaQA、AIME 类数学题 | [RUC-NLPIR/Search-o1](https://github.com/RUC-NLPIR/Search-o1) | max-over-alias EM/F1、无效答案计零、完整分母 |
| AgentSquare | [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/0ae94013da7cd459402fd77874e09ee3-Paper-Conference.pdf) | WebShop、ALFWorld | [tsinghua-fib-lab/AgentSquare](https://github.com/tsinghua-fib-lab/AgentSquare) | native WebShop score/SR、ALFWorld `won`、episode horizon |
| GiGPO / verl-agent | [NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/420c9f777c0b4f78d515e53cf74d58b2-Abstract-Conference.html) | WebShop、ALFWorld | [langfengQ/verl-agent](https://github.com/langfengQ/verl-agent) | 128 条验证配置、固定 seed、并行环境、训练 reward 与报告指标隔离 |
| Doctor-R1 | [ICLR 2026](https://openreview.net/forum?id=vQGHTyL0Jw) | HealthBench | [thu-unicorn/Doctor-R1](https://github.com/thu-unicorn/Doctor-R1) | HealthBench 条件对照；仓库未发布 HealthBench scorer |
| Learning to Generate Unit Tests for Automated Debugging | [COLM 2025](https://openreview.net/forum?id=yeVBHPLXxi) | HumanEval+、MBPP+ | [archiki/UTGenDebug](https://github.com/archiki/UTGenDebug) | agentic debugging/evaluator 隔离；不替代 EvalPlus scorer |
| HumanEval Pro and MBPP Pro | [ACL 2025 Findings](https://aclanthology.org/2025.findings-acl.686/) | HumanEval/MBPP 的扩展与 EvalPlus 数据入口 | [CodeEval-Pro/CodeEval-Pro](https://github.com/CodeEval-Pro/CodeEval-Pro) | generation contract、pass@k 管线；不能把 Pro/+ 变体冒充原版 |

此外使用三个事实标准实现校准 scorer 行为：

- HealthBench：[openai/simple-evals](https://github.com/openai/simple-evals)；
- MBPP+/HumanEval+：[evalplus/evalplus](https://github.com/evalplus/evalplus)；
- AIME 2026 的公开加载/解析参考：
  [thinking-machines-lab/tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)。

后两者不是“2025--2026 agent 会议论文”的替代品：EvalPlus 的主要论文是 COLM 2024，
tinker-cookbook 是工程实现。将它们列入是因为对应会议论文仓库常常直接依赖这些 evaluator，
而不是重新实现一个更权威的版本。

## 2. HotpotQA 与 TriviaQA

### 2.1 Search-R1（COLM 2025）

固定代码版本：
[`598e61b`](https://github.com/PeterGriffinJin/Search-R1/tree/598e61bd1d36895726d28a8d06b3a15bed19f5d3)。

具体代码：

- [`qa_em.py` 的 normalization 与 max-over-alias EM](https://github.com/PeterGriffinJin/Search-R1/blob/598e61bd1d36895726d28a8d06b3a15bed19f5d3/verl/utils/reward_score/qa_em.py#L19-L46)：小写、去英文冠词、去标点、合并空白，然后对所有 gold aliases 取最大值。
- [`qa_search_test_merge.py` 的数据映射](https://github.com/PeterGriffinJin/Search-R1/blob/598e61bd1d36895726d28a8d06b3a15bed19f5d3/scripts/data_process/qa_search_test_merge.py#L54-L110)：选择 test/dev split，并把 `golden_answers` 整体保存为 `target`，没有只保留第一条答案。
- [`evaluate.sh` 的 val-only 配置](https://github.com/PeterGriffinJin/Search-R1/blob/598e61bd1d36895726d28a8d06b3a15bed19f5d3/scripts/nq_hotpotqa/evaluate.sh#L13-L65)：完整 test parquet、4 turns、retriever top-k 3、采样 temperature 1。这说明它是 search-agent 评测，不是 direct backbone-only。

**准备参考：** normalization、完整 alias list、未生成/格式失败仍进入分母。

**明确不照搬：** 同文件的
[`extract_solution()`](https://github.com/PeterGriffinJin/Search-R1/blob/598e61bd1d36895726d28a8d06b3a15bed19f5d3/verl/utils/reward_score/qa_em.py#L62-L82)
实际要求出现至少两个 `<answer>` 区块才返回最后一个，与紧邻注释相冲突。项目 scorer 不应复制
这个条件；direct lane 应直接评分自然语言 answer，search lane 也只应要求一个闭合 final answer。

### 2.2 Search-o1（EMNLP 2025）

固定代码版本：
[`c76a700`](https://github.com/RUC-NLPIR/Search-o1/tree/c76a700fb2a948039ada577c03b3be958aa57282)。

具体代码：

- [`scripts/evaluate.py` QA 核心](https://github.com/RUC-NLPIR/Search-o1/blob/c76a700fb2a948039ada577c03b3be958aa57282/scripts/evaluate.py#L46-L117)：对每条 alias 分别计算 normalized EM、substring accuracy 和 token F1，再逐指标取最大值。
- [benchmark 路由与聚合](https://github.com/RUC-NLPIR/Search-o1/blob/c76a700fb2a948039ada577c03b3be958aa57282/scripts/evaluate.py#L219-L288)：TriviaQA、HotpotQA 都走 `mode='qa'`，所有输入都追加 metric，最终对完整列表求均值；`num_valid_answer` 只是附加诊断，不改变分母。

**准备参考的实现形状：**

```python
def qa_record_metrics(prediction: str, aliases: Sequence[str]) -> tuple[float, float]:
    pred = normalize_qa(prediction)
    em = max(float(pred == normalize_qa(gold)) for gold in aliases)
    f1 = max(token_f1(pred, normalize_qa(gold)) for gold in aliases)
    return em, f1
```

项目实现还应加入 [HotpotQA 官方 scorer](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py#L40-L57)
对 `yes/no/noanswer` 的特殊互斥处理。Search-o1 的
`extract_answer()` 使用贪婪 `\\boxed{(.*)}`，QA 聚合还通过 `eval(k)` 访问局部变量；这两点
都不应复制。应使用无代码执行的平衡花括号/显式字段解析器。

### 2.3 对本项目的直接修改建议

- HotpotQA direct lane：每条必须把该题公开 supporting context 放入 candidate prompt；scorer
  只接收候选文本与私有 aliases，不把答案或 supporting-fact labels 回灌模型。
- TriviaQA direct lane：不提供 context、不提供 search。若另做 Wikipedia/local-search 增强，
  必须命名为 retrieval-assisted lane，不能覆盖 direct backbone-only 分数。
- 两者均冻结 canonical ID 后再抽样；同一 128 panel 不能因 parse validity 重新抽样。
- 公布 EM、F1、样本数、无效候选数；headline 分母固定为 128。

## 3. AIME 2026

没有找到能够同时满足“已发表在 2025--2026 正式会议”“明确使用 AIME **2026**”“公开对应
scorer”的 agent 论文仓库。AIME 2026 在 2026 年才产生，2025 会议论文不可能对其做正式
实验；本调研不把 AIME 2024/2025 的数值或 Qwen 其它参数规模模型的分数伪装成
Qwen3.5-9B/AIME2026 target。

可核验的工程参考是 tinker-cookbook 固定版本
[`9dfcc3a`](https://github.com/thinking-machines-lab/tinker-cookbook/tree/9dfcc3a2cf44432d621a01005073dc6d76a1e87a)：

- [`aime.py`](https://github.com/thinking-machines-lab/tinker-cookbook/blob/9dfcc3a2cf44432d621a01005073dc6d76a1e87a/tinker_cookbook/eval/benchmarks/aime.py#L42-L63) 固定使用 `MathArena/aime_2026`，优先 test split；
- [单轮 prompt 与整数评分](https://github.com/thinking-machines-lab/tinker-cookbook/blob/9dfcc3a2cf44432d621a01005073dc6d76a1e87a/tinker_cookbook/eval/benchmarks/aime.py#L71-L120) 要求最终 `\\boxed{}`，解析为整数并与 gold 精确比较；
- [平衡花括号解析器](https://github.com/thinking-machines-lab/tinker-cookbook/blob/9dfcc3a2cf44432d621a01005073dc6d76a1e87a/tinker_cookbook/eval/benchmarks/_common.py#L326-L358) 比贪婪正则可靠。

**准备参考：** 固定 MathArena identity、完整 30 条、单次 generation、0--999 整数、无答案计
零、平衡花括号解析。

**需要收紧：** AIME 应优先取最后一个完整 `\\boxed{}`，并严格拒绝非整数、越界值和把
`12.7` 经 `int(float(...))` 截断成 `12` 的情况。fallback “全文最后一个数字”只能作为单独
诊断，不能悄悄改变 headline。必须在结果中冻结并报告 thinking mode、temperature、
max output tokens、pass@1/pass@k；这些条件不同的公开分数不进入 `<7 pp` 门禁。

## 4. HealthBench

### 4.1 Doctor-R1（ICLR 2026）能提供和不能提供的证据

Doctor-R1 固定代码版本：
[`2747437`](https://github.com/thu-unicorn/Doctor-R1/tree/2747437349b6ca9323a5e9448b70744db4621858)。
[README 的 HealthBench 表格](https://github.com/thu-unicorn/Doctor-R1/blob/2747437349b6ca9323a5e9448b70744db4621858/README.md#L90-L113)
证明论文确实报告 HealthBench，但仓库公开内容只有训练器、交互和 reward 文件，没有
HealthBench 数据加载、rubric prompt、grader parser 或最终 aggregate 实现。因此不能声称
“照 Doctor-R1 scorer”已经复现其分数，也不能用 README 数字给本项目 Qwen3.5-9B direct
lane 设同条件 target。

### 4.2 应采用的事实标准：OpenAI simple-evals

固定代码版本：
[`652c89d`](https://github.com/openai/simple-evals/tree/652c89d0ca9df547706735883097e9537d40dc47)。

具体代码：

- [`calculate_score()`](https://github.com/openai/simple-evals/blob/652c89d0ca9df547706735883097e9537d40dc47/healthbench_eval.py#L111-L154)：分母只累加正分 rubric；分子累加所有 `criteria_met` rubric，因此触发负分项会扣分。
- [最终聚合](https://github.com/openai/simple-evals/blob/652c89d0ca9df547706735883097e9537d40dc47/healthbench_eval.py#L222-L270)：先对 record raw score 求均值，再把总体均值 clip 到 `[0,1]`，并报告 bootstrap 标准差和样本数。
- [128 条抽样](https://github.com/openai/simple-evals/blob/652c89d0ca9df547706735883097e9537d40dc47/healthbench_eval.py#L320-L389)：完整数据读取后使用 `random.Random(0).sample(examples, num_examples)`。
- [每条 rubric 的 grader 调用](https://github.com/openai/simple-evals/blob/652c89d0ca9df547706735883097e9537d40dc47/healthbench_eval.py#L397-L493)：把原对话与候选回答拼接，每条 rubric 独立判断 `criteria_met`。
- [候选回答路径](https://github.com/openai/simple-evals/blob/652c89d0ca9df547706735883097e9537d40dc47/healthbench_eval.py#L495-L518)：candidate 直接回答原始对话，scorer 使用 candidate 实际收到的 message list；没有 agent action JSON 包装。

根据用户此前明确要求，项目仍使用 **Qwen3.5-9B SGLang 自己做 grader**，不调用 OpenAI
GPT。除 grader identity 外，其余尽量与 simple-evals 相同：direct answer、Random(0) panel、
同 rubric prompt、同逐项判断、同原生单题公式、同“先均值后总体 clip”聚合。

报告必须同时写清：

```text
HealthBench Full / 128-sample / Qwen3.5-9B-local-judge native rubric mean
```

不能写成 official GPT-4.1-comparable HealthBench score。TTB 的逐题 clip reward 和项目自定义
二元 success 可保留，但必须作为另外两列，不能覆盖 native mean。

## 5. WebShop

### 5.1 AgentSquare（ICLR 2025）

固定代码版本：
[`8f5b3fe`](https://github.com/tsinghua-fib-lab/AgentSquare/tree/8f5b3fe5d8a32f9b59d20370823bef2a2c86928c)。

- [环境终页解析](https://github.com/tsinghua-fib-lab/AgentSquare/blob/8f5b3fe5d8a32f9b59d20370823bef2a2c86928c/tasks/webshop/webshop_run.py#L112-L183) 直接读取页面上的 `[0,1]` 连续 reward。
- [episode 聚合](https://github.com/tsinghua-fib-lab/AgentSquare/blob/8f5b3fe5d8a32f9b59d20370823bef2a2c86928c/tasks/webshop/webshop_run.py#L328-L355) 报告 `mean(reward)` 与 `reward == 1` 的 SR，并跑 fixed session IDs。
- [agent horizon](https://github.com/tsinghua-fib-lab/AgentSquare/blob/8f5b3fe5d8a32f9b59d20370823bef2a2c86928c/tasks/webshop/webshop_run.py#L236-L292) 为 15 steps。

不应复制这个 wrapper 的 ad-hoc 限制：它只显示前 3 个 product，并禁用了 Next page。项目应
使用原始 WebShop 环境的完整 legal action/observation surface。

### 5.2 GiGPO / verl-agent（NeurIPS 2025）

固定代码版本：
[`20bd331`](https://github.com/langfengQ/verl-agent/tree/20bd331bdbc9026a5668e11362178e10ab7400c8)。

- [`WebshopWorker.step()`](https://github.com/langfengQ/verl-agent/blob/20bd331bdbc9026a5668e11362178e10ab7400c8/agent_system/environments/env_package/webshop/envs.py#L40-L55) 先把环境原生连续 reward 保存为 `info['task_score']`，随后才为 RL 把 reward 改成成功 10、否则 0。
- [evaluation population](https://github.com/langfengQ/verl-agent/blob/20bd331bdbc9026a5668e11362178e10ab7400c8/agent_system/environments/env_package/webshop/envs.py#L126-L177) 使用前 500 个 test goals，并从其中无放回选择。
- [GiGPO WebShop 配置](https://github.com/langfengQ/verl-agent/blob/20bd331bdbc9026a5668e11362178e10ab7400c8/examples/gigpo_trainer/run_webshop.sh#L8-L24) 把 validation batch 设为 128，但预制 pool 是 `2 * 128`；[环境配置](https://github.com/langfengQ/verl-agent/blob/20bd331bdbc9026a5668e11362178e10ab7400c8/examples/gigpo_trainer/run_webshop.sh#L48-L61) 使用 seed 0、15 steps。因此它不是本项目 frozen-128 panel 的直接来源。

**准备参考：** 环境进程与模型进程隔离、从正式 test goal range 取样、15 steps、保留 legal
actions、保存 native `task_score`。本项目的 128 个 task ID 仍需自己在 generation 前冻结。

**关键纠错：** headline Average Score 必须聚合 `task_score`，SR 为 `task_score == 1.0`；绝不能
聚合 GiGPO 为训练而覆写的 10/0 reward。

## 6. ALFWorld

### 6.1 AgentSquare（ICLR 2025）

- [split 与环境初始化](https://github.com/tsinghua-fib-lab/AgentSquare/blob/8f5b3fe5d8a32f9b59d20370823bef2a2c86928c/tasks/alfworld/alfworld_run.py#L7-L14) 使用 `eval_out_of_distribution`。
- [终止与 native success](https://github.com/tsinghua-fib-lab/AgentSquare/blob/8f5b3fe5d8a32f9b59d20370823bef2a2c86928c/tasks/alfworld/alfworld_run.py#L48-L70) 采用 50-step horizon，并从 `info['won'][0]` 取二元 reward。
- [六类任务与 134 条 OOD 聚合](https://github.com/tsinghua-fib-lab/AgentSquare/blob/8f5b3fe5d8a32f9b59d20370823bef2a2c86928c/tasks/alfworld/alfworld_run.py#L102-L132) 不丢弃失败条目。
- [`base_config.yaml`](https://github.com/tsinghua-fib-lab/AgentSquare/blob/8f5b3fe5d8a32f9b59d20370823bef2a2c86928c/tasks/alfworld/base_config.yaml#L1-L18) 同时定义 `valid_seen`、`valid_unseen`，并启用全部六个 task types。

### 6.2 GiGPO / verl-agent（NeurIPS 2025）

- [环境 split、worker 和原生 info](https://github.com/langfengQ/verl-agent/blob/20bd331bdbc9026a5668e11362178e10ab7400c8/agent_system/environments/env_package/alfworld/envs.py#L55-L106) 将 train/eval dataset 显式传给 ALFWorld，并按 seed 创建独立 worker。
- [step 返回](https://github.com/langfengQ/verl-agent/blob/20bd331bdbc9026a5668e11362178e10ab7400c8/agent_system/environments/env_package/alfworld/envs.py#L110-L144) 保留 `info`、done 与 admissible commands。
- [GiGPO ALFWorld 配置](https://github.com/langfengQ/verl-agent/blob/20bd331bdbc9026a5668e11362178e10ab7400c8/examples/gigpo_trainer/run_alfworld.sh#L7-L16) 为 128 条验证；[后续参数](https://github.com/langfengQ/verl-agent/blob/20bd331bdbc9026a5668e11362178e10ab7400c8/examples/gigpo_trainer/run_alfworld.sh#L47-L60) 为 seed 0、50 steps。

GiGPO 的 [`compute_reward()`](https://github.com/langfengQ/verl-agent/blob/20bd331bdbc9026a5668e11362178e10ab7400c8/agent_system/environments/env_package/alfworld/envs.py#L48-L53)
把 `won` 乘以 10，适合 RL 优化但不适合 headline SR。项目报告应直接平均 `won`。

**准备参考：** valid_seen 与 valid_unseen 分别冻结、分别报告、六类任务全覆盖、50 steps、
非法 action 由环境处理、成功只取 `won`。若 128 总 panel 由 seen/unseen 组成，manifest 必须
在 generation 前固定；不能在看到结果后调整两者比例。

## 7. MBPP+ 与 HumanEval

### 7.1 2025 会议论文仓库提供的经验

UTGenDebug（COLM 2025）固定代码版本：
[`03bc690`](https://github.com/archiki/UTGenDebug/tree/03bc690c28b6331e7ff0131cb001dab66f4e546f)。

- [MBPP+Fix 的 test/canonical-solution 分离](https://github.com/archiki/UTGenDebug/blob/03bc690c28b6331e7ff0131cb001dab66f4e546f/no_ut_mbpp.py#L53-L93) 表明模型生成与 gold execution 应分阶段。
- [候选执行 timeout wrapper](https://github.com/archiki/UTGenDebug/blob/03bc690c28b6331e7ff0131cb001dab66f4e546f/no_ut_mbpp.py#L227-L250) 把 crash/timeout 与 pass 区分。

但该仓库评的是 debugging 变体 `HE+Fix`、`MBPP+Fix`、`MBPP+Fix Hard`，并且部分 utility
直接 `exec`/`eval` 字符串。它可参考 agent loop 与 backtracking，**不能**替代当前 MBPP+
v0.2.0 或 HumanEval 的隔离执行 scorer。

CodeEval-Pro（ACL 2025 Findings）固定代码版本：
[`36f292e`](https://github.com/CodeEval-Pro/CodeEval-Pro/tree/36f292eafc597b38535fbc8b4aafea8e5c654e3c)。

- [generation dataset router](https://github.com/CodeEval-Pro/CodeEval-Pro/blob/36f292eafc597b38535fbc8b4aafea8e5c654e3c/eval/inference.py#L50-L76) 区分 HumanEval、MBPP 与 Pro 变体，暴露 temperature、top-p、sample count、max tokens。
- [原始命名下实际加载 EvalPlus 数据](https://github.com/CodeEval-Pro/CodeEval-Pro/blob/36f292eafc597b38535fbc8b4aafea8e5c654e3c/eval/utils.py#L67-L115) 是重要警示：一个 runner 叫 `humaneval`/`mbpp` 并不能证明其 population 是原版。
- [pass@k harness](https://github.com/CodeEval-Pro/CodeEval-Pro/blob/36f292eafc597b38535fbc8b4aafea8e5c654e3c/eval/harness.py#L91-L101) 使用 code execution metric；[结果输出](https://github.com/CodeEval-Pro/CodeEval-Pro/blob/36f292eafc597b38535fbc8b4aafea8e5c654e3c/eval/harness.py#L176-L205) 同时保留 raw 与 sanitized score。

项目不应采用该 README 中“raw/sanitized 取较高者”的选择方式；应在 generation 前冻结唯一
code extraction 规则，否则会形成 result-dependent evaluator selection。

### 7.2 MBPP+ 的事实标准：EvalPlus

固定代码版本：
[`26d6d00`](https://github.com/evalplus/evalplus/tree/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e)。

- [`check_correctness()`](https://github.com/evalplus/evalplus/blob/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e/evalplus/evaluate.py#L79-L124) 分别运行 base 与 Plus inputs。
- [数据加载与样本完整性](https://github.com/evalplus/evalplus/blob/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e/evalplus/evaluate.py#L180-L244) 按 task ID join，拒绝缺失任务，而不是只对成功生成的行求均值。
- [Plus pass 条件与 pass@k](https://github.com/evalplus/evalplus/blob/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e/evalplus/evaluate.py#L311-L349) 要求同一 candidate 的 base 与 Plus 都通过。
- [无偏 pass@k estimator](https://github.com/evalplus/evalplus/blob/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e/evalplus/eval/__init__.py#L58-L84)；本项目每题只生成一次，因此 headline 是普通 pass@1 均值。
- [隔离、time limit、异常和 timeout](https://github.com/evalplus/evalplus/blob/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e/evalplus/eval/__init__.py#L122-L283) 以及 MBPP/HumanEval 特殊 oracle 都不应被简化掉。

**MBPP+ 准备参考：** 固定 v0.2.0、canonical-ID 排序后 `Random(0)` 抽 128、模型只见官方
prompt、唯一 extraction、每题一份 candidate、base+Plus 全通过才成功、timeout/exception
是 candidate failure，evaluator 启动失败才是 infrastructure failure。

### 7.3 HumanEval

当前权威列表写的是 **HumanEval**，不是 HumanEval+。因此：

- population 必须固定为原版 164 题中的用户指定 seed-42 128 panel；
- prompt 使用官方 function signature/docstring；
- 评分只用原版官方 tests；若同时跑 HumanEval+，只能作为单独的严格诊断列；
- 不能因为 CodeEval-Pro 内部的 `get_human_eval_plus()` 或 EvalPlus 的 Plus tests，而把结果
  继续标为 HumanEval；
- pass@1 分母固定 128，syntax error、timeout、assertion failure 都是零，不能从分母移除。

建议复用 EvalPlus 的隔离与 timeout 工程结构，但给 executor 传入原版 HumanEval tests。
不要直接在主 runner 进程中 `exec` 模型代码。

## 8. 准备在本项目中采用的统一 evaluator 结构

下面是综合外部代码后计划采用的接口形状，不是第三方代码的逐字复制：

```python
@dataclass(frozen=True)
class NativeEvaluation:
    benchmark: str
    task_id: str
    metrics: Mapping[str, float]
    candidate_valid: bool
    infrastructure_ok: bool

def summarize(panel, expected_ids):
    by_id = require_exact_task_join(panel, expected_ids)
    # Candidate failure remains a scored zero. Infrastructure failure aborts publication.
    require_no_infrastructure_failures(by_id)
    return aggregate_over_all_expected_ids(by_id)
```

每个 benchmark 的 adapter 只负责把 native evaluator 输出映射到项目的
`TerminalReward(value, success)`：

| Benchmark | Native headline | `value` | `success` |
|---|---|---|---|
| HotpotQA | EM、F1 | F1 | EM |
| TriviaQA | EM、F1 | F1 | EM |
| AIME 2026 | accuracy | correct | correct |
| HealthBench | Qwen-local native rubric mean | per-record clipped score | 项目显式二元规则，另列 |
| WebShop | average native task score、SR | native task score | `task_score == 1` |
| ALFWorld | seen SR、unseen SR | `won` | `won` |
| MBPP+ | Base+Plus pass@1 | base 与 Plus 全通过 | 同左 |
| HumanEval | original-test pass@1 | 全部原版 tests 通过 | 同左 |

## 9. 优先级明确的 codebase 修复清单

1. **先修 catalog 身份：** exact-nine 改 exact-eight，删除 SpreadsheetBench 的 active
   execution、target、aggregate 和 512-episode population；历史文档保留原名，不重写历史。
2. **建立 condition registry：** 每项冻结 dataset/version/split/panel、prompt、chat template、
   thinking、decoding、horizon、scorer version 与 headline metric。只有全部条件匹配才允许
   `<7 pp` 对比。
3. **QA：** 采用显式 max-over-alias EM/F1；Hotpot context 强制存在；Trivia direct 与
   retrieval lane 分离；移除 agent-action JSON 作为 direct answer 前置条件。
4. **AIME：** MathArena AIME2026 完整 30 条，严格最后一个 boxed integer，不做 float 截断，
   报告 pass@1 condition。
5. **HealthBench：** candidate direct response；simple-evals Random(0) 128 panel、rubric prompt、
   score formula 和 aggregate；Qwen-local grader identity 单独命名。
6. **WebShop：** 官方完整环境，15 steps，连续 native score 与 SR 同时报告，训练 shaped reward
   永不进入 headline。
7. **ALFWorld：** valid_seen/unseen 分开冻结，全部六类任务，50 steps，以 native `won` 评分。
8. **代码题：** MBPP+ 固定 EvalPlus v0.2.0 base+Plus；HumanEval 保持原版 population/tests；
   两者使用隔离进程、资源限制和完整 128 分母。
9. **失败语义：** candidate failure 计零且不重跑；仅明确的 generation/evaluator
   infrastructure failure 可重试；重试按 task ID 覆盖未提交槽位，不能重复计分。
10. **报告：** 每项列出 population、条件、evaluated count、candidate/infrastructure failure、
    native metric、吞吐与 wall time；答案、测试、rubrics 和逐题结果仍只留私有存储。

## 10. 不能从本调研推出的结论

- 未找到同条件 Qwen3.5-9B + AIME2026 的正式会议 target，不能凭 Qwen 其它规模或其它年份
  AIME 数值补造。
- 未找到 Doctor-R1 公开的 HealthBench scorer；其 README 分数不能验证本地 grader parity。
- 未找到同条件 Qwen3.5-9B + MBPP+ v0.2.0、单样本 greedy/direct prompt 的会议 target；应先
  修 scorer 和执行条件，再单独确定 target provenance。
- Search-R1/Search-o1、AgentSquare/GiGPO 的模型、prompt、检索/agent scaffold 与本项目
  backbone-only 条件不同。它们能校正 evaluator 语义，不能直接提供 `<7 pp` 数值 target。
- 128 条诊断 panel 的方差可能超过 7 pp；达到工程 parity 前必须先确保 population 和 scorer
  一致，不能靠换 panel、只统计有效输出或增加 candidate retry 来“提分”。
