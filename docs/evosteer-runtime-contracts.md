# EvoSteer 运行时与采样接口

本文记录代码的实际实现约定。论文明确了机制、资源预算和 30 维特征的类别，但没有提供全部逐维公式、阶段适配 API 或前 12 步课程的来源顺序。本实现对这些空缺采用显式、可审计配置，不把自行补充的字段定义称为论文原始定义。

## 运行与终止

`GraphRuntime` 的 `max_nodes`、`max_actions` 默认都是 `None`；`FrozenToolExecutor.max_turns` 也默认为 `None`。团队大小、修复次数和轨迹长度不再有默认的 8/24/4 限制。指定数字属于调试配置，并进入公开状态或 executor identity。

控制器与节点使用同一 `EvoBudgetLedger`，输入和输出 token 合并计费；节点只结算实际执行费用。控制器每个具有选择余地的决策必须消耗真实模型预算，避免无费用图编辑循环。控制器预算不足时由 rollout collector 收窄为确定性的选输出及 STOP；图运行时不猜测控制器费用，不制造答案或奖励。可信 evaluator 仅在合法 STOP 后调用一次。

按论文 Remark C.3，`public_state.skill_menu` 只公开 slot ID 和内容 hash，不内联技能全文。绑定技能实际执行时，全文进入该节点的 `node_request`；完整审计历史保留该请求，actor prompt 的投影删除节点内部 prompt。

## 公开状态与交互阶段

公开状态包含 `debug_limits`、`actions_remaining`（无调试上限时为 null）、`feature_version`、`feature_names`、`execution_features` 和 `environment`。这些数据只来自公开执行记录及预算，不读取 evaluator 答案、最终正确性或风险审核结果。

交互 executor 默认每 8 次环境调用形成一个阶段，累计最多 12 阶段。阶段边界立即返回控制器；下一个节点操作继续同一个环境，不重置、不克隆、不回滚。技能文档读取、解析失败和本地权限拒绝不推进环境步数。成功返回的环境调用及具有准确费用的错误/超时反馈计为已观测步骤。

`public_environment_state()` 的字段包括：

- `enabled`、`steps_per_phase`、`max_phases`：阶段配置。两个配置同时为 null 可显式关闭此协议。
- `steps_completed`、`phase_index`、`steps_in_phase`、`phases_completed`、`remaining_steps`：实际观测进度；phase index 从零开始，全部耗尽时等于 max phases。
- `phase_budget_exhausted`：阶段资源用尽。它不等于环境原生终止或任务成功。
- `environment_terminal`：环境适配器实际返回的原生终止信号。
- `last_outcome`：最后一次环境调用的公开 status、terminal 和 public value；不推断任务正确性。
- `dispatches_started`、`poisoned`：已启动的调用及不确定失败状态。外部调用失败且费用/效果不明时，不增加已观测完成步数，禁止继续或恢复重试。

达到阶段总上限后，工具契约移除环境工具，但仍允许本地 COMPLETE 和授权技能读取。COMPLETE 只结束节点。结果的 `metadata.execution_outcome` 使用 `answered`、`yielded`、`failed` 表达公开本地执行结果；`answer_present` 只表示非空本地 completion，不表示答案正确。`termination` 区分阶段边界、阶段预算耗尽、本地预算耗尽、显式调试轮数上限和本地 completion。

## 30 维特征 schema 2

以下是 `evosteer-public-features@2` 的顺序。所有维度由实际公开记录计算，无额外模型调用。

| 下标 | 字段 | 定义 |
|---|---|---|
| 0 | node_count_saturated | 活节点数 n/(n+1) |
| 1 | edge_density | 有向边数 / n(n−1)，无有效分母时为 0 |
| 2 | role_diversity | 活节点角色种类 / 角色目录大小 |
| 3 | bound_skill_fraction | 活节点绑定总数 / (活节点数 × 可见技能数) |
| 4 | action_count_saturated | 历史动作数 t/(t+1) |
| 5–11 | last_add_agent … last_stop | 七种原子动作的末动作 one-hot |
| 12 | repair_attempted | RERUN、BIND 或 revise edge |
| 13 | repair_output_changed | 修复调用后输出文本实际变化 |
| 14 | last_execution_failed | 最近公开本地失败状态；不等同最终答错 |
| 15 | answered_fraction | 已有非空本地 completion 的活节点比例 |
| 16 | selected_output | 是否已选择合法输出节点 |
| 17 | answer_agreement | 非空答案两两文本完全相同的比例 |
| 18 | last_output_changed | 最近节点执行改变已有输出 |
| 19 | communication_delivered | 最近反馈边或节点调用实际携带消息 |
| 20 | communication_output_revised | 带消息的重执行改变已有输出 |
| 21 | environment_step_fraction | 观测环境步骤 / 阶段总步骤上限 |
| 22 | environment_phase_fraction | 已完成阶段 / 阶段总上限 |
| 23 | environment_terminal | 原生环境终止信号 |
| 24–25 | environment_call_succeeded / failed | 最近已观测环境调用的公开执行状态 |
| 26 | total_token_budget_used | 实际输入与输出 token 总数 / 合并 token 上限 |
| 27–29 | tool / time / model_budget_used | 对应实际费用 / 共享预算上限 |

没有交互环境或关闭阶段时，相应阶段特征为零；未观测到 outcome 时不猜测成功/失败。调试节点/动作上限不参与特征归一化。快照格式 `evosteer-graph-runtime@2` 校验相同预算、完整历史、executor identity 和公开环境状态；它不负责恢复外部世界。

## 来源平衡与课程

`SourceBalancedTaskSchedule(task_sources, curriculum=..., batch_size=56, seed=0)` 按显式 `task_id → source_id` 映射平衡来源，任务 family 不参与配额。`CurriculumStage(through_step, source_ids)` 定义包含该步的静态阶段；阶段严格递增，最后一段必须到第 12 步；之后使用全部来源。没有隐含的来源顺序或比例。

每个 batch 各来源配额最多相差 1，余数跨步轮转；每来源在批内无放回抽样。容量不足明确报错，不重复填充。跨批次可重复遇到任务，配对统计由 admission ledger 另行排除不独立的重复证据。

`plan_next()` 返回不可变 `ScheduledTaskBatch`，并不推进调度；训练成功后调用 `commit(batch)`。检查点用 `state_dict()` / `load_state_dict()`，通过完整任务来源映射、课程和种子的配置 hash 拒绝错配，并确定性复现下一批。不包含任何具体数据集 loader。
