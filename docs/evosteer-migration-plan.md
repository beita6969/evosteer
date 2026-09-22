# 在现有 SKILLEV 架构内实现 EvoSteer 的改造路线

日期：2026-09-17。本文保留最初的改造路线。第一版核心闭环现已实现，文件名和实际覆盖范围以 [EvoSteer 实现说明](evosteer-implementation.md) 为准；本文后续阶段仍是实验扩展路线，不能作为全部完成的声明。

目标是在当前项目中形成论文要求的“即时修图 → AnchorTB 训练 → 成对证据管理技能”闭环，复用已有模型服务、精确 token 记录、任务适配、预算和持久化。

## 1. 修改边界

沿用 `src/skillev/{contracts,policy,rollout,runtime,scoring,training,evolution,orchestration}` 分层。不另起一套仓库，不先重写所有基础设施。

增加显式 `EvoSteerApplication` 与方法配置，保留 `SKILLEVApplication` 和 SkillFlow baseline 的原目标、旧记录和测试。旧 TTB 的 hindsight/Beta/Z-reset 语义不进入 EvoSteer 主链；底层共有设施可复用。

历史方法文档中的“不可修改科学定义”描述针对旧方法。用户现在要求面向 EvoSteer 改造；应新增 EvoSteer 规范并更新入口说明，不让历史指令阻断当前工作，也不静默把旧实验改名为新方法。

## 2. 建议的文件落点

| 路径（均在当前仓库） | 新增/调整职责 |
|---|---|
| `src/skillev/orchestration/graph.py` | 节点、协议边、选定输出、节点结果版本、当前团队状态 |
| `src/skillev/orchestration/actions.py` | 七种图动作的结构化 schema、状态相关合法性 |
| `src/skillev/orchestration/execution.py` | 即时应用动作、调用冻结节点、生成公开执行反馈 |
| `src/skillev/orchestration/features.py` | 明确版本的 30 维公开执行特征 |
| `src/skillev/contracts/evosteer_trajectory.py` | 完整历史、原始动作 token、图变化、各模型身份 |
| `src/skillev/contracts/evosteer_batch.py` | 当前/参考/实验/对照来源与批次快照 |
| `src/skillev/policy/evosteer.py` | 当前策略、冻结基础参考、冻结执行器的独立接口 |
| `src/skillev/policy/state_value.py` | 运行时价值头与训练流残差头，分别管理 |
| `src/skillev/rollout/evosteer.py` | 一个原子图动作对应一个可追溯状态转移 |
| `src/skillev/scoring/anchor_tb.py` | 相同前缀/掩码的动作概率比与子轨迹目标 |
| `src/skillev/training/reference_statistics.py` | 参考样本、任务收缩、滞后结构统计和快照 |
| `src/skillev/training/evosteer.py` | 混合轨迹采样、梯度更新、价值监督、批次提交 |
| `src/skillev/evolution/paired_trials.py` | 同初始条件、同首角色的 +/- 候选参考续跑 |
| `src/skillev/evolution/validated_admission.py` | 候选生命周期、W/L/T0、序贯检验和全运行预算 |
| `src/skillev/evosteer_application.py` | 新方法的依赖组装与检查点状态 |
| `configs/evosteer/`、`scripts/run_evosteer.py` | 明确的新入口与配置；不复用旧入口名义 |

文件名可在实现时按现有依赖关系微调；应避免一个巨大的 trainer 同时负责建图、调用模型、计算梯度和检验技能。

## 3. 第一阶段：先完成可观察的在线修图

### 状态

至少需要 `RoleSpec`、`NodeSpec`、`ProtocolEdge`、`NodeExecutionResult`、`GraphState`、`ExecutionEvent` 和只读的 `SkillMenuSnapshot`。

图状态保存当前节点/边/输出；历史保存所有动作与观察。节点重跑产生新输出版本，删除只改变当前图，不移除旧输出和预算记录。相同图通过不同历史到达仍是不同训练状态。

### 动作语义草案

| 动作 | 首版建议语义（实现前固定在方法规范） |
|---|---|
| ADD_AGENT | 校验角色/技能/预算后建立节点，并执行一次冻结节点模型 |
| ADD_EDGE | 校验端点与协议，投递带来源输出版本的消息；是否触发执行由协议显式规定 |
| BIND_SKILL | 绑定批次可见的不可变技能版本；本动作的执行效果需明确，不能悄悄推迟 |
| SET_OUTPUT | 指定合法输出节点及输出选择规则，不调用可信评分器 |
| RERUN_AGENT | 在声明的当前输入和环境状态下再次执行，记录新结果 |
| DROP_AGENT | 移除节点及关联当前边；若涉及输出节点则要求先修正输出选择或拒绝 |
| STOP | 提交当前合法输出，进行一次可信终局评分 |

需要先明确 ADD_EDGE/BIND_SKILL 的即时作用，避免“一次动作”暗含不受限的递归执行。反馈图可以有环，但消息传播必须有明确边界；重跑由显式动作或声明的协议触发，不能无限自动循环。

交互任务只能有一个串行共享环境。RERUN 不恢复过去的环境快照，也不撤销工具副作用。节点的只读/写入权限来自角色配置。

### 验收实例

用脚本化 CPU 执行器和合成公开任务构造：

```text
添加 solver（输出一个可修正的错误）
→ 添加 verifier（指出错误）
→ 添加 verifier→solver 反馈边
→ 重跑 solver（读到反馈后修正）
→ SET_OUTPUT solver
→ STOP
```

验收必须观察到输出确实变化，而不仅仅检查动作名字存在。同时验证绑定与删除、消息版本、非法端点、环传播边界、预算耗尽、环境副作用、完整历史和终局评分次数。

首版还要规定：预算强制结束/无输出如何记分；合法失败保留为样本，基础设施错误与任务失败分开。

## 4. 第二阶段：冻结执行器与参考策略，接入实际模型

把三个角色的身份分开：

- `pi_theta`：可训练编排器。
- `rho`：按论文关闭训练 adapter 的冻结参考编排器。
- `executor`：冻结的角色执行模型，可与 rho 共用基础权重，但生成配置和调用权限独立记录。

当前 `QwenFrozenForwardReference` 捕获的是一份 forward adapter 快照；若直接复用它作为 rho，会改变论文定义。可借用它的隔离和保存机制，不能借名字跳过这一区别。

当前与参考必须对相同完整动作 token、前缀投影、停止规则、合法掩码计算概率；节点输出属于环境观察，不计入编排动作 logprob。若使用 grammar mask，生成和两侧 teacher-forcing 都必须进行相同的掩码归一化。

首版建议编排器直接生成计分的结构化动作，不另生成未计分的可训练 reasoning。若保留 reasoning，需要并入完整可计分动作，或声明为相同冻结过程产生的上下文。

价值特征全部来自公开执行记录和预算，不允许包含终局真值。模型看到的压缩上下文与完整历史状态分别记录，不能用当前图 hash 代替完整状态身份。

## 5. 第三阶段：AnchorTB 的最小正确训练实现

先实现短轨迹、直接自动微分的清晰版本，再接入现有流式/分布式引擎。

1. 每个动作采用 token logprob 求和，计算 current/reference 比值。
2. 终点锚定 `log(1 + expm1(beta)*r)`，不沿用旧 `r+epsilon`。
3. 内部流为 `stop_gradient(clip(task_anchor + structure_correction)) + residual_head`；残差头在 stop-gradient 和 clip 之外。
4. 运行时价值头预测参考续跑得分，与训练流残差头分开；不能把旧 query-only Z 改名后使用。
5. 为 actor 与各头设置明确优化器组，冻结 rho/executor/参考编码。旧 hindsight 参数不进入新目标。
6. 显式决定根节点是否硬锚定，记录论文现有实现边界；不能同时让残差自由改根值，又声称根值始终等于实测 log Z。

旧流式实现的统一轨迹系数不再适用。新目标需要逐动作和逐状态系数，相关 distributed/provisional 数据类型也需适配，不能只替换最终 loss 行。

可在正确性验证后使用以下等价式降低残差计算开销。令 `p_t=sum_{k<t} d_k`、`y_t=u_t-p_t`，则：

\[
\delta_{i:j}=y_i-y_j,\qquad
L=\frac{T+1}{K_T}\sum_{t=0}^{T}(y_t-\bar y)^2.
\]

这仅适用于论文“所有连续片段等权”的目标，不减少模型前向评分成本。应用前需与显式枚举的损失、actor 梯度和流头梯度比较。

验收包含单动作/多动作手算例、奖励端点、掩码一致性、冻结参数不变、各梯度组正确、完整检查点恢复。不用依赖真实任务的得分上涨来代替目标实现正确性。

## 6. 第四阶段：参考统计与混合 batch

每个任务建立多条计划：`2 current + 2 reference`；有候选时再加 treatment/control。记录 task/source/reset、behavior policy、current scoring policy、executor、技能菜单、价值头、锚点版本，以及 pair/candidate/forced-action 信息。

先使用同步批次验证定义，再启用论文的一批滞后异步采样。异步只影响行为版本，所有训练动作按当前 pi 与固定 rho 重评分。

参考任务基线使用自然 reference 样本；分层收缩先实现纯函数，任务层留一、类别/全局是否留一写清楚。结构统计在下一批读取。建议首版 paired 轨迹不进入任何自然参考锚点，避免强制首动作影响统计；这比论文附录披露的混入结构桶实现更严格，应在方法说明中标注。

rho 权重不变不代表背景分布永远不变：技能菜单和价值反馈会跨批改变。统计应带背景版本/时间窗口，明确何时可池化、何时重开统计，不把所有历史样本无条件看成同分布。

检查点需原子保存 actor、value、residual、参考身份、锚点统计、技能状态、批次来源和比较账本。借用现有训练提交/快照发布结构，不继承无理由的 Z 重置。

## 7. 第五阶段：候选试用与统计准入

技能生命周期新增 `candidate / validated / retired`，与不可变文档 ID、内容版本、生成材料和适用任务族关联。候选在验证前可被选择；对照组整个 episode 都不可见该候选。各节点在实际绑定执行时得到技能全文。

每个作者窗口最多生成一个候选；作者只接收公开且已评分的轨迹材料，不接收隐藏答案。复用现有作者请求、结构校验、实际用量、响应保存和内容身份。

每次比较从同一公开初始条件及同一首角色开始，分别绑定/不绑定候选，两臂由 rho 继续，其他已验证技能、环境配置、预算和价值快照一致。独立会话和完整配对是入账条件；一臂基础设施失败不能变成另一臂的胜利。

统计 reducer 保存：全运行 alpha、单调比较索引 j、观察索引 ell、W/L/T0、已消费 pair IDs、背景条件、检验 p 值、门槛及决定。按论文使用 `r>=0.5`；原始 native success 与此阈值分开保存，不能混用。

\[
\alpha_{j,\ell}=\frac{\alpha}{j(j+1)\ell(\ell+1)},\quad
p^+=P\{Bin(W+L,1/2)\ge W\}.
\]

负向类似；每方向一半预算。只有新不一致结果才触发下一次方向检验，ties 仍进入净效应分母。明确比较符号边界、重复 pair 去重、恢复时不重复花预算；alpha 账本不可因换库/换批次而重置。

晋升后需要继续抽样和重新比较的规则，才能实现论文希望的“已失效技能退休”。旧累计证据不能自动适用于不断变化的背景；新注册比较仍消耗同一个运行级预算。

论文附录仅证明预算求和。正式声称全流程错误率之前，还要保证各次检验的有效性，包括独立验证材料、重复题目处理、自适应候选来源与背景条件。实现公式不等于完成统计证明。

## 8. 第六阶段：端到端运行和实验

先在少量静态任务上验证完整链条，再加入共享环境的交互任务。使用当前实际可用的数据适配器，不在方法验证阶段同时替换全部模型、数据和基础设施。

新评测条件记录团队总 token/model/tool 成本、节点/边数、修复动作、修复后表现、停止行为、自然参考样本量、候选胜负平、待定比例和配对成本。准入“精度”需要额外独立评估，不能把通过自身检验的比例当作正确率。

对论文贡献至少分别考察：无即时执行、无修复动作、无运行时价值、无任务/结构锚点、无残差头、无验证准入，以及同动作空间下不同训练目标。不能只将完整团队对比单模型，就归因到 AnchorTB。

重用旧评测适配器时声明数据子集、阈值和任务划分。不要移除当前 single-owner 检查并将旧记录当成多智能体结果；增加独立的 EvoSteer 运行身份。

## 9. 建议首先落实的范围

第一个可交付修改应集中在 **图状态 + 七种动作 + 冻结执行接口 + 一条反馈修复实例 + 基础行为测试**。该范围直接证明论文的执行机制，又不要求先完成大型训练重构。

之后按顺序接入真实节点模型、AnchorTB 参考实现、参考样本/价值头、配对准入、分布式训练和正式实验。每一步都有可观察的行为或数值验收，避免把所有新模块一次接上后难以定位问题。
