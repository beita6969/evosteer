# SKILLEV 现有架构与 EvoSteer 论文对照

日期：2026-09-17。来源：用户提供的 `SKILLEV-new-main.zip` 与 `EvoSteer-ICLR2027.pdf`。

本文记录改造前的代码理解、接口核对和设计分析。下文“当前”“尚未实现”均指原 ZIP 的状态。随后已按用户要求完成第一版 EvoSteer 实现；实际入口、已验证范围及仍待接入的实验条件见 [实现说明](evosteer-implementation.md)。没有修改旧方法的数学目标、数据划分或用户模型权重，也没有启动 GPU 任务。

## 1. 主要判断

这套项目适合用作 EvoSteer 的工程基础，但它当前实现的是 **单策略 ReAct/技能调用 + forward/hindsight TTB + 贝叶斯技能演化**，不是已经完成的动态图多智能体系统。

最容易混淆的是两种“图”：旧方法中节点主要是交互历史状态，边是技能/工具动作；EvoSteer 的通信图节点是带角色和技能的智能体，边是带协议的消息关系。此外，EvoSteer 仍需要一棵独立的执行历史树。给旧历史图改名不会产生团队图。

建议继续使用当前仓库、`src/skillev` 包和现有分层，在方法层增加 EvoSteer 的图运行时、评分器、训练组装器与技能准入。保留旧方法作为可比较的基线；不把新语义塞进声称仍代表旧 TTB 的记录和检查点。

## 2. 实际目录分工

| 路径 | 当前职责 | EvoSteer 中的用途 |
|---|---|---|
| `src/skillev/application.py` | 完整 Bayesian TTB 方法的依赖组装 | 参考依赖注入方式，增加 EvoSteer 组装入口 |
| `contracts/` | 动作、轨迹、奖励、评分和演化记录 | 复用公共身份/序列化，新增图与参考采样记录 |
| `policy/` | 冻结骨干、forward/backward LoRA、query-only Z、精确 token 评分 | 保留骨干/评分设施，分开当前策略、参考策略、执行器 |
| `rollout/` | reasoning/action 两阶段生成、环境执行和轨迹保存 | 保留轨迹与会话能力，增加图编排循环 |
| `runtime/` | 预算、工具、环境执行、技能文档、快照和服务调用 | 大量复用，扩展团队共享预算及节点执行 |
| `scoring/`、`training/` | TTB 目标、流式/分布式梯度、固定批次和提交 | 替换目标与梯度契约，保留调度和持久化能力 |
| `diagnostics/`、`calibration/` | forward/backward 流代理与加权 Beta 统计 | 原方法诊断可保留；不能充当新参考价值或准入依据 |
| `evolution/` | 平台期检测、五类技能修改、作者调用 | 复用作者传输/材料整理，替换决策与验证流程 |
| `orchestration/` | 目前主要是成本和技能库辅助逻辑 | 新图状态与运行时的自然落点 |
| `evaluation/`、`benchmarks/` | 公开任务接口、推理协议、结果处理 | 复用适配器，增加明确的团队评测条件 |
| `packages/private-evaluation/` | 私有数据和可信终局评测实现 | 保留答案/测试/评分载荷与模型上下文隔离 |
| 根目录 `training/`、`src/executor/`、`src/skills/` | 历史 SkillFlow 路线 | 保留基线参考，谨慎复用必要设施 |
| 根目录 `run_training.py` | 调用历史 `GFlowNetTrainer` 的 SkillFlow baseline 入口 | 不直接当作新 EvoSteer 的入口 |

证据：`pyproject.toml:55`、`run_training.py:224`、`src/skillev/application.py:302,806,867`。

仓库有多代协议和评测路径。README 的“当前状态”不能概括所有最近配置；例如代码已包含多代后续协议与 catalog-then-read 技能曝光。需要沿选定入口核对配置和调用链，不能只相信文件名或历史说明。

## 3. 现在一次任务怎样运行

```text
TrainingLoop 规划当前策略/技能版本与任务
  → execute_episode 创建环境会话
  → RolloutEngine 构造初始上下文
  → 当前策略生成 reasoning
  → 当前策略生成 action，保留原始 token IDs
  → BoundedAgent 做语法/接口校验并执行
  → 追加公开环境观察和动作记录
  → 重复，直到完成或步数上限
  → 可信终局评价器给出 reward
  → TTB 评分、更新、诊断及技能演化
```

关键代码：

- `rollout/episode_executor.py:101`：单环境会话及执行器组装。
- `rollout/engine.py:218,248`：单个 `BoundedAgentState` 和顺序循环。
- `runtime/contracts.py:142`：动作类别只有 `TOOL/SKILL/COMPLETE`。
- `runtime/bounded_agent.py:151`：只执行已经生成的动作，没有模型评分权限。
- `rollout/engine.py:667`：终局评价边界。

核心路径中没有 EvoSteer 七种图动作，也没有持久的角色节点、协议边、节点输出版本、团队输出节点。`evaluation/agent_communication.py` 是历史消息解析；`evaluation/step0_integrity.py:148` 明确拒绝当前 live 多智能体条件。`agent_messages.py` 的邮箱对象可供设计参考，但不能当成已接通的团队执行器。

当前预算预留超额会抛出错误，而非自动转成正常终局轨迹；这一点应与步数上限的终局处理区别。EvoSteer 的预算强制停止语义需要单独规定。

旧 `training/environment.py:1121` 有 supervisor 调用固定 plan/decompose 等工具，`src/executor/m_exec.py:80` 会调用外部模型。这提供了委托执行的经验，但仍不支持持久通信图，且不能直接代替新运行时的模型身份、精确预算和观察记录。

## 4. 当前学习目标与论文的差别

当前实现（`scoring/objective.py:295,408`）：

\[
\Delta=\log Z(q)+\sum_t\overline{\log F_t}
-\beta\log(r+\epsilon)-\sum_t\overline{\log B_t},
\quad L=(\Delta/T)^2.
\]

动作内先取 token logprob **均值**。F 与 hindsight B 都有可训练 LoRA；F 的前缀包含当前 reasoning，B 的前缀包含当前执行观察。query-only Z 头由冻结骨干对题目编码后预测。

论文目标：

\[
d_t=\log\pi_\theta(a_t\mid s_t)-\log\rho(a_t\mid s_t),\quad
\delta_{i:j}=u(s_i)+\sum_{t=i}^{j-1}d_t-u(s_j),
\]
\[
L=\frac1{T(T+1)/2}\sum_{i<j}\delta_{i:j}^2,
\quad u(s_T)=\log[1+(e^\beta-1)r].
\]

| 维度 | 当前项目 | EvoSteer 所需 |
|---|---|---|
| 比较对象 | 当前 forward 与可训练 hindsight backward | 相同前缀上的当前策略与冻结参考策略 |
| token 归约 | 每动作平均 | 按论文为动作 token 对数概率求和 |
| 奖励变换 | `r + epsilon` 后乘 beta 的对数 | `1 + (exp(beta)-1)*r` 的对数 |
| 残差范围 | 整条轨迹一个残差 | 所有连续子轨迹 |
| 流参数 | 题目级可训练 Z | 实测任务/结构锚点 + 状态残差头 |
| 运行时价值 | 没有论文的 30 维执行价值反馈 | 独立的参考续跑价值头 |
| batch | 当前策略的一组任务轨迹 | 每题 current/reference，另有技能配对轨迹 |
| 技能评价 | 绝对终局成功标签的加权统计 | 与不使用候选的参考续跑结果比较 |

已有 `policy/reference.py:25` 可以保存冻结 forward adapter，并在 `training/stability.py:130` 用于可选 KL。它不是论文默认的“关闭训练 adapter 的基础模型参考策略”；复用其状态隔离方式时必须重新固定 rho 的定义。需要新增 rho 采样和选中动作的 logprob 评分接口。

现有低层 `PolicyBackbone.score(prefix_ids, action_ids, role)` 保留逐 token 评分，是可复用的基础。旧奖励 DTO 强制 `shifted_reward=r+epsilon`，旧梯度路径把整条轨迹梯度统一乘 `2*Delta/T^2`，这些语义必须一起更换。

## 5. 技能系统现在如何演化

1. 从 forward/backward 对数概率差累计路径流代理。
2. 在实际调用技能的边上，将流权归一化为批内均值 1。
3. 对每个技能/上下文单元更新 Beta 伪计数；同一轨迹上的调用都使用终局 success 标签。
4. 残差平台期和技能使用熵下降满足条件后，开启演化阶段。
5. 根据 flow 分位数和均值±k标准差的上下界，决定 retain/compress、refine、split、prune 或 generate。
6. 生成/改写技能直接进入 active library；修改库后重置 Z，继续训练。

证据：`diagnostics/core.py:188`、`calibration/core.py:162,178,319`、`evolution/decision.py:319`、`evolution/execution.py:95`、`evolution/loop.py:830`。

它已经有不可变技能文档、适用条件、来源、库版本、作者响应保存和批次快照。`runtime/skill_library.py:46` 的状态却没有候选/已验证/退休状态，也没有比较索引、W/L/T0 或 alpha 账本。EvoSteer 准入是新的统计流程，不是给当前 LCB 调一个阈值。

两项相邻功能不能混淆：

- `evolution/trajectory_contrast.py` 比较自然轨迹，明确 `causal_claim=False`。
- 私有评测 `experiments/skill_utility.py` 有整库 off/on 对比，但明确不验证单个技能，也不写训练或演化状态。

技能曝光也需要区分：通用默认仍可全文 inline；`configs/training/bayesianimprove_autonomous_ttb_six_domain.yaml:52` 使用 catalog-then-read。后者读取技能后只是向同一个 actor 提供建议，尚不等于给独立节点绑定技能。

## 6. 迁移中不能遗漏的接口问题

- **reasoning 的概率归属**：当前 reasoning/action 都由训练策略生成，但只评分 action。新方法若仍让未评分 reasoning 随策略变化，完整历史的 pi/rho 比值会缺项。首版建议不设单独的可训练自由 reasoning 阶段；其他方案需显式定义概率。
- **合法动作掩码**：当前契约声明 raw-full-vocabulary，事后 JSON 校验不是语法掩码。论文若保留 masked sampling，生成、pi 评分、rho 评分必须共享同一掩码和归一化。
- **历史与模型输入**：轨迹保留完整历史，但 `policy/interface.py:213` 可以裁剪输入。完整历史树身份与实际模型看到的投影应分别保存，pi/rho 使用相同投影。
- **零/多次调用的图操作**：当前 BoundedAgent 执行动作的工具计数约束不适合所有图编辑。新编排执行层应复用预算账本，按实际节点模型/工具调用结算。
- **交互环境**：团队共享一个真实环境状态；重跑不是回滚已发生的环境变化；删除节点不能删除历史或退款。
- **批次来源**：当前采样与评分身份要求相等。需分开 behavior policy 与 current scoring policy，记录 reference/paired/forced-first-action，不能关闭检查后混入。
- **统计时序**：当前库变化重置 Z 的路径不能顺带清空 EvoSteer 的全运行 alpha 账本；参考锚点、价值头和技能菜单按明确批次边界更新。
- **终局评价**：只有团队 STOP 后提交选定输出；内部 verifier 不能接触隐藏答案或提前调用可信终局评分器。

## 7. 论文与数据配置需单独对齐

附件论文表格规划 7 IID + 7 OOD；代码 `configs/evaluation/current_datasets.json` 则声明 6 IID + 6 OOD，例如 TriviaQA 在代码中是 IID、论文中是 OOD，HealthBench/MuSiQue/NQ-Open 也与论文清单不同。论文的 MBPP、GPQA Diamond、APPS 与代码的 MBPP+、BioOrganic 子集、APPS Introductory 不能直接视为相同评测。

本次不改变数据定义。先复用已有任务接口证明方法，再建立 EvoSteer 独立的训练/验证/测试配置，明确数据来源、题目去重、子集和预算。旧配置与旧结果保留原身份。

## 8. 本次检查及证据边界

- 对 `src/`、`training/`、`packages/`、`scripts/`、`tests/` 中 1,428 个 Python 文件进行 AST 语法解析，未发现语法错误。
- Python 3.11.16 下直接检查了公共核心导入、规范序列化/内容身份、共享预算预留/超额拒绝/结算、动作示例解析和非法参数拒绝。
- 运行现有 8 个文件中的 84 项 CPU 测试，全部通过：runtime budget/bounded agent/events and skills，rollout action contract/wire/surface，以及 identity/skill lineage。
- 此次使用本机缓存的 pytest 9.1.1，关闭无关的全局训练 fixture 插件；不是锁文件中 pytest 8.4.1 的完整 `make check`。默认 shell 的 Python 是 3.10，实际检查显式使用了满足项目最低要求的 Python 3.11。
- 当前检查环境没有完整 torch/transformers/peft 等训练依赖，没有运行模型梯度、GPU、真实基准任务、完整 pytest 或性能评估。
- 这些检查支持“部分基础接口可以复用”，不证明整个旧训练系统已经成功运行，也不证明尚未实现的 EvoSteer 有效。

下一步的具体文件和验收顺序见 `evosteer-migration-plan.md`。
