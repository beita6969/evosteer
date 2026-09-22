# EvoSteer：论文适配实现与运行说明

本实现根据用户提供的 `EvoSteer-ICLR2027.pdf`，直接扩展当前 SKILLEV
代码架构。本轮把先前方法差异补入完整、可训练、可恢复的 EvoSteer 主链。旧 Bayesian TTB
和 SkillFlow 的目标、记录和评测身份继续保留。

逐条公式、算法和附录对照见 [方法实现对照](evosteer-method-conformance.md)，
交互阶段和数据调度接口见 [运行时契约](evosteer-runtime-contracts.md)。
这份说明区分已实现的方法、论文未唯一指定处的实现选择、尚未执行的论文实验。
合成 smoke 和单元测试都不能当成论文主表的效果证据。

## 1. 主链和代码落点

```text
公开任务 + 角色目录 + 批次冻结技能/价值头
    ↓
当前策略 πθ 或冻结参考 ρ 选择合法图动作
    ↓
GraphRuntime 立即执行 → 冻结节点执行器/共享工具环境
    ↓
保存完整历史、真实动作 token、相同前缀/语法掩码、30 维反馈
    ↓
STOP → 可信终局评价器 → 证据绑定的完整性/风险筛选
    ↓
任务/结构独立统计快照 + AnchorTB + 价值及两个诊断头 → 一次更新
    ↓
成对 W/L/T 证据、候选准入/退休、冻结作者提案
    ↓
保存 adapter、heads、optimizer、统计、技能、检验和作者预算
```

| 文件（相对本项目） | 职责 |
|---|---|
| `src/skillev/evosteer_application.py` | 完整批次、依赖注入、冻结边界和检查点 |
| `src/skillev/evosteer_cli.py` | smoke、本地模型训练和显式检查点恢复 |
| `src/skillev/contracts/evosteer.py` | 公开任务、逐动作记录、混合来源轨迹及严格序列化 |
| `src/skillev/orchestration/{actions,graph,execution}.py` | 七种动作、团队图、版本化输出、历史与即时执行 |
| `src/skillev/orchestration/{evosteer_features,budget}.py` | 30 维公开特征、共享输入输出 token 合计预算 |
| `src/skillev/orchestration/model_executor.py` | 冻结基础模型执行文字角色 |
| `src/skillev/orchestration/tool_executor.py` | 使用现有动作协议、角色权限和共享环境执行工具 |
| `src/skillev/policy/{evosteer,action_trie}.py` | 单个 LoRA 编排策略、关闭 adapter 的参考、精确语法约束概率 |
| `src/skillev/policy/state_value.py` | CPU 运行时价值、流残差和两个诊断 outcome heads |
| `src/skillev/rollout/evosteer_risk.py` | 可信环境评估与精确轨迹证据绑定 |
| `src/skillev/training/task_schedule.py` | 来源均衡、显式前 12 步课程与可恢复采样 |
| `src/skillev/scoring/anchor_tb.py` | 全子轨迹残差、标量校验版及高效可微损失 |
| `src/skillev/training/reference_statistics.py` | 同题收缩、留一法、历史结构桶及批次快照 |
| `src/skillev/training/evosteer.py` | 混合轨迹梯度与参考价值监督 |
| `src/skillev/evolution/{paired_trials,validated_admission}.py` | 配对条件、候选生命周期与全运行 alpha 账本 |
| `src/skillev/evolution/evosteer_author.py` | 独立冻结作者、六字段技能、作者窗口及费用恢复 |

## 2. 图动作确实执行了什么

| 动作 | 即时效果 |
|---|---|
| `ADD_AGENT` | 建立角色节点，按可选技能立即执行一次 |
| `ADD_EDGE` | `feedback` 投递带来源版本的消息；`revise` 投递后立即重跑目标一次 |
| `BIND_SKILL` | 绑定批次内的技能版本，并立即重跑节点 |
| `SET_OUTPUT` | 选择最终输出节点，不调用评价器 |
| `RERUN_AGENT` | 使用最新输入消息和现有环境状态重跑，产生新输出版本 |
| `DROP_AGENT` | 删除当前节点和关联边；历史与已花预算仍保留，不能直接删选中的输出节点 |
| `STOP` | 提交选中节点的当前输出，调用一次终局评价器 |

通信图可以有环，但不会递归无限传播。反馈投递、`revise` 的一次重跑与后续策略动作
都有明确边界。节点身份单调增长，删除后不复用身份。完整历史和当前团队图分开保存。

工具节点共用同一个串行世界；重跑不会撤销之前的工具副作用。节点的局部 `COMPLETE`
只返回局部结果，团队的 `STOP` 才评分。角色权限由适配器显式配置。
工具角色的 `model_maximum.model_calls/agent_turns` 必须允许完整的多轮工具调用；
`RoleSpec` 的默认值只允许一次模型调用。

## 3. 训练目标与论文对应

动作概率取实际执行 token 的**对数概率之和**。当前策略与参考策略使用完全相同的
记录前缀、动作 token 和有限语法掩码；唯一合法 token 的位置贡献恰好为零。
没有训练用 hindsight adapter，也没有未计入动作概率的额外可训练 reasoning 阶段。

\[
d_t=\log\pi_\theta(a_t\mid s_t)-\log\rho(a_t\mid s_t),\qquad
R_\beta(x)=1+(e^\beta-1)r(x)
\]
\[
\delta_{i:j}=u(s_i)+\sum_{t=i}^{j-1}d_t-u(s_j),\quad
u(s_T)=\log R_\beta(x),\quad
L_x=\frac{1}{T(T+1)/2}\sum_{i<j}\delta_{i:j}^{2}.
\]

高效计算与逐子片段枚举的损失和梯度均有对照测试。默认先计算小张量上的逐动作/
逐状态精确梯度系数，再重算每个动作并立即反向传播，整个批次只执行一次优化器更新。
因此不会同时保留整条轨迹的模型计算图。`gradient_mode="dense"` 保留直接计算的
对照实现，`"streaming"` 是默认模式；它没有套用旧 TTB 的单系数梯度实现。

非终局状态流由三部分组成：同题参考回报的收缩锚点、落后一批读取的结构校正、
基于冻结参考编码和特征的残差 MLP。任务级统计对当前自然参考轨迹做留一；类别和
全局先验保留论文所述不做留一的边界。结构校正先平均奖励比，再取对数，采用少样本
粗桶回退、伪计数和裁剪。流残差头对冻结编码不反向传播。

根和其他非终局状态都保留可训练残差；仅终局流固定，硬固定根仅作为显式消融。
自然参考的所有合法前缀、配对强制首动作之后的合法前缀进入价值和诊断监督，
强制首动作前状态不进入这两类监督。

运行时价值头只看 30 维特征和任务类型 one-hot，发布为批次冻结的 CPU 副本。
它的数值与变化作为反馈提供给编排器，不作为可信 reward。残差头使用的参考编码
在训练阶段补算，因此运行时反馈不额外调用编码模型。两个诊断头使用冻结参考编码
和同一特征，但不进入主流值或运行时提示；主目标、价值、诊断参数分组裁剪，避免
辅助损失经由统一裁剪间接改变 actor 梯度。

主要默认值与论文一致：LoRA rank 64、alpha 128、`o_proj`、actor lr `5e-6`、
head lr `1e-3`、AdamW weight decay `0.01`、梯度裁剪 `1.0`、beta `1.0`；
收缩系数 task/category/global 为 `0.5/4/2`；结构桶最小访问数 `8`、伪计数 `8`、
校正界 `0.25`。每题默认 `2 current + 2 natural_reference`，有可验证候选时再加一对。

## 4. 成对验证和技能演化

候选从空团队开始，正组在第一个 `ADD_AGENT` 绑定候选，对照组不绑定，并在整条续跑
中排除该候选。两组使用同题、同首角色、同参考模型、同价值快照、同背景技能以及
相同随机种子继续执行。两组后续动作允许随反馈分歧。强制首动作仍按完整合法支持
计算真实的当前/参考概率。

`TaskBinding.session_factory(SessionRequest)` 必须为每条轨迹创建新会话。
要进入配对证据，适配器还必须提供 `ResetReceipt`：任务、reset、环境配置、种子、
初始状态指纹及独立会话身份。应用检查两组初始指纹一致且会话不同。
交互环境的指纹应来自实际 reset 结果；它仍依赖环境适配器真实报告，不能由模型证明。
每个 `TaskSession` 还必须注入 `risk_assessor`，返回绑定该完整轨迹的
`TrajectoryRiskAssessment`。静态文字入口使用不允许工具调用的内置评估；交互环境
应提供隔离能力证明或实现自己的评估。未知/拒绝风险会阻止整批学习，不能默认安全。

技能状态为 candidate / validated / retired。候选可试用；通过检验后才成为已验证技能。
默认每类最多 3 个 validated、1 个 candidate；整个历史库最多 60 个，每类最多 12 个，
retired 仍计入历史库容量。显式 seed 单独记录为配置种子，不伪造统计验证。

以 reward `>=0.5` 为成功，保存 wins/losses/ties。同轮多对先汇总，每个比较在
有新增非平局证据时只进行一次精确单侧二项检验，两方向分别使用
\(\alpha/[2j(j+1)\ell(\ell+1)]\)。平局不触发检验，但保留在效应分母中。
全运行比较序号、观察序号、已花 alpha、重复题目和生命周期在检查点中持续保存。
默认不把同一比较中的重复题目当成新的独立证据，但仍生成这些任务的配对轨迹
供 AnchorTB 和合法参考状态监督使用。定期 validated 重评增加一对试验，不占用
当批候选所需的一对；满槽不会使候选或退休永久相互阻塞。

**冻结窗口关系到统计功效。** `value_refresh_interval=1` 时价值头每批更新，旧比较会
在上下文改变后关闭，新比较使用新的全局编号，不能把不同快照的证据直接相加。
论文规模批次可在同一批积累多条证据；小批次调试应显式增大该间隔，例如 `10`，
在相同背景条件下累积不同任务。新技能/环境改变仍会分开比较。
间隔改变属于实验配置差异，必须记录。

作者通过独立模型显式注入，不能默认为当前 actor。每个作者窗口最多提议一个技能，
只使用公开执行材料与标量分数，优先同题成功/失败对照。六字段是 name、description、
trigger、plan、pitfall、constraint；重复内容会被过滤。明显答案复制检查不是通用的
答案泄露检测保证。无效作者响应、模型或环境基础设施错误会终止当前批次，不能用假
reward 或重试样本填满批次。作者调用费用和已用窗口也随检查点恢复。

这里证明和检查的是名义 alpha 总支出，不声称自动获得适应性选择或相关任务下的
全运行 FWER 保证；这与论文附录 C 的限制一致。

## 5. 运行与恢复

在项目根目录、Python >=3.11 且已有 policy 依赖的环境中运行：

```bash
PYTHONPATH=src python -m skillev.evosteer_cli smoke --output /tmp/evosteer-smoke --steps 2
PYTHONPATH=src python -m skillev.evosteer_cli train --help
```

`smoke` 在本地构建微型 Transformer + LoRA，执行确定性的合成修复环境，不下载模型。
它检查完整更新、文件输出和检查点通路，不能衡量论文效果。

本地模型训练从 `configs/evosteer/local.example.json` 复制配置，填写真实本地模型目录、
不可变模型身份、任务来源和预算。模型加载限定本地文件并关闭 remote code。
当前生产实现把 actor 与冻结 base 放在同一配置设备上，模型路径限定本地，不接管旧分布式训练器。

`local.example.json` 明确标为 `method_mode="debug"`，用于没有作者的本地接入。
完整方法使用 `configs/evosteer/with-author.example.json`：`method_mode="full"`、
独立冻结作者、每题 `2+2`、每批价值发布、节点/动作上限为 null、完整历史模式、
配对结构统计默认开启，batch_size 示例为 56。CLI 会拒绝把禁用作者、静默历史投影、
额外图上限或其他不一致选项标成 full。示例的 100 训练步是可修改的运行占位配置，
不声称来自论文。基础模型须实际支持所配置的上下文窗口。

作者有独立的累计调用预算。没有作者的 smoke 或 debug 不能作为完整技能演化的证据；
方法集成测试另外覆盖作者提出候选、下一批成对验证、所有训练头以及恢复。

```bash
PYTHONPATH=src python -m skillev.evosteer_cli train \
  --config /absolute/path/my-evosteer.json \
  --tasks /absolute/path/private-tasks.jsonl \
  --output /absolute/path/new-run
```

基础 JSONL 入口用于静态文字 exact-match 任务；`expected_answers` 仅供评价闭包，
不会写入 `EvoTask` 或模型提示。它不能代替 HealthBench rubric、代码沙箱或交互任务
的原生评分器。那些任务通过显式 task factory 返回现有环境支持的 `TaskBinding`，
并注入 `FrozenToolExecutor` 和可信 evaluator。

JSONL 的最小合成示例（不是任何真实数据集条目）：

```json
{"task_id":"toy-1","family":"qa","prompt":"What is 2 + 2? Return the number only.","expected_answers":["4"]}
```

接入现有基准时用 `--task-factory your_package.your_module:build_tasks` 代替 `--tasks`。
工厂接收关键字参数 `policy` 与 `config`，返回非空 `TaskBinding` 序列；原生 evaluator
留在 `TaskSession` 内。评价器、私有数据或环境 reset 语义改变时，适配器必须更新
`environment_config_id`，不能仅沿用原公开题目身份。CLI 会绑定任务源、顺序和批大小，
但单个工厂模块的摘要不能自动发现它依赖的所有外部资源改变。

恢复 smoke 的例子：

```bash
PYTHONPATH=src python -m skillev.evosteer_cli smoke \
  --resume /tmp/evosteer-smoke/checkpoints/batch-000002 \
  --output /tmp/evosteer-smoke-resumed --steps 3
```

`--steps` 是恢复后要达到的总批次数，不是额外步数。

恢复必须指向一个明确完成的检查点，并使用匹配配置/数据源及新输出目录；具体参数
见 CLI help。检查点保存模型训练参数、流残差/价值/两个诊断头、已发布价值头、优化器、参考统计、
技能/比较/alpha 历史、作者状态、版本计数与累计已完成轨迹用量。它校验 JSON 与张量
摘要以及配置/模型身份；不覆盖已有目录，也不恢复半批次或外部世界的任意中间状态。
权重内容的不可变 reference 身份仍需要模型工件管理保证。

## 6. 完整方法与实验配置的边界

本轮已补齐：可训练根残差、两个诊断 outcome heads、配对合法后缀的价值/诊断监督、
配对结构统计独立通道、显式外部先验、30 维执行/修复/通信/环境反馈、风险评估、
按轮顺序验证、无默认图规模/动作/局部工具轮次上限、8 步/12 phase 交互协议、
来源均衡采样和显式课程调度。完整历史模式是默认；超过模型支持的上下文会明确失败，
`debug_head_tail` 只允许调试，配置身份不同。

论文未逐项给出的部分已经明确实现并记录，而非留下空函数：两个诊断头采用
128/GELU/sigmoid 与 reward MSE；30 个坐标和交互 phase 的实际含义见方法对照。
附录 C.3 的配对结构前缀例外和仅暴露 skill slot 标识均被保留。
自然任务统计与配对结构统计始终分离，配对 control 的先验也直接由任务传入，
不会因该上下文没有自然参考样本而丢失外部先验。

默认滞后统计跨价值版本池化是一项近似；`strict_value_context=true` 可隔离版本。
名义 alpha 账本正确不代表相关样本和适应性技能选择下无条件 FWER 已证明。
这些是论文的统计边界或明确实现选择，不以“全部实现”掩盖它们。

Algorithm 1 的数据调度已提供通用实现，但论文没有列出前 12 步的具体来源顺序和比例。
配置可增加独立 `sampling`，它不修改 family 或轨迹行为来源。例如下例只是接口示意，
正式任务映射必须与加载数据完全相同，且每源任务数足够该批配额：

```json
{
  "sampling": {
    "task_sources": {"task-a": "dataset-a", "task-b": "dataset-b"},
    "curriculum": [{"through_step": 12, "source_ids": ["dataset-a", "dataset-b"]}]
  }
}
```

每批按来源分配近等配额、批内不重复任务；最后一个课程阶段必须结束于第 12 步，
之后所有来源参与。可配置多个明确阶段，不替作者猜课程设计。采样配置与已提交步数
随 CLI 检查点恢复。未提供 sampling 时，入口明确记录 `ordered_records`，不声称来源均衡。
因此需待数据方案确定后填写真实 manifest 和课程；现有完整方法配置没有虚构它们。

尚未开展的工作是原生数据源/评分器统一、真实数据训练、主表/消融、held-out 去重，
以及异步预取、多 GPU 模型服务等工程优化。当前使用同步采样、精确的按动作流式梯度；
后者与 dense 全图梯度和一步参数更新有数值对照。原仓库的数据划分保持原样。

## 7. 验收与版本

本轮 **429 项测试全部通过、无跳过**，其中 EvoSteer 方法测试 301 项、旧公共/架构回归 128 项；
不是全仓库测试。微型 HF/PEFT debug smoke 完成两批并从检查点恢复至第三批，
全上下文作者→候选→配对闭环另由方法集成测试覆盖。源码 lint、限定范围类型检查、
wheel 构建与既有隔离检查通过。验收记录写入 `docs/evosteer-method-validation.json`，逐项测试报告为
`docs/evosteer-method-tests.xml`。早期的 `evosteer-validation.json` 和
`evosteer-tests.xml` 是第一版历史记录，不能作为本轮源码的验收结论。

历史/运行时快照、准入/参考统计、主检查点和 CLI 配置已升级相应版本。
早期 `evosteer-checkpoint@1` 不含本轮训练头、统计及方法语义，当前入口明确拒绝，
不作无声兼容转换。新检查点可恢复完整训练状态与调度，不恢复半批次的外部世界。

本地可用 CPU 环境为 `/dev/shm/skillev-cpu-runtime-20260915/bin/python`。
所有合成结果明确标记 synthetic；不以单元测试或 smoke 宣称复现论文主表。
