# phase3-goal.md —— Bayesian TTB Self-Skill-Revolution Loop：第 3 轮 · 打分与目标层（scoring & objective）

> **Protocol-v3 immutable amendment (2026-07-25).** 本文件是历史实施记录，不再是当前跨层语义权威；当前权威见 `method-v3-final-spec.md`。TTB rollout 的 `generate_policy` 与 teacher-forced `score` 都对应 raw categorical softmax（temperature=1、top_p=1），authoring 使用独立 `generate_base` 请求类型。`z_value` 只接收 query-only token IDs；方向只接受 `AdapterRole`，张量直接使用明确接口，不做字符串兼容或 duck-typing capability probe。

本文件是本轮唯一任务清单，接续已验收的第 1 轮（轨迹数据层 contracts）与第 2 轮（模型层 policy backbone）。写法是讲解式的：每节先说做什么，紧跟着解释为什么、以及做错时的典型症状。若某处字面指令与所附理由看起来冲突，按理由描述的意图行事并留一句 note 说明。

## 0. 定位：八层导航图中的第 3 层

八层规划回顾：(1) 轨迹数据层（已完成——`src/skillev/contracts/ttb_*.py`）；(2) 模型层（已完成——`src/skillev/policy/`，PolicyBackbone + HFPolicyBackbone，481 测试全绿）；**(3) 打分与目标层（本轮）**；(4) rollout 引擎；(5) 训练循环；(6) 流诊断；(7) Bayesian 校准；(8) 进化层。本轮只做第 3 层——其余任何一层的逻辑都不写。

打分与目标层是什么（人话）：一组**纯函数**，输入一条 TrajectoryRecord + 模型层句柄（PolicyBackbone），输出每条边的 P̂_F/P̂_B、整条轨迹的 Δ(τ) 和 L_TTB，以及两类第 1 层契约记录（EdgeLogprobRecord、TrajectoryResidual）的物化。它自己不持有任何状态、不做优化器步、不采样、不碰模型内部——模型能力全部经由第 2 轮定下的接口取用。

为什么必须独立成层（这是 Owner 的原话，本轮的立身之本）：**这是全项目语义上最容易做错的地方**——前向条件含 r_t、后向 hindsight 不含 r_t、两边同一 K_t、只打分 a_t，四条里错任何一条都不报错、训练照跑、loss 照降。它一旦和训练循环搅在一起，错误就只能靠"loss 曲线看起来不对"这种无诊断力的信号发现；独立成层，每一条语义都能接受 tiny 底座上手算对照的单元测试，错在哪一步、错在哪个条件，测试直接指名。

## 1. 方法权威与翻译关系

本轮把 idea.tex 的五个式子翻译成函数：式 (4) P_F = π_θ(a_t | r_t, H_{t-1})——**前向条件含当前步的 r_t**；式 (5) P_B = P_φ(a_t | H_{t-1} ⊕ o_t^exec)——**hindsight：后向条件用执行观测 o_t 替换 r_t，不含当前步的 r_t**；式 (6) token 长度归一化——两个方向都对同一段 a_t 的逐 token logprob 求和后 ÷ 同一个 K_t，得到"每 token 均值"单位的 P̂_F(t)、P̂_B(t)；式 (7) Δ(τ) = log Z_θ(q) + Σ_t P̂_F(t) − β·log R̃(τ) − Σ_t P̂_B(t)；式 (8) L_TTB(τ) = (Δ(τ)/T)²，T 是步数（horizon），不是 token 数。

两个最经典的误读，先钉在这里：**第一**，"后向不含 r_t"指的是当前步的 r_t；历史 H_{t-1} 里的 (r_1,a_1,o_1)…(r_{t-1},a_{t-1},o_{t-1}) 是完整三元组，**历史步的 r 两个方向都有**（式 2 的 H 定义如此）。把历史 r 也从后向里删掉，和把当前 r_t 漏进后向里，是对称的两种错法。**第二**，式 (8) 是变分读法：Δ 里 ΣP̂_B 带负号、log Z 是可训练标量，最小化 (Δ/T)² **同时**训练 π_θ、P_φ、Z_θ 三个组件——没有守恒约束、没有"只训前向"的特殊化。本层的产出必须是一张把三路梯度都保住的计算图，backward 由第 5 层来调。

**钉死的**：上述五个式子的字面语义；P̂ 的单位 = 自然对数、每 token 均值（第 1 层 EdgeLogprobRecord 的注释原文就是这么写的）；R̃ = R + ε_min 的单次移位（第 1 层 shifted_reward 已含校验，本层直接取 record.shifted_reward，**绝不**自己再加一次 ε_min）；打分只针对 a_t 的 token（action_token_ids），r 和 o 永远只做条件不做打分对象。**工程默认（一句话理由即可调）**：β = 1.0（TrajectoryResidual.temperature_beta 的默认取值）；渲染模板的具体分隔符样式（见 §3.1）；log Z 的条件 = 渲染后的完整 H_0 文本（即 InitialContext 所承诺的 assembled 文本——理由：式 (1) 里轨迹分布的真正根状态是 H_0 = q ⊕ S_ret ⊕ ω_q，而"库变化改变 S_ret"对 Z 的影响已被 idea.tex §2.5.2 的"库更新后 Z 重置"吸收；严格按式 (7) 字面用裸 q 也兼容，条件粒度是工程自由度）。优先级照 AGENTS.md；idea.tex 与 AGENTS.md 都不要动。

## 2. 范围刀切

- 本轮是纯组合层：**运行时一行 torch/transformers/peft 的 import 都不写**。张量全部来自 backbone.score/z_value 的返回值，对它们只做方法与算符级操作（`.sum()`、四则运算、`**2`、`.item()`、`.detach()`），类型注解走 `if TYPE_CHECKING`。为此 import 边界测试本轮做一个精确化（见 §4）。
- 不写优化器、不写训练循环：TTBBatchStats 的组装（凑批、optimizer_step、library_version）是第 5 层的事，契约自带重算校验，本层只产出单轨迹的 TrajectoryResidual。
- 不写 rollout、不写诊断：式 (9) 的 log I(t) 本层只作为 EdgeLogprobRecord 的契约字段物化（step_importance = F − B，契约自己会验），窗口统计、流量聚合、相变判据都是第 6 层。
- 不做批量打分优化：一条 T 步轨迹 = 2T 次 score + 1 次 z_value 的朴素循环，正确性优先；拼 batch、并行化等第 5 层实测瓶颈后再谈。
- 第 1、2 层语义**不动**（contracts 与 policy 的既有行为一个都不改；唯一例外是 §4 说明的 import 边界测试精确化，那是测试自身的演进不是层语义变化）。

## 3. 组件逐个用人话讲（含代码骨架）

骨架规则同前两轮：字段语义与不变量不可减，命名与文件组织自由；注释说"需要写"的就是实现规格。

### 3.1 渲染器：把 H 变成文本的唯一权威

**是什么**：一组无 torch、无 tokenizer 的纯文本函数，负责把"初始上下文 + 若干步三元组 + 当前步的 r_t（前向）或 o_t（后向）"拼成一段前缀文本，并给出该前缀的哈希。**为什么单独成模块且放在本层**：前向/后向两种前缀"差在哪"这个方法学核心，必须只有一处代码在回答；而且第 4 层将来**生成时的 prompt 也必须经由同一渲染器产出**——这是把"采样条件 = 打分条件"钉死的机制。反面症状：rollout 和 scoring 各写各的模板，哪怕只差一个换行，P_F 打的就不是生成时的分布，且没有任何报错会告诉你。

**先弄清记录里有什么、缺什么**：TrajectoryRecord 的每一步存了 reasoning_text / action_text / observation_text 全文，但 InitialContext **只存 H_0 的 sha256 承诺（assembled_hash）和 token 数（assembled_token_count），不存正文**（第 1 层的设计：技能正文不重复入库）。所以本层所有打分入口都要显式收一个 `initial_text: str` 入参——**承诺在记录里，载荷由调用方提供**，打分前先验哈希再干活。第 4 层 rollout 时手里自然有这段文本；事件重放时由组装事件提供。

```python
TEMPLATE_VERSION: Final = "ttb-render@1"
# 需要写：模板版本常量。它进每一个前缀哈希——将来模板一改，旧记录的哈希立刻对不上，
# 这正是我们要的（模板漂移必须显式暴露，不能静默混批）。

# 工程默认模板（分隔符样式可调，两条纪律不可调）：
#   历史段（对 i = 1..t-1 逐步追加）：
#       "### Step {i}\nReasoning:\n{r_i}\nAction:\n{a_i}\nObservation:\n{o_i}\n"
#   前向尾段（式 4）： "### Step {t}\nReasoning:\n{r_t}\nAction:\n"
#   后向尾段（式 5）： "### Step {t}\nObservation:\n{o_t}\nAction:\n"
# 纪律一：两个方向的 action 引导符必须完全一致且以换行收尾——a_t 首 token 在两边看到的
#   紧邻左文一致，K_t 的"同一段 token"语义才对称；换行收尾也让 BPE 不易跨缝合并。
# 纪律二：r_t 允许为空字符串（第 1 层契约如此），渲染必须确定性地处理（空段照排），
#   不许因为 r 为空就改变结构。


@dataclass(frozen=True, slots=True)
class RenderedPrefix:
    """一次前缀渲染的完整产物。"""

    kind: str          # "forward" / "hindsight"，封闭集合
    step_index: int    # 1 起，与 TrajectoryStep.index 同基
    text: str          # 渲染后的前缀全文（含 initial_text 与尾段引导符）
    prefix_hash: str   # 见 prefix_content_hash；与 TrajectoryStep 里的对应字段比对


def assembled_context_hash(initial_text: str) -> str:
    """需要写：H_0 文本的承诺哈希，stable_hash 组成至少含 TEMPLATE_VERSION 与全文。
    它就是 InitialContext.assembled_hash 的取值约定——第 4 层组装时用它写入，
    本层打分时用它验证。"""


def prefix_content_hash(*, kind: str, text: str) -> str:
    """需要写：前缀哈希，stable_hash 组成至少含 kind、TEMPLATE_VERSION、text。
    它就是 TrajectoryStep.forward_prefix_hash / hindsight_prefix_hash 的取值约定。
    全仓库只允许这一个函数计算前缀哈希。"""


def render_forward_prefix(
    initial_text: str, steps: tuple[TrajectoryStep, ...], step_index: int
) -> RenderedPrefix:
    """需要写：initial_text + 步 1..step_index-1 的完整三元组 + 当前步的 r_t 尾段。
    只读 steps[step_index-1].reasoning_text，不读它的 observation_text。"""


def render_hindsight_prefix(
    initial_text: str, steps: tuple[TrajectoryStep, ...], step_index: int
) -> RenderedPrefix:
    """需要写：同上，但尾段用当前步的 observation_text。
    **绝不读当前步的 reasoning_text**——这行注释必须原样出现在实现里，
    它就是式 (5) 的 hindsight 语义。历史步（1..t-1）的 r 照常保留。"""
```

### 3.2 每边打分：从前缀到 P̂

**是什么**：对轨迹的第 t 步、给定方向，完成"渲染 → 验哈希 → 编码 → backbone.score → 归一化"的一条龙，产出该边的每 token 均值 logprob（带梯度的 0 维张量）。**为什么验哈希放在打分前**：record 里的 prefix_hash 是 rollout 时（或建记录时）对条件的承诺；打分前重渲染重算哈希并比对，等于机器证明"你训练用的数字，就是在当初生成动作的那个条件下算出来的"。库进化后技能文本变了、模板升级了、initial_text 传错了——全都在这一步炸出来，而不是变成悄悄错误的梯度。

```python
@dataclass(frozen=True, slots=True)
class ScoringConfig:
    temperature_beta: float = 1.0   # β（式 7）；有限且 > 0，命名与 TrajectoryResidual 一致


def edge_logprob_mean(
    backbone: PolicyBackbone,
    record: TrajectoryRecord,
    initial_text: str,
    step_index: int,
    direction: str,          # "forward" / "hindsight"（或用一个小 StrEnum，自定）
) -> "torch.Tensor":
    """需要写（按序）：
    1. 方向→渲染器→AdapterRole 的绑定钉死：forward → render_forward_prefix +
       AdapterRole.FORWARD_POLICY；hindsight → render_hindsight_prefix +
       AdapterRole.BACKWARD_POLICY。绑定写反 = π 的分数记在 φ 头上，单测必须能抓（§3.5）。
    2. 渲染前缀，prefix_hash 与 record 中该步的 forward/hindsight_prefix_hash 比对，
       不一致 → ValueError，消息里带 trajectory_id、step_index、是哪个方向的哈希。
    3. prefix_ids = tuple(backbone.tokenizer.encode(rendered.text))——前缀是重编码的；
       **action ids 从 record 取（steps[i-1].action_token_ids），绝不重编码**：
       它们是生成时的原始 token，第 1 层入库时已做过三重校验，重编码只会引入 BPE 漂移。
    4. per_token = backbone.score(prefix_ids, action_ids, role)；断言其长度 ==
       step.action_token_count（K_t）。
    5. 返回 per_token.sum() / K_t——每 token 均值单位（式 6），保梯度，0 维。
       求和与除法用张量方法/算符完成，不 import torch。"""
```

**显微镜点（本层版本的"栅栏"）**：两个方向打的是**同一段 action_token_ids、同一个 K_t**——这由"都从同一个 step 记录取"结构性保证，但测试仍要演示错误实现（比如前向按 len(prefix)+K_t 归一化、或后向对 r_t 文本重编码后打分）会被手算对照抓住。K_t 用 step.action_token_count，它与 len(action_token_ids) 的相等是第 1 层契约保证过的。

### 3.3 轨迹目标：Δ 与 L_TTB

**是什么**：把 T 条边的 P̂_F/P̂_B、一次 z_value、奖励项拼成 Δ 和 loss 的总装函数。**为什么返回张量而不是数**：第 5 层要对 loss 做 backward；三路梯度（forward adapter、backward adapter、Z 头）必须在同一张图里。第 2 轮为此专门设计过：两套 adapter 常年 requires_grad、未激活者不进前向图——所以本层可以先给两个方向逐边打分、最后一起 backward，中间随便切换角色。

```python
@dataclass(frozen=True, slots=True)
class TrajectoryScore:
    """一条轨迹的打分总成。张量字段带梯度；float 字段是 detach 后的诊断副本。"""

    trajectory_id: str
    horizon: int
    loss: "torch.Tensor"                        # 0 维，(Δ/T)²，带梯度
    delta: "torch.Tensor"                       # 0 维，与 loss 同图，带梯度
    log_z: float                                # z_value 的 detach 值
    forward_per_token_means: tuple[float, ...]  # 每步 P̂_F 的 detach 值，长度 == horizon
    backward_per_token_means: tuple[float, ...] # 每步 P̂_B 的 detach 值，长度 == horizon
    log_shifted_reward: float                   # math.log(record.shifted_reward)
    temperature_beta: float


def score_trajectory(
    backbone: PolicyBackbone,
    record: TrajectoryRecord,
    initial_text: str,
    config: ScoringConfig,
) -> TrajectoryScore:
    """需要写（按序）：
    1. 三重身份验证，全过才开始打分：(a) record.tokenizer_id ==
       backbone.tokenizer.tokenizer_id（换了 tokenizer 的模型不许给旧轨迹打分）；
       (b) assembled_context_hash(initial_text) == record.initial_context.assembled_hash；
       (c) len(backbone.tokenizer.encode(initial_text)) ==
       record.initial_context.assembled_token_count——这是白捡的 tokenizer/文本漂移金丝雀，
       契约里现成的字段，不用白不用。
    2. context_ids = encode(initial_text)，z = backbone.z_value(context_ids)（梯度只到 Z 头，
       第 2 层已保证）。
    3. 对 t = 1..horizon 各调一次 forward 与 hindsight 的 edge_logprob_mean，
       functools.reduce(operator.add, ...) 得 Σ P̂_F 与 Σ P̂_B（保持张量图）。
    4. delta = z + sum_forward - config.temperature_beta * math.log(record.shifted_reward)
       - sum_backward。奖励项是 Python float 常数——R̃ 直接取 record.shifted_reward，
       单次移位第 1 层已校验，这里再加 ε_min 就是双重移位。
    5. loss = (delta / record.horizon) ** 2——T 是步数。horizon ≠ ΣK_t，
       测试要构造 K_t ≠ T 的轨迹钉死这一点。
    6. detach 出各 float 诊断副本，组装 TrajectoryScore 返回。本函数自己不 backward。"""
```

### 3.4 契约物化：给事件总线的两类记录

**是什么**：把 TrajectoryScore 翻译成第 1 层契约实例——每步一条 EdgeLogprobRecord、每条轨迹一条 TrajectoryResidual。**为什么单独一步**：训练图（float32 张量）与审计记录（canonical float）是两个世界；物化函数保持纯数据（连 backbone 都不收，adapter 版本串由调用方显式传入），重放时不需要模型在场。契约自己的 `__post_init__` 会做第二道校验（step_importance = F − B、Δ 四项重算、raw_reward ∈ [0,1]）——本层算、第 1 层验，这个分工是故意的。

```python
def materialize_edge_records(
    score: TrajectoryScore,
    *,
    forward_adapter_version: str,    # 取值自 backbone.adapter_version(FORWARD_POLICY)
    backward_adapter_version: str,
) -> tuple[EdgeLogprobRecord, ...]:
    """需要写：每步一条；forward/backward_logprob_per_token 取 score 里的 detach 值；
    step_importance = 两者之差（用同一对 float 现算，别从张量另取一次）；
    scoring_stack_id 字面 "training-stack"（契约硬校验，两栈铁律 3 的接线端）。"""


def materialize_residual(score: TrajectoryScore, *, raw_reward: float) -> TrajectoryResidual:
    """需要写：sum_forward/sum_backward 用 math.fsum 对 per-step float 重算，
    delta 用四项 float 在 float64 里重算——**不要用 delta 张量的 .item()**：
    float32 图里的求和顺序与契约在 float64 里的重算可能差出容差，
    契约校验会拒收"数值上对但字节上偏"的记录。训练图与审计记录各自内部一致，
    桥梁是同一组 per-step detach 值。raw_reward 由调用方传 record.reward.value。"""
```

### 3.5 测试策略（本轮的显微镜清单）

tiny 底座直接复用第 2 轮的构造方式（GPT2 小配置 + WordLevel tokenizer、CPU、float32、无网络）；把 conftest 里的工厂提成两轮共享的 helper 或复制一份都行，一句话理由自定。测试轨迹用 build_trajectory_record 建，prefix 哈希用本层渲染器现算——这同时就是在测"渲染器与记录约定咬合"。**清单（每条都是行为断言）**：

- **手算对照（显微镜主镜）**：测试里用裸 Python 独立复算——逐边调 backbone.score 后手工 sum/÷K_t、手工拼 Δ 和 (Δ/T)²——与 score_trajectory 的输出 rtol=0, atol=0 逐位对照。
- **r_t 条件语义（本轮最重要的一组）**：字符串层——forward 前缀文本含当前步 reasoning_text、不含当前步 observation_text；hindsight 前缀恰好反过来；**历史步的 r/a/o 两个方向都完整在场**。张量层——构造一个"把 r_t 漏进后向"的错误前缀（直接用渲染器拼）得到与正确 P̂_B 不同的值，证明这类实现错误会被上面的手算对照抓住。
- **同一 K_t**：两个方向的 per-token 张量长度都 == action_token_count；演示按错误长度归一化的实现被抓。
- **方向↔adapter 绑定**：扰动 forward adapter 权重 → P̂_F 变、P̂_B 不变；恢复后扰动 backward adapter → 反过来。（手法照抄第 2 轮的 _replace_adapter。）
- **三路梯度汇流**：loss.backward() 后 forward adapter、backward adapter、Z 头**都有**非 None 梯度，冻结底座全 None——注意这正是变分读法的机器证据：ΣP̂_B 带负号也在训练 P_φ。
- **哈希链**：篡改 initial_text、改某步 observation_text、或换 TEMPLATE_VERSION 重算的记录 → 打分前 ValueError，消息指明轨迹、步、方向；assembled_token_count 对不上 → 同样拒绝。
- **纯函数性**：同输入两次调用逐位相等；调用前后 record 不变、全局 RNG 状态不变。
- **物化过契约**：EdgeLogprobRecord/TrajectoryResidual 构造即通过（契约的 step_importance 与 Δ 重算是第二道校验）；scoring_stack_id 是字面 "training-stack"；residual.raw_reward == record.reward.value、log_shifted_reward == math.log(record.shifted_reward)（ε_min 单次移位链）。
- **horizon 归一**：构造 K_t ≠ T 的轨迹（比如 2 步、每步 3 个 action token），断 loss == (Δ/2)² 而不是 (Δ/6)²。
- **空 r_t**：reasoning_text 为空串的步照常可打分，前后缀结构不塌。

## 4. 工程要求

- 新包建议 `src/skillev/scoring/`（命名自由）；渲染器模块无任何 torch/tokenizer 依赖，打分模块运行时零 torch import，类型注解经 `if TYPE_CHECKING`。
- **import 边界测试精确化（本轮唯一被允许触碰既有测试语义的地方）**：现版 AST 扫描会连 `if TYPE_CHECKING:` 块里的 import 一起算违规；改为跳过 TYPE_CHECKING 保护块内的 import（它们运行时不执行），其余规则不变——机器执法升级为"**运行时** torch/transformers/peft import 仅存在于 policy 包"。扫描器的自测样例相应补两条（TYPE_CHECKING 内豁免、外不豁免）。若实现中真的撞到必须运行时 torch 符号的场景，停下按 §8 报告，不要静默把 scoring 包加进白名单。
- 契约层与 policy 层语义不动；哈希一律用第 1 层 stable_hash；版本串照旧；单文件约 1000 行内拆分；测试断行为不断字面量。
- pyproject 无新依赖（本层只用标准库 + 既有两层）。

## 5. 真底座冒烟（有界，本轮唯一的直连 GPU 任务）

**GPU 执行硬约束**：只通过 SSH 直连私有配置的 GPU 服务器，不得使用
NCSA Delta、Duo、Slurm 或 scheduler 命令。本轮默认使用物理 H800 GPU `0`
并显式设置 `CUDA_VISIBLE_DEVICES=0`（这是该历史轮次的固定选择，不是当前全局限制）。
当前执行必须先检查全部 8 张卡，优先空闲卡；已有他人进程但 GPU 利用率低且该卡总显存
占用严格低于 25% 的，也允许在剩余显存足够时共用，不得终止、修改或调试他人进程。

tiny 的 WordLevel tokenizer 测不出真 BPE 的缝合行为——这是本轮冒烟存在的全部理由。在直连 GPU 服务器上装载真实 Qwen3.5-9B：用 generate 采几十 token 充当 a_t（配手写的 r/o 文本）拼一条 1–2 步的人造轨迹（走 build_trajectory_record 正门入库），然后完整跑一遍 score_trajectory → loss.backward() → materialize 两类记录，确认：数值全有限、三路梯度都在、两类契约记录构造通过、真 tokenizer 下 encode(前缀)+记录里的 action ids 拼接打分无异常。记录耗时、显存峰值、Δ/loss 的数值，commit-message 级 note。**有界**：不训练、不多轨迹、不接任务环境。被权重、私钥或 GPU 服务器阻塞 → 按 AGENTS.md 说明缺什么后停机。

## 6. 验收标准

1. 渲染器 + edge_logprob_mean + score_trajectory + 两个物化函数落地，§3.5 清单全部实现且全绿（含手算对照、r_t 条件正反例、错误归一化被抓的演示、三路梯度、哈希链拒绝）。
2. import 边界测试完成 TYPE_CHECKING 精确化且全仓库通过；scoring 包运行时零模型库 import。
3. ruff/mypy/pytest 全绿，make check 绿。
4. GPU 冒烟完成并留 note（或按 §5 阻塞条款停机说明）。
5. commit + push 到 SKILLEV-new；简短 note 说明本层完成。

## 7. 明确不做

- 不写第 4 层及以上：无 rollout/采样循环、无优化器与训练循环（TTBBatchStats 组装含在内）、无 log I(t) 的窗口/流量/相变使用（式 9 只物化为契约字段）、无后验/进化、无 vLLM。
- 不做批量打分/性能优化；不做 benchmark 数据获取。
- 不改 idea.tex、AGENTS.md、第 1/2 层语义（import 边界测试的 TYPE_CHECKING 精确化除外）；不处理历史 audit 遗留。
- 不写审批/密封仪式；不启动"审计/评审/找缺口"类子代理。

## 8. 停止条件

- §6 验收达成 → 简短总结，停机，等 phase 4 goal。
- 被必须人类介入的事情阻塞（权重、私钥、GPU 服务器）→ 说明缺什么，停机。
- idea.tex 或本文件语义歧义无法双向兼容实现（尤其：若你对式 (4)/(5) 的条件拼接读出了与 §1 不同的含义）→ 列选项与建议后停机请示。
- tiny 底座上手算对照反复对不上（3 次实现尝试后仍有系统性偏差）→ 停机报告现象与已试路径——这大概率是渲染/编码/归一化理解问题，值得人看一眼而不是继续猜。
