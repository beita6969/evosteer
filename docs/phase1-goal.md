# phase1-goal.md —— Bayesian TTB Self-Skill-Revolution Loop：第 1 轮 · 轨迹数据层（contracts）

> **Protocol-v3 immutable amendment (2026-07-25).** 本文件是历史实施记录，不再是当前跨层语义权威；当前权威见 `method-v3-final-spec.md`。终局成功规则是封闭且可重算的 `R_EQUALS_ONE` / `R_AT_THRESHOLD`，不存在 `ENV_DECLARED`。每个 action 在 admission 时只 encode 一次；进化证据与五动作使用封闭的 action-specific union，不再使用 optional mega-record、validator 或 silent dedup。Active scientific source 只有 `LIBRARY_INITIALIZED@3`、`TRAINING_STEP_COMMITTED@3`、`EVOLUTION_PHASE_OPENED@3`、`EVOLUTION_CYCLE_COMMITTED@3`。

本文件是本轮唯一任务清单，取代此前的"转向落地轮（M0–M4）"版 goal——那一版的范围改为**按层分轮**重新发放，本文件是第 1 轮。写法是讲解式的：每节先说做什么，紧跟着解释为什么、以及做错时的典型症状。若某处字面指令与所附理由看起来冲突，按理由描述的意图行事并留一句 note 说明。

## 0. 定位：分层实施策略与全局导航图

FIRE-GFN 课题已废止，方法权威是 `/idea.tex`（禁止修改）。整个算法 codebase 规划为八层，按依赖顺序分轮实施：**(1) 轨迹数据层（contracts，本轮）**；(2) 模型层（底座 + 双 LoRA + Z 头，唯一触碰 torch 的地方）；(3) 打分与目标层（teacher-forced logprob、Δ、L_TTB，纯函数）；(4) rollout 引擎；(5) 训练循环；(6) 流诊断层（log I、F、F̂，纯函数）；(7) Bayesian 校准层（z 特征 + 后验 + LCB/UCB）；(8) 进化层（相变检测 + 五动作算子 Φ）。这张图是导航，不是本轮任务清单——本轮只做第 1 层。

为什么数据层第一、且单独成轮：所有其它层只通过这些类型对话，类型定错一个字段，八层全部返工；其中 **token 边界**是全系统最脆的跨模块契约——第 4 层记录它、第 3 层消费它，边界错一个 token，将来的训练不报任何错、loss 照样下降，只是优化的已经不是 idea.tex 式 (6) 定义的目标，这类"不报错、指标照常、语义已错"的隐蔽错误只能靠**在类型上内建自校验不变量**来防（存入即校验、不一致立即抛错，而不是静默继续）。所以本轮的交付物不只是"几个 dataclass"，而是带牙齿的契约：每个类型自带不变量检查、序列化往返、以及一个演示"错误数据确实进不来"的 fixture。

## 1. 方法权威与翻译关系

本轮契约是把 idea.tex 的概念**翻译成类型**：H_t 与追加规则（式 1–2）→ TrajectoryRecord；R̃=R+ε_min（式 3）→ TerminalReward 与 TrajectoryRecord 的奖励字段；式 4–6 的两个归一化 logprob → EdgeLogprobRecord；式 7–8 的 Δ 与损失 → TTBBatchStats；式 12–14 的 (s,z) 后验 → PosteriorCellState；式 16 → PhaseTransitionEvent；§2.5.2 的五动作 → EvolutionActionRecord。**不可调的**是概念语义（每个量的定义、归一化方式、取值域）；**你的自由**是字段组织、命名、辅助子类型（允许为落账方便增设子记录，如后验更新事件），以及与现有 contracts 基础层（canonical、identity、events、schema_registry）的对接方式。发现 idea.tex 语义歧义无法双向兼容时，列出解读选项与建议后停机请示。

优先级按 AGENTS.md：用户对话指示 > AGENTS.md 与本文件 > 仓库内其它文档。AGENTS.md 与 idea.tex 你都不要动。

## 2. 前置开路（本轮唯一允许的拆除动作）

从 Makefile 摘除两道门并删除对应脚本：`check_protected_files.py`（它 pin 的是 AGENTS.md 与已废止旧方法文件的哈希——AGENTS.md 已被 Owner 替换，这道门不摘 make check 会直接失败）与 `check_training_free_source.py`（AGENTS.md 已宣布废止；本轮契约虽然不含 torch 不会触发它，但留着它等于给后续轮埋雷）。除这两个脚本外，**本轮不删除任何其它旧代码**——完整拆除是后续轮次的事，现在动手只会扩大爆炸半径。摘门后跑一次 make check 确认绿。

## 3. 七个契约，逐个用人话讲

每个契约按四段讲：它是什么（人话）、为什么需要它、关键字段、自校验不变量与典型错误症状。

**共享底型与通用约定（补充；供 §3.1–3.7 的代码架构引用）**：下面各节的代码骨架是 Owner 定稿的默认设计——**字段语义与不变量不可减**，命名与组织可在一句话理由内微调（§1 的"字段组织自由"在不减语义的前提下仍成立）。骨架省略 import 与函数体（以 `...` 代），docstring 与注释就是实现规格：注释说"需要写"的，就是该函数/校验要实现的内容。

```python
# ===== 共享底型（随契约文件一起交付；被 §3.1–3.7 引用） =====
# 依赖说明：Protocol/Mapping 来自 typing，StrEnum 来自 enum，
# JsonValue/canonical_json/stable_hash 引用现有 contracts 基础层——不自造。

class TokenizerProtocol(Protocol):
    """契约层唯一的 tokenizer 依赖面（§4 零重依赖要求的落地）。
    实现不 import transformers：生产侧给钉死的 Qwen tokenizer 写一个薄包装类，
    测试侧写一个确定性假 tokenizer（如按字符切分）。encode 必须确定性、无随机、
    不加特殊 token——它的唯一用途是重 tokenize 校验。"""

    @property
    def tokenizer_id(self) -> str:
        """需要写：稳定标识+版本，如 "qwen3.5-9b-tokenizer@<内容哈希>"。
        它被写进每条 TrajectoryRecord；校验时先比 id 再比序列，
        防"用 A 记录、用 B 校验"。"""
        ...

    def encode(self, text: str) -> list[int]:
        """需要写：纯编码，文本 → token id 列表；逐次调用结果完全一致。"""
        ...


# 命名警告：idea.tex 里 β 出现两次——式 (7) 的温度 β 与式 (12) 的 Beta 后验失败计数。
# 契约里分别命名 temperature_beta（§3.3）与 beta_count（§3.4），禁止两个都叫 beta。

# 通用校验约定（所有 __post_init__ / validate_* / build_* 遵守）：
# - 校验失败一律 raise ValueError，消息带定位（trajectory_id、step_index、字段名），
#   禁止返回 bool 静默；
# - 浮点一律 math.isfinite 检查；相等比较用显式容差（如 1e-9），容差常量集中定义一处；
# - 全部类型 frozen dataclass + slots，风格对齐仓库现有 contracts 包；
# - 身份哈希用现有 stable_hash/canonical_json；每类记录在 events 的 EventType 枚举里
#   注册对应事件类型并可包进 EventEnvelope，直接写入 runtime 的 AppendOnlyEventLog；
# - created_at 时间戳只作审计信息，不参与内容哈希（沿仓库惯例）。
```

### 3.1 TrajectoryRecord —— 一次任务执行的"录像带"

**是什么**：agent 从拿到任务到停下来的全过程存档。任务是什么（q）、开局上下文怎么拼的（H_0 = q ⊕ 检索到的技能 S_ret ⊕ 元信息 ω_q，各成分分别记录）、每一步的三件事——想了什么（r_t，推理文本）、做了什么（a_t，技能/工具调用）、环境回了什么（o_t^exec）——以及每步动作在 token 世界里的精确身份：a_t 的 **token id 序列本身**（将来 teacher-forced 打分喂的就是它）与 K_t = 该序列长度。最后是成绩单：终局奖励 R∈[0,1] 与平移后的 R̃ = R + ε_min，外加环境 id、任务族、解码快照引用（温度等采样参数的版本号）。

**为什么**：将来训练要"重放"这盘录像带逐步打分。打分要求精确知道 a_t 从哪个 token 开始、到哪个 token 结束、prefix 是怎么组装的——所以录像带必须记到 token 级，且必须记录**上下文组装函数的版本号与组装结果哈希**（防止将来 prompt 模板悄悄变了，重放时 prefix 和当年不一致，打分全错还查不出来）。

**不变量**：(a) **重 tokenize 校验**——用钉死的 tokenizer 把存入的 a_t 文本重新 tokenize，必须与记录的 token id 序列和 K_t 逐位一致，不一致当场抛错，这是全契约层最重要的一道牙齿；(b) r/a/o 三段无重叠、无遗漏、步序单调；(c) R∈[0,1]、R̃ = R + ε_min 数值自洽；(d) 每步的 o_t^exec 是 a_t 自己的执行反馈（顺序对齐检查）。**典型症状**：没有不变量 (a) 时，边界错位的轨迹会安静地流进训练，loss 曲线正常，几周后才在"R 不涨"里发现——那时已无法定位是数据错还是方法错。

**代码架构（补充）**：

```python
@dataclass(frozen=True, slots=True)
class InitialContext:
    """H_0 = q ⊕ S_ret ⊕ ω_q（式 2）。三成分分开存；拼接由将来第 4 层的组装函数做，
    这里只存成分与组装结果的哈希，供重放时重新组装并比对。"""

    query: str
    # 任务原文 q。需要写：非空校验。

    retrieved_skill_ids: tuple[str, ...]
    # S_ret：注入的技能 manifest id 列表（可为空元组）。只存 id 不复制技能文本——
    # 技能内容以 runtime/skills.py 的不可变 manifest 为唯一真相，复制会造成两处真相漂移。

    meta: Mapping[str, JsonValue]
    # ω_q：环境 id、可用工具声明、benchmark 元信息等。需要写：可 canonical 序列化校验。

    assembler_version: str
    # 上下文组装函数（prompt 模板）的版本号。模板一变这个号必须变；非空校验。

    assembled_hash: str
    # 组装出的完整 H_0 文本的 stable_hash。重放时用同版本组装函数重拼并比对——
    # 不一致 = 模板漂移，立即报错。这是"打分全错还查不出来"的唯一防线。

    assembled_token_count: int
    # 组装后 H_0 的 token 数（>0 校验）。预算记录与将来 Token Bucket 的数据源之一。


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    """一条边，即式 (1) 追加的 (r_t, a_t, o_t^exec) 三元组 + 该步的 token 级身份。"""

    index: int
    # 步号 t，从 1 起；全序列连续递增由 TrajectoryRecord 校验。

    reasoning_text: str
    # r_t 推理原文。只存档不打分（式 4 中 r_t 是条件不是结果）；允许空串（模型可以不出推理）。

    action_text: str
    # a_t 原文（结构化调用的字符串形态）。非空校验。

    action_token_ids: tuple[int, ...]
    # a_t 在钉死 tokenizer 下的 token id 序列——将来 teacher-forced 打分逐位喂的就是它。非空。

    action_token_count: int
    # K_t（式 6 的归一化分母）。不变量：== len(action_token_ids)，且经重 tokenize 三连查
    # （见 build_trajectory_record）。

    observation_text: str
    # o_t^exec：a_t 自己的执行反馈原文。非空（错误信息也是反馈）。

    observation_status: str
    # 执行状态粗分类，取值限定在允许集合 {"success","tool_error","schema_invalid",
    # "timeout","parse_error","other"}——第 7 层 FailureMode（§3.4）的原料。校验取值合法。

    invoked_skill_ids: tuple[str, ...]
    # 本步调用的技能 id（F̂(s) 聚合与后验更新的定位键）。可为空元组（非技能动作）。

    forward_prefix_hash: str
    # 前向打分前缀（H_{t-1} ⊕ r_t）组装文本的 stable_hash——含 r_t，式 (4) 的字面。

    hindsight_prefix_hash: str
    # 后向打分前缀（H_{t-1} ⊕ o_t^exec）的 stable_hash——不含 r_t，式 (5) 的字面。
    # 两个哈希的用途：第 3 层打分前重组前缀并比对，杜绝"前缀拼错但照样出数"的隐蔽错误。


@dataclass(frozen=True, slots=True)
class TrajectoryRecord:
    """录像带本体。"""

    trajectory_id: str            # 全局唯一 id，沿仓库 identity 惯例生成
    environment_id: str           # 产生本轨迹的环境/适配器 id；非空
    task_family: str              # 任务族（将来 z 的 Context 维原料）；非空
    initial_context: InitialContext
    steps: tuple[TrajectoryStep, ...]   # ≥1 步
    horizon: int                  # T。不变量：== len(steps)（÷T 的分母，冗余存以便单独校验）
    reward: TerminalReward        # §3.7 的裸输出（未平移）
    shifted_reward: float         # R̃ = reward.value + epsilon_min（式 3）。全系统唯一做平移的地方
    epsilon_min: float            # 本轨迹所用 ε_min（>0），记录下来使 R̃ 永远可复核
    tokenizer_id: str             # 记录时所用 tokenizer 标识；build 时与传入 tokenizer 比对
    decoding_snapshot_id: str     # rollout 采样参数快照引用（温度等）；非空
    created_at: str               # ISO 时间戳；只作审计，不参与内容哈希

    def __post_init__(self) -> None:
        """自足不变量（不需要 tokenizer 的部分）。需要写：
        steps 非空且 index 从 1 连续递增；horizon == len(steps)；
        0 <= reward.value <= 1（TerminalReward 自身也查，这里查引用一致性）；
        epsilon_min > 0 且 |shifted_reward − (reward.value + epsilon_min)| <= 容差；
        每步 observation_status 在允许集合内。"""
        ...


def build_trajectory_record(*, tokenizer: TokenizerProtocol, **fields) -> TrajectoryRecord:
    """工厂函数：唯一合法的构造入口。裸构造 TrajectoryRecord 不做 token 校验——
    这一事实必须写进 TrajectoryRecord 的 docstring。需要写：
    1. tokenizer.tokenizer_id 与 fields 里的 tokenizer_id 一致，否则拒绝；
    2. 对每一步做重 tokenize 三连查：
       tuple(tokenizer.encode(step.action_text)) == step.action_token_ids、
       step.action_token_count == len(step.action_token_ids)、
       重编码长度与 K_t 一致——偏一位/缺一个/多一个都要被测试逐个演示拒绝；
    3. 通过后构造返回（__post_init__ 自动跑自足校验）。
    错误消息带 trajectory_id、step index、期望/实际序列，方便定位。"""
    ...
```

### 3.2 EdgeLogprobRecord —— 某一步动作的"两份评分"存档

**是什么**：轨迹里某一步 a_t 被两个策略各打了一个分：前向策略（当时的决策者）认为这个动作有多可能（P̂_F^t），后向策略（事后诸葛亮，看到执行结果 o_t^exec 之后）认为这个动作有多合理（P̂_B^t），以及两者之差 log I(t) = P̂_F^t − P̂_B^t（步重要性——正得越多，说明前向投入的概率质量超出事后解释越多，这步越"关键"）。**本轮只定义这个记录长什么样，不实现打分**——打分是第 2/3 层的活。

**为什么**：这三个数是流诊断（第 6 层）和 TTB 目标（第 3 层）的原料，且同一步可能在不同训练时刻被重打分（adapter 更新了），所以每条记录必须能回答"这是谁、在什么版本下打的分"。

**关键字段**：指向哪条轨迹哪一步（TrajectoryRecord id + 步号）、两个 logprob（**单位钉死：自然对数、按 K_t 归一化后的每 token 均值**——单位不钉死，将来两处代码一个用总和一个用均值，log I 就是错的）、打分时的 adapter/快照版本。

**不变量**：log I 与两个 logprob 自洽（差值检查）；三个数全部有限（NaN/Inf 拒收——NaN 进日志的后果是下游聚合悄悄传染，整批诊断作废）。

**代码架构（补充）**：

```python
@dataclass(frozen=True, slots=True)
class EdgeLogprobRecord:
    """某一步的两份评分 + 步重要性。本轮只定型，不实现打分（打分是第 2/3 层的活）。"""

    trajectory_id: str            # 指向 TrajectoryRecord；非空
    step_index: int               # 指向 TrajectoryStep.index；>= 1

    forward_logprob_per_token: float
    # P̂_F^t（式 6 前向）。单位钉死：自然对数（nats）、对 a_t 全部 K_t 个 token 的
    # logprob 求和后 ÷ K_t 的"每 token 均值"。这段注释必须原样出现在代码里——
    # 它是防单位漂移（一处用总和、一处用均值）的唯一文档点。

    backward_logprob_per_token: float
    # P̂_B^t（式 6 后向）：同一 a_t 序列、同一 K_t、hindsight 前缀（不含 r_t）。单位同上。

    step_importance: float
    # log I(t) = forward − backward（式 9）。冗余存储，校验自洽。

    forward_adapter_version: str
    backward_adapter_version: str
    # 打分时 π_θ / P_φ 的 adapter 版本（优化器步数或 checkpoint id）。同一步会在不同
    # 训练时刻被重打分，没有版本字段就无法区分两条记录谁新谁旧。非空校验。

    scoring_stack_id: str
    # 打分发生在哪个栈。本轮起固定语义："training-stack"；它是两栈阶段铁规则 (a) 的
    # 审计字段——vLLM 永远不该出现在这里，出现即违规，将来审计一 grep 便知。非空校验。

    def __post_init__(self) -> None:
        """需要写：三个浮点 math.isfinite（NaN/Inf 拒收）；
        |step_importance − (forward − backward)| <= 1e-9；
        step_index >= 1；trajectory_id 与版本/栈字段非空。"""
        ...
```

### 3.3 TTBBatchStats —— 每个训练批次的"体检单"

**是什么**：一批轨迹训练一步后留下的核心健康指标：每条轨迹的残差 Δ(τ)、批均损失 L_TTB、当前 log Z_θ(q) 的取值、批平均奖励 R̄，以及成员轨迹 id 列表。

**为什么**：三个下游都靠它——损失/奖励曲线要从日志可重建（画图与审计不靠训练进程的临时变量）；"loss 降但 R 不升"这类诊断要靠批级数据事后分析；第 8 层的相变检测器将来吃的就是这个流（滑窗 Δ̄² 的原料）。没有它，训练健康状况只存在于 stdout 里，进程一退出就消失。

**不变量**：成员数 >0；统计值与成员列表对应（重放成员轨迹能重算出同样的 Δ̄²）；数值有限。

**代码架构（补充）**：

```python
@dataclass(frozen=True, slots=True)
class TrajectoryResidual:
    """批内单条轨迹的残差四项分解（式 7）。存分解而非只存 Δ 的原因："loss 降但 R 不升"
    这类事后诊断需要看是哪一项在动（Z 在单方面拟合，还是前后向差在缩）。"""

    trajectory_id: str
    log_z: float                  # log Z_θ(q) 对该轨迹 q 的当时取值
    sum_forward: float            # Σ_t P̂_F^t（每 token 均值之和——单位与 §3.2 一致，注释重申）
    sum_backward: float           # Σ_t P̂_B^t（同上）
    log_shifted_reward: float     # log R̃(τ)
    raw_reward: float             # 裸 R∈[0,1] 的副本——供批级 mean_reward 重算与 R 曲线绘制
    temperature_beta: float       # 式 (7) 的温度 β（>0）。命名避开后验的 beta_count（见共享底型警告）
    delta: float                  # Δ(τ)。__post_init__ 按式 (7) 用上面五项重算并容差比对
    horizon: int                  # T >= 1。(delta/horizon)² 即该轨迹对批损失的贡献

    def __post_init__(self) -> None:
        """需要写：全部浮点 isfinite；0<=raw_reward<=1；horizon>=1；temperature_beta>0；
        |delta − (log_z + sum_forward − temperature_beta*log_shifted_reward − sum_backward)| <= 容差。"""
        ...


@dataclass(frozen=True, slots=True)
class TTBBatchStats:
    """一个训练批的体检单。"""

    batch_id: str
    optimizer_step: int           # 优化器步号（>=0；跨批单调性由写入方保证，这里只查非负）
    library_version: str
    # 当时技能库版本。关键字段：相变检测的滑窗必须按库版本分段——Φ 之后 Z 头重置、
    # 损失面变了，跨版本混窗会把重置抖动误判成平台期破裂；曲线绘制也按它分段对齐。非空。

    residuals: tuple[TrajectoryResidual, ...]   # 批成员；非空、trajectory_id 无重复
    batch_loss: float             # L_TTB = mean((Δ/T)²)。__post_init__ 由 residuals 重算比对
    mean_reward: float            # 批平均裸 R（钉死用 raw_reward，不用 R̃——画 R 曲线的统一口径）
    created_at: str

    def __post_init__(self) -> None:
        """需要写：residuals 非空且 id 去重；batch_loss 与 mean_reward 分别由成员重算、
        容差比对（"体检单可由病历重算"就是'统计与成员对应'不变量的落地）；全部数值 isfinite。"""
        ...
```

### 3.4 PosteriorCellState —— 技能可靠性记分牌上的一个格子

**是什么**：回答"技能 s 在情境 z 下靠不靠谱"的一个格子。情境 z 由四个坐标定位：哪类任务（Context）、什么失败模式（Failure Mode）、耗了多少 token（Token Bucket）、任务多长（Horizon Bucket）。格子里存两个加权计数：α（成功证据）与 β（失败证据），由它们可推出平均成功率 μ、不确定度 σ、保守下界 LCB = μ−kσ、乐观上界 UCB = μ+kσ。

**为什么存计数不存结论**：α/β 是完备统计量——μ/σ/LCB/UCB 全能从它们现算，存结论反而会在 k 变化时留下过期数字。且事件溯源要求每次更新（哪次调用、成功与否 y_e、流权重 w_e 多大）单独落账（允许增设 PosteriorUpdateEvent 子记录），格子状态永远可以从更新流重放重建——这是第 7 层"在线更新且可审计"两个性质同时成立的基础，也是将来"换个 k 重算全部 LCB"零成本的原因。

**不变量**：α≥α_0、β≥β_0 且单调不减（后验计数只增不减）；每次更新的增量等于所记录的 w_e；w_e≥0。**典型症状**：不做单调检查时，一个重放 bug 把格子重建少了几条更新，LCB 悄悄偏高，进化层因此保留了一个其实不可靠的技能——错误一路传到论文数字。

**代码架构（补充）**：

```python
class FailureMode(StrEnum):
    """z 第二维：失败模式粗分类（≤6 类——粗颗粒防后验稀疏）。
    需要写：SUCCESS / TOOL_ERROR / SCHEMA_INVALID / TIMEOUT / PARSE_ERROR / OTHER 六个成员，
    以及 classmethod from_observation_status(status: str) -> "FailureMode"——
    把 §3.1 TrajectoryStep.observation_status 映射过来；未知字符串一律归 OTHER
    （映射必须全射、不许抛错：分类器有缺口不该让落账失败）。"""


class TokenBucket(StrEnum):
    """z 第三维。需要写：LE_1K / K1_TO_4K / GT_4K 三档 +
    classmethod from_count(n: int)，边界钉死 ≤1000、1001–4000、>4000。
    注释必须写明后果：改边界 = 改 z 语义 = 已积累的全部后验格子作废重攒。"""


class HorizonBucket(StrEnum):
    """z 第四维。LE_3 / H4_TO_8 / GT_8 + classmethod from_horizon(t: int)，
    边界 ≤3、4–8、>8。同样的改边界警告。"""


@dataclass(frozen=True, slots=True)
class ContextFeature:
    """情境 z 本体（idea.tex：z = [Context, Failure Mode, Token Bucket, Horizon Bucket]）。"""

    context: str                  # 第一维：任务族 id（如 "synthetic/tool_chain"）；非空
    failure_mode: FailureMode
    token_bucket: TokenBucket
    horizon_bucket: HorizonBucket

    def cell_key(self, skill_id: str) -> str:
        """需要写：与 skill_id 一起生成 (s,z) 格子的规范索引键——
        用 canonical_json + stable_hash，不要手拼字符串（分隔符冲突会让两个不同格子撞键）。"""
        ...


@dataclass(frozen=True, slots=True)
class PosteriorUpdateEvent:
    """一次共轭更新的凭证（式 12）——事件溯源的最小单元；格子状态由事件流重放重建。"""

    event_id: str
    skill_id: str                 # s；非空
    z: ContextFeature
    trajectory_id: str
    step_index: int               # 证据来自哪条轨迹哪一步（可回查 §3.1/§3.2）；>=1
    outcome: bool                 # y_e：该次调用的执行级成功指示
    flow_weight: float            # w_e >= 0 且 isfinite（具体归一化公式是第 7 层的事，这里只存结果值）
    alpha_after: float
    beta_count_after: float
    # 更新后的两个计数。链式不变量——等于"该格子上一事件的计数 + 本次增量"，
    # 增量 = outcome 为真时 (w_e, 0)、为假时 (0, w_e)；链式校验由重放器（第 7 层）做，
    # 本类型只查单事件自足条件（数值范围与有限性）。

    def __post_init__(self) -> None:
        """需要写：flow_weight >= 0；alpha_after/beta_count_after isfinite 且 >= 先验下限；
        step_index >= 1；各 id 非空。"""
        ...


@dataclass(frozen=True, slots=True)
class PosteriorCellState:
    """格子当前状态。可随时从 PosteriorUpdateEvent 流重建；两者不一致 = 重放 bug，
    以事件流为准（状态是缓存，事件是真相）。"""

    skill_id: str
    z: ContextFeature
    alpha: float                  # >= alpha_0
    beta_count: float             # >= beta_0；命名避开温度 β（见共享底型命名警告）
    alpha_0: float                # 先验，默认 1.0；> 0
    beta_0: float                 # 先验，默认 1.0；> 0
    update_count: int             # >= 0
    last_event_id: str | None     # 最近一次更新事件引用；None 表示纯先验格子（此时 update_count==0）

    def mean(self) -> float:
        """式 (13)：α/(α+β)。现算不缓存。"""
        ...

    def variance(self) -> float:
        """式 (13)：αβ/((α+β)²(α+β+1))。现算不缓存。"""
        ...

    def lcb(self, k: float) -> float:
        """式 (14)：mean − k·sqrt(variance)。k 是调用方参数，不存进状态——
        换 k 重算全库 LCB 零成本，这就是"存计数不存结论"的红利。"""
        ...

    def ucb(self, k: float) -> float:
        """mean + k·sqrt(variance)。Prune 的法定判据用它（§3.6）。"""
        ...

    def apply(self, event: PosteriorUpdateEvent) -> "PosteriorCellState":
        """纯函数式更新，返回新实例、不改自身。需要写：
        event 的 skill_id 与 z 必须与本格子一致，否则拒绝；
        新 α/β = 旧值 + 按 outcome 分配的 flow_weight；
        与 event.alpha_after/beta_count_after 容差比对，不一致抛错——
        这是"每次更新的增量等于所记录的 w_e"不变量的落地；
        update_count + 1、last_event_id 换成本事件。"""
        ...

    def __post_init__(self) -> None:
        """需要写：alpha >= alpha_0 > 0、beta_count >= beta_0 > 0、isfinite、
        update_count >= 0、update_count==0 ⟺ last_event_id is None（且此时 α/β 等于先验）。"""
        ...
```

### 3.5 PhaseTransitionEvent —— 宣布"该进化了"的那一声哨（连同证据）

**是什么**：训练进入平台期、允许技能库动手术的判定记录。不只记"触发了"，而是记**触发时刻的完整证据**：两个滑窗各自的 Δ̄²、相对改进值与阈值 ρ 的比较、技能使用熵的近窗序列与"连续下降"判定、触发时的库版本号。

**为什么**：idea.tex 式 (16) 用 ∧ 双条件防"把优化噪声当饱和"，论文与审计都要能**独立复核**两个条件当时确实同时成立——只存布尔值的话，"进化是不是随手触发的"就成了无法回答的问题。它也是每个 EvolutionActionRecord 的授权源头（动作要引用它）。

**不变量**：证据数值与判定结论自洽（用存的 Δ̄² 和 ρ 重算，结论必须一致）；库版本号真实存在。

**代码架构（补充）**：

```python
@dataclass(frozen=True, slots=True)
class WindowStats:
    """一个滑窗的残差证据（式 16 左条件的原料）。"""

    start_optimizer_step: int
    end_optimizer_step: int       # 闭区间端点；end >= start
    batch_count: int              # 窗内批数（== 窗宽 W；与端点跨度一致性由写入方保证，这里查 > 0）
    mean_squared_residual: float  # Δ̄²：窗内全部 (Δ/T)² 的均值；isfinite 且 >= 0
    member_batch_ids: tuple[str, ...]
    # 复核入口：审计者可拉出这些 TTBBatchStats 重算 Δ̄²。非空，数量 == batch_count。

    def __post_init__(self) -> None:
        """需要写：区间合法、batch_count == len(member_batch_ids) > 0、数值检查。"""
        ...


@dataclass(frozen=True, slots=True)
class EntropyObservation:
    """一个窗的技能使用熵观测（式 16 右条件 H(S)↓ 的操作化：
    近窗技能调用频率分布的香农熵）。"""

    window_end_step: int          # 该窗右端点（观测序列按它排序）
    entropy: float                # >= 0 且 isfinite
    total_invocations: int        # >= 0
    distinct_skills: int          # >= 0
    # 窗口已满但零技能调用必须精确表示为 entropy=0、total=0、distinct=0；
    # 它与“窗口尚未满、没有观测”是不同事实。


@dataclass(frozen=True, slots=True)
class PhaseTransitionEvent:
    """只在真触发时落账。设计决定：未触发不写事件——每窗都写会淹没日志；
    未触发时段的复核靠 TTBBatchStats 流重算，不靠事件。"""

    event_id: str
    triggered_at_step: int        # 触发时的优化器步号
    library_version: str          # 触发时库版本；本事件授权的手术必须发生在这个版本上。非空
    previous_window: WindowStats
    current_window: WindowStats
    relative_improvement: float   # (prev.msr − cur.msr) / prev.msr；__post_init__ 重算比对
    rho: float                    # 当时阈值（> 0）
    residual_condition_met: bool  # 必须 == (relative_improvement < rho)；重算比对
    entropy_series: tuple[EntropyObservation, ...]
    # 近几窗的熵观测，按 window_end_step 严格递增；长度 >= required_consecutive_drops + 1
    # （判定"连续下降 n 次"至少要 n+1 个观测点）。
    required_consecutive_drops: int   # 连续下降窗数要求（默认 2；>= 1）
    entropy_condition_met: bool   # 必须 == 序列尾部确有 required 次连续严格下降；重算比对
    triggered: bool               # 必须 == residual_condition_met AND entropy_condition_met——∧ 的字面落地

    def __post_init__(self) -> None:
        """需要写：两窗相邻不重叠且 previous 在前（prev.end < cur.start 或按写入方约定的紧邻关系）；
        prev.mean_squared_residual > 0（除数）；relative_improvement 重算比对；
        三个布尔全部由证据字段重算并比对，任何一个对不上就抛错；
        triggered 必须为 True（本类型只记录真触发）；entropy_series 排序与长度检查；数值 isfinite。"""
        ...
```

### 3.6 EvolutionActionRecord —— 技能库每次"动手术"的病历

**是什么**：五种手术（Retain&Compress 保留压缩 / Refine 修补 / Split 拆分 / Prune 下架 / Generate 新建）里做了哪种、对哪个技能、**依据是什么**（当时的 F̂(s)、LCB/UCB、log I 证据快照）、产生或修改后的技能新版本（指向 runtime/skills.py 的 lineage 记录）、由哪次 PhaseTransitionEvent 授权。

**为什么**：idea.tex 要求进化动作 "distinct, auditable"——这个词的落地形态就是：任何一次库变更都能回答四个问题：谁批准的（相变事件引用）、证据是什么（指标快照）、改了什么（lineage 前后版本）、怎么回滚（lineage 机制）。没有这份病历，技能库半年后就是一堆来历不明的文本。

**不变量**：动作类型与证据形态匹配——特别地，**Prune 必须带 UCB 证据**（idea.tex 用乐观上界判死刑：证据不足的技能 σ 大、UCB 高，不会被误删，只有"证据充分地差"才下架），Generate 必须带"无技能覆盖 + 高 |log I|"证据；lineage 与相变事件引用必须真实存在。

**代码架构（补充）**：

```python
class EvolutionActionType(StrEnum):
    """五种手术。需要写：RETAIN_COMPRESS / REFINE / SPLIT / PRUNE / GENERATE
    五个成员，一个不多一个不少（idea.tex §2.5.2 的五动作是封闭集合）。"""


@dataclass(frozen=True, slots=True)
class SkillFlowEvidence:
    """v2 动作证据。F̂ 使用单位明确的 log-domain 表示；校准模式显式持久化。"""

    calibration_enabled: bool
    posterior_decision_statistic: PosteriorDecisionStatistic
    log_skill_marginal_flow: float | None
    flow_quantile: float | None
    lcb: float | None
    ucb: float | None
    k: float | None
    posterior_mean: float | None
    posterior_event_ids: tuple[str, ...]
    importance_edge_ids: tuple[str, ...]
    split_modality: SplitModalityEvidence | None
    format: str = "skillev-evolution@2"

    def __post_init__(self) -> None:
        """校准关闭时 statistic=NONE，且 lcb/ucb/k/posterior_mean/
        split_modality/posterior_event_ids 全部为空；校准开启时 statistic 决定
        使用 bounds 还是 mean，二者不能混放。引用唯一，数值有限。"""
        ...


@dataclass(frozen=True, slots=True)
class EvolutionActionRecord:
    """病历本体。"""

    action_id: str
    action_type: EvolutionActionType
    phase_event_id: str
    # 授权来源：必须指向一条已落账的 PhaseTransitionEvent。引用真实性由落账侧核对，
    # 本类型查非空。

    library_version_before: str
    library_version_after: str
    # 不变量：after != before。五种动作都产生新库版本——包括 RETAIN_COMPRESS：
    # 压缩改写=新内容=新 manifest 哈希=新版本。

    target_skill_ids: tuple[str, ...]
    # 动作对象：RETAIN_COMPRESS/REFINE/PRUNE/SPLIT 恰 1 个；GENERATE 恰 0 个。

    produced_skill_ids: tuple[str, ...]
    # 产物：SPLIT 恰 2（两个子技能）；GENERATE 恰 1；RETAIN_COMPRESS/REFINE 恰 1（新版本技能）；
    # PRUNE 恰 0。

    lineage_ref: str
    # runtime/skills.py 的 lineage 记录 id——"改了什么、怎么回滚"由 lineage 机制承载，
    # 病历只持引用。非空。

    evidence: SkillFlowEvidence
    rationale_text: str
    # 一句人话理由（进 audit 展示）。泄漏红线：本字段与产出技能文本禁止含任何任务真值/
    # 逐题答案/verifier 私有信息——契约层无法机器判定语义泄漏，此处 docstring 挂红线，
    # 执行责任在第 8 层写入前的检查。非空校验。

    def __post_init__(self) -> None:
        """需要写：调用 validate_evolution_action(self)；id 类字段非空；after != before。"""
        ...


def validate_evolution_action(record: EvolutionActionRecord) -> None:
    """v2 closed action matrix：
    - enabled Retain/Refine/Prune 使用配置选择的 confidence bound 或 posterior mean；
      Refine 另需完整 posterior event chain；
    - Split 只允许 enabled，且必须携带完整 SplitModalityEvidence 与 posterior chain；
    - disabled 只允许 flow-only Retain 与 Generate，任何后验证据都非法；
    - Generate 在两种模式都需非空 importance_edge_ids；
    - 对象/产物比例保持 Retain/Refine 1→1、Split 1→2、Prune 1→0、Generate 0→1。
    """
    ...
```

### 3.7 TerminalReward —— 环境给轨迹打分的统一"插座"

**是什么**：不管环境是合成的、P1 系统辨识、social DGP，还是将来 14 个外部 benchmark，交出来的都是同一个三件套：归一化分数 R∈[0,1]（训练信号）、成功与否的布尔（报表用）、原生指标原文 payload（论文用——F1、Resolved Rate、Step Accuracy 各环境自报）。评分一律在环境/verifier 侧算好再进来。

**为什么现在就定死形态**：接新环境 = 写一个新适配器，训练循环零改动——这是对 `docs/benchmark-evaluation-guide.md` §2 的前向兼容（该文档定义了 14-benchmark 评测层，属后续轮次，本轮只需插座形状一致）。评分在 verifier 侧的原因是铁律 4：gold 答案、单元测试、评分逻辑永不出现在被训模型的上下文里。

**不变量**：R∈[0,1]；success 与 R 的关系由各环境显式声明（例如"R==1 ⟺ success"）并校验；payload 只进日志、有类型但不设语义约束（各 benchmark 自定义）；R̃ 的平移在 TrajectoryRecord 侧统一做，适配器只出裸 R（防止 ε_min 被加两次）。

**代码架构（补充）**：

```python
class SuccessRule(StrEnum):
    """success 与 R 关系的显式声明——三选一，不允许缺省。需要写三个成员：
    R_EQUALS_ONE:   success ⟺ R == 1（WebShop/ALFWorld 型：满分即成功）；
    R_AT_THRESHOLD: success ⟺ R >= success_threshold（F1 阈值型）；
    ENV_DECLARED:   环境独立判定（如 pass@1 与分数分离的场合）——此时 native_payload
                    里必须留有判定依据的说明字段。"""


@dataclass(frozen=True, slots=True)
class TerminalReward:
    """统一插座本体（docs/benchmark-evaluation-guide.md §2 同款形态）。"""

    value: float
    # 裸 R ∈ [0,1]，未加 ε_min。平移是 TrajectoryRecord 的事——这里出现平移 = ε_min 被加两次。

    success: bool
    success_rule: SuccessRule
    success_threshold: float | None
    # 仅 success_rule == R_AT_THRESHOLD 时必须非 None 且 ∈ (0,1]；其余规则下必须为 None
    # （多给少给都拒绝——阈值字段与规则不匹配是适配器写错的高发信号）。

    native_metric_name: str
    # 原生指标名："binary" / "f1" / "score_over_100" / "resolved_rate" / "step_accuracy" ...
    # 非空；论文报表按它分组。

    native_payload: Mapping[str, JsonValue]
    # 原生指标原文（各环境自定义结构，不设语义约束，但必须可 canonical 序列化）。
    # 红线：payload 只进日志，永不进被训模型的上下文（铁律 4）——契约层管不住使用侧，
    # 此处 docstring 写明，执行责任在第 4 层 rollout 引擎。

    environment_id: str
    verifier_version: str
    # 评分器版本。同一环境改评分逻辑必须换版本号，否则跨期数字不可比。两者非空。

    def __post_init__(self) -> None:
        """需要写：0 <= value <= 1 且 isfinite；
        按 success_rule 重算 success 并比对——
          R_EQUALS_ONE:   success == (|value − 1| <= 容差)；
          R_AT_THRESHOLD: success == (value >= success_threshold)；
          ENV_DECLARED:   不重算，但校验 native_payload 非空（判定依据必须留痕）；
        success_threshold 的存在性与 success_rule 匹配（见字段注释）；
        native_metric_name / environment_id / verifier_version 非空；
        native_payload 可 canonical 序列化（调 canonical_json 走一遍即校验）。"""
        ...
```

## 4. 工程要求

- **零逻辑、零重依赖**：契约层是纯数据 + 校验，**禁止 import torch/transformers**。重 tokenize 校验需要 tokenizer，但契约不直接依赖任何 tokenizer 实现——定义一个极小的 Tokenizer Protocol（`encode(text)->list[int]` 级别），校验函数接收它作参数；生产时传钉死的 Qwen tokenizer，测试时传一个简单的本地假 tokenizer（无网络、无权重下载）。契约里记录 tokenizer 的标识/哈希字段，防"用 A 记录、用 B 校验"。
- **沿用现有基础层**：canonical_json/stable_hash 做身份、events 的 envelope 机制对接 event_log、schema_registry 注册新 schema——不要自造序列化或哈希。风格对齐仓库现状（frozen dataclass、slots、显式校验抛 ValueError）。
- **测试三层**：(a) 不变量单元测试——每条不变量至少一个"好数据通过 + 坏数据被拒"的成对用例，特别是 token 边界错位（偏一位、缺一个、多一个）必须逐个演示被抛错；(b) 序列化往返——构造→canonical_json→读回→逐字段相等、哈希稳定；(c) 一个端到端 fixture——手工构造一条两步假轨迹（假 tokenizer），走完七类记录的完整生命周期（构造→落账→重放读取→重建后验格子），作为后续层的活文档。测试断言行为，不断言精确错误字符串或哈希字面量。
- 单文件超约 1000 行拆分；契约文件建议按记录族分文件（如 contracts/ttb_trajectory.py、ttb_training.py、ttb_calibration.py、ttb_evolution.py，命名自由）。

## 5. 验收标准

1. 七个契约类型全部落地，§3 的每条不变量都有对应的"好/坏"成对测试且全绿。
2. §4 的端到端 fixture 跑通，并演示：token 边界错位的轨迹在构造时即被拒绝（不是在下游才发现）。
3. 契约层 import 图里没有 torch/transformers；ruff/mypy/pytest 全绿；摘门后的 make check 绿。
4. schema_registry 含全部新 schema；canonical 序列化往返与哈希稳定性测试通过。
5. commit + push 到 SKILLEV-new；简短 note（commit message 即可）说明契约层完成与摘门操作。

## 6. 明确不做（本轮范围刀切在这里）

- **不写任何算法逻辑**：不建模型层、不写打分/rollout/训练/诊断/校准/进化的任何实现——哪怕一个"顺手先写了"的函数。契约层的价值恰恰在于它先于一切实现被冻结；实现混进来，契约就会被实现的便利反向扭曲。
- **不做完整拆除**：除 §2 的两道门外不删任何旧代码。旧 evolution 包等的拆除是后续轮次任务。
- 不做 benchmark 数据获取与接入（评测层规划在 `docs/benchmark-evaluation-guide.md`，仅其 §2 的 TerminalReward 形态在本轮生效）。
- 不动 idea.tex、AGENTS.md、`docs/benchmark-evaluation-guide.md`。
- 不写审批/密封/attestation 仪式；不启动"审计/评审/找缺口"类子代理（其他子代理随意）。

## 7. 停止条件

- §5 验收达成 → 写简短总结，停机，等 Owner 发第 2 轮 goal。
- idea.tex 语义歧义无法双向兼容地类型化 → 列出解读选项与你的建议后停机请示。
- 被必须人类介入的事情阻塞 → 说明缺什么，停机；不用新代码填充等待时间。
