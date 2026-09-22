# phase4-goal.md —— Bayesian TTB Self-Skill-Revolution Loop：第 4 轮 · rollout 引擎（on-policy trajectory collection）

> **Protocol-v3 immutable amendment (2026-07-26).** 本文件是历史实施记录；当前语义见 `method-v3-final-spec.md`。Production H0 只接受 `FullRetrievedSkillContext`，必须包含完整不可变技能 instructions；不存在 metadata-only placeholder、渐进披露 fallback 或第二技能来源。每条 rollout 从 fresh in-memory state 原子运行；`rollout/state.py`、state store、pause/resume 与旧 rollout schema reader 均已删除。公共环境接口也不再含 idempotency key、checkpoint 或 restore：每个 fresh session 只接受连续且恰好一次的 step。

本文件是本轮唯一任务清单，接续已验收的第 1 轮（轨迹数据层 contracts）、第 2 轮（policy backbone）与第 3 轮（scoring & objective）。写法继续沿用前几轮的讲解式口径：每一节先说明“做什么”，随后解释“为什么这样做”，并点出做错时会出现什么症状。若某处字面指令与所附理由看起来冲突，以理由描述的意图为准并留一句 note 说明。`idea.tex` 与 `AGENTS.md` 都禁止修改。

## 0. 定位：八层导航图中的第 4 层

八层规划回顾：(1) 轨迹数据层（已完成——`src/skillev/contracts/ttb_*.py`）；(2) 模型层（已完成——`src/skillev/policy/`）；(3) 打分与目标层（已完成——`src/skillev/scoring/`）；**(4) rollout 引擎（本轮）**；(5) 训练循环；(6) 流诊断；(7) Bayesian 校准；(8) 相变与五动作进化。本轮只做第 4 层，不写第 5 层及以上的逻辑。开工时先确认 main 至少包含 `67157ac` 的 Phase 1–3 完成状态与 dependency-light scoring 修复；若 HEAD 更新到更晚提交，以更晚 main 为准，但不得回退这些不变量。

rollout 引擎是什么（人话）：它是“让当前前向策略真的出去跑一遍任务，并把全过程录成合格录像带”的层。它拿到任务、检索到的技能、当前前向策略快照和环境句柄，逐步生成内部推理 `r_t` 与动作 `a_t`，把动作交给 bounded agent / 环境执行，拿回公开反馈 `o_t^exec`，到终止时再从私有 evaluator 侧取得裸终局奖励 `R(τ)`，最终只通过第 1 层的正门构造 `TrajectoryRecord`。它还必须把 Phase 3 评分所需的 `initial_text` 与 rollout 身份清单一并交付，因为 `TrajectoryRecord` 只保存 H_0 的承诺哈希，不重复保存完整技能正文。

为什么它必须单独成层：rollout 是整个算法里最“脏”的部分——I/O、工具异常、环境超时、结构化动作解析、预算、状态恢复、私有 evaluator 边界都集中在这里；但它又不能污染前 3 层的数学。将来为了吞吐量把单栈 `PolicyBackbone.generate` 换成 vLLM 客户端时，理论上只替换本层的一个生成器实现；`TrajectoryRecord`、canonical renderer、teacher-forced scoring、Δ 与 `L_TTB` 都不改。反面症状是：若 rollout 自己另写 prompt 模板、自己重算 action token、或把 vLLM 返回的 logprob 塞进训练目标，采样和评分会变成两套分布，训练仍然能跑，loss 也可能下降，但已经不是 idea.tex 的目标。

本轮的核心交付不是“能调用一次模型”，而是同时证明四件事：

1. **on-policy**：一条轨迹内所有生成都来自同一个被钉住的当前前向策略快照，不能中途换 adapter，也不能接受服务端回报的旧快照；
2. **条件一致**：生成动作时使用的前缀，和 Phase 3 之后重打分时使用的 canonical forward prefix 是字节级同一文本、tokenizer 下同一 token 序列；
3. **边界精确**：`a_t` 的 token ids 是模型第二次生成直接产生的原始 token 序列，不从全文 completion 猜、不从 action 文本重新编码后替换；
4. **答案隔离**：环境反馈只含公开执行结果，私有真值和 evaluator 诊断只用于终局评分，永远不进入后续模型上下文。

## 1. 方法权威与本轮翻译关系

本轮把 idea.tex 的式 (1)–(5) 与式 (3) 翻译成一个可执行状态机：

- 式 (2) 的 `H_0 = q ⊕ S_ret ⊕ ω_q` → 确定性 `InitialContextAssembler`，产出完整 `initial_text` 与第 1 层 `InitialContext`；
- 式 (1) 的 `H_t = H_{t-1} ⊕ (r_t, a_t, o_t^exec)` → 每一步严格按“生成 `r_t` → 生成 `a_t` → 执行 → 记录 `o_t^exec`”的顺序提交；
- 式 (4) 的 `π_θ(a_t | r_t, H_{t-1})` → 动作生成必须直接使用 Phase 3 canonical forward prefix；
- 式 (5) 的 hindsight 条件不在 rollout 中调用后向策略，但 rollout 必须在 observation 到手后，用同一个 renderer 计算并保存 hindsight prefix hash，供 Phase 3 重放验证；
- 式 (3) 的 `R̃ = R + ε_min` → evaluator 只产裸 `TerminalReward`，rollout 在最终调用 `build_trajectory_record` 时做唯一一次平移。

### 1.1 “on-policy”在本项目中的精确定义

“使用当前模型生成”不够精确。本项目把一条轨迹称为 on-policy，必须同时满足：

- 轨迹开始时钉住一个 `PolicySnapshot`，至少包含冻结底座身份、前向 adapter 版本、tokenizer 身份和生成 backend 身份；
- 该轨迹的每一次 reasoning generation 与 action generation 都显式携带 `expected_policy_snapshot_id`；
- 每次结果都回报实际使用的 `policy_snapshot_id`，与期望值不一致立即拒绝整条轨迹；
- 轨迹结束前再次读取当前快照，确认中途没有被更新；
- 同一轨迹内检索到的技能版本、library version、上下文组装版本和 decoding snapshot 也固定不变。

为什么要这么严：将来 trainer 会不断更新前向 adapter。如果 rollout 服务还挂着旧 adapter，却只在日志里写“forward-policy”，旧策略生成的轨迹会被误当成当前策略轨迹，系统悄悄变成 off-policy。这个错不会抛异常，TTB 目标也照常有数，所以必须由快照身份在本层拦住。

### 1.2 每步采用“两次生成”，不是从一个混合 completion 里猜 action span

工程默认且本轮钉死的流程是：

1. 第一次调用前向策略，只生成当前步的 reasoning token，得到 `r_t`；
2. 用 Phase 3 renderer 把 `H_{t-1}` 与 `r_t` 拼成 canonical forward prefix；
3. 第二次调用同一个前向策略快照，只生成动作文本，第二次调用返回的全部内容 token（去掉显式终止 token 后）就是 `a_t` 的原始 token span；
4. 解析 `a_t` 为结构化动作，交给 bounded agent / 环境执行。

为什么不用“一次生成 reasoning + JSON action，再从全文里找 JSON”：completion usage 只告诉我们总 token 数，不告诉 reasoning 在哪里结束、action 从哪里开始。靠字符串索引、重 tokenize 或“把整个 completion 当 action”都会重新发明 `K_t`，而 Phase 1 与现有 OpenAI-compatible provider 已经明确拒绝这种猜测。两次生成牺牲一点吞吐，换来 action 边界结构性确定；未来 vLLM 可对两种请求做连续批处理，但语义不变。

reasoning 是式 (4) 的条件文本，不进入 action logprob 的打分对象；因此它可以为空，也不需要保存进 `K_t`。不过它必须被确定性 decode 成 `reasoning_text`，并由 canonical renderer 重组为第二次调用的 action prefix。动作则不同：`action_token_ids` 必须是第二次生成的原始内容 token；`action_text` 只用于环境解析和日志，最终还要通过第 1 层的 re-tokenize 三重校验。

### 1.3 不许“重试到模型终于输出合法 JSON”

结构无效的 action 是策略真实采到的行为，不是可以偷偷擦掉的工程噪声。若模型生成了非 JSON、字段缺失或调用了不存在的工具，只要 action token span 本身精确，就把它作为这一条边记录下来，由 bounded agent 返回 `parse_error` / `schema_invalid` 公开 observation，并消耗一个 horizon step。**不允许丢掉这次生成并用同一 trajectory_id 重采，直到得到合法动作**；那会把实际策略分布条件化为“只看可解析样本”，表面仍叫 on-policy，实际已经有选择偏差。

可以重试的只有确定性的传输层故障，并且重试必须复用完全相同的请求 id、seed 与期望快照，保证逻辑上仍是一份样本。本轮单栈实现不需要传输重试。若 action 为空、token round-trip 无法成立、快照身份不一致，无法形成合法的 Phase 1 边，整条 rollout 作为 infrastructure rejection 落账，但不产出训练用 `TrajectoryRecord`。

### 1.4 终局奖励与模型上下文严格分离

bounded agent 在循环内只做公开结构校验和环境执行。一个 `COMPLETE` 动作最多得到“提交形态有效、已送交评估”这类公开 observation；科学正确性、gold answer、隐藏测试、私有参数和 evaluator 诊断都不回送给模型。轨迹结束后，`TerminalEvaluator` 在私有侧用 opaque task id 和最终 submission 计算 `TerminalReward`。`native_payload` 可进入本地事件日志，但不能进入任何后续 prompt。

agent 自身失败是合法训练数据：错误工具调用、schema invalid、超时、horizon exhaustion 通常得到 `R=0` 或环境定义的低分；评测基础设施本身失败则不是 agent 行为，不能伪装成 `R=0` 污染训练，必须拒绝该 rollout 并记录 evaluator error。

### 1.5 单栈与未来两栈的演进规则

本轮只实现 **单栈 LocalPolicyGenerator**：它调用 Phase 2 的 `PolicyBackbone.generate`，与之后评分使用同一 tokenizer 和同一模型实现。与此同时，本轮必须先定义窄的 `RolloutGenerator` Protocol，并让 engine 只依赖这个 Protocol。

未来吞吐实测不足时，可在本包内部新增 `VLLMRolloutGenerator`：vLLM 只负责返回 token 与它实际加载的 `policy_snapshot_id`；Phase 3 仍对轨迹做 teacher-forced 重打分，vLLM 的 logprob 即使存在也不进入 Δ。本轮不接 vLLM、不起服务、不写 adapter 热加载；但 fake remote generator 测试必须证明旧快照结果会被拒绝。这样未来“两栈化”是替换一个 adapter，不是重写算法。

### 1.6 钉死项与工程自由

**钉死的语义**：两次生成顺序；动作生成使用 Phase 3 canonical forward prefix；同轨迹固定快照；action token ids 取自生成原始 span；无隐藏重采；公开 observation 与私有 reward 隔离；最终只通过 `build_trajectory_record` 入库；rollout 不调用后向策略、不计算 P̂/Δ/loss。

**工程自由**：H_0 文本的具体标题样式（但必须版本化）；默认 `max_reasoning_tokens` / `max_action_tokens` / temperature / top-p；文件拆分；环境 session 是同步包装还是 async 原生；公开错误消息的具体措辞；状态存储后端。调整工程默认需留一句理由，但不得改变上面的语义。

## 2. 范围刀切与下层最小加法

### 2.1 先收口 bootstrap runtime 的“第二套科学权威”

当前仓库存在 bootstrap 时期的 `runtime.trajectory.TrajectoryStep`、`PendingTrajectoryStep`、`BackwardTokenScorer` 与由 provider 返回 forward/backward logprob 的路径。这些资产曾用于恢复通用实验室基础设施，但 Phase 1–3 已建立新的唯一权威：

- 科学轨迹类型只能是 `skillev.contracts.TrajectoryStep` / `TrajectoryRecord`；
- 前后向 logprob 只能由 Phase 3 的 teacher-forced scoring 产生；
- bounded agent 只负责有限步执行与公开 observation，不负责模型生成、后向打分或科学轨迹落账。

本轮必须做最小收口：将现有 `BoundedAgent` 重构为 execution-only 边界；移除 active path 对 `runtime.trajectory.TrajectoryStep` 与 `BackwardTokenScorer` 的依赖；`runtime.trajectory` 不再从公共 `runtime.__init__` 导出，若无其它有效调用方可直接删除。旧 `ModelProvider` / `OpenAICompatibleProvider` 可保留作通用传输资产或未来 generator adapter 的参考，但 Phase 4 engine 不得依赖其 `action_token_logprobs`，也不得让它们重新成为 action span 的科学权威。

为什么这一步属于 Phase 4：在前 3 轮，bootstrap runtime 没有进入数学路径，可以暂留；现在 rollout 正要接上它。如果不先裁定唯一权威，系统会同时出现“runtime trajectory”和“TTB TrajectoryRecord”，下游没人知道该信哪套字段，最终一定发生转换丢信息或 logprob 重复计算。

### 2.2 本轮只允许三项向下的兼容性加法

前 1–3 层行为原则上冻结，但 rollout 真实落地需要三个此前刻意没开放的能力。允许做**向后兼容的加法**，禁止改已有语义：

1. `TokenizerProtocol` / `HFTokenizer` 增加精确 `decode(token_ids) -> str`，必须 `skip_special_tokens=False`、关闭 cleanup；已有 `encode` 与 tokenizer_id 行为不变；
2. Phase 2 `GenerationResult` 增加显式 `stop_token_ids`（或等价字段），说明 output 中哪一段是 EOS/停止 token；已有 `output_token_ids` 含义不变，rollout 由此剥离 stop suffix，不靠猜；
3. Phase 3 renderer 抽出“按部件渲染”的公共入口：reasoning prompt、给定 `reasoning_text` 的 forward prefix、给定 `observation_text` 的 hindsight prefix。已有 `render_forward_prefix` / `render_hindsight_prefix` 必须委托新底层函数，输出文本与哈希逐位不变。

这三项各自要有回归测试。除此之外不改 contracts、policy scoring、Δ 或 loss。

### 2.3 本轮明确不碰的层

- 不建 optimizer、不 backward、不更新 adapter、不组装 `TTBBatchStats`；
- 不算 P̂_F/P̂_B、`EdgeLogprobRecord`、`TrajectoryResidual`、log I、F、F̂；
- 不更新 posterior、不算 LCB/UCB、不判断相变、不改技能库；
- 不接外部 benchmark 数据；
- 不实现 vLLM、DeepSpeed、跨卡采样或吞吐优化。

## 3. 组件逐个用人话讲（含代码骨架）

骨架规则同前三轮：字段语义和不变量不可减；命名与文件组织可在一句话理由内调整；注释中写“需要写”的就是实现规格。建议新包为 `src/skillev/rollout/`，但文件怎么拆由实现者决定。

### 3.1 共享身份：PolicySnapshot、DecodingSnapshot 与 RolloutRequest

**是什么**：开始跑轨迹前先拍一张“现场照片”，把策略、tokenizer、技能库和解码参数全部钉住。**为什么**：on-policy 不是靠口头保证，而是靠每次生成都能回指同一张照片。

```python
class GenerationPhase(StrEnum):
    """同一步的两类生成调用。"""

    REASONING = "reasoning"
    ACTION = "action"

class RolloutTermination(StrEnum):
    """能形成合法 TrajectoryRecord 的终止原因。"""

    COMPLETED = "completed"                    # 有结构有效的 terminal submission
    HORIZON_EXHAUSTED = "horizon-exhausted"    # 达到 max_steps；这是 agent 结果，不是 infra error

@dataclass(frozen=True, slots=True)
class PolicySnapshot:
    """一条轨迹所用前向策略的不可变身份。"""

    snapshot_id: str
    # 需要写：由下面字段 stable_hash 得到，不能由调用方随便填。

    backbone_id: str
    # Phase 2 backbone.backbone_id；审计辅助。非空。

    forward_adapter_version: str
    # backbone.adapter_version(AdapterRole.FORWARD_POLICY)。Phase 5 每次策略更新后必须换版本；
    # Phase 4 只消费，不负责递增。

    tokenizer_id: str
    backend_id: str
    # 单栈固定如 "hf-local"；未来 vLLM 为独立 id。backend 变化不应改 engine。

    def __post_init__(self) -> None:
        """需要写：字段非空；按组成字段重算 snapshot_id 并比对。"""
        ...

@dataclass(frozen=True, slots=True)
class DecodingSnapshot:
    """一整条轨迹的采样参数；其 id 写入 TrajectoryRecord.decoding_snapshot_id。"""

    snapshot_id: str
    temperature: float
    top_p: float
    max_reasoning_tokens: int
    max_action_tokens: int
    base_seed: int
    format_version: str = "two-pass-r-a@1"

    def __post_init__(self) -> None:
        """需要写：temperature > 0、top_p ∈ (0,1]、两个 token 上限 >0、
        base_seed 是 unsigned 64-bit；snapshot_id 由参数 stable_hash 重算。"""
        ...

def derive_generation_seed(
    *,
    base_seed: int,
    trajectory_id: str,
    step_index: int,
    phase: GenerationPhase,
) -> int:
    """需要写：用 stable_hash 确定性派生一个 unsigned 64-bit seed。

    同一实验仍只有 AGENTS.md 要求的一组 base seed；这里的派生只是避免同一轨迹每步、
    reasoning/action 两次调用复用完全相同随机流，不是多 seed 实验。禁止碰全局 RNG。
    """
    ...

@dataclass(frozen=True, slots=True)
class RolloutTask:
    """模型侧可见的公开任务投影。绝不含 gold answer、隐藏测试或私有生成参数。"""

    task_id: str
    environment_id: str
    task_family: str
    query: str
    public_context: JsonValue

    def __post_init__(self) -> None:
        """需要写：id/query 非空；public_context normalized canonical JSON。"""
        ...

@dataclass(frozen=True, slots=True)
class RolloutRequest:
    """跑一条轨迹的完整输入。"""

    trajectory_id: str
    task: RolloutTask
    retrieved_skills: tuple[RetrievedSkillContext, ...]
    library_version: str
    decoding: DecodingSnapshot
    epsilon_min: float

    def __post_init__(self) -> None:
        """需要写：trajectory/library id 非空；skill id 不重复且每个 manifest 版本明确；
        epsilon_min > 0。不得出现 evaluator 私有句柄之外的私有内容。"""
        ...
```

**典型错误症状**：只记录 decoding 参数、不记录策略快照时，两条由不同 adapter 生成的轨迹在日志里长得一样；Phase 5 混批后出现残差抖动，却无法判断是训练现象还是 rollout 服务陈旧。

### 3.2 H_0 组装器：把公开任务、技能与元信息变成唯一 initial_text

**是什么**：确定性拼出模型真正看到的初始文本，同时构造 Phase 1 `InitialContext`。**为什么单独成组件**：`TrajectoryRecord` 只存 assembled hash，不存技能全文；Phase 3 重打分时必须拿回一模一样的 `initial_text`。组装规则若散落在环境、CLI 和测试里，模板改一个换行就会使旧轨迹不可重放。

```python
INITIAL_CONTEXT_FORMAT_VERSION: Final = "ttb-initial-context@1"

@dataclass(frozen=True, slots=True)
class AssembledInitialContext:
    """H_0 的载荷与承诺。"""

    text: str
    contract: InitialContext

    def __post_init__(self) -> None:
        """需要写：text 非空；assembled_context_hash(text) == contract.assembled_hash。"""
        ...

    def to_value(self) -> dict[str, JsonValue]:
        """需要写：供本地 rollout artifact/event 保存完整 initial_text。
        注意它可能含许可任务题面与技能文本，只能进本地/集群私有运行产物，不能进 Git。"""
        ...

class InitialContextAssembler(Protocol):
    @property
    def assembler_version(self) -> str: ...

    def assemble(
        self,
        *,
        task: RolloutTask,
        retrieved_skills: tuple[RetrievedSkillContext, ...],
        library_version: str,
        tokenizer: "RolloutTokenizerProtocol",
    ) -> AssembledInitialContext:
        """需要写的语义：
        1. 只读取 task 的 public projection 与 skill 的 model-visible disclosure；
        2. 技能按检索结果 tuple 的既有顺序拼接（不擅自排序），id/version/content hash 一并写进公开头部；
        3. public_context 用 canonical_json，不用 str(dict)；
        4. 组装结果哈希调用 Phase 3 assembled_context_hash，禁止自造另一种算法；
        5. InitialContext.retrieved_skill_ids 与实际拼入的技能逐项一致；
        6. assembled_token_count = len(tokenizer.encode(text))；
        7. meta 可记录 task_id、task_family、library_version、公开工具 schema 版本，
           但禁止放 policy 私钥、gold answer、verifier 诊断。"""
        ...

class CanonicalInitialContextAssembler:
    """本轮需要写的默认实现。具体标题样式是工程默认，但必须版本化、确定性。"""
```

建议默认 H_0 分成固定的 Query / Retrieved Skills / Public Task Context / Available Actions 四段。技能内容可长，但同一 manifest 只能出现一次；不要在 `InitialContext.meta` 再复制全文。

### 3.3 Phase 3 renderer 的 rollout 入口：同一个模板同时服务生成与打分

**是什么**：给 renderer 增加能在“当前 step 还没构造出来”时调用的部件式入口。**为什么**：现有 `render_forward_prefix` 接收完整 `TrajectoryStep`，测试中可先放 placeholder hash 再 replace；真实 rollout 在 action/observation 尚未产生时不能造一个假 step。正确做法是抽取底层纯函数，不是在 rollout 包复制模板。

```python
# 这些函数放在 skillev.scoring.rendering（或等价的唯一 canonical renderer 模块）。
# 现有 render_forward_prefix / render_hindsight_prefix 必须委托它们，回归测试证明输出不变。

@dataclass(frozen=True, slots=True)
class RenderedReasoningPrompt:
    """第一次生成的 canonical prompt。单独成类型，不扩张 Phase 3 的
    PrefixKind={forward,hindsight} 封闭集合。"""

    step_index: int
    text: str
    prompt_hash: str


def render_reasoning_prefix(
    initial_text: str,
    previous_steps: tuple[TrajectoryStep, ...],
    step_index: int,
) -> RenderedReasoningPrompt:
    """需要写：initial_text + previous_steps 的完整 r/a/o 历史 +
    f"### Step {step_index}\nReasoning:\n"。

    这是第一次生成的输入。previous_steps 只能是 1..step_index-1；prompt_hash 使用
    stable_hash({template_version, step_index, text})，用于 RolloutManifest，不写进 Phase 1 step。
    """
    ...
def render_forward_prefix_from_parts(
    initial_text: str,
    previous_steps: tuple[TrajectoryStep, ...],
    step_index: int,
    reasoning_text: str,
) -> RenderedPrefix:
    """需要写：与现有 forward 模板字节级一致；尾部固定为 Action:\n。
    这是第二次 action generation 的输入，也是 Phase 3 teacher-forced forward scoring 的条件。"""
    ...

def render_hindsight_prefix_from_parts(
    initial_text: str,
    previous_steps: tuple[TrajectoryStep, ...],
    step_index: int,
    observation_text: str,
) -> RenderedPrefix:
    """需要写：历史完整 r/a/o + 当前 observation；绝不读取当前 reasoning。
    rollout 在 observation 到手后用它生成 TrajectoryStep.hindsight_prefix_hash。"""
    ...
```

回归不变量：对任何既有完整 steps，`render_forward_prefix(...)` 与 `_from_parts(...)` 的 text/hash 必须逐位相等，hindsight 同理。若这一条不成立，说明本轮在改 Phase 3 语义，立即停下。

### 3.4 生成器边界：RolloutGenerator 与单栈 LocalPolicyGenerator

**是什么**：engine 眼里唯一的模型生成接口。**为什么另包一层**：Phase 2 的 `PolicyBackbone` 是训练栈接口；future vLLM 是远程服务。rollout 只需要“给 token prefix 和快照，返 token”，不应知道调用是在本进程还是 HTTP。

```python
class RolloutTokenizerProtocol(TokenizerProtocol, Protocol):
    """Phase 1 tokenizer 的向后兼容扩展。"""

    def decode(self, token_ids: tuple[int, ...]) -> str:
        """需要写：精确 decode；skip_special_tokens=False，关闭 cleanup。
        不承诺任意 token 序列都 encode(decode(ids))==ids，action 入库前由 rollout 显式检查。"""
        ...

@dataclass(frozen=True, slots=True)
class RolloutGenerationRequest:
    phase: GenerationPhase
    input_ids: tuple[int, ...]
    max_new_tokens: int
    seed: int
    decoding_snapshot_id: str
    expected_policy_snapshot_id: str

    def __post_init__(self) -> None:
        """需要写：input 非空、max>0、参数范围合法、两个 snapshot id 非空。"""
        ...

@dataclass(frozen=True, slots=True)
class RolloutGenerationResult:
    """rollout 使用的生成结果；内容 token 与停止 token 已明确分开。"""

    content_token_ids: tuple[int, ...]
    stop_token_ids: tuple[int, ...]
    finish_reason: str
    policy_snapshot_id: str
    backend_id: str
    usage: BudgetVector = field(default_factory=BudgetVector)
    # 模型调用的实际 input/output/model_calls/agent_turns；engine 用它结算 BudgetLedger。

    def __post_init__(self) -> None:
        """需要写：ids 合法；快照/backend 非空；stop token 不混入 content；
        reasoning content 可空，action 是否允许空由 engine 按 phase 判断；usage 非负且
        output_tokens == len(content_token_ids) + len(stop_token_ids)。"""
        ...

class RolloutGenerator(Protocol):
    @property
    def tokenizer(self) -> RolloutTokenizerProtocol: ...

    def snapshot(self) -> PolicySnapshot:
        """返回此刻实际可用于生成的前向策略身份。"""
        ...

    async def generate(self, request: RolloutGenerationRequest) -> RolloutGenerationResult:
        """只生成 token，不解析 action、不执行环境、不返回训练用 logprob。"""
        ...

class LocalPolicyGenerator:
    """本轮需要写的单栈实现，内部持有 PolicyBackbone。运行时不得 import torch。"""

    def __init__(self, backbone: PolicyBackbone) -> None: ...

    @property
    def tokenizer(self) -> RolloutTokenizerProtocol: ...

    def snapshot(self) -> PolicySnapshot:
        """需要写：读取 backbone_id、forward adapter version、tokenizer_id，
        backend_id="hf-local"，计算 snapshot_id。"""
        ...

    async def generate(self, request: RolloutGenerationRequest) -> RolloutGenerationResult:
        """需要写：
        1. 调用前 snapshot == request.expected_policy_snapshot_id，否则拒绝；
        2. 转成 Phase 2 GenerationRequest，调用 backbone.generate；
        3. 根据 Phase 2 GenerationResult.stop_token_ids 精确拆 content/stop；
        4. 调用后再读 snapshot，若改变则拒绝（防调用期间策略被更新）；
        5. 回报实际 snapshot id；忽略任何 provider/generation logprob。
        """
        ...
```

Phase 2 的最小加法建议：`GenerationResult` 增加 `stop_token_ids: tuple[int, ...] = ()`，并校验它是 `output_token_ids` 的 suffix；HF 实现遇 EOS 时填入对应 token。不要让 rollout 根据“最后一个 token 看起来像 EOS”猜。

### 3.5 token 段解码与 action codec：文本用于执行，token 才是训练身份

**是什么**：把 reasoning/action 内容 token 转成文本，并把 action 文本解析成现有 `StructuredAction`。**为什么把 decode 与 parse 分开**：token round-trip 是训练数据完整性问题；JSON 合法性是 agent 行为问题。前者失败要拒绝轨迹，后者失败要作为 observation 记录，不能混为一谈。

```python
@dataclass(frozen=True, slots=True)
class DecodedSegment:
    text: str
    token_ids: tuple[int, ...]

def decode_reasoning_segment(
    tokenizer: RolloutTokenizerProtocol,
    token_ids: tuple[int, ...],
) -> DecodedSegment:
    """需要写：decode 成文本；允许空 token/空文本。
    reasoning 不属于 K_t，不强制 encode(text)==原 ids；但禁止插入/删除模型可见字符，
    decode 配置必须固定。可在 RolloutManifest 记录原 token count/hash供审计。"""
    ...

def decode_action_segment(
    tokenizer: RolloutTokenizerProtocol,
    token_ids: tuple[int, ...],
) -> DecodedSegment:
    """需要写：action ids 必须非空；decode 后 action_text 非空；
    tuple(tokenizer.encode(action_text)) 必须逐位 == 原 token_ids，否则抛 RolloutBoundaryError。
    **绝不能用重编码结果替换原 ids**——检查通过后仍返回生成原 ids。"""
    ...

class ActionParseStatus(StrEnum):
    VALID = "valid"
    PARSE_ERROR = "parse-error"          # 不是 JSON
    SCHEMA_INVALID = "schema-invalid"    # JSON 可读但不满足 StructuredAction

@dataclass(frozen=True, slots=True)
class ActionParseResult:
    status: ActionParseStatus
    action: StructuredAction | None
    public_error_code: str | None

    def __post_init__(self) -> None:
        """需要写：VALID ⟺ action 非 None 且 error None；失败状态反之。"""
        ...

class ActionCodec(Protocol):
    @property
    def format_version(self) -> str: ...

    def parse(self, action_text: str) -> ActionParseResult: ...

class StructuredJsonActionCodec:
    """本轮默认 action wire format：一个 JSON object，对应 runtime.StructuredAction。"""

    format_version = "structured-action-json@1"

    def parse(self, action_text: str) -> ActionParseResult:
        """需要写：
        - json.loads 失败 → PARSE_ERROR；
        - 读出后 normalize_json，再走 StructuredAction.from_value；失败 → SCHEMA_INVALID；
        - 不修改 action_text，不把 canonical_json(parsed) 回写成训练文本；
        - 错误消息只给公开粗码，不把内部堆栈/路径送进模型 observation。"""
        ...
```

典型错误：模型输出 `{"kind":"tool",...}`，parser 为了“规范化”把空格重排后拿新文本重 tokenize 入库。环境执行看起来正确，但记录中的 token 已不是模型生成的 token，Phase 3 重打分在另一个事件上。正确做法是 exact text/token 用于记录，normalized `StructuredAction` 只用于执行。

### 3.6 环境与私有 evaluator：公开交互和科学评分是两条线

**是什么**：`RolloutEnvironmentSession` 管每一步公开执行；`TerminalEvaluator` 只在轨迹结束后给奖励。**为什么分开**：环境可能知道公开工具状态，evaluator 可能持有 gold truth。把两者做成一个对象很容易在 observation 中顺手泄漏“为什么错”。

```python
@dataclass(frozen=True, slots=True)
class EnvironmentObservation:
    """bounded agent 可交回模型历史的全部信息。"""

    public_value: JsonValue
    observation_status: str
    # 必须属于 Phase 1 OBSERVATION_STATUSES：success/tool_error/schema_invalid/timeout/parse_error/other。

    invoked_skill_ids: tuple[str, ...] = ()
    terminal_submission: JsonValue = None
    terminal: bool = False
    budget_usage: BudgetVector = field(default_factory=BudgetVector)

    def __post_init__(self) -> None:
        """需要写：public_value normalized；状态合法；skill id 唯一；
        terminal=False 时 terminal_submission 必须 None；budget 非负。"""
        ...

    @property
    def observation_text(self) -> str:
        """需要写：用 canonical_json(public_value) 得到确定性非空文本，
        绝不用 repr/str(dict)。这段文本才进入 TrajectoryStep。"""
        ...

class RolloutEnvironmentSession(Protocol):
    @property
    def environment_id(self) -> str: ...

    @property
    def task_family(self) -> str: ...

    async def execute(
        self,
        action: StructuredAction,
        *,
        step_index: int,
    ) -> EnvironmentObservation:
        """执行 tool/skill action。需要保证返回的是公开 projection；
        私有真值、隐藏测试、评分理由不准出现在 public_value。"""
        ...

    def validate_completion(self, submission: JsonValue) -> bool:
        """只检查公开结构/shape，不检查科学正确性。"""
        ...

@dataclass(frozen=True, slots=True)
class TerminalEvaluationRequest:
    trajectory_id: str
    task_id: str
    termination: RolloutTermination
    submission: JsonValue
    public_transcript_hash: str

class TerminalEvaluator(Protocol):
    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        """私有侧评分。返回裸 R；不得把结果写回 environment session 或模型历史。
        evaluator 基础设施失败应抛专用异常，engine 将 rollout 标为 rejected，而不是伪造 R=0。"""
        ...
```

现有 `TrustedEvaluatorOutcome` / `TerminalRewardAdapter` 可以作为具体 evaluator adapter 的底盘。P1/social 私有包不得被 model-facing wheel 或 rollout worker import；rollout 只持有一个窄 evaluator client/句柄。

### 3.7 bounded agent：从“模型+评分总管”重构为 execution-only 有界执行器

**是什么**：bounded agent 只接收 rollout 已经生成并解析好的动作，负责 turn 上限、预算、工具/技能执行、公开错误投影、completion shape 校验和 checkpoint。**为什么要削薄**：模型生成属于 generator，token 记录属于 rollout，后向打分属于 Phase 3。bounded agent 同时做三件事会重新制造层间耦合。

```python
@dataclass(frozen=True, slots=True)
class BoundedAgentState:
    invocation_id: str
    turns_used: int = 0
    reservation_sequence: int = 0
    completed: bool = False
    completion_value: JsonValue = None
    environment_checkpoint: JsonValue = None

    def __post_init__(self) -> None:
        """需要写：计数非负；completed 与 completion_value 一致；checkpoint canonical。"""
        ...

@dataclass(frozen=True, slots=True)
class BoundedAgentTurnRequest:
    trajectory_id: str
    step_index: int
    action_text: str
    action_token_ids: tuple[int, ...]
    parse_result: ActionParseResult

@dataclass(frozen=True, slots=True)
class BoundedAgentTurnResult:
    state: BoundedAgentState
    observation: EnvironmentObservation

@dataclass(frozen=True, slots=True)
class BoundedAgentPolicy:
    max_turns: int

class BoundedAgentStateStore(Protocol):
    def load(self, *, invocation_id: str) -> BoundedAgentState | None: ...
    def save(self, *, invocation_id: str, state: BoundedAgentState) -> None: ...


class BoundedAgent:
    """执行器；构造函数不再接 model/prompt_builder/action_parser/backward_scorer。"""

    def __init__(
        self,
        *,
        environment: RolloutEnvironmentSession,
        policy: BoundedAgentPolicy,
        ledger: BudgetLedger,
        tool_call_maximum: BudgetVector,
        state_store: BoundedAgentStateStore,
        emitter: RuntimeEventEmitter,
    ) -> None:
        """需要写：复用现有工具预算、状态存储与事件基础设施；
        bounded agent 自身不生成模型 token，也不结算模型调用预算。模型预算由 RolloutEngine
        根据 RolloutGenerationResult.usage 处理，避免两个组件重复记账。"""
        ...

    async def execute_turn(
        self,
        state: BoundedAgentState,
        request: BoundedAgentTurnRequest,
    ) -> BoundedAgentTurnResult:
        """需要写：
        1. 检查 step_index == state.turns_used + 1、未超过 max_turns；
        2. PARSE_ERROR/SCHEMA_INVALID：不调用环境，构造对应公开 observation，消耗一 turn；
        3. TOOL/SKILL：调用 environment.execute；异常按 timeout/tool_error 做去敏公开投影；
        4. COMPLETE：只调用 environment.validate_completion；shape 合法则 terminal=True，
           observation 只说已接收评估，不含正确性；不合法则 schema_invalid、继续；
        5. invoked_skill_ids 取实际执行结果；不得根据 action 文本字符串猜；
        6. 更新状态、预算、checkpoint、事件后返回。
        """
        ...
```

本轮迁移完成后，`BoundedAgent` 不再拥有 `PendingTrajectoryStep`、forward logprob、backward logprob，也不再产出 `runtime.trajectory.TrajectoryStep`。若旧 API 仍有非 TTB 调用方，允许用清楚标注的 legacy adapter 暂时包住，但新 rollout 路径和公共导出只能指向 execution-only 版本。

### 3.8 step 草稿与 Phase 1 正门：把生成 token 和环境结果合成 TrajectoryStep

**是什么**：生成与执行分阶段完成，先保存临时草稿，只有 `r/a/o` 全部到齐后才构造 Phase 1 `TrajectoryStep`。**为什么**：动作已经生成但环境还没返回时，不能提前落一条缺 observation 的科学边；发生基础设施故障时应拒绝，而不是填空字符串。

```python
@dataclass(frozen=True, slots=True)
class ReasoningDraft:
    step_index: int
    prompt_text: str
    prompt_hash: str
    text: str
    generated_token_ids: tuple[int, ...]
    policy_snapshot_id: str

@dataclass(frozen=True, slots=True)
class ActionDraft:
    step_index: int
    forward_prefix_text: str
    forward_prefix_hash: str
    text: str
    token_ids: tuple[int, ...]
    parse_result: ActionParseResult
    policy_snapshot_id: str

@dataclass(frozen=True, slots=True)
class CompletedStepDraft:
    reasoning: ReasoningDraft
    action: ActionDraft
    observation: EnvironmentObservation

    def __post_init__(self) -> None:
        """需要写：step index、snapshot id 一致；observation_text 非空。"""
        ...

def materialize_trajectory_step(
    *,
    initial_text: str,
    previous_steps: tuple[TrajectoryStep, ...],
    draft: CompletedStepDraft,
) -> TrajectoryStep:
    """需要写：
    - forward hash 重新调用 render_forward_prefix_from_parts，并与 ActionDraft 承诺比对；
    - hindsight hash 调 render_hindsight_prefix_from_parts；
    - action_text/token_ids 原样来自 ActionDraft，K_t=len(token_ids)；
    - observation_text 来自 canonical public observation；
    - observation_status 与 invoked_skill_ids 来自实际 BoundedAgent result；
    - 构造 Phase 1 TrajectoryStep，不写任何 logprob。
    """
    ...

@dataclass(frozen=True, slots=True)
class RolloutManifest:
    """TrajectoryRecord 之外的 on-policy/工程身份清单。"""

    trajectory_id: str
    task_id: str
    policy_snapshot: PolicySnapshot
    library_version: str
    decoding_snapshot_id: str
    assembler_version: str
    action_format_version: str
    generator_backend_id: str
    termination: RolloutTermination
    reasoning_token_counts: tuple[int, ...]
    record_content_hash: str
    started_at: str
    completed_at: str

    def __post_init__(self) -> None:
        """需要写：step 数与 reasoning counts 对齐；时间/id 非空；
        backend 与 policy snapshot 一致。created time 不进 manifest content hash。"""
        ...

@dataclass(frozen=True, slots=True)
class RolloutArtifact:
    """Phase 5 将来消费的一条完整采样产物。"""

    initial_context: AssembledInitialContext
    record: TrajectoryRecord
    manifest: RolloutManifest

    def __post_init__(self) -> None:
        """需要写：三个 trajectory/context/hash 身份互相咬合；
        assembled hash/token count 与 record.initial_context 相同；
        manifest.record_content_hash == record.content_hash。"""
        ...

    def to_value(self) -> dict[str, JsonValue]:
        """需要写：本地事件/存储往返。包含 initial_text，因此不得提交到 Git。"""
        ...

def finalize_trajectory_record(
    *,
    request: RolloutRequest,
    assembled: AssembledInitialContext,
    steps: tuple[TrajectoryStep, ...],
    reward: TerminalReward,
    tokenizer: RolloutTokenizerProtocol,
    created_at: str,
) -> TrajectoryRecord:
    """需要写：shifted_reward = reward.value + request.epsilon_min，只加一次；
    调用 Phase 1 build_trajectory_record 正门（不是裸构造），tokenizer_id 与 decoding id 取已钉快照。"""
    ...
```

### 3.9 rollout 状态与恢复：只承诺 step-boundary，不假装能原子化外部世界

**是什么**：每完成一条 `r/a/o` 边就存状态，进程在干净边界退出后可恢复。**为什么限定边界**：外部工具可能已执行但进程还没来得及落账，通用代码无法凭空保证 exactly-once。伪称支持会导致工具动作重复执行。

```python
@dataclass(frozen=True, slots=True)
class RolloutState:
    trajectory_id: str
    policy_snapshot: PolicySnapshot
    assembled_context: AssembledInitialContext
    completed_steps: tuple[TrajectoryStep, ...]
    bounded_agent_state: BoundedAgentState
    library_version: str
    decoding_snapshot_id: str
    action_format_version: str
    next_step_index: int

    def __post_init__(self) -> None:
        """需要写：steps 连续、next==len(steps)+1、snapshot/context 身份一致；
        library/decoding/action-format id 非空。恢复时这些字段必须与新 request 全等。"""
        ...

class RolloutStateStore(Protocol):
    def load(self, trajectory_id: str) -> RolloutState | None: ...
    def save(self, state: RolloutState) -> None: ...
    def delete(self, trajectory_id: str) -> None: ...

@dataclass(slots=True)
class InMemoryRolloutStateStore:
    """测试与单进程默认实现；线程安全。"""
```

恢复规则：只从“完整 observation 已落账”的 step boundary 恢复；恢复时当前 generator snapshot 必须仍等于 state snapshot，环境 checkpoint 也必须可恢复，否则拒绝旧 trajectory，使用新 trajectory_id 重新采。发生 in-flight 外部 side effect 不确定时不自动重试，不用新代码假装解决分布式事务。

### 3.10 RolloutEngine：唯一总装状态机

**是什么**：把前面所有窄组件按固定顺序连接。它有状态但算法很薄；复杂性来自 I/O 与失败处理，不来自数学。

```python
class RolloutEngine:
    def __init__(
        self,
        *,
        generator: RolloutGenerator,
        context_assembler: InitialContextAssembler,
        action_codec: ActionCodec,
        bounded_agent: BoundedAgent,
        terminal_evaluator: TerminalEvaluator,
        state_store: RolloutStateStore,
        ledger: BudgetLedger,
        model_call_maximum: BudgetVector,
        emitter: RuntimeEventEmitter,
        clock: Callable[[], str],
    ) -> None: ...

    async def run(self, request: RolloutRequest) -> RolloutArtifact:
        """需要写，顺序不可调：

        A. 开局
        1. pinned = generator.snapshot()；组装 H_0；创建/恢复 RolloutState；
        2. 若恢复状态的 snapshot/library/decoding/assembler 任一不一致，拒绝；
        3. 记录 ROLLOUT_STARTED / POLICY_PINNED 类事件（只记录公开身份与 hash）。

        B. 每一步 t
        4. 再验 generator.snapshot()==pinned；
        5. reasoning_prompt = render_reasoning_prefix(initial_text, completed_steps, t)；encode；
        6. 为 reasoning model call 预留预算；用 derive_generation_seed(...REASONING) 调
           generator.generate；验回报 snapshot/usage，结算预算；decode_reasoning_segment，得到 r_t；
        7. forward_prefix = render_forward_prefix_from_parts(..., r_t)；encode；
           该 exact ids 作为 action generation input；
        8. 为 action model call 预留预算；用 derive_generation_seed(...ACTION) 调
           generator.generate；验回报 snapshot/usage，结算预算；action content 必须非空；
           decode_action_segment 做 token round-trip；ActionCodec.parse；
        9. bounded_agent.execute_turn：合法 action 执行；无效 action 形成 parse/schema observation；
        10. materialize_trajectory_step，追加 completed_steps，存 step-boundary state，发 step event；
        11. observation.terminal=True 则保存 submission 并退出；否则到 max_steps 后按 horizon 退出。

        C. 终局
        12. 再验 generator snapshot 未变；构造 TerminalEvaluationRequest，私有侧评分；
            evaluator infrastructure error → RolloutRejectedError，不产 TrajectoryRecord；
        13. finalize_trajectory_record 走 Phase 1 工厂；构造 RolloutManifest/Artifact；
        14. 事件落账顺序：terminal reward → trajectory/initial payload → rollout completed；
            最后删除 checkpoint，返回 artifact。

        本函数绝不调用 score_trajectory、backward、optimizer，也不读取后向 adapter。
        """
        ...
```

**没有隐藏采样循环**：对每个完成的 step，generator 调用次数严格等于 2。结构错误不是“再试一次”的触发器。测试必须断言调用计数。

### 3.11 拒绝分类与事件：区分 agent 失败和基础设施失败

**是什么**：不是所有失败都能变成 reward=0。**为什么分开**：把 tokenizer 漂移或 stale policy 当作 agent 失败，会让模型为工程 bug 背锅。

```python
class RolloutRejectionKind(StrEnum):
    POLICY_SNAPSHOT_MISMATCH = "policy-snapshot-mismatch"
    TOKEN_ROUNDTRIP_MISMATCH = "token-roundtrip-mismatch"
    EMPTY_ACTION = "empty-action"
    INITIAL_CONTEXT_MISMATCH = "initial-context-mismatch"
    ENVIRONMENT_INFRASTRUCTURE_ERROR = "environment-infrastructure-error"
    EVALUATOR_ERROR = "evaluator-error"
    UNRECOVERABLE_STATE = "unrecoverable-state"

@dataclass(frozen=True, slots=True)
class RolloutRejection:
    trajectory_id: str
    kind: RolloutRejectionKind
    stage: str
    step_index: int | None
    retryable: bool
    public_message: str


class RolloutRejectedError(RuntimeError):
    def __init__(self, rejection: RolloutRejection) -> None:
        """需要写：保存结构化 rejection；Exception 文本只用脱敏 public_message，
        不串联底层私密堆栈或 task truth。"""
        ...
```

建议新增最少 EventType：`ROLLOUT_STARTED`、`ROLLOUT_STEP_COMMITTED`、`ROLLOUT_REJECTED`、`ROLLOUT_ARTIFACT_RECORDED`、`ROLLOUT_COMPLETED`。持久化 DTO（PolicySnapshot、DecodingSnapshot、AssembledInitialContext、Manifest、Artifact、State）均需提供 canonical `to_value/from_value` 往返；代码骨架未逐个重复签名，但验收必须覆盖。事件 payload 不记录凭据；完整 initial_text 与题面只存在本地运行日志/工件，不提交 Git。

## 4. 测试策略（本轮的显微镜清单）

本轮的大部分测试必须 CPU、无网络、无真实权重。用可 encode/decode 的确定性 tokenizer、ScriptedRolloutGenerator、ScriptedEnvironmentSession 和 fake evaluator 驱动**真实 RolloutEngine**。不能另写一套只为测试服务的简化 engine。

### 4.1 两次生成与 canonical 条件

- 每个成功 step 恰好两次 generator 调用，顺序 reasoning→action；
- 第一次 input text 等于 `render_reasoning_prefix`；
- 第二次 input text 等于 `render_forward_prefix_from_parts`；
- 把最终 `TrajectoryStep` 交给原有 `render_forward_prefix`，text/hash 与 action generation prefix 逐位相等；
- hindsight hash 与当前 observation、历史完整 r/a/o 一致，且不含当前 reasoning；
- reasoning 为空串时结构不塌，action 仍可生成。

### 4.2 token 身份与边界

- action token ids 与第二次 generation 的 `content_token_ids` 是同一 tuple；
- stop token suffix 不进入 action ids/K_t；
- action decode→encode round-trip 不一致时整条 rollout 被拒，测试证明代码没有用重编码结果替换原 ids；
- 生成非 JSON action：token span 仍入 step，observation_status=`parse_error`，不重采；
- JSON 可读但 schema 错：`schema_invalid`；
- action 为空：无法形成 Phase 1 边，rejected；
- 构造 2 步、不同 K_t 的 artifact，`build_trajectory_record` 正门通过。

### 4.3 on-policy 快照

- 两次调用和多步轨迹全部携带同一 expected snapshot；
- fake generator 在第二步回报 stale snapshot → 立即拒绝，无最终 record；
- generator 在调用前后 snapshot 改变 → 拒绝；
- library version / decoding snapshot / assembler version 改变的恢复状态 → 不允许续跑；
- manifest 中的 snapshot、backend、record hash 可独立重算。

### 4.4 不隐藏失败、不选择性重采

- scripted 第一次 action 是 invalid JSON，断 generator 只调用一次 action pass、环境不执行、step 被记录；
- invalid completion shape 产生 schema observation 后可继续下一步；
- tool_error/timeout 是合法 observation，轨迹可继续或按环境规则终止；
- evaluator infrastructure error 不产 `TrajectoryRecord`，不能被转换为 R=0；
- horizon exhaustion 产合法终局 reward（通常 0）与 record；
- 传输层若做确定性 retry，调用 request id/seed/snapshot 完全相同且只提交一个样本（本轮 local 实现可不写 retry）。

### 4.5 bounded agent 权威收口

- 新 BoundedAgent constructor/API 中没有 model、prompt builder、backward scorer；
- rollout active import graph 不含 `runtime.trajectory.TrajectoryStep`、`BackwardTokenScorer`、provider logprobs；
- `runtime.__init__` 不再导出竞争性 trajectory 类型；
- BoundedAgent/environment 的 environment_id、task_family 与 RolloutRequest 一致；valid skill action 的 `invoked_skill_ids` 来自实际 execution；不存在的 skill/tool 得到公开错误而非 Python 路径/堆栈；
- completion validator 只查 shape；用一个带秘密 canary 的 evaluator，断 canary 不出现在任何 generator input、observation 或 rollout event prompt payload 中。

### 4.6 奖励、记录与 Phase 3 咬合

- evaluator 返回裸 R，最终 `shifted_reward == R + epsilon_min`，没有双重平移；
- `RolloutArtifact.initial_context.text` 的 hash/token count 与 record 完全一致；
- artifact 序列化→读回→record/content hash 稳定；
- 从 artifact 取 `record + initial_text`，在测试外部调用 Phase 3 `score_trajectory`，应无 prefix/token mismatch 并得到有限 Δ/loss；注意 production RolloutEngine 自身不调用 scoring；
- TerminalReward.native_payload 只在 artifact/event 的 log-only 区域，下一步 prompt 中不存在。

### 4.7 纯函数/确定性与状态恢复

- seed 派生对同输入稳定，reasoning/action/step 不冲突，不改变全局 RNG；
- initial context 组装对相同输入逐位稳定；retrieved skill 的检索顺序被原样保留（排序可能改变优先级语义），重复 skill id 被拒；
- 同 scripted 输入重复跑出逐字段相同 artifact（时间字段由 fake clock 固定）；
- 完成 step 1 后保存、从同一 snapshot/environment checkpoint 恢复，最终 artifact 与 uninterrupted run 相同；
- snapshot 已更新或环境不可恢复时，旧 state 被拒，不假装续跑。

### 4.8 import、wheel 与回归

- `import skillev.rollout` 在干净子进程中不加载 torch/transformers/peft；
- rollout 包运行时只 import `skillev.policy.interface` 等轻量面；
- policy tokenizer decode / GenerationResult stop-token 加法有 Phase 2 回归测试；
- renderer part-based refactor有 Phase 3 byte-for-byte 回归测试；
- `make check` 的 wheel gate 实际导入 `skillev.rollout`；
- 全部既有 594+ tests 保持通过。

## 5. 工程要求

- 新代码建议放 `src/skillev/rollout/`；模型库重依赖仍只允许在 `src/skillev/policy/`；
- engine 与 environment/generator 接口优先 async，以容纳未来 HTTP/vLLM 和 I/O 环境；单栈 wrapper 内可直接调用同步 backbone；
- canonical text 一律用 Phase 3 renderer / Phase 1 canonical JSON；不要出现第三套模板或 `str(dict)`；
- action exact text 与 normalized `StructuredAction` 必须同时保留各自职责，不互相覆盖；
- 失败异常要带 trajectory/step/stage 定位，但不要求测试精确错误句子；
- state/event 内容必须 canonical，可重放；敏感内容、许可题面、运行日志不得提交 Git；
- 单文件接近 1000 行时拆分；建议按 `contracts/context/generator/codec/environment/engine` 分，但命名自由；
- pyproject 原则上无新依赖；标准库 JSON 与既有 runtime/policy/scoring 足够；
- 不因“将来可能上 vLLM”提前写服务管理、热加载、HTTP retry 或性能 batcher。

## 6. 端到端验证与真底座冒烟

### 6.1 CPU 完整闭环（必须）

用 scripted generator + 两步 deterministic environment 跑完整流程：

- H_0 组装；
- step 1 reasoning/action 生成、工具执行、observation；
- step 2 completion；
- 私有 fake evaluator 给 `TerminalReward`；
- `build_trajectory_record`；
- artifact/event 往返；
- 测试代码在 engine 外调用 `score_trajectory` 验证 Phase 3 可直接消费。

这个测试是本轮主验收，比单独 mock 每个函数更重要；但它不能替代 §4 的负例显微镜测试。

### 6.2 直连 GPU 服务器真底座冒烟（有界）

**GPU 执行硬约束**：只通过 SSH 直连私有配置的 GPU 服务器，不得使用
NCSA Delta、Duo、Slurm 或 scheduler 命令。本轮默认使用物理 H800 GPU `0`
并显式设置 `CUDA_VISIBLE_DEVICES=0`（这是该历史轮次的固定选择，不是当前全局限制）。
当前执行必须先检查全部 8 张卡，优先空闲卡；已有他人进程但 GPU 利用率低且该卡总显存
占用严格低于 25% 的，也允许在剩余显存足够时共用，不得终止、修改或调试他人进程。

在真实 Qwen3.5-9B 单栈上跑一条 1–2 步 public debug rollout：

1. 使用 `LocalPolicyGenerator`，同一 forward adapter snapshot；
2. reasoning/action 两次生成；
3. action 无论 JSON 是否有效，只要 token span exact，就由 debug bounded agent 产生公开 observation；
4. horizon 或 completion 后产生合法 TerminalReward；
5. 得到 RolloutArtifact，再在 engine 外运行一次 Phase 3 score/materialize 验证兼容。

记录：policy snapshot id、两类 token 数、trajectory content hash、终止原因、耗时和显存峰值。**不训练、不更新参数、不接 vLLM、不接正式 benchmark**。若模型输出 action token round-trip 在真 tokenizer 下失败，这正是冒烟要发现的工程问题；报告具体 token/文本现象（不含私有题面）并停下，不得用重新编码替换原 span 糊过去。

## 7. 验收标准

1. `RolloutGenerator`、`LocalPolicyGenerator`、H_0 assembler、两次生成 codec、execution-only `BoundedAgent`、environment/evaluator 边界、RolloutEngine 与 Artifact/Manifest 落地；
2. Phase 1 的 `TrajectoryRecord` 是唯一科学轨迹权威；active rollout path 不再依赖 runtime duplicate trajectory、generation logprob 或 backward scorer；
3. §4 全部测试实现且通过，尤其是：canonical action prefix 同一性、原始 action token span、无隐藏重采、stale snapshot 拒绝、答案 canary 不泄漏、Phase 3 外部消费；
4. tokenizer decode、stop-token 字段、renderer part-based API 的下层加法保持旧输出/旧测试不变；
5. `import skillev.rollout` 源码与 built wheel 均 dependency-light；ruff、mypy、pytest、make check 全绿，GitHub Actions 3.11/3.12/3.13 全绿；
6. CPU 完整闭环通过；直连 GPU 服务器真底座冒烟完成并留 commit-message 级 note，或按 §9 阻塞条款停机；
7. commit + push 到 `YJLi-new/SKILLEV-new` main，简短总结本层产物、测试数与冒烟结果；达到后停止，等 Phase 5 goal。

## 8. 明确不做

- 不写 optimizer、训练循环、batch loss、梯度裁剪、checkpoint 训练步管理或 `TTBBatchStats` 组装；
- RolloutEngine 不调用 Phase 3 scoring；集成测试可在 engine 外调用；
- 不算流诊断、posterior、LCB/UCB、相变或技能进化；
- 不实现 vLLM/HTTP 生产 generator、adapter 热加载、并行 rollout collector、continuous batching；本轮只定 Protocol + fake stale-client 测试；
- 不获取或接入 14 个 benchmark 数据；只使用公开 debug / scripted environment；
- 不把 invalid action 丢弃后重采，不做 rejection sampling；
- 不把 evaluator 私有信息、答案、隐藏测试、native diagnostic 塞入 observation 或 prompt；
- 不修改 `idea.tex`、`AGENTS.md`，不改 Phase 1–3 数学语义；
- 不写审批、密封、attestation 仪式；不启动“审计/评审/找缺口”类子代理。

## 9. 停止条件

- §7 全部验收达成 → 写简短总结，停止，等待 Phase 5 goal；
- 发现 Phase 1 的 action token round-trip 契约与真 tokenizer 生成输出系统性不兼容，且无法在不改语义的情况下解决 → 保存最小复现、列出“扩展契约存 contextual token identity”与“约束 action wire format”等方案后停机请示，禁止静默放宽；
- 发现 Phase 3 renderer 无法在不改变既有 text/hash 的前提下抽出 part-based API → 列出冲突并停机；
- 被权重、私钥、GPU 服务器、私有 evaluator 句柄等必须人类介入的事项阻塞 → 说明缺什么，停止；
- CPU scripted engine 连续 3 次实现尝试仍不能同时满足 on-policy snapshot、exact token span 与 canonical prefix identity → 报告失败路径和最小用例，停止，不用更多代码掩盖接口理解问题。
