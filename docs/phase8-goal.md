# phase8-goal.md —— Bayesian TTB Self-Skill-Revolution Loop：第 8 轮 · 进化层（phase detector + Φ）

> **Protocol-v3 historical amendment (2026-07-26).** 本文件是历史实施记录；当前语义见 `method-v3-final-spec.md` 与 `IDEA_IMPLEMENTATION_SPEC.md`。其中“空 Φ 必须失败”的旧规则已被 Protocol 10 的可审计 `VerifiedNoOpEvolutionDecision` 取代；authoring、evaluator 或基础设施失败仍必须硬失败。Full detector 仍固定使用 confidence-bound Bayesian evidence 与 residual-AND-entropy；没有 config trigger/statistic branch。一次真实 mutation cycle 只有一个 `EvolutionMutation`，library 是 immutable in-memory state。`SKILLEVApplication` 是方法组装入口，新库必须经 full-content retriever 进入下一批 H0。closure 仅允许普通训练，不允许打开新 phase；Split 继续使用 Phase-7 的 `task_family` Context extractor。

本文件是本轮唯一任务清单，接续已验收的第 1–7 轮，是八层导航图的收官轮。写法是讲解式的：每节先说做什么，紧跟着解释为什么、以及做错时的典型症状。若某处字面指令与所附理由看起来冲突，按理由描述的意图行事并留一句 note 说明。

## 0. 定位：八层导航图的第 8 层（收官）

八层规划回顾：(1)–(7) 已完成（基线 = 当前 main，至少含 `e7717f9` 的 Phase 7 校准层）；**(8) 进化层（本轮）**。本轮结束后，idea.tex 的完整自进化循环第一次闭合成一台可运行的机器。

进化层是什么（人话）：两件东西。**相变检测器**盯着训练的体检报表（第 6 层的诊断流）问一个问题："损失平台期到了、且技能使用在坍缩收窄吗？"（式 16 的联合条件）。答案为是的那一刻，**五动作算子 Φ** 上场：按每个技能的流证据（F̂）与后验证据（LCB/UCB）决定保留压缩、修补、分裂、剪除，或从零覆盖的高不对称边生成新技能（式 17）；技能文本的生成与改写由冻结底座完成。动作落库、库版本滚动、Z 重置（`reset_partition()`）、训练继续——一个 self-skill-revolution 周期完成。

**窄接口纪律（Owner 原话，本轮的第一设计约束）**：进化层与训练循环之间只有两个窄接口——**进**：统计流（经 step_observer 收集的第 6 层 BatchDiagnostics）；**出**：技能库变更（经 runtime 的版本化 lineage 机制）+ 调第 5 层的 `reset_partition()`。为什么这么严：Φ 是全系统最"agentic"、最开放的部分——要用模型生成/改写自然语言技能文本、做多准则决策——bug 概率全系统最高。窄接口保证它**再怎么错也污染不了训练数学**：Φ 写出再烂的技能文本，最坏结果是下一段训练的 H_0 里多了一段没用的话，Δ/L_TTB/后验的每一个数仍然是对的；库的版本化 lineage 保证任何动作可回滚。反面症状：如果进化层能摸到 optimizer、scoring 或 rollout 内部，一个 Φ 的 bug 就能把 1–7 层七轮建立的正确性全部拖下水，而且没人能定位。

**技能库存储不属于本层**（Owner 边界）：不可变技能文档/清单/目录复用 `runtime/skills.py` 的现有原语（SkillDocument/SkillMetadata/SkillCatalog/RetrievedSkillContext），版本化库作为 runtime 的最小加法落在 runtime（§2）；本层只做**决策与调用**。

**完整性要求（照 phase 5 的先例）**：本轮交付完全体——不只是检测器和算子，还包括把整台机器箍起来的 **EvolutionLoop 外层驱动**（训练段 → 检测 → 触发 → Φ → 落库 → Z 重置 → 带新库继续），以及让技能真正被用起来的**最小检索器**（新技能必须能进下一段 rollout 的 H_0，否则 Generate 出来的东西永远没人调用，闭环是假的）。

## 1. 方法权威与翻译关系

- **式 (16)** (Δ̄²_{w−W} − Δ̄²_w)/Δ̄²_{w−W} < ρ ∧ H(S)↓：第一合取项按触发窗口内轨迹池化的 **raw `Δ²`** 计算，不使用训练损失 `(Δ/T)²`；第二合取项 **H(S) = 技能使用熵**——滑窗内技能调用分布（per-skill invoking_edge_count 归一）的 Shannon 熵（自然对数）。"↓"的语义由 `EntropyObservation` 与 `PhaseTransitionEvent` 契约钉死：尾部 `required_consecutive_drops` 次严格递减，且只允许持久化 `triggered=True` 的事件。窗口已满但零调用必须产生显式 `0 invocations / 0 skills / 0 entropy` 证据；它不能与“窗口未满”混为一谈。
- **冷启动裁决（本轮最重要的一条读法）**：tex 写的是"evolution boundary is triggered **if and only if**…"。空库（零调用）的熵序列恒为 0，严格递减永不成立 → 按 iff 字面，**空库永远不触发进化，这是方法语义不是 bug**。结论：**实验从非空种子库出发**（式 17 的 S^(k) 迭代以 S^(0) 存在为前提；种子库是实验设置——benchmark 轮人工写两三条通用技能即可），本层**不发明**任何绕过式 (16) 的零库自举逻辑。测试与冒烟一律种子库起步。若 Owner 将来想要零库自举，那是打破 iff 的方法变更，走 §8 请示。
- **式 (17)** S^(k+1) = Φ(S^(k); F̂(s), LCB_{s,z}, log I(t))。窗口级 F̂ 必须从触发窗口的底层 edge population 重算，持久化为单位明确的 `log_skill_marginal_flow`，不得平均每批的 log F̂。五动作判据：Retain&Compress = 高 F̂ ∧ 高 LCB；Refine/Patch = 高流 ∧ 局部情境低后验置信；Split = 高流 ∧ 两个高支持、组内紧、组间可信区间分离的 Context posterior mode；Prune = **低流 ∧ 低 UCB**；Generate = 零技能覆盖 ∧ 在**全窗口 edge population**中处于高 `|log I|` 分位的边。所有动作记录必须通过正式契约。
- **库更新后**：Z_θ 重初始化、训练继续（tex 末句）→ 调第 5 层 `reset_partition()`（参数与优化器动量的原子重置，phase 5 已备好，本轮就是它等了三轮的调用方）。
- **消融开关的消费端**：`CalibrationEngine.enabled=False` 是严格的 SkillFlow 模式，不是异常后 fallback。该分支只允许 flow-only Retain&Compress 与 Generate；Refine、Split、Prune 不产生。`SkillFlowEvidence.calibration_enabled=False` 时 LCB/UCB/k/posterior event IDs 必须全部为空，禁止伪造先验或哨兵值。
- **闭合 Phi**：verified phase 必须生成非空 `EvolutionDecision` 并完整执行全部提案，否则整个 attempt 失败。不存在 executable pending phase、no-action 成功、动作截断、Generate 关闭、authoring 局部跳过或动作子集提交。预算必须在第一次 author call 前为完整 decision 做 preflight。
- **结构与身份**：Retain/Refine/Split/Generate 只接受 action-specific exact JSON，生成结构化 `SkillDocument@2`；任何文本变更产生新 skill_id，旧文档留档停用，新后验从先验开始。authoring prompt/文档只能使用模型可见轨迹与原技能内容，禁止任务真值、答案、native payload 与私有 evaluator 信息。
- **工程默认（全部进入 protocol/config hash）**：高/低 flow 分位为 P75/P25；LCB 高/低为 0.6/0.3；UCB 低为 0.4；Split criterion 为 `min_context_evidence_mass=4.0`、`max_within_context_mean_span=0.20`、`min_between_context_mean_gap=0.40`、`confidence_k=1.0`；技能文本上限 1024 token。full method 的 importance clip、flow-weight cap、gradient clip 均为 `None`，AdamW weight decay 为 `0.0`。检索按 applicability 全量选择相关技能；超 H0 预算失败，不做 top-N 截断。

## 2. 范围刀切与跨层接口

本轮生产边界固定为：

1. **runtime**：`SkillDocument@2` 与版本化 `SkillLibrary`，活动集内容哈希版本和 immutable mutation preview；一次相变只允许一个 `EvolutionMutation`。
2. **policy**：独立的 `generate_base(AuthoringGenerationRequest)`；全部 adapter disabled，只供结构化技能 authoring。训练 rollout 仍只走 raw-softmax `generate_policy(PolicyGenerationRequest)`。
3. **training**：共享 library view、严格 `reset_partition(seed)` 与 completed-boundary exact checkpoint。EvolutionLoop 不能直接访问 optimizer、Z 参数或 scoring。
4. **evolution**：只从 step observer 的 committed `BatchDiagnostics` 与完整 posterior update chain 取证；只输出一个 durable `EvolutionCycleCommit`，再应用 live library/Z projection。
5. **retrieval**：生产唯一入口为 `TaskConditionedSkillRetriever`；以 `SkillApplicability` 匹配 task_family/context/available_tools，无 ID filler、无 silent top-N。
6. **围栏**：evolution 无运行时 torch；authoring 只经 dependency-light policy interface 传 token IDs。

不做：空库自举、embedding/LLM judge 检索、自动 rollback、benchmark 数据接入、多 seed、第二套技能存储或旧 v1 兼容 reader。禁止修改 `idea.tex`、`AGENTS.md`。

## 3. 组件逐个用人话讲（含代码骨架）

骨架规则同前七轮。建议新包 `src/skillev/evolution/`。

### 3.1 使用熵与相变检测器

**是什么**：一个折叠，吃第 6 层 `BatchDiagnostics`，维护同一 `library_version` 的批环形记录。每个已满窗口都产生 `EntropyObservation`；零调用窗也产生明确的 0/0/0 记录。残差停滞从 source residual 的 raw `delta**2` 按轨迹池化重算；不得复用训练损失或 `(delta / horizon)**2`。残差条件与熵严格下降同时成立时，构造只允许 `triggered=True` 的 `PhaseTransitionEvent`。

```python
@dataclass(frozen=True, slots=True)
class EvolutionConfig:
    format: str
    entropy_window: int
    required_consecutive_drops: int
    high_flow_quantile: float
    low_flow_quantile: float
    lcb_high: float
    lcb_low: float
    ucb_low: float
    importance_quantile: float
    max_skill_tokens: int
    maximum_retrieved_skill_tokens: int
    k: float
    split: SplitCriterionConfig


class PhaseTransitionDetector:
    """段内状态 = 批环形记录（id、step、per-skill invoking_edge_count、
    每轨迹 raw Δ²）+ 熵观测序列。observe(diag: BatchDiagnostics) ->
    PhaseTransitionEvent | None：
    1. diag.library_version 变段 → 全部状态清零（窗口与熵序列永不跨库版本）；
    2. 追加批记录；窗满后总是产生 EntropyObservation，包括零使用窗口；
    3. 用相邻窗口的 raw Δ² trajectory pool 计算 stagnation；
    4. stagnation 且 entropy 严格递减 → 构造 PhaseTransitionEvent；
    5. 触发后本段静默，只有库版本变化或显式成功 cycle 才开始新段；
    6. 数值使用 math.fsum/float64，遍历顺序固定。"""
```

配置中不存在 `evolution_check_every`、`max_actions`、`max_generate`、`split_spread` 或 `split_min_cells`。检测器逐 optimizer step 消费诊断，不能粗粒度扫描一段后选择旧触发点。

### 3.2 证据装配：EvidencePack

**是什么**：把 exact triggering window 的底层 edge flow、完整 posterior update chain 与公开轨迹节选整理成不可变快照。决策函数不能边决策边查询 live store。

```python
@dataclass(frozen=True, slots=True)
class WindowSkillFlow:
    skill_id: str
    log_skill_marginal_flow: float | None
    invoking_trajectory_count: int
    invoking_edge_count: int
    invoking_edge_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SkillEvidence:
    skill_id: str
    log_skill_marginal_flow: float | None
    flow_quantile: float
    invoking_trajectory_count: int
    invoking_edge_count: int
    zero_invocation: bool
    posterior_summary: SkillPosteriorSummary | None
    split_modality: SplitModalityEvidence | None
    exemplar_texts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvidencePack:
    phase_event: PhaseTransitionEvent
    skills: tuple[SkillEvidence, ...]
    uncovered_importance_edges: tuple[UncoveredImportanceEdge, ...]
    calibration_enabled: bool


def aggregate_window_skill_flows(
    diagnostics: tuple[BatchDiagnostics, ...],
    *,
    active_skill_ids: tuple[str, ...],
) -> tuple[WindowSkillFlow, ...]:
    """对所有 invoking edges 做 logsumexp，再除以唯一 invoking trajectory 数。
    结果必须对同一 edge population 的 1/2/4 种 batch partition 逐位不变。"""


def build_evidence_pack(...) -> EvidencePack:
    """流证据只来自 aggregate_window_skill_flows；校准证据只来自同一触发窗的
    完整 posterior event chain。Generate 的 P(importance_quantile) 先在全窗口
    edge population 上计算，再筛 invoked_skill_ids 为空的边。校准关闭时不调用
    CalibrationEngine.query，且所有 posterior 字段为空。"""
```

所有 active skill 都必须出现在 `skills`，零调用技能的 log flow 为 `None`、quantile 为 0。相同 observed log flow 使用相同经验分位。调用计数必须与 Phase 6 `SkillFlowStat` 总和交叉一致；缺轨迹、空 exemplar、残缺 posterior chain 或 source/config 不一致都使 evidence assembly 失败，不能填空值继续。

### 3.3 决策矩阵：decide_actions（纯函数）

```python
@dataclass(frozen=True, slots=True)
class ActionProposal:
    action_type: EvolutionActionType
    target_skill_ids: tuple[str, ...]
    evidence: SkillFlowEvidence
    rationale_text: str


@dataclass(frozen=True, slots=True)
class EvolutionDecision:
    phase_event_id: str
    proposals: tuple[ActionProposal, ...]


def decide_actions(pack: EvidencePack, config: EvolutionConfig) -> EvolutionDecision:
    """同 pack/config 必得同输出。
    1. enabled：每技能按 Prune > Split > Refine > Retain&Compress 取至多一动作；
       Split 只能由完整 SplitModalityEvidence 触发。
    2. disabled：只产生 flow-only Retain&Compress 与 Generate，且证据明确
       calibration_enabled=False；绝不查询/伪造 LCB、UCB、k。
    3. uncovered high-|log I| edges 非空时必须产生 Generate。
    4. 提案按固定优先级与 skill_id 排序，但不截断、不关闭 Generate。
    5. 零提案抛 NoApplicableEvolutionActionError，当前 attempt 失败。"""
```

`required_phi_budget(decision, config)` 在 authoring 前计算完整 decision 的模型调用与 token 上界。预算不足抛 `InsufficientEvolutionBudgetError`，author call/document/action/cycle/reset 均为零。

### 3.4 SkillAuthor：agentic 文本操作与泄漏红线

技能文档与模型输出均为结构化 v2 对象；模型不能生成 manifest 的 schema/license 等 trusted metadata。

```python
@dataclass(frozen=True, slots=True)
class SkillApplicability:
    task_families: tuple[str, ...]
    contexts: tuple[str, ...]
    required_tools: tuple[str, ...]
    excluded_contexts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SkillRequirement:
    requirement_id: str
    text: str


@dataclass(frozen=True, slots=True)
class AuthoredSkillDraft:
    title: str
    summary: str
    instructions: str
    applicability: SkillApplicability
    requirements: tuple[SkillRequirement, ...]


@dataclass(frozen=True, slots=True)
class AuthoringRequest:
    action_type: EvolutionActionType
    target_documents: tuple[SkillDocument, ...]
    exemplar_texts: tuple[str, ...]
    evidence_summary: str
    target_context_keys: tuple[str, ...]
    target_edge_ids: tuple[str, ...]
    split_modality: SplitModalityEvidence | None
    generate_authority: GenerateAuthoritySelection | None
    seed: int


@dataclass(frozen=True, slots=True)
class AuthoringResult:
    drafts: tuple[AuthoredSkillDraft, ...]


class SkillAuthor(Protocol):
    def author(self, request: AuthoringRequest) -> AuthoringResult: ...


class BaseModelSkillAuthor:
    """按动作选择版本化模板，调用 generate_base(AuthoringGenerationRequest)，
    只接受 exact JSON object；缺/多字段、markdown fence、纯文本、空/超长输出均失败。
    无 retry、无截断、无从首行推断 title/summary、无默认 schema/license。"""
```

四个 validator 是 action-specific 的：

- Retain/Compress：applicability 与 requirement IDs 精确保留；以 pinned tokenizer
  计算完整 canonical model-visible skill content（title/summary/instructions/
  applicability/requirements），总 token 数必须严格减少。已达最小合法表示的
  source 在生成前判为不可压缩并使 closed attempt 失败，不做 no-op，也不允许把
  内容挪入未计数字段；
- Retain prompt 必须显式携带 tokenizer identity、source 完整 canonical
  model-visible token 数、严格的 source-minus-one 输出上限、完整 applicability
  与 requirement IDs。生成前使用固定 canonical 最小合法 shape 判定可压缩性；
  生成时只调用一次 frozen base，validator 独立重算，不截断、不重试、不从多候选选择；
- Refine：applicability 保持，全部旧 requirement 保留，并为每个目标 context 增加确定性 patch requirement；
- Split：两个 draft 的 contexts 精确对应两个 posterior modes、互斥且在 source applicability 内；二者保留全部旧 requirements，并各增加对应 mode requirement；
- Generate：task family/context/tools 精确来自 `GenerateAuthoritySelection`，每个 uncovered edge 都有显式 coverage requirement。

`SkillAuthoringAuthority` 是 Generate schema/license/task/tool 的唯一来源；没有唯一 authority selection 必须在第一次 author call 前失败。native payload canary 不得进入 request、prompt、draft、document 或 event。

### 3.5 版本化技能库（runtime 最小加法）与执行器

runtime `SkillLibrary` 是结构化 v2 文档目录、活动集内容哈希版本和单次 immutable state replacement 的唯一权威。旧 v1 文档/事件不兼容读取；没有自动 metadata migration、live rollback 或 event replay。

```python
def prepare_actions(
    decision: EvolutionDecision,
    *,
    author: SkillAuthor,
    library: SkillLibrary,
    phase_event: PhaseTransitionEvent,
    config: EvolutionConfig,
    authority: SkillAuthoringAuthority,
) -> PreparedEvolutionActions:
    """先为全部提案完成 authority selection、一次 authoring、action-specific validation、
    新 document/action record 构造与 mutation preview；live library 与 event log 不变。
    返回 records、documents 与唯一 LibraryMutationPlan。无 None/空成功。"""
```

`prepare_actions()` 结束时必须证明 proposal/record 数量相等、所有 produced IDs 对应 plan documents、preview version 与旧 version 不同。`execute_actions()` 这类逐动作直接 commit 的生产捷径删除。

`EVOLUTION_PHASE_OPENED` 与 `EVOLUTION_CYCLE_COMMITTED` 是 attempt-private scientific sources；只有整个 child attempt 成功发布后才进入审计权威。cycle 内容绑定完整 `EvolutionMutation`、partition-reset seed 与实际 authoring usage。任何构造、reset、apply 或 source append 失败都作废整个 attempt；没有 `from_events()` live recovery，也不能 authoring 失败后保留前缀动作。

### 3.6 检索器：让技能进下一段 H_0

```python
class TaskConditionedSkillRetriever:
    """从 RolloutTask 的显式 task_family/context/available_tools 构造 features，
    与每个 SkillApplicability 做确定性匹配。全部匹配项进入 H0；排序只用于稳定
    展示，不用于截断。无匹配合法返回空；总 token 超预算抛
    SkillRetrievalBudgetExceededError。"""
```

匹配规则：excluded context 优先排除；task family 与 context 必须分别匹配；required tools 必须全部可用。通配只能由结构化 applicability 中显式 `"*"` 表达。`RetrievingRolloutSessionFactory` 要求 base factory 不得再注入第二个 skill source。生产路径不得导出/构造 `AllActiveSkillsRetriever`。

### 3.7 EvolutionLoop：完全体外层驱动（收官件）

```python
class EvolutionLoop:
    """薄外层：逐步 train→detect→complete Phi→cycle→continue。"""

    async def run(self, num_iterations: int) -> EvolutionRunSummary:
        """顺序不可调：
        1. 每次只调用 training_loop.run(1)，必须恰推进一个 optimizer step；
        2. 消费同一步已提交 diagnostic；
        3. 触发时 build evidence → nonempty decision → full budget preflight
           → prepare all actions/mutation/reset projections；
        4. 执行 exact reset 与 immutable library apply，写 phase/cycle source；
        5. 成功后一次发布 detector/projection reset，新 library 自动进入下一批；
           任一步失败都终止 attempt，不产生可发布的 partial success。
        """
```

成功 attempt 之后，`skillev.audit` 只从 published bundle 的四类 source event 离线重算 phase、mutation、library lineage 与 Z-reset identity。该 reducer 不是 live recovery input。

### 3.8 测试策略（本轮的显微镜清单）

全部 CPU、无网络：tiny backbone、scripted env/author、种子库（手工 2–3 条技能）。清单（每条都是行为断言）：

- **熵与检测器**：均匀/集中/零使用熵手算；窗口未满与已满零使用区分；raw `Δ²` 与 `(Δ/T)²` 反例；连降、严格阈值、换库清零、段内静默。
- **窗口 F̂**：同一 edges 分成 1/2/4 个 batch 输出逐位相同；同轨迹多 edge 分子全计、分母只计一次；wire 只出现 `log_skill_marginal_flow`；调用计数交叉校验。
- **Generate 分位**：阈值从全窗口非终局 edge population 计算后再筛零覆盖边；低不对称 uncovered 子集不能仅因内部 P90 自动产生 Generate；终局 COMPLETE/submission edge 不进入阈值 population 或 Generate candidate pool，真实 TOOL edge 保持可用。
- **决策矩阵**：五动作逐一命中；Split 的低支持、组内散、interval overlap 负例；不截断；Generate 不可关闭；零提案抛 typed error并使 attempt 失败；完整 budget 不足时零 author call。
- **校准关闭**：只可能 Retain/Generate；所有 action evidence 无 LCB/UCB/k/posterior IDs；不得调用 `CalibrationEngine.query()`。
- **结构化 authoring**：exact JSON；四类 validator 好/坏成对；缺 trusted authority 在 author 前失败；第 N 个失败导致 0 document/action/cycle/reset；无 retry/截断/metadata 推断；native payload canary 全链路不可见。
- **库与 cycle**：所有动作一次 mutation/cycle；任一故障点使 bundle quarantine；successful published bundle 可离线重建相同 library/Z identity；v1 document/event拒绝。
- **exact checkpoint**：只接受 completed-boundary exact snapshot；wrong directory/schema/arm 被拒，禁止 latest fallback、live task rewind 与 open-phase recovery。
- **retrieval**：applicability/tool/exclusion/no-match/ID rename；超预算失败；新技能下一段进入 H0；base factory 不得注入第二 skill source。
- **闭环 e2e**：真实生产 `TrainingLoop.run(1)`、diagnostics/calibration、detector、structured scripted author、cycle、partition reset、task-conditioned retrieval 与下一训练步；事件 source 顺序、Z/adapter optimizer state、新版本、新 H0 与离线 audit 一致。
- **无扰与围栏**：永不触发时 EvolutionLoop 与裸 TrainingLoop 参数逐位相同；evolution 无运行时 torch；不 import scoring/rollout internals。

## 4. 工程要求

- 新包建议 `src/skillev/evolution/`（+ runtime 的 skill_library 模块）；对 1–7 层只 import 公开导出面；无运行时 torch。
- 一切遍历与输出顺序显式固定；id/哈希用第 1 层 stable_hash/identity；事件载荷 canonical；运行时 authored skill 与题目轨迹只进本地私有 artifact/event，不进 Git；单文件约 1000 行内。
- `SKILLEV_EVOLUTION_FORMAT`、`SkillDocument`、phase-open/cycle/checkpoint/retrieval payload 均使用 v2 exact fields；不兼容读取 v1。
- 开发中只跑相关测试；push 前运行一次本地 `make check`。本轮不运行或修 GitHub CI。
- pyproject 无新依赖。

## 5. Gate 4a / Gate 4b 分拆验收

**GPU 执行硬约束**：只通过 SSH 直连私有配置的 GPU 服务器，不得使用
NCSA Delta、Duo、Slurm 或 scheduler 命令。Gate 4a 是单卡任务，可从两个批准端点的
全部物理 H800 GPU `0`–`7` 中选择，不限制卡号；启动前检查全部卡，优先空闲卡；已有他人
进程但 GPU 利用率低且该卡总显存占用严格低于 25% 的，也允许在剩余显存足够时共用。
显式设置 `CUDA_VISIBLE_DEVICES` 为选中的物理编号，不得终止、修改或调试他人进程。Gate 4b 为 tiny 闭环，不得占用 GPU。

**Gate 4a（有界真底座证据，`@4`）**：直连 GPU 服务器单卡、真 Qwen3.5-9B，以 protocol 固定的一组 seed 证明真实 generation、真实 tokenizer 上生成 token 的 exact decode/encode round-trip、teacher-forced forward/backward score 有限且 forward adapter / backward adapter / Z head 三路梯度存在、一次 optimizer step、checkpoint save/load 逐位一致，以及冻结底座 `generate_base` 对 Retain、Refine、Split、Generate 四种 method-faithful structured authoring request 各执行一次。四次必须使用独立 reservation、全部 settle 且全部通过结构和语义校验，包括每份 authored document 的完整 canonical H₀ skill block（position/id/version/content_hash/content 标签与换行）在 reachable positions 1--5 上均不超过 3,400 tokens，Gate 才通过；任一生成前 fixture rejection 是 Gate 实现失败，任一生成后 rejection 会保留私有 cause-chain 与 token/output hash 且 Gate 不通过，不重试也不 mutation。rollout action 无 exactness 要求：接受 policy 实际输出并记录 tool/skill/complete/parse_error/schema_invalid 分类；禁止 seed search。Gate 在模型加载前绑定 exact base-model artifact bytes，并证明 executing source tree/commit 与 immutable source archive 一致；同时记录 tokenizer/trainable-state/hardware identity、峰值显存与耗时。该 gate 是非 benchmark correctness evidence，不执行完整 Φ cycle。`@2` 已消费且因 Retain 未严格压缩而失败；`@3` 在新增完整 H₀ block validator 前通过但不再用于后续 protocol freeze，两者均不得重跑。

Gate 4a `@4` 前先固定公开合成 Retain development suite 与完全不相交的
untouched holdout suite。两套都覆盖短/中/长 instructions、少/多 requirements、
单/多 task family、有/无 tool、高重复、结构清晰低冗余及 near-minimal legal
source。development 只用于冻结 template 与 `AuthoringSamplingConfig`；冻结后
holdout 每 case 仅执行一次、固定 seed、零 retry，任一 rejection 阻止 Gate 与
protocol freeze。随后 Gate 4a `@4` 使用新的 commit、source package 和格式只执行
一次，要求四类 authoring 全部 validated、四次 model call/reservation 全部 settled、
`gate_passed=true` 且进程退出码为 0。

**Gate 4b（tiny 完整闭环）**：tiny backbone 运行真实 TrainingLoop diagnostics 与真实 PhaseTransitionDetector，完成 detector→Φ→library mutation→Z reset→new-library continuation 的完整闭环；`c6da682` 的生产 composition 行为测试为本 gate 的既有证据。Gate 4b 不用 scripted detector 替换触发器。完整闭环的结构正确性由 Gate 4b 证明，不再要求真 9B policy 命中一个预声明 exact action。

## 6. 验收标准

1. PhaseTransitionDetector、窗口 evidence、closed `EvolutionDecision`、structured SkillAuthor、SkillLibrary、`prepare_actions`、TaskConditionedSkillRetriever、exact-recovery EvolutionLoop 全部落地；§3.8 全绿。
2. full identity 为 policy 1/1、importance clip/flow cap/gradient clip `None`、weight decay `0.0`；一次 phase 全部动作或 attempt failure。
3. 相关 ruff/mypy/pytest 与最终本地 `make check` 全绿；不运行 GitHub CI。
4. GPU 冒烟完成并留 note（或按 §5 阻塞条款停机说明）。
5. v2 phase-goal/README/benchmark guide/protocol freeze/handoff 文档与代码一致；旧 v1 active identity 作废且零结果 ingest。
6. commit + push 到 SKILLEV-new；简短总结后停机。正式 primary benchmark 仍不在本轮启动。

## 7. 明确不做

- 不做零库自举、embedding/LLM judge 检索、技能质量评估模型、人在环审批或自动 rollback。
- 不动 Δ/L_TTB/后验数学；EvolutionLoop 不碰 optimizer/scoring/rollout 内部。
- 不接 benchmark 数据（14 benchmark 接入按 `docs/benchmark-evaluation-guide.md` 另开轮次）；不做多 seed、不并行、不 vLLM。
- 不保留 no-action、max-actions、max-generate、latest checkpoint、free-text authoring、all-active top-N 或 v1 reader。
- 不改 idea.tex、AGENTS.md；不写审批/密封仪式；不启动"审计/评审/找缺口"类子代理。

## 8. 停止条件

- §6 验收达成 → 简短总结，停机。八层完工，等 benchmark 轮 goal。
- 被必须人类介入的事情阻塞（权重、私钥、GPU 服务器）→ 说明缺什么，停机。
- 闭环 e2e 3 次实现尝试仍打不通 → 停机报告（先自查：是否消费旧段诊断、library provider/retriever 是否真实接线、failed attempt 是否误入 published source）。
- 只有发现 `idea.tex` 与本文件无法双向兼容的科学语义冲突时才停机；普通结构迁移、format bump、测试迁移由实现者直接完成。
