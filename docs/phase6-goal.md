# phase6-goal.md —— Bayesian TTB Self-Skill-Revolution Loop：第 6 轮 · 流诊断层（flow diagnostics）

> **Protocol-v3 immutable amendment (2026-07-25).** 本文件是历史实施记录；当前语义见 `method-v3-final-spec.md`。Full core 直接累计 raw `EdgeLogprobRecord.step_importance`，无 importance clip；停滞窗口池化 raw `Δ²`，不使用训练损失 `(Δ/T)²`。在线层只有 immutable preview/commit，不发派生 scientific event。全部 event-log replay/recompute 已迁到单向离线 `skillev.audit`，active diagnostics/evolution/training 不 import audit，也不提供 replay/from-events 恢复。

本文件是本轮唯一任务清单，接续已验收的第 1–5 轮。写法是讲解式的：每节先说做什么，紧跟着解释为什么、以及做错时的典型症状。若某处字面指令与所附理由看起来冲突，按理由描述的意图行事并留一句 note 说明。

## 0. 定位：八层导航图中的第 6 层

八层规划回顾：(1) 轨迹数据层（已完成）；(2) 模型层（已完成）；(3) 打分与目标层（已完成）；(4) rollout 引擎（已完成）；(5) 训练循环（已完成——`src/skillev/training/`，TrainingLoop + 事件落账）；**(6) 流诊断层（本轮）**；(7) Bayesian 校准；(8) 相变与进化。本轮只做第 6 层。开工基线 = 当前 main（至少含 `0758da6` 的 Phase 5 训练循环）。

流诊断层是什么（人话）：又一组**纯函数**，吃第 5 层落账的记录流（EdgeLogprobRecord + TTBBatchStats + 轨迹记录），产出三种诊断量——每条边的 log I(t)（局部流不对称）、每条轨迹逐步累积的状态流 F(H_t)（log 域 running product + 裁剪）、每批每技能的边际流 F̂(s)——外加式 (16) 相变判据的第一合取项：按 library_version 分段的滑窗 Δ̄² 停滞探针。它不碰模型、不碰张量、不做任何决策；它算出来的数是第 7 层后验更新的权重（w_e ∝ F(H_t)）和第 8 层进化算子 Φ 的输入（F̂(s)、log I(t)、停滞信号）。

**本轮的关键设计约束（Owner 原话，一开始就要做完整）**：这套诊断必须**既能在线挂在训练循环里跑，也能离线从 event_log 重放，两条路跑出一模一样的数**——逐位一致，不是"差不多"。这是审计要求（任何一张诊断曲线都能从原始记录复算，出了怪数不用猜是训练的锅还是诊断的锅），也是省钱利器（第 8 条的伏笔）：诊断超参的消融——换个裁剪阈值、换个窗宽 W、换个 ρ——从此全部等于**对已有 event log 重放一遍，0 GPU 时间**。做到这一点的机制只有一条：诊断的一切输入都来自**已物化的契约记录（canonical float）**，绝不碰张量、绝不碰只存在于内存的中间量；在线路径喂的对象和离线路径从事件里 from_value 重建的对象是同一批 float，等价性由构造保证、由测试逐位验收。

## 1. 方法权威与翻译关系

本轮把 idea.tex §2.4（Distributional Flow Credit Diagnostics）的三个式子和 §2.5.1 的半个式子翻译成纯函数：

- **式 (9)** log I(t) = P̂_F^t − P̂_B^t："局部 log 密度比"。它**就是** EdgeLogprobRecord.step_importance——第 1 层契约已经硬校验过 step_importance == forward − backward，本层直接取用该字段，不再自己减一遍（第 1 层验、第 6 层用，分工故意如此）。
- **式 (10)** F(H_t) = E_{τ∼π_θ}[exp(Σ_{t'≤t} log I(t')) | H_t]：状态流是"分布式估计量而非守恒标量场"。在我们的文本轨迹上，每个 H_t 只被它自己那条采样轨迹经过（文本历史几乎不可能碰撞），条件期望退化为该轨迹自身的 running product——log 域实现就是**逐步累加 log I 的前缀和**。这正是"实现是逐条轨迹独立累加"的方法学依据，不是偷懒。
- **裁剪是显式消融轴**：`importance_clip=None` 是 full method，逐边 log I 原样进入前缀和；`importance_clip=10.0` 只属于 `clipped-importance` arm。无论何种配置都记录 `clip_engaged`，且不得遇到大值后自动开启裁剪。
- **式 (11)** F̂(s) = E_{τ∼B_s}[Σ_{t: a_t invokes s} F(H_t)]：两级结构——**轨迹内**对所有调用了 s 的步求 F 之和，**批内**对 B_s 中的轨迹取均值。B_s 的读法钉死为：**本批中至少调用过一次 s 的轨迹集合**（分母 = |B_s|；不调用 s 的轨迹不进分母——否则 F̂ 会随批内无关轨迹数被稀释，"absolute credit magnitude"就不再绝对）。数值上用 logsumexp 在 log 域聚合：log F̂(s) = logsumexp(所有调用边的 log F) − log|B_s|。
- **式 (16) 第一合取项** (Δ̄²_{w−W} − Δ̄²_w)/Δ̄²_{w−W} < ρ：滑窗停滞。Δ̄²_w 是窗口内全部轨迹按轨迹池化的 **raw Δ²** 均值。训练目标仍为式 (8) 的 `(Δ/T)²`；诊断不能复用训练 loss。第二合取项 H(S)↓ 属第 8 层。

**钉死的**：诊断只消费已落账契约 float（在线离线同源）；滑窗按 library_version 分段；派生诊断永不作为 replay authority。full 显式 `importance_clip=None`，clipped ablation 为 10.0；W=50、ρ=0.05；停滞使用严格小于。前窗均值为 0 时相对改善无定义，probe 为 `None`，不得合成停滞证据。

## 2. 范围刀切与最小加法

- 本轮允许**两个最小加法**，除此之外第 1–5 层冻结：
  1. **TrainingLoop 的可选观察者钩子**：`TrainingLoop` 构造参数增加 `step_observer`（默认 None，一切既有行为与测试不变）。train_step 完成落账后，以一个小捆绑对象（本轮定义的 `TrainingStepObservation`：本批 artifacts + TTBBatchStats + 全部 EdgeLogprobRecord——**与发射进事件的是同一批对象**）调用观察者。观察者是只读消费者：加不加它，训练的参数轨迹必须逐位相同（有测试）。
  2. **EventType 增一个成员** `FLOW_DIAGNOSTICS_RECORDED`：派生诊断的便利落账，仅供人看曲线；重放路径无条件忽略它。
- 围栏零改动：本层完全无 torch（连 TYPE_CHECKING 都不需要）——现行边界测试只豁免 policy 与 training 两包，诊断包一旦手滑 import torch 会被现有扫描器自动抓住，正好。
- 不做第 7/8 层：无 z 特征分桶（Context/Failure Mode/Token Bucket/Horizon Bucket 是第 7 层的键）、无 Beta-Bernoulli/LCB/UCB、无库熵、无相变触发、无 Φ。
- 不做诊断状态的持久化/断点续算（重放本来就廉价，状态落盘只会造出第二份真相）；不做 numpy/pandas 依赖（logsumexp 和 fsum 用 math 手写，几行的事）；不做大 CLI（函数级 API 为主，简单 `__main__` 入口自由）。

## 3. 组件逐个用人话讲（含代码骨架）

骨架规则同前五轮：字段语义与不变量不可减，命名与文件组织自由；注释说"需要写"的就是实现规格。建议新包 `src/skillev/diagnostics/`。

### 3.1 配置与输入组装：两条路共用同一个装配函数

**是什么**：诊断的旋钮（DiagnosticsConfig）和批输入（BatchFlowInput），以及**在线与离线共用的唯一装配函数**。**为什么装配必须单点**：在线喂内存对象、离线喂 from_value 重建对象，如果两条路各写各的 join，等价性就要靠运气；共用一个装配函数后，等价性由构造保证，测试只是复核。

```python
@dataclass(frozen=True, slots=True)
class DiagnosticsConfig:
    """全部诊断旋钮。它的快照进每份输出——重放消融时两份结果必须能自报家门。"""

    importance_clip: float = 10.0   # c：|log I| 逐步裁剪上限（nats）；> 0 有限
    window_size: int = 50           # W：滑窗宽度（单位=批）；> 0
    stagnation_rho: float = 0.05    # ρ；(0, 1) 内有限

    def to_value(self) -> dict[str, JsonValue]: ...   # 需要写：canonical 快照


@dataclass(frozen=True, slots=True)
class TrajectoryFlowInput:
    """一条轨迹进诊断所需的最小 join：边记录 + 每步调用的技能。"""

    trajectory_id: str
    horizon: int
    edges: tuple[EdgeLogprobRecord, ...]                     # step_index 恰为 1..horizon、有序无缺
    invoked_skill_ids_by_step: tuple[tuple[str, ...], ...]   # 与 edges 逐位对齐；来自 TrajectoryStep

    def __post_init__(self) -> None:
        """需要写：edges 连续且 trajectory_id 全部一致；两个 tuple 长度 == horizon；
        错误消息带 trajectory_id 与 step_index。"""


@dataclass(frozen=True, slots=True)
class BatchFlowInput:
    """一个 optimizer 批的诊断输入。"""

    stats: TTBBatchStats
    trajectories: tuple[TrajectoryFlowInput, ...]

    def __post_init__(self) -> None:
        """需要写：trajectories 的 id 集合恰等于 stats.residuals 的 id 集合（不多不少）；
        每条轨迹的 horizon 与对应 residual.horizon 一致。"""


def assemble_batch_flow_input(
    stats: TTBBatchStats,
    records_by_id: Mapping[str, TrajectoryRecord],
    edges: tuple[EdgeLogprobRecord, ...],
) -> BatchFlowInput:
    """需要写：按 stats.residuals 的 trajectory_id 顺序（这就是全层统一的确定性顺序），
    从 records 取 horizon 与逐步 invoked_skill_ids、从 edges 按 (trajectory_id, step_index)
    归组排序，拼 TrajectoryFlowInput。缺轨迹、缺边、多余边 → ValueError 指明位置。
    在线与离线**都必须**经过本函数——这是等价性的机制本体。"""
```

### 3.2 单轨迹流：log I 与 F(H_t) 的 running product

**是什么**：式 (9)(10) 的实现——每条边取 step_importance、裁剪、前缀和。**为什么逐轨迹独立**：§1 讲过，式 (10) 的集合期望在文本轨迹上退化为单轨迹 running product；跨轨迹不存在任何耦合，这个函数就该是"tuple 进 tuple 出"的最傻形态。

```python
@dataclass(frozen=True, slots=True)
class EdgeFlowDiagnostic:
    """一条边的全部流诊断量。log 域存储；线性域由消费方自行 exp。"""

    trajectory_id: str
    step_index: int
    log_importance: float          # == 边记录的 step_importance（式 9；第 1 层已验自洽）
    clipped_log_importance: float  # 裁剪后用于累加的值
    clip_engaged: bool             # 本边是否触发了裁剪——裁剪率是重要的消融观测量
    log_state_flow: float          # log F(H_t) = Σ_{s≤t} clipped_log_importance（式 10）
    invoked_skill_ids: tuple[str, ...]   # 透传给 F̂ 与第 7 层


@dataclass(frozen=True, slots=True)
class TrajectoryFlowDiagnostic:
    trajectory_id: str
    horizon: int
    edges: tuple[EdgeFlowDiagnostic, ...]
    terminal_log_state_flow: float       # log F(H_T)；第 7 层轨迹级权重的现成量


def trajectory_flow(
    trajectory: TrajectoryFlowInput, config: DiagnosticsConfig
) -> TrajectoryFlowDiagnostic:
    """需要写：按 step_index 升序，log_importance 直接取边记录 step_importance；
    clipped = min(max(log_importance, -c), +c)；log_state_flow 为 clipped 的前缀和
    （逐步累加，float 运算顺序固定=步序，保证任何两次运行逐位一致）。纯函数：
    不改输入、无 RNG、无 I/O。"""
```

**典型症状**：忘了用裁剪值累加（裁了个寂寞）——单条病态边让某技能的 F̂ 冲到 e^80，第 8 层的 Retain&Compress 排序被一条轨迹绑架；或者在线性域累乘——三十步后 underflow 成 0.0，所有技能的边际流并列零，Φ 退化成掷骰子。

### 3.3 技能边际流：式 (11) 的两级聚合

**是什么**：对一个批，把每个技能在"调用它的边"上收到的 F 聚起来。**为什么在 log 域做**：F 是 exp 量，线性求和前先 logsumexp，数值稳且顺序可控。

```python
@dataclass(frozen=True, slots=True)
class SkillFlowStat:
    """一个批里一个技能的边际流（式 11）。"""

    skill_id: str
    invoking_trajectory_count: int   # |B_s|：批内至少调用过 s 一次的轨迹数；≥ 1
    invoking_edge_count: int         # 全批调用 s 的边数；≥ invoking_trajectory_count 不必，≥1 必
    log_skill_flow: float            # log F̂(s) = logsumexp(调用边的 log_state_flow) − log|B_s|


def skill_marginal_flows(
    trajectories: tuple[TrajectoryFlowDiagnostic, ...],
) -> tuple[SkillFlowStat, ...]:
    """需要写：收集全批 (skill_id, 调用边的 log_state_flow)；每技能按
    (trajectory_id, step_index) 排序后 logsumexp（手写：m + log(fsum(exp(x−m)))，
    m 取最大值），减 log|B_s|。返回按 skill_id 字典序排序的 tuple——
    输出顺序也是确定性契约的一部分。批内无任何技能调用 → 返回 ()。"""
```

### 3.4 滑窗停滞探针：式 (16) 第一合取项

**是什么**：维护按 library_version 分段的逐批 raw Δ² 序列，产出"当前 W 窗 vs 前一 W 窗"的相对改善。

```python
@dataclass(frozen=True, slots=True)
class StagnationProbe:
    """有足够数据时的一次停滞探测（式 16 左合取项的完整证据）。"""

    library_version: str
    window_size: int
    previous_window_mean_delta_squared: float
    current_window_mean_delta_squared: float
    relative_improvement: float
    stagnant: bool


@dataclass(frozen=True, slots=True)
class DiagnosticsState:
    """不可变折叠状态：当前库版本段内，逐批的按轨迹 raw Δ² 池化组。"""

    library_version: str | None = None
    per_batch_delta_squared: tuple[tuple[float, ...], ...] = ()

    # 需要写：observe 时 library_version 变化 → 丢弃序列从零开始（分段语义）。
    # 均值一律 math.fsum 池化（先展平窗口内全部轨迹再除以条数），不做"批均值的均值"。
```

### 3.5 批诊断与折叠入口：observe_batch

**是什么**：单批的总装——轨迹流、技能流、停滞探针、配置快照，合成一份 BatchDiagnostics；同时推进折叠状态。**为什么输出携带配置快照**：重放消融会对同一份 log 跑出多套结果，没有自报家门的输出等于没做消融。

```python
@dataclass(frozen=True, slots=True)
class BatchDiagnostics:
    """一个 optimizer 批的完整流诊断。派生数据：可落账、不可回读。"""

    batch_id: str
    optimizer_step: int
    library_version: str
    trajectories: tuple[TrajectoryFlowDiagnostic, ...]
    skill_flows: tuple[SkillFlowStat, ...]
    stagnation: StagnationProbe | None    # 段内不足 2W 批时为 None
    config: DiagnosticsConfig

    def to_value(self) -> dict[str, JsonValue]:
        """需要写：canonical 序列化，进 FLOW_DIAGNOSTICS_RECORDED 事件与等价性测试。
        **故意不提供 from_value**：派生数据永不回读，防循环 sourcing——这行注释原样保留。"""


def observe_batch(
    state: DiagnosticsState,
    batch: BatchFlowInput,
    config: DiagnosticsConfig,
) -> tuple[DiagnosticsState, BatchDiagnostics]:
    """需要写（按序）：1) library_version 换段判断；2) 逐轨迹 trajectory_flow；
    3) skill_marginal_flows；4) 本批 raw Δ² 组取自 stats.residuals（r.delta**2，
    float64——与契约同源，不从 tensor loss 取数）；5) 追加序列，段内 ≥ 2W 批则算 StagnationProbe；
    6) 返回新状态与诊断。纯函数：同 (state, batch, config) 必得逐位相同的输出。"""
```

### 3.6 在线挂载：观察者与派生落账

**是什么**：把折叠挂进训练循环的薄适配器 + §2 第 1 项的 TrainingLoop 加法。**为什么观察者必须只读**：诊断挂上去之后训练数值有任何变化，等价性口号就成了笑话——所以有一条专门的回归测试：带与不带观察者，训练完的参数逐位相同。

```python
@dataclass(frozen=True, slots=True)
class TrainingStepObservation:
    """train_step 落账后交给观察者的捆绑——与写进事件的是同一批契约对象。"""

    batch: tuple[RolloutArtifact, ...]
    stats: TTBBatchStats
    edge_records: tuple[EdgeLogprobRecord, ...]


class OnlineFlowDiagnostics:
    """需要写：持有 DiagnosticsConfig 与折叠状态；作为 step_observer 挂入 TrainingLoop。
    每次被调：用 assemble_batch_flow_input（records 取自 observation.batch 的
    artifact.record）组装 → observe_batch → 结果存入自身列表；若构造时给了 emitter，
    以 FLOW_DIAGNOSTICS_RECORDED 落账 to_value。它不 raise 进训练路径的业务异常之外的东西、
    不改任何输入、不碰模型。"""
```

### 3.7 离线重放：从 event_log 跑出同一份数

**是什么**：读事件流、重建契约对象、走同一条装配与折叠。**批边界闸门按第 5 层的真实发射顺序设计**（这是实现细节但必须写死）：train_step 的落账顺序是 TTB_BATCH_STATS_RECORDED → 逐条 EDGE_LOGPROB_RECORDED → TRAINING_STEP_COMMITTED，而 ROLLOUT_ARTIFACT_RECORDED 在采集期早已出现。所以重放的状态机是：持续缓存 artifact（按 trajectory_id 索引）与 edge（按 (trajectory_id, step_index) 索引）、暂存最近的 stats；**见到 TRAINING_STEP_COMMITTED 才关批**（以其 batch_id 与暂存 stats 对账），此刻按 stats.residuals 从缓存取货装配。凡与本层无关的事件类型（rollout/budget/agent/report/派生诊断……）一律跳过——尤其 FLOW_DIAGNOSTICS_RECORDED **必须**跳过。

```python
def replay_flow_diagnostics(
    events: Iterable[Mapping[str, JsonValue]],    # 事件封套流；读取方式复用 runtime 现有
    config: DiagnosticsConfig,                    # event_log 读取路径（phase-1 e2e 重放已有先例）
) -> tuple[BatchDiagnostics, ...]:
    """需要写：上述状态机；契约对象一律走 from_value 正门重建（canonical float
    往返精确，等价性由此成立）；关批时缺 artifact/缺边/batch_id 对不上 → ValueError
    指明批与轨迹；事件流结束时有暂存 stats 未关批 → 同样报错（半截批不静默丢弃）。
    对同一份 log、同一 config，任意两次重放输出逐位相同。"""
```

**这就是省钱利器的兑现方式**：想看 c=5 和 c=10 的 F̂ 排序差异？想把 W 从 50 调到 30 看停滞探针提前多少批？全部是对既有 log 的重放，一秒级、零 GPU。第 8 层将来做进化决策的消融，靠的就是这条路。

### 3.8 测试策略（本轮的显微镜清单）

全部 CPU、无网络、无 torch。契约对象用第 1 层正门构造。清单（每条都是行为断言）：

- **手算对照（主镜）**：手工构造 2 条轨迹（含一条多步多技能）的边记录，纸面算出每步 log I、裁剪、前缀和、logsumexp 的 F̂、窗口均值，与实现输出 rtol=0, atol=0 逐位对照——测试注释里写出手算过程。
- **裁剪**：|log I| > c 的边 clip_engaged 为 True 且累加用 ±c；c 改变 → 下游 log_state_flow/F̂ 随之变（旋钮真的接了线）；全部边都在 [−c,c] 内时输出与不裁剪逐位相同。
- **F̂ 语义**：不调用 s 的轨迹不进 |B_s| 分母；同一轨迹多次调用 s 的每条边都计入分子；技能按字典序输出；无调用 → ()；单边单轨迹的 F̂ 恰等于该边 F。
- **停滞探针**：脚本化损失序列手算窗口均值与相对改善；改善恰等于 ρ → 不停滞（严格小于）；段内不足 2W 批 → None；**library_version 中途切换 → 序列清零重来**（切换后再次凑满 2W 才有探针）——本层最重要的分段测试；前窗均值为 0 → stagnant True。
- **在线 ≡ 离线（头条测试）**：用 phase-5 的 scripted 组件整跑一个小训练（≥3 个 optimizer 批，中途含被拒 rollout），挂 OnlineFlowDiagnostics + 捕获全部事件；再对捕获的事件流 replay；两边 BatchDiagnostics.to_value() 序列**逐位相等**。
- **重放稳健性**：重放两次逐位相同；事件流里混入无关事件与"内容故意错误的 FLOW_DIAGNOSTICS_RECORDED"→ 重放结果不受影响（证明派生事件确实不被回读）；缺边/缺 artifact/半截批 → ValueError 指明位置。
- **观察者无扰**：同 seed 同脚本，带与不带 step_observer 各跑一遍训练，最终模型参数逐位相同、既有 phase-5 测试零改动照过。
- **纯函数性**：observe_batch 同输入两调逐位同输出；输入对象未被修改；全程无 RNG。
- **围栏**：诊断包无任何模型库 import（现有扫描器自动执法，此处只需别给它豁免）。

## 4. 工程要求

- 新包建议 `src/skillev/diagnostics/`；纯标准库（math/dataclasses/itertools），不引入 numpy；logsumexp 与池化均值手写并用 math.fsum。
- 一切求和/聚合的遍历顺序显式固定（步序、(trajectory_id, step_index) 序、skill_id 字典序、residuals 序）——顺序即契约，逐位等价靠它。
- 哈希用第 1 层 stable_hash；序列化 canonical；派生输出只有 to_value 没有 from_value；单文件约 1000 行内；测试断行为不断字面量。
- pyproject 无新依赖。

## 5. 冒烟（本轮无 GPU 任务）

若执行下述可选远端重放，只允许 SSH 直连私有配置的 GPU 服务器，不得使用
NCSA Delta、Duo、Slurm 或 scheduler 命令。重放本身不得使用 GPU；若辅助
检查需要 CUDA，应另按当前任务授权安排；两个批准端点的全部物理 H800 GPU `0`–`7`
均可使用，不限制卡号。启动前检查全部卡，优先空闲卡；已有他人进程但 GPU 利用率低且
该卡总显存占用严格低于 25% 的，也允许在剩余显存足够时共用。显式设置
`CUDA_VISIBLE_DEVICES`，不得终止、修改或调试他人进程。

本层是纯 float 层，在线离线等价与硬件无关，**不安排直连 GPU 任务**——这是省钱利器的第一次自我示范。可选顺手项（不阻塞验收）：若 phase-5 冒烟的 event log 还留在 GPU 服务器本地，对它跑一次 replay_flow_diagnostics，note 里记一行"真模型 log 重放成功、若干批、停滞探针状态"；没有留也不必补跑。

## 6. 验收标准

1. DiagnosticsConfig、装配函数、trajectory_flow、skill_marginal_flows、observe_batch、OnlineFlowDiagnostics、replay_flow_diagnostics 全部落地；§3.8 清单全绿（含头条的在线≡离线逐位等价与 library_version 分段测试）。
2. 两个最小加法完成且带回归证据：TrainingLoop 观察者（默认 None 行为不变、观察者无扰测试）、FLOW_DIAGNOSTICS_RECORDED 枚举（重放忽略它的反例测试）。
3. ruff/mypy/pytest 全绿，make check 绿，GitHub Actions 三版本绿。
4. commit + push 到 SKILLEV-new；简短总结本层产物与测试数后停机，等 phase 7 goal。

## 7. 明确不做

- 不写第 7/8 层：无 z 特征分桶、无 Beta-Bernoulli 更新、无 LCB/UCB、无库熵 H(S)、无相变触发布尔的第二合取项、无 Φ 与任何决策。
- 不做诊断状态持久化/断点续算；不做可视化/绘图；不做大 CLI；不并行。
- 不改第 1–5 层语义（§2 两个加法除外）；不改 idea.tex、AGENTS.md；不处理历史遗留。
- 不接 benchmark 数据；不写审批/密封仪式；不启动"审计/评审/找缺口"类子代理。

## 8. 停止条件

- §6 验收达成 → 简短总结，停机，等 phase 7 goal。
- 手算对照或在线≡离线等价 3 次实现尝试仍系统性对不上 → 停机报告现象与已试路径（等价对不上十有八九是某条路没走共用装配函数、或有量从张量而非契约 float 取数——报告里先自查这两条）。
- 发现 event log 缺少重放必需的原始字段、且无法用"补一条向后兼容的 primary 事件发射"解决 → 列缺口与选项后停机请示。
- 对式 (11) 的 B_s 或式 (16) 的 Δ̄² 读法发现与 §1 钉死版本冲突的实现证据 → 列选项与建议后停机请示，禁止静默换读法。
