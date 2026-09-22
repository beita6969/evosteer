# phase7-goal.md —— Bayesian TTB Self-Skill-Revolution Loop：第 7 轮 · 校准层（Bayesian calibration）

> **Protocol-v3 immutable amendment (2026-07-25).** 本文件是历史实施记录；当前语义见 `method-v3-final-spec.md`。Full core 的流权重是 uncapped mean-normalized `F_e/mean(F)`，没有 `enabled`、cap 或 no-flow runtime branch；六种消融均在 `experiments/arms/` 独立实现。Diagnostics 与 calibration 从同一个 `TrainingStepSource` 预览为一个 `ProjectionTransition`，随 `TRAINING_STEP_COMMITTED@3` 原子发布。不存在 posterior update/cell scientific events、live recorded-vs-raw verification、rebuild/from-events 或旧 schema reader；复算仅在离线 `skillev.audit`。

本文件是本轮唯一任务清单，接续已验收的第 1–6 轮。写法是讲解式的：每节先说做什么，紧跟着解释为什么、以及做错时的典型症状。若某处字面指令与所附理由看起来冲突，按理由描述的意图行事并留一句 note 说明。

## 0. 定位：八层导航图中的第 7 层

八层规划回顾：(1)–(6) 已完成（契约 / 模型 / 打分 / rollout / 训练循环 / 流诊断，基线 = 当前 main，至少含 `323fe00` 的 Phase 6 流诊断）；**(7) 校准层（本轮）**；(8) 相变与进化。本轮只做第 7 层。

校准层是什么（人话）：给每个技能维护一张**分情境的成绩单**。每当一条轨迹里某条边调用了技能 s，我们就问四个问题——什么任务域（Context）、那一步执行成了还是怎么砸的（Failure Mode）、初始上下文多大（Token Bucket）、整条轨迹多长（Horizon Bucket）——四个答案拼成情境 z，然后按"这条轨迹最终成没成功"往 (s, z) 那格 Beta-Bernoulli 后验里记一笔账，记账的笔画粗细由第 6 层的流credit定（w_e ∝ F(H_t)：这条边对结局越有贡献，这笔账越重）。查询侧给 LCB/UCB：LCB=μ−kσ 是保守决策边界（第 8 层 Refine/Split 用），UCB=μ+kσ 是乐观边界（第 8 层 Prune 用——证据少的技能 UCB 高，不会被错杀）。

**为什么这层是论文的命根子（Owner 原话）**：它是我们相对 SkillFlow 的 delta 本体——SkillFlow 只有流信号（F̂、log I），我们多的就是这层"分情境、带不确定性、流加权"的 Bayesian 校准。所以本轮有两条硬边界纪律：**其一**，边界干净到**一个配置开关（`CalibrationConfig.enabled=False`）就能把整层关掉退化为 SkillFlow**——这个开关同时就是论文消融实验（ours vs SkillFlow-style）的实现，第 8 层和 benchmark 轮都将读它分叉；**其二**，有状态但**事件溯源**：store 只是缓存，真相在 POSTERIOR_UPDATE_RECORDED 事件流里，状态永远可从日志重建，且更新流本身可从更深层的原始记录复算交叉验证（第 6 层"消融=重放、0 GPU"的路子在本层继续走通）。

## 1. 方法权威与翻译关系

本轮把 idea.tex §2.5.1（Feature-Conditioned Bayesian Calibration）的四个式子翻译成代码，而其中一大半已在第 1 层契约里钉死，本层是**接线不是发明**：

- **式 (12)** z = [Context, Failure Mode, Token Bucket, Horizon Bucket] → 契约 `ContextFeature` 及三个桶枚举**已存在**（`FailureMode.from_observation_status` 只接受 closed set；未知、拼写错误或空字符串立即失败，只有 producer 显式产出的 `other` 才进入 OTHER；`TokenBucket.from_count`（≤1000 / 1001–4000 / >4000）；`HorizonBucket.from_horizon`（≤3 / 4–8 / >8））。桶边界是持久化语义，本层不得另造边界。
- **式 (13)** α_{s,z}=α₀+Σ w_e·𝕀[y_e=1]、β_{s,z}=β₀+Σ w_e·𝕀[y_e=0]，w_e ∝ F(H_t^e) → 契约 `PosteriorUpdateEvent`（存 after-counts）+ `PosteriorCellState.apply()`（严格链校验：prior + 增量必须对得上）**已存在**。本层要写的是：**结果 y_e 的钉死读法**与**w_e 的归一化**。
  - **y_e = record.reward.success（轨迹终局成功）**。理由：后验回答的是"在情境 z 下调用 s，任务最终成功率几何"——技能的任务级价值；步级执行状态已经作为 z 的 Failure Mode 坐标进了条件侧，同一信息不能既当条件又当结果。若实现中读出与 tex 冲突的证据 → §8 停机请示。
  - **w_e = F_e / mean(F)**，均值取本次更新批内全部技能调用边的 F。full method 的 `flow_weight_cap=None`，不改写式 (13)；`cap=10.0` 只属于单独的 capped-flow-weight ablation。数值上在 log 域移位实现：m = max(log F)，w ∝ exp(logF−m)，除以同移位的均值，m 精确相消。若显式实验 config 提供 cap，才执行该轴的裁剪；非有限结果直接失败，不自动换配置。
- **式 (14)(15)** μ、σ²、LCB=μ−kσ → 契约 `mean()/variance()/lcb(k)/ucb(k)` **已存在**（k 由调用方给）。本层要写的是查询接口的包装：k 缺省取 config（默认 k=1.0），未见过的 (s,z) 返回先验格（α₀,β₀）的读数并标记 observed=False——保守决策必须对"没数据"有定义好的答案，而不是 KeyError。

**钉死的**：上述 y_e 与四坐标映射；更新顺序为 stats.residuals 轨迹序 → step_index 升序 → skill_id 字典序；POSTERIOR_UPDATE_RECORDED 载荷原样落账；开关关闭时零更新、零事件、零状态，查询抛类型化错误。`CalibrationEngine` 与 raw recompute 必须显式共享同一 `DiagnosticsConfig`。**full 默认**：α₀=β₀=1.0、`flow_weighting_enabled=True`、`flow_weight_cap=None`、default_k=1.0、Context=task_family。

## 2. 范围刀切

- **本轮预计零下层加法**：观察者钩子（phase 6 的 step_observer）、事件枚举（POSTERIOR_UPDATE_RECORDED / POSTERIOR_CELL_STATE_RECORDED，bootstrap 期已预留）、契约（ContextFeature/PosteriorUpdateEvent/PosteriorCellState）全部现成。若实现发现真缺口 → §8 停机请示，不静默扩改。
- 围栏零改动：本层无 torch，现有扫描器自动执法。
- 不做第 8 层：不消费 LCB/UCB 做任何决策、无相变、无 Φ、无技能库管理（store 按 skill_id 记账，技能进化后新旧 id 的 cell 去留是第 8 层的事）。
- 不做后验的跨进程持久化文件（重建即持久化：真相在事件流里，落一份 store 文件只会制造第二真相）；不做分桶边界的可配置化（契约已钉死为持久化语义）；不做在线可视化。
- 不接 benchmark 数据；不改 idea.tex、AGENTS.md、第 1–6 层语义。

## 3. 组件逐个用人话讲（含代码骨架）

骨架规则同前六轮：字段语义与不变量不可减，命名与文件组织自由；注释说"需要写"的就是实现规格。建议新包 `src/skillev/calibration/`。

### 3.1 四个 z 抽取器：从记录到坐标，各自独立可测

**是什么**：四个纯函数（+一个组合器），把 (TrajectoryRecord, step_index) 映射到 ContextFeature 的四个坐标。**为什么拆成四个而不是一个**：Owner 点名"各自独立可测"——每个坐标的映射依据不同的记录字段，各有各的边界情况；捏成一个函数，测试失败时你不知道是哪个坐标错了。桶的**边界**在契约里（persisted semantics），本层只做"从记录取对字段、喂对契约入口"这一跳。

```python
EXTRACTOR_VERSION: Final = "ttb-z-extractors@2"
# 需要写：抽取器版本常量。桶边界归契约管，但"用哪个字段喂桶"归这里管——
# 比如将来 Context 从 task_family 换成别的，就是本版本号该变的时刻。
# 它进派生快照事件（§3.5），供审计辨认两次重建是否同一套映射。


def extract_context(record: TrajectoryRecord) -> str:
    """需要写：返回 record.task_family。Context 坐标 = 任务域（工程默认，一句话理由可调）。"""


def extract_failure_mode(record: TrajectoryRecord, step_index: int) -> FailureMode:
    """需要写：取 record.steps[step_index-1].observation_status，走契约
    FailureMode.from_observation_status；未知字符串必须原样失败，不能猜到 OTHER。
    step_index 越界 → ValueError 带 trajectory_id 与 step_index。"""


def extract_token_bucket(record: TrajectoryRecord) -> TokenBucket:
    """需要写：TokenBucket.from_count(record.initial_context.assembled_token_count)。
    Token 坐标 = H_0 的规模（模型面对多大的初始上下文）——它对整条轨迹恒定，
    这是有意的：逐步累积 token 数在记录里并不完备（reasoning/observation 不存 token 数），
    而 H_0 规模是完备、可重放、且最能刻画'情境大小'的量。"""


def extract_horizon_bucket(record: TrajectoryRecord) -> HorizonBucket:
    """需要写：HorizonBucket.from_horizon(record.horizon)。注意 horizon 在轨迹结束才知道——
    没关系，校准更新本来就发生在批粒度（轨迹早已完整）。"""


def extract_z(record: TrajectoryRecord, step_index: int) -> ContextFeature:
    """需要写：四个抽取器的组合，返回契约 ContextFeature。cell_key 由契约的
    z.cell_key(skill_id) 给出，本层不自造键。"""
```

### 3.2 流权重：w_e 的归一化与封顶

```python
def flow_weights(
    flows: tuple[TrajectoryFlowDiagnostic, ...], config: CalibrationConfig
) -> dict[tuple[str, int], float]:
    """需要写：键 = (trajectory_id, step_index)，值 = 该边的 w_e。
    1. 收集本批**所有至少调用了一个技能的边**的 log_state_flow（不调用技能的边
       不产生更新、也不进归一化分母——它们不是式 (13) 的 e）；
    2. log 域移位：m = max(log F)；linear_e = exp(logF_e − m)；
       w_e = linear_e / mean(linear)；仅当 config.flow_weight_cap 明确非 None
       时才应用该实验轴的 cap；
       （移位量 m 在分子分母精确相消——这行注释保留，它解释了为什么 logF≈80 不炸）
    3. 全批无技能调用边 → 返回空 dict；
    4. 同一条边可能调用多个技能：多个更新共用同一个 w_e（权重是边的属性不是技能的）。"""
```

**典型症状**：忘了归一化——训练早期 F≈1、后期 F 漂移几个量级，同样的一次成功在不同训练阶段记的账差一万倍，后验被时间加权污染；忘了 cap——一条被裁剪救回来的高流边仍可能带 e^10 的权重，一笔账顶别人一千笔，LCB 从"保守估计"变成"单样本崇拜"。

### 3.3 更新生成：批 → 事件流（在线与复算共用的单点）

**是什么**：一个纯函数，吃（本批流诊断 + 轨迹记录 + 更新前的 cell 状态），吐按序的 PosteriorUpdateEvent 元组。**为什么必须单点**：与 phase 6 的装配函数同理——在线路径与"从原始记录复算更新流"的审计路径必须走同一段生成代码，等价性由构造保证。**为什么事件带 after-counts 还要现算**：契约 PosteriorUpdateEvent 存 alpha_after/beta_count_after，apply() 会拿 prior 严格对账——所以生成时必须按钉死顺序逐格滚动 prior，顺序一乱链就断（这是特性不是负担：任何重排/丢失/篡改都会在重建时炸出来）。

```python
def calibration_updates_for_batch(
    *,
    batch_id: str,
    flows: tuple[TrajectoryFlowDiagnostic, ...],
    records_by_id: Mapping[str, TrajectoryRecord],
    cells_before: Mapping[str, PosteriorCellState],   # 键 = z.cell_key(skill_id)
    config: CalibrationConfig,
) -> tuple[PosteriorUpdateEvent, ...]:
    """需要写：
    1. w = flow_weights(flows, config)；
    2. 按钉死顺序遍历（flows 的既有轨迹序 → step_index 升序 → sorted(invoked_skill_ids)）；
    3. 每个 (边, 技能)：z = extract_z(record, step_index)；outcome = record.reward.success；
       prior = 滚动状态中的该格（无则以 config.alpha_0/beta_0 造先验格）；
       after = 按式 (13) 现算；event_id = stable_hash({"batch_id", "skill_id",
       "trajectory_id", "step_index"})——确定性、全实验唯一（轨迹 id 唯一保证）；
    4. 构造 PosteriorUpdateEvent（契约自会校验 after-counts 为正等），并把滚动状态推进
       （同一格同批可被多条边更新——链内顺序即遍历顺序）；
    5. enabled=False 时本函数不该被调用；被调用则 raise CalibrationDisabledError
       （防御的是真实错误：绕过引擎直接调生成器）。"""
```

### 3.4 引擎与查询：PosteriorStore + LCB/UCB

```python
@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """校准层全部旋钮。enabled 就是论文的消融开关。"""

    enabled: bool = True
    flow_weighting_enabled: bool = True
    alpha_0: float = 1.0            # > 0 有限；Beta 先验
    beta_0: float = 1.0             # > 0 有限
    flow_weight_cap: float | None = None
    default_k: float = 1.0          # ≥ 0 有限；查询未显式给 k 时的默认
    format: str = "skillev-calibration@2"

    def to_value(self) -> dict[str, JsonValue]: ...


class CalibrationDisabledError(RuntimeError):
    """关掉校准后仍试图更新/查询——调用方（第 8 层）该走 SkillFlow 退化分支而不是来问我。"""


@dataclass(frozen=True, slots=True)
class CellQuery:
    """一次查询的完整读数——第 8 层拿它就够做决策，不必回头摸 store。"""

    skill_id: str
    z: ContextFeature
    alpha: float
    beta_count: float
    mean: float
    sigma: float          # sqrt(variance)
    lcb: float            # μ − kσ（式 15）
    ucb: float            # μ + kσ（Prune 的乐观边界）
    k: float              # 本次查询实际用的 k
    observed: bool        # update_count > 0；False = 你看到的是先验
    update_count: int


class CalibrationEngine:
    """有状态但事件溯源的 (s,z) 后验总管。状态 = dict[cell_key, PosteriorCellState]。"""

    def __init__(self, config: CalibrationConfig,
                 diagnostics_config: DiagnosticsConfig,
                 emitter: RuntimeEventEmitter | None = None): ...

    @property
    def enabled(self) -> bool: ...   # 第 8 层分叉读这里

    def observe(self, observation: TrainingStepObservationLike) -> tuple[PosteriorUpdateEvent, ...]:
        """需要写：enabled=False → 返回 ()（不碰任何东西，不发事件——退化为 SkillFlow 的
        全部含义就是这行）。否则：records_by_id 取自 observation.batch 的 artifact.record；
        流诊断经第 6 层公开 API 现算（assemble_batch_flow_input + trajectory_flow；
        与 OnlineFlowDiagnostics 是否同时挂载互不影响——纯函数重算廉价，不共享状态）；
        调 calibration_updates_for_batch → 逐条 apply 进 store（契约链校验第二道岗）→
        有 emitter 则逐条落 POSTERIOR_UPDATE_RECORDED（载荷 = to_value 原样）+
        每批一条 POSTERIOR_CELL_STATE_RECORDED 派生快照（§3.5）→ 返回事件元组。"""

    def query(self, skill_id: str, z: ContextFeature, k: float | None = None) -> CellQuery:
        """需要写：enabled=False → raise CalibrationDisabledError。k 缺省取 config.default_k。
        cell 不存在 → 以先验格计算读数，observed=False（μ=α₀/(α₀+β₀)，LCB/UCB 照式 15，
        对 (1,1,k=1) 手算 LCB ≈ 0.2113——测试用这个数）。数值一律走契约的
        mean/variance/lcb/ucb，本层不重写公式。"""

    def cells_for_skill(self, skill_id: str) -> tuple[PosteriorCellState, ...]:
        """需要写：该技能全部已观测 cell，按 z.content_hash 字典序——第 8 层枚举证据用。"""

    def all_cells(self) -> tuple[PosteriorCellState, ...]: ...   # 同上，全库，确定性序
```

### 3.5 事件溯源：落账、重建与审计交叉验证

**是什么**：三条路——落账（在线）、重建（从更新事件折叠）、复算（从更深层原始事件重跑第 6 层+本层生成器）。**为什么要两条重建路**：从更新事件折叠是**快速通道**（第 8 层重启、换机器恢复状态用）；从原始记录复算是**审计通道**（证明日志里的更新流没被算错/漏发/重排——两条路对不上，说明有一条在撒谎，这正是要它们同时存在的原因）。这也是省钱纪律的延续：**校准超参（cap、先验、y_e 读法……将来任何一版）的消融 = 对既有 log 复算更新流再折叠，0 GPU**。

```python
def rebuild_posterior_store(
    events: Iterable[Mapping[str, JsonValue]], config: CalibrationConfig
) -> dict[str, PosteriorCellState]:
    """需要写（快速通道）：扫事件流，只认 POSTERIOR_UPDATE_RECORDED；载荷走
    PosteriorUpdateEvent.from_value 正门；按事件流顺序对各 cell 从先验格起 apply 链——
    契约的 after-counts 对账就是防篡改/防乱序的机器执法；链断 → ValueError 指明
    event_id 与 cell。派生快照事件（POSTERIOR_CELL_STATE_RECORDED）无条件忽略。"""


def recompute_update_stream(
    events: Iterable[Mapping[str, JsonValue]], config: CalibrationConfig
) -> tuple[PosteriorUpdateEvent, ...]:
    """需要写（审计通道）：按 phase-6 重放同款状态机扫原始事件（artifact / stats / edge /
    TRAINING_STEP_COMMITTED 关批），每批经第 6 层公开 API 得 flows、
    经 calibration_updates_for_batch（滚动 cells）重新生成更新流。
    对同一份 log：本函数输出的 content_hash 序列必须与日志中 POSTERIOR_UPDATE_RECORDED
    的逐位一致——这是本层的头条测试之一。"""
```

POSTERIOR_CELL_STATE_RECORDED 快照事件：每次 observe 后对**本批被触碰的 cell** 落一条，载荷 = {"cells": [各 to_value], "extractor_version": EXTRACTOR_VERSION, "config": config.to_value()}——纯派生便利（人看曲线、快速核对），**永不回读**（phase 6 的派生纪律原样适用，测试同样要有"日志里混入内容错误的快照事件，重建结果不受影响"的反例）。

另备一个三行小工具 `compose_step_observers(*observers)`：把 OnlineFlowDiagnostics 与 CalibrationEngine.observe 同时挂上 TrainingLoop 的单一 step_observer 位——省得每个实验自己手搓 lambda。

### 3.6 测试策略（本轮的显微镜清单）

全部 CPU、无网络、无 torch。契约对象走第 1 层正门构造；训练侧用 phase-5/6 测试的 scripted 组件。清单（每条都是行为断言）：

- **四抽取器独立测**：task_family 透传；observation_status 六态映射，未知/拼写错误/空字符串失败，显式 `other` 成功；token 桶边界 1000/1001、4000/4001；horizon 桶边界 3/4、8/9；组合器产出合法 ContextFeature 且 cell_key 稳定；step_index 越界报错带位置。
- **共轭手算对照（主镜）**：两技能三更新的混合批（含成功与失败轨迹、一边多技能、同格同批两次更新），纸面算 α/β/μ/σ/LCB/UCB 与实现 rtol=0 逐位对照——测试注释写出手算；契约 lcb(k) 与式 (15) 闭式再交叉核一次。
- **w_e**：full cap=None 时批内权重均值 == 1.0 且没有 `min`；大 logF 不溢出；显式 capped arm 改变 α；`flow_weighting_enabled=False` 时每笔权重严格为 1；无技能调用边不进分母；多技能边共用同一 w_e。
- **顺序与链**：更新顺序 = 轨迹序→步序→技能字典序；event_id 全局唯一且确定；人为重排/篡改 after-counts 的事件流在重建时被契约 apply 拒绝（链校验行为）。
- **y_e 语义**：成功轨迹的全部调用边进 α、失败进 β；同批混合正确分流。
- **头条一（重建等价）**：在线跑 ≥3 批（含被拒 rollout 与无技能调用的批）→ rebuild_posterior_store(捕获事件) 与在线 store 逐 cell content_hash 相等。
- **头条二（审计交叉）**：recompute_update_stream(同一份日志) 与日志中更新事件的 content_hash 序列逐位相等。
- **消融开关**：enabled=False → observe 返回 ()、零事件、all_cells 为空、query/生成器抛 CalibrationDisabledError；挂载与否、开关开合，训练参数逐位不变（观察者无扰，phase-6 同款测法）；phase-5/6 既有测试零改动。
- **未见格查询**：observed=False、(1,1,k=1) 的 LCB ≈ 0.2113 手算对照；显式 k 覆盖 default_k；负 k 被契约拒绝。
- **派生不回读**：日志混入内容错误的 cell 快照事件 → 重建与复算结果不受影响。
- **纯函数性与确定性**：同输入两调逐位同输出；输入未被修改；无 RNG；围栏自动执法（不给本包豁免）。

## 4. 工程要求

- 新包建议 `src/skillev/calibration/`；纯标准库；对 contracts/diagnostics/training 只 import 公开导出面。
- 一切遍历顺序显式固定（轨迹序→步序→skill_id 字典序→cell 按 content_hash 序）；哈希用 stable_hash；载荷 canonical；单文件约 1000 行内；测试断行为不断字面量。
- pyproject 无新依赖。

## 5. 冒烟（本轮无 GPU 任务）

若执行下述可选远端重建，只允许 SSH 直连私有配置的 GPU 服务器，不得使用
NCSA Delta、Duo、Slurm 或 scheduler 命令。重建本身不得使用 GPU；若辅助
检查需要 CUDA，应另按当前任务授权安排；两个批准端点的全部物理 H800 GPU `0`–`7`
均可使用，不限制卡号。启动前检查全部卡，优先空闲卡；已有他人进程但 GPU 利用率低且
该卡总显存占用严格低于 25% 的，也允许在剩余显存足够时共用。显式设置
`CUDA_VISIBLE_DEVICES`，不得终止、修改或调试他人进程。

纯 float 层，与 phase 6 同理**不安排直连 GPU 任务**。可选顺手项（不阻塞）：若 GPU 服务器留有 phase-5/6 的真模型 event log，对它跑一次 recompute_update_stream + rebuild，note 记一行"真日志重建成功、N 格 M 次更新、开关两态验证"；没有就不补跑。

## 6. 验收标准

1. 四抽取器、flow_weights、calibration_updates_for_batch、CalibrationEngine（observe/query/cells_for_skill/all_cells/enabled）、rebuild、recompute、compose_step_observers 全部落地；§3.6 清单全绿（含两条头条等价与消融开关测试）。
2. 零下层改动（或 §8 请示后 Owner 批准的最小加法，附回归证据）。
3. ruff/mypy/pytest 全绿，make check 绿，GitHub Actions 三版本绿。
4. commit + push 到 SKILLEV-new；简短总结本层产物与测试数后停机，等 phase 8 goal。

## 7. 明确不做

- 不写第 8 层：不消费 LCB/UCB/F̂ 做任何决策，无相变、无 Φ、无技能库管理与 id 迁移。
- 不做 store 文件持久化、不做桶边界配置化、不做可视化、不并行。
- 不改第 1–6 层语义；不改 idea.tex、AGENTS.md；不处理历史遗留。
- 不接 benchmark 数据；不写审批/密封仪式；不启动"审计/评审/找缺口"类子代理。

## 8. 停止条件

- §6 验收达成 → 简短总结，停机，等 phase 8 goal。
- 共轭手算或两条头条等价 3 次实现尝试仍系统性对不上 → 停机报告现象与已试路径（先自查：更新顺序是否与钉死序一致、after-counts 是否用滚动 prior 现算、两条路是否真共用同一生成函数）。
- 发现 y_e 读法、Context 映射或 w_e 归一化与 idea.tex 有双向不兼容的证据 → 列选项与建议后停机请示，禁止静默换读法。
- 发现确需下层加法（事件字段、观察者形态等）→ 列缺口与最小方案后停机请示。
