# phase5-goal.md —— Bayesian TTB Self-Skill-Revolution Loop：第 5 轮 · 训练循环（trainer）

> **Protocol-v3 immutable amendment (2026-07-25).** 本文件是历史实施记录；当前语义见 `method-v3-final-spec.md`。Literal full method 的 AdamW `weight_decay=0.0` 且无 gradient clipping。每批先固定 exact B-task population，每项只尝试一次；任一 infrastructure failure 终止整个 attempt，不替换、不补采、不重试。Production `TrainingLoop.run(n)` 是完整 `collect → train` 入口；每步只追加一个 `TRAINING_STEP_COMMITTED@3` scientific source，并在 append 成功后发布已 preview 的 diagnostics/calibration state。Z reset 必须显式 seed；不存在随机默认、`emit_event=False`、optional observer、prepare/apply/recover 或旧 checkpoint reader。

本文件是本轮唯一任务清单，接续已验收的第 1–4 轮。写法是讲解式的：每节先说做什么，紧跟着解释为什么、以及做错时的典型症状。若某处字面指令与所附理由看起来冲突，按理由描述的意图行事并留一句 note 说明。

## 0. 定位：八层导航图中的第 5 层

八层规划回顾：(1) 轨迹数据层（已完成）；(2) 模型层（已完成）；(3) 打分与目标层（已完成）；(4) rollout 引擎（已完成，`src/skillev/rollout/`）；**(5) 训练循环（本轮）**；(6) 流诊断；(7) Bayesian 校准；(8) 相变与进化。本轮只做第 5 层。开工时以当前 main（至少含 `ff8f4e4` 的 Phase 4 收口）为基线。

训练循环是什么（人话）：把 2+3+4 三层组合成一台能持续转的机器——**完全体的 "collect → train" 交替驱动循环**：在钉住的当前策略快照下采一批 RolloutArtifact（第 4 层）、对每条做 teacher-forced 打分得到带梯度的 loss（第 3 层）、一次 AdamW 联合更新三个可训练组件（第 2 层的两套 adapter + Z 头）、更新策略版本让下一批采样自动 on-policy、落账批统计与曲线、按节奏存检查点。另外暴露一个窄钩子 `reset_partition()` 给将来的进化层调 Z 重置（idea.tex §2.5.2）。

为什么"薄一点是美德"（本轮的第一纪律）：**trainer 自己几乎没有算法内容**。前向/后向条件怎么拼、K_t 怎么归一、Δ 四项怎么合、token 身份怎么保真、on-policy 怎么定义——全部已经在第 1–4 层被实现且被测过。本层值得测的只有"组合是否老实"：调用次数对不对、梯度到没到该到的地方、版本换没换、账落没落。反面症状：一旦你发现自己在 trainer 里写 renderer、写 logits 切片、写"顺手归一化一下"，那就是在第 5 层重新发明第 2/3 层——立刻停手，回接口去调。

## 1. 方法权威与翻译关系

idea.tex 对本层的字面要求很少而明确：式 (7)(8) 的 L_TTB 按批最小化，联合训练 π_θ（forward adapter）、P_φ（backward adapter）、log Z_θ（Z 头）三个组件（变分读法，无守恒约束）；§2.5.2 库更新后 Z 重初始化且"重置后训练继续"。翻译成本层的机器动作：**批 loss = batch 内各轨迹 (Δ/T)² 的均值 → 一次 backward → 三组参数一起 step**；`reset_partition()` = `backbone.reset_z()` + 清 Z 头的优化器状态（第 2 轮接口注释里明确留给本层的分工——"优化器状态的清空是第 5 层的事"，现在到账了）。

on-policy 语义的消费端也在本层闭环：第 4 层已经把"轨迹必须钉住策略快照、快照不符整条拒绝"做成了机器执法，但它"只消费、不递增"版本（`docs/phase4-goal.md` §3.1 原话：Phase 5 每次策略更新后必须换版本）。所以本层每做一次 optimizer step，都必须让 `adapter_version` / snapshot 变化，从而让任何拿旧快照的采样在第 4 层被自动拒收——**这不是日志美观问题，是 off-policy 污染的唯一拦截机制**。

**钉死的**：批目标 = mean((Δ/T)²)；三组件联合更新、一步一批、**不复用样本**；每次 step 后版本必变；采集前一次性声明恰好 B 个任务且每个只执行一次；任何 infrastructure failure 使当前 run 进入终止状态，绝不换任务或补样本；agent 的 parse/schema/tool 业务失败以及 R=0 都是合法训练数据；R̃/ε_min/β 各自只有一个来源。full method 的策略采样与 teacher-forced score 都是 raw softmax（temperature=1、top_p=1），这些不是可配置训练旋钮。**工程默认**：AdamW，adapter lr 1e-4、Z 头 lr 1e-3、显式 `weight_decay=0.0`、`gradient_clip_norm=None`；batch_size B=8；ε_min=0.01；β=1.0；checkpoint 每 25 步一存、保留最近 3 份；max_reasoning/action_tokens=256；horizon 上限 max_turns=8。显式 clipping 只属于单独实验配置，不属于 full identity。seed 按 AGENTS.md 只跑一组。

## 2. 范围刀切

- 本轮允许对下层做**两个最小加法**，除此之外第 1–4 层语义冻结：
  1. **第 2 层版本推进 API**（必须做，否则 on-policy 闭环物理上无法实现）：`PolicyBackbone` 与 `HFPolicyBackbone` 增加 `mark_policy_update(optimizer_step: int) -> None`（命名自由）——把三个版本串的 `@<step>` 后缀改写为给定步数，前缀（checkpoint id 谱系）不动。廉价（纯字符串操作，不重算内容哈希），调用后 `adapter_version`/`z_version` 必变、`LocalPolicyGenerator.snapshot()` 随之变。既有 phase-2 测试保持不变，新增行为测试。
  2. **import 边界的第二次精确化**：trainer 需要运行时 `import torch`（torch.optim.AdamW、clip_grad_norm_——这些没法用"方法与算符"绕开）。边界规则升级为：**torch 运行时 import 允许出现在 policy 包与本轮新建的 training 包两处；transformers/peft 仍然只允许在 policy 包**（模型与 tokenizer 的装载单点不变——trainer 碰优化器但永远不碰模型装载，这才是围栏真正要守的东西）。扫描器自测补正反例：training 包 import torch 豁免、import transformers 违规。
- 不写第 6 层及以上：无滑窗 Δ̄² 停滞判据、无库熵、无相变、无后验、无进化动作——本层只负责把每批的 TTBBatchStats 如实落账（它正是第 6/8 层将来的输入）。`reset_partition()` 只做 Z 重置本身，**本轮没有任何代码自动调它**，进化层将来才是调用方；测试直接调它验证行为。
- 无样本复用/replay buffer/多 epoch；无 lr schedule/warmup（常数 lr，将来要再加）；无并行采集、无 DeepSpeed/多卡、无 vLLM（本轮训练冒烟只需直连 GPU 服务器的单卡；分布式等正式 benchmark 训练轮按需引入）；无 benchmark 数据获取。
- 上游遗留（README、legacy feature API 清理等）一律不碰，Owner 已裁决留独立范围。

## 3. 组件逐个用人话讲（含代码骨架）

骨架规则同前四轮：字段语义与不变量不可减，命名与文件组织自由；注释说"需要写"的就是实现规格。建议新包 `src/skillev/training/`。

### 3.1 TrainerConfig v2：分开方法、采样、优化器、执行与检查点身份

**是什么**：本层全部工程参数的唯一入口。**为什么单独一类**：曲线复现的前提是"这一步用的所有旋钮都可回放"；参数散在构造函数和局部变量里，三个月后没人能对上账。

```python
@dataclass(frozen=True, slots=True)
class TTBMethodConfig:
    epsilon_min: float
    temperature_beta: float


@dataclass(frozen=True, slots=True)
class PolicyRolloutConfig:
    base_seed: int
    max_turns: int
    max_reasoning_tokens: int
    max_action_tokens: int
    per_rollout_maximum: BudgetVector


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    adapter_learning_rate: float
    z_learning_rate: float
    weight_decay: float
    gradient_clip_norm: float | None


@dataclass(frozen=True, slots=True)
class TrainingExecutionConfig:
    experiment_id: str
    batch_size: int


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    directory: str
    every_n_steps: int
    keep_last: int


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    method: TTBMethodConfig
    rollout: PolicyRolloutConfig
    optimizer: OptimizerConfig
    execution: TrainingExecutionConfig
    checkpoint: CheckpointConfig
    format: str = "skillev-trainer-config@2"
```

各子配置独立 canonical round-trip；总 hash 包含全部子配置。`library_version`
只能来自构造时注入的 provider。v1 平铺 wire 与旧 checkpoint 一律拒绝。

### 3.2 采集侧：TaskProvider、会话工厂与 collect_batch

**是什么**：driver 的"进料口"。任务从哪来、环境和 evaluator 怎么造，本层一概不知道也不该知道——收两个窄 Protocol，测试给 scripted 实现，冒烟给 debug 实现，将来 benchmark 轮给真实现。**为什么 engine 每条轨迹现装一台**：第 4 层的 RolloutEngine 持有一个 bounded agent（包一个有状态的环境会话），天然是一次性对象；共享件（generator、assembler、codec、state store、ledger、emitter、clock）长寿命，逐轨迹只换环境与 evaluator，装配成本是纯接线。

```python
class TaskProvider(Protocol):
    def next_task(self) -> RolloutTask:
        """需要写：产出下一个公开任务投影。顺序即课程；本层不洗牌不加权。"""


@dataclass(frozen=True, slots=True)
class RolloutSessionBundle:
    """跑一条轨迹所需的逐任务部件。"""

    environment: RolloutEnvironmentSession
    evaluator: TerminalEvaluator
    retrieved_skills: tuple[RetrievedSkillContext, ...]   # 本轮通常为 ()——库在第 8 层


class RolloutSessionFactory(Protocol):
    def create(self, task: RolloutTask) -> RolloutSessionBundle:
        """需要写：给任务配环境会话与私有 evaluator。答案隔离由第 4 层契约保证，
        本层只传递不窥视。"""
```

`collect_batch()` 的 v2 语义（async，返回恰 B 个 Artifact）：

1. 开批 pin 当前 policy snapshot 与 library version；
2. **在任何 rollout 启动前**连续取出恰 B 个任务，构造
   `TrainingBatchPlan`，任务与 trajectory identity 均唯一，并先落
   `TRAINING_BATCH_PLANNED`；
3. 按 plan 顺序每个任务执行一次，不重试、不替换、不取第 B+1 个任务；
4. agent 可见业务失败仍形成 Artifact；typed infrastructure failure 则落
   `TRAINING_BATCH_FAILED`，把 loop 置为 `FAILED` 并抛
   `TrainingRunFailedError`；
5. 成功收口时，B 个 Artifact 必须与 plan、pinned snapshot、library
   version、decoding identity 逐位一致，随后进入唯一 `BATCH_READY`
   状态。失败实例不能原地 resume。

### 3.3 训练步：train_step 的固定顺序

**是什么**：一批 artifact 进、一次参数更新出。**为什么逐轨迹 backward 累积而不是先堆 B 张图再一起 backward**：打分图的激活占显存与轨迹长度成正比，B 张图同时活着 = 显存随 B 线性涨；对 `(loss_i / B)` 逐条 backward，梯度按线性性精确等于对均值 backward（浮点顺序差异在容差内），显存恒定于单条轨迹。这是本层唯一一处"算法味"的实现选择，所以把理由写死在这里。

```python
def train_step(self, batch: tuple[RolloutArtifact, ...]) -> TrainingStepReport:
    """需要写，顺序不可调：
    1. 批一致性校验（§3.2 第 5 条）；optimizer.zero_grad(set_to_none=True)；
    2. 逐条：score = score_trajectory(backbone, artifact.record,
       artifact.initial_context.text, ScoringConfig(config.temperature_beta))；
       (score.loss / B).backward()——逐条释放图；
       residual = materialize_residual(score, raw_reward=record.reward.value)；
       edges = materialize_edge_records(score, forward_adapter_version=...,
       backward_adapter_version=...)（版本串取本步更新前的现值——这些边确实是旧参数打的分）；
    3. 三组件分别测量变换前 L2 norm。仅当
       config.optimizer.gradient_clip_norm 明确非 None 时才分组件 clip；
       full method 为 None，不改变梯度；
    4. optimizer.step()；self._optimizer_step += 1；
    5. backbone.mark_policy_update(self._optimizer_step)——on-policy 闭环在此拧上：
       下一批 collect 的 pinned snapshot 从这一刻起就是新的；
    6. 组装 TTBBatchStats：batch_id = f"{experiment_id}-step-{step:06d}"，
       batch_loss 与 mean_reward **用 residual 的 float 在 float64 里按契约同款公式重算**
       （math.fsum((r.delta/r.horizon)**2)/B）——不要用 torch 批 loss 的 .item()，
       float32 图与契约 float64 重算的低位差会被容差校验拒收（phase-3 物化的同一课）；
       library_version/optimizer_step/created_at 如实；
    7. 落账：TTB_BATCH_STATS_RECORDED + 每条轨迹的 EDGE_LOGPROB_RECORDED +
       新增 TRAINING_STEP_COMMITTED（载荷 = TrainingStepReport.to_value()）；
    8. 到 checkpoint 节奏则存盘（§3.5）；返回报表。
    torch 批 loss（detach 的均值）也进报表——它和 stats.batch_loss 是两本账：
    一本是训练图的实况，一本是审计的 float64 重算，各自内部一致，低位允许不同。"""
```

### 3.4 优化器与 reset_partition：三组参数、一个窄钩子

**是什么**：单个 AdamW、三个 param group（forward-policy / backward-policy / z-head，前两组 lr 相同、Z 组更大），参数经 `backbone.trainable_parameters(component)` 取——**这是本层拿参数的唯一渠道**，直接摸 `named_parameters` 过滤字符串就是在重新发明第 2 层。param group 的组装顺序固定（三个组件名的固定序），optimizer state_dict 的往返才对得上号。

`reset_partition()` 需要写的语义：调 `backbone.reset_z()` 拿到新 z_version → **只清 Z 组参数的优化器状态**（phase-2 的 reset 是对既有 Parameter 对象原位重初始化，参数身份不变，所以从 `optimizer.state` 里删掉 Z 组参数的条目即可；两组 adapter 的动量必须原封不动——把它们也清了，等于每次库进化都把策略训练打回冷启动）→ 发一条新增的 PARTITION_RESET 事件（载荷含新旧 z_version 与当前 optimizer_step）→ 返回新 z_version。步计数**不**清零：idea.tex 说的是"Z 重置后训练继续"。

**典型症状**：忘清 Z 的优化器状态——重置后头几步，AdamW 的旧动量把刚随机化的 Z 头往重置前的旧值方向拽，Δ 曲线出现一段"幽灵回弹"，第 6 层会把它误读成停滞恢复。这正是把 reset 做成 trainer 钩子而不是让进化层直接调 backbone.reset_z() 的原因：backbone 管参数、trainer 管动量，两件事必须原子地一起发生。

### 3.5 检查点与恢复：断点等于没断过

**是什么**：`save_checkpoint()` = `backbone.save_checkpoint(目录/policy)` + `torch.save(optimizer.state_dict(), 目录/optimizer.pt)` + `trainer_state.json`（canonical：optimizer_step、experiment_id、config 摘要哈希）；按 keep_last 滚动清理。`resume(目录)` = 逆操作后 `backbone.mark_policy_update(恢复的 step)`。**为什么恢复后与不间断跑必须逐位一致**：采集是 scripted 确定性的、打分无随机、AdamW 给定梯度确定——整条链没有任何合法的不确定来源，"差不多接上了"必然意味着哪份状态没存全（十有八九是优化器动量）。

两条已知约定，写进注释防将来困惑：(a) phase-2 的 `save_checkpoint` 会把三个版本串重打为 `<新内容哈希>@0`——所以本层存盘后要**立刻**补一次 `mark_policy_update(当前 step)` 把后缀改回真实步数，否则存盘到下一步之间采的轨迹会顶着 `@0` 的版本；(b) 恢复运行与不间断运行的参数/动量/loss 逐位相等，但版本串**前缀**允许不同（一个是初始谱系、一个是检查点谱系）——on-policy 机制只依赖版本串的唯一性，不依赖字面连续性；测试断参数与 loss，不断版本串字面。

### 3.6 曲线落账：TrainingStepReport

```python
@dataclass(frozen=True, slots=True)
class TrainingStepReport:
    """一步训练的完整可回放报表；to_value() 走 canonical JSON 进事件。"""

    optimizer_step: int
    batch_id: str
    torch_batch_loss: float          # 训练图实况（detach 均值）
    audited_batch_loss: float        # == TTBBatchStats.batch_loss（float64 重算）
    mean_reward: float
    grad_norm_forward: float         # 三个裁剪前范数——本层最有价值的曲线
    grad_norm_backward: float
    grad_norm_z: float
    forward_adapter_version: str     # 本步更新后的三个版本（与下一批快照对账）
    backward_adapter_version: str
    z_version: str
    started_at: str
    completed_at: str
```

事件枚举加三个成员：`TRAINING_STEP_COMMITTED`、`PARTITION_RESET`、`TRAINING_CHECKPOINT_SAVED`（`TTB_BATCH_STATS_RECORDED`、`EDGE_LOGPROB_RECORDED` 枚举早已预留，直接用）。事件载荷不含凭据、不含题面正文；artifact 的 initial_text 只在本地运行产物里，规矩照第 4 轮。

### 3.7 driver：TrainingLoop.run

**是什么**：最外层的交替循环，薄到只剩顺序。**构造时的一条硬校验**：`generator.snapshot().backbone_id == backbone.backbone_id`——trainer 与 generator 必须持有**同一个** backbone 实例（单栈的本义）。两个实例各拿一份权重的话，训练在 A 上、采样在 B 上，off-policy 得无声无息；症状是 §3.2 第 5 条的批校验每批都炸（版本推进只发生在 A 上）。

```python
class TrainingLoop:
    def __init__(self, *, backbone, generator, task_provider, session_factory,
                 config, state_store, ledger, emitter, clock) -> None: ...

    async def run(self, num_iterations: int) -> tuple[TrainingStepReport, ...]:
        """需要写：重复 num_iterations 次 {collect_batch() → train_step()}，
        返回全部报表。异常不吞：collect 的停机条件、train 的任何错误原样上抛。
        入口边界用 asyncio.run 的调用方式与第 4 轮测试一致。"""

    def reset_partition(self) -> str: ...      # §3.4；本轮无人自动调用
    def save_checkpoint(self) -> str: ...      # §3.5；返回目录
    def resume(self, directory: str) -> None: ...
```

### 3.8 测试策略（本轮的显微镜清单）

全部 CPU、无网络：tiny 底座（phase-2 构造）+ scripted generator/environment/evaluator（phase-4 测试的现成手法）驱动**真实 TrainingLoop**。清单（每条都是行为断言）：

- **端到端两迭代**：B=2 跑 2 个 iteration；engine 调用恰 4 次、optimizer step 恰 2 次；三个组件的参数都变了（对照训练前快照），冻结底座逐位没变。
- **on-policy 闭环**：每步后 `adapter_version`/`z_version`/`generator.snapshot()` 都变；第 2 迭代的 artifact manifest 携带新 snapshot_id；人为混入旧快照 artifact 的批被拒。
- **累积等价**：对同一批，逐条 `(loss/B).backward()` 累积出的梯度与一次性 `mean(loss).backward()` 逐参数极紧容差一致（本层唯一的数值实现选择要有对照证据）。
- **两本账**：TTBBatchStats 构造即过契约（float64 重算）；演示直接塞 torch `.item()` 的批 loss 会被契约容差拒收；混 library_version 的批被拒。
- **固定总体与失败分类**：B 个任务先整体计划；第 2 个 infrastructure
  failure 时不执行后续任务、不 admission、不 optimizer step，loop 永久
  FAILED；R=0/parse/schema/tool 业务失败仍进批。
- **优化器 identity**：full 未调用 `clip_grad_norm_` 且显式
  weight_decay=0；另一个显式 clipped config 验证三组件变换及变换前范数。
- **reset_partition**：z_version 换新；Z 组优化器状态被清（动量条目消失）而 adapter 组动量原封不动；紧接着还能正常 train 一步；事件发出；adapter 参数与版本不受影响。
- **断点恢复**：训 2 步 → 存盘 → 全新 TrainingLoop + 全新 backbone 实例 resume → 再训 1 步，参数与第 3 步 loss 和不间断跑 3 步逐位一致（fake clock、scripted 输入；版本串前缀允许不同，见 §3.5(b)）。
- **确定性**：同 config 同脚本两次全程，最终参数字节一致、报表 to_value 一致（时间戳走 fake clock）。
- **围栏升级**：边界测试按 §2.2 更新且全仓通过；training 包 import transformers/peft 的样例被扫描器抓住。
- **mark_policy_update 回归**：phase-2 既有测试不动；新增测试断三个版本后缀变化、前缀不变、snapshot 随之变化。

## 4. 工程要求

- 新包建议 `src/skillev/training/`；运行时允许 `import torch`（§2.2 的围栏升级），**禁止** import transformers/peft/tokenizers；对 policy/scoring/rollout 只 import 公开导出面，不 import 内部模块。
- trainer 内不得出现任何渲染、前缀、logits、归一化逻辑；发现需要就说明下层接口缺了什么，按 §8 停机请示，不要就地实现。
- 事件与状态一律 canonical JSON；哈希用第 1 层 stable_hash；单文件约 1000 行内；测试断行为不断字面量。
- pyproject 无新依赖（torch 已在 policy 组）。

## 5. 真底座冒烟（有界，本轮唯一的直连 GPU 任务）

**GPU 执行硬约束**：只通过 SSH 直连私有配置的 GPU 服务器，不得使用
NCSA Delta、Duo、Slurm 或 scheduler 命令。本轮是单卡任务，固定使用物理
H800 GPU `0` 并显式设置 `CUDA_VISIBLE_DEVICES=0`（这是该历史轮次的固定选择，
不是当前全局限制）。当前执行必须先检查全部 8 张卡，优先空闲卡；已有他人进程但 GPU 利用率低且该卡总显存
占用严格低于 25% 的，也允许在剩余显存足够时共用，不得终止、修改或调试他人进程。

这是五层栈第一次在真模型上整体转起来。直连 GPU 服务器单卡、真实 Qwen3.5-9B：debug 环境（第 4 轮冒烟同款公开环境即可），B=2 跑 2 个 iteration（= 2 次真实 optimizer step），然后 `reset_partition()` 再跑 1 步，途中存一次检查点并 resume 验证接续。记录：每步的 torch/audited 双 loss、三组梯度范数、三个版本串的演变、显存峰值、每 iteration 耗时，commit-message 级 note（按 AGENTS 惯例带一句大致 GPU 用量）。显存吃紧再打开 BackboneConfig 现成的 gradient_checkpointing 并记一句。**有界**：不追收敛、不调参、不接 benchmark、不并行。被权重、私钥或 GPU 服务器阻塞 → 按 AGENTS.md 说明缺什么后停机。

## 6. 验收标准

1. TrainerConfig、collect_batch、train_step、优化器三组装配、reset_partition、检查点/恢复、TrainingStepReport、TrainingLoop 全部落地；§3.8 清单全绿。
2. 两个下层最小加法完成且带回归证据：mark_policy_update（phase-2 旧测试不变）、import 围栏升级（扫描器自测含正反例）。
3. ruff/mypy/pytest 全绿，make check 绿，GitHub Actions 三版本绿。
4. GPU 冒烟完成并留 note（或按 §5 阻塞条款停机说明）。
5. commit + push 到 SKILLEV-new；简短总结本层产物、测试数与冒烟结果后停机，等 phase 6 goal。

## 7. 明确不做

- 不写滑窗 Δ̄² 停滞、库熵、相变判据、后验更新、LCB/UCB、进化动作——TTBBatchStats 落账即本层对上述各层的全部义务；reset_partition 本轮无自动调用方。
- 不复用样本（无 replay、无多 epoch）、无 lr schedule、无 early stopping、无多 seed。
- 不并行采集、不 DeepSpeed/多卡、不 vLLM、不做吞吐优化。
- 不获取 benchmark 数据；只用 scripted/debug 环境。
- 不改 idea.tex、AGENTS.md；除 §2.2 两项加法外不改第 1–4 层；不处理历史遗留（README 等）。
- 不写审批/密封仪式；不启动"审计/评审/找缺口"类子代理。

## 8. 停止条件

- §6 验收达成 → 简短总结，停机，等 phase 6 goal。
- 被必须人类介入的事情阻塞（权重、私钥、GPU 服务器）→ 说明缺什么，停机。
- 发现本 goal 与第 2–4 层既有接口存在无法用 §2.2 两个加法解决的缺口（例如恢复语义需要更多 backbone 状态）→ 列选项与建议后停机请示，禁止静默扩大下层改动。
- 断点恢复或累积等价测试反复对不上（3 次实现尝试后仍有系统性偏差）→ 停机报告现象与已试路径——那说明某份状态的存取被漏了，值得人看一眼而不是继续猜。
