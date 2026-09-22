# phase2-goal.md —— Bayesian TTB Self-Skill-Revolution Loop：第 2 轮 · 模型层（policy backbone）

> **Protocol-v3 immutable amendment (2026-07-25).** 本文件是历史实施记录，不再是当前跨层语义权威；当前权威见 `method-v3-final-spec.md`。最终 production backend 是显式 pinned `QwenPolicyBackbone`/`QwenTokenizerAdapter`；不做模型、tokenizer、hidden-size、device 或 authoring 能力探测。`reset_z(seed)` 必须显式种子且确定；checkpoint 仅保存固定三组 trainable state 与版本，不存在 prepare/apply/recover reset、fingerprint、attestation 或旧 schema reader。历史段落中的 `generate` 以最终拆分后的 `generate_policy`（raw softmax）和 `generate_base`（authoring）为准。

本文件是本轮唯一任务清单，接续已验收的第 1 轮（轨迹数据层 contracts）。写法是讲解式的：每节先说做什么，紧跟着解释为什么、以及做错时的典型症状。若某处字面指令与所附理由看起来冲突，按理由描述的意图行事并留一句 note 说明。

## 0. 定位：八层导航图中的第 2 层

八层规划回顾：(1) 轨迹数据层（已完成——`src/skillev/contracts/ttb_*.py`，七契约 + 全部不变量测试已在 main 上）；**(2) 模型层（本轮）**；(3) 打分与目标层（纯函数：P̂_F/P̂_B 汇总、Δ、L_TTB）；(4) rollout 引擎；(5) 训练循环；(6) 流诊断；(7) Bayesian 校准；(8) 进化层。本轮只做第 2 层——**其余任何一层的逻辑都不写**。

模型层是什么（人话）：整个系统的"电闸房"。它是**全仓库唯一允许 import torch/transformers/peft 的地方**，负责五件事：加载冻结底座、挂载并切换 π_θ/P_φ 两套 LoRA adapter、持有 log Z_θ(q) 头、提供 `generate`（采样）与 `score`（teacher-forced 带梯度逐 token logprob）两个能力、管理版本与检查点。所有上层（打分、rollout、训练）只碰开关面板（接口），永远不进机房（不直接摸模型或 tokenizer）。

为什么单点化值三份回报：(a) 测试时可以用一个随机初始化的 tiny 模型整体替换底座，接口不变，CPU 上手算对照 score 的每一个数；(b) 将来两栈演进时，vLLM 只替换 `generate` 的实现，score/训练侧一行不改；(c) 换底座型号（9B→别的）变成改一个配置项。反面警示（这是本轮最重要的一条纪律）：**如果任何上层代码自己去摸 tokenizer 或模型对象，两栈阶段必然出现"采样和打分用了不同 tokenizer/adapter"这类无法定位的 bug**——所以接口全部只收发 token id 与张量，文本→token 的转换只发生在轨迹记录时（第 1 层的 TokenizerProtocol），而生产 tokenizer 的**唯一出口**就是本层的 `backbone.tokenizer` 属性。

## 1. 方法权威与翻译关系

本轮把 idea.tex 的三个可训练组件翻译成一个接口：式 (4) 的前向策略 π_θ 与式 (5) 的后向策略 P_φ → 同一冻结底座上两套**独立** LoRA adapter（`AdapterRole` 区分）；式 (7) 的 log Z_θ(q) → 底座池化表示上的独立 MLP 头；§2.5.2 的"Z 重初始化" → `reset_z()`。**注意分工**：hindsight 前缀怎么拼（含不含 r_t）、K_t 归一化、Δ 的四项结构——这些都**不是**本层的业务，是第 3 层的；本层只保证"给我 prefix ids + action ids + 角色，我还你这段 action 每个 token 的带梯度 logprob"。把归一化留在上层的原因：模型层不做任何 TTB 数学，它才可能被 tiny 模型/未来引擎无损替换。

**钉死的**：三个组件可训练且参数化（idea.tex 的字面要求）；两个策略是**两个不同条件分布**（"dual-policy system"），所以两套 adapter 必须独立、梯度互不渗漏；Z 可廉价重置。**工程默认（一句话理由即可调）**：底座 Qwen3.5-9B、bf16、LoRA r=16/α=32/dropout=0.05/目标模块 attention+MLP 投影、Z 头两层 MLP 宽 256、Z 的池化在 adapter 全部停用的冻结底座表示上做。优先级照 AGENTS.md；idea.tex 与 AGENTS.md 都不要动。

## 2. 范围刀切

- 本轮**写 torch 代码但不写 TTB 数学**：没有 Δ、没有 L_TTB、没有 ÷K_t、没有优化器循环、没有 rollout、没有诊断/后验/进化。测试里允许对 score 的和做一次 `backward()` 验证梯度流向——那是验收接口，不是训练。
- 上一轮 audit 提到的仓库摆放问题（README 引用、CI 启用、AGENTS/benchmark guide 入库等）**一律不处理**，Owner 已裁决忽略；除非本轮工作字面需要，不要去碰。
- 仓库里 bootstrap 期留下的 `runtime/bounded_agent.py` 有 score/logprob 的接口占位——本层是权威；如有重叠，以本层接口为准做最小调和，不扩大范围。

## 3. 组件逐个用人话讲（含代码骨架）

骨架规则同 phase1：字段语义与不变量不可减，命名与文件组织自由；注释说"需要写"的就是实现规格。

### 3.1 接口总闸：PolicyBackbone

**是什么**：上层世界看到的全部模型能力，一个 Protocol。**为什么先定接口**：第 3/4/5 层将来只 import 这个接口；tiny 测试底座和真 9B 底座是同一个类不同配置，而不是两套代码——接口错一处，三层跟着返工。

```python
class AdapterRole(StrEnum):
    """两个策略的身份标签。需要写：FORWARD_POLICY = "forward-policy"（π_θ）、
    BACKWARD_POLICY = "backward-policy"（P_φ）两个成员，封闭集合。
    所有 score/版本查询都用它定位 adapter——字符串裸奔会拼错且无法穷举测试。"""


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """一次采样请求。ids 进、ids 出——文本不经过本层。"""

    input_ids: tuple[int, ...]        # 完整 prompt 的 token id（由上层用本层导出的 tokenizer 编好）；非空
    max_new_tokens: int               # > 0
    temperature: float                # > 0（贪心用极小值不用 0，避免分支语义）
    top_p: float                      # (0, 1]
    seed: int                         # 本次采样的显式种子——可复现性不靠全局状态
    decoding_snapshot_id: str         # 参数快照引用，落进 TrajectoryRecord 的同名字段；非空


@dataclass(frozen=True, slots=True)
class GenerationResult:
    output_token_ids: tuple[int, ...] # 只含新生成的 token（不回吐 prompt）
    finish_reason: str                # "stop" / "length" 等封闭小集合


class PolicyBackbone(Protocol):
    """电闸房面板。上层只见此接口。"""

    @property
    def backbone_id(self) -> str:
        """需要写：底座标识 + 配置 + 当前检查点的内容哈希拼成的稳定 id。
        进各类记录，回答"这是哪台模型出的数"。"""

    @property
    def tokenizer(self) -> TokenizerProtocol:
        """生产 tokenizer 的唯一出口（第 1 层契约校验、第 4 层组装都从这拿）。
        需要写：返回钉死 tokenizer 的薄包装，tokenizer_id 稳定。"""

    def adapter_version(self, role: AdapterRole) -> str:
        """需要写：该 adapter 当前版本串（检查点 id + 训练步数约定）。
        它就是 EdgeLogprobRecord.forward/backward_adapter_version 的来源。"""

    @property
    def z_version(self) -> str:
        """Z 头当前版本；每次 reset_z 必变。"""

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """采样。需要写：按 request 的参数与种子采样；两栈阶段 vLLM 只替换这一个方法。
        用哪个 adapter：固定 FORWARD_POLICY（采样永远是 π_θ 的事，不留参数以免误用 P_φ 采样）。"""

    def score(
        self, prefix_ids: tuple[int, ...], action_ids: tuple[int, ...], role: AdapterRole
    ) -> "torch.Tensor":
        """teacher-forced 打分。需要写：返回一维张量，长度 == len(action_ids)，
        第 i 个元素 = 在 prefix+action[:i] 条件下 action[i] 的自然对数 logprob，
        **带梯度**（梯度只流向 role 指定的 adapter）。
        本层不做求和、不做 ÷K_t——归一化是第 3 层的业务。
        prefix_ids 非空（至少一个条件 token）；两个入参都是 ids，本方法无权 tokenize。"""

    def z_value(self, context_ids: tuple[int, ...]) -> "torch.Tensor":
        """log Z_θ(q)。需要写：返回标量张量，梯度只流向 Z 头。"""

    def reset_z(self) -> str:
        """§2.5.2 的 Z 重初始化。需要写：重新初始化 Z 头参数（新随机种子）、
        z_version 换新并返回。注意：优化器状态的清空是第 5 层的事，这里只管参数与版本——
        接口注释里写明这个分工，防止将来两边都不做或都做。"""

    def trainable_parameters(self, component: str) -> "Iterator[torch.nn.Parameter]":
        """供第 5 层建优化器。component ∈ {"forward-policy", "backward-policy", "z-head"}，
        分别只产出对应组件的参数——第 5 层按组件设不同学习率就靠这个分组。"""

    def save_checkpoint(self, directory: str) -> None: ...
    def load_checkpoint(self, directory: str) -> None:
        """需要写：两套 adapter + Z 头 + 三个版本串的存取；load 后 score/z_value
        输出与 save 前逐位一致（往返测试断这个行为）。底座权重不存（冻结的，只存路径引用）。"""
```

### 3.2 底座与双 adapter：HFPolicyBackbone

**是什么**：接口的 transformers+peft 实现，tiny 测试底座与真 9B 是**同一个类**吃不同 `BackboneConfig`。**为什么强调同一个类**：单独写个 FakeBackbone 会让测试测的不是生产代码——tiny 随机模型必须走完全一样的 forward/gather/切换路径，才配叫"手算对照"。

```python
@dataclass(frozen=True, slots=True)
class BackboneConfig:
    """换底座=改这里；一个配置项，不是一次改造。"""

    base_model_path: str              # 本地权重路径或 HF id（生产=直连 GPU 服务器上的 Qwen3.5-9B 路径，
                                      # 路径经私有配置传入，不写死进仓库）
    torch_dtype: str                  # 默认 "bfloat16"；tiny 测试用 "float32"（CPU 手算对照要精度）
    lora_rank: int                    # 默认 16
    lora_alpha: int                   # 默认 32
    lora_dropout: float               # 默认 0.05（注意：score 时必须 eval 模式关 dropout，
                                      # 否则打分带随机性，重放对不上——这是个典型隐蔽错误）
    lora_target_modules: tuple[str, ...]  # attention+MLP 投影层名；随底座架构定
    z_hidden_width: int               # 默认 256
    device: str                       # "cuda"/"cpu"/"auto"


class HFPolicyBackbone:
    """需要写的要点（按初始化顺序）：
    1. 加载底座后立即整体冻结：requires_grad_(False)、eval()；梯度检查点的开关
       留 config 项（本轮不用，第 5 层要）。
    2. 用 peft 挂两套**独立命名** adapter（名字 = AdapterRole 的值），score/generate 前
       set_adapter 切换到目标角色；确保一个 adapter 激活时另一个的参数既不参与前向
       也不收梯度。
    3. tokenizer 随底座加载，包装成 TokenizerProtocol 从 .tokenizer 暴露；
       全类内部除该包装外不得有第二个 tokenizer 引用被外部拿到。
    4. score 的实现见 §3.3（错位对齐是全轮最容易写错的一处）。
    5. z_value 的实现见 §3.5。
    6. 版本管理见 §3.6。"""
```

### 3.3 score：teacher-forced 打分与"栅栏错位"

**是什么**：把 prefix+action 拼成一条序列做**一次**前向，从 logits 里取出 action 每个 token 的 logprob。**为什么单列一节**：这里有个经典的 off-by-one——decoder 的 `logits[i]` 预测的是 `token[i+1]`。设 prefix 长 P、action 长 K，action 占位置 P..P+K−1，那么它们的 logprob 要从 **logits 的位置 P−1..P+K−2** 取。错一位不报错、训练照跑、loss 照降——只是每个 token 的"分"都是隔壁 token 的，这正是 phase1 让 TrajectoryRecord 存 token 边界、这里必须手算对照的原因。

```python
# score 实现规格（写在 HFPolicyBackbone.score 里）：
# 1. 断言 prefix_ids 非空、action_ids 非空；
# 2. full = prefix_ids + action_ids，一次前向（eval 模式、按 role set_adapter）；
# 3. 对齐：取 logits[P-1 : P+K-1]（P=len(prefix), K=len(action)），log_softmax 后
#    gather 出 action_ids 各自的 logprob——注释里必须画出这个位置图；
# 4. 计算可在 bf16 前向，gather 之后 cast 成 float32 返回（记录与比较要稳定精度）；
# 5. 返回一维张量，长度 K，requires_grad=True 且梯度只到 role 的 adapter；
# 6. 全程不改全局随机状态（score 不许有随机性——dropout 必须关死）。
```

### 3.4 generate：采样能力

**是什么**：ids 进 ids 出的采样。**为什么这么薄**：结构化动作约束、r/a 切分、停止条件这些都是第 4 层 rollout 的业务；generate 越薄，vLLM 替换越无感。需要写：固定使用 FORWARD_POLICY adapter；用 request.seed 建**局部** generator（不碰全局种子——上层可能在同进程并发多个环境）；输出只含新 token；finish_reason 如实。

### 3.5 Z 头与 reset_z

**是什么**：log Z_θ(q) = 两层 MLP（宽 256，标量输出），吃底座对 context ids 的最后层 mean-pool 表示。**两个关键设计**：(a) 池化表示在 **adapter 全部停用**（peft 的 disable_adapter 上下文）的冻结底座上取，且底座前向 no_grad——这样 Z 的输入表示不随 π/P_φ 的训练漂移，Z 的梯度只有 MLP 自己的，语义干净还省显存；(b) `reset_z()` 只重初始化 MLP 参数并换 z_version——"Z 重置后训练继续"（idea.tex §2.5.2 末句）的可审计性就靠版本串，第 1 层 TTBBatchStats 的 library_version 分段与它配合。**典型症状**：忘了 (a)，π_θ 一更新 Z 的输出就跟着动，第 3 层会把这解读成 Δ 的漂移，训练诊断全花。

### 3.6 版本、检查点与 tokenizer 出口

需要写：`adapter_version(role)` / `z_version` 的取值约定 = `"<checkpoint_id>@<step>"` 形态的稳定串（step 由第 5 层将来递增，本轮固定 0）；`save_checkpoint`/`load_checkpoint` 存取两套 adapter 权重 + Z 头 state_dict + 三个版本串 + BackboneConfig 摘要（底座只存路径引用，不复制权重）；load 后 score/z_value 与 save 前逐位一致。`backbone_id` = 底座标识+配置+检查点的 stable_hash。tokenizer 包装的 `tokenizer_id` 必须含底座 tokenizer 的内容指纹——两个不同底座绝不能共享同一个 tokenizer_id。

### 3.7 tiny 底座与测试策略

**是什么**：用 transformers 的小配置（如 Qwen 系小架构：hidden 64、2 层、2 头、vocab 128）**随机初始化**、CPU、float32，喂给同一个 HFPolicyBackbone 类。无网络、无权重下载。**测试清单（每条都是行为断言）**：

- **手算对照（本轮的显微镜）**：tiny 模型上手写 forward + log_softmax + gather，逐位对照 `score()` 的输出；再构造一个已知错位实现（取 logits[P..P+K-1]）证明测试能抓住栅栏错误。
- **梯度隔离**：对 forward 角色的 score 求和 backward → 只有 forward adapter 参数有梯度，backward adapter 与底座梯度为 None；z_value backward → 只有 Z 头有梯度。
- **adapter 独立**：扰动 forward adapter 权重 → forward 分变、backward 分不变；切换回来可复现。
- **Z 独立于策略**：扰动任一 adapter → z_value 不变（验证 §3.5(a)）；reset_z 后输出改变且 z_version 换新。
- **确定性**：同 seed 的 generate 逐位可复现；score 无随机性（连跑两次逐位相等）。
- **往返**：save→load→score/z_value 逐位一致。
- **接口纯净**：score/generate 全链路无文本参数；backbone.tokenizer 满足 TokenizerProtocol 并可直接用于第 1 层的 build_trajectory_record。
- **import 边界**：新增一个测试扫描全部 src 模块，断言 torch/transformers/peft 的 import 只出现在模型层包内——这是"唯一触碰 torch 的地方"的机器执法。

## 4. 工程要求

- 新包建议 `src/skillev/policy/`（命名自由），torch/transformers/peft 依赖以独立依赖组加入 pyproject 并进 lockfile（CPU 版即可满足 CI；组织方式一句话理由内自定），make check 必须实际跑到本层测试。
- 版本串、id、哈希沿用第 1 层的 canonical/stable_hash 与既有约定；不自造。
- 契约层（ttb_*.py）语义**不动**；如需模型层专用的小配置类型放在模型层包内，不塞进 contracts。
- 单文件超约 1000 行拆分；测试断行为不断字面量。

## 5. GPU 冒烟（本轮唯一的直连 GPU 任务，有界）

**GPU 执行硬约束**：只通过 SSH 直连私有配置的 GPU 服务器，不得使用
NCSA Delta、Duo、Slurm 或 scheduler 命令。本轮默认使用物理 H800 GPU `0`
并显式设置 `CUDA_VISIBLE_DEVICES=0`（这是该历史轮次的固定选择，不是当前全局限制）。
当前执行必须先检查全部 8 张卡，优先空闲卡；已有他人进程但 GPU 利用率低且该卡总显存
占用严格低于 25% 的，也允许在剩余显存足够时共用，不得终止、修改或调试他人进程。

在直连 GPU 服务器上装载真实 Qwen3.5-9B（权重路径经本地私有配置传入），跑一遍：generate 一次（几十 token）、对一小段人造 prefix/action 做 score 一次、z_value 一次、reset_z 一次、save/load 一次；记录显存峰值、耗时、三个版本串，写一段简短 note（不是仪式文档，commit message 级即可）。目的：证明这套栈在真底座上能跑，第 3–5 层的地基不悬空。**有界**：不训练、不采长轨迹、不接任务环境；torch 环境的搭建用能达成目的的最省路径（现有打包流程能用就用，不行就 venv/pip 直装，一句话记录选择）。若被权重缺失、私钥缺失或 GPU 服务器不可达阻塞：按 AGENTS.md 说明缺什么后停机，不用新代码填等待。

## 6. 验收标准

1. PolicyBackbone 接口 + HFPolicyBackbone 落地，§3.7 测试清单全部实现且全绿（含手算对照、栅栏错误可被测试抓住的演示、梯度隔离、import 边界扫描）。
2. tiny 底座与生产底座是同一实现类不同配置；测试无网络、无权重下载。
3. ruff/mypy/pytest 全绿，make check 绿。
4. GPU 冒烟完成并留 note（或按 §5 的阻塞条款停机说明）。
5. commit + push 到 SKILLEV-new；简短 note 说明本层完成。

## 7. 明确不做

- 不写第 3 层及以上的任何逻辑：无 Δ/L_TTB/÷K_t、无 rollout、无优化器循环、无诊断/后验/进化。
- 不接 vLLM（两栈是后续轮次的事；本轮只保证 generate 的接口形态让它将来可无感替换）。
- 不改 idea.tex、AGENTS.md、phase1 契约语义；不处理上轮 audit 的仓库摆放遗留。
- 不做 benchmark 数据获取；不写审批/密封仪式；不启动"审计/评审/找缺口"类子代理。

## 8. 停止条件

- §6 验收达成 → 简短总结，停机，等 phase 3 goal。
- 被必须人类介入的事情阻塞（权重、私钥、GPU 服务器）→ 说明缺什么，停机。
- idea.tex 或本文件语义歧义无法双向兼容实现 → 列选项与建议后停机请示。
- tiny 模型上手算对照反复对不上（3 次实现尝试后仍有系统性偏差）→ 停机报告现象与已试路径——这大概率是对齐/精度理解问题，值得人看一眼而不是继续猜。
