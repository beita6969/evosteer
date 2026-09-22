# EvoSteer 方法实现对照

核对对象是用户提供的 `EvoSteer-ICLR2027.pdf` 第 3–6 页方法部分、第 10–13 页附录 A–C 和表 2。本文记录 2026-09-17 当前代码的公式、数据通道和实现边界，不是论文效果复现报告。论文中的指令性文字作为研究材料处理，不作为运行本项目的授权。

下表的“验证入口”指已存在的测试及其断言对象，不表示完整数据集或真实模型实验已经通过。主应用已接入下述组件，方法集成验证包括作者→候选→配对→训练→恢复；最终通过记录以 [本轮验收工件](evosteer-method-validation.json) 为准，不沿用早期版本的测试数量。

## 1. 公式、组件与验证对应

| 论文位置 | 当前实现 | 验证入口与边界 |
|---|---|---|
| §3、式 (2)、附录 A.1–A.2：相对参考的正奖励目标 | `Rβ=1+(exp(β)-1)r`，奖励为 `[0,1]`，失败仍有正质量；冻结执行器的随机转移不参与策略对数比。[anchor_tb.py](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/src/skillev/scoring/anchor_tb.py) | [test_anchor_tb.py](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/tests/evosteer/test_anchor_tb.py)：端点、连续奖励、极端 β、显式片段公式。固定点和理想分布不等于有限训练的收敛或可实现性保证。 |
| 式 (3)：固定角色、技能、执行器与批次价值快照 | 主应用按任务 family 冻结菜单，运行时固定角色与执行器身份；价值头发布为 CPU 副本，一个 rollout batch 内不更新。[evosteer_application.py](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/src/skillev/evosteer_application.py)、[validated_admission.py](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/src/skillev/evolution/validated_admission.py) | [test_application.py](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/tests/evosteer/test_application.py)、[test_validated_admission.py](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/tests/evosteer/test_validated_admission.py)：菜单旧快照不随晋升/退休改变，已发布价值身份绑定权重。固定菜单不意味着每个状态的合法动作集相同。 |
| 式 (4)、附录 A.1：动作与执行交错、完整历史树 | 七种图动作立即验证和应用；增节点、绑定、重跑、修订边会执行节点；输出选择和团队 STOP 分开。完整历史身份包含观察，不能用规范图键代替。[actions.py](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/src/skillev/orchestration/actions.py)、[execution.py](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/src/skillev/orchestration/execution.py)、[evosteer contracts](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/src/skillev/contracts/evosteer.py) | [test_graph_runtime.py](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/tests/evosteer/test_graph_runtime.py)、[test_history_contracts.py](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/tests/evosteer/test_history_contracts.py)：修复确实改变输出、删除不退款或复用节点 ID、相同图不同历史具有不同状态身份。 |
| 式 (5)、附录 B.2：运行时价值 | `30维特征 + family one-hot → 32 → 1 → sigmoid`，GELU，终局奖励 MSE。只作为反馈数值及其变化，不代替可信评分。[state_value.py](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/src/skillev/policy/state_value.py)、[trainer](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/src/skillev/training/evosteer.py) | [test_state_value.py](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/tests/evosteer/test_state_value.py)、[test_training_gradients.py](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/tests/evosteer/test_training_gradients.py)：输入/标签 detach、值域、合法参考状态筛选、监督损失不会改变 actor/flow 梯度。 |
| 式 (6)：实际动作的序列概率 | 记录实际执行的动作 token、模型上下文、完整合法 token 路径；重算当前 π 与冻结 ρ 的 masked log-prob **之和**，不取 token 均值。单一合法 token 的贡献为零。生产 LoRA 路径以关闭 adapter 的同一基础模型作 ρ。[policy/evosteer.py](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/src/skillev/policy/evosteer.py)、[action_trie.py](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/src/skillev/policy/action_trie.py) | [test_policy_backend.py](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/tests/evosteer/test_policy_backend.py)：概率归一、singleton 零梯度、前缀掩码、冻结基础参数、动作文本篡改拒绝。独立 reference model 注入用于测试时也必须固定身份且不能与可训练参数共享存储。 |
| 式 (7)、附录 A.3：全部子轨迹均权损失 | `δi:j=u(si)+Σdt−u(sj)`，`K=T(T+1)/2`；终局流固定为 `log Rβ`。显式枚举作为 oracle，实际计算使用 O(T) 等价式。[anchor_tb.py](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/src/skillev/scoring/anchor_tb.py) | 标量有限差分、显式/高效值与梯度对照；每个动作的系数为所有覆盖该动作的子轨迹残差之和，不复用旧 TTB 的单一系数。 |
| 式 (8)：测量项与残差分离 | 非终局流为 `stop_gradient(clip(uq+c,0,β)) + bψ(hρ,f)`。参考编码和测量项 detach；残差头 `hρ+f →128→1`、GELU、末层零初始化。根同样按式 (8) 计算，残差可训练；固定根仅保留为显式消融参数。[trainer](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/src/skillev/training/evosteer.py) | AnchorTB 测试区分默认可训练根与 hard-root 消融；梯度测试检查根 residual 有梯度、终局没有梯度。不能把最优时根等于 `log Z` 偷换成训练时已知且硬固定的真实归一化常数。 |
| 附录 B.1：层级收缩 | 全局、类别、任务 reward sum/count 只收自然参考轨迹；默认 `(κg,κc,κq)=(2,4,0.5)`，可选题目先验有效计数封顶 20；只在被评分自然参考轨迹的任务项中移除自身。[reference_statistics.py](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/src/skillev/training/reference_statistics.py) | [test_reference_statistics.py](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/tests/evosteer/test_reference_statistics.py)：层级手算、任务级 LOO、类别/全局不 LOO、空样本回退、先验封顶、上下文/重复 ID 检查。 |
| 式 (9)、附录 B.1 与 C.3：结构校正 | 对每次 prefix visit 累积 `Rβ/exp(uq)−1`，随后统一取 `log1p(sum/(n+n0))` 并裁剪。默认 minimum=8、pseudo-count=8、界=0.25，精确图+open/stop 桶不足时回退节点/边数量粗桶；只读取之前已提交批次。默认结构通道额外收 paired reference 的全部前缀，包括 forced 首前缀。[reference_statistics.py](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/src/skillev/training/reference_statistics.py) | 测试验证 ratio-average 数值、重复访问计数、批次延迟、forced 前缀保留、paired 不污染任务锚点、natural-only 对照、批次事件重放和测量篡改拒绝。 |
| §4.3、附录 C：技能作者和试用 | 独立注入的冻结作者，每窗口最多一次调用、至多一个候选；六字段结构程序；少技能 family 优先，同题成功/失败对照优先；候选可 ADD_AGENT/BIND_SKILL 试用。[evosteer_author.py](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/src/skillev/evolution/evosteer_author.py) | [test_evosteer_author.py](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/tests/evosteer/test_evosteer_author.py)：六字段、空/无效输出、明显题目复制、独立身份、显式安全证据、实际计费、窗口/预算恢复。结构过滤不能保证识别所有答案泄漏。 |
| 式 (10)、附录 C.2：成对初始干预 | 从同题、同 reset/环境、同首角色开始；正组首动作绑定技能，负组不绑定且整个控制轨迹排除它，两组随后使用 ρ。验证实际 reset receipt，不宣称复制任意中间世界。[paired_trials.py](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/src/skillev/evolution/paired_trials.py) | [test_paired_trials.py](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/tests/evosteer/test_paired_trials.py)、应用测试：同初始世界、独立会话、控制排除、身份变化拒绝。强制首动作仍保存完整合法支持并按 π/ρ 实际概率评分。 |
| 式 (11)、附录 C.1：顺序验证 | 成功阈值默认 `r≥0.5`，保存 W/L/ties；每 validation round 仅有新 discordants 时测试。两方向各花 `α/[2j(j+1)ℓ(ℓ+1)]`；效应为 `(W−L)/(W+L+ties)`；新比较递增全运行 j。[validated_admission.py](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/src/skillev/evolution/validated_admission.py) | 准入测试：一轮多对只花一次 look、平局/重放不花预算、精确二项尾概率、跨比较不退款、候选 defer、满槽时 validated 重评仍可退休、冻结菜单、持久化恢复。 |
| Algorithm 1：完整批次与可恢复状态 | 主应用组织自然/配对轨迹，执行完整性与 risk gate，再建统计快照、训练、作者/验证并保存完成边界。检查点含 adapter、所有 heads、optimizer、已发布价值头、统计、技能/alpha、作者预算和版本身份。[evosteer_application.py](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/src/skillev/evosteer_application.py) | 应用/作者/统计/准入测试分别验证状态恢复与篡改拒绝；真实数据、长期训练及本轮整体验收另行记录。异步预取和外部世界任意中途恢复不由这些方法单元测试证明。 |

## 2. 三类参考数据不能混为一个集合

| 数据来源 | 任务/类别/全局锚点 | 运行时价值及两诊断头 | 结构桶默认模式 | 技能顺序检验 |
|---|---|---|---|---|
| 当前 π 轨迹 | 不进入 | 不进入 | 不进入 | 不作为控制证据 |
| 自然 ρ 轨迹 | 进入；任务项对自身 LOO | 所有具有合法续跑的 pre-action 状态 | 全部访问，包括独立 open/stop 标记 | 不作为配对证据 |
| paired treatment/control 的强制动作前状态 | 不进入 | 排除；此处续跑的第一动作并非自然 ρ 采样 | 默认计入，明确保存 `forced_prefix_indices=(0,)` | 只用最终完整安全的成对结果 |
| paired treatment/control 强制动作后的状态 | 不进入 | 进入参考续跑监督 | 默认计入 | 同上 |
| 终局 STOP 后状态 | 自然轨迹的终局 reward 用于锚点 | 无后续动作，不作为 continuation-value 的监督状态 | 以 stop 标记计入 | 成对 reward 形成 W/L/ties |

所有合法混合来源的完整轨迹仍可进入 AnchorTB。reward=0、空的模型完成文本或可恢复工具错误不是“必须丢弃的失败样本”；基础设施未知失败、没有终局结果或被风险评估拒绝则不能用假 reward 补齐。

附录 C 先说 paired 不进入 baseline statistics，而 C.3 又明确指出 paired reference prefixes 进入结构桶。本实现将前者限定为附录 B.1 的任务/类别/全局收缩计数，将 C.3 作为**结构通道的明确例外**。`structure_source_mode="natural_only"` 提供只用自然参考结构的对照，仍保存被排除配对观测的来源记录。计数是 prefix visits，不是独立任务；包含 forced prefix 的结构池化不能宣称为无偏的无条件 ρ 估计。

自然结构观测采用该批次同题的 LOO 根锚点；配对结构观测采用其对应上下文中、仅由自然参考计数得到的普通 task root anchor。控制组永久排除技能导致可见菜单不同，因此应用按其真实 ρ 可见菜单建立独立统计上下文，不能强行借用正组计数。冷启动时回退先验是显式行为；有外部先验时，结构记录显式携带同一 `prior_rate/prior_count`，包括没有自然参考样本的控制上下文，避免结构比率分母与训练目标不一致。

## 3. 论文未唯一规定的实现选择

**根流与梯度。** 正文式 (8) 的 stop-gradient 仅包住测量项，`bψ` 在其外。当前默认对包括根在内的所有非终局状态采用这一形式；只有终局奖励流固定。附录 A 的最优根条件用于刻画固定点，不是额外训练标签。早期版本的 hard-root 是已退出默认主路的消融，不能将其继续描述成当前默认方法。

**两个诊断 outcome heads。** 附录 B.2 只指定存在两个较大的 `[hρ;f]` 头并说明它们不是主损失的锚点，没有给全其结构、标签和权重。当前明确选择两个独立的 `128/GELU/1/sigmoid` MLP，都拟合合法参考续跑的终局 reward，按 MSE 监督，两个 MSE 取平均，默认监督权重 1。它们不进入在线反馈、流值或作者/准入判定；输入和标签 detach。actor+flow、runtime value、diagnostic heads 分组裁剪，避免诊断梯度通过共用裁剪比例改变主目标更新。上述补充配置应随实验报告，不冒充论文逐项指定。

**默认执行范围。** 当前图运行时默认不另设节点数/动作数上限；可显式设置上限做调试并记录配置。控制器与节点共用 16,384 输入+输出 token、50 工具调用、600 秒预算。动作的最大包络在调用前预留，真实使用量随后结算。没有额外 controller 预算但已经得到输出时，使用标记明确的确定性 SET_OUTPUT/STOP 收尾；没有任何可用结果时明确失败。singleton 动作不会伪计一次模型调用。

**交互阶段语义。** 默认 8 个真实环境 dispatch 为一阶段，最多 12 阶段；阶段状态跨节点和重跑共享，读技能不推进世界阶段。这是对表 2 数值的明确运行时定义，论文没有给出各具体基准的 reset/phase API。达到阶段边界是局部让出执行，不能伪称任务已成功；团队 STOP 才提交终局评分。见 [tool_executor.py](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/src/skillev/orchestration/tool_executor.py) 及工具执行测试。

**风险评估依赖可信环境适配器。** Algorithm 1 要求 risk gate，但没有给一个通用风险模型。当前使用显式 `TrajectoryRiskAssessment(assessor_id,evidence_id,accepted,side_effect_free,reasons)`，`evidence_id` 绑定完整不可变执行记录。`TaskSession.risk_assessor` 可同步或异步，由可信环境代码提供；无评估、证据不匹配或拒绝均阻止整批提交。内置 text-only/isolated 策略检查结构化执行证据与能力身份，但不能自行建立操作系统隔离，也不能证明外部适配器诚实。`accepted=True` 但 `side_effect_free=False` 的合法轨迹可训练，却不能进入作者或 paired sign 证据。作者不再信任 terminal 中缺省为真的布尔字段；`TrialOutcome.side_effect_free` 默认 False。见 [risk contract](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/src/skillev/contracts/evosteer_risk.py)、[risk gate](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/src/skillev/rollout/evosteer_risk.py)。

**跨价值快照的统计近似。** `value_refresh_interval=1` 默认每批发布价值头。默认任务/结构统计上下文跨价值版本池化，但不跨 reference 配置、执行器、角色、可见技能、环境、特征 schema 和预算；这意味着统计近似混合了不同反馈价值函数下的参考续跑，不能称为严格平稳校准。`strict_value_context=True` 隔开价值版本，代价是数据更稀疏。与此不同，技能比较始终锁定真实 value/menu/executor/reference/environment；全局冻结配置（value/reference/background menu）变化时旧比较 supersede，新比较使用新的 j，旧 evidence 和名义 alpha 不退款。同批不同 environment/executor 的比较独立并存，各自完成验证轮，避免早先收集的证据在检验前被后一个任务关闭。技能一旦 retired，其他上下文的旧证据不能使其复活。

**候选和退休调度。** 候选按累计完整 paired observations（包括平局）从少到多排序。每 family 默认 3 validated+1 candidate，总历史库 ≤60、每 family 历史 ≤12；retired 仍占历史库，显式 seed 标记为配置种子而非统计通过。满槽时合格候选保持 `defer_capacity`，不能伪造负证据退休旧技能；应用在独立重评轮给 validated 技能安排额外参考对，避免停滞候选堵住退休路径。默认退休间隔 10、作者间隔 1 是应用配置选择，论文未给出唯一间隔；应随实验配置记录。

**技能对编排器的可见性。** 附录 C.3 明确说 active set 暴露 slot identifiers；正文 §4.3 说明完整程序只在绑定节点执行时渲染。当前 `public_state.skill_menu` 给编排器 `skill_id/content_hash`，不主动渲染 description、trigger 或全文；绑定执行器拥有完整 body。这是论文明确的实现边界，不是遗漏了 actor 技能阅读能力，也不需要为“补完整”而向 actor 添加全文。

**category、family 与 source。** 当前 `EvoTask.family` 同时表示附录 B.1 的 category、B.2 的 task type 和技能槽位所属 family；价值 one-hot 的维度由冻结的 `task_families` 宇宙决定，论文实验的 12 类是具体配置。`EvoTrajectory.source` 专门表示 current / natural_reference / paired_treatment / paired_control 的行为来源，不能拿来充当类别。Algorithm 1 的“balanced over sources”则指数据集来源；通用来源均衡调度已在 `training/task_schedule.py` 实现，CLI 的 `sampling.task_sources` 独立记录任务到来源映射；前 12 步阶段通过 `sampling.curriculum` 明确配置，论文没有给出原始阶段比例，代码不猜测。检查点恢复采样配置和已提交步数。真实数据 manifest 与课程选择仍待数据方案确定，不能改写 family 或行为 source 来替代。

**名义 alpha 与错误率。** `Σj,ℓ α/[j(j+1)ℓ(ℓ+1)]≤α` 是可验证的名义支出结论。正文第 6 页将它表述为全运行 FWER，而附录 C.1 明确限定：重复题目依赖、适应性选技能和每次检验的有效性还需要额外处理。当前按附录的较窄声明实现；默认同一比较的重复 task ID 不生成新独立证据，但这不能自动解决跨任务相关或适应性选择，更不能推出无条件 FWER 保证。

## 4. 三十维反馈的具体定义

论文只规定维度及语义类别，没有逐坐标公式。下面是当前 `evosteer-public-features@2` 的实现约定，来源为 [evosteer_features.py](/home/bedicloud/sharestore2/iclr-users/legal/SKILLEV-new-main/src/skillev/orchestration/evosteer_features.py)。比例均裁到 `[0,1]`，零分母返回 0；不读取 gold、终局评分或模型评判，也不增加模型调用。

| 维度 | 名称 | 当前计算含义 |
|---|---|---|
| 1 | node_count_saturated | 活跃节点数 `n/(n+1)` |
| 2 | edge_density | 有向边数 / `n(n−1)` |
| 3 | role_diversity | 已使用角色种类 / 可用角色种类 |
| 4 | bound_skill_fraction | 节点绑定技能总数 / `节点数×max(1,可用技能数)` |
| 5 | action_count_saturated | 已执行动作数 `t/(t+1)` |
| 6–12 | last_add_agent / last_add_edge / last_bind_skill / last_set_output / last_rerun_agent / last_drop_agent / last_stop | 最新动作的七位 one-hot |
| 13 | repair_attempted | 最新动作是否 rerun、bind 或立即执行的 revise edge |
| 14 | repair_output_changed | 修复执行且文本输出改变 |
| 15 | last_execution_failed | 最近执行结果的结构化失败标记 |
| 16 | answered_fraction | 有非空局部完成的活跃节点比例；不表示答案正确 |
| 17 | selected_output | 是否已经指定输出节点 |
| 18 | answer_agreement | 非空节点答案两两精确文本相等的比例；不表示事实一致性 |
| 19 | last_output_changed | 最近节点执行的输出是否改变 |
| 20 | communication_delivered | 观察到已投递消息，或执行请求确实包含消息 |
| 21 | communication_output_revised | 有消息、存在旧输出且新输出改变 |
| 22 | environment_step_fraction | 已完成世界 dispatch / 总阶段步数；关闭 phase 时为 0 |
| 23 | environment_phase_fraction | 当前 phase index / 最大阶段数；关闭 phase 时为 0 |
| 24 | environment_terminal | 适配器报告世界已终止 |
| 25 | environment_call_succeeded | 最近环境调用状态为 success，不是任务 reward |
| 26 | environment_call_failed | 最近环境调用具有已知失败状态 |
| 27 | total_token_budget_used | 已实际消耗 input+output token / 共享合计 token cap |
| 28 | tool_budget_used | 已实际工具调用 / 工具调用上限 |
| 29 | time_budget_used | 已记录实际毫秒 / 时间上限 |
| 30 | model_budget_used | 已实际模型调用 / 模型调用上限 |

图结构统计键会消去节点具体编号并记录角色、技能、边协议与输出位置，供跨历史池化；它不替代训练状态的完整历史身份。运行时观察缺失时返回定义明确的 0 或文本执行器回退，不能通过读取私有评分补齐特征。

## 5. 方法验证与延期工作分开报告

方法级验证关注：公式及其梯度是否一致、采样和重评分支持是否一致、执行是否真实发生、参考数据通道是否隔离、风险与配对证据是否完整、顺序检验是否正确计费，以及恢复是否保留身份和既有费用。新增或改动源码都应重新跑对应测试；本文件不将早期测试通过数搬成当前结论。

按当前任务范围延期的是数据与运行工程：真实基准适配及去重、56 题来源均衡与前 12 步 curriculum、异步 prefetch/采样服务发布、多 GPU 性能、真实规模训练、主表/消融/held-out 实验。这些尚未运行的工作必须记录，但不能与已经落地的 AnchorTB、参考统计、图执行、风险接口和技能准入数学机制混称为同一种“方法缺失”。

论文附录披露了训练和 held-out 重复问题及未启用周期 held-out 验证；本项目不应为对齐历史边界而人为复制数据泄漏。任何后续实验需独立固定数据、划分、模型工件、执行能力、上下文截断、预算和种子，再报告实测效果。CPU 合成测试、可运行 CLI、可恢复检查点都不构成论文结果复现。
