# 单 owner、无投票与 AIME thinking：后续修复清单

来源：已完整阅读 `SKILLEV_SINGLE_OWNER_NO_VOTING_AIME_THINKING_REPAIR_PLAN_2026-09-07.md`
（852 行）。本次用户要求是**阅读并加入后续问题清单**；下列未勾选项不是已实现、已通过测试
或已获新成绩的声明。文件中的代码和测试是方案示例，不是仓库验证结果。

后续 [来源与恢复集成记录](STEP0_SINGLE_OWNER_SOURCE_IMPLEMENTATION_2026-09-07.md) 区分已实现项
与剩余工作。用户最新已接受 AIME 23/30，当前优先修复 ALFWorld/WebShop；未启动新的
AIME thinking-on 评测。原始清单保留，执行优先级按该最新指示调整。

## 与当前工作合并，而不是覆盖历史

- 新计划检查的是 `4dd26d3`；当前修复源码是 `7235e93`。实施时按实际入口确认差异，
  不把历史 selector 行为误报为当前 clean controller 正在投票。
- 保留已完成的 F01–F16 通信修复、active selector 拒绝路径、peer advice-only、原生指标、
  owner 确定性投影，以及 AIME 等值整数兼容。`7235e93` 的多个等值声明只在原有唯一整数
  投影一致时接受，不按多数选值；它仍须纳入新的真实 final 通道验证。
- **后续 AIME 新条件使用 native thinking-on，全部 30 题一致。** 此项覆盖此前未来轮次
  一律 thinking-off 的限制，不更改正在运行或已经完成的旧条件。其他七项延续现有 off 条件，
  并非永久禁止后续另行声明的条件。独立 reasoning pass 不因 native thinking 而强制增加。
- 当前只评 A2、一个 seed；不因计划讨论 A1/训练后对照而自动启动这些实验。若以后实际对照，
  必须保持同题、同 thinking 和同预算，分别报告 base 与指定训练策略。
- IID 仍为 HotpotQA、TriviaQA、AIME2026、HealthBench、WebShop、ALFWorld、
  **MBPP+ hard**、HumanEval。MBPP+ hard 是项目名称，实际人群为 MBPP v0.2，
  成功需 Base AND Plus；不新增或伪称官方 hard split。
- `idea.tex` 和 `AGENTS.md` 不修改；不因本清单重做训练吞吐实验或改 TTB 方法。

## A：关闭遗留选择入口（低／中风险）

- [x] A1：使 `legacy_candidate_selection.py` 的两套选择器也无条件不可执行；保留历史说明，
  不留环境变量重启开关。兼容 shim 只抛出明确的禁止选择异常。
- [x] A2：修正 `public_validation_guard.py` 关于 reference fallback 的过时说明；
  clean 入口继续拒绝 legacy，历史复现不得发布成 clean 成绩。
- [ ] A3：检查实际入口、依赖与私有启动器，移除答案池、S/T 选择、标签交换投票、
  Step-0 回退和按输出成本挑题追加推导的可执行路径。正常 token `top_k`、技能检索排序
  不等于候选答案选择；不以关键词搜索零命中代替调用关系检查。
- [x] A4：历史 26/30、direct-policy 26/30、组合 27/30 及 Health/Web 选择结果保留原数字，
  在相邻表格明确 withdrawn/composite，不删掉直接策略列后留下组合增益。

验证：合成 S/T、任意偏好或平局均不能执行选择；legacy import/dispatch/实际入口依赖测试；
改变其他题 token 数或调度顺序不改变本题预算、成员或提交次数。拟议脚本检查只处理实际
选择路径，不恢复已经废止的审批、收据或 training-free 门禁。

## B：受评策略与实际 owner 输出闭环（高风险）

- [ ] B1：从冻结配置构造明确的 policy/model/tokenizer/optimizer-step/checkpoint/adapter
  绑定。adapter-free Step-0 保持零更新；训练后使用独立 schema，不放宽旧 Step-0 标签。
- [ ] B2：实际 generator、请求、响应和 final 使用一致策略身份；核对服务中 adapter 与
  指定 checkpoint 的映射。缺失、加载失败、路由不符均失败，不能退回 base。actor/peer
  身份分别记录；Health judge 独立冻结在 Qwen base 路由。
- [x] B3：broker 收到实际生成结果后、返回 actor 前，私有持久化 `model_outputs`：
  run/arm/episode/attempt/call、policy、participant、purpose、原始 token、final token、
  通道状态、finish reason、usage 和实际 adapter route。逐题内容不进入 Git。
- [x] B4：`FinalCandidate` 明确 attempt、owner call、submission 与结构化终止状态，
  不再仅靠逻辑 `final_message_id` 或可选 payload 判断来源。
- [ ] B5：提交、恢复和首次评分共用来源校验：同 scope/attempt/policy/checkpoint，
  owner 已完成的实际 call、存在合法 final 通道、确定性投影一致、来源未重复消费。
  peer 输出、旧 run 输出、另一策略输出不得伪装为本次 owner final。
- [ ] B6：有效 final 立即提交；提交后不能再生成并替换。允许提交前固定预算内的 owner
  自检和公开接口修复，不允许评分后低分重答、隐藏测试修复或 alternative-derivation 重开。
- [x] B7：显式区分 submitted、model_no_final、model_format_invalid、budget_exhausted、
  infrastructure_unresolved。空答案必须有真实调用／通道／预算事件支持；有效 final
  不得被上层丢成空串，基础设施未解决不得伪装为模型零分，也不能找旧答案补空。
- [x] B8：交互任务评的是本 episode 的完整真实轨迹；逐步 action 关联同策略 owner call
  与 native execution acknowledgement，不强制静态文本 final，不从其他轨迹取 reward。

验证：合成伪来源、peer/base/旧 run 冒充、响应与 payload 不符、提交后调用、空答案分类、
崩溃边界和同 call 恢复；覆盖真实 actor/broker 接口，但不读评测答案。

## C：仅恢复原运行，不改写答案归属（高风险，与 B 合并）

- [x] C1：显式区分 formal-fresh、same-run-resume、historical-diagnostic。
  正式新 run 禁用 `--reuse-completed-from` 和跨 run 候选导入；不修改训练 checkpoint 恢复。
- [x] C2：同 run 恢复保留原 run、policy、parser、输入、scorer、thinking 和冻结 controls；
  完整比较条件，不能只遍历旧 controls 的 key。既有候选不能未经 B 的校验直接返回或评分。
- [x] C3：历史诊断保留 origin run，不把旧 candidate 改 run ID 当作新生成；旧 schema /
  `legacy-unverified` 保持 unverified，禁止补造从未存在的 broker 证据。
- [x] C4：同 candidate/scorer 的 definitive score 直接复用，尤其不能隐式重抽 Health judge。
  真正缺失或失败评分仅按冻结恢复规则处理，候选保持不变，不重复 judge 择高。
- [ ] C5：进程恢复只取回同一次已完成 call 的输出；HTTP 丢失时优先凭 request ID 取回原输出。
  无完成证据的调用保留 unknown/infrastructure 及成本，不能免费重采样。

验证：跨 run/arm/attempt/policy/thinking 拒绝、完整 controls 差异、旧 schema 标签、
crash 恢复和 definitive score 复用。禁止跨运行回填不等于禁止同一次输出的正常缓存。

## D：AIME thinking 配置与 final 通道贯通（中风险）

- [x] D1：版本化按八项 benchmark 声明的 thinking map；新条件 AIME=true，其余当前=false。
  全局 arm 默认值不能覆盖 resolved 值；owner 与 peer 使用同一已声明条件。
- [x] D2：贯通 runtime profile → actor payload → tokenizer 模板 → generation → broker
  重新编码 → controls → final/score，实际请求与报告保持一致。
- [x] D3：actor/broker 共用实际 token 边界解析，token ID 来自固定 tokenizer。
  若 API 明确分离 reasoning/content，则使用其字段，不与原始 token 模式悄悄混用。
- [x] D4：区分 complete、reasoning-unfinished、malformed、final-empty；thinking 未闭合、
  多重非法边界或 final 为空时，不能从推理草稿中的整数/boxed 内容提取备用答案。
  合法 final 内等值表示继续接受，不同值不能以多数或最后出现者决定。
- [ ] D5：冻结 context、输出上限、每题总预算与超时；不在失败后切换 thinking，不额外
  强制第二次完整推导，不照抄模型卡长度建议当作已验证参数。全 30 题独立 fresh 运行，
  不将旧 off 的答案和新 on 的部分题混合，也不以 token 长度挑题追加重答。

验证：resolved 配置端到端、真实 tokenizer 模板和合成 token 通道、未闭合 boxed 草稿、
等值／冲突 final、预算不变与 thinking 变更拒绝 resume；必要时一个合成真实服务 canary。
新 thinking 条件尚未实现或启动，当前 off 轮次不热修改。

## E：单候选评分、报告与最终验收（中风险）

- [ ] E1：Health grader 只接收一份已提交候选及私有 rubric，输出 verdict/score；
  不提供 candidate A/B、选择或替换答案接口，judge 反馈不回流 owner。
- [ ] E2：八项统一 owner 答案权威：QA alias 只在 scorer；程序只评 owner 的一个程序；
  Web/ALF 只计真实轨迹。公开样例不能在多个策略程序间择优，隐藏测试不触发再答。
- [ ] E3：新增简明 `EVALUATED_POLICY_ANSWER_AUTHORITY.md`；报告 policy/checkpoint/adapter、
  实际 thinking map、完整 planned/completed/failure/infra 分母、已验证 owner 来源数、
  跨 run 导入／fallback／selector／提交后生成计数，以及模型、peer、unknown 成本。
  零违规声明来自真实运行边界，不采信 actor 自报几个零。
- [ ] E4：分别报告 integrity_pass、execution_complete、performance_improved。
  不达目标不等于有 bug；高分也不能证明无选择。旧数据不因新代码升级为 verified。
  其他七项仅在实际条件或相关代码改变时安排评测，不拼不同版本各 benchmark 最高分。

验证：单候选评分不可变、scorer 无生成能力、失败不丢分母、版本不拼表、结果 schema 与集成。
正式验收须同时满足计划第 14 节全部十条，不能仅以目标分或测试通过代替真实来源证据。

## 实施与验证节奏

顺序：A → B+C 集成 → D → E/最终交付。源码全部由主线程实现和自检；子代理仅 watching，
不启动代码、审计或评审代理。B+C 的持久化与跨模块接口集中做一次主线程综合检查，
不对每个小修改设门禁。拟议测试优先合并到已有相关文件，避免重复覆盖和僵硬字面量断言。

本次清单整理只增加文档，未启动新的评测、训练或常驻进程，也未运行新的测试；已在运行的
`@13` AIME thinking-off 轮次按原条件继续。代码集成里程碑／push 前所需完整验证使用
22049 CPU：`CUDA_VISIBLE_DEVICES="" make check`；此次仅文档变更，沿用 `7235e93`
的 77 项定向测试及 2,952 项全量测试、Ruff/mypy、双 wheel 构建成功记录，不重复执行。
无哈希检查，不恢复已废止流程。
