# 七项 IID：thinking-on 复测与 MBPP 优先修复

## 当前状态

用户于本轮末尾要求停止 ALFWorld 超过 20 分钟的轨迹；已执行，不再重跑。
五项 32 题复测完成，ALFWorld 保留 31 条完成结果和一条人工中止，AIME 已经
thinking-on，本轮未重复运行。机器可读记录见
[汇总 JSON](machine-results/step0_seven_iid_thinking_followup_2026-09-08.json)。

| Benchmark | 本轮 thinking-on 结果 | 状态 |
| --- | --- | --- |
| MBPP+ | **84.375%（27/32）** | 达到 80；Base AND Plus |
| HotpotQA | **79.5939 F1** | 32/32；仍低于 80 |
| TriviaQA | **86.7424 F1** | 32/32 |
| HumanEval | **84.375%（27/32）** | 32/32 评分；包含一条输出预算耗尽 |
| HealthBench | **51.0355/100** | 32/32；固定本地 Qwen judge，超过 45 |
| ALFWorld | **31 成功／32 计划；1 条人工中止** | 96.875% 为按计划数计算的运营口径，非完整原生轮次 |
| AIME2026 | 本轮未重跑 | 已 thinking-on；不混合完整／中断旧记录 |

**MBPP 最新 thinking-on：27/32 = 84.375%，达到 80 目标。**
Base 28/32、独立 Plus 27/32，32 次 owner 调用、零修复／工具／peer 调用；
全部 32 条实际请求 thinking-on，原生评分和最终答案来源验证通过。
22,880 输入／22,215 输出 token，无基础设施失败。2026-09-08
20:07:54–20:11:53 UTC，端到端 239.76 秒，约 480.5 题／小时。
MBPP 达标后未继续尝试更高分或追加模型调用。

已完成的 TriviaQA thinking-on 为 **86.7424 F1**（EM 84.375，32/32），
32 次 owner 调用、55,555 输入／22,158 输出 token，零工具／peer 调用。
HumanEval thinking-on 已完成 **27/32 = 84.375%**，37 次 owner 调用，
117,448 输入／78,014 输出 token；两项测试失败、两项运行错误、一项预算耗尽空提交，
零基础设施失败。长尾按原 32,768-token 预算结束，没有切回 off 或用旧答案填补。
该空提交使整项标记为 `communication-not-complete / unresolved-output-failure`；
32 条评分完成不等于 32 条输出通道全部完成，不能宣称整项完整性检查全通过。
它低于此前 thinking-off 的 29/32；这次复测不能证明 thinking 总会提升成绩。
HealthBench thinking-on 已完成 **51.0355/100**，32/32，超过 45 目标。
Owner 37 次调用、33,900 输入／19,463 输出 token、3 次工具调用、零 peer；
独立 judge 400 次请求／响应、502,682 输入／41,080 输出 token，零语义修复、
零未知用量或失败调用。仍是固定本地 Qwen judge，不是官方外部 judge 等价成绩。
HotpotQA 首轮 thinking-on 为 **76.7651 F1**，32/32；其实际请求逐条核查
确认全部 **320/320 段公开原文和 32/32 个问题**在上下文内，不是只发送问题。
但发现一条加粗的 `**Final answer: …**` 未被当作字段，整段解释被误提交。
已修复 QA 字段外层 Markdown 包装，保留原文和冲突拒绝；不改 AIME、代码或临床
自然语言输出规则。只读比较原 64 份 QA 输出：Hotpot 一份投影改变、Trivia 零份。
因此仅追加 Hotpot 完整 32 题验证，不无故重复已完成的 Trivia，也不追改旧评分。

追加 Hotpot 已完成：**79.5939 F1**，EM 59.375，仍低于 80，不宣称目标达成。
32/32、34 次 owner 调用、一次 history 读取与一次接口修复、零 peer 调用；
77,197 输入／21,169 输出 token。20:34:55–20:42:04 UTC，用时 429 秒，
两路任务利用 HumanEval 长尾空槽，与旧四路轮次不是严格同吞吐条件。
本轮最终字段投影正常；剩余失分包含错误实体、把各自参与误解成共同合作、
过长／不完整的答案短语与数字词形差异。保持原生 F1，不做答案相关归一化或再次
重跑以碰到阈值。新旧生成并不逐字相同，不能把差值完全归因于 Markdown 修复。

仅评 A2：一个 Qwen3.5-9B owner，Step-0、无 adapter、skills-off、seed0。
当前 IID 严格为 HotpotQA、TriviaQA、AIME2026、HealthBench、ALFWorld、MBPP+、HumanEval。
本次先跑 MBPP，再补跑此前 thinking-off 的五项。AIME 已 thinking-on，不重复启动；
其 30 题完整轮与后来的 28 题中断记录不能合并，不能补成虚假的 32 题。

第一轮 MBPP thinking-on 已于 2026-09-08 19:51:58–20:05:57 UTC 完成：
**24/32 = 75.00%**，Base 26/32、独立 Plus 24/32；未达到 80。
该轮沿用旧采样参数 0.7/0.8/20、presence penalty 0，未开启原流分块配置。
34 次 owner 调用，47,111 输入／35,252 输出 token，两个接口修复、零 peer/tool 调用。
所有 32 题形成最终提交；八项失败是三项执行失败、两项 Base 测试失败、
两项 Plus 测试失败和一项语法错误，无基础设施失败。

31 题完成后，CPU 先对这些本轮唯一提交评分，写入同一 insert-once journal；
正式流程只复用这些原生分数，随后评分最后一题，不重复评分或混入历史答案。
最后一题两次长度截断，原始推理包含长段重复，最终由同一 owner 提交。
前 31 题约四分钟完成，总耗时约十四分钟；这不是服务请求排队造成的尾延迟。

修正采样与续写配置后的独立新轮按 MBPP+、HotpotQA、TriviaQA、HumanEval、
HealthBench、ALFWorld 顺序执行，每项固定来源前 32 题。最新六项中五项已完整结束，
ALFWorld 按用户指示中止，不能写成六项完整完成。首轮 thinking-off 曾按用户新指示中断于 23 条，
只保留私有记录，不与上述任何轮次合并。

## 本批修复与边界

1. **公开接口说明**：MBPP 基础任务说明明确使用公开示例中的函数名与调用方式，
   示例属于任务规格。没有添加题目专属提示、强制解法、自写求解器或隐藏反馈。
   核查失败原文后，确认本轮语法错误由 owner 本身生成，并非围栏解码损坏；
   错误函数名、owner 自加的失败断言与算法错误仍按原生评分，不改写答案。
2. **单项加载与七项清单**：source loader 只加载请求的 benchmark，MBPP 单项不再
   依赖未请求的 WebShop/其他题库文件。当前目录和新 panel 排除 WebShop，历史记录
   仍按原身份可读。公开 prompt 的顺序、文本和私有答案隔离不变。
   后续 HotpotQA 实际启动又暴露 runtime 无条件初始化 MBPP scorer 的独立依赖问题；
   现仅在面板包含 MBPP 时初始化该解释器与原生 profile。非 MBPP 不需要伪造 EvalPlus
   配置，已请求 MBPP 的原生评分条件不变。该启动失败发生在任何模型调用之前。
3. **实际 thinking 配置**：默认 map 和七项启动覆盖文件均为七项 thinking-on。
   后者曾残留旧的六项 off 配置，已同步修正并增加回归测试。HealthBench 的被评测
   owner 开 thinking；独立本地 Qwen rubric judge 保持已冻结的 thinking-off 评分协议。
4. **同轨迹续写与采样**：非 AIME 新轮在原有整题预算内按最多 8,192 token 分块，
   尾部 2,048 token 仍可使用；截断后接原始 prompt + 已生成 token，而非重新开推导。
   AIME 既有 32,768/4,096 配置保留。预留不是强制 final，也不保证没有重复或耗尽。
   新轮采用 Qwen 官方 thinking 参数：编码任务温度 0.6、top-p 0.95、presence 0；
   其他任务温度 1、top-p 0.95、presence 1.5，top-k 20、repetition 1。
   参见 [Qwen 官方 model card](https://huggingface.co/Qwen/Qwen3.5-9B#best-practices)。
   这是按任务类型预先声明的新条件，不是按题号、长度排名、答案或得分选择参数。
5. **交互预算与上下文容量**：ALFWorld 首次 thinking-on 启动暴露整局预算误作
   单次上下文预留的问题：163,840 大于 98,304，导致第一步就被错误拒绝，零模型
   调用。现预留的是当前响应／动作在剩余调用数、每块上限及整局余额内实际可用的
   最大输出，不为尚未开始的其他动作预留同一个上下文。原始 token 流续写不丢弃，
   总输出 163,840、总调用 160、每动作最多 8 次、环境 100 步均保持不变。
   真实隔离 actor 的两动作／中途续写回归通过，旧静态 AIME/QA/代码边界一并验证。
   修复后的 ALFWorld 使用 8 路独立轨迹协调，现有服务同时执行 4 请求；
   新快照单独记录，没有热改其他已完成轮次。

总输出／调用预算、公开题目、原生 scorer、单 owner 约束不降低或放宽。没有投票、
两候选选择、Step-0 回退或不同轮次的逐题最优合并。新旧轮同时改变了采样与续写，
不能把分数变化单独归因于 thinking、某项代码修复或 BayesianImprove 的训练收益。
这些是已用于开发的固定 32 题，没有匹配 backbone 对照或未见数据泛化证明。

## ALFWorld 中止与后续 20 分钟上限

2026-09-08 22:14:47 UTC 按用户指示终止最后一条长尾；31 条已完成记录全部
经原生评分成功，未生成第 32 条最终提交或伪造其环境失败／成功。31/32 = 96.875%
仅表示计划 32 条中的成功比例；31 条完成子集的 100% 不能冒充完整轮次分数。
此前个别长轨迹在 20 分钟后由 owner 自行纠正并成功，不能把此次结果追认为所有题
都预先限制 20 分钟的实验，也不追删这些已完成记录。

末条拿错物品后反复搜索；初始任务、当前物品／动作和环境回包均在轨迹内。
运行至中止约 66 分钟，长尾期间近期完成率曾降至零；主线程已经报告“需要人工
决策”并保留预算，收到中止指示后立即执行。没有替模型选动作、改目标或补答案。
全部已完成模型响应累计 815 次、7,924,679 输入／200,003 输出 token，全部
thinking-on、仅 owner；另有一条在途请求被中止，其未返回 token 用量未知，
不把已知响应之和宣称为精确总成本。CPU 只评分了本轮已保存的 31 份动作轨迹。

后续 **ALFWorld-only** 运行使用
[`step0_alfworld20min_timeout.yaml`](../configs/evaluation/step0_alfworld20min_timeout.yaml)：
既有隔离 actor 的整条轨迹 timer 设置为 1,200 秒，包含交互与推理等待，不按动作
重置；其他 benchmark 不应用该覆盖。超时仍记录为不完整，不能伪造 native outcome。
已准备新的私有运行配置，**本轮未启动新评测**；旧轮冻结配置保持不变。

## 资源与验证

- 仅复用 22049 **物理 GPU6** 的 SGLang 服务；四路独立任务并行，其他进程未修改。
  20:08 UTC 主线程确认服务健康 200、GPU6 利用率 100%。没有训练、额外 GPU 或新
  SGLang 副本；新建的是 CPU 评测顺序驱动、临时 CPU 评分及全检进程。
  最终 22:17 UTC 主线程检查：评测与检查进程已退出、GPU6 利用率 0%，SGLang
  健康 200 且保持在线；最后请求已经因客户端断开而被服务中止。仅保留该推理服务。
- 复制的 ALFWorld CPU 环境完成真实 reset/close，HealthBench 官方评分模块完成导入。
  缺失的官方依赖 blobfile 仅安装到私有依赖目录，未修改共享服务环境。
  首次跨服务器启动暴露旧 bubblewrap 路径和未请求 scorer 的初始化依赖，全部在
  模型调用前失败；保留失败目录，修复后使用新目录启动，不禁用隔离，不把这些
  无调用的启动失败算作 benchmark 失败。
- 本地运行受影响文件 Ruff/format；CPU 定向覆盖真实隔离 actor、公开输入非干扰、
  Python 载体、MBPP native profile、按域加载、七域清单及 thinking 传递。
  早期分组为 47、18、5 项，后续 thinking 与清单变更分别为 6、45 项；这些集合有
  重叠，不相加冒充独立用例数。最新续写／启动配置 17 项通过，采样配置追加后只
  重跑新增受影响断言（1 项通过）。一次私有 TMPDIR 配置错误只重跑相应新用例。
- 主实现冻结快照在 22049 CPU 完成 `CUDA_VISIBLE_DEVICES="" make check`：
  **3,554 passed、1 CUDA opt-in skipped**，pytest 257.77 秒，Ruff/format、mypy、
  两个 wheel 与模型包边界检查通过。启动覆盖配置变更后的全检为 **3,555 passed、
  1 skipped**，pytest 275.10 秒，其余检查亦通过。随后发现的 scorer 初始化依赖问题
  完成 25 项构造／分派／指标定向测试及 **3,557 passed、1 skipped** 的 CPU 全检
  （pytest 251.68 秒，其他检查通过）。QA Markdown 修复的 74 项定向用例通过：
  首次 73 项通过，新测试夹具漏填公开 context，补齐后只重跑该 1 项；最终 QA
  冻结快照 CPU 全检为 **3,571 passed、1 skipped**（pytest 255.81 秒），
  Ruff/format、mypy、两个 wheel 和模型包边界检查通过。
  随后的真实 ALFWorld 启动缺陷完成 51 项交互、续写、预算与 broker 定向测试；
  该新快照最终 CPU 全检为 **3,572 passed、1 skipped**（pytest 261.72 秒），
  Ruff/format、mypy、两个 wheel 和模型包边界检查通过；之后没有修改源码。
  收尾新增的 ALF-only timeout 覆盖只做配置解析，沿用既有 episode timer；
  按用户要求不再启动 GPU 验证。31 条保存轨迹的来源验证与 CPU 原生评分通过。
  同一已通过状态不重复全检，文档更新不重复构建；未运行无关 E2E。
- 未使用编码或独立审查子代理；只有用户指定的 Git-only luna/max 自动同步代理，
  不编辑文件、不读源码、不运行测试。没有手动哈希校验。共享工作树存在另一个训练
  会话的修改，本报告的验证只覆盖本会话冻结源码，不覆盖所有自动 WIP 提交。

题面、答案、逐题结果、环境私有路径和连接凭据仅保存于私有存储。

## Later HotpotQA-only follow-up

The owner subsequently requested deeper thinking specifically for HotpotQA.
A fresh same-panel run with uniform deliberation guidance and repaired recurrent
prefix-cache precision scored **82.5875 F1 / 68.75 EM**, 32/32 complete. Actual
reasoning did not lengthen: mean per model call was 518.34 tokens, versus 540.24
above. Other six IID were not rerun or relabelled. The preceding table remains
its original run record. See the [follow-up and limitations](step0_hotpot_thinking_cache_repair_2026-09-08.md).
