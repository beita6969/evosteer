# 新架构无训练：其余七项 IID 当前结果

> 后续更新：[同一冻结条件的七项32-sample真实结果](STEP0_SEVEN_IID_32_REPAIR_RESULTS_2026-09-08.md)
> 已完成。下表保留05:00 UTC的历史快照，不用新轮答案覆盖；其中旧ALFWorld轮随后完成
> 103/128（80.46875%）。新轮AIME回退与输出失败同样公开，不能继续拿旧25/30当作最新版。

应 owner 要求，本次汇总排除 WebShop，并停止其后续修复/新评测工作。
这不修改 `idea.tex` 或项目原有八项 IID 目录。截止时间：**2026-09-08 05:00 UTC**。
[机器可读汇总](machine-results/step0_a2_seven_iid_current_2026-09-08.json)。

**以下是每项最新可用记录，不是同一冻结版本的一轮七项结果，也不是挑选历史最高分。**
全部为 A2、Qwen3.5-9B、零优化器更新、无 adapter、skills off、单组实验 seed0。
数据均已用于开发，不能称为未见测试集；没有匹配 backbone 对照或架构因果增益结论。
“无训练”不等于“无 thinking”：AIME 单独启用 native thinking，其余六项关闭。

## 成绩（0–100）

| Benchmark | 已评 / 计划 | 最新分数 | 指标与状态 | 来源 |
|---|---:|---:|---|---|
| HotpotQA | 128/128 | **72.13** | Answer F1；EM 53.13；低于 >75 目标 | `5a1517a` / `@11` |
| TriviaQA | 128/128 | **82.62** | Answer F1；EM 77.34；超过 75 | `5a1517a` / `@11` |
| AIME2026 | 30/30 | **83.33** | 25/30；thinking-on；三条预算耗尽失败保留在分母 | `6c930c4` / AIME thinking `@1` |
| HealthBench | 128/128 | **43.59** | 本地 Qwen rubric 均分；旧配置实际调用过咨询 Agent | `5a1517a` / `@11` |
| ALFWorld | **126/128** | **81.75（部分）** | 已完成部分 103/126 成功；两条未完成；不是最终 128 条均分 | `d2c4af9` / owner interaction `@5` |
| MBPP+ hard（要求的名称） | 128/128 | **72.66（实际为 MBPP+）** | Base AND Plus 93/128；固定 hard 子集尚未验证；旧配置调用过咨询 Agent | `5a1517a` / `@11` |
| HumanEval | 128/128 | **85.94** | Pass@1，110/128；旧配置调用过咨询 Agent | `5a1517a` / `@11` |

### 必须保留的口径区别

- **还没有严格单主 Agent、当前同一源码的完整七项结果。** `@11` 配置暴露了可选咨询
  对象：HotpotQA、TriviaQA 实际咨询调用为零；HealthBench、MBPP+、HumanEval 分别为
  **37、4、9 次**，实际 Agent 数分别为 3、2、2。后面三项只能列为历史成绩，不能重命名
  为现行严格单主 Agent 结果。前面两项实际仅 owner 推理，也不等于已在禁用咨询对象的
  当前配置上重新运行。AIME thinking 轮同样实际只有 owner，但旧配置仍暴露可选 peers。
- **HumanEval 的历史验收 118/128 = 92.19% 仍保留**，来自 `3f37ec0` 的独立轮次；
  不撤销 owner 的验收，也不拿它覆盖最新历史轮次的 110/128。
- AIME thinking-off 的独立最新记录为 **23/30 = 76.67%**。它和 thinking-on 不投票、
  不拼接答案、不取逐题较优结果。thinking-on 达到分数阈值，但三条输出失败仍使其
  communication status 为 `unresolved-output-failure`。
- HealthBench 使用本地 Qwen3.5-9B 评分，不是官方外部 GPT judge 分数；不能据此声称
  严格高于匹配 backbone。MBPP+ 的实际数据是 EvalPlus MBPP v0.2，不能凭名称或分数
  将常规 128 条样本当作已经验证的 hard 子集。
- ALFWorld 的 103 个已确认成功意味着最终分数可能在 **80.47–82.03** 之间，但这只是
  以剩余两条成败计算的范围，不是填零或最终分数。原轮保留、自然完成，不用新轮答案
  补齐。其未解决的输出/控制问题仍记录为失败，不能写成通信全通过。

## 来源与未采用的结果

- 五项静态任务：[最新完整 `@11` 原始汇总](machine-results/step0_a2_native_feedback_full_2026-09-07.json)，
  `full-a2-native-tools-5a1517a-20260907T185427Z`，2026-09-07 19:53:57 UTC 完成。
- [AIME thinking-on 原始汇总](machine-results/step0_a2_owner_thinking_aime30_2026-09-07.json)，
  `focused-aime-a2-owner-native-thinking-6c930c4-20260907T214451Z`。
- [ALFWorld 126 条只读评分快照](machine-results/step0_a2_alfworld_partial126_2026-09-08.json)，
  `focused-full-a2-native-interaction-v5-fullcatalog-d2c4af9-20260908T022312Z`。
  评分于 04:51 UTC 读取既有候选和 native outcome，无新模型调用、环境动作或原评分表改写。
- WebShop 最新完整旧轮为 **37.46/100，128 条**，故按 owner 指示撤下后续工作。
  不采用早期已撤销的规则路由、投票、候选选择或回退组合成绩来填补任何表格。
- `6bb921e` 修复版 `@6` 混合 128+128 轮于 04:57:23 UTC 启动，owner 改变范围后
  于 04:59:56 UTC 请求停止。保留 **44 条 ALFWorld 候选、0 条 WebShop 候选、0 个分数**；
  05:00 UTC 亲查 coordinator 已退出且无该轮残留。它是未完成轮次，完全不进入上表。
  上一批合成诊断的上下文缩短/采样变化，不是这次被停止轮次的准确率或端到端加速结果。

## 本次交付与资源

此次只汇总记录、更新文档和停止刚启动的混合轮；没有追加七项评测、重评分或训练。
05:00 UTC 主线程亲查：旧 `@5` 仍 254/256，仅剩 ALFWorld 两条；此前约五分钟无新增
完成，低于 40 条/小时阈值，已报告需要人工决策。按最近约 28 tokens/s 与剩余输出预算，
粗估还需 **15–25 分钟**，长尾波动大，不能沿用累计完成率给出的分钟内 ETA。
owner 本次已决定不再继续 WebShop；ALFWorld 原样保留，不切预算或替换答案。

22048 现有 **GPU5、7** SGLang 仍在线，显存约 67.7/68.2 GiB、100% utilization，两个 health
检查均为 200；本次未新增或停止推理服务，也未操作其它会话训练或其它 GPU。
这些是此次主线程实查，不转述其它训练会话的资源状态。

验证只检查七行数据与来源一致、分母/历史配置/未完成标识、JSON 和文档链接及 diff。
没有产品源码变化，不重复前一批已完成的完整检查与构建；其 245 项定向测试及
3381 项通过、一个 CUDA skip 的完整检查/定向补完记录见
[性能修复说明](STEP0_NATIVE_INTERACTION_PERFORMANCE_REPAIR_2026-09-08.md)。
没有新增子代理、独立审查或哈希检查；既有子代理仅 watching。题库、答案、逐题分数、
连接凭据和私有运行路径均不进入 Git。
