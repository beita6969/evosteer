# Qwen3.5-9B SWE-bench Verified backbone-only 评测方案

## 1. 结论先行

SWE-bench 的有效 Direct-Qwen backbone-only 轨道不能把 supervisor 的编辑
指令委托给第二次 MExec 模型调用，也不能在运行中获得环境编辑、
语法检查或测试反馈。正式轨道是 **28-step 单模型只读仓库检查**：
Qwen3.5-9B 可搜索并阅读全仓库，然后一次性提交一个 unified diff。评测时
skill workspace 为空，不加载 LoRA、adapter 或 learned skill，MExec 明确禁用。

旧 one-shot 轨道仍保留为解析器/基础设施诊断，但不能替代 repository-agent parity
结果。旧的 supervisor+MExec 轨道保留为“同底座双调用编排”对照，也不再冒充
Direct-Qwen。三条轨道使用同一个 frozen 128-instance IID panel 和同一个官方
SWE-bench Verified evaluator。

## 2. 根因与修复

先前低分来自三个相互独立的问题：

1. **生成协议不一致**：旧实现只给 issue 和最多三个源码摘录，让模型一次性返回 diff；
   上游实际提供完整 repository 和多步 `list_files`、`search_code`、`view_file`、
   `edit_file`。
2. **diff 提取过窄**：Markdown fence 标签写错、多个 fence 配对和重复候选会造成假
   invalid。`unified-diff@2` 现在按行解析 fence、按 diff 结构判断、合并完全相同的
   重复候选，并拒绝多个不同 patch。
3. **rootless archive ownership**：官方 harness 生成的 tar 继承宿主 UID/GID；单 UID
   user namespace 无法在容器内表示该 ownership，导致 `put_archive` 在 patch 应用前
   失败。桥接器现在只把传输 tar 的 UID/GID 规范化为 0；patch 字节、镜像、测试和
   scorer 语义均不改变。
4. **声明 seed 未进入请求**：早期 repository-agent 运行声明 seed 42，但上游
   supervisor/MExec 请求没有发送 `seed`。runner 现在把 YAML 的单一 formal seed 注入
   每个请求，并拒绝冲突 seed。
5. **Direct-Qwen 身份漂移**：上游 `edit_file` 让 supervisor 把编辑指令交给第二个
   Qwen MExec 调用；即使改成单模型 `str_replace_editor`，每步编辑后的 workspace
   反馈仍使合同明显强于直接基线。正式轨道因此只暴露只读导航工具，让同一 Qwen
   最终一次提交 unified diff；disabled executor 证明不存在第二模型调用。
6. **环境额外引导**：上游 SWE 环境会注入跨轮 progress ledger、从 issue 预提取的
   member hints 和重复工具调用 steering。正式只读合同关闭这些非请求的提示，
   每轮 observation 只包含模型主动请求的 repository evidence。

## 3. 正式生成合同

生成实现：`scripts/generate_qwen35_direct_swe_repo_agent.py`。

| 项目 | 合同 |
|---|---|
| Population | released IID frozen 128，顺序由 manifest 固定 |
| Base model | adapter-free `Qwen/Qwen3.5-9B` |
| Skill workspace | empty |
| Repository | public repo 的 frozen base revision，独立 Git worktree |
| Horizon | 28 steps |
| Model tools | `list_files`, `search_code`, `view_file`（只读） |
| Supervisor decoding | non-thinking，temperature 0.8，max tokens 512 |
| Executor | disabled；禁止 MExec 或其它第二模型调用 |
| Seed | YAML 中唯一 seed 42；注入每个 OpenAI-compatible 请求 |
| Candidate | 一次最终 `unified-diff@2` patch；不在生成阶段应用 |
| Candidate retry | 禁止 |
| Infrastructure retry | 最多三次，只重试明确的请求、repo 或超时基础设施失败 |

runner 是 append-only journal：已产生 candidate 或 candidate failure 的任务永不重跑；
只有 infrastructure failure 可在上限内恢复。最终 prediction 始终保持 frozen 128 顺序。

### 3.1 答案隔离

模型可见对象只有：

- public problem statement；
- public repository identity 和 base revision；
- 当前独立 worktree 中通过工具读取的源码；
- 本 episode 的工具历史。

以下内容不会进入模型上下文：gold patch、官方 test patch、FAIL_TO_PASS/PASS_TO_PASS、
gold-affected-file 提示、scorer verdict、其它样本输出。官方 evaluator 只在 generation
终止后接收 candidate patch。

所有读写路径先解析为 canonical path 并验证仍位于当前独立 worktree；绝对路径、`..`
和 symlink 均不能越过该边界。搜索和文件列表只以 worktree 为工作目录。

## 4. 官方 CPU evaluator

评分实现：

- `scripts/score_qwen35_direct_swe.py`：128 条 coverage gate 和 aggregate；
- `scripts/run_local_swebench_official.py`：typed protocol 到官方 harness 的桥接；
- 官方包：`swebench==5.0.2`，dataset variant 为
  `SWE-bench/SWE-bench_Verified`。

当前可复现 runtime 使用 22048 的 CPU 和大容量 Linux 本地盘，不占 Windows C 盘：

1. 在隔离 user namespace 中运行非特权 Podman service；
2. image/layer/work-root 全部位于大容量本地盘；
3. 使用 fuse-overlayfs 和 `overlay.ignore_chown_errors=true` 解包官方镜像；
4. Docker SDK 通过私有 Unix socket 连接该 service；
5. bridge 把 copy archive 的 UID/GID 设为容器 root 可表示的 0；
6. 不需要 sudo，不修改系统 Docker，不接触其他用户的 container 或进程。

transport ownership 修复只解决 rootless 文件复制，不绕过官方镜像、patch application、
测试或 resolved 判定。

## 5. restart-safe scorer

bridge 将每个实例归类为以下 definitive verdict：

```text
resolved
apply-failure
test-failure
timeout
candidate-invalid
```

空 patch 直接成为 `candidate-invalid`。已有 `report.json`、明确 apply failure、timeout
或显式导入的 prior verdict 会被复用；只把未知实例交给官方 harness。prior 与当前
candidate 或 work-root 证据冲突时 fail closed。

镜像拉取、容器创建、archive copy、harness 输出缺失等仍属于 infrastructure failure，
不能计零。失败时 `--partial-output` 保存已完成 verdict 和 pending 数量，便于同一
run-id 继续。

## 6. 运行模板

路径均为公开占位符；真实题库、patch、逐题日志和内部路径不得进入 Git。

```bash
export CUDA_VISIBLE_DEVICES=""
export RUN_ROOT=/large-local-disk/swe-repo-agent
export PYTHONPATH="$RUN_ROOT/source/src:$RUN_ROOT/source/packages/private-evaluation/src:$RUN_ROOT/source"

python scripts/generate_qwen35_direct_swe_repo_agent.py \
  --config configs/evaluation/qwen35_skillflow_direct_reference.yaml \
  --iid-population /private/frozen-iid.json \
  --population-manifest /private/swe-manifest.json \
  --private-output-dir "$RUN_ROOT/generation" \
  --skillflow-source /path/to/locked/SkillFlow \
  --repo-cache "$RUN_ROOT/repos" \
  --endpoint-base http://127.0.0.1:PORT_A/v1 \
  --endpoint-base http://127.0.0.1:PORT_B/v1 \
  --endpoint-base http://127.0.0.1:PORT_C/v1 \
  --served-model-name qwen35-direct-base \
  --concurrency 12
```

评分命令使用相同 frozen predictions，设置
`--generation-contract qwen-direct-readonly-repository-agent`，并把 local evaluator command 指向
`run_local_swebench_official.py`。所有长任务用 `nohup` 或 `tmux` 保存 PID 和日志。

## 7. 结果轨道

| 轨道 | Coverage | Resolved | 用途 |
|---|---:|---:|---|
| issue-only one-shot diagnostic | 128/128 | 8/128 = 6.25% | 解析器/官方 scorer 诊断；没有全仓库检查 |
| delegated supervisor+MExec（未绑定 seed） | 128/128 | 35/128 = 27.34375% | 历史编排对照 |
| delegated supervisor+MExec（seed 42） | 128/128 | 38/128 = 29.6875% | 种子修复后的编排对照 |
| 单 Qwen 可编辑 workspace | 128/128 | 33/128 = 25.78125% | 无 MExec，但有逐步编辑反馈 |
| 单 Qwen raw-tools 可编辑 workspace | 128/128 | 39/128 = 30.46875% | 去除环境 hints，但仍有逐步编辑反馈 |
| **单 Qwen 只读检查 + 一次最终 diff** | **128/128** | **16/128 = 12.5%** | **本轮正式 Direct-Qwen 数值轨道** |

只读正式 generation 在 1,411.05 秒内完成 128 条（约 5.44 tasks/min）：
55 条提交非空 unified diff，73 条是合法 candidate failure，共 2,987 次单 Qwen
supervisor call，generation infrastructure failure 为 0。官方 evaluator 给出 16 resolved、
20 test failure、19 apply failure、73 candidate invalid、0 timeout；128/128 都有 definitive
verdict，generation、scorer 和 environment infrastructure failure 均为 0。

本轮没有重跑成功生成、candidate failure 或官方终态；也没有用 scorer 结果选择
seed、temperature、sample 或 patch。不同轨道之间的新运行由可观测的合同偏差驱动，
不是在同一合同下重试候选。

相对 17.19% 参考值，12.5% 的绝对差为 **4.69 个百分点**，严格 `<7 pp`
数值门禁为 **PASS**。由于 paper 的 prompt/decoding/seed aggregation 比较语义仍未公开，
这只是 SWE-bench 数值 parity，不升级为科学等价，也不单独决定 formal training GO。

## 8. 发布与清理

正式发布必须同时满足：

- 128 个唯一 frozen instances；
- generation candidate response coverage 为 128/128；
- official definitive verdict coverage 为 128/128；
- generation/scorer/environment infrastructure failure 均为 0；
- 只发布 aggregate，不提交题面、答案、patch、逐题 verdict 或私有路径；
- evaluator 完成后清理本项目的临时 worktree、container 和额外 CPU service；
- 用户要求保留的 adapter-free SGLang 不因 scorer 完成而停止。
