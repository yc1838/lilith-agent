# Lilith-Agent 改进计划

> 基于 Marina Wyss "Context Engineering for AI Agents" 视频分析、Anthropic 官方博文、2026 年 5 月最新趋势研究，以及对 lilith-agent 现有代码的逐行审计。
>
> 日期：2026-05-22
> 更新：2026-05-23（整合深度 Code Review 反馈与工程落地可行性调整）

> **Review note:** 以下新增的 `Review note` 只补充 repo 审计后发现的工程落地风险，不删除或改写原计划内容；实现时应把这些 caveat 当作验收条件的一部分。

---

## 目标定义

本计划分两条线：

- **路线 A（GAIA 提分）**：针对 GAIA benchmark 特点（466 题，全对或全错，三级难度），以最小改动换最大分数提升。当前榜首 Claude Sonnet 4.5 在 Princeton HAL 上 74.6%（有脚手架），裸模型只有 44.8%，说明 agent 框架的价值巨大。
- **路线 B（追赶最新趋势）**：让 Lilith 成为架构上最先进的 agent，不只看分数，还看多 agent 编排、context 管理等能力。

下面按最新的执行顺序排列，每项标注属于哪条路线。

---

## Phase 0（前置条件）：Eval Pipeline 稳定化与 Blockers 修复

**在所有改动之前，必须先确保评估基础设施可靠、结果可复现，同时修复目前存在的严重状态泄露和参数解析 Bug。**

### 为什么这是 Phase 0

每个 Phase 合并后都必须跑完整的 GAIA eval 来验证改动是加分还是减分。如果 eval pipeline 本身不稳定（比如 API rate limit 导致随机失败、checkpoint 残留影响重跑结果、不同次 run 分数波动过大），后续所有改进的效果评估都不可信。

### 具体任务

1. **修复 Blockers**：
   - **`--force` 状态泄露**：当前仅删除 JSON 文件，未清理 checkpointer 的 SQLite persistence。旧的 thread state 在 SQLite 中依然存活，会严重污染重跑结果。
   - **CLI `--level` 解析错误**：`dev_run_gaia.py` 传入的是字符串，而数据集匹配需要严格类型，导致 `--level 1,2,3` 匹配出空集。
     > **Review note:** repo 当前在 `gaia_dataset.py` 中已经把 `row.get("Level")` 和 `level` 都转成字符串比较；真正的缺口不是严格类型，而是 `--level 1,2,3` / `--level all` 这类多 level 输入没有被解析成集合过滤。
   - **`scoring API split` 冲突**：`os.getenv("GAIA_DATASET_SPLIT","test")` 的 fallback 逻辑会与本地 validation runs 冲突，也可能影响 leaderboard 提交脚本，需显式隔离。
   - **Checkpoint 职责分离**：当前的 checkpoint 混杂了 leaderboard payload 和 debug 状态。需要剥离 metadata 到 `runs/<run_id>/<task_id>.json` 中。
     > **Review note:** 如果 metadata 或 per-run checkpoint 迁移到 `runs/<run_id>/...`，必须同步更新 `scripts/build_leaderboard_submission.py` 和相关测试；该脚本当前只从单一 checkpoint 目录 glob `*.json` 生成 submission。
2. **Baseline 锁定与回归**：跑 3 次精简的 stratified golden set (约 20 题，覆盖各 level) 和 1 次 full GAIA eval，记录分数和各 level 的正确率。
3. **Eval 报告自动化**：每次 eval 后自动生成 JSON 报告，包含：
   - 总正确率、各 level 正确率
   - 每题的 tool call 数、耗时、是否触发 fail_safe
   - 与上次 run 的 diff（新增正确 / 新增错误的题目列表）

### 涉及文件

- `src/lilith_agent/runner.py`：`run_agent_on_questions()` 增加统计输出与 env var 修复
- `src/lilith_agent/app.py`：修复 checkpointer clear 逻辑
- `scripts/dev_run_gaia.py`：修复 level parse 逻辑
- 新增：`scripts/eval_report.py`（生成对比报告）
- 新增：`tests/golden_set.json`（稳定题目子集）

---

## Phase 0.5：Normalizer 鲁棒性审计

**路线：A | 预估 GAIA 提升：免费加分项**

### 问题与思路

GAIA 评分是严格的 exact-match。如果经历了复杂的多步推理，仅仅因为提取的答案多了一个空格、或者大小写不匹配而丢分，是非常不划算的。在进行任何复杂的架构改动前，提升答案 Normalizer 的保真度（fidelity）是最便宜的"免费加分项"。

1. 审计并强化 Supervisor 的 `best_answer` 提取逻辑。
2. 确保对比与最终提取时，正确处理空白字符、大小写归一化及标点符号过滤。

---

## P3：Supervisor 升级与安全拦截 (Guard)

**路线：A | 预估 GAIA 提升：+1-2%**

### 问题

当前 supervisor 用的是 cheap model，判断质量有限。特别是在 Level 2/3 题上，cheap model 可能无法准确评估 agent 是否已经收集到足够的证据。此外，它可能会错误地抽取出诸如 "unknown", "N/A" 的 placeholder，导致提前放弃任务。

### 实现思路

在 `src/lilith_agent/app.py` 的 `build_react_agent()` 中：

1. **模型升级**：将 `supervisor_model = get_cheap_model(cfg)` 改为 `supervisor_model = get_extra_strong_model(cfg)` 或新增一个 `get_strong_model(cfg)` 中间档。
2. **安全拦截 (Placeholder Guard)**：增加护栏（Guard）逻辑，如果 Supervisor 提取出 "unknown/n/a" 等无意义的 placeholder，强制拒绝结束任务，要求 agent 继续探索。
3. **阈值调整**：由于 strong model 更贵，提高 supervisor 触发阈值：`_SUPERVISOR_MIN_TOOL_CALLS` 从 5 提到 8。
4. 同时给 supervisor prompt 加入更明确的评判标准，比如：
   - "如果 agent 已经找到了一个具体的数字/名字/日期，且该答案与已收集的证据一致，则 status=finalize"
   - "如果 agent 在最近 3 次 tool call 中没有获得新信息，则 status=finalize"

### 涉及文件

- `src/lilith_agent/app.py`：`build_react_agent()` 中的 supervisor 初始化与 guard 逻辑
- `src/lilith_agent/config.py`：新增 `supervisor_model_tier` 配置

---

## P4：fetch_url 结果二次清洗

**路线：A | 预估 GAIA 提升：+0.5-1%**

### 问题

`fetch_url` 提取正文后截断到 `max_chars=8000`。但网页输出经常残留导航栏、cookie 提示、重复 header 等噪音。同样 8000 字符，噪音越少有效信息密度越高。当前逻辑可能只处理了 trafilatura fallback，遗漏了 Jina Reader 主路径。

### 实现思路

在 `src/lilith_agent/tools/web.py` 的 `fetch_url` 函数中，对 Jina Reader 和 trafilatura 提取的结果，在截断之前，加一步轻量清洗。

**Noise patterns 外置成 JSON 配置文件**（不使用 YAML，避免引入新依赖）：

```json
{
  "en": [
    "cookie",
    "privacy policy",
    "terms of service",
    "subscribe",
    "newsletter",
    "follow us",
    "share on",
    "tweet this",
    "skip to content",
    "advertisement",
    "sponsored"
  ],
  "zh": [
    "隐私政策",
    "使用条款",
    "订阅",
    "关注我们",
    "分享到",
    "跳转到主内容",
    "广告"
  ]
}
```

清洗函数从 JSON 配置文件读取 patterns 进行清洗。

### 涉及文件

- `src/lilith_agent/tools/web.py`：`fetch_url()` / `_fetch_url()`
- 新增：`src/lilith_agent/config/noise_patterns.json`（噪音模式配置）

> **Review note:** `src/lilith_agent/config.py` 目前是模块文件，`pyproject.toml` 也没有 package-data 配置；如果把 JSON 放到 `src/lilith_agent/config/noise_patterns.json`，打包后可能无法被可靠读取。优先考虑放到明确的 packaged data 路径，或同步补充 setuptools package-data 配置与读取测试。

---

## P5：动态 Budget 管理

**路线：A | 预估 GAIA 提升：+0.5-1%**

### 问题

当前 `budget_hard_cap=25` 和 `budget_warn_at=15` 是全局配置（固定值），不区分题目难度。Level 1 题通常 3-5 步就能回答，而 Level 3 可能需要 30+ 步。由于 graph 编译是一次性的，无法在 runtime 随意修改 cfg。

### 实现思路

放弃修改 cfg，直接在 `AgentState` 中新增 field 来管理。放弃调用模型来猜测难度。

1. **直接使用 Level 标签**：在任务开始时，直接读取 GAIA 数据集自带的 Level 标签。
2. **状态驱动的 Budget**：在 `AgentState` 内维护 `current_budget_warn` 和 `current_budget_cap`。
   - Level 1：warn=8, cap=12
   - Level 2：warn=15, cap=25
   - Level 3：warn=25, cap=40（配合更积极的压缩）

> **Review note:** 需要一个统一的 state-derived budget helper 同时驱动 `model_node()` 的 warning prompt、`_route_after_model()` 的 hard cap，以及 graph compile 时的 recursion sizing；只在 `AgentState` 加字段会留下 cfg 默认值继续生效的路径。

### 涉及文件

- `src/lilith_agent/app.py`：`AgentState` 定义，路由逻辑 `_route_after_model()`
- `src/lilith_agent/runner.py`：传递 level 标签到状态中

---

## P0：状态级里程碑感知压缩（State-Level Milestone）

**路线：A + B | 预估 GAIA 提升：+2-4%**

### 问题

当前 `_compact_old_tool_messages()` 按位置压缩，这是"时间盲"的，会丢失关键发现。如果直接往消息流强行插入 `SystemMessage` 来标记里程碑，会破坏 `AIMessage(tool_calls)` 与 `ToolMessage` 必须相邻的 LLM 厂商协议，极易触发 API 400 报错。

### 实现思路

在 `src/lilith_agent/app.py` 中改造，**绝不直接向消息流插入标记**，而是维护一个独立的事实账本。

1. **State-Level Ledger**：在 `AgentState` 中新增一个独立的 `facts_ledger` 列表。
2. **复用压缩器检测以控制成本**：为了节省 extra cheap model 调用，不在每次 AIMessage 产出时检测。将里程碑检测逻辑合并到已有的 compaction summarizer 流程中，或者当 `tool_calls >= 6` 时才触发检测。提炼出的关键事实追加到 `facts_ledger` 中。
3. **注入上下文**：在生成 prompt 时，将 `facts_ledger` 作为整体系统上下文注入到前端，保证早期的关键发现不会被后续的时间盲压缩丢失。
   > **Review note:** `facts_ledger` 的来源是 tool/web 输出，属于 untrusted evidence，不应被提升成高优先级指令。实现时要用明确的 evidence wrapper、保留来源/provenance，并增加恶意网页内容不能注入系统指令的测试。
4. **可选：接入 Anthropic Compaction API** 替代自研总结逻辑。

### 涉及文件

- `src/lilith_agent/app.py`：`AgentState` 定义、`_compact_old_tool_messages()` 改造

---

## P2：经验反思学习（ERL）

**路线：A | 预估 GAIA 提升：+5-8%**

### 问题

Lilith 当前的 `ephemeral_memory` 会在任务结束后被完全清空，无法跨任务继承经验，导致无法实现"从失败中学习"。ERL（arxiv 2603.24639）在 GAIA2 上比 baseline 提高了 7.8%，且完全在 prompt-time 运行。

### 实现思路（分两步）

**Step 1：Golden Heuristics 验证上限（1-2 天）**

1. 人工挑选 20-30 条高质量 heuristics（分析 `.last_failures.txt` 失败轨迹提取），写入 JSON 格式的种子文件。
2. 直接注入 system prompt，跑一轮 stratified golden eval，验证 ERL 在当前架构下的理论上限。若提升 < 2%，需重新审视。

**Step 2：自动化反思管线与独立持久化**

1. **独立持久化 DB**：新增 `src/lilith_agent/heuristics.py`，建立独立的 SQLite 表（不与 ephemeral memory 混用），记录 rule、成功率等统计数据。
   ```sql
   CREATE TABLE heuristics (
       id TEXT PRIMARY KEY,
       rule TEXT NOT NULL,
       source_task_id TEXT,
       success BOOLEAN,
       keywords TEXT,
       times_applied INTEGER DEFAULT 0,
       times_helped INTEGER DEFAULT 0,
       times_hurt INTEGER DEFAULT 0,
       confidence REAL DEFAULT 0.5,
       created_at TEXT,
       retired_at TEXT
   );
   ```
2. **任务后反思**：`run_agent_on_questions` 结束后，不管成功失败，调用 cheap model 分析轨迹，生成规则并存入。
   > **Review note:** `run_agent_on_questions()` 当前没有 ground-truth success signal；validation/golden 以外的 leaderboard/test split 任务不能可靠判断 `success`、`times_helped`、`times_hurt`。自动学习应限制在有 expected answer / 显式 scoring 结果的 run，或把 unlabeled run 只作为候选规则来源而不更新置信度。
3. **规则检索与自动淘汰**：每次任务记录 applied heuristic IDs；当 `confidence < 0.3` 且 `times_applied >= 5` 时，自动 retire。必须在起步阶段就做好淘汰机制，防止低质量规则污染 prompt。

### 涉及文件

- 新增：`src/lilith_agent/heuristics.py`（独立持久化存储）
- `src/lilith_agent/runner.py`：任务后反思调用与 heuristic 效果追踪
- `src/lilith_agent/app.py`：`model_node()` 中的规则检索与注入
- 新增：`scripts/heuristic_health_report.py`（规则健康报告）

---

## P1：Sub-Agent 架构

**路线：A（Level 3 题）+ B | 预估 GAIA 提升：+2-3%**

### 问题

单 agent 在几十步的 Level 3 题中极易"lost in the middle"。引入独立 context 的子代理可以缓解。但当前工具注册机制（如 `build_tools`）较为 monolithic，直接引入会导致循环依赖；同时，所有任务共享同一线程 state 会导致严重污染。

### 实现思路

必须先解决工程底座问题，才能安全引入 `spawn_sub_agent`。

1. **前置重构：工具与 Agent 解耦**：重构 `build_react_agent()` 和 `build_tools()` 的绑定关系。工具构建必须支持按需裁剪（如 `build_tools(cfg, allowed_names)`），解决循环依赖。
2. **Checkpointer 隔离**：子 Agent **绝不能**使用主线程的同一个 SQLite checkpointer。必须为子任务分配独立的 `in-memory checkpointer`，或唯一的 `child thread ID`，或设为无持久化状态，确保上下文纯净。
3. **硬限制保护**：
   - 最大深度：`depth=1`，防止递归爆炸。
   - 操作限制：子代理禁止调用 `write_file`，除非明确分配。
   - 独立的 budget 和 timeout。
4. **精简的子工具集与动态摘要**：子 agent 根据任务类型（research, compute, vision）拿到裁剪后的纯净工具集。返回摘要时按任务类型动态限制长度（research ≤ 3000 chars, compute ≤ 500 chars），汇总回主代理。

### 涉及文件

- `src/lilith_agent/app.py`：解耦并重构工具组装、注册新工具
- 新增：`src/lilith_agent/tools/sub_agent.py`（包含工具集裁剪逻辑）
- `src/lilith_agent/config.py`：新增子代理配置

---

## 实施顺序建议

```
Phase 0 & 0.5（基础设施与提分低垂果实，2-3 天）
├── 修复所有 Blockers (状态泄露, CLI bug)
├── 剥离 Checkpoint Metadata，跑 Baseline
└── 强化 Normalizer 和 best_answer 提取逻辑

Phase 1（快速见效，1-2 周）
├── P3：强模型 Supervisor + Placeholder Guard
├── P4：fetch_url 清洗（覆盖 Jina 主路径，JSON 配置）
├── P5：基于 Level 的 AgentState Budget 控制
└── ⚡ 跑完整 GAIA eval，对比 baseline

Phase 2（状态记忆与反思，2-4 周）
├── P0：State-level 里程碑事实提炼（复用 Summarizer 控制成本）
├── P2 Step 1：建立独立持久化 DB，JSON 注入验证 Golden ERL 上限
├── P2 Step 2：自动化任务后反思与淘汰管线
└── ⚡ 跑完整 GAIA eval，对比 Phase 1

Phase 3（高阶架构，3-5 周）
├── P1 前置：重构 Tool Registry，解耦循环依赖
├── P1 核心：Checkpointer 隔离与 Sub-agent 实现
└── ⚡ 跑完整 GAIA eval，验收最终成果
```

**关键原则**：每个 Phase 结束时必须跑评估。如果改动导致 golden_set 回归，立即 revert 并分析。

---

## 参考来源

- [Anthropic - Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [ERL 论文 - arxiv 2603.24639](https://arxiv.org/abs/2603.24639)（ICLR 2026 MemAgents Workshop）
- [Constraint Compliance Decay - arxiv 2604.20911](https://arxiv.org/abs/2604.20911)（4416 次试验）
- [GAIA Leaderboard - Princeton HAL](https://hal.cs.princeton.edu/gaia)
- [Marina Wyss - Context Engineering for AI Agents](https://www.youtube.com/watch?v=-h9VVJIqtvA)
- [Perplexity MCP Criticism](https://nevo.systems/blogs/news/perplexity-drops-mcp-protocol-72-percent-context-window-waste)
- [Multi-Agent Orchestration Guide 2026](https://fungies.io/ai-agent-orchestration-developers-guide-2026/)
- [Anthropic Compaction API](https://platform.claude.com/cookbook/tool-use-context-engineering-context-engineering-tools)（beta）
