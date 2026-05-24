# Lilith-Agent 改进计划

> 基于 Marina Wyss "Context Engineering for AI Agents" 视频分析、Anthropic 官方博文、2026 年 5 月最新趋势研究，以及对 lilith-agent 现有代码的逐行审计。
>
> 日期：2026-05-22
> 更新：2026-05-22（整合第一轮 review 反馈）

---

## 目标定义

本计划分两条线：

- **路线 A（GAIA 提分）**：针对 GAIA benchmark 特点（466 题，全对或全错，三级难度），以最小改动换最大分数提升。当前榜首 Claude Sonnet 4.5 在 Princeton HAL 上 74.6%（有脚手架），裸模型只有 44.8%，说明 agent 框架的价值巨大。
- **路线 B（追赶最新趋势）**：让 Lilith 成为架构上最先进的 agent，不只看分数，还看多 agent 编排、context 管理等能力。

下面按优先级排序，每项标注属于哪条路线。

---

## Phase 0（前置条件）：Eval Pipeline 稳定化

**在所有改动之前，必须先确保评估基础设施可靠、结果可复现。**

### 为什么这是 Phase 0

每个 Phase 合并后都必须跑完整的 GAIA eval 来验证改动是加分还是减分。如果 eval pipeline 本身不稳定（比如 API rate limit 导致随机失败、checkpoint 残留影响重跑结果、不同次 run 分数波动过大），后续所有改进的效果评估都不可信。

### 具体任务

1. **Baseline 锁定**：在当前代码（无改动）上跑 3 次完整 GAIA eval（`--split test --level 1,2,3 --limit -1`），记录每次分数和各 level 的正确率。如果三次结果波动 > 2%，先排查原因（rate limit retry、模型温度、随机工具失败等）。
2. **Checkpoint 清理规范**：确保 `--force` 模式行为正确——删除 `.checkpoints/` 下对应文件后重跑。避免旧 checkpoint 污染新结果。
3. **Eval 报告自动化**：每次 eval 后自动生成 JSON 报告，包含：
   - 总正确率、各 level 正确率
   - 每题的 tool call 数、耗时、是否触发 fail_safe
   - 与上次 run 的 diff（新增正确 / 新增错误的题目列表）
4. **回归检测**：维护一个 `golden_set`（约 20 题，覆盖各 level 和各工具组合），Phase 1/2/3 的每次 PR 都必须在 golden_set 上 pass 才能合并。题目应按难度分层抽取，例如 L1×6、L2×8、L3×6，确保各难度均有覆盖，避免 golden_set 偏向简单题而漏测 Level 3 回归。

### 涉及文件

- `src/lilith_agent/runner.py`：`run_agent_on_questions()` 增加统计输出
- 新增：`scripts/eval_report.py`（生成对比报告）
- 新增：`tests/golden_set.json`（稳定题目子集）

---

## P0：里程碑感知压缩（Milestone-Based Compaction）

**路线：A + B | 预估 GAIA 提升：+2-4%**

### 问题

当前 `_compact_old_tool_messages()` 按位置压缩——保留最近 4 条 ToolMessage，老的用 cheap model 总结或截断。这是"时间盲"的：不管 agent 是否完成了一个子任务，压缩都机械地按顺序来。Level 2/3 题通常需要 15-25 步，后半段会丢失前面的关键发现。

研究数据支撑：arxiv 2604.20911 显示，没有记忆缓解措施时，constraint compliance 从 turn 5 的 73% 降到 turn 16 的 33%。

### 实现思路

在 `src/lilith_agent/app.py` 的 `_compact_old_tool_messages()` 基础上改造：

1. **子任务边界检测（用 cheap model，不用关键词匹配）**：
   在 model_node 中，每次 agent 产出 AIMessage 后，调用 cheap model 做一次二分类判断："这条回复是否标志着一个子任务的完成？"返回 yes/no + 一句话摘要。

   **不用关键词匹配的原因**：agent 回复中英文混合，"找到了 X"、"The result is Y"、"根据以上分析"等模式太多太碎，关键词方案脆弱且维护成本高。cheap model 做分类判断更靠谱，虽然多一次 API call，但里程碑检测本身只在长对话中触发（短对话不需要压缩），一次 cheap call 的成本相对可控。

   实现方式：
   ```python
   _MILESTONE_DETECT_PROMPT = (
       "Does this agent message indicate completion of a sub-task or discovery of a key fact? "
       "Answer JSON: {\"is_milestone\": true/false, \"summary\": \"one-line summary if true\"}"
   )
   ```

   > **⚠️ 延迟预算**：cheap model 响应超过 2 秒时，跳过分类，降级为关键词匹配（检测"找到"/"发现"/"完成"/"The result is"等模式）。里程碑检测本身不应成为每步推理的瓶颈，超时降级确保 compaction 逻辑对主流程延迟影响可控。

2. **里程碑摘要生成**：当 cheap model 判断 `is_milestone=true` 时，将其 summary 包装为结构化标记注入消息流：
   ```
   [MILESTONE @ step N]
   - 已完成：{summary}
   - 下一步：由 agent 自行决定
   ```
3. **压缩时保留里程碑**：修改 `_compact_old_tool_messages()` 逻辑——以 `[MILESTONE` 开头的消息永远不被压缩，普通 ToolMessage 正常压缩。这样即使在 turn 20，agent 仍然能看到 turn 3 发现的关键事实。
4. **可选：接入 Anthropic Compaction API**（`compact-2026-01-12`，目前 beta）替代自研总结逻辑。注意 beta 状态有稳定性风险，建议作为可选后端。

### 涉及文件

- `src/lilith_agent/app.py`：`_compact_old_tool_messages()`、`model_node()`
- 新增：`_detect_milestone()` 函数（调用 cheap model 做二分类）

---

## P1：Sub-Agent 架构

**路线：A（Level 3 题）+ B | 预估 GAIA 提升：+2-3%**

### 问题

Lilith 本质是单 agent 循环。Supervisor 只是一个 cheap model 做的旁观者，不能分派子任务。对于 Level 3 题（需要数十步 + 多种工具），单 agent 在长链推理中不可避免地遭遇 "lost in the middle" 效应。

Anthropic 自己的研究表明，子 agent 各自在干净的 context window 中做深度探索，最后只返回 1000-2000 token 的浓缩摘要，在复杂研究任务上表现显著优于单 agent。

### 实现思路

利用 LangGraph 现有能力，在 `build_react_agent()` 中增加子 agent 调度：

1. **新增 `spawn_sub_agent` 工具**：主 agent 可以调用这个工具，传入一个聚焦的子任务描述和任务类型（如 `type="research"` 或 `type="compute"`），spawn 一个独立的 ReAct agent 实例：
   - 子 agent 有自己干净的 context window
   - **精简工具集（按任务类型裁剪）**：子 agent 不应该拿到全部工具，工具越多走偏概率越大。按类型分配：
     - `research` 类型：`web_search`, `fetch_url`, `read_file`, `inspect_pdf`
     - `compute` 类型：`run_python`, `read_file`, `write_file`
     - `vision` 类型：`inspect_visual_content`, `read_file`
     - 所有类型都**不包含** `spawn_sub_agent`（防止递归爆炸，depth=1 硬限制）
   - 子 agent 有独立的、更小的 budget（如 cap=10）
   - **摘要长度按任务类型动态调整**：`research` 类型可能需要更多空间来传递多个发现（≤3000 chars），`compute` 类型通常一个数字就够了（≤500 chars）。不用一刀切 2000 chars。
2. **主 agent 综合结果**：子 agent 的摘要作为 ToolMessage 返回给主 agent，主 agent 基于多个子 agent 的汇报做最终推理。
3. **并行 vs 串行**：初期实现串行 spawn（简单可靠），后续可考虑并行。

### 涉及文件

- `src/lilith_agent/app.py`：`build_react_agent()` 中注册新工具
- 新增：`src/lilith_agent/tools/sub_agent.py`（包含工具集裁剪逻辑）
- `src/lilith_agent/config.py`：新增 `sub_agent_budget`、`sub_agent_summary_caps` 等配置

### 注意事项

- 子 agent 不应该再 spawn 子 agent（防止递归爆炸），加一个 depth=1 限制
- GAIA 的 Level 1 题不需要子 agent，可以通过 supervisor 或 heuristic 判断何时触发
- 子 agent 的 system prompt 应该更聚焦：只描述子任务目标，不包含主 agent 的完整指令集

---

## P2：经验反思学习（ERL）

**路线：A | 预估 GAIA 提升：+5-8%**

### 问题

Lilith 的 `memory.py` 有长期记忆存储（SQLite），但只存事实性记忆，没有"从失败中学习"的反思机制。每次任务都是从零开始，不会利用过往任务的经验。

ERL（arxiv 2603.24639）在 GAIA2 上比 ReAct baseline 提高了 7.8%，且完全在 prompt-time 运行，不需要梯度更新。

### 澄清：ERL 不是强化学习

尽管名字里有 "Learning"，ERL 不涉及 reward signal 或参数更新。流程是：
1. 跑完一个任务后，让 LLM 回顾轨迹（成功或失败）
2. 提炼出文字形式的启发式规则（heuristics），例如：
   - "遇到 Wikipedia 表格数据时，用 `run_python` + pandas 解析比直接读文本更准确"
   - "当搜索返回付费墙页面时，立即换搜索词而非重试同一 URL"
   - "多步数学题必须在 Python 中验证，不要心算"
3. 将规则存入数据库，标注适用的任务类型/关键词
4. 下次遇到类似任务时，检索相关规则注入 system prompt

### 实现思路（分两步：先验证上限，再自动化）

**Step 1：Golden Heuristics 验证上限（1-2 天）**

在投入自动化反思管线之前，先手动验证 ERL 的潜力上限：

1. 手动挑选 20-30 条高质量 heuristics，来源包括：
   - 分析 `.last_failures.txt` 中的失败题目轨迹，人工总结失败原因
   - 复盘已有的成功题目，提取可复用的策略
   - 参考 ERL 论文中的 heuristic 示例
2. 将这些 golden heuristics 硬编码注入 system prompt
3. 跑一轮完整 GAIA eval，对比 baseline
4. 如果提升 < 2%，需要重新审视检索精度和规则质量，再决定是否投入 Step 2

**Step 2：自动化反思管线**

确认上限足够高之后再搭建：

1. **任务后反思**（Post-Task Reflection）：在 `runner.py` 的 `run_agent_on_questions` 中，每个任务完成后（无论成功失败），调用 cheap model 分析轨迹，生成 heuristics。
2. **规则存储**：在 `memory.py` 的 `MemoryStore` 中新增 `heuristics` 表：
   ```sql
   CREATE TABLE heuristics (
       id TEXT PRIMARY KEY,
       rule TEXT NOT NULL,            -- 启发式规则文本
       source_task_id TEXT,           -- 来源任务
       success BOOLEAN,               -- 该任务是否成功
       keywords TEXT,                 -- 适用关键词（用于检索）
       times_applied INTEGER DEFAULT 0,   -- 被检索注入的次数
       times_helped INTEGER DEFAULT 0,    -- 注入后任务成功的次数
       times_hurt INTEGER DEFAULT 0,      -- 注入后任务失败的次数
       confidence REAL DEFAULT 0.5,       -- 动态置信度 = helped / (helped + hurt)
       created_at TEXT,
       retired_at TEXT                    -- 被淘汰的时间（NULL = 活跃）
   );
   ```
3. **规则检索与注入**：在 `model_node()` 的 iteration 0 阶段，除了现有的 `retrieve_relevant_context()` 之外，额外检索匹配的 heuristics 并注入 system prompt。只注入 `confidence >= 0.4` 且 `retired_at IS NULL` 的规则。
4. **规则淘汰机制（必须在 Step 2 一开始就设计好）**：
   - 每次任务结束后，回溯本次注入了哪些 heuristics，更新 `times_applied`、`times_helped` 或 `times_hurt`
   - 当 `confidence < 0.3` 且 `times_applied >= 5` 时，自动 retire（设置 `retired_at`）
   - 保留 retired 规则不删除（方便分析），但不再注入
   - 定期（每 50 个任务）输出一次规则健康报告：活跃/退役/低置信度分布

   **为什么淘汰机制要早做**：低质量规则积累起来会污染 system prompt。一条错误的 heuristic（如"Wikipedia 表格总是在第二个 section"）可能让 agent 在本来能做对的题上反而出错。

### 涉及文件

- `src/lilith_agent/memory.py`：扩展 `MemoryStore`，新增 heuristics 表和淘汰逻辑
- `src/lilith_agent/runner.py`：任务后反思 + heuristic 效果追踪
- `src/lilith_agent/app.py`：`model_node()` 中的规则注入（带置信度过滤）
- 新增：`src/lilith_agent/reflection.py`（反思逻辑独立模块）
- 新增：`scripts/heuristic_health_report.py`（规则健康报告）

---

## P3：Supervisor 升级为 Strong Model

**路线：A | 预估 GAIA 提升：+1-2%**

### 问题

当前 supervisor 用的是 cheap model，判断质量有限。特别是在 Level 2/3 题上，cheap model 可能无法准确评估 agent 是否已经收集到足够的证据。

### 实现思路

在 `src/lilith_agent/app.py` 的 `build_react_agent()` 中：

1. 将 `supervisor_model = get_cheap_model(cfg)` 改为 `supervisor_model = get_extra_strong_model(cfg)` 或新增一个 `get_strong_model(cfg)` 中间档。
2. 由于 strong model 更贵，可以提高 supervisor 触发阈值：`_SUPERVISOR_MIN_TOOL_CALLS` 从 5 提到 8，减少不必要的调用。
3. 同时给 supervisor prompt 加入更明确的评判标准，比如：
   - "如果 agent 已经找到了一个具体的数字/名字/日期，且该答案与已收集的证据一致，则 status=finalize"
   - "如果 agent 在最近 3 次 tool call 中没有获得新信息，则 status=finalize"

### 涉及文件

- `src/lilith_agent/app.py`：`build_react_agent()` 中的 supervisor 初始化
- `src/lilith_agent/config.py`：新增 `supervisor_model_tier` 配置（cheap/strong/extra_strong）

---

## P4：fetch_url 结果二次清洗

**路线：A | 预估 GAIA 提升：+0.5-1%**

### 问题

`fetch_url` 用 trafilatura 提取正文后截断到 `max_chars=8000`。但 trafilatura 输出经常还残留导航栏文字、cookie 提示、重复 header、社交分享按钮文字等噪音。同样 8000 字符，噪音越少有效信息密度越高。

### 实现思路

在 `src/lilith_agent/tools/web.py` 的 `fetch_url` 函数中，trafilatura 提取之后、截断之前，加一步轻量清洗。

**Noise patterns 外置成配置文件**（不硬编码在函数里）：

```yaml
# config/noise_patterns.yaml
# 不同语言的页面噪音模式差异大，外置方便维护和扩展
en:
  - cookie
  - privacy policy
  - terms of service
  - subscribe
  - newsletter
  - follow us
  - share on
  - tweet this
  - skip to content
  - advertisement
  - sponsored
zh:
  - 隐私政策
  - 使用条款
  - 订阅
  - 关注我们
  - 分享到
  - 跳转到主内容
  - 广告
```

清洗函数从配置文件读取 patterns：

```python
import yaml
from pathlib import Path

_NOISE_CONFIG = Path(__file__).parent.parent / "config" / "noise_patterns.yaml"

def _load_noise_patterns() -> list[str]:
    """Load all noise patterns from config, flattened across languages."""
    if not _NOISE_CONFIG.exists():
        return []  # graceful fallback: no cleaning if config missing
    with open(_NOISE_CONFIG) as f:
        data = yaml.safe_load(f) or {}
    patterns = []
    for lang_patterns in data.values():
        if isinstance(lang_patterns, list):
            patterns.extend(lang_patterns)
    return patterns

def _post_clean(text: str) -> str:
    """Remove common trafilatura residual noise."""
    noise_patterns = _load_noise_patterns()
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped or len(stripped) < 3:
            continue
        lower = stripped.lower()
        if any(p in lower for p in noise_patterns) and len(stripped) < 80:
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)
```

### 涉及文件

- `src/lilith_agent/tools/web.py`：`fetch_url()` / `_fetch_url()`
- 新增：`src/lilith_agent/config/noise_patterns.yaml`（噪音模式配置）

---

## P5：动态 Budget 管理

**路线：A | 预估 GAIA 提升：+0.5-1%**

### 问题

当前 `budget_hard_cap=25` 和 `budget_warn_at=15` 是固定值，不区分题目难度。Level 1 题通常 3-5 步就能回答，但 agent 可能因为不够果断而用掉 10+ 步；Level 3 题可能真的需要 30+ 步。

### 实现思路

1. 在任务开始时，让 model 或 cheap model 快速评估题目难度（Level 1/2/3 风格），设置对应的 budget：
   - 简单题：warn=8, cap=12
   - 中等题：warn=15, cap=25（当前默认）
   - 复杂题：warn=25, cap=40（配合更积极的压缩）
2. 或者更简单：如果 GAIA 数据集本身带 level 标签，直接用标签设 budget。

### 涉及文件

- `src/lilith_agent/app.py`：`_route_after_model()`、`model_node()`
- `src/lilith_agent/config.py`：新增 per-level budget 配置
- `src/lilith_agent/runner.py`：传递 level 信息给 agent

---

## 实施顺序建议

```
Phase 0（前置条件，2-3 天）
└── Eval Pipeline 稳定化
    ├── 跑 3 次 baseline，锁定当前分数
    ├── 搭建 eval 报告脚本
    └── 建立 golden_set 回归测试

Phase 1（快速见效，1-2 周）
├── P3：Supervisor 升级（改一行代码 + 调 prompt）
├── P4：fetch_url 清洗（加函数 + 外置 noise config）
├── P5：动态 Budget（按 level 调参数）
└── ⚡ 跑完整 GAIA eval，对比 Phase 0 baseline

Phase 2（核心改造，2-4 周）
├── P2 Step 1：Golden Heuristics 手动验证（1-2 天）
│   └── 如果提升 < 2%，暂停 P2，重新审视
├── P0：里程碑压缩（cheap model 做分类 + 改造 compaction）
├── P2 Step 2：自动化反思管线（含淘汰机制）
└── ⚡ 跑完整 GAIA eval，对比 Phase 1

Phase 3（架构升级，3-5 周）
├── P1：Sub-Agent 架构（精简工具集 + 动态摘要上限）
└── ⚡ 跑完整 GAIA eval，对比 Phase 2
```

**关键原则**：每个 Phase 结束时必须跑完整 GAIA eval。如果某个改动导致 golden_set 回归，立即 revert 并分析原因，不带入下一个 Phase。

Phase 0 是所有后续工作的基础。Phase 1 的三项改动都很小、风险低，适合快速验证。Phase 2 的 ERL 先用手动 golden heuristics 验上限再决定是否投入自动化——避免花两周搭管线结果发现提升不大。Phase 3 改动最大但对 Level 3 长链题帮助最显著。

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
