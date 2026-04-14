from __future__ import annotations

import json
import logging
import re
import string
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import MessagesState, add_messages
from typing import Annotated, TypedDict

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    iterations: int


from lilith_agent.config import Config
from lilith_agent.models import get_extra_strong_model

log = logging.getLogger(__name__)



def _call_key(name: str, args) -> tuple[str, str]:
    try:
        norm = json.dumps(args or {}, sort_keys=True, default=str)
    except Exception:
        norm = repr(args)
    return (name, norm)


def _collect_seen_calls(messages) -> set[tuple[str, str]]:
    """All (tool_name, args) pairs already requested in prior AI messages."""
    seen: set[tuple[str, str]] = set()
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in getattr(m, "tool_calls", None) or []:
                seen.add(_call_key(tc.get("name"), tc.get("args")))
    return seen


_COMPACT_KEEP_RECENT = 4
_COMPACT_MAX_CHARS = 300

_BUDGET_WARN_AT = 15
_BUDGET_HARD_CAP = 25

_SEMANTIC_DEDUP_THRESHOLD = 0.5
_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "for", "to", "and", "or", "but",
    "is", "are", "was", "were", "be", "been", "being", "by", "with", "from",
    "as", "that", "this", "it", "its", "which", "who", "whom", "what", "when",
    "where", "why", "how", "do", "does", "did", "can", "could", "should",
    "would", "will", "about", "into", "over", "under", "than", "then", "so",
    "if", "not", "no", "yes", "any", "all", "some", "each", "every",
}


def _normalize_query_tokens(q: str) -> frozenset[str]:
    q = q.lower()
    q = q.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in re.split(r"\s+", q) if t and t not in _STOPWORDS]
    return frozenset(tokens)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _count_tool_calls_since_last_human(messages: list) -> int:
    """Count AIMessage tool_calls made after the most recent HumanMessage."""
    count = 0
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            count += len(m.tool_calls)
    return count


def _prior_tavily_queries(messages: list) -> list[tuple[str, frozenset[str]]]:
    """All tavily_search queries from prior AIMessages in this turn (since last HumanMessage)."""
    out: list[tuple[str, frozenset[str]]] = []
    collecting = True
    for m in messages:
        if isinstance(m, HumanMessage):
            out = []
            collecting = True
            continue
        if collecting and isinstance(m, AIMessage):
            for tc in getattr(m, "tool_calls", None) or []:
                if tc.get("name") == "tavily_search":
                    q = (tc.get("args") or {}).get("query", "")
                    if q:
                        out.append((q, _normalize_query_tokens(q)))
    return out


def _compact_old_tool_messages(messages: list, keep_recent: int = _COMPACT_KEEP_RECENT, max_chars: int = _COMPACT_MAX_CHARS) -> list:
    """Return a shallow-copied message list where older ToolMessage contents are truncated.

    Tool results often dominate context (search dumps, page fetches). Keep the `keep_recent`
    most recent ToolMessages verbatim; older ones get shrunk to `max_chars` with a clear
    "[COMPACTED: N chars dropped]" suffix so the model knows data was pruned but still
    sees the lead-in to recall what the call was about.
    """
    tool_indices = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    keep_indices = set(tool_indices[-keep_recent:])

    out = []
    for i, m in enumerate(messages):
        if isinstance(m, ToolMessage) and i not in keep_indices:
            content = str(m.content)
            if len(content) > max_chars:
                dropped = len(content) - max_chars
                new_content = content[:max_chars] + f"\n...[COMPACTED: {dropped} chars dropped from older tool result]..."
                m = m.model_copy(update={"content": new_content})
        out.append(m)
    return out


def _build_tool_node(tools: list[BaseTool]) -> Callable:
    """Tool executor with dedup + exception-to-ToolMessage feedback.

    Dedup rule: if the same (tool_name, args) pair appeared in any prior
    AIMessage in history, short-circuit with a synthetic ToolMessage telling
    the model it already tried this, without invoking the tool.
    """
    tools_by_name = {t.name: t for t in tools}

    def tool_node(state):
        messages = state["messages"]
        last = messages[-1]
        tool_calls = getattr(last, "tool_calls", None) or []

        # "seen" = calls that appeared in AIMessages strictly BEFORE the current one.
        seen = _collect_seen_calls(messages[:-1])
        prior_tavily = _prior_tavily_queries(messages[:-1])

        def count_recent_errors(tool_name: str) -> int:
            count = 0
            for m in reversed(messages):
                if isinstance(m, ToolMessage) and m.name == tool_name:
                    if getattr(m, "status", "") == "error":
                        count += 1
                    else:
                        break
                # Only check contiguous blocks of prior tools/AI messages
                elif isinstance(m, AIMessage):
                    continue
                else: 
                    break
            return count

        results: list[ToolMessage] = []
        for tc in tool_calls:
            name = tc.get("name")
            args = tc.get("args") or {}
            tc_id = tc.get("id", "")

            key = _call_key(name, args)
            if key in seen:
                log.warning("[dedup] short-circuited repeat tool call: %s %s", name, args)
                results.append(ToolMessage(
                    tool_call_id=tc_id,
                    name=name or "unknown",
                    content=(
                        f"You already called `{name}` with these exact arguments earlier "
                        "in this conversation and received a result. Do not repeat the same "
                        "call — try different arguments, a different tool, or use the prior "
                        "result to answer the user."
                    ),
                    status="error",
                ))
                continue

            if name == "tavily_search":
                q = (args or {}).get("query", "")
                if q:
                    q_tokens = _normalize_query_tokens(q)
                    best_prior, best_score = None, 0.0
                    for prior_q, prior_tokens in prior_tavily:
                        score = _jaccard(q_tokens, prior_tokens)
                        if score > best_score:
                            best_prior, best_score = prior_q, score
                    if best_score >= _SEMANTIC_DEDUP_THRESHOLD:
                        log.warning("[semantic_dedup] %.2f match vs prior: %r ~ %r", best_score, q, best_prior)
                        results.append(ToolMessage(
                            tool_call_id=tc_id,
                            name=name,
                            content=(
                                f"PATH EXHAUSTED (similarity={best_score:.2f}). "
                                f"Your query {q!r} is too similar to a prior search {best_prior!r}. "
                                "This search path is likely dead. DO NOT keep tweaking the query. "
                                "Summarize the best possible guess from existing snippets and move to a final answer immediately. "
                                "If you must continue searching, use a COMPLETELY different keyword or strategy."
                            ),
                            status="error",
                        ))
                        continue

            if count_recent_errors(name) >= 3:
                log.warning("[loop_breaker] force-cooldown %s", name)
                results.append(ToolMessage(
                    tool_call_id=tc_id,
                    name=name,
                    content=(
                        f"SYSTEM CRITICAL: You have consecutively failed to use `{name}` 3 times. "
                        "This tool is now placed on temporary COOLDOWN. "
                        "You MUST shift your strategy and use a different tool or provide a final answer based on what you know."
                    ),
                    status="error"
                ))
                continue

            tool = tools_by_name.get(name)
            if tool is None:
                results.append(ToolMessage(
                    tool_call_id=tc_id,
                    name=name or "unknown",
                    content=f"ERROR: unknown tool {name!r}. Available: {sorted(tools_by_name)}",
                    status="error",
                ))
                continue

            try:
                out = tool.invoke(args)
            except Exception as e:
                log.warning("[tool_node] %s raised: %s", name, e)
                out = f"ERROR: {type(e).__name__}: {e}"
                if len(out) > 1000:
                    out = out[:1000] + "\n...[TRUNCATED BY SYSTEM TO PREVENT CONTEXT COLLAPSE]..."
                results.append(ToolMessage(tool_call_id=tc_id, name=name, content=str(out), status="error"))
                continue

            results.append(ToolMessage(tool_call_id=tc_id, name=name, content=str(out)))

        return {"messages": results}

    return tool_node


CAVEMAN_SYSTEM = (
    "Respond terse like smart caveman. All technical substance stay. Only fluff die.\n\n"
    "Intensity: {mode}\n\n"
    "RULES:\n"
    "Drop: articles (a/an/the), filler (just/really/basically/actually/simply), pleasantries (sure/certainly/of course/happy to), hedging.\n"
    "Fragments OK. Short synonyms (big not extensive, fix not 'implement a solution for').\n"
    "Technical terms exact. Code blocks unchanged. Errors quoted exact.\n\n"
    "Pattern: [thing] [action] [reason]. [next step].\n\n"
    "INTENSITY LEVELS:\n"
    "- lite: No filler/hedging. Keep articles + full sentences. Professional but tight.\n"
    "- full: Drop articles, fragments OK, short synonyms. Classic caveman.\n"
    "- ultra: Abbreviate (DB/auth/config/req/res/fn/impl), strip conjunctions, arrows for causality (X -> Y), one word when one word enough.\n"
)


def apply_caveman(base_prompt: str, caveman_enabled: bool, mode: str = "full") -> str:
    if not caveman_enabled:
        return base_prompt
    
    caveman_instructions = CAVEMAN_SYSTEM.format(mode=mode)
    return f"{caveman_instructions}\n\nREMAINING SYSTEM INSTRUCTIONS (FOLLOW THESE EXACTLY BUT IN CAVEMAN STYLE):\n{base_prompt}"


def build_react_agent(cfg: Config):
    """Explicit ReAct graph with tool-call dedup, error feedback, and recursion cap."""
    try:
        from lilith_agent.tools import build_tools
        tools = build_tools(cfg)
    except ImportError:
        log.warning("Tools not found; running with zero tools.")
        tools = []

    model = get_extra_strong_model(cfg).bind_tools(tools)

    def model_node(state):
        from langchain_core.messages import SystemMessage
        base_prompt = (
            "You are Lilith, an autonomous ReAct research assistant operating in a continuous session.\n\n"
            "CRITICAL DIRECTIVES FOR EXECUTION:\n"
            "1. STOP AT CONFIDENCE: Once you have gathered the key facts needed to answer the user's core question, "
            "IMMEDIATELY stop calling tools and output your final conclusion. Do not seek absolute 100% certainty if you already have a highly probable answer.\n"
            "2. ANTI-RABBIT-HOLE: If you find yourself running multiple variations of the same search or Python script to find a specific missing link (e.g., trying to link a specific name to a specific file or commit), STOP. "
            "Use the strongest evidence you have gathered so far. Exhaustive verification leads to infinite loops.\n"
            "3. NO REDUNDANT CHECKS: Do NOT run redundant tools just to double-check. "
            "If you found a name, number, or fact that fits the constraints, output it as the answer immediately.\n"
            "4. CONTEXT RESOLUTION: Treat the conversation history purely as read-only background context. "
            "Your active formatting rules are dictated ENTIRELY by the user's most recent message.\n"
            "5. NO-RETRY GUIDELINES: If you encounter a paywall, CAPTCHA, or 'Semantic Duplicate' error, consider that path dead. "
            "Summarize the best possible guess from snippets and move to Final Answer immediately. NEVER output an empty response."
        )
        sys_prompt = apply_caveman(base_prompt, cfg.caveman, cfg.caveman_mode)
        sys_msg = SystemMessage(sys_prompt)
        
        # Message Compaction
        compacted = _compact_old_tool_messages(state["messages"])

        prompt_msgs = [sys_msg]
        tool_calls_this_turn = _count_tool_calls_since_last_human(state["messages"])
        if tool_calls_this_turn >= _BUDGET_WARN_AT:
            prompt_msgs.append(SystemMessage(
                f"[BUDGET WARNING: {tool_calls_this_turn} tool calls used on this question "
                f"(hard cap at {_BUDGET_HARD_CAP}). STOP exploring. Commit to your best answer "
                "NOW using evidence already gathered. Further searches are almost certainly wasted — "
                "you already have enough to answer or you need to pivot strategy entirely.]"
            ))
        prompt_msgs += compacted

        response = model.invoke(prompt_msgs)
        
        # Fallback for empty responses
        if not response.content and not getattr(response, "tool_calls", None):
            log.warning("[model_node] blank response detected; injecting system nudge")
            response = AIMessage(content=(
                "SYSTEM: Your previous response was empty. If you have enough information, "
                "provide a 'Final Answer:'. If you are stuck, return a best guess based on "
                "available snippets or explain why you cannot complete the task."
            ))

        return {"messages": [response], "iterations": state.get("iterations", 0) + 1}

    def fail_safe_node(state):
        from langchain_core.messages import SystemMessage
        sys_prompt = (
            "SYSTEM EMERGENCY OVERRIDE: You have hit the absolute maximum iteration limit for this task. "
            "You are FORCED to stop. Provide a brief 'Final Answer:' summarizing what you have tried, "
            "why it failed, and what the best conclusion is so far."
        )
        compacted = _compact_old_tool_messages(state["messages"])
        response = model.invoke([SystemMessage(sys_prompt)] + compacted)
        return {"messages": [response]}

    tool_node = _build_tool_node(tools)

    graph = StateGraph(AgentState)
    graph.add_node("model", model_node)
    graph.add_node("tools", tool_node)
    graph.add_node("fail_safe", fail_safe_node)
    def _route_after_model(state) -> str:
        if state.get("iterations", 0) >= cfg.recursion_limit - 2:
            return "fail_safe"
        if _count_tool_calls_since_last_human(state["messages"]) >= _BUDGET_HARD_CAP:
            log.warning("[hard_cap] per-question tool-call cap hit; forcing fail_safe")
            return "fail_safe"
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return END

    graph.set_entry_point("model")
    graph.add_conditional_edges("model", _route_after_model, {"tools": "tools", "fail_safe": "fail_safe", END: END})
    graph.add_edge("tools", "model")
    graph.add_edge("fail_safe", END)

    compiled = graph.compile(checkpointer=MemorySaver())
    return compiled.with_config({"recursion_limit": cfg.recursion_limit})
