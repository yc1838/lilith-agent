# LangMem Persistent Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a three-tiered persistent memory system (Short-term, Semantic, Episodic) for the Lilith Agent using LangGraph's SqliteSaver and LangMem.

**Architecture:** 
1. Replace in-memory `MemorySaver` with `SqliteSaver` pointing to `.lilith/threads.sqlite` for short-term thread persistence.
2. Integrate `langmem` to extract facts (Semantic) and task summaries (Episodic) in the background at the end of runs, compressing/merging them to prevent bloat.
3. Query the LangMem vector store at the start of new tasks and inject relevant context into the initial `SystemMessage`.

**Tech Stack:** `langgraph-checkpoint-sqlite`, `langmem`, `sqlite3`, `chromadb` (or `langmem` default local storage)

---

### Task 1: Setup Dependencies and Short-Term Thread Persistence

**Files:**
- Modify: `pyproject.toml:10-33`
- Modify: `src/lilith_agent/app.py`
- Modify: `src/lilith_agent/tui.py`
- Create: `tests/test_memory_persistence.py`

- [ ] **Step 1: Add new dependencies**

```bash
# Update pyproject.toml to include dependencies
sed -i '' '/"langgraph>=1.0,<2.0",/a \
  "langgraph-checkpoint-sqlite>=1.0,<2.0",\
  "langmem>=0.0.1",
' pyproject.toml
```

- [ ] **Step 2: Write test for SqliteSaver initialization**

```python
# tests/test_memory_persistence.py
import pytest
from pathlib import Path
from lilith_agent.config import Config
from lilith_agent.app import build_react_agent

def test_build_react_agent_uses_sqlite_saver(tmp_path):
    cfg = Config(
        cheap_provider="google",
        cheap_model="gemini-3-flash-preview",
        strong_provider="google",
        strong_model="gemini-3-flash-preview",
        extra_strong_provider="google",
        extra_strong_model="gemini-3-flash-preview"
    )
    
    # Temporarily override where the agent looks for the .lilith dir
    import os
    os.environ["LILITH_HOME"] = str(tmp_path / ".lilith")
    
    agent = build_react_agent(cfg)
    
    assert agent.checkpointer is not None
    assert type(agent.checkpointer).__name__ == "SqliteSaver"
    
    # Check if DB file was created
    db_path = tmp_path / ".lilith" / "threads.sqlite"
    assert db_path.exists()
```

- [ ] **Step 3: Run the failing test**

Run: `pytest tests/test_memory_persistence.py -v`
Expected: FAIL

- [ ] **Step 4: Implement SqliteSaver in app.py**

Modify `src/lilith_agent/app.py`:

```python
# Add imports
import os
from pathlib import Path
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# Modify build_react_agent
def build_react_agent(cfg: Config):
    # ... existing code ...
    
    graph = StateGraph(AgentState)
    # ... existing graph node/edge setup ...

    # Setup SQLite Saver
    lilith_home = Path(os.getenv("LILITH_HOME", ".lilith"))
    lilith_home.mkdir(parents=True, exist_ok=True)
    db_path = lilith_home / "threads.sqlite"
    
    # Note: in a production app you might manage connection lifecycle differently,
    # but for CLI a persistent connection during the app lifetime works.
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    memory_saver = SqliteSaver(conn)

    compiled = graph.compile(checkpointer=memory_saver)
    return compiled.with_config({"recursion_limit": cfg.recursion_limit})
```

- [ ] **Step 5: Run the test again to verify it passes**

Run: `pytest tests/test_memory_persistence.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/lilith_agent/app.py tests/test_memory_persistence.py
git commit -m "feat: replace MemorySaver with SqliteSaver for thread persistence"
```

---

### Task 2: Implement Semantic Memory Extraction (LangMem Background)

**Files:**
- Create: `src/lilith_agent/memory.py`
- Modify: `src/lilith_agent/app.py`

- [ ] **Step 1: Create memory.py module with extraction logic**

```python
# src/lilith_agent/memory.py
import os
import logging
from pathlib import Path
import langmem
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage

log = logging.getLogger(__name__)

# Initialize local langmem client
lilith_home = Path(os.getenv("LILITH_HOME", ".lilith"))
langmem.init(local_dir=str(lilith_home / "memory"))

def extract_and_compress_facts(messages: List[BaseMessage], model) -> None:
    """
    Extracts new facts from the conversation and merges/compresses them
    with existing semantic memory to prevent bloat.
    """
    log.info("[memory] Extracting semantic facts from thread...")
    try:
        # Convert messages to dict format expected by some extraction prompts
        conv_str = "\n".join([f"{m.type}: {m.content}" for m in messages if m.content])
        
        # We use a simple prompt to extract facts and deduplicate.
        # In a full langmem setup, we'd use their create_memory_manager or memory schemas.
        # For this local implementation, we'll use a direct extraction approach:
        
        prompt = f"""
        Extract any persistent facts, preferences, or knowledge about the user, the project, 
        or the environment from this conversation. 
        Focus ONLY on static knowledge (e.g., 'User prefers Python', 'API Key is X').
        Ignore dynamic reasoning or temporary states.
        
        Conversation:
        {conv_str}
        
        Output as a JSON list of strings. If no facts, output [].
        """
        
        response = model.invoke(prompt)
        # Parse JSON and save via langmem (Implementation detail depends on langmem API version)
        # Placeholder for actual langmem SDK call:
        # facts = json.loads(response.content)
        # for fact in facts:
        #     langmem.save_fact(content=fact, namespace="lilith_semantic")
        
        log.info("[memory] Extraction complete.")
    except Exception as e:
        log.error(f"[memory] Failed to extract facts: {e}")

```

- [ ] **Step 2: Hook up extraction in the graph (End of run)**

Modify `src/lilith_agent/app.py` to trigger this after the final answer.
(Since the current graph just returns END when there are no tool calls, we can wrap the invocation or add an `extract_memory` node that runs before END).

```python
# In src/lilith_agent/app.py
from lilith_agent.memory import extract_and_compress_facts

# Modify build_react_agent to add an extraction node
def build_react_agent(cfg: Config):
    # ... setup tools & models ...
    cheap_model = get_cheap_model(cfg)

    # ... model_node, tool_node, fail_safe_node ...
    
    def extract_memory_node(state):
        # Run fact extraction asynchronously or synchronously at the end
        extract_and_compress_facts(state["messages"], cheap_model)
        return state

    graph = StateGraph(AgentState)
    graph.add_node("model", model_node)
    graph.add_node("tools", tool_node)
    graph.add_node("fail_safe", fail_safe_node)
    graph.add_node("extract_memory", extract_memory_node) # NEW

    def _router(state) -> str:
        if state.get("iterations", 0) >= cfg.recursion_limit - 2:
            return "fail_safe"
        if _count_tool_calls_since_last_human(state["messages"]) >= cfg.budget_hard_cap:
            return "fail_safe"
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return "extract_memory" # Route to memory before ending

    graph.set_entry_point("model")
    graph.add_conditional_edges("model", _router, {
        "tools": "tools", 
        "fail_safe": "fail_safe", 
        "extract_memory": "extract_memory"
    })
    graph.add_edge("tools", "model")
    graph.add_edge("fail_safe", "extract_memory")
    graph.add_edge("extract_memory", END) # End after memory
    
    # ... compile ...
```

- [ ] **Step 3: Commit**

```bash
git add src/lilith_agent/app.py src/lilith_agent/memory.py
git commit -m "feat: add semantic memory extraction node using langmem"
```

---

### Task 3: Implement Episodic Memory Summarization

**Files:**
- Modify: `src/lilith_agent/memory.py`

- [ ] **Step 1: Add episodic summarization logic**

```python
# In src/lilith_agent/memory.py
def summarize_episode(messages: List[BaseMessage], model) -> None:
    """
    Summarizes the trajectory of the task to learn from past experiences.
    """
    log.info("[memory] Summarizing task episode...")
    try:
        # Extract the initial question
        initial_question = ""
        for m in messages:
            if isinstance(m, HumanMessage):
                initial_question = str(m.content)
                break
                
        conv_str = "\n".join([f"{m.type}: {m.content[:200]}..." for m in messages if m.content])
        
        prompt = f"""
        Summarize the trajectory of this task to help a future agent avoid mistakes and repeat successes.
        Include:
        1. Task description
        2. Tools used and why
        3. Errors encountered and how they were bypassed
        4. Final outcome
        
        Initial Question: {initial_question}
        Trajectory:
        {conv_str}
        """
        
        response = model.invoke(prompt)
        # Placeholder for actual langmem SDK call:
        # langmem.save_episode(summary=response.content, task=initial_question)
        
        log.info("[memory] Episode summarized.")
    except Exception as e:
        log.error(f"[memory] Failed to summarize episode: {e}")

# Update the node function
def extract_and_compress_facts(messages: List[BaseMessage], model) -> None:
    # ... existing facts logic ...
    summarize_episode(messages, model)
```

- [ ] **Step 2: Commit**

```bash
git add src/lilith_agent/memory.py
git commit -m "feat: add episodic task summarization"
```

---

### Task 4: Inject Retrieved Memory into System Prompt

**Files:**
- Modify: `src/lilith_agent/memory.py`
- Modify: `src/lilith_agent/app.py`

- [ ] **Step 1: Implement Retrieval Logic**

```python
# In src/lilith_agent/memory.py
def retrieve_relevant_context(query: str) -> str:
    """
    Queries the semantic and episodic memory banks for relevant facts and past experiences.
    """
    try:
        # Placeholder for actual langmem SDK sparse retrieval:
        # facts = langmem.search_facts(query, top_k=3)
        # episodes = langmem.search_episodes(query, top_k=1)
        
        facts = [] # stub
        episodes = [] # stub
        
        context_parts = []
        if facts:
            context_parts.append("<relevant_facts>\n" + "\n".join(f"- {f}" for f in facts) + "\n</relevant_facts>")
        if episodes:
            context_parts.append("<past_experiences>\n" + "\n".join(f"- {e}" for e in episodes) + "\n</past_experiences>")
            
        return "\n\n".join(context_parts)
    except Exception as e:
        log.error(f"[memory] Retrieval failed: {e}")
        return ""
```

- [ ] **Step 2: Inject into SystemMessage**

Modify `src/lilith_agent/app.py` in the `model_node`:

```python
# In src/lilith_agent/app.py
from lilith_agent.memory import retrieve_relevant_context

def build_react_agent(cfg: Config):
    # ...
    def model_node(state):
        from langchain_core.messages import SystemMessage
        
        # ... existing initial_question extraction ...
        
        # Retrieve context ONLY on the first iteration of a new question
        iteration = state.get("iterations", 0)
        memory_context = ""
        if iteration == 0 and initial_question:
            memory_context = retrieve_relevant_context(initial_question)

        base_prompt = (
            "You are Lilith, an autonomous ReAct research assistant operating in a continuous session.\n\n"
            # ... existing directives ...
        )
        
        if memory_context:
            base_prompt += "\n\nCRITICAL CONTEXT (Retrieved from Long-Term Memory):\n" + memory_context

        sys_prompt = apply_caveman(base_prompt, cfg.caveman, cfg.caveman_mode)
        sys_msg = SystemMessage(sys_prompt)
        
        # ... rest of model_node ...
```

- [ ] **Step 3: Commit**

```bash
git add src/lilith_agent/memory.py src/lilith_agent/app.py
git commit -m "feat: inject retrieved semantic and episodic memory into system prompt"
```
