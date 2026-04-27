# Persistent Memory Design for Lilith Agent (LangMem)

## Overview

This document outlines the design for a three-tiered persistent memory system for the Lilith Agent, utilizing `langmem` and inspired by DeepSeek V4's architecture to prevent memory explosion.

The goal is to enable Lilith to:
1.  **Resume interrupted conversations** (Short-Term/Thread Persistence).
2.  **Remember user preferences and static facts** across sessions (Semantic Memory).
3.  **Learn from past complex tasks** to avoid repeating mistakes (Episodic Memory).

## Architecture: Three-Tiered Memory

Inspired by DeepSeek V4's separation of static knowledge (Engram) from dynamic reasoning, and its aggressive compression mechanisms (HCA), we will implement the following tiers:

### 1. Short-Term Memory (Thread Persistence)
*   **Purpose:** Allows resuming the exact state of a conversation after the CLI/TUI is closed.
*   **Mechanism:** Replace LangGraph's in-memory `MemorySaver` with `SqliteSaver`.
*   **Storage:** Local SQLite database at `.lilith/threads.sqlite`.
*   **Implementation:**
    *   Add `langgraph-checkpoint-sqlite` to dependencies.
    *   Initialize `SqliteSaver(conn)` in `app.py::build_react_agent`.
    *   Ensure the TUI and batch runner correctly pass and reuse `thread_id`s.

### 2. Semantic Memory (Static Knowledge / "Engram")
*   **Purpose:** Stores atomic facts, user preferences, and environmental knowledge extracted from conversations (e.g., "User prefers Python 3.11", "API key X is used for service Y").
*   **Mechanism:** Background extraction and active compression using `langmem`.
*   **Storage:** Local vector store (e.g., Chroma or local SQLite + vector extensions) managed by `langmem` at `.lilith/semantic_memory`.
*   **Implementation:**
    *   Add `langmem` to dependencies.
    *   Create an asynchronous background task that runs at the end of a successful graph execution (or periodically).
    *   Use the `cheap_model` (e.g., `gemini-3-flash-preview`) to extract new facts from the recent conversation.
    *   **Anti-Bloat Compression (HCA equivalent):** Before saving, query the vector store for similar existing facts. Instruct the model to merge, update, or delete existing facts to resolve contradictions and maintain a highly compressed, deduplicated knowledge base.
    *   **Sparse Retrieval (Lightning Indexer equivalent):** At the start of a new task, embed the user's query, perform a Top-K (e.g., K=3) search against the semantic memory, and inject only the relevant facts into the `SystemMessage`.

### 3. Episodic Memory (Task Experiences)
*   **Purpose:** Remembers the trajectories and outcomes of complex tasks (e.g., "When parsing a GAIA PDF, `pypdf` failed but `pdfplumber` worked").
*   **Mechanism:** Summarization of successful (or informatively failed) task executions.
*   **Storage:** Local vector store managed by `langmem` at `.lilith/episodic_memory`.
*   **Implementation:**
    *   When the agent reaches the `END` node with a final answer, trigger an episodic summarizer (using the `cheap_model`).
    *   The summarizer condenses the entire ReAct trajectory (tools used, errors encountered, successful path) into a concise "episode" summary.
    *   **Sparse Retrieval:** Similar to semantic memory, fetch Top-K (e.g., K=1 or 2) relevant past episodes based on the new task's initial query and inject them into the system prompt as historical context.

## Component Interactions

1.  **Start of Task:**
    *   User inputs a query.
    *   Agent embeds the query and searches Semantic and Episodic memory.
    *   Agent constructs the initial `SystemMessage`, injecting retrieved facts and past episodes.
2.  **During Task (Reasoning):**
    *   Agent executes the ReAct loop, saving state to the `SqliteSaver` (Short-Term memory) at each step.
    *   Context is managed using existing compaction logic to prevent context window explosion.
3.  **End of Task (Extraction & Compression):**
    *   Task concludes (success or failure).
    *   Background process reads the thread history from `SqliteSaver`.
    *   **Extract:** Identify new facts and summarize the episode.
    *   **Compress:** Deduplicate and merge new facts with existing facts in the Semantic vector store.
    *   **Store:** Save the updated facts and the new episode summary.

## Data Schema (Conceptual)

**Semantic Fact (Engram)**
```json
{
  "id": "uuid",
  "content": "User strictly uses Python 3.11 for all scripts.",
  "type": "preference",
  "last_updated": "2026-04-26T...",
  "embedding": [0.1, 0.2, ...]
}
```

**Episode Summary**
```json
{
  "id": "uuid",
  "task_description": "Extract text from a scanned PDF for GAIA benchmark.",
  "summary": "Attempted pypdf first, which failed due to scanned image format. Pivoted to using OCR via inspect_visual_content, which successfully extracted the text.",
  "outcome": "success",
  "timestamp": "2026-04-26T...",
  "embedding": [0.3, 0.4, ...]
}
```

## Dependencies
*   `langgraph-checkpoint-sqlite`
*   `langmem`
*   Vector database client (e.g., `chromadb` or equivalent compatible with `langmem` for local storage).

## Next Steps (Implementation Plan)
1.  Implement `SqliteSaver` for thread persistence.
2.  Set up the `langmem` infrastructure (local vector store).
3.  Implement Semantic Memory extraction and sparse retrieval.
4.  Implement active compression/merging logic for facts to prevent bloat.
5.  Implement Episodic Memory summarization and retrieval.
