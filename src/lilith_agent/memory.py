import os
import logging
from pathlib import Path
import langmem
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage

log = logging.getLogger(__name__)

# Initialize local langmem client
lilith_home = Path(os.getenv("LILITH_HOME", ".lilith"))
# langmem.init(local_dir=str(lilith_home / "memory"))  # Placeholder, SDK API may vary


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
        
        # Placeholder for langmem save_episode logic
        log.info(f"[memory] Episode summarized: {response.content[:100]}...")
    except Exception as e:
        log.error(f"[memory] Failed to summarize episode: {e}")

def extract_and_compress_facts(messages: List[BaseMessage], model) -> None:
    """
    Extracts new facts from the conversation and merges/compresses them
    with existing semantic memory to prevent bloat.
    """
    log.info("[memory] Extracting semantic facts from thread...")
    try:
        # Convert messages to dict format expected by some extraction prompts
        conv_str = "\n".join([f"{m.type}: {m.content}" for m in messages if m.content])
        
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
        
        # Placeholder for langmem save_fact logic
        log.info(f"[memory] Facts extracted: {response.content[:100]}...")
        
        log.info("[memory] Extraction complete.")
    except Exception as e:
        log.error(f"[memory] Failed to extract facts: {e}")
        
    summarize_episode(messages, model)

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
