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
        
        # Placeholder for langmem save_fact logic depending on their local SDK version.
        # In a full langmem cloud setup, you might use memory_manager.
        # Here we just log it as a stub until local vector is fully set up.
        log.info(f"[memory] Facts extracted: {response.content[:100]}...")
        
        log.info("[memory] Extraction complete.")
    except Exception as e:
        log.error(f"[memory] Failed to extract facts: {e}")
