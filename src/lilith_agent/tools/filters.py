from __future__ import annotations
import logging
import json
from typing import Any

log = logging.getLogger(__name__)

def filter_entities(entities: list[dict[str, Any]], keep_conditions: list[str] | None = None, remove_conditions: list[str] | None = None) -> str:
    """Filter a list of entities based on conditions.
    
    Args:
        entities: List of dictionaries to filter.
        keep_conditions: List of substrings that MUST be present in any string field of the entry.
        remove_conditions: List of substrings that MUST NOT be present in any string field of the entry.
    """
    if not entities:
        return "No entities to filter."
        
    filtered = []
    for item in entities:
        # Flatten all string values for easy checking
        all_vals = " ".join([str(v) for v in item.values() if isinstance(v, (str, int, float))]).lower()
        
        keep = True
        if keep_conditions:
            for cond in keep_conditions:
                if cond.lower() not in all_vals:
                    keep = False
                    break
                    
        if keep and remove_conditions:
            for cond in remove_conditions:
                if cond.lower() in all_vals:
                    keep = False
                    break
        
        if keep:
            filtered.append(item)
            
    res = {
        "value": len(filtered),
        "data_source": "filter_entities_tool",
        "record_type": "filtered-list",
        "type_strictness": "exact" if (keep_conditions or remove_conditions) else "medium",
        "original_count": len(entities),
        "filtered_count": len(filtered),
        "items": filtered[:50] # Limit output for context
    }
    
    return json.dumps(res, indent=2)
