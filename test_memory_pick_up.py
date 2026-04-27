import os
from lilith_agent.memory import _store, retrieve_relevant_context

# Initialize DB via _store
_store._init_db()

# Manually inject
from datetime import datetime
import sqlite3
conn = sqlite3.connect(str(_store.db_path))
with conn:
    conn.execute("DELETE FROM memories")
    now = datetime.now().isoformat()
    conn.execute(
        "INSERT INTO memories (id, content, type, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        ("test-fact-1", "The user's secret code name is 'Blue Butterfly'.", "fact", now, now)
    )
conn.close()

context = retrieve_relevant_context("Who am I?")
print("\nRetrieved Context:")
print(context)
