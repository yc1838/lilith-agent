import pytest
from pathlib import Path
from lilith_agent.config import Config
from lilith_agent.app import build_react_agent

def test_build_react_agent_uses_sqlite_saver(tmp_path):
    cfg = Config.from_env()
    
    # Temporarily override where the agent looks for the .lilith dir
    import os
    os.environ["LILITH_HOME"] = str(tmp_path / ".lilith")
    
    agent = build_react_agent(cfg)
    
    assert agent.checkpointer is not None
    assert type(agent.checkpointer).__name__ == "SqliteSaver"
    
    # Check if DB file was created
    db_path = tmp_path / ".lilith" / "threads.sqlite"
    assert db_path.exists()
