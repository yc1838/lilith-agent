import pytest
from pathlib import Path
import logging
from lilith_agent.config import Config
from lilith_agent.app import build_react_agent
from langchain_core.messages import HumanMessage

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


def test_summarize_episode_stores_list_block_content_as_text(tmp_path, monkeypatch):
    from lilith_agent import memory

    class FakeModel:
        def invoke(self, prompt):
            class Response:
                content = [
                    {"type": "text", "text": "Captured lesson"},
                    {"type": "non_text", "value": "ignored"},
                ]

            return Response()

    store = memory.MemoryStore(tmp_path / "long_term_memory.sqlite")
    monkeypatch.setattr(memory, "_store", store)

    memory.summarize_episode(
        [HumanMessage(content=[{"type": "text", "text": "Remember this"}])],
        FakeModel(),
    )

    episodes = store.get_recent_episodes()
    assert episodes[0]["task"] == "Remember this"
    assert episodes[0]["summary"] == "Captured lesson"


def test_summarize_episode_prints_saved_episode_details(tmp_path, monkeypatch, caplog, capsys):
    from lilith_agent import memory

    class FakeModel:
        def invoke(self, prompt):
            class Response:
                content = "Used read_file successfully and learned to inspect spreadsheet rows."

            return Response()

    store = memory.MemoryStore(tmp_path / "long_term_memory.sqlite")
    monkeypatch.setattr(memory, "_store", store)

    with caplog.at_level(logging.INFO, logger="lilith_agent.memory"):
        memory.summarize_episode([HumanMessage(content="Find oldest Blu-Ray title")], FakeModel())

    printed = capsys.readouterr().out
    assert "[memory] Episode saved: task='Find oldest Blu-Ray title'" in printed
    assert "outcome='success'" in printed
    assert "summary='Used read_file successfully" in printed

    record = next(r for r in caplog.records if "[memory] Episode saved:" in r.message)
    assert "task='Find oldest Blu-Ray title'" in record.message
    assert "outcome='success'" in record.message
    assert "summary='Used read_file successfully" in record.message


def test_summarize_episode_logs_traceback_on_failure(tmp_path, monkeypatch, caplog):
    from lilith_agent import memory

    class BrokenModel:
        def invoke(self, prompt):
            raise RuntimeError("boom")

    store = memory.MemoryStore(tmp_path / "long_term_memory.sqlite")
    monkeypatch.setattr(memory, "_store", store)

    with caplog.at_level(logging.ERROR, logger="lilith_agent.memory"):
        memory.summarize_episode([HumanMessage(content="Remember this")], BrokenModel())

    record = next(r for r in caplog.records if "Summarization failed" in r.message)
    assert record.exc_info is not None


def test_extract_and_compress_facts_passes_existing_memories_with_ids(tmp_path, monkeypatch):
    from lilith_agent import memory
    import langmem

    captured = {}

    class FakeManager:
        def invoke(self, payload):
            captured["existing"] = payload["existing"]
            return []

    store = memory.MemoryStore(tmp_path / "long_term_memory.sqlite")
    store.save_memories([{"id": "memory-1", "content": "Existing fact"}])
    monkeypatch.setattr(memory, "_store", store)
    monkeypatch.setattr(langmem, "create_memory_manager", lambda model, enable_deletes: FakeManager())

    class FakeModel:
        def invoke(self, prompt):
            class Response:
                content = "Lesson"

            return Response()

    memory.extract_and_compress_facts([HumanMessage(content="New fact")], FakeModel())

    existing_id, existing_memory = captured["existing"][0]
    assert existing_id == "memory-1"
    assert existing_memory.content == "Existing fact"
