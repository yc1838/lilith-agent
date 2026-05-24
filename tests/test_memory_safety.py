"""Tests for memory safety guards: empty-list wipe prevention and ID stability."""
from langchain_core.messages import HumanMessage

from lilith_agent.memory import MemoryStore, MIN_MESSAGES_FOR_EXTRACTION


class _FakeEpisodeModel:
    def invoke(self, prompt):
        class Response:
            content = "Test lesson"

        return Response()


class TestSaveMemoriesEmptyGuard:
    """save_memories([]) must not wipe existing facts."""

    def test_refuses_empty_list_when_facts_exist(self, tmp_path):
        store = MemoryStore(tmp_path / "test.sqlite")
        store.save_memories([
            {"id": "fact-1", "content": "User prefers dark theme"},
            {"id": "fact-2", "content": "User name is Yujing"},
        ])
        assert len(store.get_all_memories()) == 2

        # Empty list should NOT wipe
        store.save_memories([])

        facts = store.get_all_memories()
        assert len(facts) == 2, "Empty list wiped all facts!"

    def test_empty_list_on_empty_store_is_noop(self, tmp_path):
        store = MemoryStore(tmp_path / "test.sqlite")
        assert len(store.get_all_memories()) == 0

        # Empty save on empty store is fine — no error, still empty
        store.save_memories([])
        assert len(store.get_all_memories()) == 0


class TestExtractDoesNotWipeOnEmptyResult:
    """extract_and_compress_facts must not wipe store when LangMem returns []."""

    def test_langmem_empty_result_preserves_existing_facts(self, tmp_path, monkeypatch):
        from lilith_agent import memory
        import langmem

        class FakeManager:
            def invoke(self, payload):
                return []  # LangMem found nothing new

        store = MemoryStore(tmp_path / "test.sqlite")
        store.save_memories([
            {"id": "fact-1", "content": "Important fact"},
        ])
        monkeypatch.setattr(memory, "_store", store)
        monkeypatch.setattr(langmem, "create_memory_manager", lambda model, enable_deletes: FakeManager())

        memory.extract_and_compress_facts(
            [HumanMessage(content="Hello"), HumanMessage(content="How are you?")],
            _FakeEpisodeModel(),
        )

        facts = store.get_all_memories()
        assert len(facts) == 1, "LangMem empty result wiped existing facts!"
        assert facts[0]["content"] == "Important fact"


class TestMemoryIdStability:
    def test_passes_existing_memories_with_stable_ids_to_langmem(self, tmp_path, monkeypatch):
        from lilith_agent import memory
        import langmem

        captured = {}

        class FakeManager:
            def invoke(self, payload):
                captured["existing"] = payload["existing"]
                return [
                    {"id": "memory-1", "content": "Existing fact"},
                    {"content": "New fact"},
                ]

        store = MemoryStore(tmp_path / "test.sqlite")
        store.save_memories([
            {"id": "memory-1", "content": "Existing fact"},
        ])
        original = store.get_all_memories()[0]
        monkeypatch.setattr(memory, "_store", store)
        monkeypatch.setattr(langmem, "create_memory_manager", lambda model, enable_deletes: FakeManager())

        memory.extract_and_compress_facts(
            [HumanMessage(content="Remember a new fact"), HumanMessage(content="New fact")],
            _FakeEpisodeModel(),
        )

        facts_by_content = {fact["content"]: fact for fact in store.get_all_memories()}
        existing_payload = captured["existing"][0]
        assert existing_payload[0] == "memory-1"
        assert getattr(existing_payload[1], "content") == "Existing fact"
        assert facts_by_content["Existing fact"]["id"] == "memory-1"
        assert facts_by_content["Existing fact"]["created_at"] == original["created_at"]
        assert facts_by_content["New fact"]["id"] != "memory-1"

    def test_langmem_remove_doc_deletes_fact_instead_of_persisting_marker(self, tmp_path, monkeypatch):
        from lilith_agent import memory
        import langmem

        class RemoveDocLike:
            def __repr_name__(self):
                return "RemoveDoc"

        class FakeExtractedMemory:
            def __init__(self, id, content):
                self.id = id
                self.content = content

        class FakeMemory:
            content = "Keep fact"

        class FakeManager:
            def invoke(self, payload):
                return [
                    FakeExtractedMemory("fact-1", RemoveDocLike()),
                    FakeExtractedMemory("fact-2", FakeMemory()),
                ]

        store = MemoryStore(tmp_path / "test.sqlite")
        store.save_memories([
            {"id": "fact-1", "content": "Delete fact"},
            {"id": "fact-2", "content": "Keep fact"},
        ])
        monkeypatch.setattr(memory, "_store", store)
        monkeypatch.setattr(langmem, "create_memory_manager", lambda model, enable_deletes: FakeManager())

        memory.extract_and_compress_facts(
            [HumanMessage(content="Forget delete fact"), HumanMessage(content="Keep fact")],
            _FakeEpisodeModel(),
        )

        facts_by_id = {fact["id"]: fact for fact in store.get_all_memories()}
        assert "fact-1" not in facts_by_id
        assert facts_by_id["fact-2"]["content"] == "Keep fact"

    def test_successful_fact_extraction_also_records_episode(self, tmp_path, monkeypatch):
        from lilith_agent import memory
        import langmem

        class FakeManager:
            def invoke(self, payload):
                return [{"content": "New fact"}]

        class FakeModel:
            def invoke(self, prompt):
                class Response:
                    content = "Successful lesson"

                return Response()

        store = MemoryStore(tmp_path / "test.sqlite")
        monkeypatch.setattr(memory, "_store", store)
        monkeypatch.setattr(langmem, "create_memory_manager", lambda model, enable_deletes: FakeManager())

        memory.extract_and_compress_facts(
            [HumanMessage(content="Remember useful fact"), HumanMessage(content="New fact")],
            FakeModel(),
        )

        episodes = store.get_recent_episodes()
        assert episodes[0]["summary"] == "Successful lesson"

    def test_one_turn_exchange_is_eligible_for_memory_extraction(self):
        assert MIN_MESSAGES_FOR_EXTRACTION <= 2


class TestForgetSafety:
    def test_delete_memory_prefix_does_not_treat_sql_wildcards_as_patterns(self, tmp_path):
        store = MemoryStore(tmp_path / "test.sqlite")
        store.save_memories([
            {"id": "abc-1", "content": "First fact"},
            {"id": "def-1", "content": "Second fact"},
        ])

        assert store.delete_memory_prefix("%") == 0
        assert len(store.get_all_memories()) == 2
        assert store.delete_memory_prefix("abc") == 1
        assert [fact["id"] for fact in store.get_all_memories()] == ["def-1"]


class TestSearchMemoryLimits:
    def test_max_results_applies_to_combined_facts_and_episodes(self, tmp_path, monkeypatch):
        from lilith_agent import memory

        store = MemoryStore(tmp_path / "test.sqlite")
        store.save_memories([
            {"id": "fact-1", "content": "alpha fact one"},
            {"id": "fact-2", "content": "alpha fact two"},
        ])
        store.add_episode("alpha task one", "alpha summary one", "success")
        store.add_episode("alpha task two", "alpha summary two", "success")
        monkeypatch.setattr(memory, "_store", store)

        result = memory.search_memory_store("alpha", max_results=3)

        result_count = result.count("[fact]") + result.count("[episode]")
        assert result_count == 3
