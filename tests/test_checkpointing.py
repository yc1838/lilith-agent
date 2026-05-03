from __future__ import annotations

import importlib
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from lilith_agent.checkpointing import build_checkpointer


class CheckpointingTests(unittest.TestCase):
    def test_build_checkpointer_uses_sqlite_when_available(self) -> None:
        tmp_path = Path(self._testMethodName)

        class FakeSqliteSaver:
            def __init__(self, conn) -> None:
                self.conn = conn

        def fake_import_module(name: str):
            if name == "langgraph.checkpoint.sqlite":
                return types.SimpleNamespace(SqliteSaver=FakeSqliteSaver)
            raise AssertionError(f"unexpected import: {name}")

        with patch.object(importlib, "import_module", side_effect=fake_import_module):
            saver = build_checkpointer(tmp_path)

        self.assertIsInstance(saver, FakeSqliteSaver)
        self.assertTrue((tmp_path / "threads.sqlite").exists())

    def test_build_checkpointer_falls_back_to_in_memory_when_sqlite_package_missing(self) -> None:
        tmp_path = Path(self._testMethodName)

        class FakeInMemorySaver:
            pass

        def fake_import_module(name: str):
            if name == "langgraph.checkpoint.sqlite":
                raise ModuleNotFoundError("No module named 'langgraph.checkpoint.sqlite'")
            if name == "langgraph.checkpoint.memory":
                return types.SimpleNamespace(InMemorySaver=FakeInMemorySaver)
            raise AssertionError(f"unexpected import: {name}")

        with patch.object(importlib, "import_module", side_effect=fake_import_module):
            with self.assertLogs("lilith_agent.checkpointing", level="WARNING") as captured:
                saver = build_checkpointer(tmp_path)

        self.assertIsInstance(saver, FakeInMemorySaver)
        self.assertIn("langgraph-checkpoint-sqlite missing", "\n".join(captured.output))

    def test_requirements_include_langgraph_checkpoint_sqlite(self) -> None:
        requirements = Path("requirements.txt").read_text().splitlines()
        self.assertIn("langgraph-checkpoint-sqlite", requirements)


if __name__ == "__main__":
    unittest.main()
