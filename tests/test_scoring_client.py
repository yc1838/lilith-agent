from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from lilith_agent.scoring_client import ScoringApiClient


class ScoringApiClientTests(unittest.TestCase):
    def test_get_questions_returns_api_payload_on_success(self) -> None:
        response = Mock()
        response.json.return_value = [{"task_id": "t1", "question": "Q"}]
        response.raise_for_status.return_value = None

        session = Mock()
        session.get.return_value = response

        client = ScoringApiClient(api_url="https://example.com", session=session)

        payload = client.get_questions()

        self.assertEqual(payload, [{"task_id": "t1", "question": "Q"}])
        self.assertIsNone(client.last_warning)
        session.get.assert_called_once_with("https://example.com/questions", timeout=30)

    def test_get_questions_falls_back_to_dataset_on_429(self) -> None:
        response = Mock(status_code=429, headers={})
        error = requests.HTTPError("Too Many Requests", response=response)
        response.raise_for_status.side_effect = error

        session = Mock()
        session.get.return_value = response
        dataset_client = Mock()
        dataset_client.get_questions.return_value = [{"task_id": "fallback", "question": "Q2"}]

        client = ScoringApiClient(
            api_url="https://example.com",
            session=session,
            dataset_client=dataset_client,
        )

        payload = client.get_questions()

        self.assertEqual(payload, [{"task_id": "fallback", "question": "Q2"}])
        self.assertIn("429", client.last_warning or "")
        dataset_client.get_questions.assert_called_once_with()

    def test_download_file_falls_back_to_dataset_on_429(self) -> None:
        response = Mock(status_code=429, headers={})
        error = requests.HTTPError("Too Many Requests", response=response)
        response.raise_for_status.side_effect = error

        session = Mock()
        session.get.return_value = response
        dataset_client = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            fallback_path = Path(tmpdir) / "from-dataset.txt"
            fallback_path.write_text("ok")
            dataset_client.download_file.return_value = fallback_path

            client = ScoringApiClient(
                api_url="https://example.com",
                session=session,
                dataset_client=dataset_client,
            )

            payload = client.download_file("task-123", tmpdir)

        self.assertEqual(payload, fallback_path)
        self.assertIn("429", client.last_warning or "")
        dataset_client.download_file.assert_called_once_with("task-123", Path(tmpdir))

    def test_default_dataset_client_is_built_lazily(self) -> None:
        session = Mock()
        response = Mock(status_code=429, headers={})
        error = requests.HTTPError("Too Many Requests", response=response)
        response.raise_for_status.side_effect = error
        session.get.return_value = response

        dataset_instance = Mock()
        dataset_instance.get_questions.return_value = []

        with patch.object(ScoringApiClient, "_get_dataset_client", return_value=dataset_instance) as getter:
            client = ScoringApiClient(api_url="https://example.com", session=session)
            client.get_questions()

        getter.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
