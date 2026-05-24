from __future__ import annotations

from unittest.mock import Mock, patch

from lilith_agent.gaia_dataset import GaiaDatasetClient


def _client_with_rows(rows):
    with patch("lilith_agent.gaia_dataset.load_dataset", return_value=rows):
        return GaiaDatasetClient(config="2023_all", split="test", level=None, token="hf-token")


def test_download_file_prints_when_task_row_is_missing(tmp_path):
    client = _client_with_rows([])

    with patch("builtins.print") as printed:
        path = client.download_file("missing-task", tmp_path)

    assert path is None
    printed.assert_any_call(
        "[gaia_dataset] file row missing task=missing-task split=test config=2023_all",
        flush=True,
    )


def test_download_file_prints_hf_download_status_on_failure(tmp_path):
    client = _client_with_rows([
        {
            "task_id": "task-img",
            "Question": "Q",
            "Level": "1",
            "file_name": "task-img.png",
            "Final answer": "",
        }
    ])
    response = Mock(status_code=404)

    with patch("lilith_agent.gaia_dataset.requests.get", return_value=response) as get, patch("builtins.print") as printed:
        path = client.download_file("task-img", tmp_path)

    assert path is None
    assert get.call_args.args[0] == "https://huggingface.co/datasets/gaia-benchmark/GAIA/resolve/main/2023/test/task-img.png"
    printed.assert_any_call(
        "[gaia_dataset] hf download failed task=task-img status=404 url=https://huggingface.co/datasets/gaia-benchmark/GAIA/resolve/main/2023/test/task-img.png",
        flush=True,
    )
