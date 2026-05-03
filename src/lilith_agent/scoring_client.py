from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from lilith_agent.gaia_dataset import GaiaDatasetClient


class ScoringApiClient:
    """Client for the GAIA scoring Space with dataset fallback on transient failures."""

    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        session: requests.Session | None = None,
        dataset_client: Any | None = None,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.session = session or requests.Session()
        self._dataset_client = dataset_client
        self.last_warning: str | None = None

    def get_questions(self) -> list[dict]:
        try:
            response = self.session.get(f"{self.api_url}/questions", timeout=30)
            response.raise_for_status()
            self.last_warning = None
            return response.json()
        except requests.RequestException as exc:
            fallback = self._fallback_dataset_client_for(exc, action="fetch questions")
            if fallback is None:
                raise
            return fallback.get_questions()

    def download_file(self, task_id: str, dest_dir: str | Path) -> Path | None:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            response = self.session.get(f"{self.api_url}/files/{task_id}", timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:
            fallback = self._fallback_dataset_client_for(exc, action=f"download file for {task_id}")
            if fallback is None:
                return None
            return fallback.download_file(task_id, dest_dir)

        filename = task_id
        cd = response.headers.get("content-disposition", "")
        if "filename=" in cd:
            filename = cd.split("filename=")[-1].strip().strip('"')
        out = dest_dir / filename
        out.write_bytes(response.content)
        self.last_warning = None
        return out

    def _fallback_dataset_client_for(
        self,
        exc: requests.RequestException,
        *,
        action: str,
    ) -> Any | None:
        if not self._should_use_dataset_fallback(exc):
            return None
        try:
            dataset_client = self._get_dataset_client()
        except Exception:
            return None

        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        detail = f"status={status_code}" if status_code is not None else exc.__class__.__name__
        self.last_warning = (
            f"Scoring API unavailable while trying to {action} ({detail}); "
            f"falling back to GAIA dataset."
        )
        log.warning(self.last_warning)
        return dataset_client

    def _get_dataset_client(self) -> Any:
        if self._dataset_client is None:
            from lilith_agent.gaia_dataset import GaiaDatasetClient

            token = os.getenv("HF_TOKEN") or os.getenv("GAIA_HUGGINGFACE_API_KEY")
            self._dataset_client = GaiaDatasetClient(
                config=os.getenv("GAIA_DATASET_CONFIG", "2023_all"),
                split=os.getenv("GAIA_DATASET_SPLIT", "test"),
                level=None,
                token=token,
            )
        return self._dataset_client

    @staticmethod
    def _should_use_dataset_fallback(exc: requests.RequestException) -> bool:
        if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
            return True
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        return status_code in {429, 502, 503, 504}
