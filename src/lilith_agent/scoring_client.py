from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


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
            fallback = self._fallback_dataset_client_for(
                exc,
                action=f"download file for {task_id}",
                missing_is_fallback=True,
            )
            if fallback is None:
                print(
                    f"[scoring] file download failed without fallback task={task_id} error={exc.__class__.__name__}",
                    flush=True,
                )
                return None
            path = fallback.download_file(task_id, dest_dir)
            print(f"[scoring] fallback file result task={task_id} path={path}", flush=True)
            return path

        filename = task_id
        cd = response.headers.get("content-disposition", "")
        if "filename=" in cd:
            filename = cd.split("filename=")[-1].strip().strip('"')
        content_type = response.headers.get("content-type", "")
        if self._is_invalid_file_payload(response.content, content_type):
            print(
                f"[scoring] invalid file payload task={task_id} status={response.status_code} "
                f"bytes={len(response.content)} content_type={content_type!r}; trying dataset fallback",
                flush=True,
            )
            try:
                fallback = self._get_dataset_client()
            except Exception as exc:
                print(
                    f"[scoring] dataset fallback unavailable task={task_id} type={type(exc).__name__} error={exc}",
                    flush=True,
                )
                return None
            path = fallback.download_file(task_id, dest_dir)
            print(f"[scoring] fallback file result task={task_id} path={path}", flush=True)
            return path
        out = dest_dir / filename
        out.write_bytes(response.content)
        self.last_warning = None
        print(
            f"[scoring] file downloaded task={task_id} path={out} bytes={len(response.content)} "
            f"content_type={response.headers.get('content-type', '')!r}",
            flush=True,
        )
        return out

    def _fallback_dataset_client_for(
        self,
        exc: requests.RequestException,
        *,
        action: str,
        missing_is_fallback: bool = False,
    ) -> Any | None:
        if not self._should_use_dataset_fallback(exc, missing_is_fallback=missing_is_fallback):
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
        print(self.last_warning, flush=True)
        log.warning(self.last_warning)
        return dataset_client

    def _get_dataset_client(self) -> Any:
        if self._dataset_client is None:
            from lilith_agent.gaia_dataset import GaiaDatasetClient

            token = os.getenv("HF_TOKEN") or os.getenv("GAIA_HUGGINGFACE_API_KEY")
            self._dataset_client = GaiaDatasetClient(
                config=os.getenv("GAIA_DATASET_CONFIG", "2023_all"),
                split=os.getenv("GAIA_DATASET_SPLIT", "validation"),
                level=None,
                token=token,
            )
        return self._dataset_client

    @staticmethod
    def _should_use_dataset_fallback(exc: requests.RequestException, *, missing_is_fallback: bool = False) -> bool:
        if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
            return True
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if missing_is_fallback and status_code == 404:
            return True
        return status_code in {429, 502, 503, 504}

    @staticmethod
    def _is_invalid_file_payload(content: bytes, content_type: str) -> bool:
        return "application/json" in content_type.lower() and len(content) < 4096
