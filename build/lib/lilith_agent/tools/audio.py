from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=2)
def _get_model(model_size: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe_audio(path: str, model_size: str = "base") -> str:
    model = _get_model(model_size)
    segments, _info = model.transcribe(path)
    return " ".join(seg.text.strip() for seg in segments).strip()
