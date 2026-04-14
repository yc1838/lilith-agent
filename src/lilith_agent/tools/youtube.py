from __future__ import annotations

import re

from youtube_transcript_api import YouTubeTranscriptApi

_PATTERNS = [
    re.compile(r"youtube\.com/watch\?v=([A-Za-z0-9_-]{11})"),
    re.compile(r"youtu\.be/([A-Za-z0-9_-]{11})"),
    re.compile(r"youtube\.com/embed/([A-Za-z0-9_-]{11})"),
]


def extract_video_id(url: str) -> str:
    for pat in _PATTERNS:
        m = pat.search(url)
        if m:
            return m.group(1)
    raise ValueError(f"Could not extract video id from {url}")


def youtube_transcript(url: str) -> str:
    try:
        vid = extract_video_id(url)
        fetched = YouTubeTranscriptApi().fetch(vid)
        return " ".join(
            (seg["text"] if isinstance(seg, dict) else seg.text) for seg in fetched
        )
    except Exception as e:
        return f"Video Transcript Error: {e}"
