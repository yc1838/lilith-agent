from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

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


def _ffmpeg_path() -> str:
    """Locate an ffmpeg binary. Prefer system ffmpeg, fall back to imageio-ffmpeg's bundled copy."""
    which = shutil.which("ffmpeg")
    if which:
        return which
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError(f"ffmpeg not found and imageio-ffmpeg unavailable: {e}")


def youtube_frame_at(url: str, timestamp_seconds: float) -> str:
    """Download a YouTube video and extract the frame at `timestamp_seconds` as a local PNG.

    Returns the local PNG path on success (feed this to `inspect_visual_content`),
    or an "ERROR: ..." string on failure.
    """
    try:
        import yt_dlp
    except ImportError as e:
        return f"ERROR: yt-dlp not installed ({e})"

    try:
        ffmpeg = _ffmpeg_path()
    except Exception as e:
        return f"ERROR: {e}"

    tmp_dir = Path(tempfile.mkdtemp(prefix="ytframe_"))
    try:
        vid = extract_video_id(url)
        video_out = tmp_dir / f"{vid}.mp4"
        # Pre-merged single-file format -> no ffmpeg merge step needed.
        ydl_opts = {
            "format": "best[ext=mp4]/best",
            "outtmpl": str(video_out),
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if not video_out.exists():
            matches = list(tmp_dir.glob(f"{vid}.*"))
            if not matches:
                return "ERROR: yt-dlp produced no file"
            video_out = matches[0]

        frame_out = tmp_dir / f"{vid}_t{int(timestamp_seconds)}.png"
        cmd = [
            ffmpeg, "-y", "-ss", str(timestamp_seconds),
            "-i", str(video_out), "-frames:v", "1",
            "-q:v", "2", str(frame_out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode != 0 or not frame_out.exists():
            return f"ERROR: ffmpeg failed: {proc.stderr[-500:]}"

        try:
            video_out.unlink(missing_ok=True)
        except Exception:
            pass
        return str(frame_out)
    except Exception as e:
        return f"ERROR: {e}"
