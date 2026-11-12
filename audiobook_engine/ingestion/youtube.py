"""yt-dlp wrapper for audio extraction.

This module performs real network I/O and shells out to FFmpeg via yt-dlp's
postprocessor — per the architecture, this is "computationally/IO expensive"
and is meant to run in the Colab execution environment, not as a unit test.
Keep this module's logic thin and push anything unit-testable (URL parsing,
error classification) into separate pure functions so most of it CAN be
tested without a real download.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ExtractionErrorKind(Enum):
    PRIVATE = "private"
    UNAVAILABLE = "unavailable"
    AGE_RESTRICTED = "age_restricted"
    GEO_RESTRICTED = "geo_restricted"
    UNKNOWN = "unknown"


class ExtractionError(Exception):
    def __init__(self, kind: ExtractionErrorKind, message: str):
        self.kind = kind
        super().__init__(message)


def classify_ytdlp_error(message: str) -> ExtractionErrorKind:
    """Map a yt-dlp DownloadError message to a coarse error kind.

    Pure string matching — kept separate from the actual download call so
    it's unit-testable against real yt-dlp error strings without a network
    call.
    """
    lowered = message.lower()
    if "private video" in lowered:
        return ExtractionErrorKind.PRIVATE
    if "video unavailable" in lowered or "has been removed" in lowered:
        return ExtractionErrorKind.UNAVAILABLE
    if "sign in to confirm your age" in lowered or "age-restricted" in lowered:
        return ExtractionErrorKind.AGE_RESTRICTED
    if "not available in your country" in lowered or "geo" in lowered:
        return ExtractionErrorKind.GEO_RESTRICTED
    return ExtractionErrorKind.UNKNOWN


@dataclass(frozen=True)
class ExtractedAudio:
    audio_path: Path
    title: str
    duration_seconds: float
    video_id: str
    source_url: str


def extract_audio(url: str, output_dir: Path) -> ExtractedAudio:
    """Download the best available audio stream for a single video and
    convert it to WAV via FFmpeg (through yt-dlp's postprocessor chain).

    Requires network access and a working ffmpeg on PATH — run this in
    Colab, not as part of the unit test suite.
    """
    import yt_dlp

    output_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(output_dir / "%(id)s.%(ext)s")

    ydl_opts: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as exc:
        kind = classify_ytdlp_error(str(exc))
        raise ExtractionError(kind, str(exc)) from exc

    video_id = info["id"]
    audio_path = output_dir / f"{video_id}.wav"
    if not audio_path.exists():
        raise ExtractionError(
            ExtractionErrorKind.UNKNOWN,
            f"Expected output file not found: {audio_path}",
        )

    return ExtractedAudio(
        audio_path=audio_path,
        title=info.get("title", "unknown"),
        duration_seconds=float(info.get("duration") or 0.0),
        video_id=video_id,
        source_url=url,
    )
