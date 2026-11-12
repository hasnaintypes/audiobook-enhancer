"""Source URL validation — kept separate from extraction so it can be unit
tested without network access or yt-dlp.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class SourceKind(Enum):
    VIDEO = "video"
    PLAYLIST = "playlist"
    UNSUPPORTED = "unsupported"


_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com"}

_VIDEO_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{11}$")


@dataclass(frozen=True)
class SourceValidationResult:
    kind: SourceKind
    reason: str | None = None

    @property
    def is_supported(self) -> bool:
        return self.kind is not SourceKind.UNSUPPORTED


def classify_source(url: str) -> SourceValidationResult:
    """Classify a source URL without making any network calls.

    This only rules out obviously unsupported input (wrong host, malformed
    URL). It does NOT verify the video/playlist actually exists or is
    accessible — that requires a real yt-dlp call and belongs in extraction.
    """
    url = url.strip()
    if not url:
        return SourceValidationResult(SourceKind.UNSUPPORTED, "Empty URL")

    try:
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(url)
    except ValueError:
        return SourceValidationResult(SourceKind.UNSUPPORTED, "Malformed URL")

    host = parsed.netloc.lower()
    if host not in _YOUTUBE_HOSTS:
        return SourceValidationResult(
            SourceKind.UNSUPPORTED, f"Unsupported host: {host or '(none)'}"
        )

    query = parse_qs(parsed.query)
    if "list" in query:
        return SourceValidationResult(SourceKind.PLAYLIST)

    if host == "youtu.be":
        video_id = parsed.path.lstrip("/")
        if _VIDEO_ID_RE.match(video_id):
            return SourceValidationResult(SourceKind.VIDEO)
        return SourceValidationResult(SourceKind.UNSUPPORTED, "Invalid youtu.be path")

    if parsed.path == "/watch" and "v" in query:
        video_id = query["v"][0]
        if _VIDEO_ID_RE.match(video_id):
            return SourceValidationResult(SourceKind.VIDEO)
        return SourceValidationResult(SourceKind.UNSUPPORTED, "Invalid video id")

    return SourceValidationResult(SourceKind.UNSUPPORTED, "Unrecognized YouTube URL shape")
