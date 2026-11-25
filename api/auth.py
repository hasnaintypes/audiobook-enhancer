"""Minimal API key auth (docs/06-Roadmap.md Phase 6's explicit "minimal
API key/token auth" scope -- no user accounts, no OAuth; this is still a
single-user project, see docs/01-PRD.md Open Assumption #5).
"""

from __future__ import annotations

import os

from fastapi import Header, HTTPException


def _allowed_keys() -> set[str]:
    raw = os.environ.get("API_KEYS", "")
    return {key.strip() for key in raw.split(",") if key.strip()}


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    allowed = _allowed_keys()
    if not allowed:
        raise HTTPException(status_code=500, detail="API_KEYS is not configured on the server")

    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")

    token = authorization.removeprefix("Bearer ").strip()
    if token not in allowed:
        raise HTTPException(status_code=401, detail="Invalid API key")
