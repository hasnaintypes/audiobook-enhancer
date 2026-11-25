"""FastAPI app entrypoint (docs/06-Roadmap.md Phase 6).

Run locally with: uvicorn api.main:app --reload
Deploying this to a real host (Render, per gap.md Step 9's decision) is a
separate follow-up, not part of this increment.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router

app = FastAPI(title="Audiobook Enhancer API")

# Narration audio is private (docs/07-Decisions.md D007: "never serve this
# bucket publicly") -- restrict to the frontend's own origin, not "*".
_frontend_origin = os.environ.get("FRONTEND_ORIGIN")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_frontend_origin] if _frontend_origin else [],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
