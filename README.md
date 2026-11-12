# Audiobook Enhancer

AI-assisted enhancement pipeline for YouTube-sourced Urdu audiobooks: extracts narration, analyzes quality, applies tiered DSP/AI enhancement while preserving the narrator's voice, and optionally mixes in subtle background music.

Full planning docs (PRD, architecture, tech stack, roadmap, decisions) live in the Obsidian vault at `~/Vaults/Main/Projects/Audiobook Enhancer/` — this repo does not duplicate them. Once the vault doc set is reviewed and approved, docs will be linked in via `project-doc-linking`.

## Status

Scaffolding only (Phase 0). No pipeline code has been implemented yet. See `06-Roadmap.md` in the vault for the phased plan.

## Structure

- `audiobook_engine/` — pure processing package (ingestion, analysis, enhancement, postprocess, music, chunking, quality, config, jobstore). No Colab/FastAPI imports allowed here.
- `api/` — FastAPI job-lifecycle layer (Phase 6).
- `frontend/` — Next.js app (Phase 7).
- `notebooks/` — Colab bootstrap notebook only; never edit pipeline code here.
- `tests/unit/` — fast, no-GPU tests, run in CI.
- `tests/gpu/` — GPU-dependent tests, Colab-only.
- `config/` — pinned dependency lockfile and enhancement presets.
