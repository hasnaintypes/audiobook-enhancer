"""Minimal per-chunk job runner loop (docs/06-Roadmap.md Phase 4).

Consumes the chunk-planning mechanism built in chunking/boundaries.py and
chunking/manifest.py: plans a long file into chunks, processes whichever
chunks aren't done yet through the existing Tier 0/1 pipeline, uploads each
chunk's output, and persists the manifest after EVERY successful chunk. That
re-upload-after-each-success is the actual resumability guarantee a crash
between chunks relies on — it must never be batched or deferred.

Pure logic (chunk_filename, slice_chunk_samples, resolve_manifest) is kept
separate from run_job's real I/O (ffmpeg subprocesses, network storage, local
file writes) the same way chunking/manifest.py is split from
jobstore/storage.py, and postprocess/dsp.py / enhancement/rnnoise.py split
their pure command-builders from their run_* execution functions. run_job
itself is excluded from the unit suite like those run_* functions already
are — run it in Colab, not locally without ffmpeg/network access.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from audiobook_engine.analysis.quality import analyze_quality
from audiobook_engine.analysis.strategy import (
    EnhancementTier,
    resolve_effective_tier,
    select_strategy,
)
from audiobook_engine.chunking.boundaries import ChunkBoundary, ChunkingConfig, plan_chunks
from audiobook_engine.chunking.manifest import (
    ChunkManifest,
    mark_chunk_complete,
    pending_chunk_indices,
)


def chunk_filename(index: int) -> str:
    """Zero-padded so lexicographic and numeric chunk order match — matters
    once stitching reads these back in order.
    """
    return f"chunk_{index:04d}.wav"


def slice_chunk_samples(
    samples: np.ndarray, sample_rate: int, boundary: ChunkBoundary
) -> np.ndarray:
    """Pure numpy slice of a chunk's samples from a planned boundary."""
    start_idx = int(round(boundary.start_s * sample_rate))
    end_idx = int(round(boundary.end_s * sample_rate))
    return samples[start_idx:end_idx]


def resolve_manifest(
    job_id: str, total_chunks: int, existing: ChunkManifest | None
) -> ChunkManifest:
    """Decide which manifest to resume from (or start fresh).

    Chunk planning is deterministic given the same audio + ChunkingConfig, so
    an existing manifest's total_chunks should always match a fresh replan.
    A mismatch means the plan changed between runs (e.g. a config edit) —
    resuming against stale chunk indices would silently corrupt the job, so
    this refuses rather than guesses.
    """
    if existing is None:
        return ChunkManifest(job_id=job_id, total_chunks=total_chunks, completed_indices=frozenset())

    if existing.job_id != job_id:
        raise ValueError(
            f"Manifest job_id mismatch: expected {job_id!r}, found {existing.job_id!r}"
        )
    if existing.total_chunks != total_chunks:
        raise ValueError(
            f"Chunk plan changed since last run for job {job_id!r}: existing manifest "
            f"has {existing.total_chunks} chunks, replanned to {total_chunks} — refusing "
            "to resume against a stale chunk plan"
        )
    return existing


def run_job(
    job_id: str,
    input_audio_path: Path,
    work_dir: Path,
    chunking_config: ChunkingConfig = ChunkingConfig(),
    model_path: Path | None = None,
    force_tier2: bool = False,
    tier2_venv_python: Path | None = None,
    max_tier: EnhancementTier | None = None,
) -> ChunkManifest:
    """Run (or resume) a chunked enhancement job end-to-end.

    force_tier2=True routes every chunk through Tier 2 (Resemble Enhance)
    regardless of what select_strategy decides — this is the explicit,
    human-driven opt-in Tier 2 requires (analysis/strategy.py never assigns
    Tier 2 on its own). Resemble Enhance only runs inside its own isolated
    legacy venv (enhancement/resemble.py's ensure_resemble_enhance_env), so
    force_tier2=True requires tier2_venv_python to already point at that
    venv's python binary.

    max_tier caps (never escalates) the tier select_strategy would
    otherwise choose -- see analysis/strategy.py's resolve_effective_tier,
    which this delegates to. Used by runner/poller.py to enforce a
    submitted job's preset (config/presets.yaml's tier_max).

    Requires ffmpeg on PATH and network access to Neon Object Storage — run
    in Colab, not as part of the unit test suite.
    """
    import dataclasses

    if force_tier2 and tier2_venv_python is None:
        raise ValueError("force_tier2=True requires tier2_venv_python")

    import soundfile as sf
    from botocore.exceptions import ClientError

    from audiobook_engine.analysis.io import load_mono
    from audiobook_engine.chunking.boundaries import estimate_adaptive_silence_threshold_db
    from audiobook_engine.enhancement.resemble import run_tier2_processing
    from audiobook_engine.enhancement.rnnoise import download_default_model, run_tier1_processing
    from audiobook_engine.jobstore.storage import (
        download_manifest,
        object_key,
        upload_file,
        upload_manifest,
    )
    from audiobook_engine.postprocess.dsp import run_tier0_processing

    work_dir.mkdir(parents=True, exist_ok=True)

    samples, sample_rate = load_mono(input_audio_path)

    # A single global silence_rms_threshold_db can't work across recordings
    # with different ambient noise floors (confirmed on real samples — see
    # gap.md evidence, 2026-11-20: a lecture recording's higher noise floor
    # made the old fixed default find almost no silence gaps at all). Derive
    # it from this sample's own frame-RMS distribution instead.
    adaptive_threshold_db = estimate_adaptive_silence_threshold_db(
        samples, sample_rate, frame_ms=chunking_config.frame_ms
    )
    effective_chunking_config = dataclasses.replace(
        chunking_config, silence_rms_threshold_db=adaptive_threshold_db
    )
    boundaries = plan_chunks(samples, sample_rate, effective_chunking_config)

    try:
        existing = download_manifest(job_id)
    except ClientError:
        existing = None

    manifest = resolve_manifest(job_id, len(boundaries), existing)
    if existing is None:
        upload_manifest(manifest)

    resolved_model_path = model_path

    for idx in pending_chunk_indices(manifest):
        boundary = boundaries[idx]
        chunk_samples = slice_chunk_samples(samples, sample_rate, boundary)

        input_chunk_path = work_dir / f"input_{chunk_filename(idx)}"
        sf.write(str(input_chunk_path), chunk_samples, sample_rate)

        report = analyze_quality(chunk_samples, sample_rate)
        decision = select_strategy(report)

        processed_path = work_dir / chunk_filename(idx)
        effective_tier = resolve_effective_tier(decision.tier, force_tier2, max_tier)

        if effective_tier is EnhancementTier.TIER_2_GPU_HEAVY:
            run_tier2_processing(
                input_chunk_path, processed_path, sample_rate, venv_python=tier2_venv_python
            )
        elif effective_tier is EnhancementTier.TIER_1_LIGHT_AI:
            if resolved_model_path is None:
                resolved_model_path = download_default_model(work_dir / "models" / "std.rnnn")
            run_tier1_processing(input_chunk_path, processed_path, resolved_model_path, sample_rate)
        else:
            run_tier0_processing(input_chunk_path, processed_path, decision, sample_rate)

        upload_file(processed_path, object_key(job_id, "chunks", chunk_filename(idx)))

        manifest = mark_chunk_complete(manifest, idx)
        upload_manifest(manifest)

    return manifest
