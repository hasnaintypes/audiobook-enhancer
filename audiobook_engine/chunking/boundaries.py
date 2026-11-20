"""Chunk-boundary planning for long-form audio — pure numpy, no I/O.

Hybrid VAD + max-duration chunking (docs/07-Decisions.md D005): prefer
cutting at a silence gap, never exceed a max chunk duration, never cut
mid-word if a nearby silence gap exists. Pure fixed-duration chunking is
explicitly rejected because it risks splitting mid-word/mid-phrase, which
conflicts with this project's top priority of preserving the narrator's
pacing (see postprocess/dsp.py's docstring on never trimming internal
pauses).

All ChunkingConfig defaults below are PROVISIONAL placeholders, not
calibrated against real audiobook samples — D005's status is "proposed,
pending experiment" (docs/06-Roadmap.md Phase 4 task 1). Expect to tune
these once that real-sample experiment runs on Colab, the same way
analysis/quality.py's thresholds were (eventually) validated.

Extends analysis/quality.py's energy-gate VAD (frame_signal/rms_db) with
debouncing: a silence gap must span a contiguous run of frames under
threshold for at least min_silence_duration_s, not a single isolated
frame — quality.py's analyze_quality() has no such debouncing since it
only needs an aggregate silence_fraction, not boundary-grade gaps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from audiobook_engine.analysis.framing import frame_signal, frame_times_s, rms_db

_DEFAULT_MAX_CHUNK_DURATION_S = 600.0
_DEFAULT_SILENCE_RMS_THRESHOLD_DB = -45.0
_DEFAULT_MIN_SILENCE_DURATION_S = 0.5
_DEFAULT_FRAME_MS = 20.0


@dataclass(frozen=True)
class ChunkingConfig:
    max_chunk_duration_s: float = _DEFAULT_MAX_CHUNK_DURATION_S
    silence_rms_threshold_db: float = _DEFAULT_SILENCE_RMS_THRESHOLD_DB
    min_silence_duration_s: float = _DEFAULT_MIN_SILENCE_DURATION_S
    frame_ms: float = _DEFAULT_FRAME_MS


@dataclass(frozen=True)
class ChunkBoundary:
    start_s: float
    end_s: float
    cut_reason: str  # "silence_gap" | "max_duration_fallback" | "end_of_audio"


def estimate_adaptive_silence_threshold_db(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: float = _DEFAULT_FRAME_MS,
    percentile: float = 15.0,
) -> float:
    """Estimate a per-sample silence threshold from this sample's own
    frame-level RMS distribution, instead of assuming one fixed dB value
    works across recordings with different ambient noise floors.

    Confirmed necessary on real samples (gap.md evidence, 2026-11-20): a
    70-minute lecture recording's ambient noise floor sits far above a quiet
    studio narration's, so a single global threshold (e.g. -45dB) found
    almost no silence gaps at all on the lecture — 7 of 8 chunk boundaries
    fell back to a hard mid-audio cut. This computes directly from the raw
    frame-RMS distribution with NO pre-filtering by any fixed constant
    (unlike analysis/quality.py's noise_floor_db, which filters frames by
    its own fixed -45dB threshold first — circular for this exact problem).
    """
    frames = frame_signal(samples, sample_rate, frame_ms)
    frame_db = np.array([rms_db(f) for f in frames])
    return float(np.percentile(frame_db, percentile))


def find_silence_gaps(
    samples: np.ndarray, sample_rate: int, config: ChunkingConfig
) -> list[tuple[float, float]]:
    """Return (start_s, end_s) of debounced silence gaps.

    A gap must span a contiguous run of frames all under
    config.silence_rms_threshold_db for at least config.min_silence_duration_s
    — a single isolated quiet frame doesn't count (avoids treating a brief
    in-word dip as a splice point).
    """
    frames = frame_signal(samples, sample_rate, config.frame_ms)
    frame_db = np.array([rms_db(f) for f in frames])
    times = frame_times_s(len(frames), sample_rate, config.frame_ms)
    frame_duration_s = config.frame_ms / 1000.0

    silence_mask = frame_db < config.silence_rms_threshold_db

    gaps: list[tuple[float, float]] = []
    run_start_idx: int | None = None
    for i, is_silent in enumerate(silence_mask):
        if is_silent and run_start_idx is None:
            run_start_idx = i
        elif not is_silent and run_start_idx is not None:
            gap_start = float(times[run_start_idx])
            gap_end = float(times[i - 1]) + frame_duration_s
            if gap_end - gap_start >= config.min_silence_duration_s:
                gaps.append((gap_start, gap_end))
            run_start_idx = None
    if run_start_idx is not None:
        gap_start = float(times[run_start_idx])
        gap_end = float(times[-1]) + frame_duration_s
        if gap_end - gap_start >= config.min_silence_duration_s:
            gaps.append((gap_start, gap_end))

    return gaps


def plan_chunks(
    samples: np.ndarray, sample_rate: int, config: ChunkingConfig
) -> list[ChunkBoundary]:
    """Plan chunk boundaries covering the full audio, in order, contiguous.

    Raises ValueError on an empty waveform. A non-empty waveform shorter
    than max_chunk_duration_s (with no cut needed at all) returns a single
    whole-file boundary with cut_reason="end_of_audio".
    """
    if samples.size == 0:
        raise ValueError("Empty waveform")

    duration_s = samples.size / sample_rate
    if duration_s <= config.max_chunk_duration_s:
        return [ChunkBoundary(start_s=0.0, end_s=duration_s, cut_reason="end_of_audio")]

    gaps = find_silence_gaps(samples, sample_rate, config)

    boundaries: list[ChunkBoundary] = []
    cursor = 0.0
    while cursor < duration_s:
        cap = cursor + config.max_chunk_duration_s
        if cap >= duration_s:
            boundaries.append(
                ChunkBoundary(start_s=cursor, end_s=duration_s, cut_reason="end_of_audio")
            )
            break

        # Prefer the gap whose midpoint (the actual cut point) is closest to
        # (but not past) the cap, among gaps that start after cursor — the
        # gap itself may extend slightly past the cap as long as the cut
        # point doesn't.
        candidate_gaps = [
            g for g in gaps if g[0] > cursor and (g[0] + g[1]) / 2.0 <= cap
        ]
        if candidate_gaps:
            gap_start, gap_end = max(candidate_gaps, key=lambda g: (g[0] + g[1]) / 2.0)
            cut_at = (gap_start + gap_end) / 2.0
            boundaries.append(
                ChunkBoundary(start_s=cursor, end_s=cut_at, cut_reason="silence_gap")
            )
            cursor = cut_at
        else:
            boundaries.append(
                ChunkBoundary(start_s=cursor, end_s=cap, cut_reason="max_duration_fallback")
            )
            cursor = cap

    return boundaries
