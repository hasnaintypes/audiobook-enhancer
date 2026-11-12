"""Audio quality analysis — pure numpy, operates on an in-memory waveform.

Deliberately has no yt-dlp/ffmpeg/Colab dependency so it's fully covered by
tests/unit with synthetic signals. Reading a real file into a numpy array
is the caller's job (see analysis/io.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Thresholds below are starting points, not calibrated against real audiobook
# samples yet — see docs/01-PRD.md Open Assumptions. Expect to tune these
# once Phase 1's real-sample experiment runs.
_CLIPPING_PEAK_THRESHOLD = 0.99
_CLIPPING_MIN_FRACTION = 0.0001
_SILENCE_RMS_THRESHOLD_DB = -45.0
_FRAME_MS = 20
_LOUDNESS_WINDOW_S = 3.0
_NOISE_FLOOR_PERCENTILE = 50.0
_INCONSISTENT_LOUDNESS_STD_DB = 6.0


@dataclass(frozen=True)
class QualityReport:
    peak_amplitude: float
    rms_db: float
    clipping_fraction: float
    is_clipped: bool
    silence_fraction: float
    speech_fraction: float
    duration_seconds: float
    noise_floor_db: float
    loudness_std_db: float
    is_loudness_inconsistent: bool


def _frame_signal(samples: np.ndarray, sample_rate: int, frame_ms: int) -> np.ndarray:
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    n_frames = len(samples) // frame_len
    if n_frames == 0:
        return samples.reshape(1, -1)
    trimmed = samples[: n_frames * frame_len]
    return trimmed.reshape(n_frames, frame_len)


def _rms_db(frame: np.ndarray) -> float:
    rms = np.sqrt(np.mean(np.square(frame))) if frame.size else 0.0
    if rms <= 1e-9:
        return -120.0
    return float(20 * np.log10(rms))


def _estimate_noise_floor_db(frame_db: np.ndarray, silence_mask: np.ndarray) -> float:
    """Median loudness of the quietest frames — a proxy for hiss/hum level,
    distinct from true digital silence (which frame_db already excludes via
    the -120dB floor only on fully-zero frames).
    """
    quiet_frames = frame_db[silence_mask]
    if quiet_frames.size == 0:
        # No silence detected at all (e.g. continuous loud narration) — fall
        # back to the overall 10th percentile as the best available proxy.
        quiet_frames = frame_db
        return float(np.percentile(quiet_frames, 10))
    return float(np.percentile(quiet_frames, _NOISE_FLOOR_PERCENTILE))


def _loudness_consistency(
    samples: np.ndarray, sample_rate: int, speech_mask_frame_ms: int
) -> tuple[float, bool]:
    """Std deviation of RMS loudness across multi-second windows, restricted
    to windows that aren't mostly silence — flags volume jumps between
    sections (e.g. a louder re-recorded chapter) rather than silence/speech
    alternation, which would otherwise dominate the variance.
    """
    windows = _frame_signal(samples, sample_rate, int(_LOUDNESS_WINDOW_S * 1000))
    window_db = np.array([_rms_db(w) for w in windows])
    non_silent = window_db[window_db > _SILENCE_RMS_THRESHOLD_DB]
    if non_silent.size < 2:
        return 0.0, False
    std_db = float(np.std(non_silent))
    return std_db, std_db > _INCONSISTENT_LOUDNESS_STD_DB


def analyze_quality(samples: np.ndarray, sample_rate: int) -> QualityReport:
    """Compute basic quality signals for a mono float32 waveform in [-1, 1].

    This is Tier-0 analysis only (cheap, no AI) — used to decide whether a
    recording needs anything beyond DSP. See docs/03-Tech-Stack.md for the
    tiering rationale.
    """
    if samples.ndim != 1:
        raise ValueError("Expected a mono (1-D) waveform; downmix before calling")
    if samples.size == 0:
        raise ValueError("Empty waveform")

    peak = float(np.max(np.abs(samples)))
    clipped_samples = np.sum(np.abs(samples) >= _CLIPPING_PEAK_THRESHOLD)
    clipping_fraction = float(clipped_samples) / samples.size
    is_clipped = clipping_fraction > _CLIPPING_MIN_FRACTION

    overall_rms_db = _rms_db(samples)

    frames = _frame_signal(samples, sample_rate, _FRAME_MS)
    frame_db = np.array([_rms_db(frame) for frame in frames])
    silence_mask = frame_db < _SILENCE_RMS_THRESHOLD_DB
    silence_fraction = float(np.mean(silence_mask))
    speech_fraction = 1.0 - silence_fraction

    noise_floor_db = _estimate_noise_floor_db(frame_db, silence_mask)
    loudness_std_db, is_loudness_inconsistent = _loudness_consistency(
        samples, sample_rate, _FRAME_MS
    )

    return QualityReport(
        peak_amplitude=peak,
        rms_db=overall_rms_db,
        clipping_fraction=clipping_fraction,
        is_clipped=is_clipped,
        silence_fraction=silence_fraction,
        speech_fraction=speech_fraction,
        duration_seconds=samples.size / sample_rate,
        noise_floor_db=noise_floor_db,
        loudness_std_db=loudness_std_db,
        is_loudness_inconsistent=is_loudness_inconsistent,
    )
