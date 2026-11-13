"""Shared frame-based signal helpers — pure numpy, no I/O.

Extracted out of analysis/quality.py so both quality.py's detectors and
chunking/boundaries.py can share the same framing math instead of each
reimplementing it (or one reaching into the other's private internals).
"""

from __future__ import annotations

import numpy as np


def frame_signal(samples: np.ndarray, sample_rate: int, frame_ms: float) -> np.ndarray:
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    n_frames = len(samples) // frame_len
    if n_frames == 0:
        return samples.reshape(1, -1)
    trimmed = samples[: n_frames * frame_len]
    return trimmed.reshape(n_frames, frame_len)


def rms_db(frame: np.ndarray) -> float:
    rms = np.sqrt(np.mean(np.square(frame))) if frame.size else 0.0
    if rms <= 1e-9:
        return -120.0
    return float(20 * np.log10(rms))


def frame_times_s(n_frames: int, sample_rate: int, frame_ms: float) -> np.ndarray:
    """Start time, in seconds, of each frame produced by frame_signal() with
    the same sample_rate/frame_ms — frame i starts at i * frame_ms / 1000.
    """
    frame_duration_s = frame_ms / 1000.0
    return np.arange(n_frames) * frame_duration_s
