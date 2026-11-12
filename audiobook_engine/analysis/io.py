"""Audio file loading — thin wrapper around soundfile, kept separate from
quality.py so the analysis math stays testable without real files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    """Load an audio file as a mono float32 waveform.

    Requires `soundfile` (and libsndfile) — run in Colab or a local dev env
    with audio deps installed, not assumed available in the base unit-test
    environment.
    """
    import soundfile as sf

    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples, sample_rate
