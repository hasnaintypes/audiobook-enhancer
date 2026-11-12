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

# Spectral detectors (background music / reverb / lossy-compression cutoff)
# need wider frames than the 20ms loudness frames above for usable frequency
# resolution. Also unvalidated against real samples — see module docstring.
_SPECTRAL_FRAME_MS = 40.0

# "Background music under narration" shows up as tonal (low spectral-
# flatness) energy in frames that are quiet enough to be a narration pause
# but not quiet enough to be true digital silence — pure hiss/hum in that
# same quiet band is broadband (flatness near 1), a music bed is peaky
# (flatness well below 1).
_MUSIC_QUIET_RMS_LOW_DB = _SILENCE_RMS_THRESHOLD_DB
_MUSIC_QUIET_RMS_HIGH_DB = -25.0
_MUSIC_FLATNESS_THRESHOLD = 0.35
_MUSIC_MIN_QUIET_FRAMES = 5

# Reverb/echo: after a sharp energy drop ("offset"), a dry recording falls
# straight to the noise floor; a reverberant one keeps audible energy above
# the floor for a while longer. Looks at how much of the window right after
# each offset still sits above floor+margin.
_REVERB_OFFSET_DROP_DB = 15.0
_REVERB_TAIL_WINDOW_MS = 300.0
_REVERB_TAIL_RATIO_THRESHOLD = 0.4
_REVERB_MIN_OFFSETS = 3
_REVERB_FLOOR_MARGIN_DB = 6.0

# Lossy-compression artifacts: a hard spectral cutoff well below Nyquist
# (rather than a gradual natural roll-off) is consistent with audio that was
# encoded at a low bitrate (e.g. mp3) and later upsampled/re-encoded losslessly.
_COMPRESSION_ROLLOFF_DB = -50.0
_COMPRESSION_CUTOFF_RATIO_THRESHOLD = 0.75


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
    has_background_music: bool
    background_music_confidence: float
    has_reverb: bool
    reverb_tail_score: float
    has_compression_artifacts: bool
    spectral_cutoff_hz: float


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


def _spectral_flatness(frame: np.ndarray) -> float:
    """Geometric mean / arithmetic mean of the power spectrum ("Wiener
    entropy"). Near 1.0 for broadband noise, well below 1.0 for tonal
    content (music, hum) — the ratio a flat spectrum vs. a peaky one.
    """
    if frame.size < 2:
        return 1.0
    window = np.hanning(frame.size)
    spectrum = np.abs(np.fft.rfft(frame * window))
    spectrum = spectrum[spectrum > 1e-12]
    if spectrum.size == 0:
        return 1.0
    geometric_mean = np.exp(np.mean(np.log(spectrum)))
    arithmetic_mean = np.mean(spectrum)
    if arithmetic_mean <= 1e-12:
        return 1.0
    return float(geometric_mean / arithmetic_mean)


def _detect_background_music(samples: np.ndarray, sample_rate: int) -> tuple[bool, float]:
    frames = _frame_signal(samples, sample_rate, int(_SPECTRAL_FRAME_MS))
    frame_db = np.array([_rms_db(f) for f in frames])
    quiet_mask = (frame_db >= _MUSIC_QUIET_RMS_LOW_DB) & (frame_db <= _MUSIC_QUIET_RMS_HIGH_DB)
    quiet_frames = frames[quiet_mask]
    if quiet_frames.shape[0] < _MUSIC_MIN_QUIET_FRAMES:
        # Not enough narration-pause frames to judge — report "no music"
        # rather than guessing from too little evidence.
        return False, 1.0
    flatness = np.array([_spectral_flatness(f) for f in quiet_frames])
    mean_flatness = float(np.mean(flatness))
    return mean_flatness < _MUSIC_FLATNESS_THRESHOLD, mean_flatness


def _detect_reverb(frame_db: np.ndarray, frame_ms: float, noise_floor_db: float) -> tuple[bool, float]:
    frames_per_window = max(1, int(_REVERB_TAIL_WINDOW_MS / frame_ms))
    drop = frame_db[:-1] - frame_db[1:]
    offset_indices = np.nonzero(drop > _REVERB_OFFSET_DROP_DB)[0] + 1
    if offset_indices.size < _REVERB_MIN_OFFSETS:
        return False, 0.0

    floor_margin_db = noise_floor_db + _REVERB_FLOOR_MARGIN_DB
    tail_ratios = []
    for idx in offset_indices:
        tail = frame_db[idx : idx + frames_per_window]
        if tail.size == 0:
            continue
        tail_ratios.append(float(np.mean(tail > floor_margin_db)))

    if not tail_ratios:
        return False, 0.0
    mean_ratio = float(np.mean(tail_ratios))
    return mean_ratio > _REVERB_TAIL_RATIO_THRESHOLD, mean_ratio


def _detect_compression_artifacts(samples: np.ndarray, sample_rate: int) -> tuple[bool, float]:
    frames = _frame_signal(samples, sample_rate, int(_SPECTRAL_FRAME_MS))
    if frames.shape[1] < 2:
        return False, 0.0

    window = np.hanning(frames.shape[1])
    spectra = np.abs(np.fft.rfft(frames * window, axis=1))
    power = np.mean(np.square(spectra), axis=0)
    freqs = np.fft.rfftfreq(frames.shape[1], d=1.0 / sample_rate)

    power_db = 10 * np.log10(power + 1e-12)
    peak_db = np.max(power_db)
    above_floor = (power_db - peak_db) > _COMPRESSION_ROLLOFF_DB
    if not np.any(above_floor):
        return False, 0.0

    cutoff_idx = int(np.max(np.nonzero(above_floor)[0]))
    cutoff_hz = float(freqs[cutoff_idx])
    nyquist = sample_rate / 2.0
    if nyquist <= 0:
        return False, cutoff_hz
    cutoff_ratio = cutoff_hz / nyquist
    return cutoff_ratio < _COMPRESSION_CUTOFF_RATIO_THRESHOLD, cutoff_hz


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

    has_background_music, background_music_confidence = _detect_background_music(
        samples, sample_rate
    )
    has_reverb, reverb_tail_score = _detect_reverb(frame_db, _FRAME_MS, noise_floor_db)
    has_compression_artifacts, spectral_cutoff_hz = _detect_compression_artifacts(
        samples, sample_rate
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
        has_background_music=has_background_music,
        background_music_confidence=background_music_confidence,
        has_reverb=has_reverb,
        reverb_tail_score=reverb_tail_score,
        has_compression_artifacts=has_compression_artifacts,
        spectral_cutoff_hz=spectral_cutoff_hz,
    )
