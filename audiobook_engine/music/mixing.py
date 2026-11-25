"""Optional background-music mixing with ducking/fades (docs/06-Roadmap.md
Phase 5) — subordinate-volume music under narration, fully skippable per job.

Runs once on the whole stitched narration file from stitching/stitch.py's
stitch_job (jobs/<job_id>/enhanced/final.wav), not per-chunk, since Phase 5's
only dependency is "Phase 2 complete" — same reasoning that keeps stitch_job
itself out of runner/job_runner.py's per-chunk loop.

Ducking derives a speech/non-speech mask the same way chunking/boundaries.py
derives chunk-cut silence gaps: analysis/framing.py's frame_signal/rms_db,
gated by an adaptively-estimated threshold (estimate_adaptive_silence_
threshold_db) rather than a new fixed constant — a fixed threshold was
already shown not to generalize across recordings with different noise
floors (gap.md evidence, 2026-11-20), so reusing the adaptive estimator here
instead of adding a fourth hardcoded silence dB value is deliberate.

Config defaults below are provisional starting points, not calibrated
against real audiobook samples — same caveat as every other threshold in
this codebase (see analysis/quality.py, chunking/boundaries.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audiobook_engine.analysis.framing import frame_signal, rms_db
from audiobook_engine.chunking.boundaries import estimate_adaptive_silence_threshold_db

_DEFAULT_DUCK_GAIN_DB = -18.0
_DEFAULT_ATTACK_S = 0.3
_DEFAULT_RELEASE_S = 1.0
_DEFAULT_INTRO_FADE_S = 2.0
_DEFAULT_OUTRO_FADE_S = 2.0
_DEFAULT_FRAME_MS = 20.0


@dataclass(frozen=True)
class DuckingConfig:
    duck_gain_db: float = _DEFAULT_DUCK_GAIN_DB
    attack_s: float = _DEFAULT_ATTACK_S
    release_s: float = _DEFAULT_RELEASE_S
    intro_fade_s: float = _DEFAULT_INTRO_FADE_S
    outro_fade_s: float = _DEFAULT_OUTRO_FADE_S
    frame_ms: float = _DEFAULT_FRAME_MS


def _smooth_frame_gains(target_gains: np.ndarray, frame_duration_s: float, config: DuckingConfig) -> np.ndarray:
    """One-pole attack/release smoothing of a frame-level target-gain sequence.

    Asymmetric time constants: ducking IN (gain falling, music getting
    quieter under speech) uses attack_s, coming back UP uses release_s —
    release is deliberately slower by default so the music doesn't snap back
    to full volume between words, only across real pauses.
    """
    if target_gains.size == 0:
        return target_gains.copy()

    smoothed = np.empty_like(target_gains)
    smoothed[0] = target_gains[0]
    for i in range(1, target_gains.size):
        prev = smoothed[i - 1]
        target = target_gains[i]
        time_const = config.attack_s if target < prev else config.release_s
        alpha = 1.0 - np.exp(-frame_duration_s / max(time_const, 1e-6))
        smoothed[i] = prev + alpha * (target - prev)
    return smoothed


def compute_duck_envelope(
    narration: np.ndarray, sample_rate: int, config: DuckingConfig = DuckingConfig()
) -> np.ndarray:
    """Compute a per-sample music-gain envelope in [0, 1] from the narration's
    own speech/non-speech pattern, with the config's intro/outro fade applied.

    Returns an array the same length as narration.
    """
    if narration.size == 0:
        return narration.copy()

    frames = frame_signal(narration, sample_rate, config.frame_ms)
    frame_db = np.array([rms_db(f) for f in frames])
    threshold_db = estimate_adaptive_silence_threshold_db(narration, sample_rate, config.frame_ms)
    is_speech = frame_db >= threshold_db

    duck_gain_linear = 10 ** (config.duck_gain_db / 20.0)
    target_gains = np.where(is_speech, duck_gain_linear, 1.0)

    frame_duration_s = config.frame_ms / 1000.0
    smoothed = _smooth_frame_gains(target_gains, frame_duration_s, config)

    frame_len = max(1, int(sample_rate * config.frame_ms / 1000))
    envelope = np.repeat(smoothed, frame_len)
    if envelope.size < narration.size:
        # Trailing partial frame that frame_signal() trimmed off — hold the
        # last frame's smoothed value across it rather than leaving it unset.
        pad = np.full(narration.size - envelope.size, smoothed[-1])
        envelope = np.concatenate([envelope, pad])
    else:
        envelope = envelope[: narration.size]

    duration_s = narration.size / sample_rate
    fade = np.ones(narration.size, dtype=np.float64)
    intro_samples = min(int(config.intro_fade_s * sample_rate), narration.size)
    if intro_samples > 0:
        fade[:intro_samples] *= np.linspace(0.0, 1.0, intro_samples)
    outro_samples = min(int(config.outro_fade_s * sample_rate), narration.size)
    if outro_samples > 0 and duration_s > config.intro_fade_s:
        fade[-outro_samples:] *= np.linspace(1.0, 0.0, outro_samples)

    return envelope * fade


def mix_narration_and_music(
    narration: np.ndarray, music: np.ndarray, envelope: np.ndarray
) -> np.ndarray:
    """Loop or truncate music to narration's length, apply the duck envelope,
    and sum with the narration.

    Looping is a simple tile-and-slice with no crossfade at the loop seam —
    an accepted v1 limitation (same spirit as stitch.py's accepted limitation
    on mid-word max_duration_fallback cuts), not something to silently
    over-engineer here. A simple global peak-safety scale-down guards against
    clipping from the sum; it is not a true limiter — same caution as
    postprocess/dsp.py's note on not guessing at an unvalidated DSP feature.
    """
    if narration.size == 0:
        return narration.copy()
    if music.size == 0:
        raise ValueError("Music track is empty")

    n = narration.size
    if music.size >= n:
        music_fitted = music[:n]
    else:
        repeats = -(-n // music.size)  # ceil division
        music_fitted = np.tile(music, repeats)[:n]

    mixed = narration + music_fitted * envelope

    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 1.0:
        mixed = mixed / peak

    return mixed.astype(narration.dtype, copy=False)


def mix_audio(
    narration: np.ndarray,
    sample_rate: int,
    music: np.ndarray | None,
    config: DuckingConfig = DuckingConfig(),
) -> np.ndarray:
    """Mix an optional music track under the narration with ducking/fades.

    music=None is the skip path: fully skippable per FR-011/roadmap Phase 5
    means no-music jobs get the narration back completely untouched (a copy,
    byte-identical) — no envelope computed, no mixing attempted.
    """
    if music is None:
        return narration.copy()

    envelope = compute_duck_envelope(narration, sample_rate, config)
    return mix_narration_and_music(narration, music, envelope)


def mix_job(
    job_id: str, music_path: Path | None, work_dir: Path, config: DuckingConfig = DuckingConfig()
) -> Path | None:
    """Download a job's stitched narration, optionally mix in a music track,
    and upload the result as the music_mixed artifact.

    music_path=None is the real-world skip path: returns None and uploads
    nothing at all — downstream consumers simply keep using Phase 2's
    jobs/<job_id>/enhanced/final.wav unchanged, so "identical to Phase 2's
    output" holds trivially with no duplicate copy ever written to diverge
    from it.

    Requires ffmpeg on PATH and network access to Neon Object Storage — run
    in Colab, not as part of the unit test suite.
    """
    import subprocess

    import soundfile as sf

    from audiobook_engine.analysis.io import load_mono
    from audiobook_engine.jobstore.storage import download_file, object_key, upload_file
    from audiobook_engine.postprocess.dsp import build_resample_command

    if music_path is None:
        return None

    work_dir.mkdir(parents=True, exist_ok=True)

    narration_path = work_dir / "download_narration.wav"
    download_file(object_key(job_id, "enhanced", "final.wav"), narration_path)
    narration, sample_rate = load_mono(narration_path)

    resampled_music_path = work_dir / "music_resampled.wav"
    resample_command = build_resample_command(music_path, resampled_music_path, sample_rate)
    subprocess.run(resample_command, check=True, capture_output=True, text=True)
    music, _ = load_mono(resampled_music_path)

    mixed = mix_audio(narration, sample_rate, music, config)

    output_path = work_dir / "music_mixed.wav"
    sf.write(str(output_path), mixed, sample_rate)
    upload_file(output_path, object_key(job_id, "music_mixed", "final.wav"))

    return output_path
