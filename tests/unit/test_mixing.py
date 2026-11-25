import numpy as np
import pytest

from audiobook_engine.music.mixing import (
    DuckingConfig,
    compute_duck_envelope,
    mix_audio,
    mix_narration_and_music,
)

SAMPLE_RATE = 16000


def test_mix_audio_skip_path_returns_narration_unchanged():
    narration = np.linspace(-0.5, 0.5, SAMPLE_RATE, dtype=np.float32)

    result = mix_audio(narration, SAMPLE_RATE, music=None)

    assert np.array_equal(result, narration)
    assert result is not narration  # a copy, not the same array


def test_compute_duck_envelope_stays_within_unit_range():
    rng = np.random.default_rng(0)
    narration = (rng.standard_normal(SAMPLE_RATE * 10) * 0.2).astype(np.float32)

    envelope = compute_duck_envelope(narration, SAMPLE_RATE)

    assert envelope.shape == narration.shape
    assert np.all(envelope >= 0.0)
    assert np.all(envelope <= 1.0 + 1e-9)


def test_compute_duck_envelope_ducks_during_loud_region_and_recovers_during_silence():
    # Quiet fraction kept a minority of the total (~13%) — the adaptive
    # threshold (15th percentile of frame RMS) is estimated the same way
    # chunking/boundaries.py uses it for real narration, where brief pauses
    # are a minority of total duration; a quiet region that's as large a
    # share as the loud one would confuse a single global percentile, same
    # as boundaries.py's own adaptive-threshold rationale about noise floors
    # varying — not a scenario this estimator is meant to handle.
    config = DuckingConfig(intro_fade_s=0.0, outro_fade_s=0.0, attack_s=0.05, release_s=0.1)
    rng = np.random.default_rng(1)
    loud = (rng.standard_normal(SAMPLE_RATE * 10) * 0.3).astype(np.float32)
    quiet = (rng.standard_normal(int(SAMPLE_RATE * 1.5)) * 0.0005).astype(np.float32)
    narration = np.concatenate([loud, quiet])

    envelope = compute_duck_envelope(narration, SAMPLE_RATE, config)

    duck_gain_linear = 10 ** (config.duck_gain_db / 20.0)
    # Deep into the loud region, the envelope should have settled near the
    # ducked gain; 1s into the trailing quiet region (release_s=0.1 gives
    # ~10 time constants to settle), it should have recovered to full (1.0).
    mid_loud = envelope[SAMPLE_RATE * 5]
    settled_quiet = envelope[SAMPLE_RATE * 10 + SAMPLE_RATE]
    assert mid_loud == pytest.approx(duck_gain_linear, abs=0.05)
    assert settled_quiet == pytest.approx(1.0, abs=0.05)


def test_compute_duck_envelope_applies_intro_and_outro_fade():
    config = DuckingConfig(intro_fade_s=1.0, outro_fade_s=1.0)
    narration = np.zeros(SAMPLE_RATE * 5, dtype=np.float32)

    envelope = compute_duck_envelope(narration, SAMPLE_RATE, config)

    assert envelope[0] == pytest.approx(0.0, abs=1e-6)
    assert envelope[-1] == pytest.approx(0.0, abs=1e-6)


def test_mix_narration_and_music_loops_short_music_to_cover_narration():
    narration = np.zeros(1000, dtype=np.float32)
    music = np.full(300, 0.1, dtype=np.float32)
    envelope = np.ones(1000, dtype=np.float64)

    result = mix_narration_and_music(narration, music, envelope)

    assert result.shape == narration.shape
    assert np.all(np.abs(result) > 0.0)


def test_mix_narration_and_music_truncates_long_music_to_narration_length():
    narration = np.zeros(1000, dtype=np.float32)
    music = np.full(5000, 0.1, dtype=np.float32)
    envelope = np.ones(1000, dtype=np.float64)

    result = mix_narration_and_music(narration, music, envelope)

    assert result.shape == narration.shape


def test_mix_narration_and_music_raises_on_empty_music():
    narration = np.zeros(1000, dtype=np.float32)
    envelope = np.ones(1000, dtype=np.float64)

    with pytest.raises(ValueError):
        mix_narration_and_music(narration, np.array([], dtype=np.float32), envelope)


def test_mix_narration_and_music_applies_peak_safety_scaledown():
    narration = np.full(1000, 0.9, dtype=np.float32)
    music = np.full(1000, 0.9, dtype=np.float32)
    envelope = np.ones(1000, dtype=np.float64)

    result = mix_narration_and_music(narration, music, envelope)

    assert np.max(np.abs(result)) <= 1.0 + 1e-6
