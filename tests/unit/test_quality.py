import numpy as np
import pytest

from audiobook_engine.analysis.quality import analyze_quality

SAMPLE_RATE = 16_000


def _tone(duration_s: float, amplitude: float, freq_hz: float = 440.0) -> np.ndarray:
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def test_clean_tone_is_not_clipped():
    samples = _tone(1.0, amplitude=0.5)
    report = analyze_quality(samples, SAMPLE_RATE)
    assert not report.is_clipped
    assert report.peak_amplitude < 0.6


def test_full_scale_square_wave_is_clipped():
    samples = np.full(SAMPLE_RATE, 1.0, dtype=np.float32)
    report = analyze_quality(samples, SAMPLE_RATE)
    assert report.is_clipped
    assert report.clipping_fraction > 0.9


def test_silence_is_detected():
    samples = np.zeros(SAMPLE_RATE, dtype=np.float32)
    report = analyze_quality(samples, SAMPLE_RATE)
    assert report.silence_fraction == pytest.approx(1.0)
    assert report.speech_fraction == pytest.approx(0.0)


def test_loud_tone_has_low_silence_fraction():
    samples = _tone(1.0, amplitude=0.5)
    report = analyze_quality(samples, SAMPLE_RATE)
    assert report.silence_fraction < 0.1


def test_duration_matches_input_length():
    samples = _tone(2.0, amplitude=0.3)
    report = analyze_quality(samples, SAMPLE_RATE)
    assert report.duration_seconds == pytest.approx(2.0, abs=0.01)


def test_empty_waveform_raises():
    with pytest.raises(ValueError):
        analyze_quality(np.array([], dtype=np.float32), SAMPLE_RATE)


def test_stereo_input_raises():
    stereo = np.zeros((SAMPLE_RATE, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        analyze_quality(stereo, SAMPLE_RATE)
