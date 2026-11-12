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


def test_quiet_signal_has_low_noise_floor():
    rng = np.random.default_rng(0)
    quiet_hiss = (rng.normal(0, 0.0005, SAMPLE_RATE)).astype(np.float32)
    report = analyze_quality(quiet_hiss, SAMPLE_RATE)
    assert report.noise_floor_db < -45.0


def test_loud_hiss_raises_noise_floor():
    rng = np.random.default_rng(0)
    loud_hiss = (rng.normal(0, 0.05, SAMPLE_RATE)).astype(np.float32)
    report = analyze_quality(loud_hiss, SAMPLE_RATE)
    assert report.noise_floor_db > -45.0


def test_consistent_loudness_is_not_flagged():
    samples = _tone(9.0, amplitude=0.3)
    report = analyze_quality(samples, SAMPLE_RATE)
    assert not report.is_loudness_inconsistent
    assert report.loudness_std_db < 1.0


def test_volume_jump_between_sections_is_flagged():
    quiet_section = _tone(4.5, amplitude=0.05)
    loud_section = _tone(4.5, amplitude=0.5)
    samples = np.concatenate([quiet_section, loud_section])
    report = analyze_quality(samples, SAMPLE_RATE)
    assert report.is_loudness_inconsistent
    assert report.loudness_std_db > 6.0


def _speech_with_pauses(
    n_words: int, pause_amplitude_adder: np.ndarray | None = None
) -> np.ndarray:
    rng = np.random.default_rng(1)
    word_len = int(SAMPLE_RATE * 0.3)
    pause_len = int(SAMPLE_RATE * 0.4)
    chunks = []
    for _ in range(n_words):
        word = (0.3 * rng.normal(0, 1, word_len)).astype(np.float32)
        pause = np.zeros(pause_len, dtype=np.float32)
        if pause_amplitude_adder is not None:
            pause = pause + pause_amplitude_adder[:pause_len]
        chunks.extend([word, pause])
    return np.concatenate(chunks).astype(np.float32)


def test_clean_pauses_have_no_background_music():
    samples = _speech_with_pauses(n_words=10)
    report = analyze_quality(samples, SAMPLE_RATE)
    assert not report.has_background_music


def test_tonal_bed_under_pauses_is_detected_as_background_music():
    t = np.linspace(0, 10.0, SAMPLE_RATE * 10, endpoint=False)
    music_bed = (0.03 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    samples = _speech_with_pauses(n_words=10, pause_amplitude_adder=music_bed)
    report = analyze_quality(samples, SAMPLE_RATE)
    assert report.has_background_music
    assert report.background_music_confidence < 0.35


def test_dry_impulses_have_no_reverb():
    rng = np.random.default_rng(2)
    bursts = []
    for _ in range(6):
        burst = (0.5 * rng.normal(0, 1, int(SAMPLE_RATE * 0.15))).astype(np.float32)
        silence = np.zeros(int(SAMPLE_RATE * 0.3), dtype=np.float32)
        bursts.extend([burst, silence])
    samples = np.concatenate(bursts)
    report = analyze_quality(samples, SAMPLE_RATE)
    assert not report.has_reverb


def test_decaying_tail_after_bursts_is_detected_as_reverb():
    rng = np.random.default_rng(2)
    decay_len = int(SAMPLE_RATE * 0.3)
    # Direct sound drops sharply (>15dB, like the dry case), but a quieter
    # reverberant tail lingers afterward with a slow decay rather than
    # falling straight to silence.
    decay_envelope = (0.15 * np.exp(-np.linspace(0, 3, decay_len))).astype(np.float32)
    bursts = []
    for _ in range(6):
        burst = (0.5 * rng.normal(0, 1, int(SAMPLE_RATE * 0.15))).astype(np.float32)
        tail_noise = (0.5 * rng.normal(0, 1, decay_len)).astype(np.float32)
        tail = tail_noise * decay_envelope
        bursts.extend([burst, tail])
    samples = np.concatenate(bursts)
    report = analyze_quality(samples, SAMPLE_RATE)
    assert report.has_reverb
    assert report.reverb_tail_score > 0.4


def test_full_bandwidth_noise_has_no_compression_artifacts():
    rng = np.random.default_rng(3)
    samples = (0.3 * rng.normal(0, 1, SAMPLE_RATE * 2)).astype(np.float32)
    report = analyze_quality(samples, SAMPLE_RATE)
    assert not report.has_compression_artifacts


def test_low_pass_filtered_noise_is_detected_as_compression_artifact():
    rng = np.random.default_rng(3)
    # Simulate a lossy-encoded source re-saved at full sample rate: energy
    # only below ~1/4 of Nyquist, a hard cliff above it.
    t = np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE
    cutoff_hz = SAMPLE_RATE / 8
    narrowband = np.zeros_like(t, dtype=np.float32)
    for freq in np.linspace(100, cutoff_hz, 20):
        narrowband += np.sin(2 * np.pi * freq * t).astype(np.float32)
    narrowband = (0.3 * narrowband / np.max(np.abs(narrowband))).astype(np.float32)
    report = analyze_quality(narrowband, SAMPLE_RATE)
    assert report.has_compression_artifacts
    assert report.spectral_cutoff_hz < SAMPLE_RATE / 4
