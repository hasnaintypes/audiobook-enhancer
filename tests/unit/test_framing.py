import numpy as np

from audiobook_engine.analysis.framing import frame_signal, frame_times_s, rms_db

SAMPLE_RATE = 16_000


def _tone(duration_s: float, amplitude: float, freq_hz: float = 440.0) -> np.ndarray:
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def test_frame_signal_shapes_match_expected_frame_count():
    samples = _tone(1.0, 0.5)
    frame_ms = 20
    frames = frame_signal(samples, SAMPLE_RATE, frame_ms)
    frame_len = int(SAMPLE_RATE * frame_ms / 1000)
    expected_n_frames = len(samples) // frame_len
    assert frames.shape == (expected_n_frames, frame_len)


def test_rms_db_of_silence_is_floor_value():
    silence = np.zeros(SAMPLE_RATE // 10, dtype=np.float32)
    assert rms_db(silence) == -120.0


def test_frame_times_s_starts_at_zero_and_increments_by_frame_duration():
    frame_ms = 20.0
    times = frame_times_s(5, SAMPLE_RATE, frame_ms)
    assert times[0] == 0.0
    expected_step = frame_ms / 1000.0
    assert np.allclose(np.diff(times), expected_step)
