import numpy as np
import pytest

from audiobook_engine.chunking.boundaries import ChunkingConfig, find_silence_gaps, plan_chunks

SAMPLE_RATE = 16_000


def _tone(duration_s: float, amplitude: float, freq_hz: float = 440.0) -> np.ndarray:
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _silence(duration_s: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * duration_s), dtype=np.float32)


def test_single_long_silence_gap_is_detected():
    word = _tone(2.0, 0.5)
    pause = _silence(1.0)
    signal = np.concatenate([word, pause, word])
    config = ChunkingConfig(min_silence_duration_s=0.5)

    gaps = find_silence_gaps(signal, SAMPLE_RATE, config)

    assert len(gaps) == 1
    gap_start, gap_end = gaps[0]
    assert gap_start == pytest.approx(2.0, abs=0.05)
    assert gap_end == pytest.approx(3.0, abs=0.05)


def test_short_blip_of_silence_is_not_a_gap():
    word = _tone(1.0, 0.5)
    blip = _silence(0.1)
    signal = np.concatenate([word, blip, word])
    config = ChunkingConfig(min_silence_duration_s=0.5)

    gaps = find_silence_gaps(signal, SAMPLE_RATE, config)

    assert gaps == []


def test_plan_chunks_cuts_at_silence_gap_when_available():
    word = _tone(5.0, 0.5)
    pause = _silence(1.0)
    signal = np.concatenate([word, pause, word])
    config = ChunkingConfig(max_chunk_duration_s=8.0, min_silence_duration_s=0.5)

    boundaries = plan_chunks(signal, SAMPLE_RATE, config)

    assert len(boundaries) == 2
    assert boundaries[0].cut_reason == "silence_gap"
    assert boundaries[0].end_s == pytest.approx(5.5, abs=0.05)
    assert boundaries[1].start_s == boundaries[0].end_s


def test_plan_chunks_never_exceeds_max_duration_with_no_silence():
    signal = _tone(10.0, 0.5)
    config = ChunkingConfig(max_chunk_duration_s=3.0, min_silence_duration_s=0.5)

    boundaries = plan_chunks(signal, SAMPLE_RATE, config)

    for b in boundaries:
        assert b.end_s - b.start_s <= config.max_chunk_duration_s + 1e-6
    assert any(b.cut_reason == "max_duration_fallback" for b in boundaries)


def test_plan_chunks_prefers_nearby_gap_over_hard_cutoff():
    word = _tone(2.5, 0.5)
    pause = _silence(0.5)
    tail = _tone(2.0, 0.5)
    signal = np.concatenate([word, pause, tail])
    config = ChunkingConfig(max_chunk_duration_s=3.0, min_silence_duration_s=0.3)

    boundaries = plan_chunks(signal, SAMPLE_RATE, config)

    assert boundaries[0].cut_reason == "silence_gap"
    assert boundaries[0].end_s < 3.0


def test_plan_chunks_covers_full_audio_with_no_gaps_or_overlaps():
    signal = _tone(25.0, 0.5)
    config = ChunkingConfig(max_chunk_duration_s=7.0, min_silence_duration_s=0.5)

    boundaries = plan_chunks(signal, SAMPLE_RATE, config)

    assert boundaries[0].start_s == 0.0
    expected_duration = signal.size / SAMPLE_RATE
    assert boundaries[-1].end_s == pytest.approx(expected_duration, abs=1e-6)
    for prev, nxt in zip(boundaries, boundaries[1:]):
        assert prev.end_s == nxt.start_s


def test_plan_chunks_raises_on_empty_audio():
    config = ChunkingConfig()
    with pytest.raises(ValueError):
        plan_chunks(np.array([], dtype=np.float32), SAMPLE_RATE, config)


def test_plan_chunks_returns_single_chunk_for_short_nonempty_audio():
    signal = _tone(1.0, 0.5)
    config = ChunkingConfig(max_chunk_duration_s=600.0)

    boundaries = plan_chunks(signal, SAMPLE_RATE, config)

    assert len(boundaries) == 1
    assert boundaries[0].cut_reason == "end_of_audio"
    assert boundaries[0].start_s == 0.0
    assert boundaries[0].end_s == pytest.approx(1.0, abs=1e-6)
