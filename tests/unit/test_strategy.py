import numpy as np

from audiobook_engine.analysis.quality import analyze_quality
from audiobook_engine.analysis.strategy import (
    EnhancementTier,
    resolve_effective_tier,
    select_strategy,
)

SAMPLE_RATE = 16_000


def _tone(duration_s: float, amplitude: float, freq_hz: float = 440.0) -> np.ndarray:
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def test_clean_speech_like_recording_selects_tier_0():
    # Alternating tone/near-silence, like speech with pauses — unlike a
    # continuous tone, this gives the noise-floor estimator real quiet
    # frames to measure, so a clean recording's floor reads as actually low.
    word = _tone(0.3, amplitude=0.3)
    pause = np.zeros(int(SAMPLE_RATE * 0.2), dtype=np.float32)
    samples = np.concatenate([word, pause] * 5)
    report = analyze_quality(samples, SAMPLE_RATE)
    decision = select_strategy(report)
    assert decision.tier is EnhancementTier.TIER_0_DSP_ONLY
    assert not decision.needs_declipping


def test_noisy_recording_selects_tier_1():
    rng = np.random.default_rng(0)
    noisy = _tone(1.0, amplitude=0.3) + rng.normal(0, 0.05, SAMPLE_RATE).astype(
        np.float32
    )
    report = analyze_quality(noisy, SAMPLE_RATE)
    decision = select_strategy(report)
    assert decision.tier is EnhancementTier.TIER_1_LIGHT_AI
    assert any("noise floor" in reason for reason in decision.reasons)


def test_clipped_recording_flags_declipping_without_forcing_tier_1():
    samples = np.full(SAMPLE_RATE, 1.0, dtype=np.float32)
    report = analyze_quality(samples, SAMPLE_RATE)
    decision = select_strategy(report)
    assert decision.needs_declipping
    assert any("clipping" in reason for reason in decision.reasons)


def test_never_recommends_tier_2():
    rng = np.random.default_rng(0)
    very_noisy = rng.normal(0, 0.2, SAMPLE_RATE).astype(np.float32)
    report = analyze_quality(very_noisy, SAMPLE_RATE)
    decision = select_strategy(report)
    assert decision.tier <= EnhancementTier.TIER_1_LIGHT_AI


def test_resolve_effective_tier_force_tier2_wins_regardless_of_max_tier():
    result = resolve_effective_tier(
        EnhancementTier.TIER_0_DSP_ONLY, force_tier2=True, max_tier=EnhancementTier.TIER_0_DSP_ONLY
    )
    assert result is EnhancementTier.TIER_2_GPU_HEAVY


def test_resolve_effective_tier_no_cap_passes_through():
    result = resolve_effective_tier(
        EnhancementTier.TIER_1_LIGHT_AI, force_tier2=False, max_tier=None
    )
    assert result is EnhancementTier.TIER_1_LIGHT_AI


def test_resolve_effective_tier_caps_down_to_max_tier():
    result = resolve_effective_tier(
        EnhancementTier.TIER_1_LIGHT_AI, force_tier2=False, max_tier=EnhancementTier.TIER_0_DSP_ONLY
    )
    assert result is EnhancementTier.TIER_0_DSP_ONLY


def test_resolve_effective_tier_never_escalates_above_decision_tier():
    result = resolve_effective_tier(
        EnhancementTier.TIER_0_DSP_ONLY, force_tier2=False, max_tier=EnhancementTier.TIER_2_GPU_HEAVY
    )
    assert result is EnhancementTier.TIER_0_DSP_ONLY
