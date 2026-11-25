from audiobook_engine.analysis.strategy import EnhancementTier
from audiobook_engine.config.presets import DEFAULT_PRESETS_PATH, load_preset_tier_caps, resolve_max_tier


def test_load_preset_tier_caps_reads_the_real_committed_file():
    caps = load_preset_tier_caps(DEFAULT_PRESETS_PATH)

    assert caps["gentle"] is EnhancementTier.TIER_0_DSP_ONLY
    assert caps["balanced"] is EnhancementTier.TIER_1_LIGHT_AI
    assert caps["strong"] is EnhancementTier.TIER_2_GPU_HEAVY


def test_resolve_max_tier_returns_none_for_preset_absent_from_caps():
    caps = {"gentle": EnhancementTier.TIER_0_DSP_ONLY}

    assert resolve_max_tier("custom", caps) is None


def test_resolve_max_tier_returns_the_configured_cap():
    caps = {"balanced": EnhancementTier.TIER_1_LIGHT_AI}

    assert resolve_max_tier("balanced", caps) is EnhancementTier.TIER_1_LIGHT_AI
