"""Strategy selection — maps a QualityReport to the cheapest enhancement
tier that's likely sufficient.

Per docs/03-Tech-Stack.md: Tier 0 is FFmpeg DSP only, Tier 1 adds a light
denoiser (RNNoise/DeepFilterNet, often CPU-only), Tier 2 is a heavier
GPU model, opt-in only (07-Decisions.md) — never selected automatically.
This module intentionally never recommends Tier 2; a human (or a later,
explicit "aggressive mode" toggle) decides to escalate there.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from audiobook_engine.analysis.quality import QualityReport

# Starting point, not calibrated against real recordings yet — same caveat
# as quality.py's thresholds. Revisit once Phase 1's real-sample experiment
# runs (docs/06-Roadmap.md Phase 1 acceptance criteria).
_NOISY_FLOOR_DB_THRESHOLD = -50.0


class EnhancementTier(IntEnum):
    TIER_0_DSP_ONLY = 0
    TIER_1_LIGHT_AI = 1
    TIER_2_GPU_HEAVY = 2


@dataclass(frozen=True)
class StrategyDecision:
    tier: EnhancementTier
    reasons: tuple[str, ...]
    needs_loudness_normalization: bool
    needs_declipping: bool


def select_strategy(report: QualityReport) -> StrategyDecision:
    reasons: list[str] = []
    tier = EnhancementTier.TIER_0_DSP_ONLY

    if report.noise_floor_db > _NOISY_FLOOR_DB_THRESHOLD:
        tier = EnhancementTier.TIER_1_LIGHT_AI
        reasons.append(
            f"noise floor {report.noise_floor_db:.1f}dB exceeds "
            f"{_NOISY_FLOOR_DB_THRESHOLD:.1f}dB threshold"
        )

    if report.is_clipped:
        reasons.append(
            f"clipping detected ({report.clipping_fraction:.4%} of samples)"
        )

    if report.is_loudness_inconsistent:
        reasons.append(
            f"loudness varies by {report.loudness_std_db:.1f}dB across sections"
        )

    # These three are detected but not actionable yet — there's no
    # music-separation, dereverb, or spectral-restoration stage in the
    # pipeline, so they're surfaced for visibility (and to explain a
    # recording that still sounds bad despite a clean tier decision)
    # rather than used to pick a tier.
    if report.has_background_music:
        reasons.append(
            f"background music detected under narration (spectral flatness "
            f"{report.background_music_confidence:.2f}) — no music-separation "
            f"stage exists yet, flagged for visibility only"
        )

    if report.has_reverb:
        reasons.append(
            f"reverb/echo detected (tail persistence {report.reverb_tail_score:.2f}) "
            f"— no dereverb stage exists yet, flagged for visibility only"
        )

    if report.has_compression_artifacts:
        reasons.append(
            f"lossy-compression artifacts detected (spectral cutoff "
            f"~{report.spectral_cutoff_hz:.0f}Hz) — lost high frequencies "
            f"can't be recovered, flagged for visibility only"
        )

    if not reasons:
        reasons.append("no significant quality issues detected")

    return StrategyDecision(
        tier=tier,
        reasons=tuple(reasons),
        needs_loudness_normalization=report.is_loudness_inconsistent,
        needs_declipping=report.is_clipped,
    )


def resolve_effective_tier(
    decision_tier: EnhancementTier, force_tier2: bool, max_tier: EnhancementTier | None
) -> EnhancementTier:
    """Resolve the tier a chunk actually runs at (runner/job_runner.py's
    run_job), given select_strategy's recommendation plus the two separate
    ways a caller can override it.

    force_tier2 (the existing explicit human opt-in, 07-Decisions.md) wins
    outright, unchanged from before this function existed. Otherwise a
    preset's tier_max (config/presets.yaml, config/presets.py) only ever
    caps the tier DOWN, never escalates it — select_strategy already never
    recommends Tier 2 on its own (this module's own docstring), so a
    preset's tier_max=2 does not mean "always run Tier 2"; it just declines
    to cap what select_strategy already decided.
    """
    if force_tier2:
        return EnhancementTier.TIER_2_GPU_HEAVY
    if max_tier is not None:
        return min(decision_tier, max_tier)
    return decision_tier
