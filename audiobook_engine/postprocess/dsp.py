"""Tier 0 DSP postprocessing — FFmpeg filter chain only, no AI models.

Scope is deliberately conservative: this only trims LEADING/TRAILING
silence, not pauses within the narration — internal pauses are part of
the narrator's pacing, which is the project's top priority to preserve
(see docs/01-PRD.md). Shortening long internal pauses is a separate,
riskier feature left for later, not bundled in here.

De-essing is NOT implemented yet — there's no detection signal for
sibilance in analysis/quality.py yet, and guessing a de-ess filter's
parameters without one would be exactly the kind of unverified
threshold this project is trying to avoid stacking on top of each
other. Revisit once there's a real detection signal or a real sample
that needs it.

Filter parameters below are starting points, not calibrated against
real audiobook samples — same caveat as analysis/quality.py.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from audiobook_engine.analysis.strategy import StrategyDecision

_LOUDNORM_TARGET_LUFS = -16.0
_LOUDNORM_TRUE_PEAK = -1.5
_LOUDNORM_LRA = 11.0

_COMPRESSOR_THRESHOLD_DB = -18.0
_COMPRESSOR_RATIO = 2.0

_SILENCE_TRIM_THRESHOLD_DB = -45.0
_SILENCE_TRIM_MIN_DURATION_S = 0.3


def build_filter_chain(decision: StrategyDecision) -> list[str]:
    """Build the ffmpeg -af filter chain for a given strategy decision.

    Pure function — no subprocess, no filesystem — so the chain logic is
    unit-testable without ffmpeg installed.
    """
    filters: list[str] = []

    if decision.needs_declipping:
        filters.append("adeclip")

    # Trim leading/trailing silence only — never internal pauses. Trimming
    # the trailing end requires reversing, trimming "start" silence again,
    # then reversing back — ffmpeg's silenceremove has no direct "end" mode.
    _silence_trim = (
        f"silenceremove=start_periods=1:start_threshold={_SILENCE_TRIM_THRESHOLD_DB}dB:"
        f"start_silence={_SILENCE_TRIM_MIN_DURATION_S}"
    )
    filters.append(_silence_trim)
    filters.append("areverse")
    filters.append(_silence_trim)
    filters.append("areverse")

    filters.append(
        f"acompressor=threshold={_COMPRESSOR_THRESHOLD_DB}dB:ratio={_COMPRESSOR_RATIO}"
    )

    filters.append(
        f"loudnorm=I={_LOUDNORM_TARGET_LUFS}:TP={_LOUDNORM_TRUE_PEAK}:LRA={_LOUDNORM_LRA}"
    )

    return filters


def build_ffmpeg_command(
    input_path: Path, output_path: Path, decision: StrategyDecision, sample_rate: int
) -> list[str]:
    """Build the full ffmpeg argv for applying Tier 0 DSP processing.

    sample_rate must be the INPUT's actual sample rate (e.g. from the same
    load_mono() call used for analysis) — without pinning -ar explicitly,
    the filter chain was observed silently resampling the output to ~4x
    the input rate (confirmed on Colab: 48kHz in -> ~192kHz out).

    Pure function (no execution) — see run_tier0_processing for the part
    that actually shells out.
    """
    filter_chain = build_filter_chain(decision)
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-af",
        ",".join(filter_chain),
        # Pin both explicitly: -c:a alone was NOT enough to stop the
        # filter chain's sample-rate drift observed above.
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]


def run_tier0_processing(
    input_path: Path, output_path: Path, decision: StrategyDecision, sample_rate: int
) -> None:
    """Run the Tier 0 DSP chain via ffmpeg. Requires ffmpeg on PATH — run in
    Colab, not as part of the unit test suite.
    """
    command = build_ffmpeg_command(input_path, output_path, decision, sample_rate)
    subprocess.run(command, check=True, capture_output=True, text=True)
