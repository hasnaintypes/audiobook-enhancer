from pathlib import Path

from audiobook_engine.analysis.strategy import EnhancementTier, StrategyDecision
from audiobook_engine.postprocess.dsp import build_filter_chain, build_ffmpeg_command

SAMPLE_RATE = 48_000


def _decision(needs_declipping: bool, needs_loudness_normalization: bool = False) -> StrategyDecision:
    return StrategyDecision(
        tier=EnhancementTier.TIER_0_DSP_ONLY,
        reasons=("test",),
        needs_loudness_normalization=needs_loudness_normalization,
        needs_declipping=needs_declipping,
    )


def test_declip_filter_included_when_needed():
    chain = build_filter_chain(_decision(needs_declipping=True))
    assert "adeclip" in chain


def test_declip_filter_omitted_when_not_needed():
    chain = build_filter_chain(_decision(needs_declipping=False))
    assert "adeclip" not in chain


def test_loudnorm_always_present():
    chain = build_filter_chain(_decision(needs_declipping=False))
    assert any(f.startswith("loudnorm") for f in chain)


def test_silence_trim_only_targets_leading_trailing():
    chain = build_filter_chain(_decision(needs_declipping=False))
    silence_filters = [f for f in chain if "silenceremove" in f]
    assert len(silence_filters) == 2  # once forward, once reversed


def test_ffmpeg_command_structure():
    decision = _decision(needs_declipping=True)
    command = build_ffmpeg_command(Path("in.wav"), Path("out.wav"), decision, SAMPLE_RATE)
    assert command[0] == "ffmpeg"
    assert "-i" in command
    assert str(Path("in.wav")) in command
    assert str(Path("out.wav")) in command
    assert command[-1] == str(Path("out.wav"))


def test_ffmpeg_command_pins_16bit_pcm_output():
    # Without this, ffmpeg defaults to a high-precision float codec —
    # observed ~8x the expected file size for 16-bit mono input.
    command = build_ffmpeg_command(Path("in.wav"), Path("out.wav"), _decision(True), SAMPLE_RATE)
    assert "-c:a" in command
    assert command[command.index("-c:a") + 1] == "pcm_s16le"


def test_ffmpeg_command_pins_sample_rate():
    # -c:a alone was NOT enough — the filter chain was observed silently
    # resampling output to ~4x the input rate without an explicit -ar.
    command = build_ffmpeg_command(Path("in.wav"), Path("out.wav"), _decision(True), SAMPLE_RATE)
    assert "-ar" in command
    assert command[command.index("-ar") + 1] == str(SAMPLE_RATE)


def test_ffmpeg_command_declip_affects_filter_arg():
    with_declip = build_ffmpeg_command(Path("in.wav"), Path("out.wav"), _decision(True), SAMPLE_RATE)
    without_declip = build_ffmpeg_command(Path("in.wav"), Path("out.wav"), _decision(False), SAMPLE_RATE)
    filter_arg_index = with_declip.index("-af") + 1
    assert "adeclip" in with_declip[filter_arg_index]
    assert "adeclip" not in without_declip[filter_arg_index]
