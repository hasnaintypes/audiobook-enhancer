from pathlib import Path

from audiobook_engine.enhancement.rnnoise import build_ffmpeg_command

SAMPLE_RATE = 48_000


def test_command_includes_arnndn_filter_with_model_path():
    command = build_ffmpeg_command(
        Path("in.wav"), Path("out.wav"), Path("/content/models/std.rnnn"), SAMPLE_RATE
    )
    filter_arg_index = command.index("-af") + 1
    assert "arnndn=model=/content/models/std.rnnn" == command[filter_arg_index]


def test_command_pins_sample_rate_and_codec():
    command = build_ffmpeg_command(
        Path("in.wav"), Path("out.wav"), Path("model.rnnn"), SAMPLE_RATE
    )
    assert command[command.index("-ar") + 1] == str(SAMPLE_RATE)
    assert command[command.index("-c:a") + 1] == "pcm_s16le"


def test_command_structure():
    command = build_ffmpeg_command(
        Path("in.wav"), Path("out.wav"), Path("model.rnnn"), SAMPLE_RATE
    )
    assert command[0] == "ffmpeg"
    assert command[-1] == str(Path("out.wav"))
