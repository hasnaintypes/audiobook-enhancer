from pathlib import Path

from audiobook_engine.enhancement.resemble import (
    build_resemble_worker_command,
    resolve_enhance_params,
)


def test_resolve_enhance_params_denoise_only_prioritizes_noise_removal():
    params = resolve_enhance_params(denoise_only=True)

    assert params["lambd"] == 0.9


def test_resolve_enhance_params_enhance_prioritizes_naturalness():
    params = resolve_enhance_params(denoise_only=False)

    assert params["lambd"] == 0.1


def test_resolve_enhance_params_fixed_values():
    params = resolve_enhance_params(denoise_only=False)

    assert params["nfe"] == 32
    assert params["solver"] == "midpoint"
    assert params["tau"] == 0.5


def test_build_resemble_worker_command_includes_denoise_only_flag():
    command = build_resemble_worker_command(
        Path("/venv/bin/python"),
        Path("/pkg/_resemble_worker.py"),
        Path("in.wav"),
        Path("out.wav"),
        "cuda",
        denoise_only=True,
    )

    assert command[0] == "/venv/bin/python"
    assert command[1] == "/pkg/_resemble_worker.py"
    assert "--denoise-only" in command


def test_build_resemble_worker_command_omits_flag_when_not_denoise_only():
    command = build_resemble_worker_command(
        Path("/venv/bin/python"),
        Path("/pkg/_resemble_worker.py"),
        Path("in.wav"),
        Path("out.wav"),
        "cpu",
        denoise_only=False,
    )

    assert "--denoise-only" not in command


def test_build_resemble_worker_command_passes_input_output_device():
    command = build_resemble_worker_command(
        Path("/venv/bin/python"),
        Path("/pkg/_resemble_worker.py"),
        Path("in.wav"),
        Path("out.wav"),
        "cpu",
        denoise_only=False,
    )

    assert "--input" in command
    assert command[command.index("--input") + 1] == "in.wav"
    assert "--output" in command
    assert command[command.index("--output") + 1] == "out.wav"
    assert "--device" in command
    assert command[command.index("--device") + 1] == "cpu"
