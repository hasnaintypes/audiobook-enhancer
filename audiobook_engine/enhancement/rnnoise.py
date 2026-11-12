"""Tier 1 light-AI enhancement — RNNoise via ffmpeg's `arnndn` filter.

Originally attempted with DeepFilterNet2 (see docs/03-Tech-Stack.md), but
its Rust-based dependency (deepfilterlib) has no prebuilt wheel for
Colab's Python 3.13 and no Rust toolchain is present to build it from
source — confirmed by an actual failed install, not assumed. RNNoise was
the documented fallback (pure C, ffmpeg-native, no extra Python deps) and
is used here instead. DeepFilterNet2 can be revisited later (e.g. a
pinned older Python env, or once prebuilt wheels catch up), but this
project needs an actually-running Tier 1, not a theoretical one.

Model file: ffmpeg's arnndn filter requires an explicit .rnnn model path
— there is no bundled default. See download_default_model() for a known
community model (richardpl/arnndn-models, "std.rnnn" — a general-purpose
model, not tuned for Urdu specifically; same unverified-for-this-language
caveat as every other model choice in this project).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

DEFAULT_MODEL_URL = (
    "https://raw.githubusercontent.com/richardpl/arnndn-models/master/std.rnnn"
)


def download_default_model(destination: Path) -> Path:
    """Download the default RNNoise model if not already present. Requires
    network access — run in Colab, not as part of the unit test suite.
    """
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["wget", "-q", "-O", str(destination), DEFAULT_MODEL_URL],
        check=True,
    )
    return destination


def build_ffmpeg_command(
    input_path: Path, output_path: Path, model_path: Path, sample_rate: int
) -> list[str]:
    """Build the ffmpeg argv for RNNoise denoising.

    Pure function — no execution, no network. sample_rate must be pinned
    explicitly (see postprocess/dsp.py for why: ffmpeg filter chains have
    been observed silently resampling output without it).
    """
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-af",
        f"arnndn=model={model_path}",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]


def run_tier1_processing(
    input_path: Path, output_path: Path, model_path: Path, sample_rate: int
) -> None:
    """Run RNNoise denoising via ffmpeg. Requires ffmpeg on PATH and the
    model file to already exist — run in Colab, not as part of the unit
    test suite.
    """
    command = build_ffmpeg_command(input_path, output_path, model_path, sample_rate)
    subprocess.run(command, check=True, capture_output=True, text=True)
