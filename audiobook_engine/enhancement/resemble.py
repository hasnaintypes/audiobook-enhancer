"""Tier 2 GPU enhancement via Resemble Enhance — explicitly opt-in only.

Resemble Enhance (github.com/resemble-ai/resemble-enhance, MIT-licensed) is
confirmed effectively unmaintained (no commits to `main` since Dec 2024,
checked 2026-11-20) and, worse, doesn't install/import on any current Python
stack: its hard `==`-pinned deps are all late-2023 versions (torch==2.1.1,
numpy==1.26.2, pandas==2.1.3, deepspeed==0.12.4, ...), several with no wheels
for current Python, and importing it under a modern numpy fails outright
(`AttributeError: module 'numpy' has no attribute '_no_nep50_warning'`). The
only known fork (Archkik/resemble-enhance) is equally dead. Per an explicit
user decision, this is still used — isolated in its own legacy-pinned venv
(ensure_resemble_enhance_env) rather than installed alongside
audiobook_engine's own numpy>=2.0 dependency, which it's incompatible with.

Tier 2 is never auto-selected (analysis/strategy.py never assigns it), so
this abandoned dependency carries far less risk than it would as a default
path. Per docs/07-Decisions.md D002, using this beyond experimentation
additionally requires a documented >=3-sample Urdu voice-preservation
listening test — not yet done, so treat any output from this module as
experimental until that happens.

Split the same way as enhancement/rnnoise.py: pure command/param builders
(unit-tested) and impure functions (real I/O + subprocess + system setup,
excluded from the unit suite, run in Colab only).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_WORKER_SCRIPT = Path(__file__).with_name("_resemble_worker.py")

# Resemble Enhance's own hard pins (confirmed 2026-11-20 via `pip download`
# against its PyPI metadata) -- the exact versions it needs simultaneously,
# none of which coexist with audiobook_engine's own numpy>=2.0 dependency.
_PINNED_REQUIREMENTS = [
    "torch==2.1.1",
    "torchaudio==2.1.1",
    "torchvision==0.16.1",
    "numpy==1.26.2",
    "pandas==2.1.3",
    "scipy==1.11.4",
    "librosa==0.10.1",
    "resampy==0.4.2",
    "soundfile==0.12.1",
    "omegaconf==2.3.0",
    "deepspeed==0.12.4",
    "celluloid==0.2.0",
    "matplotlib==3.8.1",
    "ptflops==0.7.1.2",
    "rich==13.7.0",
    "tqdm==4.66.1",
    "tabulate==0.8.10",
    "gradio==4.8.0",
    "resemble-enhance",
]


def resolve_enhance_params(denoise_only: bool) -> dict:
    """Pure: picks resemble_enhance's enhance()/denoise() call parameters.

    lambd=0.9 when denoise_only (prioritize noise removal over naturalness),
    else 0.1 (prioritize naturalness) — matches the published reference usage
    of the resemble_enhance.enhancer.inference module. Canonical source of
    truth for these values; _resemble_worker.py inlines a copy (it can't
    import this module) that must be kept in sync.
    """
    return {
        "nfe": 32,
        "solver": "midpoint",
        "lambd": 0.9 if denoise_only else 0.1,
        "tau": 0.5,
    }


def build_resemble_worker_command(
    python_bin: Path,
    worker_script: Path,
    input_path: Path,
    output_path: Path,
    device: str,
    denoise_only: bool,
) -> list[str]:
    """Pure: argv for invoking _resemble_worker.py under a given interpreter.

    Same role as rnnoise.py's build_ffmpeg_command — isolates the inherently
    testable "what command do we run" logic from the actual subprocess call.
    """
    command = [
        str(python_bin),
        str(worker_script),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--device",
        device,
    ]
    if denoise_only:
        command.append("--denoise-only")
    return command


def ensure_resemble_enhance_env(venv_dir: Path) -> Path:
    """Impure: creates (if needed) a dedicated Python 3.10 venv pinned to
    Resemble Enhance's exact 2023-era dependency set, isolated from both the
    main kernel's packages and audiobook_engine's own numpy>=2.0 requirement.
    Idempotent — returns the existing venv's python binary if already set up.

    Requires apt (python3.10/python3.10-venv) and network access — run in
    Colab, not as part of the unit test suite.
    """
    python_bin = venv_dir / "bin" / "python"
    if python_bin.exists():
        return python_bin

    subprocess.run(
        ["apt-get", "install", "-y", "-qq", "python3.10", "python3.10-venv"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["python3.10", "-m", "venv", str(venv_dir)], check=True, capture_output=True, text=True
    )
    subprocess.run(
        [str(python_bin), "-m", "pip", "install", "-q", *_PINNED_REQUIREMENTS],
        check=True,
        capture_output=True,
        text=True,
    )
    return python_bin


def run_tier2_processing(
    input_path: Path,
    output_path: Path,
    sample_rate: int,
    venv_python: Path,
    device: str = "cuda",
    denoise_only: bool = False,
) -> None:
    """Impure: runs _resemble_worker.py under the isolated legacy venv's
    interpreter to process one chunk through Resemble Enhance. Requires
    ensure_resemble_enhance_env to have already set up venv_python — run in
    Colab, not as part of the unit test suite.
    """
    import soundfile as sf

    _, sr = sf.read(str(input_path), frames=1)
    if sr != sample_rate:
        raise ValueError(
            f"Input file sample rate {sr} doesn't match expected {sample_rate}"
        )

    command = build_resemble_worker_command(
        venv_python, _WORKER_SCRIPT, input_path, output_path, device, denoise_only
    )
    subprocess.run(command, check=True, capture_output=True, text=True)
