"""Tier 2 GPU enhancement via Resemble Enhance — explicitly opt-in only.

Resemble Enhance (github.com/resemble-ai/resemble-enhance, MIT-licensed) is
confirmed effectively unmaintained (no commits to `main` since Dec 2024,
checked 2026-11-20) but still used here per an explicit, informed decision:
Tier 2 is never auto-selected (analysis/strategy.py never assigns it), so an
abandoned-but-functional model carries far less risk than it would as a
default path. Per docs/07-Decisions.md D002, using this beyond experimentation
additionally requires a documented >=3-sample Urdu voice-preservation
listening test — not yet done, so treat any output from this module as
experimental until that happens.

Requires a CUDA GPU and the `resemble-enhance` package, pip-installed in
Colab (not declared in pyproject.toml — same precedent as soundfile/botocore,
which are also Colab-provided rather than pinned here). Split the same way as
enhancement/rnnoise.py: a pure parameter-selection function (unit-tested) and
an impure processing function (real I/O + GPU, excluded from the unit suite,
run in Colab only).
"""

from __future__ import annotations

from pathlib import Path


def resolve_enhance_params(denoise_only: bool) -> dict:
    """Pure: picks resemble_enhance's enhance()/denoise() call parameters.

    lambd=0.9 when denoise_only (prioritize noise removal over naturalness),
    else 0.1 (prioritize naturalness) — matches the published reference usage
    of the resemble_enhance.enhancer.inference module.
    """
    return {
        "nfe": 32,
        "solver": "midpoint",
        "lambd": 0.9 if denoise_only else 0.1,
        "tau": 0.5,
    }


def run_tier2_processing(
    input_path: Path,
    output_path: Path,
    sample_rate: int,
    device: str = "cuda",
    denoise_only: bool = False,
) -> None:
    """Impure: loads audio, runs resemble_enhance's enhance() (or denoise()
    if denoise_only), writes the result. Requires a CUDA GPU and the
    resemble-enhance package — run in Colab, not as part of the unit test
    suite.
    """
    import soundfile as sf
    import torch
    from resemble_enhance.enhancer.inference import denoise, enhance

    samples, sr = sf.read(str(input_path))
    if sr != sample_rate:
        raise ValueError(
            f"Input file sample rate {sr} doesn't match expected {sample_rate}"
        )
    dwav = torch.from_numpy(samples).float()

    if denoise_only:
        wav, out_sr = denoise(dwav, sr, device)
    else:
        params = resolve_enhance_params(denoise_only)
        wav, out_sr = enhance(dwav, sr, device, **params)

    sf.write(str(output_path), wav.cpu().numpy(), out_sr)
