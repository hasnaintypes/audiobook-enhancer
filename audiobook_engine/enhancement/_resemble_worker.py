"""Standalone CLI worker that runs Resemble Enhance inside its own legacy
pinned venv (see enhancement/resemble.py's ensure_resemble_enhance_env).

Deliberately imports NOTHING from audiobook_engine: this script only ever
runs inside a dedicated Python 3.10 venv pinned to Resemble Enhance's
2023-era dependency set (torch==2.1.1, numpy==1.26.2, ...), which conflicts
with audiobook_engine's own numpy>=2.0 dependency. Keeping this file
import-independent of the main package is what makes that isolation work.

The nfe/solver/lambd/tau values below are inlined copies of
enhancement/resemble.py's resolve_enhance_params — keep them in sync; that
function is the canonical, unit-tested source of truth for these values,
this file just can't import it.
"""

import argparse

import soundfile as sf
import torch
from resemble_enhance.enhancer.inference import denoise, enhance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoise-only", action="store_true")
    args = parser.parse_args()

    samples, sr = sf.read(args.input)
    dwav = torch.from_numpy(samples).float()

    if args.denoise_only:
        wav, out_sr = denoise(dwav, sr, args.device)
    else:
        wav, out_sr = enhance(
            dwav, sr, args.device, nfe=32, solver="midpoint", lambd=0.1, tau=0.5
        )

    sf.write(args.output, wav.cpu().numpy(), out_sr)


if __name__ == "__main__":
    main()
