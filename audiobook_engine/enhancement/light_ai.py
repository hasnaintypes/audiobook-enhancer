"""Tier 1 light-AI enhancement — DeepFilterNet2, chosen per docs/03-Tech-Stack.md:
published PESQ 3.08 vs 2.81 baseline on VoiceBank+DEMAND, reported real-time
factor 0.19 on a single CPU thread (i.e. often GPU-unnecessary). No Urdu-
specific benchmark exists — this is a starting point, not a verified choice
for this project's actual material (see docs/01-PRD.md Open Assumptions).

This is the heaviest-weight module in the pipeline so far: it loads a real
model and is NOT covered by tests/unit (no torch/model dependency there).
Verify behavior on real audio in Colab, same pattern as ingestion/youtube.py
and postprocess/dsp.py.
"""

from __future__ import annotations

from pathlib import Path

_model = None
_df_state = None


def _get_model():
    """Lazily load the DeepFilterNet model once per process — loading it
    per-chunk would be wasteful for long-form audio (see docs/06-Roadmap.md
    Phase 4 chunking plan, which will reuse this cached model across chunks).
    """
    global _model, _df_state
    if _model is None:
        from df.enhance import init_df

        _model, _df_state, _ = init_df()
    return _model, _df_state


def run_tier1_processing(input_path: Path, output_path: Path) -> None:
    """Run DeepFilterNet2 denoising on a file. Requires the `deepfilternet`
    package (and its model weights, downloaded on first use) — run in
    Colab, not as part of the unit test suite.
    """
    from df.enhance import enhance, load_audio, save_audio

    model, df_state = _get_model()
    audio, _ = load_audio(str(input_path), sr=df_state.sr())
    enhanced = enhance(model, df_state, audio)
    save_audio(str(output_path), enhanced, df_state.sr())
