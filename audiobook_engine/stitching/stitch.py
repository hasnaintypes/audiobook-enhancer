"""Crossfade stitching of chunked job output (docs/06-Roadmap.md Phase 2
remainder) back into a single deliverable file.

runner/job_runner.py slices chunks with NO overlap — exact, contiguous
boundaries — and that code already shipped and is proven working on Colab,
so this module does not touch it. Instead it crossfades using the tail of
chunk i and the head of chunk i+1 as they already exist: chunking/
boundaries.py's plan_chunks() prefers cutting at detected silence gaps, so
the common case already has near-silent audio on both sides of a boundary
(a crossfade there is a safety net against a sample-level click, not a save
for word-splitting). The rare "max_duration_fallback" cut (no silence gap
found in range) can land mid-word — a crossfade softens the amplitude jump
but can't reconstruct audio that was never captured on either side; that's
an accepted limitation of anything but an ideal silence-based cut, not
something fixable here.

Pure logic (stitch_chunks) is kept separate from stitch_job's real I/O
(network storage, file writes), the same split used throughout this
package (chunking/manifest.py + jobstore/storage.py; postprocess/dsp.py's
builders vs. its run_tier0_processing). stitch_job is excluded from the
unit suite like those run_* functions already are — run it in Colab, not
locally without network access.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def stitch_chunks(chunks: list[np.ndarray], crossfade_samples: int) -> np.ndarray:
    """Concatenate chunks in order, crossfading each adjacent pair's boundary.

    Equal-power (cos/sin) crossfade — fade_out**2 + fade_in**2 == 1 — avoids
    the audible mid-crossfade volume dip a plain linear ramp produces.

    Each pair's crossfade length is clamped to min(crossfade_samples,
    len(prev), len(curr)) so a short final chunk can't raise an index error.
    """
    if not chunks:
        raise ValueError("No chunks to stitch")
    if len(chunks) == 1:
        return chunks[0].copy()

    result: list[np.ndarray] = [chunks[0]]
    for curr in chunks[1:]:
        prev = result[-1]
        n = min(crossfade_samples, len(prev), len(curr))
        if n <= 0:
            result.append(curr)
            continue

        prev_head, prev_tail = prev[:-n], prev[-n:]
        curr_fade, curr_tail = curr[:n], curr[n:]

        ramp = np.linspace(0.0, np.pi / 2, n, dtype=np.float32)
        fade_out = np.cos(ramp)
        fade_in = np.sin(ramp)
        blended = prev_tail * fade_out + curr_fade * fade_in

        result[-1] = prev_head
        result.append(blended)
        result.append(curr_tail)

    return np.concatenate(result)


def stitch_job(job_id: str, work_dir: Path, crossfade_ms: float = 50.0) -> Path:
    """Download a completed job's per-chunk output, stitch it, and upload the
    whole-file result.

    Requires network access to Neon Object Storage — run in Colab, not as
    part of the unit test suite.
    """
    import soundfile as sf

    from audiobook_engine.chunking.manifest import pending_chunk_indices
    from audiobook_engine.jobstore.storage import (
        download_file,
        download_manifest,
        object_key,
        upload_file,
    )
    from audiobook_engine.runner.job_runner import chunk_filename

    work_dir.mkdir(parents=True, exist_ok=True)

    manifest = download_manifest(job_id)
    if pending_chunk_indices(manifest):
        raise ValueError(
            f"Job {job_id!r} is not complete: chunks "
            f"{pending_chunk_indices(manifest)} are still pending — refusing to "
            "stitch a truncated job"
        )

    chunk_arrays: list[np.ndarray] = []
    sample_rate: int | None = None
    for idx in range(manifest.total_chunks):
        local_path = work_dir / f"download_{chunk_filename(idx)}"
        download_file(object_key(job_id, "chunks", chunk_filename(idx)), local_path)
        samples, sr = sf.read(str(local_path), dtype="float32", always_2d=False)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(
                f"Sample rate mismatch across chunks for job {job_id!r}: "
                f"chunk 0 is {sample_rate}Hz, chunk {idx} is {sr}Hz"
            )
        chunk_arrays.append(samples)

    assert sample_rate is not None  # total_chunks >= 1 or download_manifest would have 0 chunks
    crossfade_samples = int(crossfade_ms / 1000 * sample_rate)
    stitched = stitch_chunks(chunk_arrays, crossfade_samples)

    output_path = work_dir / "stitched.wav"
    sf.write(str(output_path), stitched, sample_rate)
    upload_file(output_path, object_key(job_id, "enhanced", "final.wav"))

    return output_path
