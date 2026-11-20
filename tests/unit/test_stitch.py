import numpy as np
import pytest

from audiobook_engine.stitching.stitch import stitch_chunks


def test_stitch_chunks_raises_on_empty_list():
    with pytest.raises(ValueError):
        stitch_chunks([], crossfade_samples=10)


def test_stitch_chunks_returns_single_chunk_unchanged():
    chunk = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    result = stitch_chunks([chunk], crossfade_samples=10)

    assert np.array_equal(result, chunk)
    assert result is not chunk  # returns a copy, not the same array


def test_stitch_chunks_output_length_accounts_for_crossfade_overlap():
    chunks = [np.ones(100, dtype=np.float32) for _ in range(3)]
    crossfade_samples = 10

    result = stitch_chunks(chunks, crossfade_samples)

    expected_length = sum(len(c) for c in chunks) - crossfade_samples * (len(chunks) - 1)
    assert len(result) == expected_length


def test_stitch_chunks_preserves_non_crossfaded_regions():
    a = np.full(100, 1.0, dtype=np.float32)
    b = np.full(100, 2.0, dtype=np.float32)
    crossfade_samples = 10

    result = stitch_chunks([a, b], crossfade_samples)

    assert np.allclose(result[: 100 - crossfade_samples], 1.0)
    assert np.allclose(result[100:], 2.0)


def test_stitch_chunks_crossfade_region_is_equal_power_not_equal_gain():
    # Equal-power (cos/sin, fade_out**2 + fade_in**2 == 1) deliberately does NOT
    # sum to a flat 1.0 for two equal-amplitude, in-phase sources -- it bumps up
    # to sqrt(2) at the midpoint. A flat/dipping sum there would mean the code
    # had regressed to a plain linear (equal-gain) crossfade instead, which is
    # exactly the perceived-loudness dip this module's docstring says to avoid.
    a = np.full(100, 1.0, dtype=np.float32)
    b = np.full(100, 1.0, dtype=np.float32)
    crossfade_samples = 10

    result = stitch_chunks([a, b], crossfade_samples)
    crossfade_region = result[100 - crossfade_samples : 100]

    assert crossfade_region.max() == pytest.approx(np.sqrt(2), abs=0.05)
    assert crossfade_region.min() >= 1.0 - 1e-5


def test_stitch_chunks_crossfade_endpoints_match_source_edges():
    a = np.full(100, 1.0, dtype=np.float32)
    b = np.full(100, 5.0, dtype=np.float32)
    crossfade_samples = 10

    result = stitch_chunks([a, b], crossfade_samples)
    crossfade_region = result[100 - crossfade_samples : 100]

    # First blended sample should be dominated by a (fade_out ~= 1, fade_in ~= 0)
    assert crossfade_region[0] == pytest.approx(1.0, abs=0.1)
    # Last blended sample should be dominated by b (fade_out ~= 0, fade_in ~= 1)
    assert crossfade_region[-1] == pytest.approx(5.0, abs=0.5)


def test_stitch_chunks_clamps_crossfade_to_shortest_chunk():
    a = np.full(100, 1.0, dtype=np.float32)
    tiny = np.full(3, 2.0, dtype=np.float32)  # shorter than crossfade_samples

    result = stitch_chunks([a, tiny], crossfade_samples=10)

    # Clamped to len(tiny)=3, so only 3 samples should be blended, not 10.
    expected_length = 100 + 3 - 3
    assert len(result) == expected_length


def test_stitch_chunks_handles_more_than_two_chunks_in_order():
    chunks = [np.full(50, float(i), dtype=np.float32) for i in range(4)]

    result = stitch_chunks(chunks, crossfade_samples=5)

    expected_length = 50 * 4 - 5 * 3
    assert len(result) == expected_length
    # Start and end should reflect the first and last chunk's values.
    assert result[0] == pytest.approx(0.0)
    assert result[-1] == pytest.approx(3.0)
