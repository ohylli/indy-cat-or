"""Tests for the decide-stage scoring core.

All pure and fast -- no model, no real data. Synthetic vectors exercise the
normalize/aggregate/score contract (shapes, best-match selection, the two
aggregations) rather than any learned values.
"""

from __future__ import annotations

import numpy as np
import pytest

from indycat.decision import (
    Gallery,
    Match,
    aggregate,
    l2_normalize,
    score,
    score_many,
)

# --------------------------------------------------------------------------- #
# l2_normalize
# --------------------------------------------------------------------------- #


def test_l2_normalize_single_vector_is_unit_norm() -> None:
    out = l2_normalize(np.array([3.0, 4.0], dtype=np.float32))
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out), 1.0, atol=1e-6)


def test_l2_normalize_batch_rows_are_unit_norm() -> None:
    out = l2_normalize(np.array([[3, 4], [0, 5]], dtype=np.float32))
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0], atol=1e-6)


def test_l2_normalize_zero_vector_has_no_nan() -> None:
    out = l2_normalize(np.zeros(4, dtype=np.float32))
    assert not np.isnan(out).any()
    np.testing.assert_allclose(out, 0.0)


# --------------------------------------------------------------------------- #
# aggregate
# --------------------------------------------------------------------------- #


def test_aggregate_max() -> None:
    sims = np.array([0.1, 0.9, 0.5], dtype=np.float32)
    assert aggregate(sims, "max") == pytest.approx(0.9)


def test_aggregate_mean_top3_averages_best_three() -> None:
    sims = np.array([0.1, 0.2, 0.9, 0.5, 0.8], dtype=np.float32)
    assert aggregate(sims, "mean-top3") == pytest.approx((0.9 + 0.8 + 0.5) / 3)


def test_aggregate_mean_top3_degrades_on_small_gallery() -> None:
    sims = np.array([0.4, 0.6], dtype=np.float32)
    assert aggregate(sims, "mean-top3") == pytest.approx(0.5)


def test_aggregate_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        aggregate(np.array([], dtype=np.float32), "max")


def test_aggregate_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown aggregation"):
        aggregate(np.array([0.1], dtype=np.float32), "median")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Gallery
# --------------------------------------------------------------------------- #


def test_gallery_from_raw_normalizes_and_keeps_names() -> None:
    gallery = Gallery.from_raw(["a", "b"], np.array([[3, 0], [0, 5]], dtype=np.float32))
    np.testing.assert_allclose(
        np.linalg.norm(gallery.vectors, axis=1), [1, 1], atol=1e-6
    )
    assert gallery.names == ("a", "b")


def test_gallery_rejects_row_mismatch() -> None:
    with pytest.raises(ValueError, match="row-aligned"):
        Gallery.from_raw(["a"], np.zeros((2, 3), dtype=np.float32))


def test_gallery_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        Gallery.from_raw([], np.zeros((0, 3), dtype=np.float32))


# --------------------------------------------------------------------------- #
# score / score_many
# --------------------------------------------------------------------------- #


def basis_gallery() -> Gallery:
    """A 3-vector orthonormal-basis gallery (each axis a distinct 'photo')."""
    raw = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    return Gallery.from_raw(["x", "y", "z"], raw)


def test_score_picks_the_closest_gallery_vector() -> None:
    gallery = basis_gallery()
    query = np.array([0.1, 0.9, 0.0], dtype=np.float32)  # nearest the 'y' axis
    match = score(query, gallery, "max")
    assert match.best_name == "y"
    assert match.best_index == 1
    assert match.score == pytest.approx(0.9 / np.linalg.norm(query), abs=1e-6)


def test_score_max_exceeds_mean_top3() -> None:
    # Two near-duplicate vectors plus one orthogonal: max keeps the single best,
    # mean-top3 is dragged down by the orthogonal third.
    raw = np.array([[1, 0], [1, 0.1], [0, 1]], dtype=np.float32)
    gallery = Gallery.from_raw(["a", "b", "c"], raw)
    query = np.array([1, 0], dtype=np.float32)
    assert score(query, gallery, "max").score > score(query, gallery, "mean-top3").score


def test_score_many_returns_one_match_per_query() -> None:
    gallery = Gallery.from_raw(["a", "b"], np.array([[1, 0], [0, 1]], dtype=np.float32))
    queries = np.array([[1, 0], [0, 1]], dtype=np.float32)
    matches = score_many(queries, gallery)
    assert all(isinstance(m, Match) for m in matches)
    assert [m.best_name for m in matches] == ["a", "b"]
