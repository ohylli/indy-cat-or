"""Tests for the calibration measurement/report and the embeddings-cache reader.

All pure and fast -- synthetic vectors and tmp CSV/npy files; no model, no real
(gitignored) data. They cover the V0 report sections, the look-alike grouping,
the loud failures (missing manifest image, inconsistent cache), and the
per-image score CSV.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

import calibration_report as cr
from _common import BASE_METADATA_COLUMNS, load_cached_embeddings
from indycat.decision import Gallery


def make_gallery() -> Gallery:
    raw = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    return Gallery.from_raw(["g0", "g1"], raw)


# --------------------------------------------------------------------------- #
# summarize
# --------------------------------------------------------------------------- #


def test_summarize_basic_stats() -> None:
    stats = cr.summarize([0.0, 0.5, 1.0])
    assert stats.n == 3
    assert stats.min == 0.0
    assert stats.max == 1.0
    assert stats.mean == pytest.approx(0.5)
    assert stats.p50 == pytest.approx(0.5)


def test_summarize_empty_is_nan() -> None:
    stats = cr.summarize([])
    assert stats.n == 0
    assert math.isnan(stats.mean)


# --------------------------------------------------------------------------- #
# select_vectors / score_role
# --------------------------------------------------------------------------- #


def test_select_vectors_missing_name_raises() -> None:
    lookup = {"a": np.zeros(3, dtype=np.float32)}
    with pytest.raises(KeyError, match="absent from the embeddings cache"):
        cr.select_vectors(["a", "b"], lookup)


def test_score_role_attaches_breed_and_best_match() -> None:
    gallery = make_gallery()
    lookup = {"n0": np.array([1, 0, 0], dtype=np.float32)}
    scored = cr.score_role(["n0"], lookup, gallery, "max", breeds={"n0": "Persian"})
    assert scored[0].breed == "Persian"
    assert scored[0].best_match == "g0"
    assert scored[0].score == pytest.approx(1.0)


def test_score_role_missing_name_raises() -> None:
    gallery = make_gallery()
    with pytest.raises(KeyError, match="absent from the embeddings cache"):
        cr.score_role(["missing"], {}, gallery, "max")


# --------------------------------------------------------------------------- #
# build_report
# --------------------------------------------------------------------------- #


def test_build_report_has_all_sections_and_clean_gap() -> None:
    positives = [
        cr.ScoredImage("p0", 0.8, "g0", None),
        cr.ScoredImage("p1", 0.7, "g1", None),
    ]
    negatives = [
        cr.ScoredImage("Persian_1.jpg", 0.5, "g0", "Persian"),
        cr.ScoredImage("Abyssinian_1.jpg", 0.2, "g1", "Abyssinian"),
    ]
    report = cr.build_report("m.yaml", 5, positives, negatives, "max")
    assert "Score distribution" in report
    assert "Negatives by group" in report
    assert "Per-breed negative scores" in report
    assert "look-alike" in report
    assert "Persian" in report
    assert "false-positive risks" in report
    assert "clean gap" in report  # lowest pos 0.7 > highest neg 0.5


def test_build_report_flags_overlap() -> None:
    positives = [cr.ScoredImage("p0", 0.4, "g0", None)]
    negatives = [cr.ScoredImage("Persian_1.jpg", 0.5, "g0", "Persian")]
    report = cr.build_report("m", 5, positives, negatives, "max")
    assert "OVERLAP" in report


def test_build_report_groups_persian_as_lookalike() -> None:
    negatives = [
        cr.ScoredImage("Persian_1.jpg", 0.5, "g0", "Persian"),
        cr.ScoredImage("Maine_Coon_1.jpg", 0.4, "g0", "Maine_Coon"),
        cr.ScoredImage("Abyssinian_1.jpg", 0.2, "g0", "Abyssinian"),
    ]
    report = cr.build_report(
        "m", 5, [cr.ScoredImage("p", 0.9, "g0", None)], negatives, "max"
    )
    # The look-alike group line reports 2 breeds (Persian + Maine_Coon).
    assert "look-alike  (2 breeds" in report


# --------------------------------------------------------------------------- #
# write_scores_csv
# --------------------------------------------------------------------------- #


def test_write_scores_csv_roundtrips(tmp_path: Path) -> None:
    positives = [cr.ScoredImage("p0", 0.8, "g0", None)]
    negatives = [
        cr.ScoredImage("Persian_1.jpg", 0.5, "g0", "Persian"),
        cr.ScoredImage("Persian_2.jpg", 0.6, "g1", "Persian"),
    ]
    out = tmp_path / "scores.csv"
    cr.write_scores_csv(out, positives, negatives)
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert rows[0]["role"] == "negative"  # negatives first, worst on top
    assert rows[0]["source_filename"] == "Persian_2.jpg"  # 0.6 before 0.5
    assert rows[0]["breed"] == "Persian"
    assert any(r["role"] == "positive" for r in rows)


# --------------------------------------------------------------------------- #
# load_cached_embeddings (in _common)
# --------------------------------------------------------------------------- #


def write_cache(tmp_path: Path, names: list[str], dim: int = 4) -> tuple[Path, Path]:
    """Write a tiny aligned metadata.csv + embeddings.npy; return their paths."""
    meta = tmp_path / "metadata.csv"
    with meta.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(BASE_METADATA_COLUMNS)
        for i, name in enumerate(names):
            writer.writerow([i, name, True, "0.9", 0, 0, 1, 1, "0.5"])
    emb = tmp_path / "embeddings.npy"
    vectors = np.arange(len(names) * dim, dtype=np.float32).reshape(len(names), dim)
    np.save(emb, vectors)
    return meta, emb


def test_load_cached_embeddings_returns_aligned_names_and_vectors(
    tmp_path: Path,
) -> None:
    meta, emb = write_cache(tmp_path, ["a.jpg", "b.jpg"])
    names, vectors = load_cached_embeddings(meta, emb)
    assert names == ["a.jpg", "b.jpg"]
    assert vectors.shape == (2, 4)
    assert vectors.dtype == np.float32


def test_load_cached_embeddings_row_mismatch_raises(tmp_path: Path) -> None:
    meta, emb = write_cache(tmp_path, ["a.jpg", "b.jpg"])
    np.save(emb, np.zeros((3, 4), dtype=np.float32))  # drift the row count
    with pytest.raises(ValueError, match="inconsistent"):
        load_cached_embeddings(meta, emb)
