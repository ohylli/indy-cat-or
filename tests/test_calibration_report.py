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

import calibration as cr
from _common import BASE_METADATA_COLUMNS, load_cached_embeddings
from calibration import report_common
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
# threshold sweep (V1)
# --------------------------------------------------------------------------- #


def test_sweep_thresholds_are_round_and_bracket_the_range() -> None:
    positives = [cr.ScoredImage("p", 0.83, "g0", None)]
    negatives = [cr.ScoredImage("n", 0.12, "g0", "Persian")]
    thresholds = cr.sweep_thresholds(positives, negatives, 0.05)
    # Round multiples of the step, spanning below the min (0.12 -> 0.10) to above
    # the max (0.83 -> 0.85), so the grid brackets the full trade-off.
    assert thresholds[0] == pytest.approx(0.10)
    assert thresholds[-1] == pytest.approx(0.85)
    assert all(round(t / 0.05) == pytest.approx(t / 0.05) for t in thresholds)


def test_sweep_thresholds_empty_input_is_empty() -> None:
    assert cr.sweep_thresholds([], [], 0.05) == []


def test_sweep_thresholds_rejects_nonpositive_step() -> None:
    with pytest.raises(ValueError, match="positive"):
        cr.sweep_thresholds([cr.ScoredImage("p", 0.5, "g0", None)], [], 0.0)


def test_build_sweep_fpr_and_recall_with_lookalike_split() -> None:
    positives = [
        cr.ScoredImage("p0", 0.8, "g0", None),
        cr.ScoredImage("p1", 0.4, "g0", None),
    ]
    negatives = [
        cr.ScoredImage("Persian_1.jpg", 0.6, "g0", "Persian"),  # look-alike
        cr.ScoredImage("Persian_2.jpg", 0.3, "g0", "Persian"),  # look-alike
        cr.ScoredImage("Abyssinian_1.jpg", 0.5, "g0", "Abyssinian"),  # easy
    ]
    [row] = cr.build_sweep(positives, negatives, [0.5])
    assert row.cutoff == 0.5
    assert row.fpr_overall == pytest.approx(2 / 3)  # Persian_1 + Abyssinian_1 >= 0.5
    assert row.fpr_lookalike == pytest.approx(0.5)  # 1 of 2 Persians
    assert row.fpr_easy == pytest.approx(1.0)  # the one Abyssinian
    assert row.recall == pytest.approx(0.5)  # 1 of 2 positives (>= uses 0.8 only)


def test_build_sweep_empty_group_is_nan() -> None:
    # No look-alike negatives -> look-alike FPR is NaN (renders as a dash).
    negatives = [cr.ScoredImage("Abyssinian_1.jpg", 0.5, "g0", "Abyssinian")]
    [row] = cr.build_sweep([cr.ScoredImage("p", 0.9, "g0", None)], negatives, [0.5])
    assert math.isnan(row.fpr_lookalike)


def test_build_breed_sweep_is_sorted_worst_first() -> None:
    negatives = [
        cr.ScoredImage("Abyssinian_1.jpg", 0.3, "g0", "Abyssinian"),
        cr.ScoredImage("Persian_1.jpg", 0.7, "g0", "Persian"),  # highest max
    ]
    breeds, fpr_by_breed = cr.build_breed_sweep(negatives, [0.5])
    assert breeds == ["Persian", "Abyssinian"]  # Persian leads (max 0.7 > 0.3)
    assert fpr_by_breed["Persian"] == [pytest.approx(1.0)]
    assert fpr_by_breed["Abyssinian"] == [pytest.approx(0.0)]


def test_build_report_includes_sweep_sections() -> None:
    positives = [cr.ScoredImage("p0", 0.8, "g0", None)]
    negatives = [cr.ScoredImage("Persian_1.jpg", 0.5, "g0", "Persian")]
    report = cr.build_report("m.yaml", 5, positives, negatives, "max")
    assert "Threshold sweep" in report
    assert "Per-breed FPR by cutoff" in report
    assert "recall" in report


# --------------------------------------------------------------------------- #
# threshold pick (V2)
# --------------------------------------------------------------------------- #


def _pick_data() -> tuple[list[cr.ScoredImage], list[cr.ScoredImage]]:
    """A small fixed split with hand-checkable cutoffs.

    Positives 0.9, 0.6; look-alike (Persian) negatives 0.5, 0.7; easy negative 0.3.
    Distinct scores {0.3,0.5,0.6,0.7,0.9} -> candidate midpoints {0.4,0.55,0.65,0.8}
    plus bracketing endpoints, so picks land on those midpoints.
    """
    positives = [
        cr.ScoredImage("p0", 0.9, "g0", None),
        cr.ScoredImage("p1", 0.6, "g0", None),
    ]
    negatives = [
        cr.ScoredImage("Persian_1.jpg", 0.5, "g0", "Persian"),
        cr.ScoredImage("Persian_2.jpg", 0.7, "g0", "Persian"),
        cr.ScoredImage("Abyssinian_1.jpg", 0.3, "g0", "Abyssinian"),
    ]
    return positives, negatives


def test_candidate_cutoffs_are_sorted_midpoints_bracketing_the_range() -> None:
    positives = [cr.ScoredImage("p", 0.8, "g0", None)]
    negatives = [
        cr.ScoredImage("n0", 0.2, "g0", "Abyssinian"),
        cr.ScoredImage("n1", 0.5, "g0", "Abyssinian"),
    ]
    cuts = cr.candidate_cutoffs(positives, negatives)
    assert cuts == sorted(cuts)
    assert cuts[0] < 0.2 and cuts[-1] > 0.8  # endpoints bracket the data
    # Interior candidates are midpoints between adjacent distinct scores.
    assert cuts[1] == pytest.approx(0.35)  # (0.2 + 0.5) / 2
    assert cuts[2] == pytest.approx(0.65)  # (0.5 + 0.8) / 2
    # No candidate coincides with an observed score (keeps >= unambiguous).
    assert all(c not in (0.2, 0.5, 0.8) for c in cuts)


def test_candidate_cutoffs_empty_is_empty() -> None:
    assert cr.candidate_cutoffs([], []) == []


def test_pick_target_fpr_lookalike_default_respects_budget() -> None:
    positives, negatives = _pick_data()
    choice = cr.pick_threshold(positives, negatives, "target-fpr")  # 0.05, look-alike
    # Only cutoffs above both Persians (>= 0.7) keep look-alike FPR within 0.05;
    # the lowest such midpoint is 0.8, which keeps recall = 0.5 (only the 0.9 pos).
    assert choice.policy == "target-fpr"
    assert choice.row.cutoff == pytest.approx(0.8)
    assert choice.row.fpr_lookalike == pytest.approx(0.0)
    assert choice.row.recall == pytest.approx(0.5)
    assert "look-alike" in choice.rationale


def test_pick_target_fpr_looser_budget_lowers_cutoff_for_more_recall() -> None:
    positives, negatives = _pick_data()
    choice = cr.pick_threshold(positives, negatives, "target-fpr", target_fpr=0.5)
    # FPR(look-alike) = 0.5 is now within budget at cutoff 0.55, recall climbs to 1.0.
    assert choice.row.cutoff == pytest.approx(0.55)
    assert choice.row.fpr_lookalike == pytest.approx(0.5)
    assert choice.row.recall == pytest.approx(1.0)


def test_pick_target_fpr_overall_group() -> None:
    positives, negatives = _pick_data()
    choice = cr.pick_threshold(
        positives, negatives, "target-fpr", target_fpr=0.5, target_group="overall"
    )
    # Overall FPR drops to 1/3 at cutoff 0.55 (only the 0.7 Persian clears it).
    assert choice.row.cutoff == pytest.approx(0.55)
    assert choice.row.fpr_overall == pytest.approx(1 / 3)
    assert "overall" in choice.rationale


def test_pick_youdens_j_maximises_recall_minus_fpr() -> None:
    positives, negatives = _pick_data()
    choice = cr.pick_threshold(positives, negatives, "youdens-j")
    # J peaks at 0.55: recall 1.0 - FPR(all) 1/3 = 0.667, beating every other cutoff.
    assert choice.row.cutoff == pytest.approx(0.55)


def test_pick_youdens_j_ties_break_to_higher_cutoff() -> None:
    positives = [
        cr.ScoredImage("p0", 0.9, "g0", None),
        cr.ScoredImage("p1", 0.3, "g0", None),
    ]
    negatives = [
        cr.ScoredImage("n0", 0.1, "g0", "Abyssinian"),
        cr.ScoredImage("n1", 0.7, "g0", "Abyssinian"),
    ]
    # J = 0.5 at both cutoff 0.2 and 0.8; the tie-break prefers the higher (fewer FPs).
    choice = cr.pick_threshold(positives, negatives, "youdens-j")
    assert choice.row.cutoff == pytest.approx(0.8)


def test_pick_equal_error_balances_fpr_and_miss_rate() -> None:
    positives, negatives = _pick_data()
    choice = cr.pick_threshold(positives, negatives, "equal-error")
    # |FPR - (1-recall)| is smallest (0.167) at cutoff 0.65.
    assert choice.row.cutoff == pytest.approx(0.65)


def test_pick_threshold_empty_groups_raise() -> None:
    neg = [cr.ScoredImage("n", 0.3, "g0", "Persian")]
    pos = [cr.ScoredImage("p", 0.8, "g0", None)]
    with pytest.raises(ValueError, match="positive"):
        cr.pick_threshold([], neg, "youdens-j")
    with pytest.raises(ValueError, match="negative"):
        cr.pick_threshold(pos, [], "youdens-j")


def test_pick_target_fpr_lookalike_without_lookalikes_raises() -> None:
    positives = [cr.ScoredImage("p", 0.8, "g0", None)]
    negatives = [cr.ScoredImage("Abyssinian_1.jpg", 0.5, "g0", "Abyssinian")]
    with pytest.raises(ValueError, match="look-alike"):
        cr.pick_threshold(positives, negatives, "target-fpr")


def test_build_report_includes_chosen_threshold_only_when_given() -> None:
    positives, negatives = _pick_data()
    without = cr.build_report("m.yaml", 5, positives, negatives, "max")
    assert "Chosen threshold" not in without
    choice = cr.pick_threshold(positives, negatives, "youdens-j")
    with_choice = cr.build_report(
        "m.yaml", 5, positives, negatives, "max", choice=choice
    )
    assert "Chosen threshold (policy=youdens-j)" in with_choice
    assert "cutoff 0.550" in with_choice


# --------------------------------------------------------------------------- #
# render_report_html / write_report_html
# --------------------------------------------------------------------------- #


def _html_fixture(tmp_path: Path) -> tuple[str, Path, Path]:
    """Render an HTML report over synthetic data; return (html, indy_dir, html_path).

    Image files need not exist -- only ``src`` strings are generated -- so the dirs
    are plain tmp_path subdirs.
    """
    indy_dir = tmp_path / "images" / "indy"
    oxford_dir = tmp_path / "images" / "oxford"
    html_path = tmp_path / "data" / "reports" / "r.html"
    positives = [
        cr.ScoredImage("indy_a.jpg", 0.8, "g0.jpg", None),
        cr.ScoredImage("indy_b.jpg", 0.6, "g1.jpg", None),
    ]
    negatives = [
        cr.ScoredImage("Persian_1.jpg", 0.5, "g0.jpg", "Persian"),
        cr.ScoredImage("Abyssinian_1.jpg", 0.2, "g1.jpg", "Abyssinian"),
    ]
    document = cr.render_report_html(
        "m.yaml",
        ["g0.jpg", "g1.jpg"],
        positives,
        negatives,
        "max",
        html_path=html_path,
        indy_image_dir=indy_dir,
        oxford_image_dir=oxford_dir,
    )
    return document, indy_dir, html_path


def test_render_report_html_has_sections_tables_and_risk_list(tmp_path: Path) -> None:
    document, _, _ = _html_fixture(tmp_path)
    assert document.startswith("<!DOCTYPE html>")
    assert "<h2>Score distribution</h2>" in document
    assert "<h2>Overlap</h2>" in document
    assert "<h2>Per-breed negative scores</h2>" in document
    assert "<h2>Threshold sweep</h2>" in document
    assert "<h2>Per-breed FPR by cutoff</h2>" in document
    assert "<h2>Gallery</h2>" in document
    assert "<table>" in document
    assert '<ol class="risks">' in document


def test_render_report_html_image_alt_and_visible_caption(tmp_path: Path) -> None:
    document, _, _ = _html_fixture(tmp_path)
    # The highest negative shows both alt= and a visible figcaption of its name,
    # plus its best-match gallery photo.
    assert 'alt="Persian_1.jpg"' in document
    assert "<figcaption>Persian_1.jpg</figcaption>" in document
    assert 'alt="g0.jpg"' in document


def test_render_report_html_src_is_relative_and_forward_slashed(tmp_path: Path) -> None:
    document, _, _ = _html_fixture(tmp_path)
    # An Oxford candidate resolves under the oxford dir, its best-match under indy;
    # paths are relative (climb out of data/reports) and use forward slashes.
    assert 'src="../../images/oxford/Persian_1.jpg"' in document
    assert 'src="../../images/indy/g0.jpg"' in document
    assert "\\" not in document


def test_render_report_html_lists_every_gallery_photo(tmp_path: Path) -> None:
    document, _, _ = _html_fixture(tmp_path)
    assert 'src="../../images/indy/g0.jpg"' in document
    assert 'src="../../images/indy/g1.jpg"' in document


def test_render_report_html_chosen_threshold_section(tmp_path: Path) -> None:
    positives = [cr.ScoredImage("indy_a.jpg", 0.9, "g0.jpg", None)]
    negatives = [cr.ScoredImage("Persian_1.jpg", 0.5, "g0.jpg", "Persian")]
    choice = cr.pick_threshold(positives, negatives, "youdens-j")
    html_path = tmp_path / "data" / "reports" / "r.html"
    without = cr.render_report_html(
        "m.yaml",
        ["g0.jpg"],
        positives,
        negatives,
        "max",
        html_path=html_path,
        indy_image_dir=tmp_path / "images" / "indy",
        oxford_image_dir=tmp_path / "images" / "oxford",
    )
    assert "<h2>Chosen threshold</h2>" not in without
    with_choice = cr.render_report_html(
        "m.yaml",
        ["g0.jpg"],
        positives,
        negatives,
        "max",
        html_path=html_path,
        choice=choice,
        indy_image_dir=tmp_path / "images" / "indy",
        oxford_image_dir=tmp_path / "images" / "oxford",
    )
    assert "<h2>Chosen threshold</h2>" in with_choice
    assert "<code>youdens-j</code>" in with_choice


def test_write_report_html_roundtrips(tmp_path: Path) -> None:
    out = tmp_path / "reports" / "r.html"
    cr.write_report_html(
        out,
        "m.yaml",
        ["g0.jpg"],
        [cr.ScoredImage("indy_a.jpg", 0.8, "g0.jpg", None)],
        [cr.ScoredImage("Persian_1.jpg", 0.5, "g0.jpg", "Persian")],
        "max",
        indy_image_dir=tmp_path / "images" / "indy",
        oxford_image_dir=tmp_path / "images" / "oxford",
    )
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


# --------------------------------------------------------------------------- #
# report_common: shared rendering primitives (the evaluate report will reuse)
# --------------------------------------------------------------------------- #


def test_fmt_and_fmt_html_handle_nan_and_finite() -> None:
    assert report_common.fmt(float("nan")) == "  -  "
    assert report_common.fmt(0.1234) == "0.123"
    assert report_common.fmt_html(float("nan")) == "&ndash;"
    assert report_common.fmt_html(0.1234) == "0.123"


def test_scoped_table_empty_corner_is_blank_header() -> None:
    table = report_common.scoped_table(["a", "b"], [("row1", ["1", "2"])], corner="")
    # Empty corner -> a blank, unscoped <th>; column headers carry scope="col".
    assert table.startswith("<table><thead><tr><th></th>")
    assert '<th scope="col">a</th><th scope="col">b</th>' in table
    # Body row: a scope="row" header then the cells, in order, as <td>.
    assert '<tbody><tr><th scope="row">row1</th><td>1</td><td>2</td></tr></tbody>' in (
        table
    )


def test_scoped_table_labelled_corner_gets_col_scope() -> None:
    table = report_common.scoped_table(["x"], [("r", ["v"])], corner="cutoff")
    # A non-empty corner labels the row-header column with its own scope="col".
    assert '<tr><th scope="col">cutoff</th><th scope="col">x</th>' in table


def test_scoped_table_does_not_escape_pre_rendered_cells() -> None:
    # Cells are trusted HTML (formatter output / entities); the builder must not
    # double-escape them.
    table = report_common.scoped_table(["c"], [("r", ["&ndash;"])])
    assert "<td>&ndash;</td>" in table
    assert "&amp;ndash;" not in table


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
