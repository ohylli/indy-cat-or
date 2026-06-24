"""Smoke tests for the cat-breeds HTML evaluation report renderer.

Render-level only: call ``render_catbreeds_report_html`` with hand-built
``ScoredImage`` lists (no cache/manifest fixtures, no GPU), mirroring
``test_evaluate``'s HTML-renderer tests. Covers the cat-breeds-specific shape:
the NFC-only rate row, the look-alike framing, the *absence* of a drift table,
the false-positive image cap with its count note, and the nested per-breed
image-path resolver.
"""

from __future__ import annotations

from pathlib import Path

from calibration.artifact import (
    ARTIFACT_FORMAT_VERSION,
    CalibrationArtifact,
    ChosenBy,
    EmbeddingIdentity,
    GalleryImageRef,
    MetricsAtThreshold,
)
from calibration.evaluate_catbreeds_report_html import (
    MAX_FP_FIGURES,
    render_catbreeds_report_html,
)
from calibration.metrics import ScoredImage

NFC = "Norwegian Forest Cat"


def make_artifact(threshold: float = 0.5) -> CalibrationArtifact:
    return CalibrationArtifact(
        format_version=ARTIFACT_FORMAT_VERSION,
        threshold=threshold,
        aggregation="max",
        comparison=">=",
        embedding=EmbeddingIdentity(
            model_id="facebook/dinov2-base",
            embedding_dim=8,
            detect=True,
            margin=0.1,
            min_confidence=0.25,
        ),
        gallery_vectors_file="m.gallery.npy",
        gallery_fingerprint="sha256:deadbeef",
        gallery_count=2,
        gallery_images=[
            GalleryImageRef(n, "lying", "3q") for n in ("g0.jpg", "g1.jpg")
        ],
        chosen_by=ChosenBy(
            manifest="data/splits/m.yaml",
            seed=123,
            policy="target-fpr",
            target_fpr=0.05,
            target_fpr_group="look-alike",
        ),
        metrics_at_threshold=MetricsAtThreshold(
            fpr_all=0.01,
            fpr_look_alike=0.048,
            fpr_easy=0.0,
            recall_indy=1.0,
            n_pos=10,
            n_neg=700,
        ),
        aggregation_comparison=[],
        winner="max",
        sweep=[],
    )


def _scored() -> tuple[list[ScoredImage], list[ScoredImage], dict[str, str]]:
    """Positives all clear the bar; 25 negatives clear it (> MAX_FP_FIGURES FPs).

    NFC negatives score highest so they land inside the shown cap; the rest spread
    across a look-alike and a non-look-alike breed.
    """
    positives = [
        ScoredImage("t0.jpg", 0.9, "g0.jpg", None),
        ScoredImage("t1.jpg", 0.8, "g1.jpg", None),
    ]
    negatives: list[ScoredImage] = []
    breeds: dict[str, str] = {}
    # 5 NFC false positives at the top (highest scores).
    for i in range(5):
        name = f"nfc_{i}.jpg"
        negatives.append(ScoredImage(name, 0.95, "g0.jpg", NFC))
        breeds[name] = NFC
    # 20 more false positives split between a look-alike and a shorthair breed.
    for i in range(20):
        name = f"other_{i}.jpg"
        breed = "Maine Coon" if i % 2 else "Abyssinian"
        negatives.append(ScoredImage(name, 0.6, "g1.jpg", breed))
        breeds[name] = breed
    return positives, negatives, breeds


def _render(tmp_path: Path) -> str:
    positives, negatives, breeds = _scored()
    return render_catbreeds_report_html(
        "art.yaml",
        "the cat-breeds dataset",
        make_artifact(),
        positives,
        negatives,
        breeds,
        html_path=tmp_path / "eval.html",
    )


def test_has_nfc_row_and_lookalike_framing(tmp_path: Path) -> None:
    document = _render(tmp_path)
    assert "FPR (NFC only)" in document
    assert "FPR (long-haired)" in document
    assert "Recall (Indy)" in document
    assert "the real exam" in document  # the unseen-exam framing
    assert NFC in document  # NFC appears in the per-breed table


def test_has_scoped_tables(tmp_path: Path) -> None:
    document = _render(tmp_path)
    assert "<table>" in document
    assert '<th scope="col">' in document
    assert '<th scope="row">' in document


def test_per_breed_table_has_cats_column(tmp_path: Path) -> None:
    document = _render(tmp_path)
    assert '<th scope="col">cats</th>' in document  # the new per-breed cat-count column
    # NFC has 5 negatives in the fixture; its count cell appears in the table.
    assert "<td>5</td>" in document


def test_no_drift_table(tmp_path: Path) -> None:
    document = _render(tmp_path)
    assert "Generalization" not in document
    assert "recall_indy" not in document  # a drift-table row label


def test_false_positive_cap_and_note(tmp_path: Path) -> None:
    document = _render(tmp_path)
    # Exactly MAX_FP_FIGURES figure-rows render (figure_list rows open with <li><p>;
    # the summary <ul> items do not). All positives clear the bar -> no FN list.
    assert document.count("<li><p>") == MAX_FP_FIGURES
    assert f"+{25 - MAX_FP_FIGURES} more not shown" in document
    assert "None: every Indy photo cleared the bar." in document


def test_nested_resolver_and_no_backslash(tmp_path: Path) -> None:
    document = _render(tmp_path)
    # The candidate src resolves through the <breed>/ subpath...
    assert f"{NFC}/nfc_0.jpg" in document
    # ...while the caption/alt stays the bare filename.
    assert 'alt="nfc_0.jpg"' in document
    # Forward-slash src even on Windows.
    assert "\\" not in document
