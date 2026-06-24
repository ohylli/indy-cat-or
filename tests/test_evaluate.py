"""Tests for the evaluation tool (``calibration.evaluate`` + its E0 renderers).

Covers the honest-grade path with synthetic data so it never touches the
gitignored real data or a GPU: the ``confusion_at`` count helper, the disjointness
guard and the empty-test guard, the run flow under a monkeypatched embeddings
cache, and the text/HTML renderers' section/table substrings (mirroring
``test_calibration_report``). The artifact load/validation itself is covered in
``test_artifact``; here we build artifacts directly to control the frozen numbers.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from _common import EmbeddingsMeta, EmbeddingsVariant, write_embeddings_meta
from calibration import evaluate
from calibration import manifest as manifest_mod
from calibration.artifact import (
    ARTIFACT_FORMAT_VERSION,
    CalibrationArtifact,
    ChosenBy,
    EmbeddingIdentity,
    GalleryImageRef,
    MetricsAtThreshold,
)
from calibration.evaluate_report_html import render_report_html
from calibration.evaluate_report_text import build_report, write_scores_csv
from calibration.manifest import (
    MANIFEST_FORMAT_VERSION,
    EmbeddingProvenance,
    GenerationParams,
    SplitConfigError,
    SplitManifest,
)
from calibration.metrics import ScoredImage, confusion_at, select_error_rows

GALLERY = ["g0.jpeg", "g1.jpeg"]
INDY_TEST = ["t0.jpeg", "t1.jpeg"]
OXFORD_TEST = ["Persian_5.jpg", "Ragdoll_5.jpg"]

#: The baseline variant the artifact's frozen embedding identity dictates; the
#: fake test caches are written into its dir + sidecar so evaluate's variant
#: resolution + cross-check pass.
BASELINE = EmbeddingsVariant(model_id="facebook/dinov2-base", detect=True, margin=0.1)
EMBED_DIM = 8


def make_embedding_identity(**overrides: object) -> EmbeddingIdentity:
    """The artifact's frozen embedding block (baseline variant by default)."""
    defaults: dict[str, object] = {
        "model_id": "facebook/dinov2-base",
        "embedding_dim": EMBED_DIM,
        "detect": True,
        "margin": 0.1,
        "min_confidence": 0.25,
    }
    defaults.update(overrides)
    return EmbeddingIdentity(**defaults)  # type: ignore[arg-type]


def make_artifact(
    *,
    gallery: list[str] = GALLERY,
    threshold: float = 0.5,
    embedding: EmbeddingIdentity | None = None,
) -> CalibrationArtifact:
    return CalibrationArtifact(
        format_version=ARTIFACT_FORMAT_VERSION,
        threshold=threshold,
        aggregation="max",
        comparison=">=",
        embedding=embedding if embedding is not None else make_embedding_identity(),
        gallery_vectors_file="m.gallery.npy",
        gallery_fingerprint="sha256:deadbeef",
        gallery_count=len(gallery),
        gallery_images=[GalleryImageRef(n, "lying", "3q") for n in gallery],
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


def make_manifest(
    *,
    gallery: list[str] = GALLERY,
    indy_test: list[str] = INDY_TEST,
    oxford_test: list[str] = OXFORD_TEST,
) -> SplitManifest:
    params = GenerationParams(
        strategy="three_way",
        seed=123,
        gallery=len(gallery),
        calibration=2,
        test=len(indy_test),
        oxford_test_fraction=0.3,
        prefer=None,
    )
    return SplitManifest(
        format_version=MANIFEST_FORMAT_VERSION,
        params=params,
        embedding=EmbeddingProvenance(
            model_id="facebook/dinov2-base",
            detect=True,
            margin=0.1,
            min_confidence=0.25,
        ),
        generated_at="2026-01-01T00:00:00+00:00",
        random_seed_drawn=False,
        indy_gallery=list(gallery),
        indy_calibration=["c0.jpeg", "c1.jpeg"],
        indy_test=list(indy_test),
        oxford_setup=["Persian_1.jpg", "Ragdoll_1.jpg"],
        oxford_test=list(oxford_test),
        oxford_setup_breed_counts={"Persian": 1, "Ragdoll": 1},
        oxford_test_breed_counts={"Persian": 1, "Ragdoll": 1},
    )


def _write_metadata_csv(path: Path, names: list[str], *, breeds: bool) -> None:
    """Write a minimal embeddings ``metadata.csv`` (loader reads source_filename)."""
    columns = [
        "row",
        "source_filename",
        "detect_used",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "area_fraction",
    ]
    if breeds:
        columns = [*columns, "breed"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for i, name in enumerate(names):
            base = [i, name, True, "0.9", 0, 0, 1, 1, "0.5"]
            writer.writerow([*base, name.split("_")[0]] if breeds else base)


def baseline_sidecar(row_count: int, **overrides: object) -> EmbeddingsMeta:
    """A sidecar matching the baseline variant, with optional field overrides."""
    defaults: dict[str, object] = {
        "format_version": 1,
        "model_id": "facebook/dinov2-base",
        "embedding_dim": EMBED_DIM,
        "normalized": False,
        "detect": True,
        "margin": 0.1,
        "min_confidence": 0.25,
        "row_count": row_count,
    }
    defaults.update(overrides)
    return EmbeddingsMeta(**defaults)  # type: ignore[arg-type]


def write_variant_cache(
    out_dir: Path, names: list[str], *, breeds: bool, meta: EmbeddingsMeta
) -> None:
    """Write a full variant cache: metadata.csv + embeddings.npy + the sidecar."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_metadata_csv(out_dir / "metadata.csv", names, breeds=breeds)
    rng = np.random.default_rng(len(names))
    vectors = rng.standard_normal((len(names), EMBED_DIM)).astype(np.float32)
    np.save(out_dir / "embeddings.npy", vectors)
    write_embeddings_meta(meta, out_dir)


@pytest.fixture
def fake_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point evaluate at synthetic baseline-variant test caches under a tmp root.

    Writes real Indy + Oxford variant dirs (metadata.csv + embeddings.npy +
    sidecar) matching the artifact's frozen embedding block, so the full
    ``load_embeddings_variant`` + variant cross-check + scoring path runs without a
    GPU, real weights, or the gitignored data. Returns the tmp embeddings root so a
    test can overwrite a sidecar to force the artifact-vs-cache mismatch.
    """
    embeddings_root = tmp_path / "embeddings"
    monkeypatch.setattr(manifest_mod, "EMBEDDINGS_ROOT", embeddings_root)

    write_variant_cache(
        BASELINE.dir(embeddings_root / "indy"),
        INDY_TEST,
        breeds=False,
        meta=baseline_sidecar(len(INDY_TEST)),
    )
    write_variant_cache(
        BASELINE.dir(embeddings_root / "oxford"),
        OXFORD_TEST,
        breeds=True,
        meta=baseline_sidecar(len(OXFORD_TEST)),
    )
    return embeddings_root


def raw_gallery() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.standard_normal((len(GALLERY), 8)).astype(np.float32)


# --------------------------------------------------------------------------- #
# confusion_at
# --------------------------------------------------------------------------- #


def test_confusion_at_counts() -> None:
    positives = [ScoredImage("a", 0.9, "g", None), ScoredImage("b", 0.4, "g", None)]
    negatives = [
        ScoredImage("x", 0.6, "g", "Persian"),
        ScoredImage("y", 0.2, "g", "Beagle"),
        ScoredImage("z", 0.5, "g", "Ragdoll"),
    ]
    c = confusion_at(positives, negatives, 0.5)
    assert (c.tp, c.fn) == (1, 1)  # 0.9 >= 0.5, 0.4 < 0.5
    assert (c.fp, c.tn) == (2, 1)  # 0.6 and 0.5 clear; 0.2 does not


# --------------------------------------------------------------------------- #
# run_evaluation flow + guards
# --------------------------------------------------------------------------- #


def test_run_evaluation_prints_grade(
    fake_cache: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    evaluate.run_evaluation(
        make_artifact(), raw_gallery(), make_manifest(), "art.yaml", "m.yaml", None
    )
    out = capsys.readouterr().out
    assert "Confusion at the frozen threshold" in out
    assert "Recall (Indy)" in out
    assert "Generalization (calibration vs test" in out
    assert "fpr_look_alike" in out  # the drift table maps the artifact's key
    assert "NOT the unseen-breed exam" in out  # honest labeling


def test_run_evaluation_writes_html(
    fake_cache: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    html_out = tmp_path / "reports" / "eval.html"
    evaluate.run_evaluation(
        make_artifact(), raw_gallery(), make_manifest(), "art.yaml", "m.yaml", html_out
    )
    assert html_out.exists()
    assert "HTML report written to" in capsys.readouterr().out


def test_disjointness_guard_is_loud(fake_cache: Path) -> None:
    # A manifest whose gallery differs from the artifact's frozen gallery.
    bad = make_manifest(gallery=["other0.jpeg", "other1.jpeg"])
    with pytest.raises(SplitConfigError, match="different experiments"):
        evaluate.run_evaluation(
            make_artifact(), raw_gallery(), bad, "art.yaml", "m.yaml", None
        )


def test_empty_test_is_loud(fake_cache: Path) -> None:
    empty = make_manifest(indy_test=[], oxford_test=[])
    with pytest.raises(SplitConfigError, match="nothing to grade"):
        evaluate.run_evaluation(
            make_artifact(), raw_gallery(), empty, "art.yaml", "m.yaml", None
        )


def test_artifact_vs_cache_variant_mismatch_is_loud(
    fake_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Re-stamp the Indy test cache's sidecar with a different model than the
    # artifact's frozen embedding identity records. Evaluate has no --model flag,
    # so the variant is dictated by the artifact -- a drift must fail loudly.
    indy_dir = BASELINE.dir(fake_cache / "indy")
    write_embeddings_meta(
        baseline_sidecar(len(INDY_TEST), model_id="facebook/dinov2-large"), indy_dir
    )
    with pytest.raises(SplitConfigError, match="different footing"):
        evaluate.run_evaluation(
            make_artifact(), raw_gallery(), make_manifest(), "art.yaml", "m.yaml", None
        )


# --------------------------------------------------------------------------- #
# CLI wiring (main)
# --------------------------------------------------------------------------- #


def test_main_resolves_missing_manifest_loudly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    art = make_artifact()
    monkeypatch.setattr(evaluate, "load_artifact", lambda p: (art, raw_gallery()))
    with pytest.raises(SystemExit, match="manifest not found"):
        # The artifact's recorded chosen_by.manifest (data/splits/m.yaml) is not
        # resolvable from the test cwd and no --manifest override is given.
        evaluate.main(["--artifact", str(tmp_path / "a.yaml")])


def test_main_end_to_end_writes_html(
    fake_cache: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    art = make_artifact()
    manifest_file = tmp_path / "m.yaml"
    manifest_file.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(evaluate, "load_artifact", lambda p: (art, raw_gallery()))
    monkeypatch.setattr(evaluate, "load_manifest", lambda p: make_manifest())
    html_out = tmp_path / "eval.html"
    evaluate.main(
        [
            "--artifact",
            str(tmp_path / "a.yaml"),
            "--manifest",
            str(manifest_file),
            "--html",
            str(html_out),
        ]
    )
    assert html_out.exists()


# --------------------------------------------------------------------------- #
# Renderers (direct, mirroring test_calibration_report's substring style)
# --------------------------------------------------------------------------- #


def _scored() -> tuple[list[ScoredImage], list[ScoredImage]]:
    positives = [
        ScoredImage("t0.jpeg", 0.8, "g0.jpeg", None),
        ScoredImage("t1.jpeg", 0.4, "g1.jpeg", None),
    ]
    negatives = [
        ScoredImage("Persian_5.jpg", 0.6, "g0.jpeg", "Persian"),
        ScoredImage("Ragdoll_5.jpg", 0.3, "g1.jpeg", "Ragdoll"),
    ]
    return positives, negatives


def test_text_report_has_all_sections() -> None:
    positives, negatives = _scored()
    report = build_report("art.yaml", "m.yaml", make_artifact(), positives, negatives)
    assert "Evaluation: art.yaml  on test set m.yaml" in report
    assert "Confusion at the frozen threshold:" in report
    assert "Rates at the frozen threshold:" in report
    assert "Per-breed FPR at the frozen threshold" in report
    assert "breeds sorted by FPR, highest first" in report  # the FPR-descending order
    assert "cats" in report  # the per-breed cat-count column
    assert "Generalization (calibration vs test" in report
    assert "recall_indy" in report  # a drift-table row


def test_html_report_has_scoped_tables_and_note(tmp_path: Path) -> None:
    positives, negatives = _scored()
    document = render_report_html(
        "art.yaml",
        "m.yaml",
        make_artifact(),
        positives,
        negatives,
        html_path=tmp_path / "eval.html",
    )
    assert "<h2>Confusion at the frozen threshold</h2>" in document
    assert "<h2>Generalization (calibration vs test)</h2>" in document
    assert "<table>" in document
    assert '<th scope="col">' in document
    assert '<th scope="row">' in document
    assert "NOT the unseen-breed exam" in document  # honest labeling
    assert "\\" not in document  # no backslash paths even on Windows


# --------------------------------------------------------------------------- #
# E1: error lists + scores CSV
# --------------------------------------------------------------------------- #


def test_select_error_rows_partitions_at_threshold() -> None:
    positives = [
        ScoredImage("p_hi.jpeg", 0.9, "g", None),
        ScoredImage("p_lo.jpeg", 0.3, "g", None),  # false negative
    ]
    negatives = [
        ScoredImage("Persian_5.jpg", 0.7, "g", "Persian"),  # false positive
        ScoredImage("Ragdoll_5.jpg", 0.6, "g", "Ragdoll"),  # false positive
        ScoredImage("Beagle_5.jpg", 0.2, "g", "Beagle"),
    ]
    false_pos, false_neg = select_error_rows(positives, negatives, 0.5)
    # FP = negatives >= threshold, highest first; FN = positives < threshold.
    assert [s.name for s in false_pos] == ["Persian_5.jpg", "Ragdoll_5.jpg"]
    assert [s.name for s in false_neg] == ["p_lo.jpeg"]


def test_text_report_has_error_lists() -> None:
    positives, negatives = _scored()
    # At threshold 0.5: t1.jpeg (0.4) is a false negative, Persian_5.jpg (0.6) a FP.
    report = build_report("art.yaml", "m.yaml", make_artifact(), positives, negatives)
    assert "False positives (negatives that cleared the bar):" in report
    assert "False negatives (Indy missed):" in report
    assert "Persian_5.jpg" in report  # the false positive is listed
    assert "t1.jpeg" in report  # the false negative is listed


def test_html_report_has_error_list_figures(tmp_path: Path) -> None:
    positives, negatives = _scored()
    document = render_report_html(
        "art.yaml",
        "m.yaml",
        make_artifact(),
        positives,
        negatives,
        html_path=tmp_path / "eval.html",
    )
    assert "<h2>False positives (negatives that cleared the bar)</h2>" in document
    assert "<h2>False negatives (Indy missed)</h2>" in document
    assert 'class="risks"' in document
    assert "<figure>" in document
    assert 'alt="Persian_5.jpg"' in document  # the FP crop is embedded
    assert "\\" not in document  # forward-slash src even on Windows


def test_write_scores_csv_has_verdict_column(tmp_path: Path) -> None:
    positives, negatives = _scored()
    out = tmp_path / "scores.csv"
    write_scores_csv(out, positives, negatives, 0.5)
    rows = list(csv.reader(out.read_text(encoding="utf-8").splitlines()))
    assert rows[0] == [
        "role",
        "source_filename",
        "score",
        "verdict",
        "best_match",
        "breed",
    ]
    by_name = {r[1]: r for r in rows[1:]}
    assert by_name["Persian_5.jpg"][3] == "Indy"  # 0.6 >= 0.5
    assert by_name["t1.jpeg"][3] == "not"  # 0.4 < 0.5
    assert by_name["Persian_5.jpg"][5] == "Persian"  # breed provenance
    assert by_name["t0.jpeg"][5] == ""  # positives carry no breed


def test_main_end_to_end_writes_scores_csv(
    fake_cache: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    art = make_artifact()
    manifest_file = tmp_path / "m.yaml"
    manifest_file.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(evaluate, "load_artifact", lambda p: (art, raw_gallery()))
    monkeypatch.setattr(evaluate, "load_manifest", lambda p: make_manifest())
    scores_out = tmp_path / "scores.csv"
    evaluate.main(
        [
            "--artifact",
            str(tmp_path / "a.yaml"),
            "--manifest",
            str(manifest_file),
            "--scores-out",
            str(scores_out),
        ]
    )
    assert scores_out.exists()
    header = scores_out.read_text(encoding="utf-8").splitlines()[0]
    assert "verdict" in header
