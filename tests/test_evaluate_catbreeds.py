"""Tests for the cat-breeds evaluation tool (``calibration.evaluate_catbreeds``).

A sibling of ``test_evaluate.py``: same synthetic-cache style (no GPU, no weights,
no gitignored data), but the negative source is a whole **cat-breeds** variant
cache rather than the manifest's Oxford ``test`` role. Covers the honest-grade
run, the HTML/scores outputs, and -- the loud guards that stop a silently-wrong
grade -- the artifact-vs-cache variant cross-check, the same-experiment guard, and
the empty-cache / empty-positives failures. A focused ``build_sweep`` test pins
the custom ``lookalike_breeds`` partition the cat-breeds report depends on.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from _common import EmbeddingsMeta, EmbeddingsVariant, write_embeddings_meta
from calibration import evaluate_catbreeds
from calibration import manifest as manifest_mod
from calibration.artifact import (
    ARTIFACT_FORMAT_VERSION,
    CalibrationArtifact,
    ChosenBy,
    EmbeddingIdentity,
    GalleryImageRef,
    MetricsAtThreshold,
)
from calibration.manifest import (
    MANIFEST_FORMAT_VERSION,
    EmbeddingProvenance,
    GenerationParams,
    SplitConfigError,
    SplitManifest,
)
from calibration.metrics import LOOKALIKE_BREEDS, ScoredImage, build_sweep

GALLERY = ["g0.jpeg", "g1.jpeg"]
INDY_TEST = ["t0.jpeg", "t1.jpeg"]

#: Cat-breeds negatives, named ``<breed>__<n>.jpg`` so the synthetic metadata can
#: carry a breed column. One NFC (the headline look-alike), one other long-haired
#: look-alike, one shorthair -- enough to exercise per-breed/NFC bucketing.
CATBREEDS = ["Norwegian Forest Cat__0.jpg", "Siberian__0.jpg", "Bengal__0.jpg"]
CATBREEDS_BREEDS = ["Norwegian Forest Cat", "Siberian", "Bengal"]

#: The baseline variant the artifact's frozen embedding identity dictates.
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
        oxford_test=["Persian_5.jpg", "Ragdoll_5.jpg"],
        oxford_setup_breed_counts={"Persian": 1, "Ragdoll": 1},
        oxford_test_breed_counts={"Persian": 1, "Ragdoll": 1},
    )


def _write_metadata_csv(
    path: Path, names: list[str], *, breeds: list[str] | None
) -> None:
    """Write a minimal embeddings ``metadata.csv`` (loader reads source_filename).

    When ``breeds`` is given, a trailing ``breed`` column carries each row's label
    (the cat-breeds cache shape that the breed join reads).
    """
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
    if breeds is not None:
        columns = [*columns, "breed"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for i, name in enumerate(names):
            base = [i, name, True, "0.9", 0, 0, 1, 1, "0.5"]
            writer.writerow([*base, breeds[i]] if breeds is not None else base)


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
    out_dir: Path,
    names: list[str],
    *,
    breeds: list[str] | None,
    meta: EmbeddingsMeta,
) -> None:
    """Write a full variant cache: metadata.csv + embeddings.npy + the sidecar."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_metadata_csv(out_dir / "metadata.csv", names, breeds=breeds)
    rng = np.random.default_rng(len(names) + 1)
    vectors = rng.standard_normal((len(names), EMBED_DIM)).astype(np.float32)
    np.save(out_dir / "embeddings.npy", vectors)
    write_embeddings_meta(meta, out_dir)


@pytest.fixture
def fake_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the eval at synthetic baseline-variant Indy + cat-breeds caches.

    Writes real Indy + cat-breeds variant dirs (metadata.csv + embeddings.npy +
    sidecar) matching the artifact's frozen embedding block, so the full
    ``load_embeddings_variant`` + variant cross-check + scoring path runs without a
    GPU or the gitignored data. Returns the tmp embeddings root so a test can
    overwrite a sidecar to force the artifact-vs-cache mismatch.
    """
    embeddings_root = tmp_path / "embeddings"
    monkeypatch.setattr(manifest_mod, "EMBEDDINGS_ROOT", embeddings_root)

    write_variant_cache(
        BASELINE.dir(embeddings_root / "indy"),
        INDY_TEST,
        breeds=None,
        meta=baseline_sidecar(len(INDY_TEST)),
    )
    write_variant_cache(
        BASELINE.dir(embeddings_root / "catbreeds"),
        CATBREEDS,
        breeds=CATBREEDS_BREEDS,
        meta=baseline_sidecar(len(CATBREEDS)),
    )
    return embeddings_root


def raw_gallery() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.standard_normal((len(GALLERY), EMBED_DIM)).astype(np.float32)


# --------------------------------------------------------------------------- #
# run_evaluation flow
# --------------------------------------------------------------------------- #


def test_run_evaluation_prints_grade(
    fake_cache: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    evaluate_catbreeds.run_evaluation(
        make_artifact(),
        raw_gallery(),
        make_manifest(),
        "art.yaml",
        "the cat-breeds dataset",
        None,
    )
    out = capsys.readouterr().out
    assert "Cat-breeds evaluation: art.yaml" in out
    assert "Confusion at the frozen threshold:" in out
    assert "Recall (Indy):" in out
    assert "FPR (NFC only):" in out  # the NFC breakout
    assert "Per-breed FPR at the frozen threshold" in out
    # The whole cat-breeds cache is the negative pool.
    assert f"Negatives: {len(CATBREEDS)} cat-breeds cats" in out


def test_run_evaluation_writes_html(
    fake_cache: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    html_out = tmp_path / "reports" / "eval.html"
    evaluate_catbreeds.run_evaluation(
        make_artifact(),
        raw_gallery(),
        make_manifest(),
        "art.yaml",
        "the cat-breeds dataset",
        html_out,
    )
    assert html_out.exists()
    assert "HTML report written to" in capsys.readouterr().out


def test_run_evaluation_writes_scores_csv(fake_cache: Path, tmp_path: Path) -> None:
    scores_out = tmp_path / "scores.csv"
    evaluate_catbreeds.run_evaluation(
        make_artifact(),
        raw_gallery(),
        make_manifest(),
        "art.yaml",
        "the cat-breeds dataset",
        None,
        scores_out,
    )
    assert scores_out.exists()
    header = scores_out.read_text(encoding="utf-8").splitlines()[0]
    assert "verdict" in header


# --------------------------------------------------------------------------- #
# Loud guards
# --------------------------------------------------------------------------- #


def test_artifact_vs_cache_variant_mismatch_is_loud(
    fake_cache: Path,
) -> None:
    # Re-stamp the cat-breeds cache's sidecar with a different model than the
    # artifact records. The eval has no --model flag, so the variant is dictated
    # by the artifact -- a drift must fail loudly.
    cb_dir = BASELINE.dir(fake_cache / "catbreeds")
    write_embeddings_meta(
        baseline_sidecar(len(CATBREEDS), model_id="facebook/dinov2-large"), cb_dir
    )
    with pytest.raises(SplitConfigError, match="different footing"):
        evaluate_catbreeds.run_evaluation(
            make_artifact(),
            raw_gallery(),
            make_manifest(),
            "art.yaml",
            "the cat-breeds dataset",
            None,
        )


def test_same_experiment_guard_is_loud(fake_cache: Path) -> None:
    # A manifest whose gallery differs from the artifact's frozen gallery.
    bad = make_manifest(gallery=["other0.jpeg", "other1.jpeg"])
    with pytest.raises(SplitConfigError, match="different experiments"):
        evaluate_catbreeds.run_evaluation(
            make_artifact(),
            raw_gallery(),
            bad,
            "art.yaml",
            "the cat-breeds dataset",
            None,
        )


def test_empty_catbreeds_cache_is_loud(
    fake_cache: Path,
) -> None:
    # Overwrite the cat-breeds cache with a 0-row one (header only + empty .npy).
    cb_dir = BASELINE.dir(fake_cache / "catbreeds")
    write_variant_cache(cb_dir, [], breeds=[], meta=baseline_sidecar(0))
    with pytest.raises(SplitConfigError, match="is empty"):
        evaluate_catbreeds.run_evaluation(
            make_artifact(),
            raw_gallery(),
            make_manifest(),
            "art.yaml",
            "the cat-breeds dataset",
            None,
        )


def test_empty_indy_test_is_loud(fake_cache: Path) -> None:
    empty = make_manifest(indy_test=[])
    with pytest.raises(SplitConfigError, match="Indy test role is empty"):
        evaluate_catbreeds.run_evaluation(
            make_artifact(),
            raw_gallery(),
            empty,
            "art.yaml",
            "the cat-breeds dataset",
            None,
        )


# --------------------------------------------------------------------------- #
# CLI wiring (main)
# --------------------------------------------------------------------------- #


def test_main_end_to_end_writes_html(
    fake_cache: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    art = make_artifact()
    manifest_file = tmp_path / "m.yaml"
    manifest_file.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(
        evaluate_catbreeds, "load_artifact", lambda p: (art, raw_gallery())
    )
    monkeypatch.setattr(evaluate_catbreeds, "load_manifest", lambda p: make_manifest())
    html_out = tmp_path / "eval.html"
    evaluate_catbreeds.main(
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


def test_main_resolves_missing_manifest_loudly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    art = make_artifact()
    monkeypatch.setattr(
        evaluate_catbreeds, "load_artifact", lambda p: (art, raw_gallery())
    )
    with pytest.raises(SystemExit, match="manifest not found"):
        # The artifact's recorded chosen_by.manifest is not resolvable here and no
        # --manifest override is given.
        evaluate_catbreeds.main(["--artifact", str(tmp_path / "a.yaml")])


# --------------------------------------------------------------------------- #
# metrics: custom lookalike_breeds partition (what the cat-breeds report relies on)
# --------------------------------------------------------------------------- #


def test_build_sweep_custom_lookalike_partition() -> None:
    positives = [ScoredImage("t0", 0.9, "g", None)]
    negatives = [
        ScoredImage("nfc", 0.8, "g", "Norwegian Forest Cat"),  # look-alike
        ScoredImage("bengal", 0.6, "g", "Bengal"),  # easy
    ]
    custom = frozenset({"Norwegian Forest Cat"})
    row = build_sweep(positives, negatives, [0.7], lookalike_breeds=custom)[0]
    assert row.fpr_overall == 0.5  # 1 of 2 negatives clears 0.7
    assert row.fpr_lookalike == 1.0  # the NFC (0.8) clears
    assert row.fpr_easy == 0.0  # Bengal (0.6) does not


def test_build_sweep_default_lookalike_is_oxford() -> None:
    # With the default set, an Oxford look-alike counts in fpr_lookalike and a
    # cat-breeds breed name does not -- the default is unchanged.
    assert "Persian" in LOOKALIKE_BREEDS
    assert "Norwegian Forest Cat" not in LOOKALIKE_BREEDS
    negatives = [
        ScoredImage("p", 0.8, "g", "Persian"),
        ScoredImage("nfc", 0.8, "g", "Norwegian Forest Cat"),
    ]
    row = build_sweep([ScoredImage("t", 0.9, "g", None)], negatives, [0.7])[0]
    assert row.fpr_lookalike == 1.0  # only the Persian is a default look-alike
    assert row.fpr_easy == 1.0  # the NFC falls in the easy bucket by default
