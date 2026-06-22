"""Tests for the V3 calibration artifact (``calibration.artifact``).

Covers the pure pieces in isolation -- the gallery fingerprint, the FPR-first
aggregation winner, and the YAML+npy round-trip with its loud format/fingerprint
validation -- with synthetic data, so they never touch the gitignored real data.
The CLI wiring (the ``--artifact`` flag) is exercised in ``test_calibrate``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from calibration.artifact import (
    ARTIFACT_FORMAT_VERSION,
    AggregationResult,
    CalibrationArtifact,
    EmbeddingIdentity,
    build_artifact,
    gallery_fingerprint,
    load_artifact,
    write_artifact,
)
from calibration.manifest import (
    EmbeddingProvenance,
    GenerationParams,
    SplitConfigError,
    SplitManifest,
)
from calibration.metrics import ScoredImage, SweepRow, ThresholdChoice
from indycat.decision import Aggregation


def make_manifest() -> SplitManifest:
    params = GenerationParams(
        strategy="three_way",
        seed=12345,
        gallery=2,
        calibration=2,
        test=0,
        oxford_test_fraction=0.3,
        prefer=None,
    )
    return SplitManifest(
        format_version=2,
        params=params,
        embedding=EmbeddingProvenance(
            model_id="facebook/dinov2-base",
            detect=True,
            margin=0.1,
            min_confidence=0.25,
        ),
        generated_at="2026-01-01T00:00:00+00:00",
        random_seed_drawn=False,
        indy_gallery=["g0", "g1"],
        indy_calibration=["c0", "c1"],
        indy_test=[],
        oxford_setup=["Persian_1", "Ragdoll_1"],
        oxford_test=[],
        oxford_setup_breed_counts={"Persian": 1, "Ragdoll": 1},
        oxford_test_breed_counts={},
    )


def make_results(
    fpr_look_by_agg: dict[Aggregation, float],
    recall_by_agg: dict[Aggregation, float] | None = None,
) -> dict[Aggregation, AggregationResult]:
    """A per-aggregation result bundle with controllable chosen-row metrics."""
    positives = [
        ScoredImage("c0", 0.9, "g0", None),
        ScoredImage("c1", 0.7, "g1", None),
    ]
    negatives = [
        ScoredImage("Persian_1", 0.4, "g0", "Persian"),
        ScoredImage("Ragdoll_1", 0.3, "g1", "Ragdoll"),
    ]
    results: dict[Aggregation, AggregationResult] = {}
    for agg, fpr_look in fpr_look_by_agg.items():
        recall = 1.0 if recall_by_agg is None else recall_by_agg[agg]
        row = SweepRow(
            cutoff=0.5,
            fpr_overall=fpr_look,
            fpr_lookalike=fpr_look,
            fpr_easy=0.0,
            recall=recall,
        )
        choice = ThresholdChoice(policy="target-fpr", row=row, rationale="x")
        results[agg] = (positives, negatives, choice)
    return results


def make_embedding(embedding_dim: int = 3) -> EmbeddingIdentity:
    """The variant identity stamped into the artifact's embedding block."""
    return EmbeddingIdentity(
        model_id="facebook/dinov2-base",
        embedding_dim=embedding_dim,
        detect=True,
        margin=0.1,
        min_confidence=0.25,
    )


def build_default_artifact(
    results: dict[Aggregation, AggregationResult],
    raw_vectors: NDArray[np.float32] | None = None,
    embedding: EmbeddingIdentity | None = None,
) -> CalibrationArtifact:
    if raw_vectors is None:
        raw_vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    if embedding is None:
        embedding = make_embedding(raw_vectors.shape[1])
    return build_artifact(
        make_manifest(),
        "data/splits/m.yaml",
        ["g0", "g1"],
        raw_vectors,
        {"g0": ("lying", "3q"), "g1": ("sitting", "side")},
        results,
        "m.gallery.npy",
        embedding,
        policy="target-fpr",
        target_fpr=0.05,
        target_fpr_group="look-alike",
        sweep_step=0.05,
    )


# --------------------------------------------------------------------------- #
# Fingerprint
# --------------------------------------------------------------------------- #


def test_fingerprint_is_deterministic_and_sensitive() -> None:
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    assert gallery_fingerprint(a) == gallery_fingerprint(a.copy())
    assert gallery_fingerprint(a).startswith("sha256:")
    b = a.copy()
    b[0, 0] += 0.001
    assert gallery_fingerprint(a) != gallery_fingerprint(b)


# --------------------------------------------------------------------------- #
# Winner selection
# --------------------------------------------------------------------------- #


def test_winner_is_lower_fpr_lookalike() -> None:
    results = make_results({"max": 0.048, "mean-top3": 0.061})
    artifact = build_default_artifact(results)
    assert artifact.winner == "max"
    assert artifact.aggregation == "max"
    # mean-top3 would win if it had the lower FPR instead.
    flipped = make_results({"max": 0.061, "mean-top3": 0.048})
    assert build_default_artifact(flipped).winner == "mean-top3"


def test_recall_breaks_fpr_ties() -> None:
    results = make_results(
        {"max": 0.05, "mean-top3": 0.05},
        recall_by_agg={"max": 0.9, "mean-top3": 1.0},
    )
    assert build_default_artifact(results).winner == "mean-top3"


def test_exact_tie_prefers_max() -> None:
    results = make_results(
        {"max": 0.05, "mean-top3": 0.05},
        recall_by_agg={"max": 1.0, "mean-top3": 1.0},
    )
    assert build_default_artifact(results).winner == "max"


def test_operative_fields_track_the_winner() -> None:
    results = make_results({"max": 0.048, "mean-top3": 0.061})
    artifact = build_default_artifact(results)
    assert artifact.comparison == ">="
    assert artifact.threshold == 0.5  # the winner row's cutoff
    assert artifact.metrics_at_threshold.n_pos == 2
    assert artifact.metrics_at_threshold.n_neg == 2
    assert artifact.gallery_count == 2
    assert [img.source_filename for img in artifact.gallery_images] == ["g0", "g1"]
    assert artifact.gallery_images[0].position == "lying"
    assert artifact.sweep  # the V1 curve is present


def test_missing_gallery_position_is_loud() -> None:
    results = make_results({"max": 0.048, "mean-top3": 0.061})
    with pytest.raises(SplitConfigError, match="missing from"):
        build_artifact(
            make_manifest(),
            "m.yaml",
            ["g0", "g1"],
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            {"g0": ("lying", "3q")},  # g1 absent
            results,
            "m.gallery.npy",
            make_embedding(2),
            policy="target-fpr",
            target_fpr=0.05,
            target_fpr_group="look-alike",
            sweep_step=0.05,
        )


# --------------------------------------------------------------------------- #
# Round-trip and loud validation
# --------------------------------------------------------------------------- #


def test_write_then_load_round_trips(tmp_path: Path) -> None:
    raw = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
    results = make_results({"max": 0.048, "mean-top3": 0.061})
    artifact = build_default_artifact(results, raw_vectors=raw)
    yaml_path = tmp_path / "calibration-test.yaml"

    vectors_path = write_artifact(artifact, raw, yaml_path)
    assert yaml_path.exists()
    assert vectors_path.exists()
    assert vectors_path.name == "m.gallery.npy"  # the artifact's vectors basename

    loaded, loaded_vectors = load_artifact(yaml_path)
    assert loaded.format_version == ARTIFACT_FORMAT_VERSION
    assert loaded.threshold == artifact.threshold
    assert loaded.aggregation == artifact.aggregation
    assert loaded.winner == artifact.winner
    assert loaded.gallery_fingerprint == artifact.gallery_fingerprint
    assert np.array_equal(loaded_vectors, raw)
    assert [i.source_filename for i in loaded.gallery_images] == ["g0", "g1"]
    assert len(loaded.aggregation_comparison) == 2


def test_fingerprint_mismatch_is_loud(tmp_path: Path) -> None:
    raw = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    results = make_results({"max": 0.048, "mean-top3": 0.061})
    artifact = build_default_artifact(
        results, raw_vectors=np.array([[1.0, 0.0]], dtype=np.float32)
    )
    yaml_path = tmp_path / "c.yaml"
    vectors_path = write_artifact(artifact, raw, yaml_path)
    # Tamper with the saved vectors so the fingerprint no longer matches.
    np.save(vectors_path, raw + 1.0)
    with pytest.raises(SplitConfigError, match="fingerprint"):
        load_artifact(yaml_path)


def test_format_version_mismatch_is_loud(tmp_path: Path) -> None:
    raw = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    results = make_results({"max": 0.048, "mean-top3": 0.061})
    artifact = build_default_artifact(results, raw_vectors=raw)
    yaml_path = tmp_path / "c.yaml"
    write_artifact(artifact, raw, yaml_path)
    text = yaml_path.read_text(encoding="utf-8").replace(
        "format_version: 2", "format_version: 99"
    )
    yaml_path.write_text(text, encoding="utf-8")
    with pytest.raises(SplitConfigError, match="format_version"):
        load_artifact(yaml_path)


def test_format_version_is_two() -> None:
    assert ARTIFACT_FORMAT_VERSION == 2


def test_embedding_block_round_trips(tmp_path: Path) -> None:
    raw = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
    results = make_results({"max": 0.048, "mean-top3": 0.061})
    embedding = EmbeddingIdentity(
        model_id="facebook/dinov2-with-registers-base",
        embedding_dim=3,
        detect=True,
        margin=0.2,
        min_confidence=0.3,
    )
    artifact = build_default_artifact(results, raw_vectors=raw, embedding=embedding)
    yaml_path = tmp_path / "c.yaml"
    write_artifact(artifact, raw, yaml_path)
    loaded, _ = load_artifact(yaml_path)
    assert loaded.embedding == embedding


def test_embedding_dim_vs_vector_width_cross_check_is_loud(tmp_path: Path) -> None:
    # The artifact declares embedding_dim 2 but the vectors are 3-wide.
    raw = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    results = make_results({"max": 0.048, "mean-top3": 0.061})
    artifact = build_default_artifact(
        results, raw_vectors=raw, embedding=make_embedding(2)
    )
    yaml_path = tmp_path / "c.yaml"
    write_artifact(artifact, raw, yaml_path)
    with pytest.raises(SplitConfigError, match="embedding_dim"):
        load_artifact(yaml_path)
