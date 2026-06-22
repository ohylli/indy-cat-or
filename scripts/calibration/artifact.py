"""The frozen calibration artifact (V3): freeze the chosen threshold + gallery.

The last calibrate stage of ``docs/calibration_design.md`` Sec. 5: serialize V2's
``ThresholdChoice`` into the artifact the live *decide* stage and the future
``evaluate.py`` consume. It is a **pair**, bundled so a deployed decide stage
needs nothing else:

* a human-readable ``calibration.yaml`` -- the operative fields decide reads, plus
  provenance and the trade-off curve that make a frozen number auditable;
* a companion ``<stem>.gallery.npy`` -- the **raw** (un-normalized) gallery
  vectors, row-aligned to the ``gallery.images`` list, so ``decision.Gallery``
  keeps L2-normalizing on construction and floats stay out of the YAML.

A ``sha256`` fingerprint over the raw vectors lives in the YAML, so an accidental
vector/threshold mismatch is a loud failure at load, not a silently-wrong verdict.

Writing the artifact runs the threshold pick under **both** aggregations (``max``
and ``mean-top3``) and records the comparison; the winner -- chosen FPR-first
(lowest ``fpr_look_alike`` at the chosen threshold, ``recall_indy`` breaking ties,
``max`` winning exact ties) -- becomes the operative ``aggregation``. This module
owns the artifact's data model, fingerprinting, build, and YAML/npy I/O only; the
threshold math is reused verbatim from ``calibration.metrics``.
"""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from calibration.manifest import (
    INDY_MAPPING,
    SplitConfigError,
    SplitManifest,
)
from calibration.metrics import (
    ScoredImage,
    SweepRow,
    ThresholdChoice,
    build_sweep,
    sweep_thresholds,
)
from indycat.decision import Aggregation

#: Asserted loudly at load, like the manifest's ``format_version``.
ARTIFACT_FORMAT_VERSION = 2

#: The companion vectors file is named off the YAML stem so the pair travels
#: together; the YAML stores only the basename, keeping it relocatable.
GALLERY_VECTORS_SUFFIX = ".gallery.npy"

#: The score >= threshold convention, recorded explicitly so a frozen artifact
#: never leaves it implicit.
COMPARISON = ">="


# --------------------------------------------------------------------------- #
# Fingerprint
# --------------------------------------------------------------------------- #


def gallery_fingerprint(raw_vectors: NDArray[np.float32]) -> str:
    """A ``sha256:`` fingerprint over the raw gallery vectors (loud drift check)."""
    data = np.ascontiguousarray(raw_vectors, dtype=np.float32).tobytes()
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GalleryImageRef:
    """One gallery photo: filename plus the ``mapping.csv`` attributes decide names."""

    source_filename: str
    position: str
    view: str


@dataclass(frozen=True)
class AggComparisonEntry:
    """One aggregation's chosen-threshold headline, for the max-vs-mean-top3 run."""

    aggregation: str
    threshold: float
    fpr_all: float
    fpr_look_alike: float
    recall_indy: float


@dataclass(frozen=True)
class MetricsAtThreshold:
    """What the frozen threshold buys: the chosen ``SweepRow`` plus the counts."""

    fpr_all: float
    fpr_look_alike: float
    fpr_easy: float
    recall_indy: float
    n_pos: int
    n_neg: int


@dataclass(frozen=True)
class EmbeddingIdentity:
    """The embedding variant the frozen gallery was built with.

    Operative: the predict app loads ``Embedder(model_id)`` and a detector at
    ``min_confidence`` / ``detect`` / ``margin`` from this, so the live
    detect->crop->embed matches the gallery's footing (invariant #2 of
    ``docs/embeddings_provenance.md``). Evaluate asserts the test caches' variant
    against it. ``margin``/``min_confidence`` are ``None`` when ``detect`` is off.
    """

    model_id: str
    embedding_dim: int
    detect: bool
    margin: float | None
    min_confidence: float | None


@dataclass(frozen=True)
class ChosenBy:
    """V2's rationale, serialized: how the threshold was picked."""

    manifest: str
    seed: int
    policy: str
    target_fpr: float
    target_fpr_group: str


@dataclass(frozen=True)
class CalibrationArtifact:
    """The frozen calibration result -- operative fields, provenance, and curve."""

    format_version: int
    # operative -- read by decide
    threshold: float
    aggregation: str
    comparison: str
    embedding: EmbeddingIdentity
    gallery_vectors_file: str
    gallery_fingerprint: str
    gallery_count: int
    gallery_images: list[GalleryImageRef]
    # provenance -- audit, not consumed at decide time
    chosen_by: ChosenBy
    metrics_at_threshold: MetricsAtThreshold
    aggregation_comparison: list[AggComparisonEntry]
    winner: str
    # the curve -- trade-off context for the frozen point
    sweep: list[SweepRow]


# --------------------------------------------------------------------------- #
# Loading gallery position/view (the columns load_indy_metadata ignores)
# --------------------------------------------------------------------------- #


def load_indy_positions(
    mapping_path: Path = INDY_MAPPING,
) -> dict[str, tuple[str, str]]:
    """Map each Indy ``new_filename`` to its ``(position, view)`` from mapping.csv.

    These let a verdict name the best-matching gallery photo's pose without
    ``mapping.csv`` at decide time (``docs/calibration_design.md`` Sec. 1).
    """
    positions: dict[str, tuple[str, str]] = {}
    with mapping_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            positions[row["new_filename"]] = (row["position"], row["view"])
    return positions


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

#: A per-aggregation calibration result: the scored positives/negatives and the
#: threshold V2 picked for that aggregation.
AggregationResult = tuple[list[ScoredImage], list[ScoredImage], ThresholdChoice]


def _winner_key(entry: AggComparisonEntry) -> tuple[float, float]:
    """FPR-first, recall tiebreak. NaN ``fpr_look_alike`` falls back to ``fpr_all``."""
    fpr = entry.fpr_all if math.isnan(entry.fpr_look_alike) else entry.fpr_look_alike
    return (fpr, -entry.recall_indy)


def _comparison_entry(aggregation: str, choice: ThresholdChoice) -> AggComparisonEntry:
    row = choice.row
    return AggComparisonEntry(
        aggregation=aggregation,
        threshold=row.cutoff,
        fpr_all=row.fpr_overall,
        fpr_look_alike=row.fpr_lookalike,
        recall_indy=row.recall,
    )


def build_artifact(
    manifest: SplitManifest,
    manifest_label: str,
    gallery_names: list[str],
    raw_gallery_vectors: NDArray[np.float32],
    positions: dict[str, tuple[str, str]],
    results: dict[Aggregation, AggregationResult],
    gallery_vectors_file: str,
    embedding: EmbeddingIdentity,
    *,
    policy: str,
    target_fpr: float,
    target_fpr_group: str,
    sweep_step: float,
) -> CalibrationArtifact:
    """Assemble the artifact: pick the winning aggregation and freeze its threshold.

    ``results`` holds the scored positives/negatives and ``ThresholdChoice`` for
    *both* aggregations. The winner is FPR-first (see :func:`_winner_key`); its
    chosen cutoff and aggregation become the operative threshold, and the curve
    is the V1 sweep under that same aggregation so the frozen point is on a
    visible trade-off. ``embedding`` records the variant the gallery was built
    with, so the live decide stage and evaluate can match the same footing.
    """
    # ``AGGREGATIONS`` order (max first) is the exact-tie break, so iterate it.
    entries = [
        _comparison_entry(agg, results[agg][2])
        for agg in ("max", "mean-top3")
        if agg in results
    ]
    winner_entry = min(entries, key=_winner_key)
    winner: Aggregation = winner_entry.aggregation  # type: ignore[assignment]

    win_pos, win_neg, win_choice = results[winner]
    row = win_choice.row
    sweep = build_sweep(
        win_pos, win_neg, sweep_thresholds(win_pos, win_neg, sweep_step)
    )

    missing = [name for name in gallery_names if name not in positions]
    if missing:
        raise SplitConfigError(f"gallery photos missing from {INDY_MAPPING}: {missing}")
    images = [
        GalleryImageRef(name, positions[name][0], positions[name][1])
        for name in gallery_names
    ]

    return CalibrationArtifact(
        format_version=ARTIFACT_FORMAT_VERSION,
        threshold=row.cutoff,
        aggregation=winner,
        comparison=COMPARISON,
        embedding=embedding,
        gallery_vectors_file=gallery_vectors_file,
        gallery_fingerprint=gallery_fingerprint(raw_gallery_vectors),
        gallery_count=len(gallery_names),
        gallery_images=images,
        chosen_by=ChosenBy(
            manifest=manifest_label,
            seed=manifest.params.seed,
            policy=policy,
            target_fpr=target_fpr,
            target_fpr_group=target_fpr_group,
        ),
        metrics_at_threshold=MetricsAtThreshold(
            fpr_all=row.fpr_overall,
            fpr_look_alike=row.fpr_lookalike,
            fpr_easy=row.fpr_easy,
            recall_indy=row.recall,
            n_pos=len(win_pos),
            n_neg=len(win_neg),
        ),
        aggregation_comparison=entries,
        winner=winner,
        sweep=sweep,
    )


# --------------------------------------------------------------------------- #
# YAML write / load
# --------------------------------------------------------------------------- #


def _sweep_row_to_dict(row: SweepRow) -> dict[str, float]:
    return {
        "cutoff": row.cutoff,
        "fpr_all": row.fpr_overall,
        "fpr_look_alike": row.fpr_lookalike,
        "fpr_easy": row.fpr_easy,
        "recall_indy": row.recall,
    }


def artifact_to_dict(artifact: CalibrationArtifact) -> dict[str, Any]:
    """Render the artifact as the ordered, YAML-serialisable mapping.

    Ordering carries the doc's operative / provenance / curve grouping (PyYAML's
    ``safe_dump`` cannot emit section comments).
    """
    return {
        "format_version": artifact.format_version,
        # operative -- read by decide
        "threshold": artifact.threshold,
        "aggregation": artifact.aggregation,
        "comparison": artifact.comparison,
        "embedding": {
            "model_id": artifact.embedding.model_id,
            "embedding_dim": artifact.embedding.embedding_dim,
            "detect": artifact.embedding.detect,
            "margin": artifact.embedding.margin,
            "min_confidence": artifact.embedding.min_confidence,
        },
        "gallery": {
            "vectors": artifact.gallery_vectors_file,
            "fingerprint": artifact.gallery_fingerprint,
            "count": artifact.gallery_count,
            "images": [
                {
                    "source_filename": img.source_filename,
                    "position": img.position,
                    "view": img.view,
                }
                for img in artifact.gallery_images
            ],
        },
        # provenance -- audit, not consumed at decide time
        "chosen_by": {
            "manifest": artifact.chosen_by.manifest,
            "seed": artifact.chosen_by.seed,
            "policy": artifact.chosen_by.policy,
            "target_fpr": artifact.chosen_by.target_fpr,
            "target_fpr_group": artifact.chosen_by.target_fpr_group,
        },
        "metrics_at_threshold": {
            "fpr_all": artifact.metrics_at_threshold.fpr_all,
            "fpr_look_alike": artifact.metrics_at_threshold.fpr_look_alike,
            "fpr_easy": artifact.metrics_at_threshold.fpr_easy,
            "recall_indy": artifact.metrics_at_threshold.recall_indy,
            "n_pos": artifact.metrics_at_threshold.n_pos,
            "n_neg": artifact.metrics_at_threshold.n_neg,
        },
        "aggregation_comparison": {
            **{
                entry.aggregation: {
                    "threshold": entry.threshold,
                    "fpr_all": entry.fpr_all,
                    "fpr_look_alike": entry.fpr_look_alike,
                    "recall_indy": entry.recall_indy,
                }
                for entry in artifact.aggregation_comparison
            },
            "winner": artifact.winner,
        },
        # the curve -- trade-off context
        "sweep": [_sweep_row_to_dict(row) for row in artifact.sweep],
    }


def write_artifact(
    artifact: CalibrationArtifact,
    raw_vectors: NDArray[np.float32],
    yaml_path: Path,
) -> Path:
    """Write the YAML + companion ``.gallery.npy`` pair; return the npy path.

    The companion's basename is taken verbatim from ``artifact.gallery_vectors_file``
    (the single source of truth, set by :func:`build_artifact`) and saved beside the
    YAML, so the YAML and the file it names can never drift apart.
    """
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    vectors_path = yaml_path.with_name(artifact.gallery_vectors_file)
    np.save(vectors_path, np.ascontiguousarray(raw_vectors, dtype=np.float32))
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            artifact_to_dict(artifact),
            f,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
    return vectors_path


def _artifact_from_dict(data: dict[str, Any]) -> CalibrationArtifact:
    gallery = data["gallery"]
    comparison_data = dict(data["aggregation_comparison"])
    winner = comparison_data.pop("winner")
    return CalibrationArtifact(
        format_version=data["format_version"],
        threshold=data["threshold"],
        aggregation=data["aggregation"],
        comparison=data["comparison"],
        embedding=EmbeddingIdentity(
            model_id=data["embedding"]["model_id"],
            embedding_dim=data["embedding"]["embedding_dim"],
            detect=data["embedding"]["detect"],
            margin=data["embedding"]["margin"],
            min_confidence=data["embedding"]["min_confidence"],
        ),
        gallery_vectors_file=gallery["vectors"],
        gallery_fingerprint=gallery["fingerprint"],
        gallery_count=gallery["count"],
        gallery_images=[
            GalleryImageRef(img["source_filename"], img["position"], img["view"])
            for img in gallery["images"]
        ],
        chosen_by=ChosenBy(
            manifest=data["chosen_by"]["manifest"],
            seed=data["chosen_by"]["seed"],
            policy=data["chosen_by"]["policy"],
            target_fpr=data["chosen_by"]["target_fpr"],
            target_fpr_group=data["chosen_by"]["target_fpr_group"],
        ),
        metrics_at_threshold=MetricsAtThreshold(
            fpr_all=data["metrics_at_threshold"]["fpr_all"],
            fpr_look_alike=data["metrics_at_threshold"]["fpr_look_alike"],
            fpr_easy=data["metrics_at_threshold"]["fpr_easy"],
            recall_indy=data["metrics_at_threshold"]["recall_indy"],
            n_pos=data["metrics_at_threshold"]["n_pos"],
            n_neg=data["metrics_at_threshold"]["n_neg"],
        ),
        aggregation_comparison=[
            AggComparisonEntry(
                aggregation=agg,
                threshold=entry["threshold"],
                fpr_all=entry["fpr_all"],
                fpr_look_alike=entry["fpr_look_alike"],
                recall_indy=entry["recall_indy"],
            )
            for agg, entry in comparison_data.items()
        ],
        winner=winner,
        sweep=[
            SweepRow(
                cutoff=row["cutoff"],
                fpr_overall=row["fpr_all"],
                fpr_lookalike=row["fpr_look_alike"],
                fpr_easy=row["fpr_easy"],
                recall=row["recall_indy"],
            )
            for row in data["sweep"]
        ],
    )


def load_artifact(
    yaml_path: Path,
) -> tuple[CalibrationArtifact, NDArray[np.float32]]:
    """Load + validate an artifact and its raw gallery vectors (the decide read path).

    Validation is loud, mirroring ``load_manifest``: a wrong ``format_version`` or
    a gallery fingerprint that disagrees with the companion ``.npy`` (an accidental
    vector/threshold mismatch) both raise ``SplitConfigError`` -- never a
    silently-wrong verdict.
    """
    with yaml_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    artifact = _artifact_from_dict(data)

    if artifact.format_version != ARTIFACT_FORMAT_VERSION:
        raise SplitConfigError(
            f"{yaml_path} has format_version {artifact.format_version}; "
            f"this tool writes/reads version {ARTIFACT_FORMAT_VERSION}"
        )

    vectors_path = yaml_path.with_name(artifact.gallery_vectors_file)
    raw_vectors: NDArray[np.float32] = np.load(vectors_path).astype(np.float32)
    actual = gallery_fingerprint(raw_vectors)
    if actual != artifact.gallery_fingerprint:
        raise SplitConfigError(
            f"{vectors_path} fingerprint {actual} disagrees with "
            f"{yaml_path} ({artifact.gallery_fingerprint}); the vectors and the "
            "frozen threshold no longer match"
        )
    if raw_vectors.shape[0] != artifact.gallery_count:
        raise SplitConfigError(
            f"{vectors_path} has {raw_vectors.shape[0]} rows but "
            f"{yaml_path} declares gallery.count {artifact.gallery_count}"
        )
    if raw_vectors.shape[1] != artifact.embedding.embedding_dim:
        raise SplitConfigError(
            f"{vectors_path} vectors are {raw_vectors.shape[1]}-dimensional but "
            f"{yaml_path} declares embedding.embedding_dim "
            f"{artifact.embedding.embedding_dim}; the gallery vectors and the "
            "frozen embedding identity no longer match"
        )
    return artifact, raw_vectors
