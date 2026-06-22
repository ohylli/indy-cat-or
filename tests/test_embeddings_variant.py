"""Tests for the embedding-provenance variant + sidecar layer in ``_common``.

All pure and fast -- string/path composition and tiny tmp npy/CSV/YAML files;
no model, no real (gitignored) data. They cover the slugs (incl. the ``:g``
margin format and the ``nocrop`` case), ``EmbeddingsVariant`` path composition,
the sidecar write->read round-trip (including the detect-off ``null`` writes),
``variant_key`` normalization, and ``load_embeddings_variant``'s happy path plus
its two loud failures (missing sidecar, sidecar-vs-npy row drift).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import yaml

from _common import (
    BASE_METADATA_COLUMNS,
    SIDECAR_NAME,
    EmbeddingsMeta,
    EmbeddingsVariant,
    crop_slug,
    load_embeddings_variant,
    model_slug,
    read_embeddings_meta,
    write_embeddings_meta,
)

# --------------------------------------------------------------------------- #
# Slugs
# --------------------------------------------------------------------------- #


def test_model_slug_replaces_slash() -> None:
    assert model_slug("facebook/dinov2-base") == "facebook--dinov2-base"


def test_model_slug_no_slash_unchanged() -> None:
    assert model_slug("dinov2-base") == "dinov2-base"


def test_crop_slug_detect_on_formats_margin_g() -> None:
    # :g keeps 0.1 as "0.1" rather than "0.100000".
    assert crop_slug(True, 0.1) == "crop-m0.1"
    assert crop_slug(True, 0.25) == "crop-m0.25"
    assert crop_slug(True, 0.0) == "crop-m0"


def test_crop_slug_detect_off_is_nocrop() -> None:
    # Margin is irrelevant under detect-off and must be omitted.
    assert crop_slug(False, 0.1) == "nocrop"
    assert crop_slug(False, 0.9) == "nocrop"


# --------------------------------------------------------------------------- #
# EmbeddingsVariant path composition
# --------------------------------------------------------------------------- #


def test_variant_subdir_crop_on() -> None:
    variant = EmbeddingsVariant("facebook/dinov2-base", detect=True, margin=0.1)
    assert variant.subdir() == Path("facebook--dinov2-base") / "crop-m0.1"


def test_variant_subdir_detect_off() -> None:
    variant = EmbeddingsVariant("facebook/dinov2-base", detect=False, margin=0.1)
    assert variant.subdir() == Path("facebook--dinov2-base") / "nocrop"


def test_variant_dir_under_dataset_root() -> None:
    variant = EmbeddingsVariant("facebook/dinov2-base", detect=True, margin=0.1)
    root = Path("data") / "embeddings" / "indy"
    assert variant.dir(root) == root / "facebook--dinov2-base" / "crop-m0.1"


# --------------------------------------------------------------------------- #
# Sidecar write -> read round-trip
# --------------------------------------------------------------------------- #


def test_sidecar_round_trip_crop_on(tmp_path: Path) -> None:
    meta = EmbeddingsMeta(
        format_version=1,
        model_id="facebook/dinov2-base",
        embedding_dim=768,
        normalized=False,
        detect=True,
        margin=0.1,
        min_confidence=0.25,
        row_count=33,
    )
    path = write_embeddings_meta(meta, tmp_path)
    assert path == tmp_path / SIDECAR_NAME
    assert read_embeddings_meta(tmp_path) == meta


def test_sidecar_detect_off_writes_null_margin_and_min_confidence(
    tmp_path: Path,
) -> None:
    # Even if the caller hands in stale margin/min_confidence, a nocrop sidecar
    # must not advertise them: they are written as null on disk.
    meta = EmbeddingsMeta(
        format_version=1,
        model_id="facebook/dinov2-base",
        embedding_dim=768,
        normalized=False,
        detect=False,
        margin=0.1,
        min_confidence=0.25,
        row_count=10,
    )
    write_embeddings_meta(meta, tmp_path)

    raw = yaml.safe_load((tmp_path / SIDECAR_NAME).read_text(encoding="utf-8"))
    assert raw["margin"] is None
    assert raw["min_confidence"] is None
    assert raw["format_version"] == 1

    # Read back: the loud-on-disk null becomes None on the model too.
    read_back = read_embeddings_meta(tmp_path)
    assert read_back.margin is None
    assert read_back.min_confidence is None
    assert read_back.detect is False


# --------------------------------------------------------------------------- #
# variant_key normalization
# --------------------------------------------------------------------------- #


def test_variant_key_crop_on_carries_margin_and_min_confidence() -> None:
    meta = EmbeddingsMeta(
        format_version=1,
        model_id="facebook/dinov2-base",
        embedding_dim=768,
        normalized=False,
        detect=True,
        margin=0.1,
        min_confidence=0.25,
        row_count=33,
    )
    assert meta.variant_key() == ("facebook/dinov2-base", True, 0.1, 0.25)


def test_variant_key_nocrop_ignores_margin_and_min_confidence() -> None:
    # Two nocrop metas with different recorded margins compare equal.
    a = EmbeddingsMeta(
        format_version=1,
        model_id="facebook/dinov2-base",
        embedding_dim=768,
        normalized=False,
        detect=False,
        margin=0.1,
        min_confidence=0.25,
        row_count=10,
    )
    b = EmbeddingsMeta(
        format_version=1,
        model_id="facebook/dinov2-base",
        embedding_dim=768,
        normalized=False,
        detect=False,
        margin=0.9,
        min_confidence=0.5,
        row_count=10,
    )
    assert a.variant_key() == ("facebook/dinov2-base", False, None, None)
    assert a.variant_key() == b.variant_key()


# --------------------------------------------------------------------------- #
# load_embeddings_variant
# --------------------------------------------------------------------------- #


def write_variant_cache(
    out_dir: Path, names: list[str], dim: int = 4, row_count: int | None = None
) -> EmbeddingsMeta:
    """Write a tiny metadata.csv + embeddings.npy + sidecar in ``out_dir``.

    ``row_count`` defaults to ``len(names)`` (a consistent cache); pass a
    different value to drive the sidecar-vs-npy drift error.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(BASE_METADATA_COLUMNS)
        for i, name in enumerate(names):
            writer.writerow([i, name, True, "0.9", 0, 0, 1, 1, "0.5"])
    vectors = np.arange(len(names) * dim, dtype=np.float32).reshape(len(names), dim)
    np.save(out_dir / "embeddings.npy", vectors)
    meta = EmbeddingsMeta(
        format_version=1,
        model_id="facebook/dinov2-base",
        embedding_dim=dim,
        normalized=False,
        detect=True,
        margin=0.1,
        min_confidence=0.25,
        row_count=len(names) if row_count is None else row_count,
    )
    write_embeddings_meta(meta, out_dir)
    return meta


def test_load_embeddings_variant_happy_path(tmp_path: Path) -> None:
    write_variant_cache(tmp_path, ["a.jpg", "b.jpg"])
    names, vectors, meta = load_embeddings_variant(tmp_path)
    assert names == ["a.jpg", "b.jpg"]
    assert vectors.shape == (2, 4)
    assert vectors.dtype == np.float32
    assert meta.model_id == "facebook/dinov2-base"
    assert meta.row_count == 2


def test_load_embeddings_variant_missing_sidecar_raises(tmp_path: Path) -> None:
    # A consistent metadata/npy pair but no sidecar at all.
    write_variant_cache(tmp_path, ["a.jpg", "b.jpg"])
    (tmp_path / SIDECAR_NAME).unlink()
    with pytest.raises(FileNotFoundError, match="no embeddings sidecar"):
        load_embeddings_variant(tmp_path)


def test_load_embeddings_variant_row_count_mismatch_raises(tmp_path: Path) -> None:
    # metadata/npy agree (2 rows) but the sidecar claims 3 -> loud drift error.
    write_variant_cache(tmp_path, ["a.jpg", "b.jpg"], row_count=3)
    with pytest.raises(ValueError, match="drifted apart"):
        load_embeddings_variant(tmp_path)
