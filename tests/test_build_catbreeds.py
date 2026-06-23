"""Tests for the cat-breeds (Kaggle ma7555) negatives builder.

Covers the pure, network-free helpers with a tiny fake ``images/<breed>/`` tree
in ``tmp_path``: the breed listing, the seeded per-breed sample, the catalog and
metadata writers, and -- the discipline that keeps this Petfinder scrape's broken
JPEGs out of the cache -- the corrupt-image skip path through ``detect_crop_stream``.
``ensure_dataset`` is a thin kagglehub wrapper and is not exercised here.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image

import build_catbreeds_negatives as builder
from _common import BASE_METADATA_COLUMNS
from build_catbreeds_negatives import (
    CatbreedsRow,
    CatImage,
    detect_crop_stream,
    list_cat_images,
    sample_per_breed,
    write_catalog,
    write_metadata,
)


def make_image(path: Path, *, width: int = 8, height: int = 8) -> None:
    """Write a tiny valid JPEG at ``path`` (parents created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), (123, 80, 40)).save(path)


def build_tree(root: Path, layout: dict[str, list[str]]) -> None:
    """Create ``root/images/<breed>/<file>.jpg`` for every breed -> files entry."""
    for breed, files in layout.items():
        for name in files:
            make_image(root / "images" / breed / name)


# --------------------------------------------------------------------------- #
# list_cat_images
# --------------------------------------------------------------------------- #


def test_list_cat_images_derives_breed_and_sorts(tmp_path: Path) -> None:
    build_tree(
        tmp_path,
        {
            "Norwegian Forest Cat": ["b.jpg", "a.jpg"],
            "Persian": ["p.jpg"],
        },
    )
    cats = list_cat_images(tmp_path)

    # Breed is the parent folder; source_filename is the bare name; sorted by
    # (breed, source_filename).
    assert [(c.breed, c.source_filename) for c in cats] == [
        ("Norwegian Forest Cat", "a.jpg"),
        ("Norwegian Forest Cat", "b.jpg"),
        ("Persian", "p.jpg"),
    ]
    # The path points at the located image, under images/<breed>/.
    assert cats[0].path == tmp_path / "images" / "Norwegian Forest Cat" / "a.jpg"


def test_list_cat_images_ignores_non_dirs(tmp_path: Path) -> None:
    build_tree(tmp_path, {"Persian": ["p.jpg"]})
    # A stray file directly under images/ must not be mistaken for a breed folder.
    (tmp_path / "images" / "README.txt").write_text("not a breed", encoding="utf-8")
    cats = list_cat_images(tmp_path)
    assert [c.breed for c in cats] == ["Persian"]


# --------------------------------------------------------------------------- #
# sample_per_breed
# --------------------------------------------------------------------------- #


def _named(breed: str, n: int) -> list[CatImage]:
    return [
        CatImage(Path(f"{breed}/{i}.jpg"), f"{breed}_{i}.jpg", breed) for i in range(n)
    ]


def test_sample_per_breed_caps_each_breed(tmp_path: Path) -> None:
    cats = _named("Persian", 5) + _named("Siberian", 2)
    sampled = sample_per_breed(cats, per_breed_limit=3, seed=20240601)
    by_breed: dict[str, int] = {}
    for c in sampled:
        by_breed[c.breed] = by_breed.get(c.breed, 0) + 1
    assert by_breed == {"Persian": 3, "Siberian": 2}  # cap respected; under-cap whole


def test_sample_per_breed_zero_and_negative_are_unlimited() -> None:
    cats = _named("Persian", 5)
    assert sample_per_breed(cats, 0, seed=1) == cats
    assert sample_per_breed(cats, -1, seed=1) == cats


def test_sample_per_breed_is_deterministic_and_resorted() -> None:
    cats = _named("Persian", 10) + _named("Maine Coon", 10)
    first = sample_per_breed(cats, 4, seed=20240601)
    second = sample_per_breed(cats, 4, seed=20240601)
    assert first == second  # same seed -> identical pick
    # The kept rows are re-sorted by (breed, source_filename) regardless of shuffle.
    assert first == sorted(first, key=lambda c: (c.breed, c.source_filename))


def test_sample_per_breed_seed_changes_pick() -> None:
    cats = _named("Persian", 10)
    a = {c.source_filename for c in sample_per_breed(cats, 3, seed=1)}
    b = {c.source_filename for c in sample_per_breed(cats, 3, seed=2)}
    assert a != b  # different seed draws a different subset


# --------------------------------------------------------------------------- #
# write_catalog / write_metadata
# --------------------------------------------------------------------------- #


def test_write_catalog_lists_full_set(tmp_path: Path) -> None:
    cats = _named("Persian", 2) + _named("Siberian", 1)
    out = tmp_path / "catalog.csv"
    write_catalog(cats, out)
    rows = list(csv.reader(out.read_text(encoding="utf-8").splitlines()))
    assert rows[0] == ["source_filename", "breed"]
    assert {(r[0], r[1]) for r in rows[1:]} == {
        ("Persian_0.jpg", "Persian"),
        ("Persian_1.jpg", "Persian"),
        ("Siberian_0.jpg", "Siberian"),
    }


def test_write_metadata_has_breed_column_aligned_by_index(tmp_path: Path) -> None:
    rows = [
        CatbreedsRow("a.jpg", "Persian", None),
        CatbreedsRow("b.jpg", "Siberian", None),
    ]
    out = tmp_path / "metadata.csv"
    write_metadata(rows, out)
    table = list(csv.reader(out.read_text(encoding="utf-8").splitlines()))
    assert table[0] == [*BASE_METADATA_COLUMNS, "breed"]
    # One row per embedding, aligned by index; trailing breed cell present.
    assert table[1][0] == "0" and table[1][1] == "a.jpg" and table[1][-1] == "Persian"
    assert table[2][0] == "1" and table[2][1] == "b.jpg" and table[2][-1] == "Siberian"


# --------------------------------------------------------------------------- #
# detect_crop_stream: the corrupt-image and detector-miss skip discipline
# --------------------------------------------------------------------------- #


def test_corrupt_image_is_counted_and_skipped(tmp_path: Path) -> None:
    good = tmp_path / "images" / "Persian" / "good.jpg"
    make_image(good)
    broken = tmp_path / "images" / "Persian" / "broken.jpg"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_bytes(b"not an image")  # PIL raises UnidentifiedImageError (OSError)

    cats = [
        CatImage(good, "good.jpg", "Persian"),
        CatImage(broken, "broken.jpg", "Persian"),
    ]
    misses = [0]
    corrupt = [0]
    # detector=None -> full-frame path, so no YOLO weights are needed and the only
    # skip a broken file can take is the corrupt branch.
    out = list(detect_crop_stream(cats, None, 0.1, misses, corrupt))

    assert corrupt == [1]  # the broken file is counted...
    assert misses == [0]
    kept = [item for item in out if item is not None]
    assert len(kept) == 1  # ...and never embedded
    row, image = kept[0]
    assert row.source_filename == "good.jpg"
    assert isinstance(image, Image.Image)
    assert out[1] is None  # the broken file yields a skip in stream order


def test_detector_miss_is_counted_and_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    img = tmp_path / "images" / "Persian" / "p.jpg"
    make_image(img)
    cats = [CatImage(img, "p.jpg", "Persian")]
    misses = [0]
    corrupt = [0]
    # Stub the detector + the detect_and_crop call so no real YOLO is loaded; an
    # empty result is a no-cat miss.
    monkeypatch.setattr(builder, "detect_and_crop", lambda image, det, margin: [])
    out = list(detect_crop_stream(cats, object(), 0.1, misses, corrupt))  # type: ignore[arg-type]

    assert misses == [1]
    assert corrupt == [0]
    assert out == [None]  # nothing embedded
