"""Build the Oxford-IIIT Pet negatives: download -> detect -> crop -> embed.

The negatives never enter the live decision (best-match-vs-threshold against
Indy's gallery). Their job is calibration and evaluation: locating the cutoff
where Indy and non-Indy score distributions separate, and measuring the
false-positive rate on look-alike long-haired cats. For those numbers to mean
anything the negatives must be produced on *identical footing* to the gallery --
same loader, same YOLO re-detection (Oxford's shipped boxes are head-only ROIs
and are ignored), same frozen DINOv2, same raw/un-normalized vectors. So this
script mirrors ``build_indy_gallery.py`` and shares its helpers via ``_common``.

torchvision is used only to fetch + md5-verify the archives into
``--data-dir/oxford-iiit-pet/``; iteration is done over ``annotations/list.txt``
ourselves, which yields *all* cat images across both paper splits (the dataset
object would load only one split, skip EXIF, and hide filenames). Cats are rows
with species 1; the breed is the image-id prefix (e.g. ``Maine_Coon``).

Two stages, selected by ``--download-only``:
    download  ensure the data is present, write ``catalog.csv`` (every cat image
              -> breed) and print per-breed counts -- a text check that the 12
              breeds at ~200 each are there, without opening any image.
    embed     (default) also detect+crop+embed every cat and write, into the
              resolved ``--out-dir``, the row-aligned ``embeddings.npy`` +
              ``metadata.csv`` (the Indy columns plus a ``breed`` column) and
              the per-variant ``embeddings.meta.yaml`` provenance sidecar (the
              source of truth the folder name only mirrors; see
              ``docs/embeddings_provenance.md``). ``catalog.csv`` stays beside
              the raw dataset -- it describes the dataset, not a given run.

When ``--out-dir`` is omitted it is derived as
``data/embeddings/oxford/<model_slug>/<crop_slug>`` so each (model, crop)
variant lands in its own subdir and large vectors can never drop into the base
folder by accident.

Like the gallery builder, the top-confidence crop is the one kept per image; an
image with no cat detected is counted and skipped, never embedded as a full
frame. ``--no-detect`` embeds full frames instead (the detect toggle).

Usage:
    uv run python scripts/build_oxford_negatives.py --download-only
    uv run python scripts/build_oxford_negatives.py --limit 16
    uv run python scripts/build_oxford_negatives.py
    uv run python scripts/build_oxford_negatives.py --no-detect
"""

import argparse
import csv
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.datasets import OxfordIIITPet

from _common import (
    BASE_METADATA_COLUMNS,
    EmbeddingsMeta,
    EmbeddingsVariant,
    base_metadata_cells,
    embed_stream,
    load_image,
    write_embeddings_meta,
)
from indycat.detection import CatDetector, Detection, detect_and_crop
from indycat.embedding import Embedder

REPO_ROOT = Path(__file__).parent.parent

#: SPECIES column value for cats in Oxford's annotation files (2 = dog).
CAT_SPECIES = "1"


@dataclass(frozen=True)
class CatImage:
    """One Oxford cat image: where it is and what breed it is labelled."""

    path: Path
    source_filename: str
    breed: str


@dataclass
class OxfordRow:
    """Provenance for one embedding, with the Oxford breed label."""

    source_filename: str
    breed: str
    detection: Detection | None


def list_cat_images(data_dir: Path) -> list[CatImage]:
    """All cat images from ``annotations/list.txt``, sorted for reproducibility.

    ``list.txt`` rows are ``image_id class_id species breed_id`` with ``#``
    comment lines; cats are species 1. The breed is the image-id prefix.
    """
    base = data_dir / "oxford-iiit-pet"
    images_folder = base / "images"
    cats: list[CatImage] = []
    with (base / "annotations" / "list.txt").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            image_id, _class_id, species, _breed_id = line.split()
            if species != CAT_SPECIES:
                continue
            cats.append(
                CatImage(
                    path=images_folder / f"{image_id}.jpg",
                    source_filename=f"{image_id}.jpg",
                    breed=image_id.rsplit("_", 1)[0],
                )
            )
    cats.sort(key=lambda c: (c.breed, c.source_filename))
    return cats


def write_catalog(cats: list[CatImage], csv_path: Path) -> None:
    """The reviewable cat-image -> breed listing (the full set, pre-detection)."""
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_filename", "breed"])
        for cat in cats:
            writer.writerow([cat.source_filename, cat.breed])


def print_breed_counts(cats: list[CatImage]) -> None:
    """Per-breed counts -- a text check that the 12 breeds at ~200 each exist."""
    counts = Counter(cat.breed for cat in cats)
    print(f"{len(counts)} cat breeds, {len(cats)} images total:")
    for breed, count in sorted(counts.items()):
        print(f"  {breed}: {count}")


def write_metadata(rows: list[OxfordRow], csv_path: Path) -> None:
    """One CSV row per embedding, aligned by index to the .npy array rows.

    The Indy provenance columns plus a trailing ``breed`` column.
    """
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([*BASE_METADATA_COLUMNS, "breed"])
        for index, row in enumerate(rows):
            cells = base_metadata_cells(index, row.source_filename, row.detection)
            writer.writerow([*cells, row.breed])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Oxford-IIIT Pet negatives (download -> embed)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "images",
        help="download root; data lands in <dir>/oxford-iiit-pet (default: images)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="folder for embeddings.npy + metadata.csv + the sidecar; default "
        "derives the variant subdir under data/embeddings/oxford from "
        "(model, detect, margin)",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="just fetch the data and write catalog.csv; do not embed",
    )
    parser.add_argument(
        "--model",
        default="facebook/dinov2-base",
        help="DINOv2 model id (default: facebook/dinov2-base)",
    )
    parser.add_argument(
        "--no-detect",
        action="store_true",
        help="embed the full frame instead of the cat crop (the detect toggle)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.25,
        help="discard detections below this confidence (default: 0.25)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.1,
        help="crop margin as a fraction of the box size (default: 0.1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="images per embedding forward pass (default: 32)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="embed only the first N cats (for a fast pipeline smoke test); "
        "catalog.csv is always written from the full set",
    )
    args = parser.parse_args()

    # torchvision only fetches + md5-verifies; idempotent if already present.
    print("Ensuring Oxford-IIIT Pet data is present...")
    OxfordIIITPet(root=str(args.data_dir), download=True)

    cats = list_cat_images(args.data_dir)
    if not cats:
        raise SystemExit(f"no cat images found under {args.data_dir}")

    # Co-located with the data (beside images/indy/mapping.csv's analogue), not
    # with the embeddings -- it describes the raw dataset, not a given run.
    catalog_path = args.data_dir / "oxford-iiit-pet" / "catalog.csv"
    write_catalog(cats, catalog_path)
    print_breed_counts(cats)
    print(f"catalog written to {catalog_path}")

    if args.download_only:
        return

    if args.limit is not None:
        cats = cats[: args.limit]

    detect = not args.no_detect
    out_dir: Path = args.out_dir
    if out_dir is None:
        # Derive the per-variant subdir so the no-arg run lands in its own
        # (model, crop) folder rather than the shared base dir.
        out_dir = EmbeddingsVariant(args.model, detect, args.margin).dir(
            REPO_ROOT / "data" / "embeddings" / "oxford"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = (
        None
        if args.no_detect
        else CatDetector(model="yolo11x.pt", min_confidence=args.min_confidence)
    )

    misses = [0]

    def detect_crop_stream() -> Iterator[tuple[OxfordRow, Image.Image] | None]:
        """Detect+crop one cat at a time so the embedder consumes batches.

        Yields ``(row, crop)`` for a kept crop or ``None`` for a no-cat skip;
        ``misses[0]`` tallies the skips for the closing summary. Unlike the Indy
        builder this stays quiet per image -- 2400 lines would bury the bar.
        """
        for cat in cats:
            image = load_image(cat.path)
            if detector is None:
                yield OxfordRow(cat.source_filename, cat.breed, None), image
                continue
            pairs = detect_and_crop(image, detector, args.margin)
            if not pairs:
                misses[0] += 1
                yield None
                continue
            # Highest-confidence detection is the cat; extras would be background.
            detection, crop = pairs[0]
            yield OxfordRow(cat.source_filename, cat.breed, detection), crop

    print(f"Loading embedder ({args.model})...")
    embedder = Embedder(model=args.model)
    start = time.perf_counter()
    rows, embeddings = embed_stream(
        embedder, detect_crop_stream(), args.batch_size, total=len(cats), desc="oxford"
    )
    elapsed = time.perf_counter() - start

    np.save(out_dir / "embeddings.npy", embeddings)
    write_metadata(rows, out_dir / "metadata.csv")
    # margin/min_confidence are None when detect is off so the on-disk sidecar
    # matches the normalized variant key (write_embeddings_meta also nulls them).
    write_embeddings_meta(
        EmbeddingsMeta(
            format_version=1,
            model_id=embedder.model_id,
            embedding_dim=embedder.embedding_dim,
            normalized=False,
            detect=detect,
            margin=args.margin if detect else None,
            min_confidence=args.min_confidence if detect else None,
            row_count=len(rows),
        ),
        out_dir,
    )

    print(
        f"\nSummary: {len(cats)} cats processed, {len(rows)} embeddings "
        f"of dim {embedder.embedding_dim} written to {out_dir}"
    )
    print(f"  device: {embedder.device}")
    print(f"  detected + embedded in {elapsed:.1f}s")
    if detector is not None:
        print(f"  {misses[0]} with NO cat detected (skipped)")


if __name__ == "__main__":
    main()
