"""Build the cat-breeds (Kaggle ma7555) eval negatives: download -> detect -> embed.

A second negatives source alongside Oxford, used as a held-out **false-positive
stress test**: every image is a non-Indy cat, so any crop the frozen artifact
scores above threshold is a false positive. Its draw is the 67 "breed" folders
of ``ma7555/cat-breeds-dataset`` (a Petfinder scrape) -- including a Norwegian
Forest Cat slice, the headline long-haired look-alike. The labels are noisy
(many are coat patterns, not breeds, and mislabelling is common), but that does
not bias an FPR measurement: a mislabelled cat is still a valid negative; the
breed only buckets the results.

For the numbers to mean anything the negatives must be on *identical footing* to
the gallery -- same loader, same YOLO re-detection, same frozen DINOv2, same
raw/un-normalized vectors -- so this mirrors ``build_oxford_negatives.py`` and
shares its helpers via ``_common``. The cache lands in the same variant-nested
layout under ``data/embeddings/catbreeds/<model>/<crop>/``.

Two differences from the Oxford builder:
    download  via ``kagglehub`` (anonymous; no credentials needed) straight into
              ``--data-dir/cat-breeds`` using ``output_dir``; idempotent, so a
              re-run skips the 1.9 GB fetch. The breed is the image's parent
              folder name (e.g. ``Norwegian Forest Cat``).
    sampling  126k images is too many to embed wholesale, so a seeded,
              breed-stratified ``--per-breed-limit`` caps each breed (0 =
              unlimited). The drop is logged, never silent. ``catalog.csv``
              always describes the *full* set; sampling only affects the embed.

Usage:
    uv run python scripts/build_catbreeds_negatives.py --download-only
    uv run python scripts/build_catbreeds_negatives.py --limit 16
    uv run python scripts/build_catbreeds_negatives.py --per-breed-limit 200
    uv run python scripts/build_catbreeds_negatives.py --no-detect
"""

import argparse
import csv
import random
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import kagglehub
import numpy as np
from PIL import Image

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

#: Kaggle dataset handle (public; kagglehub fetches it anonymously).
DATASET_HANDLE = "ma7555/cat-breeds-dataset"

#: Built-in seed so the stratified per-breed sample is bit-for-bit repeatable.
DEFAULT_SEED = 20240601

#: Default per-breed cap. ~67 breeds; the set is dominated by Domestic Short Hair
#: (53k of 126k), so an uncapped embed is hours on the 3070. 0 = unlimited.
DEFAULT_PER_BREED_LIMIT = 200


@dataclass(frozen=True)
class CatImage:
    """One cat-breeds image: where it is, its filename, and its breed folder."""

    path: Path
    source_filename: str
    breed: str


@dataclass
class CatbreedsRow:
    """Provenance for one embedding, with the cat-breeds folder label."""

    source_filename: str
    breed: str
    detection: Detection | None


def ensure_dataset(data_dir: Path) -> Path:
    """Download the dataset into ``data_dir/cat-breeds`` (idempotent); return root.

    ``kagglehub.dataset_download`` writes via ``output_dir`` and skips the fetch
    when the data is already present (a ``.complete`` marker), so a re-run is
    free. Returns the resolved dataset root, which contains ``images/<breed>/``
    and ``data/cats.csv``.
    """
    out_dir = data_dir / "cat-breeds"
    path = kagglehub.dataset_download(DATASET_HANDLE, output_dir=str(out_dir))
    return Path(path)


def list_cat_images(dataset_root: Path) -> list[CatImage]:
    """All images under ``images/<breed>/``, sorted for reproducibility.

    The breed is the immediate parent folder name; filenames are globally unique
    across breeds, so ``source_filename`` is the bare filename (an image is
    located as ``images/<breed>/<source_filename>``).
    """
    images_root = dataset_root / "images"
    cats: list[CatImage] = []
    for breed_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
        for image_path in sorted(breed_dir.glob("*.jpg")):
            cats.append(
                CatImage(
                    path=image_path,
                    source_filename=image_path.name,
                    breed=breed_dir.name,
                )
            )
    cats.sort(key=lambda c: (c.breed, c.source_filename))
    return cats


def sample_per_breed(
    cats: list[CatImage], per_breed_limit: int, seed: int
) -> list[CatImage]:
    """Cap each breed at ``per_breed_limit`` via a seeded shuffle (0 = unlimited).

    Each breed is shuffled with a per-breed sub-seed drawn from one
    ``Random(seed)`` (breeds processed in sorted order for determinism), then
    truncated. The result is re-sorted by ``(breed, source_filename)`` so the
    embedded rows keep a stable, inspectable order regardless of the shuffle.
    """
    if per_breed_limit <= 0:
        return cats
    by_breed: dict[str, list[CatImage]] = {}
    for cat in cats:
        by_breed.setdefault(cat.breed, []).append(cat)

    rng = random.Random(seed)
    kept: list[CatImage] = []
    for breed in sorted(by_breed):
        sub_seed = rng.randrange(2**31)
        breed_cats = list(by_breed[breed])
        random.Random(sub_seed).shuffle(breed_cats)
        kept.extend(breed_cats[:per_breed_limit])
    kept.sort(key=lambda c: (c.breed, c.source_filename))
    return kept


def write_catalog(cats: list[CatImage], csv_path: Path) -> None:
    """The reviewable image -> breed listing (the full set, pre-sample)."""
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_filename", "breed"])
        for cat in cats:
            writer.writerow([cat.source_filename, cat.breed])


def print_breed_counts(cats: list[CatImage], header: str) -> None:
    """Per-breed counts -- a text check on the breed folders without opening one."""
    counts = Counter(cat.breed for cat in cats)
    print(f"{header}: {len(counts)} breeds, {len(cats)} images total:")
    for breed, count in sorted(counts.items()):
        print(f"  {breed}: {count}")


def write_metadata(rows: list[CatbreedsRow], csv_path: Path) -> None:
    """One CSV row per embedding, aligned by index to the .npy array rows.

    The base provenance columns plus a trailing ``breed`` column (same shape as
    the Oxford metadata, so the calibration/eval breed join is reused as-is).
    """
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([*BASE_METADATA_COLUMNS, "breed"])
        for index, row in enumerate(rows):
            cells = base_metadata_cells(index, row.source_filename, row.detection)
            writer.writerow([*cells, row.breed])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the cat-breeds (Kaggle ma7555) eval negatives."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "images",
        help="download root; data lands in <dir>/cat-breeds (default: images)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="folder for embeddings.npy + metadata.csv + the sidecar; default "
        "derives the variant subdir under data/embeddings/catbreeds from "
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
        "--per-breed-limit",
        type=int,
        default=DEFAULT_PER_BREED_LIMIT,
        help=f"cap each breed at N images via a seeded sample (default "
        f"{DEFAULT_PER_BREED_LIMIT}; 0 = unlimited)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"seed for the per-breed sample (default {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="embed only the first N cats after sampling (fast smoke test); "
        "catalog.csv is always written from the full set",
    )
    args = parser.parse_args()

    print("Ensuring cat-breeds data is present (kagglehub, anonymous)...")
    dataset_root = ensure_dataset(args.data_dir)

    cats = list_cat_images(dataset_root)
    if not cats:
        raise SystemExit(f"no cat images found under {dataset_root / 'images'}")

    # catalog.csv describes the full dataset (pre-sample), beside the raw data.
    catalog_path = dataset_root / "catalog.csv"
    write_catalog(cats, catalog_path)
    print_breed_counts(cats, "full dataset")
    print(f"catalog written to {catalog_path}")

    if args.download_only:
        return

    sampled = sample_per_breed(cats, args.per_breed_limit, args.seed)
    if len(sampled) != len(cats):
        print(
            f"\nsampled {len(sampled)} of {len(cats)} images "
            f"(per-breed cap {args.per_breed_limit}, seed {args.seed}); "
            f"{len(cats) - len(sampled)} dropped"
        )
    if args.limit is not None:
        sampled = sampled[: args.limit]
        print(f"--limit: embedding only the first {len(sampled)} (smoke test)")

    detect = not args.no_detect
    out_dir: Path = args.out_dir
    if out_dir is None:
        out_dir = EmbeddingsVariant(args.model, detect, args.margin).dir(
            REPO_ROOT / "data" / "embeddings" / "catbreeds"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = (
        None
        if args.no_detect
        else CatDetector(model="yolo11x.pt", min_confidence=args.min_confidence)
    )

    misses = [0]
    corrupt = [0]

    def detect_crop_stream() -> Iterator[tuple[CatbreedsRow, Image.Image] | None]:
        """Detect+crop one cat at a time so the embedder consumes batches.

        Yields ``(row, crop)`` for a kept crop or ``None`` for a skip; ``misses[0]``
        tallies no-cat skips and ``corrupt[0]`` tallies unreadable images for the
        closing summary. Unlike the clean Oxford set, this Petfinder scrape has
        truncated/broken JPEGs, so a decode failure is a counted skip (never an
        embed of garbage pixels), the same loud-but-recoverable discipline as a
        detector miss -- the file is catalogued but absent from metadata.csv.
        """
        for cat in sampled:
            try:
                image = load_image(cat.path)
                pairs = (
                    None
                    if detector is None
                    else detect_and_crop(image, detector, args.margin)
                )
            except OSError:
                corrupt[0] += 1
                yield None
                continue
            if detector is None:
                yield CatbreedsRow(cat.source_filename, cat.breed, None), image
                continue
            if not pairs:
                misses[0] += 1
                yield None
                continue
            detection, crop = pairs[0]
            yield CatbreedsRow(cat.source_filename, cat.breed, detection), crop

    print(f"Loading embedder ({args.model})...")
    embedder = Embedder(model=args.model)
    start = time.perf_counter()
    rows, embeddings = embed_stream(
        embedder,
        detect_crop_stream(),
        args.batch_size,
        total=len(sampled),
        desc="catbreeds",
    )
    elapsed = time.perf_counter() - start

    np.save(out_dir / "embeddings.npy", embeddings)
    write_metadata(rows, out_dir / "metadata.csv")
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
        f"\nSummary: {len(sampled)} cats processed, {len(rows)} embeddings "
        f"of dim {embedder.embedding_dim} written to {out_dir}"
    )
    print(f"  device: {embedder.device}")
    print(f"  detected + embedded in {elapsed:.1f}s")
    if detector is not None:
        print(f"  {misses[0]} with NO cat detected (skipped)")
    print(f"  {corrupt[0]} unreadable/corrupt image(s) (skipped)")


if __name__ == "__main__":
    main()
