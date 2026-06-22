"""Build Indy's gallery embeddings: detect -> crop -> embed, in one pass.

This is the real producer of the gallery the live decision compares against.
It detects the cat in each Indy photo on the fly (it does NOT depend on
``detect_indy_gallery.py`` having run -- that script is a separate, purely
textual sanity-check of the detector). For each photo it embeds the
highest-confidence crop, which review of the detections established is Indy
himself; any lower-confidence extra detection is an error (a dog beside him, a
cat-print blanket) and is dropped. A photo with no cat detected is reported and
skipped -- never embedded as a full frame.

Pass ``--no-detect`` to embed the full frame instead of the cat crop; this is
the toggle that lets the detect-and-crop stage's effect on accuracy be measured
rather than assumed.

Output (three files in the resolved ``--out-dir``):
    embeddings.npy   float32 array of shape (n_rows, embedding_dim); row i is
                     the vector for the crop described by row i of the CSV.
    metadata.csv     one row per embedding, recording provenance so the
                     gallery/calibration/test split can be done later as a
                     text check.
    embeddings.meta.yaml
                     the per-variant provenance sidecar (model id, embedding
                     dim, crop setting, row count) -- the source of truth the
                     folder name only mirrors. See ``docs/embeddings_provenance.md``.

When ``--out-dir`` is omitted it is derived as
``data/embeddings/indy/<model_slug>/<crop_slug>`` so each (model, crop) variant
lands in its own subdir and large vectors can never drop into the base folder
by accident.

Usage:
    uv run python scripts/build_indy_gallery.py
    uv run python scripts/build_indy_gallery.py --no-detect
    uv run python scripts/build_indy_gallery.py --model facebook/dinov2-large
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from _common import (
    BASE_METADATA_COLUMNS,
    EmbeddingsMeta,
    EmbeddingsVariant,
    GalleryRow,
    base_metadata_cells,
    embed_in_batches,
    iter_images,
    load_image,
    write_embeddings_meta,
)
from indycat.detection import CatDetector, detect_and_crop
from indycat.embedding import Embedder

REPO_ROOT = Path(__file__).parent.parent


def write_metadata(rows: list[GalleryRow], csv_path: Path) -> None:
    """One CSV row per embedding, aligned by index to the .npy array rows."""
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(BASE_METADATA_COLUMNS)
        for index, row in enumerate(rows):
            writer.writerow(
                base_metadata_cells(index, row.source_filename, row.detection)
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Indy's gallery embeddings (detect -> crop -> embed)."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=REPO_ROOT / "images" / "indy",
        help="folder of input photos (default: images/indy)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="folder for embeddings.npy + metadata.csv + the sidecar; default "
        "derives the variant subdir under data/embeddings/indy from "
        "(model, detect, margin)",
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
    args = parser.parse_args()

    detect = not args.no_detect
    out_dir: Path = args.out_dir
    if out_dir is None:
        # Derive the per-variant subdir so the no-arg run lands in its own
        # (model, crop) folder rather than the shared base dir.
        out_dir = EmbeddingsVariant(args.model, detect, args.margin).dir(
            REPO_ROOT / "data" / "embeddings" / "indy"
        )

    photos = iter_images(args.images_dir)
    if not photos:
        raise SystemExit(f"no images found in {args.images_dir}")

    detector = (
        None
        if args.no_detect
        else CatDetector(model="yolo11x.pt", min_confidence=args.min_confidence)
    )

    crops: list[Image.Image] = []
    rows: list[GalleryRow] = []
    no_cat: list[str] = []
    for path in photos:
        image = load_image(path)
        if detector is None:
            crops.append(image)
            rows.append(GalleryRow(path.name, None))
            print(f"{path.name}: full frame (detect off)")
            continue
        pairs = detect_and_crop(image, detector, args.margin)
        if not pairs:
            no_cat.append(path.name)
            print(f"{path.name}: NO CAT DETECTED -- skipped")
            continue
        # Highest-confidence detection is Indy; any extras are errors.
        detection, crop = pairs[0]
        crops.append(crop)
        rows.append(GalleryRow(path.name, detection))
        extra = f" ({len(pairs)} detections, kept top)" if len(pairs) > 1 else ""
        print(f"{path.name}: confidence {detection.confidence:.2f}{extra}")

    print(f"\nLoading embedder ({args.model})...")
    embedder = Embedder(model=args.model)
    embeddings = embed_in_batches(embedder, crops, args.batch_size)

    out_dir.mkdir(parents=True, exist_ok=True)
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
        f"\nSummary: {len(photos)} photos, {len(rows)} embeddings "
        f"of dim {embedder.embedding_dim} written to {out_dir}"
    )
    print(f"  device: {embedder.device}")
    if no_cat:
        print(f"  {len(no_cat)} with NO cat detected (skipped): {', '.join(no_cat)}")
    else:
        print("  every photo produced an embedding")


if __name__ == "__main__":
    main()
