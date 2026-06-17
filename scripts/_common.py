"""Shared helpers for the gallery/dataset driver scripts.

These are I/O concerns (opening files, walking folders) deliberately kept out
of the I/O-free core in ``src/indycat``. Both the detect-review script and the
embed script need them, so they live here rather than being duplicated. The
core never imports from ``scripts/``; scripts import from the core and from
here.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from indycat.detection import Detection
from indycat.embedding import Embedder

#: Image extensions the driver scripts will pick up from a folder.
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")

#: Provenance columns shared by every embeddings ``metadata.csv``. Producers
#: that need extra columns (e.g. a breed label) append to this list and to each
#: row of ``base_metadata_cells``.
BASE_METADATA_COLUMNS = [
    "row",  # explicit index, redundant with order but greppable
    "source_filename",
    "detect_used",  # True / False -- records the detect toggle per row
    "confidence",  # empty when detection is None (--no-detect)
    "x1",
    "y1",
    "x2",
    "y2",
    "area_fraction",  # empty when detection is None
]


@dataclass
class GalleryRow:
    """Provenance for one embedding; ``detection`` is None under --no-detect."""

    source_filename: str
    detection: Detection | None


def load_image(path: Path) -> Image.Image:
    """Open an image with EXIF orientation applied.

    Phone photos often store rotation as an EXIF tag; applying it here makes
    box coordinates, saved crops, and embeddings all match the photo as
    displayed.
    """
    return ImageOps.exif_transpose(Image.open(path))


def iter_images(directory: Path) -> list[Path]:
    """Sorted image files directly in ``directory`` (non-recursive)."""
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def embed_in_batches(
    embedder: Embedder, images: list[Image.Image], batch_size: int
) -> np.ndarray:
    """Embed ``images`` in chunks so large datasets don't blow up GPU memory.

    For Indy's ~35 crops one batch would do, but the chunking is what lets this
    same routine scale to the ~2400 Oxford crops.
    """
    if not images:
        return np.empty((0, embedder.embedding_dim), dtype=np.float32)
    chunks = [
        embedder.embed_batch(images[start : start + batch_size])
        for start in range(0, len(images), batch_size)
    ]
    return np.concatenate(chunks, axis=0)


def base_metadata_cells(
    index: int, source_filename: str, detection: Detection | None
) -> list[object]:
    """The ``BASE_METADATA_COLUMNS`` cells for one embedding row.

    Detection-derived cells are left empty when ``detection`` is None (the
    --no-detect / full-frame case), so the column count stays fixed. Takes
    primitives rather than a row class so producers with extra columns can
    append to the returned list.
    """
    if detection is None:
        return [index, source_filename, False, "", "", "", "", "", ""]
    x1, y1, x2, y2 = detection.box_xyxy
    return [
        index,
        source_filename,
        True,
        f"{detection.confidence:.4f}",
        x1,
        y1,
        x2,
        y2,
        f"{detection.area_fraction:.4f}",
    ]
