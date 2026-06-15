"""Shared helpers for the gallery/dataset driver scripts.

These are I/O concerns (opening files, walking folders) deliberately kept out
of the I/O-free core in ``src/indycat``. Both the detect-review script and the
embed script need them, so they live here rather than being duplicated. The
core never imports from ``scripts/``; scripts import from the core and from
here.
"""

from pathlib import Path

from PIL import Image, ImageOps

#: Image extensions the driver scripts will pick up from a folder.
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


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
