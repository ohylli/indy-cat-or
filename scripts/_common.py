"""Shared helpers for the gallery/dataset driver scripts.

These are I/O concerns (opening files, walking folders) deliberately kept out
of the I/O-free core in ``src/indycat``. Both the detect-review script and the
embed script need them, so they live here rather than being duplicated. The
core never imports from ``scripts/``; scripts import from the core and from
here.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray
from PIL import Image, ImageOps

from indycat.detection import Detection
from indycat.embedding import Embedder

#: Image extensions the driver scripts will pick up from a folder.
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")

#: Filename of the per-variant provenance sidecar written beside each dataset's
#: ``embeddings.npy`` / ``metadata.csv``. See ``docs/embeddings_provenance.md``.
SIDECAR_NAME = "embeddings.meta.yaml"

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


def load_cached_embeddings(
    metadata_path: Path, embeddings_path: Path
) -> tuple[list[str], NDArray[np.float32]]:
    """Read a cached embeddings dataset back as ``(source_filenames, vectors)``.

    The I/O inverse of ``base_metadata_cells``: the ``source_filename`` column of
    ``metadata.csv`` (in row order) and the row-aligned ``embeddings.npy``. The
    row counts must match -- a mismatch means the two files drifted apart, which
    would silently misalign every vector with its provenance, so it is a loud
    error rather than a truncation. Reusable by the calibrate/evaluate drivers.
    """
    with metadata_path.open(encoding="utf-8", newline="") as f:
        names = [row["source_filename"] for row in csv.DictReader(f)]
    vectors: NDArray[np.float32] = np.load(embeddings_path).astype(np.float32)
    if len(names) != vectors.shape[0]:
        raise ValueError(
            f"{metadata_path} has {len(names)} rows but {embeddings_path} has "
            f"{vectors.shape[0]}; the embeddings cache is inconsistent"
        )
    return names, vectors


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


# --------------------------------------------------------------------------- #
# Embedding provenance: variant identity + the on-disk sidecar.
#
# An embeddings cache is identified by *which model* embedded it and *which crop
# setting* produced the images it embedded -- not by its folder name, which is
# only a human handle. This layer is the source of truth those names mirror, and
# the loud cross-checks that keep a query from being scored against a cache built
# with a different backbone or crop. The full rationale is in
# ``docs/embeddings_provenance.md``.
# --------------------------------------------------------------------------- #


def model_slug(model_id: str) -> str:
    """Filesystem-safe handle for a HF model id (``/`` -> ``--``).

    ``"facebook/dinov2-base"`` -> ``"facebook--dinov2-base"``. The ``/`` would
    otherwise be a path separator; keeping the org segment disambiguates future
    ids like ``facebook/dinov3-vitb16-...``. Purely a directory name -- the
    authoritative model id lives in the sidecar, never parsed back from this.
    """
    return model_id.replace("/", "--")


def crop_slug(detect: bool, margin: float) -> str:
    """Filesystem-safe handle for the crop setting.

    ``crop-m0.1`` when detection is on (the margin ``:g``-formatted so ``0.1``
    stays ``0.1`` rather than ``0.100000``); ``nocrop`` when detection is off,
    with the margin omitted because it does not apply to a full-frame embed.
    """
    if not detect:
        return "nocrop"
    return f"crop-m{margin:g}"


@dataclass(frozen=True)
class EmbeddingsVariant:
    """The crop+model axes that select an embeddings cache directory.

    Only the three axes that name a folder: the model and the crop setting
    (``detect`` + ``margin``). ``min_confidence`` is provenance recorded in the
    sidecar but is not a directory axis (see ``docs/embeddings_provenance.md``),
    so it does not live here.
    """

    model_id: str
    detect: bool
    margin: float

    def subdir(self) -> Path:
        """Variant-relative path: ``<model_slug>/<crop_slug>``."""
        return Path(model_slug(self.model_id)) / crop_slug(self.detect, self.margin)

    def dir(self, dataset_root: Path) -> Path:
        """Absolute cache dir for this variant under a dataset root.

        ``dataset_root`` is e.g. ``data/embeddings/indy``; the returned path is
        where ``embeddings.npy`` / ``metadata.csv`` / the sidecar live.
        """
        return dataset_root / self.subdir()


@dataclass(frozen=True)
class EmbeddingsMeta:
    """In-memory model of the ``embeddings.meta.yaml`` sidecar.

    The sidecar is the authoritative record of how a cache was built. ``margin``
    and ``min_confidence`` are ``None`` when ``detect`` is false (they do not
    apply to a full-frame embed), and the builder writes them as ``null`` on
    disk too, so a ``nocrop`` sidecar never carries a misleading margin.
    """

    format_version: int
    model_id: str
    embedding_dim: int
    normalized: bool
    detect: bool
    margin: float | None
    min_confidence: float | None
    row_count: int

    def variant_key(self) -> tuple[str, bool, float | None, float | None]:
        """Normalized identity for equality checks across caches.

        ``(model_id, detect, margin, min_confidence)`` with ``margin`` and
        ``min_confidence`` forced to ``None`` when ``detect`` is false -- so two
        ``nocrop`` caches compare equal regardless of an irrelevant recorded
        margin. This is the key asserted-equal wherever two identities meet
        (cache-vs-cache, cache-vs-manifest, cache-vs-artifact).
        """
        if not self.detect:
            return (self.model_id, self.detect, None, None)
        return (self.model_id, self.detect, self.margin, self.min_confidence)


def write_embeddings_meta(meta: EmbeddingsMeta, out_dir: Path) -> Path:
    """Write ``meta`` as ``embeddings.meta.yaml`` in ``out_dir``; return its path.

    When ``detect`` is false, ``margin`` and ``min_confidence`` are written as
    ``null`` regardless of what ``meta`` carries -- the on-disk sidecar must not
    advertise a margin a full-frame embed never used, so the file matches the
    normalized :meth:`EmbeddingsMeta.variant_key` rather than just the key
    masking it at compare time.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    margin = meta.margin if meta.detect else None
    min_confidence = meta.min_confidence if meta.detect else None
    payload = {
        "format_version": meta.format_version,
        "model_id": meta.model_id,
        "embedding_dim": meta.embedding_dim,
        "normalized": meta.normalized,
        "detect": meta.detect,
        "margin": margin,
        "min_confidence": min_confidence,
        "row_count": meta.row_count,
    }
    sidecar_path = out_dir / SIDECAR_NAME
    with sidecar_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            payload,
            f,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
    return sidecar_path


def read_embeddings_meta(out_dir: Path) -> EmbeddingsMeta:
    """Read the sidecar from ``out_dir``; raise loudly if it is missing.

    A missing sidecar means an unprovenance'd cache (e.g. a pre-migration flat
    folder), which must fail rather than be scored blindly -- same loud-on-drift
    discipline as :func:`load_cached_embeddings`'s row-count guard.
    """
    sidecar_path = out_dir / SIDECAR_NAME
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"no embeddings sidecar at {sidecar_path}; the cache has no provenance "
            f"(rebuild it so a {SIDECAR_NAME} is written)"
        )
    with sidecar_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return EmbeddingsMeta(
        format_version=data["format_version"],
        model_id=data["model_id"],
        embedding_dim=data["embedding_dim"],
        normalized=data["normalized"],
        detect=data["detect"],
        margin=data["margin"],
        min_confidence=data["min_confidence"],
        row_count=data["row_count"],
    )


def load_embeddings_variant(
    out_dir: Path,
) -> tuple[list[str], NDArray[np.float32], EmbeddingsMeta]:
    """Load a cache *with* its provenance from a variant directory.

    Composes :func:`load_cached_embeddings` (the ``metadata.csv`` /
    ``embeddings.npy`` row-count guard) with :func:`read_embeddings_meta`, then
    adds one more loud cross-check: the sidecar's ``row_count`` must equal the
    ``.npy`` row count. A drift here means the sidecar was written against a
    different build than the vectors on disk, so it is an error rather than a
    silent trust of one over the other.
    """
    metadata_path = out_dir / "metadata.csv"
    embeddings_path = out_dir / "embeddings.npy"
    names, vectors = load_cached_embeddings(metadata_path, embeddings_path)
    meta = read_embeddings_meta(out_dir)
    if meta.row_count != vectors.shape[0]:
        raise ValueError(
            f"{out_dir / SIDECAR_NAME} records row_count={meta.row_count} but "
            f"{embeddings_path} has {vectors.shape[0]} rows; the sidecar and "
            f"vectors drifted apart"
        )
    return names, vectors, meta
