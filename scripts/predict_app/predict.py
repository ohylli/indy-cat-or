"""Recognition core for the "Is it Indy?" prediction app.

UI-agnostic and streamlit-free, like the rest of the pipeline core: it composes
the existing detect -> crop -> embed -> decide stages into a single
:func:`classify` over one opened image and returns plain data. The thin
Streamlit layer (``app.py``) owns model loading/caching, I/O, and rendering.

The live gallery and threshold come from the *frozen calibration artifact*
(``calibration.artifact.load_artifact``), never from the full cached Indy
embeddings -- the artifact's gallery is only the calibration photos, so scoring
against it keeps the held-out test photos out of the live decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from calibration.artifact import CalibrationArtifact
from calibration.manifest import SplitConfigError
from indycat.decision import Aggregation, Gallery, score
from indycat.detection import CatDetector, Detection, detect_and_crop
from indycat.embedding import Embedder

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INDY_DIR = REPO_ROOT / "images" / "indy"
ARTIFACTS_DIR = REPO_ROOT / "data" / "artifacts"

#: The only comparison the decide stage uses (artifact records ``">="``).
SUPPORTED_COMPARISON = ">="

#: The detect-and-crop margin used when the artifact recorded none (detect was
#: off when the gallery was built, so ``margin`` is ``None``). Matches
#: ``detection.detect_and_crop``'s own default so an override-to-detect path
#: behaves like the gallery builders' default.
DEFAULT_MARGIN = 0.1


def find_artifacts(artifacts_dir: Path = ARTIFACTS_DIR) -> list[Path]:
    """Calibration artifact YAMLs on disk, newest-looking last (sorted by name).

    The companion ``.gallery.npy`` is excluded -- only the ``.yaml`` entry points
    are returned. Empty list means none have been produced yet (run calibrate).
    """
    if not artifacts_dir.is_dir():
        return []
    return sorted(
        p for p in artifacts_dir.glob("*.yaml") if not p.name.endswith(".gallery.npy")
    )


def build_gallery(
    artifact: CalibrationArtifact,
    raw_vectors: NDArray[np.float32],
    embedder: Embedder | None = None,
) -> tuple[Gallery, dict[str, tuple[str, str]]]:
    """Build the live :class:`Gallery` and a ``filename -> (position, view)`` map.

    ``raw_vectors`` are the artifact's companion gallery vectors (row-aligned to
    ``artifact.gallery_images``); :meth:`Gallery.from_raw` L2-normalizes them. The
    positions map lets a verdict describe the matched photo's pose in words.

    Pass ``embedder`` (the *live* backbone the app loaded from
    ``artifact.embedding.model_id``) to assert its output width matches the frozen
    gallery vectors. ``load_artifact`` already checks the vectors against the
    recorded ``embedding_dim``; this is the matching loud guard on the live model,
    so a backbone whose output width disagrees with the gallery is caught at
    bundle build rather than producing a silently-wrong verdict.
    """
    if embedder is not None and embedder.embedding_dim != raw_vectors.shape[1]:
        raise SplitConfigError(
            f"live embedder {embedder.model_id!r} produces "
            f"{embedder.embedding_dim}-dimensional vectors but the gallery vectors "
            f"are {raw_vectors.shape[1]}-dimensional; the live backbone does not "
            "match the one the gallery was built with"
        )
    names = [img.source_filename for img in artifact.gallery_images]
    positions = {
        img.source_filename: (img.position, img.view) for img in artifact.gallery_images
    }
    return Gallery.from_raw(names, raw_vectors), positions


@dataclass(frozen=True)
class CropResult:
    """The verdict for one query (one detected crop, or the whole frame).

    ``detection`` is ``None`` when detection was toggled off (the query is the
    full image). ``crop_image`` is exactly the image that was embedded -- the
    cropped cat (detect on) or the full frame (detect off) -- carried back so the
    UI can show it without re-running the detector. ``margin`` is
    ``score - threshold`` -- positive means Indy.
    """

    index: int
    detection: Detection | None
    crop_image: Image.Image
    score: float
    margin: float
    is_indy: bool
    best_name: str
    best_position: str
    best_view: str


@dataclass(frozen=True)
class ClassifyResult:
    """The outcome of classifying one image against the Indy gallery.

    ``no_cat`` is the explicit detect-on-but-nothing-found case: ``crops`` is
    empty and the frame is deliberately *not* embedded. With multiple crops the
    image is Indy if *any* crop clears the threshold.
    """

    detect_used: bool
    no_cat: bool
    crops: list[CropResult]
    overall_is_indy: bool
    threshold: float
    aggregation: Aggregation


def _score_crop(
    index: int,
    detection: Detection | None,
    crop: Image.Image,
    *,
    embedder: Embedder,
    gallery: Gallery,
    positions: dict[str, tuple[str, str]],
    threshold: float,
    aggregation: Aggregation,
) -> CropResult:
    match = score(embedder.embed(crop), gallery, aggregation)
    position, view = positions.get(match.best_name, ("", ""))
    return CropResult(
        index=index,
        detection=detection,
        crop_image=crop,
        score=match.score,
        margin=match.score - threshold,
        is_indy=match.score >= threshold,
        best_name=match.best_name,
        best_position=position,
        best_view=view,
    )


def classify(
    image: Image.Image,
    *,
    detector: CatDetector,
    embedder: Embedder,
    gallery: Gallery,
    positions: dict[str, tuple[str, str]],
    threshold: float,
    aggregation: Aggregation,
    comparison: str = SUPPORTED_COMPARISON,
    detect: bool = True,
    margin: float | None = None,
) -> ClassifyResult:
    """Classify one (already EXIF-corrected) image as Indy or not.

    ``detect`` on: every detected cat is cropped and scored separately; an empty
    detection set is the ``no_cat`` case and the frame is not embedded. ``detect``
    off: the full image is the single query (``detection=None``). The caller owns
    EXIF correction so the query matches the upright gallery.

    ``margin`` is the crop margin the gallery was built with (from
    ``artifact.embedding.margin``); ``None`` -- the artifact's value when detect
    was off -- falls back to :data:`DEFAULT_MARGIN` so an override-to-detect run
    still crops like the builders' default rather than silently using a stray
    value. Ignored when ``detect`` is off (the whole frame is embedded).
    """
    if comparison != SUPPORTED_COMPARISON:
        raise ValueError(
            f"unsupported comparison {comparison!r}; this app scores "
            f"{SUPPORTED_COMPARISON} (the artifact's recorded convention)"
        )

    def score_one(
        index: int, detection: Detection | None, crop: Image.Image
    ) -> CropResult:
        return _score_crop(
            index,
            detection,
            crop,
            embedder=embedder,
            gallery=gallery,
            positions=positions,
            threshold=threshold,
            aggregation=aggregation,
        )

    if not detect:
        crops = [score_one(0, None, image)]
        return ClassifyResult(
            detect_used=False,
            no_cat=False,
            crops=crops,
            overall_is_indy=crops[0].is_indy,
            threshold=threshold,
            aggregation=aggregation,
        )

    crop_margin = DEFAULT_MARGIN if margin is None else margin
    detected = detect_and_crop(image, detector, crop_margin)
    crops = [
        score_one(i, detection, crop) for i, (detection, crop) in enumerate(detected)
    ]
    return ClassifyResult(
        detect_used=True,
        no_cat=len(crops) == 0,
        crops=crops,
        overall_is_indy=any(c.is_indy for c in crops),
        threshold=threshold,
        aggregation=aggregation,
    )
