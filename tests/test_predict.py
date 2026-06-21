"""Tests for the prediction app's streamlit-free recognition core.

Fast and model-free: the detector and embedder are faked (subclasses that skip
weight loading and return canned data), so ``classify`` is exercised purely on
its orchestration -- the two mandated edge cases (no cat, multiple cats), the
detect toggle, and the score/threshold verdict -- without a GPU or real photos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from calibration.artifact import (
    CalibrationArtifact,
    ChosenBy,
    GalleryImageRef,
    MetricsAtThreshold,
)
from indycat.decision import Gallery
from indycat.detection import CatDetector, Detection, DetectionResult
from indycat.embedding import Embedder
from predict_app import predict

# --------------------------------------------------------------------------- #
# Fakes (subclass so they pass as CatDetector/Embedder, but load no weights)
# --------------------------------------------------------------------------- #


class FakeDetector(CatDetector):
    def __init__(self, detections: list[Detection]) -> None:
        self._detections = detections
        self.detect_calls = 0

    def detect(self, image: Image.Image) -> DetectionResult:
        self.detect_calls += 1
        return DetectionResult(image_size=image.size, detections=self._detections)


class FakeEmbedder(Embedder):
    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = [np.asarray(v, dtype=np.float32) for v in vectors]
        self.embed_calls = 0

    def embed(self, image: Image.Image) -> NDArray[np.float32]:
        vector = self._vectors[self.embed_calls]
        self.embed_calls += 1
        return vector


def basis_gallery() -> tuple[Gallery, dict[str, tuple[str, str]]]:
    """A 2-vector gallery on the axes, with poses, for predictable scoring."""
    gallery = Gallery.from_raw(["a", "b"], np.eye(2, dtype=np.float32))
    positions = {"a": ("lying", "side"), "b": ("sitting", "front")}
    return gallery, positions


def a_detection() -> Detection:
    return Detection(confidence=0.9, box_xyxy=(0, 0, 50, 50), area_fraction=0.25)


def square() -> Image.Image:
    return Image.new("RGB", (100, 100))


# --------------------------------------------------------------------------- #
# classify: detect on
# --------------------------------------------------------------------------- #


def test_no_cat_is_explicit_and_does_not_embed() -> None:
    gallery, positions = basis_gallery()
    embedder = FakeEmbedder([])
    result = predict.classify(
        square(),
        detector=FakeDetector([]),
        embedder=embedder,
        gallery=gallery,
        positions=positions,
        threshold=0.8,
        aggregation="max",
        detect=True,
    )
    assert result.no_cat is True
    assert result.crops == []
    assert result.overall_is_indy is False
    assert embedder.embed_calls == 0  # the frame was NOT embedded as a fallback


def test_single_crop_above_threshold_is_indy() -> None:
    gallery, positions = basis_gallery()
    result = predict.classify(
        square(),
        detector=FakeDetector([a_detection()]),
        embedder=FakeEmbedder([[1.0, 0.0]]),  # cosine 1.0 with "a"
        gallery=gallery,
        positions=positions,
        threshold=0.8,
        aggregation="max",
        detect=True,
    )
    assert result.overall_is_indy is True
    (crop,) = result.crops
    assert crop.is_indy is True
    assert crop.best_name == "a"
    assert crop.best_position == "lying"
    assert crop.best_view == "side"
    assert crop.score == pytest.approx(1.0)
    assert crop.margin == pytest.approx(0.2)
    assert crop.detection is not None


def test_single_crop_below_threshold_is_not_indy() -> None:
    gallery, positions = basis_gallery()
    result = predict.classify(
        square(),
        detector=FakeDetector([a_detection()]),
        embedder=FakeEmbedder([[0.5, 0.5]]),  # cosine ~0.707 with "a"
        gallery=gallery,
        positions=positions,
        threshold=0.8,
        aggregation="max",
        detect=True,
    )
    assert result.overall_is_indy is False
    (crop,) = result.crops
    assert crop.is_indy is False
    assert crop.score == pytest.approx(np.sqrt(0.5))
    assert crop.margin < 0


def test_multiple_crops_indy_if_any_clears_threshold() -> None:
    gallery, positions = basis_gallery()
    detector = FakeDetector([a_detection(), a_detection()])
    result = predict.classify(
        square(),
        detector=detector,
        embedder=FakeEmbedder([[0.5, 0.5], [1.0, 0.0]]),  # below, then above
        gallery=gallery,
        positions=positions,
        threshold=0.8,
        aggregation="max",
        detect=True,
    )
    assert len(result.crops) == 2
    assert [c.is_indy for c in result.crops] == [False, True]
    assert result.overall_is_indy is True
    assert [c.index for c in result.crops] == [0, 1]


# --------------------------------------------------------------------------- #
# classify: detect off, and the comparison guard
# --------------------------------------------------------------------------- #


def test_detect_off_embeds_full_frame_without_detector() -> None:
    gallery, positions = basis_gallery()
    detector = FakeDetector([])
    image = square()
    result = predict.classify(
        image,
        detector=detector,
        embedder=FakeEmbedder([[1.0, 0.0]]),
        gallery=gallery,
        positions=positions,
        threshold=0.8,
        aggregation="max",
        detect=False,
    )
    assert result.detect_used is False
    assert result.no_cat is False
    assert detector.detect_calls == 0  # detection skipped entirely
    (crop,) = result.crops
    assert crop.detection is None
    assert crop.crop_image is image  # the whole frame is the query
    assert result.overall_is_indy is True


def test_unsupported_comparison_raises() -> None:
    gallery, positions = basis_gallery()
    with pytest.raises(ValueError, match="comparison"):
        predict.classify(
            square(),
            detector=FakeDetector([a_detection()]),
            embedder=FakeEmbedder([[1.0, 0.0]]),
            gallery=gallery,
            positions=positions,
            threshold=0.8,
            aggregation="max",
            comparison="<",
            detect=True,
        )


# --------------------------------------------------------------------------- #
# build_gallery / find_artifacts wiring
# --------------------------------------------------------------------------- #


def _artifact(gallery_images: list[GalleryImageRef]) -> CalibrationArtifact:
    """A minimal artifact carrying just the gallery refs build_gallery reads."""
    return CalibrationArtifact(
        format_version=1,
        threshold=0.5,
        aggregation="max",
        comparison=">=",
        gallery_vectors_file="x.gallery.npy",
        gallery_fingerprint="sha256:0",
        gallery_count=len(gallery_images),
        gallery_images=gallery_images,
        chosen_by=ChosenBy("m", 0, "p", 0.05, "look_alike"),
        metrics_at_threshold=MetricsAtThreshold(0.0, 0.0, 0.0, 1.0, 1, 1),
        aggregation_comparison=[],
        winner="max",
        sweep=[],
    )


def test_build_gallery_aligns_names_and_positions() -> None:
    artifact = _artifact(
        [
            GalleryImageRef("a.jpg", "lying", "side"),
            GalleryImageRef("b.jpg", "sitting", "front"),
        ]
    )
    raw = np.array([[3.0, 0.0], [0.0, 5.0]], dtype=np.float32)
    gallery, positions = predict.build_gallery(artifact, raw)
    assert gallery.names == ("a.jpg", "b.jpg")
    norms = np.linalg.norm(gallery.vectors, axis=1)
    np.testing.assert_allclose(norms, [1, 1], atol=1e-6)
    assert positions == {"a.jpg": ("lying", "side"), "b.jpg": ("sitting", "front")}


def test_find_artifacts_lists_yaml_sorted_excluding_companions(
    tmp_path: Path,
) -> None:
    (tmp_path / "z.yaml").write_text("")
    (tmp_path / "a.yaml").write_text("")
    (tmp_path / "thing.gallery.npy").write_bytes(b"")
    found = predict.find_artifacts(tmp_path)
    assert [p.name for p in found] == ["a.yaml", "z.yaml"]


def test_find_artifacts_missing_dir_is_empty() -> None:
    assert predict.find_artifacts(Path("does/not/exist")) == []
