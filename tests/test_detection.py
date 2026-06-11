"""Tests for the detect-and-crop stage.

The crop-geometry tests are pure and fast. The detector test needs the YOLO
weights (auto-downloaded on first run) and a real photo, so it is marked
``slow``; deselect with ``pytest -m "not slow"``.
"""

from pathlib import Path

import pytest
from PIL import Image

from indycat.detection import CatDetector, crop_with_margin

INDY_DIR = Path(__file__).parent.parent / "images" / "indy"


def blank_image(width: int = 100, height: int = 100) -> Image.Image:
    return Image.new("RGB", (width, height))


def test_crop_without_margin_is_exact_box() -> None:
    crop = crop_with_margin(blank_image(), (40, 40, 60, 60), margin=0.0)
    assert crop.size == (20, 20)


def test_crop_margin_expands_each_side() -> None:
    # 20x20 box, 10% margin -> 2 px on every side -> 24x24.
    crop = crop_with_margin(blank_image(), (40, 40, 60, 60), margin=0.1)
    assert crop.size == (24, 24)


def test_crop_margin_scales_with_box_not_image() -> None:
    # 50% margin on a 20x40 box adds 10 px left/right and 20 px top/bottom.
    crop = crop_with_margin(blank_image(200, 200), (100, 100, 120, 140), margin=0.5)
    assert crop.size == (40, 80)


def test_crop_clamps_to_image_bounds() -> None:
    # Box touching the top-left corner: margin cannot go below (0, 0).
    crop = crop_with_margin(blank_image(), (0, 0, 20, 20), margin=0.5)
    assert crop.size == (30, 30)
    # And against the bottom-right edge.
    crop = crop_with_margin(blank_image(), (80, 80, 100, 100), margin=0.5)
    assert crop.size == (30, 30)


def test_crop_full_frame_box_is_noop() -> None:
    crop = crop_with_margin(blank_image(), (0, 0, 100, 100), margin=0.2)
    assert crop.size == (100, 100)


def test_crop_rejects_degenerate_box() -> None:
    with pytest.raises(ValueError):
        crop_with_margin(blank_image(), (50, 50, 50, 60))
    with pytest.raises(ValueError):
        crop_with_margin(blank_image(), (60, 50, 50, 60))


def test_crop_rejects_negative_margin() -> None:
    with pytest.raises(ValueError):
        crop_with_margin(blank_image(), (40, 40, 60, 60), margin=-0.1)


@pytest.mark.slow
def test_detector_finds_one_cat_in_indy_photo() -> None:
    photos = sorted(p for p in INDY_DIR.iterdir() if p.suffix.lower() != ".csv")
    if not photos:
        pytest.skip("Indy photos not present (gitignored, local only)")
    result = CatDetector().detect(Image.open(photos[0]))
    assert len(result.detections) == 1
    best = result.detections[0]
    assert best.confidence > 0.5
    assert 0.0 < best.area_fraction <= 1.0
    x1, y1, x2, y2 = best.box_xyxy
    width, height = result.image_size
    assert 0 <= x1 < x2 <= width
    assert 0 <= y1 < y2 <= height
