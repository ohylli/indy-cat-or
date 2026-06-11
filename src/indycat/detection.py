"""Detect-and-crop stage: find cats in an image and crop to them.

This module is the first pipeline stage (image -> detect -> crop). It is
deliberately I/O-free: the detector takes an already-opened PIL image and
returns plain data, so callers decide where images come from (disk, web
request, ...) and what to do with the boxes. Cropping is a separate pure
function so its geometry can be tested without a model.

License note: the detector backend is ultralytics YOLO, which is AGPL-3.0
(copyleft extends to running it as a network service). Fine for local and
open-source use; if that ever becomes a problem, torchvision's COCO-pretrained
detection models (Faster R-CNN et al., BSD-3) can replace it behind the same
CatDetector interface.
"""

from dataclasses import dataclass

from PIL import Image
from ultralytics import YOLO

#: "cat" in the COCO class list the pretrained detector was trained on.
COCO_CAT_CLASS_ID = 15


@dataclass(frozen=True)
class Detection:
    """One detected cat."""

    confidence: float
    #: Pixel coordinates (x1, y1, x2, y2), origin top-left, x2/y2 exclusive.
    box_xyxy: tuple[int, int, int, int]
    #: Box area as a fraction of the whole image; ~1.0 means cropping is a
    #: no-op, very small values suggest a misdetection or a distant cat.
    area_fraction: float


@dataclass(frozen=True)
class DetectionResult:
    """All cat detections for one image.

    An empty ``detections`` list is the explicit "no cat found" case; callers
    must handle it rather than silently falling back to the full frame.
    """

    image_size: tuple[int, int]  # (width, height)
    detections: list[Detection]  # sorted by confidence, highest first


class CatDetector:
    """COCO-pretrained YOLO detector restricted to the "cat" class.

    Default model: measured on the 35 Indy photos (2026-06-11), yolo11n missed
    the cat in 8 of them and yolo11m in 2, while yolo11x found it in all 35.
    Detection misses are the costly failure, so the large variant is the
    default; pass a smaller one (yolo11m.pt, yolo11n.pt) if footprint matters.
    """

    def __init__(self, model: str = "yolo11x.pt", min_confidence: float = 0.25) -> None:
        self._model = YOLO(model)
        self.min_confidence = min_confidence

    def detect(self, image: Image.Image) -> DetectionResult:
        # YOLO expects 3-channel input; flatten alpha/palette images.
        rgb = image.convert("RGB")
        results = self._model.predict(
            rgb,
            classes=[COCO_CAT_CLASS_ID],
            conf=self.min_confidence,
            verbose=False,
        )
        width, height = rgb.size
        detections = []
        boxes = results[0].boxes
        for box in boxes if boxes is not None else []:
            x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    confidence=float(box.conf[0]),
                    box_xyxy=(x1, y1, x2, y2),
                    area_fraction=((x2 - x1) * (y2 - y1)) / (width * height),
                )
            )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return DetectionResult(image_size=(width, height), detections=detections)


def crop_with_margin(
    image: Image.Image,
    box_xyxy: tuple[int, int, int, int],
    margin: float = 0.1,
) -> Image.Image:
    """Crop ``image`` to ``box_xyxy`` expanded by ``margin`` on every side.

    The margin is a fraction of the box's own width/height (0.1 adds 10% of
    the box width to the left and right, 10% of its height above and below),
    clamped to the image bounds.
    """
    x1, y1, x2, y2 = box_xyxy
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"box must have positive width and height, got {box_xyxy}")
    if margin < 0:
        raise ValueError(f"margin must be non-negative, got {margin}")
    dx = round((x2 - x1) * margin)
    dy = round((y2 - y1) * margin)
    width, height = image.size
    return image.crop(
        (max(0, x1 - dx), max(0, y1 - dy), min(width, x2 + dx), min(height, y2 + dy))
    )
