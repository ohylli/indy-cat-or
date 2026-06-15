"""Run the detect-and-crop stage over all Indy photos.

By default writes one crop per photo -- the highest-confidence detection -- to
the output folder (for sighted spot-review). Pass ``--all-crops`` to instead
write a crop for every detection. Either way it prints a textual per-image
report plus an aggregate summary (the primary, screen-reader-friendly
verification) describing *all* detections, and optionally records every
detection in a CSV.

Usage:
    uv run python scripts/detect_indy_gallery.py
    uv run python scripts/detect_indy_gallery.py --all-crops
    uv run python scripts/detect_indy_gallery.py --csv data/crops/indy/detections.csv
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from indycat.detection import CatDetector, DetectionResult, crop_with_margin

REPO_ROOT = Path(__file__).parent.parent

#: Below this confidence a detection is called out as suspect in the report.
LOW_CONFIDENCE = 0.8
#: Above this fraction of the frame, cropping is nearly a no-op — worth noting.
NEAR_FULL_FRAME = 0.9

CSV_COLUMNS = [
    "source_filename",
    "n_detections",
    "detection_index",
    "crop_filename",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "image_width",
    "image_height",
    "area_fraction",
]


@dataclass
class ImageReport:
    filename: str
    result: DetectionResult
    #: Aligned 1:1 with ``result.detections``; ``None`` where no crop was saved
    #: (secondary detections when not running with --all-crops).
    crop_filenames: list[str | None]


def load_image(path: Path) -> Image.Image:
    """Open an image with EXIF orientation applied.

    Phone photos often store rotation as an EXIF tag; applying it here makes
    the box coordinates and the saved crops match the photo as displayed.
    """
    return ImageOps.exif_transpose(Image.open(path))


def save_crop(crop: Image.Image, path: Path) -> None:
    if path.suffix.lower() in (".jpg", ".jpeg"):
        # The crops feed the embedding stage; don't recompress harshly.
        crop.convert("RGB").save(path, quality=95)
    else:
        crop.save(path)


def process_image(
    path: Path, detector: CatDetector, out_dir: Path, margin: float, all_crops: bool
) -> ImageReport:
    image = load_image(path)
    result = detector.detect(image)
    # Detections are sorted highest-confidence-first, so index 1 is the top
    # crop. By default only it is written; --all-crops writes every detection.
    crop_filenames: list[str | None] = []
    for index, detection in enumerate(result.detections, start=1):
        if all_crops or index == 1:
            crop_name = f"{path.stem}_crop{index}{path.suffix.lower()}"
            save_crop(
                crop_with_margin(image, detection.box_xyxy, margin),
                out_dir / crop_name,
            )
            crop_filenames.append(crop_name)
        else:
            crop_filenames.append(None)
    return ImageReport(path.name, result, crop_filenames)


def describe(report: ImageReport) -> str:
    """One textual report block per image; anomalies are spelled out in words."""
    detections = report.result.detections
    if not detections:
        return f"{report.filename}: NO CAT DETECTED"
    count = len(detections)
    lines = []
    if count > 1:
        lines.append(f"{report.filename}: MULTIPLE CATS ({count})")
    for detection, crop_name in zip(detections, report.crop_filenames, strict=True):
        flags = ""
        if detection.confidence < LOW_CONFIDENCE:
            flags += ", LOW CONFIDENCE"
        if detection.area_fraction > NEAR_FULL_FRAME:
            flags += ", NEARLY FULL FRAME"
        crop_note = f"-> {crop_name}" if crop_name else "(crop not saved)"
        line = (
            f"confidence {detection.confidence:.2f}, "
            f"{detection.area_fraction:.0%} of frame{flags} {crop_note}"
        )
        if count == 1:
            lines.append(f"{report.filename}: 1 cat, {line}")
        else:
            lines.append(f"  {line}")
    return "\n".join(lines)


def summarize(reports: list[ImageReport], out_dir: Path) -> str:
    """Aggregate verdict: how many images behaved, and which ones did not."""
    none = [r.filename for r in reports if not r.result.detections]
    multiple = [
        f"{r.filename} ({len(r.result.detections)})"
        for r in reports
        if len(r.result.detections) > 1
    ]
    # Three decimals so a 0.799 doesn't print as an apparently-fine "0.80".
    low = [
        f"{r.filename} ({r.result.detections[0].confidence:.3f})"
        for r in reports
        if r.result.detections and r.result.detections[0].confidence < LOW_CONFIDENCE
    ]
    full_frame = [
        r.filename
        for r in reports
        if any(d.area_fraction > NEAR_FULL_FRAME for d in r.result.detections)
    ]
    clean = sum(
        1
        for r in reports
        if len(r.result.detections) == 1
        and r.result.detections[0].confidence >= LOW_CONFIDENCE
    )
    n_crops = sum(1 for r in reports for c in r.crop_filenames if c)
    lines = [
        f"Summary: {len(reports)} images, {n_crops} crops written to {out_dir}",
        f"  {clean} images with exactly one cat at confidence >= {LOW_CONFIDENCE}",
    ]
    if none:
        lines.append(f"  {len(none)} with NO cat detected: {', '.join(none)}")
    if multiple:
        lines.append(f"  {len(multiple)} with multiple cats: {', '.join(multiple)}")
    if low:
        lines.append(
            f"  {len(low)} with best confidence below {LOW_CONFIDENCE}: "
            f"{', '.join(low)}"
        )
    if full_frame:
        lines.append(
            f"  {len(full_frame)} with a detection covering over "
            f"{NEAR_FULL_FRAME:.0%} of the frame: {', '.join(full_frame)}"
        )
    if not (none or multiple or low or full_frame):
        lines.append("  No anomalies.")
    return "\n".join(lines)


def write_csv(reports: list[ImageReport], csv_path: Path) -> None:
    """One row per detection; zero-detection images get a row with empty fields."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for report in reports:
            width, height = report.result.image_size
            if not report.result.detections:
                writer.writerow(
                    [report.filename, 0, "", "", "", "", "", "", "", width, height, ""]
                )
                continue
            n = len(report.result.detections)
            for index, (detection, crop_name) in enumerate(
                zip(report.result.detections, report.crop_filenames, strict=True),
                start=1,
            ):
                x1, y1, x2, y2 = detection.box_xyxy
                writer.writerow(
                    [
                        report.filename,
                        n,
                        index,
                        crop_name or "",
                        f"{detection.confidence:.4f}",
                        x1,
                        y1,
                        x2,
                        y2,
                        width,
                        height,
                        f"{detection.area_fraction:.4f}",
                    ]
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect and crop the cat in every Indy photo."
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
        default=REPO_ROOT / "data" / "crops" / "indy",
        help="folder for cropped images (default: data/crops/indy)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="also write detection details to this CSV",
    )
    parser.add_argument(
        "--all-crops",
        action="store_true",
        help=(
            "save a crop for every detection "
            "(default: only the highest-confidence one)"
        ),
    )
    parser.add_argument(
        "--model",
        default="yolo11x.pt",
        help="ultralytics model name or path (default: yolo11x.pt)",
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
    args = parser.parse_args()

    photos = sorted(
        p
        for p in args.images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )
    if not photos:
        raise SystemExit(f"no images found in {args.images_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stale = [p for p in args.out_dir.iterdir() if p.is_file()]
    for p in stale:
        p.unlink()
    if stale:
        print(f"Cleared {len(stale)} old files from {args.out_dir}")

    detector = CatDetector(model=args.model, min_confidence=args.min_confidence)
    reports = []
    for path in photos:
        report = process_image(
            path, detector, args.out_dir, args.margin, args.all_crops
        )
        print(describe(report))
        reports.append(report)

    print()
    print(summarize(reports, args.out_dir))
    if args.csv is not None:
        write_csv(reports, args.csv)
        print(f"Detection details written to {args.csv}")


if __name__ == "__main__":
    main()
