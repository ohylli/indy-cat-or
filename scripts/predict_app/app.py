"""Streamlit app: upload an image and ask "is this Indy?".

The deliverable UI for the recognition pipeline -- distinct from the
``scripts/data_review`` dev tool. It is a thin, disposable layer over the
streamlit-free :mod:`predict_app.predict` core: load the frozen calibration
artifact (the live gallery + threshold), detect/crop/embed/score the upload, and
render a *text-first* verdict (the answer is readable as prose without seeing any
image; the crop and closest-match images are a sighted-helper supplement).

Run:
    uv run streamlit run scripts/predict_app/app.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import cast

# Running via `streamlit run scripts/predict_app/app.py` puts this file's own
# directory on sys.path but not its parent `scripts/`, where the `calibration`
# package lives. Add it so the package-qualified imports below resolve the same
# way they do under pytest (pythonpath=["scripts"]) and mypy.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import streamlit as st  # noqa: E402
from PIL import Image, ImageOps, UnidentifiedImageError  # noqa: E402

from calibration.artifact import CalibrationArtifact, load_artifact  # noqa: E402
from indycat.decision import Aggregation, Gallery  # noqa: E402
from indycat.detection import CatDetector  # noqa: E402
from indycat.embedding import Embedder  # noqa: E402
from predict_app import predict  # noqa: E402

UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]


@st.cache_resource
def load_detector() -> CatDetector:
    """The YOLO cat detector, loaded once per server (weights are heavy)."""
    return CatDetector()


@st.cache_resource
def load_embedder() -> Embedder:
    """The frozen DINOv2 embedder, loaded once per server."""
    return Embedder()


@st.cache_resource
def load_bundle(
    artifact_path: str,
) -> tuple[CalibrationArtifact, Gallery, dict[str, tuple[str, str]]]:
    """Load + validate the artifact and build the live gallery (cached by path)."""
    artifact, raw_vectors = load_artifact(Path(artifact_path))
    gallery, positions = predict.build_gallery(artifact, raw_vectors)
    return artifact, gallery, positions


def _pct(value: float) -> str:
    """Format a rate as a percentage, or ``n/a`` for NaN (no group present)."""
    return "n/a" if math.isnan(value) else f"{value * 100:.1f}%"


def pick_artifact() -> Path | None:
    """Choose the calibration artifact: auto if one, sidebar select if several."""
    artifacts = predict.find_artifacts()
    if not artifacts:
        st.error(
            "No calibration artifact found in `data/artifacts/`. Produce one with "
            "`uv run python scripts/calibrate.py` before using this app -- the "
            "frozen threshold and gallery live in that file."
        )
        return None
    if len(artifacts) == 1:
        return artifacts[0]
    names = [p.name for p in artifacts]
    chosen = st.sidebar.selectbox("Calibration artifact", options=names)
    return artifacts[names.index(chosen)]


def render_context(artifact: CalibrationArtifact) -> None:
    """Sidebar: what the frozen artifact decided, as text."""
    m = artifact.metrics_at_threshold
    st.sidebar.markdown("### Calibration")
    st.sidebar.markdown(
        f"- **Aggregation:** {artifact.aggregation}\n"
        f"- **Frozen threshold:** {artifact.threshold:.3f} "
        f"(`{artifact.comparison}`)\n"
        f"- **Gallery photos:** {artifact.gallery_count}\n"
        f"- **At this threshold** (calibration split): look-alike FPR "
        f"{_pct(m.fpr_look_alike)}, recall on Indy {_pct(m.recall_indy)}"
    )


def threshold_slider(artifact: CalibrationArtifact) -> float:
    """A threshold slider defaulting to the frozen value, ranged over the sweep."""
    cutoffs = [row.cutoff for row in artifact.sweep]
    lo = min(cutoffs) if cutoffs else 0.0
    hi = max(cutoffs) if cutoffs else 1.0
    default = min(max(artifact.threshold, lo), hi)
    value = st.slider(
        "Decision threshold",
        min_value=float(lo),
        max_value=float(hi),
        value=float(default),
        step=0.005,
        help=(
            "Defaults to the frozen calibrated threshold. Move it to explore "
            "borderline cases; the verdict and margin track the value you pick."
        ),
    )
    if abs(value - artifact.threshold) < 1e-9:
        st.caption(f"Using the frozen calibrated threshold ({artifact.threshold:.3f}).")
    else:
        st.warning(
            f"Threshold overridden to {value:.3f} (frozen calibrated value is "
            f"{artifact.threshold:.3f}). This is exploration, not the calibrated "
            "decision.",
            icon="⚠️",
        )
    return value


def render_crop(result: predict.CropResult, n_crops: int, detect_used: bool) -> None:
    """One crop's verdict + supporting images, heading-led and text-first."""
    if detect_used and n_crops > 1:
        st.subheader(f"Crop {result.index + 1} of {n_crops}")

    verdict = "Indy" if result.is_indy else "not Indy"
    relation = "≥" if result.is_indy else "<"
    st.markdown(
        f"**This crop: {verdict}.** Score {result.score:.3f} {relation} threshold "
        f"({result.margin:+.3f} margin)."
    )
    if result.detection is not None:
        d = result.detection
        st.markdown(
            f"- Cat detection confidence: {d.confidence:.3f}\n"
            f"- Crop covers {d.area_fraction * 100:.1f}% of the frame"
        )
    else:
        st.markdown("- Detection was off; the whole frame was embedded.")

    pose = ", ".join(p for p in (result.best_position, result.best_view) if p)
    pose_text = f" ({pose})" if pose else ""
    st.markdown(f"- Closest gallery photo: **{result.best_name}**{pose_text}")

    crop_col, match_col = st.columns(2)
    with crop_col:
        st.image(
            result.crop_image,
            caption="Embedded crop" if detect_used else "Embedded frame",
            width="stretch",
        )
    with match_col:
        match_path = predict.INDY_DIR / result.best_name
        if match_path.exists():
            st.image(str(match_path), caption=result.best_name, width="stretch")
        else:
            st.info(f"Gallery image file not found: {result.best_name}")


def render_result(result: predict.ClassifyResult, upload: Image.Image) -> None:
    """Render the whole verdict: headline first, then per-crop detail."""
    st.subheader("Result")

    if result.no_cat:
        st.warning(
            "No cat detected. Nothing was embedded (the full frame is deliberately "
            "not used as a fallback). Try toggling detection off to embed the whole "
            "image instead.",
            icon="🐈",
        )
    elif result.overall_is_indy:
        st.success(f"This is Indy. (threshold {result.threshold:.3f})", icon="✅")
    else:
        st.error(
            f"Not recognized as Indy. (threshold {result.threshold:.3f})", icon="❌"
        )

    if result.detect_used and len(result.crops) > 1:
        indy_crops = sum(c.is_indy for c in result.crops)
        st.markdown(
            f"Detected **{len(result.crops)} cats**; Indy if any one matches "
            f"({indy_crops} did)."
        )

    st.subheader("Uploaded image")
    st.image(upload, caption="Uploaded image", width="stretch")

    for crop in result.crops:
        st.divider()
        render_crop(crop, len(result.crops), result.detect_used)


def main() -> None:
    st.set_page_config(page_title="Is it Indy?", layout="wide")
    st.title("Is it Indy?")
    st.write(
        "Upload a photo and the verifier compares it to Indy's gallery: is this "
        "the specific cat Indy, or not. The answer, score, and closest match are "
        "all reported as text."
    )

    artifact_path = pick_artifact()
    if artifact_path is None:
        return
    artifact, gallery, positions = load_bundle(str(artifact_path))
    render_context(artifact)

    threshold = threshold_slider(artifact)

    with st.form("classify"):
        upload = st.file_uploader("Image to classify", type=UPLOAD_TYPES)
        detect = st.toggle(
            "Detect and crop the cat first",
            value=True,
            help=(
                "On: find the cat and embed only the crop (the calibrated path). "
                "Off: embed the whole frame -- which may latch onto the background."
            ),
        )
        submitted = st.form_submit_button("Classify")

    if not submitted:
        return
    if upload is None:
        st.error("Choose an image first.")
        return

    try:
        # EXIF-transpose to match the upright gallery (the builders use the same
        # correction); st.file_uploader hands raw bytes that Image.open won't rotate.
        image = ImageOps.exif_transpose(Image.open(upload))
    except (UnidentifiedImageError, OSError):
        st.error("That file could not be read as an image.")
        return
    if image is None:  # exif_transpose can return None on a closed/empty image
        st.error("That file could not be read as an image.")
        return

    with st.status("Detecting, embedding and scoring…", expanded=False) as status:
        result = predict.classify(
            image,
            detector=load_detector(),
            embedder=load_embedder(),
            gallery=gallery,
            positions=positions,
            threshold=threshold,
            aggregation=cast(Aggregation, artifact.aggregation),
            comparison=artifact.comparison,
            detect=detect,
        )
        status.update(label="Done.", state="complete")

    render_result(result, image)


if __name__ == "__main__":
    main()
