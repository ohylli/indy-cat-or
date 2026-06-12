"""Streamlit review view for the Indy image mapping and crop results.

Throwaway experiment with these goals:

1. Give Indy's owner a scrollable, image-by-image view of the mapping metadata
   so the labels in ``images/indy/mapping.csv`` can be sanity-checked.
2. Evaluate Streamlit's screen-reader accessibility *in practice* (an open
   question in this project), by offering two views of the same mapping data:
     - "Rows": each photo is a heading-led section with text fields. The most
       accessible shape Streamlit can reasonably produce (navigable by heading).
     - "Data grid": the literal ``st.dataframe`` + ``ImageColumn`` table, which
       has known accessibility limits -- included so the difference is felt
       directly rather than assumed.
3. Review the detect-and-crop results in ``data/crops/indy/detections.csv`` via
   a third view, "Crop review": a grid showing each detection's original image
   next to its crop, plus the detection metrics and the mapping notes -- so the
   detector's crops can be eyeballed for correctness.

Run:
    uv run streamlit run experiments/review_mapping.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

INDY_DIR = Path(__file__).resolve().parent.parent / "images" / "indy"
MAPPING_CSV = INDY_DIR / "mapping.csv"

CROPS_DIR = Path(__file__).resolve().parent.parent / "data" / "crops" / "indy"
DETECTIONS_CSV = CROPS_DIR / "detections.csv"

# Streamlit serves files under this script's sibling `static/` dir at the URL
# `app/static/<file>` when `server.enableStaticServing` is on (see
# .streamlit/config.toml). The grid references images this way -- a short path
# rather than an inline base64 data URL a screen reader would read out in full.
STATIC_DIR = Path(__file__).resolve().parent / "static"
APP_STATIC_URL = "app/static"

# Columns the owner does not need for a label review.
HIDDEN_COLUMNS = ("number", "original_filename")


@st.cache_data
def load_mapping() -> pd.DataFrame | None:
    """Load the mapping CSV as strings (no type coercion of the label fields).

    Returns ``None`` if the file is absent, so callers can report it rather than
    crash with a ``FileNotFoundError``.
    """
    if not MAPPING_CSV.exists():
        return None
    return pd.read_csv(MAPPING_CSV, dtype=str)


@st.cache_data
def load_detections() -> pd.DataFrame | None:
    """Load the detect-and-crop results CSV as strings, or ``None`` if absent.

    ``keep_default_na=False`` renders blank cells (no-detection rows, where the
    crop/confidence fields are empty) as empty strings rather than ``NaN``.
    """
    if not DETECTIONS_CSV.exists():
        return None
    return pd.read_csv(DETECTIONS_CSV, dtype=str, keep_default_na=False)


def sync_static_images(filenames: list[str], src_dir: Path = INDY_DIR) -> None:
    """Copy images from ``src_dir`` into ``static/`` so the app can serve them.

    The images' source of truth stays ``src_dir``; this keeps a local mirror in
    sync, copying only files that are missing or have changed since last run.
    Originals (``images/indy``) and crops (``data/crops/indy``) share the mirror;
    crop filenames carry a ``_crop{N}`` suffix so they never collide.
    """
    STATIC_DIR.mkdir(exist_ok=True)
    for name in filenames:
        src = src_dir / name
        if not src.exists():
            continue
        dst = STATIC_DIR / name
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(src, dst)


def static_url(filename: str, src_dir: Path = INDY_DIR) -> str | None:
    """URL for serving an image via static serving, or ``None`` if missing."""
    if filename and (src_dir / filename).exists():
        return f"{APP_STATIC_URL}/{filename}"
    return None


def info_fields(df: pd.DataFrame, include_filename: bool = False) -> list[str]:
    """Label columns to show: everything except hidden and optionally the filename
    handle."""
    return [
        c
        for c in df.columns
        if c not in HIDDEN_COLUMNS and (include_filename or c != "new_filename")
    ]


def render_rows(df: pd.DataFrame) -> None:
    """Heading-led, one-section-per-photo layout (accessibility-leaning)."""
    fields = info_fields(df)
    for _, row in df.iterrows():
        filename = row["new_filename"]
        st.subheader(filename)
        image_col, info_col = st.columns([1, 2])
        with image_col:
            path = INDY_DIR / filename
            if path.exists():
                st.image(str(path), caption=filename, width="stretch")
            else:
                st.warning(f"Missing image file: {filename}")
        with info_col:
            for field in fields:
                label = field.replace("_", " ").capitalize()
                st.markdown(f"**{label}:** {row[field]}")
        st.divider()


def render_grid(df: pd.DataFrame) -> None:
    """Literal data grid with inline thumbnails (known a11y limits)."""
    fields = info_fields(df, True)
    filenames = df["new_filename"].tolist()
    sync_static_images(filenames)
    grid = pd.DataFrame(
        {
            "image": [static_url(name) for name in filenames],
            **{field: df[field] for field in fields},
        }
    )
    st.dataframe(
        grid,
        hide_index=True,
        row_height=120,
        # "content" sizes the grid to all rows (no inner scroll box -- the page
        # scrolls instead), rather than the default ~10-default-row height cap.
        height="content",
        column_config={
            "image": st.column_config.ImageColumn("image", width="medium"),
        },
    )


def render_crops(df_detections: pd.DataFrame | None, df_mapping: pd.DataFrame) -> None:
    """Crop-review grid: original image beside its crop, with detection metrics.

    One row per detection (an image with two cats yields two rows). No-detection
    rows are kept: the original shows, the crop cell is blank.
    """
    if df_detections is None:
        st.warning(
            "Detection results not found at `data/crops/indy/detections.csv`. "
            "Run `uv run python scripts/detect_indy_gallery.py` to generate them."
        )
        return

    merged = df_detections.merge(
        df_mapping[["new_filename", "notes"]],
        left_on="source_filename",
        right_on="new_filename",
        how="left",
    )

    sync_static_images(merged["source_filename"].tolist(), INDY_DIR)
    sync_static_images([c for c in merged["crop_filename"] if c], CROPS_DIR)

    grid = pd.DataFrame(
        {
            "original": [
                static_url(name, INDY_DIR) for name in merged["source_filename"]
            ],
            "crop": [static_url(name, CROPS_DIR) for name in merged["crop_filename"]],
            "source_filename": merged["source_filename"],
            "n_detections": merged["n_detections"],
            "crop_filename": merged["crop_filename"],
            "confidence": merged["confidence"],
            "area_fraction": merged["area_fraction"],
            "notes": merged["notes"],
        }
    )
    st.dataframe(
        grid,
        hide_index=True,
        row_height=120,
        height="content",
        column_config={
            "original": st.column_config.ImageColumn("original", width="medium"),
            "crop": st.column_config.ImageColumn("crop", width="medium"),
        },
    )


def main() -> None:
    st.set_page_config(page_title="Indy mapping review", layout="wide")
    st.title("Indy image mapping review")

    df = load_mapping()
    if df is None:
        st.error(
            "Mapping CSV not found at `images/indy/mapping.csv`. It is the source "
            "of truth for this review -- there is nothing to show without it."
        )
        return
    st.write(f"{len(df)} photos. Reviewing labels in `images/indy/mapping.csv`.")

    view = st.sidebar.radio(
        "View",
        options=("Rows (accessible)", "Data grid", "Crop review"),
        help=(
            "Rows/Data grid are two renderings of the mapping data, for comparing "
            "screen-reader behaviour. Crop review shows the detect-and-crop results."
        ),
    )

    if view == "Rows (accessible)":
        render_rows(df)
    elif view == "Data grid":
        render_grid(df)
    else:
        render_crops(load_detections(), df)


if __name__ == "__main__":
    main()
