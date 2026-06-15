"""Crop review: the detect-and-crop results in ``data/crops/indy/detections.csv``.

A grid showing each detection's original image beside its crop, plus the
detection metrics and the mapping notes -- so the detector's crops can be
eyeballed for correctness.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from common import CROPS_DIR, DETECTIONS_CSV, INDY_DIR, static_url, sync_static_images


@st.cache_data
def load_detections() -> pd.DataFrame | None:
    """Load the detect-and-crop results CSV as strings, or ``None`` if absent.

    ``keep_default_na=False`` renders blank cells (no-detection rows, where the
    crop/confidence fields are empty) as empty strings rather than ``NaN``.
    """
    if not DETECTIONS_CSV.exists():
        return None
    return pd.read_csv(DETECTIONS_CSV, dtype=str, keep_default_na=False)


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
