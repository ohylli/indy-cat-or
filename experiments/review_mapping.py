"""Streamlit review view for the Indy image mapping CSV.

Throwaway experiment with two goals:

1. Give Indy's owner a scrollable, image-by-image view of the mapping metadata
   so the labels in ``images/indy/mapping.csv`` can be sanity-checked.
2. Evaluate Streamlit's screen-reader accessibility *in practice* (an open
   question in this project), by offering two layouts of the same data:
     - "Rows": each photo is a heading-led section with text fields. The most
       accessible shape Streamlit can reasonably produce (navigable by heading).
     - "Data grid": the literal ``st.dataframe`` + ``ImageColumn`` table, which
       has known accessibility limits -- included so the difference is felt
       directly rather than assumed.

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

# Streamlit serves files under this script's sibling `static/` dir at the URL
# `app/static/<file>` when `server.enableStaticServing` is on (see
# .streamlit/config.toml). The grid references images this way -- a short path
# rather than an inline base64 data URL a screen reader would read out in full.
STATIC_DIR = Path(__file__).resolve().parent / "static"
APP_STATIC_URL = "app/static"

# Columns the owner does not need for a label review.
HIDDEN_COLUMNS = ("number", "original_filename")


@st.cache_data
def load_mapping() -> pd.DataFrame:
    """Load the mapping CSV as strings (no type coercion of the label fields)."""
    return pd.read_csv(MAPPING_CSV, dtype=str)


def sync_static_images(filenames: list[str]) -> None:
    """Copy the mapping's photos into ``static/`` so the app can serve them.

    The photos' source of truth stays ``images/indy``; this keeps a local mirror
    in sync, copying only files that are missing or have changed since last run.
    """
    STATIC_DIR.mkdir(exist_ok=True)
    for name in filenames:
        src = INDY_DIR / name
        if not src.exists():
            continue
        dst = STATIC_DIR / name
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(src, dst)


def static_url(filename: str) -> str | None:
    """URL for serving an image via static serving, or ``None`` if missing."""
    if (INDY_DIR / filename).exists():
        return f"{APP_STATIC_URL}/{filename}"
    return None


def info_fields(df: pd.DataFrame) -> list[str]:
    """Label columns to show: everything except hidden and the filename handle."""
    return [c for c in df.columns if c not in HIDDEN_COLUMNS and c != "new_filename"]


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
    fields = info_fields(df)
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


def main() -> None:
    st.set_page_config(page_title="Indy mapping review", layout="wide")
    st.title("Indy image mapping review")

    df = load_mapping()
    st.write(f"{len(df)} photos. Reviewing labels in `images/indy/mapping.csv`.")

    layout = st.sidebar.radio(
        "Layout",
        options=("Rows (accessible)", "Data grid"),
        help="Two renderings of the same data, for comparing screen-reader behaviour.",
    )

    if layout == "Rows (accessible)":
        render_rows(df)
    else:
        render_grid(df)


if __name__ == "__main__":
    main()
