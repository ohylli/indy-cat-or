"""Mapping review: two renderings of ``images/indy/mapping.csv``.

- ``render_rows``: each photo is a heading-led section with text fields -- the
  most accessible shape Streamlit can reasonably produce (navigable by heading).
- ``render_grid``: the literal ``st.dataframe`` + ``ImageColumn`` table, which
  has known accessibility limits -- kept so the difference is felt directly
  rather than assumed.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from common import INDY_DIR, MAPPING_CSV, static_url, sync_static_images

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
