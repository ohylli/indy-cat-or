"""Streamlit review app for the Indy image mapping and crop results.

Gives Indy's owner a screen-reader-friendly, image-by-image view of the project
data, with three views selectable in the sidebar:

- "Rows": each photo as a heading-led section of text fields -- the most
  accessible rendering of the ``images/indy/mapping.csv`` labels.
- "Data grid": the literal ``st.dataframe`` rendering of the same mapping data,
  kept alongside "Rows" so the screen-reader difference is felt directly.
- "Crop review": the detect-and-crop results, each original beside its crop with
  the detection metrics -- so the crops can be checked for correctness.

The mapping and crop views live in their own modules (``mapping``, ``crops``);
shared paths and static-image serving live in ``common``.

Run:
    uv run streamlit run scripts/data_review/data_review.py
"""

from __future__ import annotations

import crops
import mapping
import streamlit as st


def main() -> None:
    st.set_page_config(page_title="Indy data review", layout="wide")
    st.title("Indy image mapping review")

    df = mapping.load_mapping()
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
        mapping.render_rows(df)
    elif view == "Data grid":
        mapping.render_grid(df)
    else:
        crops.render_crops(crops.load_detections(), df)


if __name__ == "__main__":
    main()
