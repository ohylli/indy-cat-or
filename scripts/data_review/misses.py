"""Detect-miss review: Oxford cats in the catalog but absent from the embeddings.

The embed stage re-runs YOLO over every Oxford cat image and drops images where
no cat is detected (~4% of the 2371). Those misses are catalog rows whose
``source_filename`` has no matching row in
``data/embeddings/oxford/metadata.csv``. This view shows them as a paginated
image grid so a sighted helper can eyeball *why* the detector missed each one --
cat too small, occluded, odd pose, etc.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from common import OXFORD_CATALOG_CSV, OXFORD_IMAGES_DIR, OXFORD_METADATA_CSV

COLUMNS = 4
PER_PAGE = 24


@st.cache_data
def load_misses() -> list[str] | None:
    """Catalog filenames with no embedding row, sorted. ``None`` if a file is absent.

    The miss set is ``catalog - metadata`` on ``source_filename``: every cat
    image the catalog knows about that the embed stage did not write a row for.
    """
    if not OXFORD_CATALOG_CSV.exists() or not OXFORD_METADATA_CSV.exists():
        return None
    catalog = pd.read_csv(OXFORD_CATALOG_CSV, dtype=str)
    metadata = pd.read_csv(OXFORD_METADATA_CSV, dtype=str)
    embedded = set(metadata["source_filename"])
    return sorted(n for n in catalog["source_filename"] if n not in embedded)


def render_misses(misses: list[str] | None) -> None:
    """Paginated grid of the missed images, ``COLUMNS`` per row, filename caption."""
    if misses is None:
        st.warning(
            "Need both `images/oxford-iiit-pet/catalog.csv` and "
            "`data/embeddings/oxford/metadata.csv`. Run "
            "`uv run python scripts/build_oxford_negatives.py` to generate them."
        )
        return
    if not misses:
        st.success("No misses: every catalog image has an embedding row.")
        return

    n_pages = (len(misses) + PER_PAGE - 1) // PER_PAGE
    st.write(
        f"{len(misses)} cat images were not detected, so they were never "
        f"embedded. Showing {PER_PAGE} per page across {n_pages} pages."
    )

    page = 1
    if n_pages > 1:
        page = st.number_input("Page", min_value=1, max_value=n_pages, value=1, step=1)
    start = (page - 1) * PER_PAGE
    page_items = misses[start : start + PER_PAGE]

    for i in range(0, len(page_items), COLUMNS):
        cols = st.columns(COLUMNS)
        for col, name in zip(cols, page_items[i : i + COLUMNS], strict=False):
            with col:
                path = OXFORD_IMAGES_DIR / name
                if path.exists():
                    st.image(str(path), caption=name, width="stretch")
                else:
                    st.warning(f"Missing image file: {name}")
