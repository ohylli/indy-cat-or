"""Shared paths and static-image serving for the data-review app."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Running via `streamlit run scripts/data_review/data_review.py` puts this file's
# own directory on sys.path (for the bare `import crops`/`mapping`/`misses`) but
# not its parent `scripts/`, where `_common` and the `calibration` package live.
# Add it so the imports below resolve the same way they do under pytest
# (pythonpath=["scripts"]) and mypy, matching `scripts/predict_app/app.py`.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _common import EmbeddingsVariant  # noqa: E402
from calibration.manifest import (  # noqa: E402
    DEFAULT_DETECT,
    DEFAULT_MARGIN,
    DEFAULT_MODEL,
    EMBEDDINGS_ROOT,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

INDY_DIR = REPO_ROOT / "images" / "indy"
MAPPING_CSV = INDY_DIR / "mapping.csv"

CROPS_DIR = REPO_ROOT / "data" / "crops" / "indy"
DETECTIONS_CSV = CROPS_DIR / "detections.csv"

OXFORD_DIR = REPO_ROOT / "images" / "oxford-iiit-pet"
OXFORD_IMAGES_DIR = OXFORD_DIR / "images"
OXFORD_CATALOG_CSV = OXFORD_DIR / "catalog.csv"

# Oxford detect-miss review (catalog minus embedding metadata) reads the
# *baseline crop-on* variant's metadata, not a fixed flat path. Detection misses
# depend on the detector + min_confidence + the detect toggle, NOT on the
# embedding model (invariant #3 of docs/embeddings_provenance.md), so the
# baseline crop-on metadata is the right, stable source whatever model is being
# tried -- the misses are the same regardless of backbone.
OXFORD_BASELINE_VARIANT = EmbeddingsVariant(
    model_id=DEFAULT_MODEL, detect=DEFAULT_DETECT, margin=DEFAULT_MARGIN
)
OXFORD_METADATA_CSV = (
    OXFORD_BASELINE_VARIANT.dir(EMBEDDINGS_ROOT / "oxford") / "metadata.csv"
)

# Streamlit serves files under the main script's sibling `static/` dir at the URL
# `app/static/<file>` when `server.enableStaticServing` is on (see
# .streamlit/config.toml). The grid references images this way -- a short path
# rather than an inline base64 data URL a screen reader would read out in full.
STATIC_DIR = Path(__file__).resolve().parent / "static"
APP_STATIC_URL = "app/static"


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
