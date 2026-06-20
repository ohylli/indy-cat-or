"""Shared rendering primitives for the calibration and evaluation reports.

The accessibility-critical markup -- the scoped-header ``<table>`` scaffolding,
the NaN-aware score formatters, and the ``<figure>``/relative-``src`` helpers --
lives here so the calibrate report (:mod:`calibration.report_html`) and the
future evaluate report cannot drift. This mirrors how :mod:`calibration.metrics`
keeps the *numbers* in one place: this module keeps the *screen-reader table
semantics* in one place. No measurement happens here; it only formats.
"""

from __future__ import annotations

import html
import math
import os
from collections.abc import Sequence
from pathlib import Path

#: Minimal styling -- just enough to lay two figures of a risk row side by side
#: and keep the scoped-header tables legible. No JavaScript; semantics carry the
#: report. Shared so both reports render identical accessible markup.
HTML_STYLE = """
body { font-family: sans-serif; max-width: 60rem; margin: 2rem auto; padding: 0 1rem; }
table { border-collapse: collapse; margin: 0.5rem 0; }
th, td { border: 1px solid #ccc; padding: 0.2rem 0.6rem; text-align: right; }
th[scope="row"], caption { text-align: left; }
figure { display: inline-block; margin: 0.3rem; vertical-align: top; }
figure img { max-height: 12rem; max-width: 16rem; }
figcaption { font-size: 0.85rem; word-break: break-all; max-width: 16rem; }
ol.risks > li { margin-bottom: 1rem; }
"""


def fmt(value: float) -> str:
    """Format a score for plain text (NaN -> dash)."""
    return "  -  " if math.isnan(value) else f"{value:.3f}"


def fmt_html(value: float) -> str:
    """Format a score for HTML (NaN -> en dash, no padding)."""
    return "&ndash;" if math.isnan(value) else f"{value:.3f}"


def rel_src(filename: str, image_dir: Path, html_dir: Path) -> str:
    """A browser-correct relative ``src`` from the HTML file to an image.

    Relative to the HTML file's own directory (so the report stays portable if the
    repo moves), with forward slashes regardless of OS.
    """
    rel = os.path.relpath(image_dir / filename, html_dir)
    return rel.replace(os.sep, "/")


def figure(filename: str, image_dir: Path, html_dir: Path) -> str:
    """One ``<figure>``: the image, with its filename as both ``alt`` and caption."""
    src = html.escape(rel_src(filename, image_dir, html_dir))
    name = html.escape(filename)
    return (
        f'<figure><img src="{src}" alt="{name}">'
        f"<figcaption>{name}</figcaption></figure>"
    )


def scoped_table(
    headers: Sequence[str],
    rows: Sequence[tuple[str, Sequence[str]]],
    *,
    corner: str = "",
) -> str:
    """Build a semantic table with scoped headers (the shared accessible shape).

    The column-header row is an optional top-left ``corner`` cell followed by
    ``headers`` as ``<th scope="col">`` cells; each ``rows`` entry is a
    ``(row_header, cells)`` pair rendered as a ``<th scope="row">`` then ``<td>``
    cells. An empty ``corner`` emits a blank ``<th>`` (over a row-label column);
    a non-empty ``corner`` labels that column with its own ``scope="col"``.

    All content is treated as **trusted, pre-rendered HTML** -- this builder
    escapes nothing. Cells carry formatter output (``&ndash;``, numbers) and entity
    fragments that must not be double-escaped, so callers ``html.escape`` their own
    text (breed names, labels) before passing it in.
    """
    corner_cell = f'<th scope="col">{corner}</th>' if corner else "<th></th>"
    head = corner_cell + "".join(f'<th scope="col">{h}</th>' for h in headers)
    body = "".join(
        f'<tr><th scope="row">{row_header}</th>'
        + "".join(f"<td>{cell}</td>" for cell in cells)
        + "</tr>"
        for row_header, cells in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
