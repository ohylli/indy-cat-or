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
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

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


def figure(
    filename: str, image_dir: Path, html_dir: Path, *, rel_path: str | None = None
) -> str:
    """One ``<figure>``: the image, with its filename as both ``alt`` and caption.

    The image is located at ``image_dir / (rel_path or filename)``; ``rel_path``
    lets a caller whose ``source_filename`` is a bare name point at a nested file
    (e.g. cat-breeds' ``<breed>/<file>``) while keeping the bare ``filename`` as
    the screen-reader alt text and caption.
    """
    src = html.escape(rel_src(rel_path or filename, image_dir, html_dir))
    name = html.escape(filename)
    return (
        f'<figure><img src="{src}" alt="{name}">'
        f"<figcaption>{name}</figcaption></figure>"
    )


class ScoredRow(Protocol):
    """The shape :func:`figure_list` needs of a scored image (duck-typed).

    Kept as a local ``Protocol`` so this primitives module stays decoupled from
    :mod:`calibration.metrics` (whose frozen ``ScoredImage`` satisfies it). Members
    are read-only properties so a frozen dataclass's read-only fields match.
    """

    @property
    def score(self) -> float: ...
    @property
    def name(self) -> str: ...
    @property
    def best_match(self) -> str: ...
    @property
    def breed(self) -> str | None: ...


def figure_list(
    rows: Sequence[ScoredRow],
    candidate_dir: Path,
    best_match_dir: Path,
    html_dir: Path,
    *,
    show_breed: bool,
    candidate_resolver: Callable[[str], str] | None = None,
) -> str:
    """An ordered list of scored rows: a text line plus candidate + best-match figures.

    Shared by calibrate's risk lists and evaluate's error lists so the figure-row
    markup (and the screen-reader caption/alt text inside :func:`figure`) lives in
    one place. The candidate image comes from ``candidate_dir``; its ``best_match``
    from ``best_match_dir`` (always the Indy gallery dir). ``show_breed`` appends
    the breed to the text line (negatives have one, positives do not).

    ``candidate_resolver`` maps a candidate's ``name`` to a path relative to
    ``candidate_dir`` -- for a source whose ``source_filename`` is a bare name but
    whose image is nested (cat-breeds' ``<breed>/<file>``). The caption/alt stays
    the bare ``name``. ``None`` (the default) locates the candidate flat by name,
    so existing callers are unchanged.
    """
    items = []
    for s in rows:
        text = f"{fmt_html(s.score)} &mdash; {html.escape(s.name)}"
        if show_breed:
            text += f" ({html.escape(str(s.breed))})"
        text += f" &rarr; best match {html.escape(s.best_match)}"
        rel_path = candidate_resolver(s.name) if candidate_resolver else None
        figures = figure(s.name, candidate_dir, html_dir, rel_path=rel_path) + figure(
            s.best_match, best_match_dir, html_dir
        )
        items.append(f"<li><p>{text}</p>{figures}</li>")
    return '<ol class="risks">' + "".join(items) + "</ol>"


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
