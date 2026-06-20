"""Render the calibration report as a self-contained semantic-HTML document.

The HTML half of the V0+V1 report: it mirrors ``report_text`` section-for-section
(distributions, overlap, group/per-breed breakdowns, the V1 threshold and
per-breed FPR tables, the risk lists) but additionally **embeds the actual cat
photos** the risk lists name -- each candidate beside the gallery photo it best
matched -- plus the full gallery. The document is screen-reader-first: headings,
lists and scoped-header tables for navigation, no JavaScript, and each image's
filename as both its ``alt`` text and a visible caption. All measurement comes
from ``calibration.metrics``; this module only renders.
"""

from __future__ import annotations

import html
import math
import os
from pathlib import Path

from calibration.manifest import INDY_IMAGE_DIR, OXFORD_IMAGE_DIR
from calibration.metrics import (
    LOOKALIKE_BREEDS,
    RISK_ROWS,
    ScoredImage,
    Stats,
    build_breed_sweep,
    build_sweep,
    select_risk_rows,
    summarize,
    sweep_thresholds,
)
from indycat.decision import Aggregation

#: Minimal styling -- just enough to lay the two figures of a risk row side by
#: side and keep tables legible. No JavaScript; semantics carry the report.
_HTML_STYLE = """
body { font-family: sans-serif; max-width: 60rem; margin: 2rem auto; padding: 0 1rem; }
table { border-collapse: collapse; margin: 0.5rem 0; }
th, td { border: 1px solid #ccc; padding: 0.2rem 0.6rem; text-align: right; }
th[scope="row"], caption { text-align: left; }
figure { display: inline-block; margin: 0.3rem; vertical-align: top; }
figure img { max-height: 12rem; max-width: 16rem; }
figcaption { font-size: 0.85rem; word-break: break-all; max-width: 16rem; }
ol.risks > li { margin-bottom: 1rem; }
"""


def _fmt_html(value: float) -> str:
    """Format a score for HTML (NaN -> en dash, no padding)."""
    return "&ndash;" if math.isnan(value) else f"{value:.3f}"


def _rel_src(filename: str, image_dir: Path, html_dir: Path) -> str:
    """A browser-correct relative ``src`` from the HTML file to an image.

    Relative to the HTML file's own directory (so the report stays portable if the
    repo moves), with forward slashes regardless of OS.
    """
    rel = os.path.relpath(image_dir / filename, html_dir)
    return rel.replace(os.sep, "/")


def _figure(filename: str, image_dir: Path, html_dir: Path) -> str:
    """One ``<figure>``: the image, with its filename as both ``alt`` and caption."""
    src = html.escape(_rel_src(filename, image_dir, html_dir))
    name = html.escape(filename)
    return (
        f'<figure><img src="{src}" alt="{name}">'
        f"<figcaption>{name}</figcaption></figure>"
    )


def _stats_table(rows: list[tuple[str, Stats]], columns: tuple[str, ...]) -> str:
    """A stats ``<table>``: a labelled row header then the requested stat columns.

    ``columns`` names the ``Stats`` attributes to show (e.g. ``("mean", "p95",
    "max")``); ``n`` is always shown first as an integer count.
    """
    head = '<th scope="col">n</th>' + "".join(
        f'<th scope="col">{html.escape(c)}</th>' for c in columns
    )
    body = []
    for label, st in rows:
        cells = "".join(f"<td>{_fmt_html(getattr(st, c))}</td>" for c in columns)
        body.append(
            f'<tr><th scope="row">{html.escape(label)}</th><td>{st.n}</td>{cells}</tr>'
        )
    return (
        "<table><thead><tr><th></th>"
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def _html_overlap(positives: list[ScoredImage], negatives: list[ScoredImage]) -> str:
    if not positives or not negatives:
        return "<p>Need both positives and negatives to compare.</p>"
    lowest_pos = min(positives, key=lambda s: s.score)
    highest_neg = max(negatives, key=lambda s: s.score)
    items = [
        f"Lowest positive {_fmt_html(lowest_pos.score)} "
        f"({html.escape(lowest_pos.name)})",
        f"Highest negative {_fmt_html(highest_neg.score)} "
        f"({html.escape(highest_neg.name)}, {html.escape(str(highest_neg.breed))})",
    ]
    if lowest_pos.score > highest_neg.score:
        gap = lowest_pos.score - highest_neg.score
        items.append(f"Clean gap of {_fmt_html(gap)} (distributions separate)")
    else:
        neg_above = sum(1 for s in negatives if s.score >= lowest_pos.score)
        pos_below = sum(1 for s in positives if s.score <= highest_neg.score)
        items.append(
            f"<strong>OVERLAP</strong>: {neg_above} negative(s) score "
            f"&ge; the lowest positive; {pos_below} positive(s) score "
            f"&le; the highest negative"
        )
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def _html_per_breed(negatives: list[ScoredImage]) -> str:
    by_breed: dict[str, list[float]] = {}
    for s in negatives:
        by_breed.setdefault(s.breed or "(unknown)", []).append(s.score)
    rows = [(breed, summarize(scores)) for breed, scores in by_breed.items()]
    rows.sort(key=lambda r: r[1].max, reverse=True)
    body = []
    for breed, st in rows:
        look = "yes" if breed in LOOKALIKE_BREEDS else ""
        body.append(
            f'<tr><th scope="row">{html.escape(breed)}</th><td>{st.n}</td>'
            f"<td>{_fmt_html(st.mean)}</td><td>{_fmt_html(st.p95)}</td>"
            f"<td>{_fmt_html(st.max)}</td><td>{look}</td></tr>"
        )
    return (
        '<table><thead><tr><th></th><th scope="col">n</th>'
        '<th scope="col">mean</th><th scope="col">p95</th>'
        '<th scope="col">max</th><th scope="col">look-alike</th></tr></thead>'
        "<tbody>" + "".join(body) + "</tbody></table>"
    )


def _html_sweep(
    positives: list[ScoredImage],
    negatives: list[ScoredImage],
    thresholds: list[float],
) -> str:
    cols = ("FPR (all)", "FPR (look-alike)", "FPR (easy)", "recall (Indy)")
    head = "".join(f'<th scope="col">{html.escape(c)}</th>' for c in cols)
    body = []
    for r in build_sweep(positives, negatives, thresholds):
        cells = "".join(
            f"<td>{_fmt_html(v)}</td>"
            for v in (r.fpr_overall, r.fpr_lookalike, r.fpr_easy, r.recall)
        )
        body.append(f'<tr><th scope="row">{r.cutoff:.2f}</th>{cells}</tr>')
    return (
        '<table><thead><tr><th scope="col">cutoff</th>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def _html_breed_sweep(negatives: list[ScoredImage], thresholds: list[float]) -> str:
    breeds, fpr_by_breed = build_breed_sweep(negatives, thresholds)
    head = "".join(f'<th scope="col">{t:.2f}</th>' for t in thresholds)
    body = []
    for breed in breeds:
        cells = "".join(f"<td>{_fmt_html(f)}</td>" for f in fpr_by_breed[breed])
        body.append(f'<tr><th scope="row">{html.escape(breed)}</th>{cells}</tr>')
    return (
        '<table><thead><tr><th scope="col">breed</th>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def _html_risk_list(
    rows: list[ScoredImage],
    candidate_dir: Path,
    html_dir: Path,
    *,
    show_breed: bool,
) -> str:
    """An ordered list of risk rows: a text line plus candidate + best-match figures.

    The candidate image comes from ``candidate_dir``; its ``best_match`` is always
    a gallery (Indy) photo, so that figure always uses the Indy image dir.
    """
    items = []
    for s in rows:
        text = f"{_fmt_html(s.score)} &mdash; {html.escape(s.name)}"
        if show_breed:
            text += f" ({html.escape(str(s.breed))})"
        text += f" &rarr; best match {html.escape(s.best_match)}"
        figures = _figure(s.name, candidate_dir, html_dir) + _figure(
            s.best_match, INDY_IMAGE_DIR, html_dir
        )
        items.append(f"<li><p>{text}</p>{figures}</li>")
    return '<ol class="risks">' + "".join(items) + "</ol>"


def render_report_html(
    label: str,
    gallery_names: list[str],
    positives: list[ScoredImage],
    negatives: list[ScoredImage],
    aggregation: Aggregation,
    *,
    html_path: Path,
    sweep_step: float = 0.05,
    indy_image_dir: Path = INDY_IMAGE_DIR,
    oxford_image_dir: Path = OXFORD_IMAGE_DIR,
) -> str:
    """Render the calibration report as a self-contained semantic HTML document.

    Mirrors :func:`calibration.report_text.build_report` section-for-section (V0
    distributions + the V1 threshold sweep), and additionally embeds the actual
    cat photos the risk lists refer to (each candidate beside the gallery photo it
    best matched) and lists the whole gallery. Image ``src`` paths are relative to
    ``html_path`` so the file is portable; the filename is each image's ``alt``
    text *and* its visible caption (screen-reader-first).
    """
    html_dir = html_path.parent
    thresholds = sweep_thresholds(positives, negatives, sweep_step)
    pos_stats = summarize([s.score for s in positives])
    neg_stats = summarize([s.score for s in negatives])
    look = [s for s in negatives if s.breed in LOOKALIKE_BREEDS]
    easy = [s for s in negatives if s.breed not in LOOKALIKE_BREEDS]
    group_rows = [
        (
            f"{name} ({len({s.breed for s in group})} breeds)",
            summarize([s.score for s in group]),
        )
        for name, group in (("look-alike", look), ("easy", easy))
    ]
    breeds = len({s.breed for s in negatives})
    worst_neg, hardest_pos = select_risk_rows(positives, negatives)
    esc_label = html.escape(label)

    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>Calibration report: {esc_label}</title>",
        f"<style>{_HTML_STYLE}</style>",
        "</head>",
        "<body>",
        "<h1>Calibration report</h1>",
        f"<p>Manifest: <code>{esc_label}</code> "
        f"(aggregation = {html.escape(aggregation)})</p>",
        "<h2>Summary</h2>",
        "<ul>"
        f"<li>Gallery: {len(gallery_names)} Indy photos</li>"
        f"<li>Positives: {len(positives)} Indy photos (calibration)</li>"
        f"<li>Negatives: {len(negatives)} Oxford cats, {breeds} breeds</li>"
        "</ul>",
        "<h2>Score distribution</h2>",
        "<p>Cosine to best gallery match.</p>",
        _stats_table(
            [("Indy (positives)", pos_stats), ("Oxford (negatives)", neg_stats)],
            ("mean", "min", "p50", "p95", "max"),
        ),
        "<h2>Overlap</h2>",
        _html_overlap(positives, negatives),
        "<h2>Negatives by group</h2>",
        _stats_table(group_rows, ("mean", "p95", "max")),
        "<h2>Per-breed negative scores</h2>",
        "<p>Sorted by max descending.</p>",
        _html_per_breed(negatives),
        "<h2>Threshold sweep</h2>",
        "<p>A query is Indy when its score is at or above the cutoff. No cutoff "
        "is chosen here -- this only shows the trade-off.</p>",
        _html_sweep(positives, negatives, thresholds),
        "<h2>Per-breed FPR by cutoff</h2>",
        "<p>Breeds sorted worst-first (highest max negative score).</p>",
        _html_breed_sweep(negatives, thresholds),
        f"<h2>Highest-scoring negatives (false-positive risks, top {RISK_ROWS})</h2>",
        _html_risk_list(worst_neg, oxford_image_dir, html_dir, show_breed=True),
        f"<h2>Lowest-scoring positives (recognition risks, bottom {RISK_ROWS})</h2>",
        _html_risk_list(hardest_pos, indy_image_dir, html_dir, show_breed=False),
        "<h2>Gallery</h2>",
        f"<p>{len(gallery_names)} Indy gallery photos.</p>",
        "".join(_figure(name, indy_image_dir, html_dir) for name in gallery_names),
        "</body>",
        "</html>",
    ]
    return "\n".join(parts)


def write_report_html(
    path: Path,
    label: str,
    gallery_names: list[str],
    positives: list[ScoredImage],
    negatives: list[ScoredImage],
    aggregation: Aggregation,
    *,
    sweep_step: float = 0.05,
    indy_image_dir: Path = INDY_IMAGE_DIR,
    oxford_image_dir: Path = OXFORD_IMAGE_DIR,
) -> None:
    """Render the HTML report and write it to ``path`` (parents created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    document = render_report_html(
        label,
        gallery_names,
        positives,
        negatives,
        aggregation,
        html_path=path,
        sweep_step=sweep_step,
        indy_image_dir=indy_image_dir,
        oxford_image_dir=oxford_image_dir,
    )
    path.write_text(document, encoding="utf-8")
