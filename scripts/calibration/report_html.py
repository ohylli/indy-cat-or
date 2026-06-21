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
from pathlib import Path

from calibration.manifest import INDY_IMAGE_DIR, OXFORD_IMAGE_DIR
from calibration.metrics import (
    LOOKALIKE_BREEDS,
    RISK_ROWS,
    ScoredImage,
    Stats,
    SweepRow,
    ThresholdChoice,
    build_breed_sweep,
    build_sweep,
    select_risk_rows,
    summarize,
    sweep_thresholds,
)
from calibration.report_common import (
    HTML_STYLE,
    figure,
    figure_list,
    fmt_html,
    scoped_table,
)
from indycat.decision import Aggregation


def _stats_table(rows: list[tuple[str, Stats]], columns: tuple[str, ...]) -> str:
    """A stats ``<table>``: a labelled row header then the requested stat columns.

    ``columns`` names the ``Stats`` attributes to show (e.g. ``("mean", "p95",
    "max")``); ``n`` is always shown first as an integer count.
    """
    headers = ["n", *(html.escape(c) for c in columns)]
    body = [
        (
            html.escape(label),
            [str(st.n), *(fmt_html(getattr(st, c)) for c in columns)],
        )
        for label, st in rows
    ]
    return scoped_table(headers, body)


def _html_overlap(positives: list[ScoredImage], negatives: list[ScoredImage]) -> str:
    if not positives or not negatives:
        return "<p>Need both positives and negatives to compare.</p>"
    lowest_pos = min(positives, key=lambda s: s.score)
    highest_neg = max(negatives, key=lambda s: s.score)
    items = [
        f"Lowest positive {fmt_html(lowest_pos.score)} "
        f"({html.escape(lowest_pos.name)})",
        f"Highest negative {fmt_html(highest_neg.score)} "
        f"({html.escape(highest_neg.name)}, {html.escape(str(highest_neg.breed))})",
    ]
    if lowest_pos.score > highest_neg.score:
        gap = lowest_pos.score - highest_neg.score
        items.append(f"Clean gap of {fmt_html(gap)} (distributions separate)")
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
    body = [
        (
            html.escape(breed),
            [
                str(st.n),
                fmt_html(st.mean),
                fmt_html(st.p95),
                fmt_html(st.max),
                "yes" if breed in LOOKALIKE_BREEDS else "",
            ],
        )
        for breed, st in rows
    ]
    return scoped_table(["n", "mean", "p95", "max", "look-alike"], body)


#: Column headers for the FPR/recall sweep tables (also the V2 choice table).
_SWEEP_COLUMNS = ("FPR (all)", "FPR (look-alike)", "FPR (easy)", "recall (Indy)")


def _sweep_cells(r: SweepRow) -> list[str]:
    """The FPR/recall cells of a sweep row, in ``_SWEEP_COLUMNS`` order."""
    return [fmt_html(v) for v in (r.fpr_overall, r.fpr_lookalike, r.fpr_easy, r.recall)]


def _html_sweep(
    positives: list[ScoredImage],
    negatives: list[ScoredImage],
    thresholds: list[float],
) -> str:
    headers = [html.escape(c) for c in _SWEEP_COLUMNS]
    body = [
        (f"{r.cutoff:.2f}", _sweep_cells(r))
        for r in build_sweep(positives, negatives, thresholds)
    ]
    return scoped_table(headers, body, corner="cutoff")


def _html_breed_sweep(negatives: list[ScoredImage], thresholds: list[float]) -> str:
    breeds, fpr_by_breed = build_breed_sweep(negatives, thresholds)
    headers = [f"{t:.2f}" for t in thresholds]
    body = [
        (html.escape(breed), [fmt_html(f) for f in fpr_by_breed[breed]])
        for breed in breeds
    ]
    return scoped_table(headers, body, corner="breed")


def _html_choice(choice: ThresholdChoice) -> str:
    """The V2 chosen-threshold section: rationale plus a one-row metrics table."""
    r = choice.row
    headers = [html.escape(c) for c in _SWEEP_COLUMNS]
    rows = [(f"{r.cutoff:.3f}", _sweep_cells(r))]
    table = scoped_table(headers, rows, corner="cutoff")
    return (
        f"<p>Policy <code>{html.escape(choice.policy)}</code>: "
        f"{html.escape(choice.rationale)}.</p>" + table
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
    a gallery (Indy) photo, so that figure always uses the Indy image dir. Thin
    wrapper over the shared :func:`figure_list` so calibrate's risk lists and
    evaluate's error lists render identical markup.
    """
    return figure_list(
        rows, candidate_dir, INDY_IMAGE_DIR, html_dir, show_breed=show_breed
    )


def render_report_html(
    label: str,
    gallery_names: list[str],
    positives: list[ScoredImage],
    negatives: list[ScoredImage],
    aggregation: Aggregation,
    *,
    html_path: Path,
    sweep_step: float = 0.05,
    choice: ThresholdChoice | None = None,
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
        f"<style>{HTML_STYLE}</style>",
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
        *(
            ["<h2>Chosen threshold</h2>", _html_choice(choice)]
            if choice is not None
            else []
        ),
        f"<h2>Highest-scoring negatives (false-positive risks, top {RISK_ROWS})</h2>",
        _html_risk_list(worst_neg, oxford_image_dir, html_dir, show_breed=True),
        f"<h2>Lowest-scoring positives (recognition risks, bottom {RISK_ROWS})</h2>",
        _html_risk_list(hardest_pos, indy_image_dir, html_dir, show_breed=False),
        "<h2>Gallery</h2>",
        f"<p>{len(gallery_names)} Indy gallery photos.</p>",
        "".join(figure(name, indy_image_dir, html_dir) for name in gallery_names),
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
    choice: ThresholdChoice | None = None,
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
        choice=choice,
        indy_image_dir=indy_image_dir,
        oxford_image_dir=oxford_image_dir,
    )
    path.write_text(document, encoding="utf-8")
