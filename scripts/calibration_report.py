"""Measure and report the calibration score distributions and trade-off (V0+V1).

The V0+V1 slices of ``docs/calibration_design.md`` Sec. 5. This module scores the
held-back Indy positives and the Oxford negatives against the gallery (via
``indycat.decision``) and renders the screen-reader-first textual report from
Sec. 4: the V0 distribution tables, the positive/negative overlap, a
look-alike-vs-easy breakdown, per-breed negative scores, and the worst
false-positive / hardest-positive rows -- plus the **V1 threshold sweep**, a
trade-off curve of ``cutoff -> FPR (overall / look-alike / easy) , recall-on-Indy``
with a separate per-breed FPR table.

The sweep deliberately picks **no cutoff** -- it only makes the trade-off visible
(choosing a threshold by policy is V2). Kept separate from ``calibrate.py`` so the
CLI stays a thin driver and the measurement/report is unit-testable. ``test`` is
never touched here -- only ``gallery`` (the references), ``calibration``
(positives), and the Oxford ``setup`` (negatives) roles take part.
"""

from __future__ import annotations

import csv
import html
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from indycat.decision import Aggregation, Gallery, score
from split_manifest import INDY_IMAGE_DIR, OXFORD_IMAGE_DIR

#: The long-haired Oxford breeds that look most like Indy -- the hard negative
#: tail the threshold will have to sit above. Reported as one group *and* per
#: breed, so Persian (et al.) stay individually visible regardless of grouping.
LOOKALIKE_BREEDS = frozenset({"Maine_Coon", "Ragdoll", "Birman", "Persian"})

#: How many worst-case rows the report lists for each risk section.
_RISK_ROWS = 10


@dataclass(frozen=True)
class ScoredImage:
    """One image scored against the gallery; ``breed`` is None for Indy positives."""

    name: str
    score: float
    best_match: str
    breed: str | None


@dataclass(frozen=True)
class Stats:
    """Summary statistics of a score distribution (``n`` is the count)."""

    n: int
    mean: float
    min: float
    p50: float
    p95: float
    max: float


@dataclass(frozen=True)
class SweepRow:
    """One row of the threshold sweep: the trade-off at a single cutoff.

    A query is called *Indy* when its score ``>= cutoff`` (the ``>=`` convention
    matches the overlap counts). ``fpr_*`` are false-positive rates over the
    respective negative groups; ``recall`` is the fraction of positives kept.
    Any rate is NaN when its group is empty.
    """

    cutoff: float
    fpr_overall: float
    fpr_lookalike: float
    fpr_easy: float
    recall: float


def build_name_to_vector(
    names: list[str], vectors: NDArray[np.float32]
) -> dict[str, NDArray[np.float32]]:
    """Map each ``source_filename`` to its embedding row."""
    return {name: vectors[i] for i, name in enumerate(names)}


def select_vectors(
    names: list[str], name_to_vector: dict[str, NDArray[np.float32]]
) -> NDArray[np.float32]:
    """Stack the vectors for ``names``; a name absent from the cache is loud.

    A manifest references images by ``source_filename``; one missing from the
    embeddings cache means the manifest and the cache disagree (a re-embed or an
    Oxford no-cat miss), which must surface rather than silently shrink the role.
    """
    missing = [name for name in names if name not in name_to_vector]
    if missing:
        raise KeyError(
            f"{len(missing)} manifest image(s) absent from the embeddings cache: "
            f"{missing[:5]}{' ...' if len(missing) > 5 else ''}"
        )
    return np.stack([name_to_vector[name] for name in names])


def score_role(
    role_names: list[str],
    name_to_vector: dict[str, NDArray[np.float32]],
    gallery: Gallery,
    aggregation: Aggregation,
    breeds: dict[str, str] | None = None,
) -> list[ScoredImage]:
    """Score every image in a role against the gallery."""
    missing = [name for name in role_names if name not in name_to_vector]
    if missing:
        raise KeyError(
            f"{len(missing)} manifest image(s) absent from the embeddings cache: "
            f"{missing[:5]}{' ...' if len(missing) > 5 else ''}"
        )
    scored: list[ScoredImage] = []
    for name in role_names:
        match = score(name_to_vector[name], gallery, aggregation)
        breed = breeds.get(name) if breeds is not None else None
        scored.append(ScoredImage(name, match.score, match.best_name, breed))
    return scored


def summarize(scores: list[float]) -> Stats:
    """Distribution summary; an empty list yields zero count and NaN stats."""
    if not scores:
        nan = float("nan")
        return Stats(0, nan, nan, nan, nan, nan)
    arr = np.asarray(scores, dtype=np.float64)
    return Stats(
        n=len(scores),
        mean=float(arr.mean()),
        min=float(arr.min()),
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        max=float(arr.max()),
    )


# --------------------------------------------------------------------------- #
# Threshold sweep (V1: the trade-off curve, no cutoff is chosen)
# --------------------------------------------------------------------------- #


def sweep_thresholds(
    positives: list[ScoredImage], negatives: list[ScoredImage], step: float
) -> list[float]:
    """A round, data-ranged grid of cutoffs spanning the observed scores.

    Cutoffs are multiples of ``step`` from just below the minimum to just above
    the maximum of *all* scores (positives and negatives together), so the grid
    brackets the full trade-off: at the low end almost everything clears the bar
    (FPR ~= 1, recall ~= 1); at the high end almost nothing does. Returns an
    empty list when there is nothing to score.
    """
    if step <= 0:
        raise ValueError(f"sweep step must be positive, got {step}")
    scores = [s.score for s in positives] + [s.score for s in negatives]
    if not scores:
        return []
    lo = math.floor(min(scores) / step)
    hi = math.ceil(max(scores) / step)
    return [round(k * step, 10) for k in range(lo, hi + 1)]


def _rate(scores: list[float], cutoff: float) -> float:
    """Fraction of ``scores`` at or above ``cutoff``; NaN for an empty group."""
    if not scores:
        return float("nan")
    return sum(1 for s in scores if s >= cutoff) / len(scores)


def build_sweep(
    positives: list[ScoredImage],
    negatives: list[ScoredImage],
    thresholds: list[float],
) -> list[SweepRow]:
    """The main trade-off: FPR (overall / look-alike / easy) and recall per cutoff."""
    pos = [s.score for s in positives]
    neg_all = [s.score for s in negatives]
    neg_look = [s.score for s in negatives if s.breed in LOOKALIKE_BREEDS]
    neg_easy = [s.score for s in negatives if s.breed not in LOOKALIKE_BREEDS]
    return [
        SweepRow(
            cutoff=t,
            fpr_overall=_rate(neg_all, t),
            fpr_lookalike=_rate(neg_look, t),
            fpr_easy=_rate(neg_easy, t),
            recall=_rate(pos, t),
        )
        for t in thresholds
    ]


def build_breed_sweep(
    negatives: list[ScoredImage], thresholds: list[float]
) -> tuple[list[str], dict[str, list[float]]]:
    """Per-breed FPR at each cutoff; breeds sorted worst-first (by max score desc).

    Returns ``(breeds, fpr_by_breed)`` where ``fpr_by_breed[breed]`` is the list
    of FPRs aligned to ``thresholds`` -- the same row order the text/HTML tables use.
    """
    by_breed: dict[str, list[float]] = {}
    for s in negatives:
        by_breed.setdefault(s.breed or "(unknown)", []).append(s.score)
    breeds = sorted(by_breed, key=lambda b: max(by_breed[b]), reverse=True)
    fpr_by_breed = {b: [_rate(by_breed[b], t) for t in thresholds] for b in breeds}
    return breeds, fpr_by_breed


# --------------------------------------------------------------------------- #
# Report rendering
# --------------------------------------------------------------------------- #


def _fmt(value: float) -> str:
    """Format a score for the report (NaN -> dash)."""
    return "  -  " if math.isnan(value) else f"{value:.3f}"


def _stats_row(label: str, stats: Stats, width: int) -> str:
    return (
        f"  {label:<{width}} {stats.n:>5}  {_fmt(stats.mean)}  {_fmt(stats.min)}  "
        f"{_fmt(stats.p50)}  {_fmt(stats.p95)}  {_fmt(stats.max)}"
    )


def _distribution_section(
    positives: list[ScoredImage], negatives: list[ScoredImage], aggregation: Aggregation
) -> list[str]:
    pos = summarize([s.score for s in positives])
    neg = summarize([s.score for s in negatives])
    width = len("Oxford (neg)")
    columns = f"{'n':>5}  {'mean':>5}  {'min':>5}  {'p50':>5}  {'p95':>5}  {'max':>5}"
    header = " " * (2 + width + 1) + columns
    return [
        f"Score distribution (cosine to best gallery match, "
        f"aggregation={aggregation}):",
        header,
        _stats_row("Indy (pos)", pos, width),
        _stats_row("Oxford (neg)", neg, width),
    ]


def _overlap_section(
    positives: list[ScoredImage], negatives: list[ScoredImage]
) -> list[str]:
    if not positives or not negatives:
        return ["", "  Overlap: need both positives and negatives to compare."]
    lowest_pos = min(positives, key=lambda s: s.score)
    highest_neg = max(negatives, key=lambda s: s.score)
    lines = [
        "",
        f"  Lowest positive  {_fmt(lowest_pos.score)} ({lowest_pos.name})",
        f"  Highest negative {_fmt(highest_neg.score)} "
        f"({highest_neg.name}, {highest_neg.breed})",
    ]
    if lowest_pos.score > highest_neg.score:
        gap = lowest_pos.score - highest_neg.score
        lines.append(f"  -> clean gap of {_fmt(gap)} (distributions separate)")
    else:
        neg_above = sum(1 for s in negatives if s.score >= lowest_pos.score)
        pos_below = sum(1 for s in positives if s.score <= highest_neg.score)
        lines.append(
            f"  -> OVERLAP: {neg_above} negative(s) score >= the lowest positive; "
            f"{pos_below} positive(s) score <= the highest negative"
        )
    return lines


def _group_section(negatives: list[ScoredImage]) -> list[str]:
    look = [s for s in negatives if s.breed in LOOKALIKE_BREEDS]
    easy = [s for s in negatives if s.breed not in LOOKALIKE_BREEDS]
    lines = ["", "Negatives by group:"]
    for label, group in (("look-alike", look), ("easy", easy)):
        stats = summarize([s.score for s in group])
        breeds = len({s.breed for s in group})
        lines.append(
            f"  {label:<11} ({breeds} breeds, n={stats.n}):  "
            f"mean {_fmt(stats.mean)}  p95 {_fmt(stats.p95)}  max {_fmt(stats.max)}"
        )
    return lines


def _per_breed_section(negatives: list[ScoredImage]) -> list[str]:
    by_breed: dict[str, list[float]] = {}
    for s in negatives:
        by_breed.setdefault(s.breed or "(unknown)", []).append(s.score)
    rows = [(breed, summarize(scores)) for breed, scores in by_breed.items()]
    rows.sort(key=lambda r: r[1].max, reverse=True)
    width = max((len(b) for b, _ in rows), default=5)
    lines = [
        "",
        "Per-breed negative scores (sorted by max desc):",
        f"  {'breed':<{width}} {'n':>5}  {'mean':>5}  {'p95':>5}  {'max':>5}",
    ]
    for breed, stats in rows:
        tag = " *" if breed in LOOKALIKE_BREEDS else ""
        lines.append(
            f"  {breed:<{width}} {stats.n:>5}  {_fmt(stats.mean)}  "
            f"{_fmt(stats.p95)}  {_fmt(stats.max)}{tag}"
        )
    lines.append("  (* = look-alike group)")
    return lines


def _sweep_section(
    positives: list[ScoredImage],
    negatives: list[ScoredImage],
    thresholds: list[float],
) -> list[str]:
    rows = build_sweep(positives, negatives, thresholds)
    lines = [
        "",
        "Threshold sweep (a query is Indy when score >= cutoff; no cutoff is "
        "chosen here):",
        f"  {'cutoff':>6}  {'FPR(all)':>8}  {'FPR(look)':>9}  {'FPR(easy)':>9}  "
        f"{'recall':>6}",
    ]
    for r in rows:
        lines.append(
            f"  {r.cutoff:>6.2f}  {_fmt(r.fpr_overall):>8}  "
            f"{_fmt(r.fpr_lookalike):>9}  {_fmt(r.fpr_easy):>9}  {_fmt(r.recall):>6}"
        )
    return lines


def _per_breed_sweep_section(
    negatives: list[ScoredImage], thresholds: list[float]
) -> list[str]:
    breeds, fpr_by_breed = build_breed_sweep(negatives, thresholds)
    width = max((len(b) for b in breeds), default=5)
    header_cells = "  ".join(f"{t:>5.2f}" for t in thresholds)
    lines = [
        "",
        "Per-breed FPR by cutoff (breeds sorted worst-first):",
        f"  {'breed':<{width}}  {header_cells}",
    ]
    for breed in breeds:
        cells = "  ".join(f"{_fmt(f):>5}" for f in fpr_by_breed[breed])
        lines.append(f"  {breed:<{width}}  {cells}")
    return lines


def select_risk_rows(
    positives: list[ScoredImage], negatives: list[ScoredImage]
) -> tuple[list[ScoredImage], list[ScoredImage]]:
    """The two risk lists: ``(worst_negatives, hardest_positives)``, each top-N.

    Shared by the text and HTML renderers so they cannot diverge: negatives by
    score descending (highest false-positive risk first), positives by score
    ascending (hardest to recognise first), both capped at ``_RISK_ROWS``.
    """
    worst_neg = sorted(negatives, key=lambda s: s.score, reverse=True)[:_RISK_ROWS]
    hardest_pos = sorted(positives, key=lambda s: s.score)[:_RISK_ROWS]
    return worst_neg, hardest_pos


def _risk_sections(
    positives: list[ScoredImage], negatives: list[ScoredImage]
) -> list[str]:
    worst_neg, hardest_pos = select_risk_rows(positives, negatives)
    lines = ["", f"Highest-scoring negatives (false-positive risks, top {_RISK_ROWS}):"]
    for s in worst_neg:
        lines.append(
            f"  {_fmt(s.score)}  {s.name}  ({s.breed})  -> best match {s.best_match}"
        )
    lines += ["", f"Lowest-scoring positives (recognition risks, bottom {_RISK_ROWS}):"]
    for s in hardest_pos:
        lines.append(f"  {_fmt(s.score)}  {s.name}  -> best match {s.best_match}")
    return lines


def build_report(
    label: str,
    gallery_size: int,
    positives: list[ScoredImage],
    negatives: list[ScoredImage],
    aggregation: Aggregation,
    sweep_step: float = 0.05,
) -> str:
    """Render the full textual calibration report (V0 distributions + V1 sweep)."""
    breeds = len({s.breed for s in negatives})
    header = [
        f"Calibration: {label}   (aggregation={aggregation})",
        f"  Gallery:    {gallery_size} Indy photos",
        f"  Positives:  {len(positives)} Indy photos (calibration)",
        f"  Negatives:  {len(negatives)} Oxford cats, {breeds} breeds",
        "",
    ]
    thresholds = sweep_thresholds(positives, negatives, sweep_step)
    sections = [
        *_distribution_section(positives, negatives, aggregation),
        *_overlap_section(positives, negatives),
        *_group_section(negatives),
        *_per_breed_section(negatives),
        *_sweep_section(positives, negatives, thresholds),
        *_per_breed_sweep_section(negatives, thresholds),
        *_risk_sections(positives, negatives),
    ]
    return "\n".join([*header, *sections])


# --------------------------------------------------------------------------- #
# HTML report rendering
# --------------------------------------------------------------------------- #

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

    Mirrors :func:`build_report` section-for-section (V0 distributions + the V1
    threshold sweep), and additionally embeds the actual cat photos the risk lists
    refer to (each candidate beside the gallery photo it best matched) and lists
    the whole gallery. Image ``src`` paths are relative to ``html_path`` so the
    file is portable; the filename is each image's ``alt`` text *and* its visible
    caption (screen-reader-first).
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
        f"<h2>Highest-scoring negatives (false-positive risks, top {_RISK_ROWS})</h2>",
        _html_risk_list(worst_neg, oxford_image_dir, html_dir, show_breed=True),
        f"<h2>Lowest-scoring positives (recognition risks, bottom {_RISK_ROWS})</h2>",
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


def write_scores_csv(
    path: Path, positives: list[ScoredImage], negatives: list[ScoredImage]
) -> None:
    """Write per-image scores joined with provenance for offline inspection.

    Negatives first (worst false-positive risk on top), then positives (hardest
    to recognise on top), each ordered so the rows that matter most lead.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["role", "source_filename", "score", "best_match", "breed"])
        for s in sorted(negatives, key=lambda s: s.score, reverse=True):
            writer.writerow(
                ["negative", s.name, f"{s.score:.6f}", s.best_match, s.breed]
            )
        for s in sorted(positives, key=lambda s: s.score):
            writer.writerow(["positive", s.name, f"{s.score:.6f}", s.best_match, ""])
