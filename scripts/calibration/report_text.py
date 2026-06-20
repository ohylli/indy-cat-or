"""Render the calibration report as screen-reader-first plain text, plus the CSV.

The textual half of the V0+V1 report (``docs/calibration_design.md`` Sec. 4):
the distribution tables, the positive/negative overlap, the look-alike-vs-easy
breakdown, per-breed negative scores, the V1 threshold sweep and per-breed FPR
tables, and the worst-case risk lists. All measurement comes from
``calibration.metrics``; this module only formats it. ``write_scores_csv`` writes
the per-image scores for offline inspection.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

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


def _risk_sections(
    positives: list[ScoredImage], negatives: list[ScoredImage]
) -> list[str]:
    worst_neg, hardest_pos = select_risk_rows(positives, negatives)
    lines = ["", f"Highest-scoring negatives (false-positive risks, top {RISK_ROWS}):"]
    for s in worst_neg:
        lines.append(
            f"  {_fmt(s.score)}  {s.name}  ({s.breed})  -> best match {s.best_match}"
        )
    lines += ["", f"Lowest-scoring positives (recognition risks, bottom {RISK_ROWS}):"]
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
