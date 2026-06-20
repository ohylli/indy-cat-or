"""Calibration measurement: score distributions, the threshold sweep, risk lists.

The pure-computation half of the V0+V1 report (``docs/calibration_design.md``
Sec. 5): the data structures, distribution summaries, the trade-off sweep, and
the worst-case risk selection. No rendering and no I/O live here -- the text and
HTML renderers (``report_text``/``report_html``) and the future ``evaluate.py``
all consume these, so the metrics stay in one place and cannot diverge between
outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: The long-haired Oxford breeds that look most like Indy -- the hard negative
#: tail the threshold will have to sit above. Reported as one group *and* per
#: breed, so Persian (et al.) stay individually visible regardless of grouping.
LOOKALIKE_BREEDS = frozenset({"Maine_Coon", "Ragdoll", "Birman", "Persian"})

#: How many worst-case rows the report lists for each risk section.
RISK_ROWS = 10


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


def select_risk_rows(
    positives: list[ScoredImage], negatives: list[ScoredImage]
) -> tuple[list[ScoredImage], list[ScoredImage]]:
    """The two risk lists: ``(worst_negatives, hardest_positives)``, each top-N.

    Shared by the text and HTML renderers so they cannot diverge: negatives by
    score descending (highest false-positive risk first), positives by score
    ascending (hardest to recognise first), both capped at ``RISK_ROWS``.
    """
    worst_neg = sorted(negatives, key=lambda s: s.score, reverse=True)[:RISK_ROWS]
    hardest_pos = sorted(positives, key=lambda s: s.score)[:RISK_ROWS]
    return worst_neg, hardest_pos
