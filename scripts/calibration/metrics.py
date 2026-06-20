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
from typing import Literal

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


@dataclass(frozen=True)
class Confusion:
    """The confusion matrix at one cutoff (the ``>=`` convention, like the sweep).

    ``tp``/``fn`` partition the positives (Indy), ``fp``/``tn`` the negatives
    (Oxford): a query is called *Indy* when its score ``>= threshold``. This is
    evaluate's fixed-rule view -- the actual counts at the frozen threshold -- and
    is consistent with :func:`build_sweep`'s rates (recall = tp / (tp + fn),
    FPR(all) = fp / (fp + tn)).
    """

    tp: int
    fn: int
    fp: int
    tn: int


def confusion_at(
    positives: list[ScoredImage], negatives: list[ScoredImage], threshold: float
) -> Confusion:
    """Confusion counts at ``threshold`` (``score >= threshold`` -> Indy)."""
    tp = sum(1 for s in positives if s.score >= threshold)
    fp = sum(1 for s in negatives if s.score >= threshold)
    return Confusion(tp=tp, fn=len(positives) - tp, fp=fp, tn=len(negatives) - fp)


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


# --------------------------------------------------------------------------- #
# Threshold pick (V2: an explicit policy chooses one cutoff off the curve)
# --------------------------------------------------------------------------- #

#: Which automated policy chooses the cutoff (``docs/calibration_design.md`` V2).
PickPolicy = Literal["target-fpr", "youdens-j", "equal-error"]
PICK_POLICIES: tuple[PickPolicy, ...] = ("target-fpr", "youdens-j", "equal-error")

#: Which negative group the ``target-fpr`` budget applies to.
TargetGroup = Literal["overall", "look-alike"]
TARGET_GROUPS: tuple[TargetGroup, ...] = ("overall", "look-alike")

#: Pad placing the below-min / above-max endpoints just outside the data range.
#: Only the *reported* endpoint cutoff shifts by this; it never changes which
#: scores clear the bar (the endpoint stays strictly outside the observed range).
_ENDPOINT_PAD = 1e-3


@dataclass(frozen=True)
class ThresholdChoice:
    """A cutoff chosen by an explicit policy, with its trade-off at that cutoff.

    ``row`` is the :class:`SweepRow` evaluated at the chosen cutoff (so it carries
    the cutoff plus FPR(all/look-alike/easy) and recall); ``rationale`` is the
    human-readable reason the policy landed there.
    """

    policy: str
    row: SweepRow
    rationale: str


def candidate_cutoffs(
    positives: list[ScoredImage], negatives: list[ScoredImage]
) -> list[float]:
    """A fine candidate grid for policy picking: midpoints + bracketing endpoints.

    Unlike the round V1 sweep grid (multiples of ``--sweep-step``), this is derived
    from the *actual* scores so the chosen threshold is precise. Candidates are the
    midpoints between adjacent distinct observed scores -- so no candidate equals an
    observed score and the ``>=`` convention is unambiguous -- plus an endpoint just
    below the minimum (everything clears: FPR=1, recall=1) and just above the
    maximum (nothing clears: FPR=0, recall=0). Returns an empty list with no scores.
    """
    scores = sorted({s.score for s in positives} | {s.score for s in negatives})
    if not scores:
        return []
    midpoints = [(lo + hi) / 2 for lo, hi in zip(scores, scores[1:], strict=False)]
    return [scores[0] - _ENDPOINT_PAD, *midpoints, scores[-1] + _ENDPOINT_PAD]


def pick_threshold(
    positives: list[ScoredImage],
    negatives: list[ScoredImage],
    policy: PickPolicy,
    *,
    target_fpr: float = 0.05,
    target_group: TargetGroup = "look-alike",
) -> ThresholdChoice:
    """Choose one cutoff off the trade-off curve by an explicit policy (V2).

    All policies score the fine :func:`candidate_cutoffs` grid via
    :func:`build_sweep` (same ``>=`` convention and FPR/recall math as V1), then
    pick one row:

    - ``target-fpr`` -- the *lowest* cutoff (max recall) whose FPR over
      ``target_group`` is ``<= target_fpr``. The above-max endpoint (FPR=0) makes
      a feasible pick guaranteed.
    - ``youdens-j`` -- the cutoff maximising ``recall - fpr_overall``.
    - ``equal-error`` -- the cutoff minimising ``|fpr_overall - (1 - recall)|``.

    Ties break toward the *higher* cutoff (fewer false positives). Raises
    ``ValueError`` for empty positives/negatives, or a ``look-alike`` target with
    no look-alike negatives -- never returns a silently-wrong (NaN-driven) pick.
    """
    if not positives:
        raise ValueError("pick_threshold needs at least one positive")
    if not negatives:
        raise ValueError("pick_threshold needs at least one negative")
    rows = build_sweep(positives, negatives, candidate_cutoffs(positives, negatives))

    if policy == "target-fpr":
        use_lookalike = target_group == "look-alike"
        fprs = [r.fpr_lookalike if use_lookalike else r.fpr_overall for r in rows]
        if use_lookalike and all(math.isnan(f) for f in fprs):
            raise ValueError(
                "target-fpr with group=look-alike needs look-alike negatives"
            )
        feasible = [r for r, f in zip(rows, fprs, strict=True) if f <= target_fpr]
        # Lowest cutoff within budget = most recall; feasible is non-empty because
        # the above-max endpoint has FPR=0 <= target_fpr.
        chosen = min(feasible, key=lambda r: r.cutoff)
        rationale = (
            f"lowest cutoff with FPR({target_group}) <= {target_fpr:.3f} "
            f"(maximises recall within the budget)"
        )
    elif policy == "youdens-j":
        chosen = max(rows, key=lambda r: (r.recall - r.fpr_overall, r.cutoff))
        rationale = "cutoff maximising Youden's J = recall - FPR(all)"
    elif policy == "equal-error":
        chosen = min(
            rows, key=lambda r: (abs(r.fpr_overall - (1 - r.recall)), -r.cutoff)
        )
        rationale = "cutoff where FPR(all) ~= 1 - recall (equal-error rate)"
    else:  # pragma: no cover - exhaustive over PickPolicy
        raise ValueError(f"unknown policy: {policy}")

    return ThresholdChoice(policy=policy, row=chosen, rationale=rationale)
