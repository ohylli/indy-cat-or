"""Render the evaluation report as screen-reader-first plain text.

The textual half of ``evaluate.py``'s E0 output (``docs/calibration_design.md``
Sec. 7): the confusion matrix at the *frozen* threshold, the headline rates, the
per-breed FPR at that one cutoff, and the calibration-vs-test drift table. Unlike
calibrate's *curve*, evaluate applies a single fixed rule, so the natural shape is
a confusion matrix plus derived rates -- no sweep, no policy pick. All measurement
comes from ``calibration.metrics`` (so the grade cannot diverge from calibrate's
math); this module only formats it.
"""

from __future__ import annotations

from calibration.artifact import CalibrationArtifact
from calibration.metrics import (
    ScoredImage,
    build_breed_sweep,
    build_sweep,
    confusion_at,
)
from calibration.report_common import fmt as _fmt

#: The honest-labeling caveat (``docs/calibration_design.md`` Sec. 2 and 7): under
#: the breed-stratified split the look-alike breeds appear on both sides, so this
#: FPR is measured on look-alike breeds *also seen during calibration* -- not the
#: unseen-breed generalization exam the handoff's "real exam" language implies.
LOOKALIKE_NOTE = (
    "look-alike breeds also seen during calibration -- NOT the unseen-breed "
    "exam; see Sec. 8"
)


def _confusion_section(
    positives: list[ScoredImage], negatives: list[ScoredImage], threshold: float
) -> list[str]:
    c = confusion_at(positives, negatives, threshold)
    pos_label = f"Indy ({len(positives)})"
    neg_label = f"not  ({len(negatives)})"
    width = max(len(pos_label), len(neg_label))
    return [
        "",
        "Confusion at the frozen threshold:",
        f"  {'':<{width}}  {'pred Indy':>9}  {'pred not':>9}",
        f"  {pos_label:<{width}}  {c.tp:>9}  {c.fn:>9}",
        f"  {neg_label:<{width}}  {c.fp:>9}  {c.tn:>9}",
    ]


def _rates_section(
    positives: list[ScoredImage], negatives: list[ScoredImage], threshold: float
) -> list[str]:
    row = build_sweep(positives, negatives, [threshold])[0]
    return [
        "",
        "Rates at the frozen threshold:",
        f"  Recall (Indy):     {_fmt(row.recall)}",
        f"  FPR (all):         {_fmt(row.fpr_overall)}",
        f"  FPR (look-alike):  {_fmt(row.fpr_lookalike)}   [{LOOKALIKE_NOTE}]",
        f"  FPR (easy):        {_fmt(row.fpr_easy)}",
    ]


def _per_breed_section(negatives: list[ScoredImage], threshold: float) -> list[str]:
    breeds, fpr_by_breed = build_breed_sweep(negatives, [threshold])
    width = max((len(b) for b in breeds), default=5)
    lines = [
        "",
        "Per-breed FPR at the frozen threshold (breeds sorted worst-first):",
        f"  {'breed':<{width}}  {'FPR':>5}",
    ]
    for breed in breeds:
        lines.append(f"  {breed:<{width}}  {_fmt(fpr_by_breed[breed][0]):>5}")
    return lines


def _drift_section(
    artifact: CalibrationArtifact,
    positives: list[ScoredImage],
    negatives: list[ScoredImage],
) -> list[str]:
    row = build_sweep(positives, negatives, [artifact.threshold])[0]
    cal = artifact.metrics_at_threshold
    # Each row: (metric label, calibration value, test value). The dataclass
    # attribute names differ between the two sources (artifact uses fpr_all /
    # recall_indy; SweepRow uses fpr_overall / recall) -- mapped here.
    rows = [
        ("recall_indy", cal.recall_indy, row.recall),
        ("fpr_all", cal.fpr_all, row.fpr_overall),
        ("fpr_look_alike", cal.fpr_look_alike, row.fpr_lookalike),
        ("fpr_easy", cal.fpr_easy, row.fpr_easy),
    ]
    width = max(len(label) for label, _, _ in rows)
    lines = [
        "",
        "Generalization (calibration vs test, at the same frozen threshold):",
        f"  {'metric':<{width}}  {'calibration':>11}  {'test':>11}",
    ]
    for label, cal_v, test_v in rows:
        lines.append(f"  {label:<{width}}  {_fmt(cal_v):>11}  {_fmt(test_v):>11}")
    return lines


def build_report(
    artifact_label: str,
    manifest_label: str,
    artifact: CalibrationArtifact,
    positives: list[ScoredImage],
    negatives: list[ScoredImage],
) -> str:
    """Render the full textual evaluation report (E0): the honest grade.

    Applies ``artifact.threshold`` verbatim under ``artifact.aggregation`` and
    reports the confusion matrix, headline rates, per-breed FPR, and the
    calibration-vs-test drift table. No threshold is chosen here -- evaluate only
    grades the frozen one.
    """
    breeds = len({s.breed for s in negatives})
    header = [
        f"Evaluation: {artifact_label}  on test set {manifest_label}",
        f"  Frozen threshold: {artifact.threshold:.4f}   "
        f"(aggregation={artifact.aggregation}, "
        f"score {artifact.comparison} threshold -> Indy)",
        f"  Test positives: {len(positives)} Indy photos "
        "(held-out; never seen during setup)",
        f"  Test negatives: {len(negatives)} Oxford cats, {breeds} breeds",
    ]
    sections = [
        *_confusion_section(positives, negatives, artifact.threshold),
        *_rates_section(positives, negatives, artifact.threshold),
        *_per_breed_section(negatives, artifact.threshold),
        *_drift_section(artifact, positives, negatives),
    ]
    return "\n".join([*header, *sections])
