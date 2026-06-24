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

import csv
from pathlib import Path

from calibration.artifact import CalibrationArtifact
from calibration.metrics import (
    ScoredImage,
    build_breed_table,
    build_sweep,
    confusion_at,
    select_error_rows,
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
    rows = build_breed_table(negatives, threshold)
    width = max((len(r.breed) for r in rows), default=5)
    lines = [
        "",
        "Per-breed FPR at the frozen threshold (breeds sorted by FPR, highest first):",
        f"  {'breed':<{width}}  {'FPR':>5}  {'cats':>5}",
    ]
    for r in rows:
        lines.append(f"  {r.breed:<{width}}  {_fmt(r.fpr):>5}  {r.count:>5}")
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


def _error_lists_section(
    positives: list[ScoredImage], negatives: list[ScoredImage], threshold: float
) -> list[str]:
    """The actual mistakes at the frozen cutoff (E1): false positives, then negatives.

    Inverts calibrate's *risk* lists -- these are the real errors the frozen rule
    makes, uncapped (every one shown). The HTML report adds the crops; the text
    lines mirror calibrate's ``_risk_sections`` format so a screen reader gets the
    error inventory without images.
    """
    false_pos, false_neg = select_error_rows(positives, negatives, threshold)
    lines = ["", "False positives (negatives that cleared the bar):"]
    if false_pos:
        for s in false_pos:
            lines.append(
                f"  {_fmt(s.score)}  {s.name}  ({s.breed})  "
                f"-> best match {s.best_match}"
            )
    else:
        lines.append("  (none)")
    lines += ["", "False negatives (Indy missed):"]
    if false_neg:
        for s in false_neg:
            lines.append(f"  {_fmt(s.score)}  {s.name}  -> best match {s.best_match}")
    else:
        lines.append("  (none)")
    return lines


def write_scores_csv(
    path: Path,
    positives: list[ScoredImage],
    negatives: list[ScoredImage],
    threshold: float,
) -> None:
    """Write per-image test scores joined with provenance + the frozen verdict (E1).

    Parallel to calibrate's ``write_scores_csv`` but with a ``verdict`` column
    unique to evaluate (the threshold is fixed here): ``Indy`` when ``score >=
    threshold`` else ``not``. Negatives first (worst false-positive risk on top),
    then positives (hardest to recognise on top), so the rows that matter lead.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    def verdict(score: float) -> str:
        return "Indy" if score >= threshold else "not"

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["role", "source_filename", "score", "verdict", "best_match", "breed"]
        )
        for s in sorted(negatives, key=lambda s: s.score, reverse=True):
            writer.writerow(
                [
                    "negative",
                    s.name,
                    f"{s.score:.6f}",
                    verdict(s.score),
                    s.best_match,
                    s.breed,
                ]
            )
        for s in sorted(positives, key=lambda s: s.score):
            writer.writerow(
                [
                    "positive",
                    s.name,
                    f"{s.score:.6f}",
                    verdict(s.score),
                    s.best_match,
                    "",
                ]
            )


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
        *_error_lists_section(positives, negatives, artifact.threshold),
    ]
    return "\n".join([*header, *sections])
