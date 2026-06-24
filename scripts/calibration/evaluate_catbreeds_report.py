"""Render the cat-breeds evaluation as screen-reader-first plain text.

A sibling of ``evaluate_report_text`` for a different negatives source: the whole
cat-breeds (Kaggle ma7555) dataset rather than the manifest's Oxford ``test``
role. The shape is the same fixed-rule grade -- confusion matrix at the frozen
threshold, headline rates, per-breed FPR, the actual errors -- so it reuses
``calibration.metrics`` verbatim (the grade cannot diverge from calibrate's
math); only the look-alike grouping and the framing differ.

Two differences from the Oxford report:
* **Look-alike set.** Oxford's four long-haired breeds are replaced by this
  dataset's long-haired breeds (:data:`CATBREEDS_LOOKALIKE_BREEDS`), and
  Norwegian Forest Cat -- the headline look-alike -- is broken out on its own.
* **Framing.** These breeds never appear in the gallery or calibration, so this
  *is* the unseen-breed exam the Oxford report's ``LOOKALIKE_NOTE`` warns it is
  not. There is no calibration-vs-test drift table: the calibration look-alikes
  (Oxford) and these are different breed sets, so the comparison would mislead.
"""

from __future__ import annotations

from calibration.artifact import CalibrationArtifact
from calibration.metrics import (
    ScoredImage,
    build_breed_table,
    build_sweep,
    confusion_at,
    select_error_rows,
)
from calibration.report_common import fmt as _fmt

#: Norwegian Forest Cat, the headline long-haired look-alike, named once.
NFC_BREED = "Norwegian Forest Cat"

#: The cat-breeds long-haired breeds that look most like Indy -- this dataset's
#: hard-negative tail, the analogue of Oxford's ``LOOKALIKE_BREEDS``. Folder names
#: (spaces, not Oxford's underscores). Reported as one group, with NFC also broken
#: out and every breed visible in the per-breed table regardless.
CATBREEDS_LOOKALIKE_BREEDS = frozenset(
    {
        NFC_BREED,
        "Maine Coon",
        "Ragdoll",
        "Siberian",
        "Persian",
        "Himalayan",
        "Turkish Angora",
        "Birman",
        "Nebelung",
        "Somali",
        "Ragamuffin",
        "Domestic Long Hair",
        "Balinese",
        "Turkish Van",
    }
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
    positives: list[ScoredImage],
    negatives: list[ScoredImage],
    threshold: float,
    nfc_fpr: float,
) -> list[str]:
    row = build_sweep(
        positives, negatives, [threshold], lookalike_breeds=CATBREEDS_LOOKALIKE_BREEDS
    )[0]
    return [
        "",
        "Rates at the frozen threshold:",
        f"  Recall (Indy):       {_fmt(row.recall)}",
        f"  FPR (all):           {_fmt(row.fpr_overall)}",
        f"  FPR (long-haired):   {_fmt(row.fpr_lookalike)}   "
        "[unseen long-haired look-alikes -- the real exam]",
        f"  FPR (NFC only):      {_fmt(nfc_fpr)}",
        f"  FPR (other):         {_fmt(row.fpr_easy)}",
    ]


def _per_breed_section(
    negatives: list[ScoredImage], threshold: float
) -> tuple[list[str], float]:
    """The per-breed FPR table; also returns the NFC FPR for the rates section."""
    rows = build_breed_table(negatives, threshold)
    width = max((len(r.breed) for r in rows), default=5)
    lines = [
        "",
        "Per-breed FPR at the frozen threshold (breeds sorted by FPR, highest first):",
        f"  {'breed':<{width}}  {'FPR':>5}  {'cats':>5}",
    ]
    for r in rows:
        lines.append(f"  {r.breed:<{width}}  {_fmt(r.fpr):>5}  {r.count:>5}")
    nfc_fpr = next((r.fpr for r in rows if r.breed == NFC_BREED), float("nan"))
    return lines, nfc_fpr


def _error_lists_section(
    positives: list[ScoredImage], negatives: list[ScoredImage], threshold: float
) -> list[str]:
    """The actual mistakes at the frozen cutoff: false positives, then false negatives.

    Uncapped (every error shown). False positives are the cat-breeds cats wrongly
    called Indy -- the headline failure this eval hunts for.
    """
    false_pos, false_neg = select_error_rows(positives, negatives, threshold)
    lines = ["", "False positives (cat-breeds cats wrongly called Indy):"]
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


def build_report(
    artifact_label: str,
    dataset_label: str,
    artifact: CalibrationArtifact,
    positives: list[ScoredImage],
    negatives: list[ScoredImage],
) -> str:
    """Render the full textual cat-breeds evaluation report (the honest grade).

    Applies ``artifact.threshold`` verbatim under ``artifact.aggregation`` and
    reports the confusion matrix, headline rates (with NFC broken out), and
    per-breed FPR over the whole cat-breeds dataset as negatives, plus the actual
    errors. No threshold is chosen here -- this only grades the frozen one.
    """
    breeds = len({s.breed for s in negatives})
    header = [
        f"Cat-breeds evaluation: {artifact_label}  on {dataset_label}",
        f"  Frozen threshold: {artifact.threshold:.4f}   "
        f"(aggregation={artifact.aggregation}, "
        f"score {artifact.comparison} threshold -> Indy)",
        f"  Positives: {len(positives)} held-out Indy photos "
        "(from the artifact's manifest test role)",
        f"  Negatives: {len(negatives)} cat-breeds cats, {breeds} breeds "
        "(all unseen during calibration)",
    ]
    per_breed_lines, nfc_fpr = _per_breed_section(negatives, artifact.threshold)
    sections = [
        *_confusion_section(positives, negatives, artifact.threshold),
        *_rates_section(positives, negatives, artifact.threshold, nfc_fpr),
        *per_breed_lines,
        *_error_lists_section(positives, negatives, artifact.threshold),
    ]
    return "\n".join([*header, *sections])
