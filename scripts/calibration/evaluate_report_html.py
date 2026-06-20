"""Render the evaluation report as a self-contained semantic-HTML document.

The HTML half of ``evaluate.py``'s E0 output (``docs/calibration_design.md``
Sec. 7). A screen reader navigates scoped-header tables -- real ``<th
scope="col">``/``<th scope="row">`` cells -- far better than monospace ASCII, and
the confusion matrix, per-breed FPR, and drift table *are* tables, so HTML ships
from E0 (the error-list images arrive in E1). It builds **only** on
``calibration.report_common`` (the shared scoped-table/format primitives), so the
accessibility-critical markup stays in one place and cannot drift from the
calibration report. All measurement comes from ``calibration.metrics``.
"""

from __future__ import annotations

import html
from pathlib import Path

from calibration.artifact import CalibrationArtifact
from calibration.evaluate_report_text import LOOKALIKE_NOTE
from calibration.metrics import (
    ScoredImage,
    build_breed_sweep,
    build_sweep,
    confusion_at,
)
from calibration.report_common import HTML_STYLE, fmt_html, scoped_table


def _html_confusion(
    positives: list[ScoredImage], negatives: list[ScoredImage], threshold: float
) -> str:
    c = confusion_at(positives, negatives, threshold)
    body = [
        (f"Indy ({len(positives)})", [str(c.tp), str(c.fn)]),
        (f"not ({len(negatives)})", [str(c.fp), str(c.tn)]),
    ]
    return scoped_table(["pred Indy", "pred not"], body, corner="actual")


def _html_rates(
    positives: list[ScoredImage], negatives: list[ScoredImage], threshold: float
) -> str:
    row = build_sweep(positives, negatives, [threshold])[0]
    body = [
        ("Recall (Indy)", [fmt_html(row.recall)]),
        ("FPR (all)", [fmt_html(row.fpr_overall)]),
        ("FPR (look-alike)", [fmt_html(row.fpr_lookalike)]),
        ("FPR (easy)", [fmt_html(row.fpr_easy)]),
    ]
    table = scoped_table(["value"], body, corner="metric")
    return f"<p>FPR (look-alike): {html.escape(LOOKALIKE_NOTE)}.</p>" + table


def _html_per_breed(negatives: list[ScoredImage], threshold: float) -> str:
    breeds, fpr_by_breed = build_breed_sweep(negatives, [threshold])
    body = [
        (html.escape(breed), [fmt_html(fpr_by_breed[breed][0])]) for breed in breeds
    ]
    return scoped_table(["FPR"], body, corner="breed")


def _html_drift(
    artifact: CalibrationArtifact,
    positives: list[ScoredImage],
    negatives: list[ScoredImage],
) -> str:
    row = build_sweep(positives, negatives, [artifact.threshold])[0]
    cal = artifact.metrics_at_threshold
    rows = [
        ("recall_indy", cal.recall_indy, row.recall),
        ("fpr_all", cal.fpr_all, row.fpr_overall),
        ("fpr_look_alike", cal.fpr_look_alike, row.fpr_lookalike),
        ("fpr_easy", cal.fpr_easy, row.fpr_easy),
    ]
    body = [
        (label, [fmt_html(cal_v), fmt_html(test_v)]) for label, cal_v, test_v in rows
    ]
    return scoped_table(["calibration", "test"], body, corner="metric")


def render_report_html(
    artifact_label: str,
    manifest_label: str,
    artifact: CalibrationArtifact,
    positives: list[ScoredImage],
    negatives: list[ScoredImage],
) -> str:
    """Render the evaluation report as a self-contained semantic HTML document.

    Confusion matrix, headline rates, per-breed FPR at the frozen cutoff, and the
    calibration-vs-test drift table -- the same sections as the text report, each
    a scoped-header ``<table>``. No images in E0.
    """
    breeds = len({s.breed for s in negatives})
    esc_artifact = html.escape(artifact_label)
    esc_manifest = html.escape(manifest_label)
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>Evaluation report: {esc_artifact}</title>",
        f"<style>{HTML_STYLE}</style>",
        "</head>",
        "<body>",
        "<h1>Evaluation report</h1>",
        f"<p>Artifact: <code>{esc_artifact}</code></p>",
        f"<p>Test set: <code>{esc_manifest}</code></p>",
        "<h2>Summary</h2>",
        "<ul>"
        f"<li>Frozen threshold: {artifact.threshold:.4f} "
        f"(aggregation = {html.escape(artifact.aggregation)}, score "
        f"{html.escape(artifact.comparison)} threshold &rarr; Indy)</li>"
        f"<li>Test positives: {len(positives)} Indy photos "
        "(held-out; never seen during setup)</li>"
        f"<li>Test negatives: {len(negatives)} Oxford cats, {breeds} breeds</li>"
        "</ul>",
        "<h2>Confusion at the frozen threshold</h2>",
        _html_confusion(positives, negatives, artifact.threshold),
        "<h2>Rates at the frozen threshold</h2>",
        _html_rates(positives, negatives, artifact.threshold),
        "<h2>Per-breed FPR at the frozen threshold</h2>",
        "<p>Breeds sorted worst-first (highest max negative score).</p>",
        _html_per_breed(negatives, artifact.threshold),
        "<h2>Generalization (calibration vs test)</h2>",
        "<p>Each metric at the same frozen threshold, so the generalization gap "
        "is visible rather than a test number read in isolation.</p>",
        _html_drift(artifact, positives, negatives),
        "</body>",
        "</html>",
    ]
    return "\n".join(parts)


def write_report_html(
    path: Path,
    artifact_label: str,
    manifest_label: str,
    artifact: CalibrationArtifact,
    positives: list[ScoredImage],
    negatives: list[ScoredImage],
) -> None:
    """Render the HTML evaluation report and write it to ``path`` (parents made)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    document = render_report_html(
        artifact_label, manifest_label, artifact, positives, negatives
    )
    path.write_text(document, encoding="utf-8")
