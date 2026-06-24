"""Render the cat-breeds evaluation as a self-contained semantic-HTML document.

The HTML sibling of :mod:`calibration.evaluate_catbreeds_report` (the text grade),
the same way :mod:`calibration.evaluate_report_html` is the HTML sibling of the
Oxford text report. A screen reader navigates scoped-header tables -- real ``<th
scope="col">``/``<th scope="row">`` cells -- far better than monospace ASCII, so
the confusion matrix, headline rates, and per-breed FPR ship as tables, followed
by the actual error lists with crops. It builds only on
:mod:`calibration.report_common` (the shared accessible-markup primitives) and
:mod:`calibration.metrics` (all measurement), so it cannot drift from the rest.

Two differences from the Oxford HTML report mirror the text report's:
* **Look-alike set + NFC.** The headline rates use this dataset's long-haired
  :data:`~calibration.evaluate_catbreeds_report.CATBREEDS_LOOKALIKE_BREEDS` and
  break Norwegian Forest Cat out on its own row.
* **No drift table.** The calibration look-alikes (Oxford) and these are
  different breed sets, so a calibration-vs-test comparison would mislead.

The false-positive list is capped at :data:`MAX_FP_FIGURES` embedded figure-rows
(124+ FPs is too many crops for a screen reader); a count note points at the text
report / ``--scores-out`` CSV for the full, uncapped list.
"""

from __future__ import annotations

import html
from pathlib import Path

from calibration.artifact import CalibrationArtifact
from calibration.evaluate_catbreeds_report import CATBREEDS_LOOKALIKE_BREEDS, NFC_BREED
from calibration.manifest import CATBREEDS_IMAGE_DIR, INDY_IMAGE_DIR
from calibration.metrics import (
    BreedFpr,
    ScoredImage,
    build_breed_table,
    build_sweep,
    confusion_at,
    select_error_rows,
)
from calibration.report_common import (
    HTML_STYLE,
    figure_list,
    fmt_html,
    scoped_table,
)

#: Max false-positive figure-rows embedded in the HTML; the rest get a count note.
#: The full uncapped list lives in the text report and the --scores-out CSV.
MAX_FP_FIGURES = 20


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
    positives: list[ScoredImage],
    negatives: list[ScoredImage],
    threshold: float,
    nfc_fpr: float,
) -> str:
    row = build_sweep(
        positives, negatives, [threshold], lookalike_breeds=CATBREEDS_LOOKALIKE_BREEDS
    )[0]
    body = [
        ("Recall (Indy)", [fmt_html(row.recall)]),
        ("FPR (all)", [fmt_html(row.fpr_overall)]),
        ("FPR (long-haired)", [fmt_html(row.fpr_lookalike)]),
        ("FPR (NFC only)", [fmt_html(nfc_fpr)]),
        ("FPR (other)", [fmt_html(row.fpr_easy)]),
    ]
    table = scoped_table(["value"], body, corner="metric")
    note = (
        "<p>FPR (long-haired): over this dataset's long-haired breeds "
        "&mdash; unseen during calibration, so this is the real exam.</p>"
    )
    return note + table


def _html_per_breed(rows: list[BreedFpr]) -> str:
    body = [(html.escape(r.breed), [fmt_html(r.fpr), str(r.count)]) for r in rows]
    return scoped_table(["FPR", "cats"], body, corner="breed")


def _html_error_lists(
    positives: list[ScoredImage],
    negatives: list[ScoredImage],
    threshold: float,
    html_dir: Path,
    breeds: dict[str, str],
    indy_image_dir: Path,
    catbreeds_image_dir: Path,
) -> list[str]:
    """The error lists at the frozen cutoff, with crops.

    False positives are the cat-breeds cats wrongly called Indy -- the headline
    failure this eval hunts for. Cat-breeds images are nested per breed, so the
    candidate figure is resolved to ``<breed>/<file>`` via ``breeds``. The list is
    capped at :data:`MAX_FP_FIGURES` crops with a count note (no silent
    truncation -- the full list is in the text report / CSV). False negatives are
    few (Indy missed) so they stay uncapped.
    """

    def resolve_catbreeds(name: str) -> str:
        # source_filename is bare; the image lives at <breed>/<file>.
        return f"{breeds[name]}/{name}"

    false_pos, false_neg = select_error_rows(positives, negatives, threshold)
    parts = ["<h2>False positives (cat-breeds cats wrongly called Indy)</h2>"]
    if false_pos:
        shown = false_pos[:MAX_FP_FIGURES]
        parts.append(
            f"<p>{len(false_pos)} cat(s) cleared the bar; showing the "
            f"{len(shown)} worst with crops.</p>"
        )
        parts.append(
            figure_list(
                shown,
                catbreeds_image_dir,
                indy_image_dir,
                html_dir,
                show_breed=True,
                candidate_resolver=resolve_catbreeds,
            )
        )
        if len(false_pos) > MAX_FP_FIGURES:
            parts.append(
                f"<p>+{len(false_pos) - MAX_FP_FIGURES} more not shown; see the "
                "text report or the --scores-out CSV for the full list.</p>"
            )
    else:
        parts.append("<p>None: no cat-breeds cat cleared the bar.</p>")
    parts.append("<h2>False negatives (Indy missed)</h2>")
    if false_neg:
        parts.append(
            f"<p>{len(false_neg)} Indy photo(s) scored below the frozen threshold.</p>"
        )
        parts.append(
            figure_list(
                false_neg, indy_image_dir, indy_image_dir, html_dir, show_breed=False
            )
        )
    else:
        parts.append("<p>None: every Indy photo cleared the bar.</p>")
    return parts


def render_catbreeds_report_html(
    artifact_label: str,
    dataset_label: str,
    artifact: CalibrationArtifact,
    positives: list[ScoredImage],
    negatives: list[ScoredImage],
    breeds: dict[str, str],
    *,
    html_path: Path,
    indy_image_dir: Path = INDY_IMAGE_DIR,
    catbreeds_image_dir: Path = CATBREEDS_IMAGE_DIR,
) -> str:
    """Render the cat-breeds evaluation as a self-contained semantic HTML document.

    Confusion matrix, headline rates (with NFC broken out), and per-breed FPR at
    the frozen cutoff -- each a scoped-header ``<table>`` -- followed by the error
    lists, which embed the actual misclassified cats beside the gallery photo each
    best matched. No drift table (see the module docstring). ``breeds`` maps each
    negative's ``source_filename`` to its breed folder so the nested cat-breeds
    image path resolves. Image ``src`` paths are relative to ``html_path`` so the
    file is portable.
    """
    html_dir = html_path.parent
    n_breeds = len({s.breed for s in negatives})
    esc_artifact = html.escape(artifact_label)
    esc_dataset = html.escape(dataset_label)
    breed_rows = build_breed_table(negatives, artifact.threshold)
    nfc_fpr = next((r.fpr for r in breed_rows if r.breed == NFC_BREED), float("nan"))
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>Cat-breeds evaluation report: {esc_artifact}</title>",
        f"<style>{HTML_STYLE}</style>",
        "</head>",
        "<body>",
        "<h1>Cat-breeds evaluation report</h1>",
        f"<p>Artifact: <code>{esc_artifact}</code></p>",
        f"<p>Dataset: <code>{esc_dataset}</code></p>",
        "<h2>Summary</h2>",
        "<ul>"
        f"<li>Frozen threshold: {artifact.threshold:.4f} "
        f"(aggregation = {html.escape(artifact.aggregation)}, score "
        f"{html.escape(artifact.comparison)} threshold &rarr; Indy)</li>"
        f"<li>Positives: {len(positives)} held-out Indy photos "
        "(from the artifact's manifest test role)</li>"
        f"<li>Negatives: {len(negatives)} cat-breeds cats, {n_breeds} breeds "
        "(all unseen during calibration)</li>"
        "</ul>",
        "<h2>Confusion at the frozen threshold</h2>",
        _html_confusion(positives, negatives, artifact.threshold),
        "<h2>Rates at the frozen threshold</h2>",
        _html_rates(positives, negatives, artifact.threshold, nfc_fpr),
        "<h2>Per-breed FPR at the frozen threshold</h2>",
        "<p>Breeds sorted by FPR, highest first.</p>",
        _html_per_breed(breed_rows),
        *_html_error_lists(
            positives,
            negatives,
            artifact.threshold,
            html_dir,
            breeds,
            indy_image_dir,
            catbreeds_image_dir,
        ),
        "</body>",
        "</html>",
    ]
    return "\n".join(parts)


def write_report_html(
    path: Path,
    artifact_label: str,
    dataset_label: str,
    artifact: CalibrationArtifact,
    positives: list[ScoredImage],
    negatives: list[ScoredImage],
    breeds: dict[str, str],
) -> None:
    """Render the cat-breeds HTML report and write it to ``path`` (parents made)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    document = render_catbreeds_report_html(
        artifact_label,
        dataset_label,
        artifact,
        positives,
        negatives,
        breeds,
        html_path=path,
    )
    path.write_text(document, encoding="utf-8")
