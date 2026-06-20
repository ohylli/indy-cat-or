"""Calibration of the decide stage: split manifests, scoring, and the V0+V1 report.

The V0+V1 slice of ``docs/calibration_design.md`` Sec. 5, packaged into focused
modules:

* :mod:`calibration.manifest` -- generate/load the reproducible split manifest.
* :mod:`calibration.scoring` -- score a manifest's roles against the gallery via
  the ``indycat.decision`` core.
* :mod:`calibration.metrics` -- pure measurement: distributions, the threshold
  sweep, and the risk lists (shared by both renderers and the future evaluator).
* :mod:`calibration.report_text` / :mod:`calibration.report_html` -- the two
  renderings of the report.
* :mod:`calibration.cli` -- the ``calibrate`` command tying it together (the
  ``scripts/calibrate.py`` shim is its entry point).

The names below are re-exported as the package's public surface so callers (and
tests) can ``import calibration`` without reaching into submodules.
"""

from __future__ import annotations

from calibration.metrics import (
    LOOKALIKE_BREEDS,
    PICK_POLICIES,
    RISK_ROWS,
    TARGET_GROUPS,
    ScoredImage,
    Stats,
    SweepRow,
    ThresholdChoice,
    build_breed_sweep,
    build_sweep,
    candidate_cutoffs,
    pick_threshold,
    select_risk_rows,
    summarize,
    sweep_thresholds,
)
from calibration.report_html import render_report_html, write_report_html
from calibration.report_text import build_report, write_scores_csv
from calibration.scoring import build_name_to_vector, score_role, select_vectors

__all__ = [
    "LOOKALIKE_BREEDS",
    "PICK_POLICIES",
    "RISK_ROWS",
    "TARGET_GROUPS",
    "ScoredImage",
    "Stats",
    "SweepRow",
    "ThresholdChoice",
    "build_breed_sweep",
    "build_name_to_vector",
    "build_report",
    "build_sweep",
    "candidate_cutoffs",
    "pick_threshold",
    "render_report_html",
    "score_role",
    "select_risk_rows",
    "select_vectors",
    "summarize",
    "sweep_thresholds",
    "write_report_html",
    "write_scores_csv",
]
