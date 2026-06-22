"""Evaluate the decide stage: grade a frozen threshold against the held-out exam.

The mirror of calibrate on the other side of the boundary (``docs/calibration_
design.md`` Sec. 7). Where calibrate *explores* the gallery/calibration split to
choose a threshold, evaluate *grades* a frozen threshold against the ``test``
exam. It is the **only** place the ``test`` role is ever read, and it makes
exactly one decision rule -- the frozen threshold applied verbatim. No sweep, no
policy pick: those were already made and frozen into the artifact.

E0 is the first stage: load the artifact + the manifest's ``test`` roles, score
under the frozen aggregation, and emit the core tables (confusion matrix,
headline rates, per-breed FPR at the frozen cutoff, calibration-vs-test drift) in
both plain text (stdout) and semantic HTML (``--html``). No images, no
``--scores-out`` -- those arrive in E1.

It builds squarely on the calibration package (``artifact``, ``manifest``,
``scoring``, ``metrics``) so the numbers cannot diverge from calibrate's math.
``scripts/evaluate.py`` is the thin entry-point shim that calls :func:`main`.

Usage::

    # grade the frozen artifact against its own recorded test set
    uv run python scripts/evaluate.py --artifact data/artifacts/<...>.yaml

    # grade a different test set drawn from the SAME gallery (disjointness-guarded)
    uv run python scripts/evaluate.py --artifact <...>.yaml --manifest <other>.yaml

    # also write the semantic-HTML report (bare flag auto-names into data/reports/)
    uv run python scripts/evaluate.py --artifact <...>.yaml --html
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from _common import load_embeddings_variant
from calibration.artifact import CalibrationArtifact, load_artifact
from calibration.cache_variant import artifact_variant, assert_cache_matches_artifact
from calibration.evaluate_report_html import write_report_html
from calibration.evaluate_report_text import build_report, write_scores_csv
from calibration.manifest import (
    REPORTS_DIR,
    SplitConfigError,
    SplitManifest,
    indy_variant_dir,
    load_manifest,
    load_oxford_metadata,
    oxford_variant_dir,
)
from calibration.scoring import build_name_to_vector, score_role
from indycat.decision import Aggregation, Gallery

#: Sentinel for a bare ``--html`` (no path given) -> auto-name into REPORTS_DIR.
_HTML_AUTO = "<auto>"


def default_report_name(artifact_path: Path) -> str:
    """Auto-name the HTML report off the artifact stem (parallel to calibrate)."""
    return f"eval-{artifact_path.stem}.html"


def assert_same_experiment(
    manifest: SplitManifest, artifact: CalibrationArtifact
) -> None:
    """The disjointness guard: manifest and artifact must describe one experiment.

    Evaluate's gallery comes from the artifact; the test membership from the
    manifest. If a ``--manifest`` override points at a different-seed split, its
    ``test`` set could overlap the artifact's gallery and silently inflate recall
    by self-matching. Requiring the gallery sets to be identical defends that
    path; a mismatch is loud (``docs/calibration_design.md`` Sec. 7). Shared with
    the cat-breeds eval, whose Indy positives come from the same manifest role.
    """
    artifact_gallery = {img.source_filename for img in artifact.gallery_images}
    if set(manifest.indy_gallery) != artifact_gallery:
        raise SplitConfigError(
            "manifest and artifact describe different experiments: the manifest's "
            "Indy gallery does not match the artifact's frozen gallery. Pass the "
            "--manifest that the artifact was calibrated against (a different test "
            "set must still be drawn from the SAME gallery)."
        )


def run_evaluation(
    artifact: CalibrationArtifact,
    raw_gallery_vectors: NDArray[np.float32],
    manifest: SplitManifest,
    artifact_label: str,
    manifest_label: str,
    html_out: Path | None,
    scores_out: Path | None = None,
) -> None:
    """Score the ``test`` roles against the frozen gallery and print the grade.

    The gallery comes from the artifact (never re-derived from the manifest -- the
    binding the fingerprint protects); scoring uses ``artifact.aggregation`` and
    the frozen ``artifact.threshold`` verbatim.
    """
    assert_same_experiment(manifest, artifact)
    if not manifest.indy_test or not manifest.oxford_test:
        raise SplitConfigError(
            "test set is empty (indy_test or oxford_test has no images); there is "
            "nothing to grade. Regenerate the split with --test > 0."
        )

    # The test-set caches are dictated by the frozen artifact's embedding identity
    # -- no CLI flag selects them. Resolve both variant dirs, load with provenance,
    # and assert each loaded sidecar matches the artifact before any scoring runs.
    variant = artifact_variant(artifact.embedding)
    oxford_dir = oxford_variant_dir(variant)
    indy_names, indy_vectors, indy_meta = load_embeddings_variant(
        indy_variant_dir(variant)
    )
    oxford_names, oxford_vectors, oxford_meta = load_embeddings_variant(oxford_dir)
    assert_cache_matches_artifact(indy_meta, artifact.embedding, "Indy")
    assert_cache_matches_artifact(oxford_meta, artifact.embedding, "Oxford")

    indy_lookup = build_name_to_vector(indy_names, indy_vectors)
    oxford_lookup = build_name_to_vector(oxford_names, oxford_vectors)
    breeds = {
        record.source_filename: record.breed
        for record in load_oxford_metadata(oxford_dir / "metadata.csv")
    }

    # The aggregation is read from the artifact, never a CLI flag: the threshold
    # is only meaningful under the aggregation that produced it.
    aggregation = cast(Aggregation, artifact.aggregation)
    gallery_names = [img.source_filename for img in artifact.gallery_images]
    gallery = Gallery.from_raw(gallery_names, raw_gallery_vectors)
    positives = score_role(manifest.indy_test, indy_lookup, gallery, aggregation)
    negatives = score_role(
        manifest.oxford_test, oxford_lookup, gallery, aggregation, breeds=breeds
    )

    print()
    print(build_report(artifact_label, manifest_label, artifact, positives, negatives))
    if scores_out is not None:
        write_scores_csv(scores_out, positives, negatives, artifact.threshold)
        print(f"\nPer-image scores written to {scores_out}")
    if html_out is not None:
        write_report_html(
            html_out, artifact_label, manifest_label, artifact, positives, negatives
        )
        print(f"\nHTML report written to {html_out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grade a frozen calibration artifact against the held-out test "
        "exam (the only place the test role is read)."
    )
    parser.add_argument(
        "--artifact",
        required=True,
        help="path to the frozen calibration.yaml artifact to grade.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="split manifest for the test membership (defaults to the artifact's "
        "recorded chosen_by.manifest; override to grade a different test set drawn "
        "from the SAME gallery).",
    )
    parser.add_argument(
        "--html",
        nargs="?",
        const=_HTML_AUTO,
        default=None,
        help="also write a semantic-HTML report; bare flag auto-names into "
        "data/reports/, or pass an explicit path.",
    )
    parser.add_argument(
        "--scores-out",
        default=None,
        help="optional CSV of per-image test scores joined with provenance and the "
        "frozen verdict.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        artifact_path = Path(args.artifact)
        artifact, raw_vectors = load_artifact(artifact_path)

        manifest_label = (
            args.manifest if args.manifest is not None else artifact.chosen_by.manifest
        )
        manifest_path = Path(manifest_label)
        if not manifest_path.exists():
            raise SplitConfigError(
                f"manifest not found: {manifest_path}. The artifact records "
                f"'{artifact.chosen_by.manifest}', which may be relative to a "
                "different working directory; pass --manifest with a resolvable path."
            )
        manifest = load_manifest(manifest_path)

        if args.html is None:
            html_out = None
        elif args.html == _HTML_AUTO:
            html_out = REPORTS_DIR / default_report_name(artifact_path)
        else:
            html_out = Path(args.html)

        scores_out = Path(args.scores_out) if args.scores_out is not None else None

        run_evaluation(
            artifact,
            raw_vectors,
            manifest,
            str(artifact_path),
            str(manifest_path),
            html_out,
            scores_out,
        )
    except SplitConfigError as err:
        raise SystemExit(str(err)) from err


if __name__ == "__main__":
    main()
