"""Grade a frozen artifact against the whole cat-breeds dataset as negatives.

A second held-out exam alongside ``evaluate.py``. Where ``evaluate.py`` grades
the frozen threshold against the manifest's Oxford ``test`` negatives, this grades
it against the **entire cat-breeds (Kaggle ma7555) dataset** -- a false-positive
stress test on tens of thousands of unseen cats, none of which appear in the
gallery or calibration. The Indy positives are the same held-out ``test`` photos
from the artifact's manifest, so Indy recall and cat-breeds FPR are reported side
by side under one frozen rule.

It reuses calibrate's machinery wholesale -- the artifact, the manifest, the
``scoring`` bridge, and ``metrics`` (via the cat-breeds report) -- so the grade
cannot diverge from calibrate's math. The only thing that differs from
``evaluate.py`` is the negative source: every cat-breeds row, not a manifest role.
``scripts/evaluate_catbreeds.py`` is the thin entry-point shim.

Usage::

    uv run python scripts/evaluate_catbreeds.py --artifact data/artifacts/<...>.yaml
    uv run python scripts/evaluate_catbreeds.py --artifact <...>.yaml --scores-out s.csv
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
from calibration.evaluate import assert_same_experiment
from calibration.evaluate_catbreeds_report import build_report
from calibration.evaluate_report_text import write_scores_csv
from calibration.manifest import (
    SplitConfigError,
    SplitManifest,
    catbreeds_variant_dir,
    indy_variant_dir,
    load_manifest,
    load_oxford_metadata,
)
from calibration.scoring import build_name_to_vector, score_role
from indycat.decision import Aggregation, Gallery


def run_evaluation(
    artifact: CalibrationArtifact,
    raw_gallery_vectors: NDArray[np.float32],
    manifest: SplitManifest,
    artifact_label: str,
    dataset_label: str,
    scores_out: Path | None = None,
) -> None:
    """Score the Indy test positives + all cat-breeds negatives against the gallery.

    The gallery, threshold, and aggregation all come from the frozen artifact; the
    Indy positives come from the manifest's ``test`` role (guarded same-experiment
    against the artifact's gallery). Every cat-breeds row is scored as a negative.
    """
    assert_same_experiment(manifest, artifact)
    if not manifest.indy_test:
        raise SplitConfigError(
            "the manifest's Indy test role is empty; there are no positives to "
            "grade. Regenerate the split with --test > 0."
        )

    # Both caches are dictated by the frozen artifact's embedding identity -- no CLI
    # flag selects them. Load with provenance and assert each matches before scoring.
    variant = artifact_variant(artifact.embedding)
    catbreeds_dir = catbreeds_variant_dir(variant)
    indy_names, indy_vectors, indy_meta = load_embeddings_variant(
        indy_variant_dir(variant)
    )
    cb_names, cb_vectors, cb_meta = load_embeddings_variant(catbreeds_dir)
    assert_cache_matches_artifact(indy_meta, artifact.embedding, "Indy")
    assert_cache_matches_artifact(cb_meta, artifact.embedding, "cat-breeds")
    if not cb_names:
        raise SplitConfigError(
            f"the cat-breeds cache at {catbreeds_dir} is empty; build it first with "
            "scripts/build_catbreeds_negatives.py"
        )

    indy_lookup = build_name_to_vector(indy_names, indy_vectors)
    cb_lookup = build_name_to_vector(cb_names, cb_vectors)
    breeds = {
        record.source_filename: record.breed
        for record in load_oxford_metadata(catbreeds_dir / "metadata.csv")
    }

    aggregation = cast(Aggregation, artifact.aggregation)
    gallery_names = [img.source_filename for img in artifact.gallery_images]
    gallery = Gallery.from_raw(gallery_names, raw_gallery_vectors)
    positives = score_role(manifest.indy_test, indy_lookup, gallery, aggregation)
    # The whole dataset is the negative pool: score every cat-breeds row.
    negatives = score_role(cb_names, cb_lookup, gallery, aggregation, breeds=breeds)

    print()
    print(build_report(artifact_label, dataset_label, artifact, positives, negatives))
    if scores_out is not None:
        write_scores_csv(scores_out, positives, negatives, artifact.threshold)
        print(f"\nPer-image scores written to {scores_out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grade a frozen calibration artifact against the whole "
        "cat-breeds dataset as negatives (a false-positive stress test)."
    )
    parser.add_argument(
        "--artifact",
        required=True,
        help="path to the frozen calibration.yaml artifact to grade.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="split manifest for the Indy positives (defaults to the artifact's "
        "recorded chosen_by.manifest; override to use a different test set drawn "
        "from the SAME gallery).",
    )
    parser.add_argument(
        "--scores-out",
        default=None,
        help="optional CSV of per-image scores joined with provenance and the "
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

        scores_out = Path(args.scores_out) if args.scores_out is not None else None

        run_evaluation(
            artifact,
            raw_vectors,
            manifest,
            str(artifact_path),
            "the cat-breeds dataset",
            scores_out,
        )
    except SplitConfigError as err:
        raise SystemExit(str(err)) from err


if __name__ == "__main__":
    main()
