"""Calibrate the decide stage: split, score, report, and freeze the artifact.

The full V0--V3 command of ``docs/calibration_design.md`` Sec. 5. It generates
(or replays) a reproducible split manifest, then scores the held-back Indy
positives and the Oxford negatives against the gallery and prints the textual
report: the V0 distributions (do the positive and negative scores separate at
all?) and the V1 threshold sweep (a ``cutoff -> FPR , recall`` trade-off table
plus a per-breed FPR table). With ``--policy`` it also picks a cutoff (V2); with
``--artifact`` it freezes the V3 calibration artifact -- a ``calibration.yaml`` +
companion ``.gallery.npy`` decide consumes -- running the pick under both
aggregations and recording the FPR-first winner as the operative one.

Generation is folded into this one command (no separate generate/calibrate
dance). The split logic lives in the reusable ``calibration.manifest`` module;
the scoring/report logic in ``calibration.scoring`` / ``calibration.report_text``
/ ``calibration.report_html`` over the ``indycat.decision`` core; the artifact in
``calibration.artifact``. ``--generate-only`` stops after writing the manifest.
``scripts/calibrate.py`` is the thin entry-point shim that calls :func:`main`.

Usage::

    # zero-arg baseline: built-in defaults + built-in seed -> identical every time
    uv run python scripts/calibrate.py

    # generate only, then stop (skip scoring)
    uv run python scripts/calibrate.py --generate-only

    # specify the split
    uv run python scripts/calibrate.py --gallery 15 --calibration 10 --test 10 --seed 42

    # the alternate aggregation, and an optional per-image score dump
    uv run python scripts/calibrate.py --aggregation mean-top3 \
        --scores-out data/splits/scores.csv

    # a coarser threshold-sweep grid (fewer cutoff rows)
    uv run python scripts/calibrate.py --sweep-step 0.1

    # also write an HTML report with embedded images (bare flag auto-names
    # into data/reports/; or pass an explicit path)
    uv run python scripts/calibrate.py --html
    uv run python scripts/calibrate.py --html data/reports/run.html

    # freeze the calibration artifact (V3): runs both aggregations, auto-selects
    # the FPR-first winner, writes data/artifacts/<auto>.yaml + .gallery.npy
    uv run python scripts/calibrate.py --policy target-fpr --artifact

    # fresh random seed (the drawn seed is recorded in the written manifest)
    uv run python scripts/calibrate.py --random-seed

    # replay an exact prior split (scores it too)
    uv run python scripts/calibrate.py --manifest data/splits/run-<...>.yaml
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from _common import load_cached_embeddings
from calibration.artifact import (
    GALLERY_VECTORS_SUFFIX,
    AggregationResult,
    build_artifact,
    load_indy_positions,
    write_artifact,
)
from calibration.manifest import (
    ARTIFACTS_DIR,
    DEFAULT_CALIBRATION,
    DEFAULT_GALLERY,
    DEFAULT_OXFORD_TEST_FRACTION,
    DEFAULT_SEED,
    DEFAULT_TEST,
    INDY_EMBEDDINGS,
    INDY_METADATA,
    OXFORD_EMBEDDINGS,
    OXFORD_METADATA,
    PREFER_CHOICES,
    REPORTS_DIR,
    SPLITS_DIR,
    STRATEGY_THREE_WAY,
    GenerationParams,
    SplitConfigError,
    SplitManifest,
    generate_three_way,
    load_indy_metadata,
    load_manifest,
    load_oxford_metadata,
    write_manifest,
)
from calibration.metrics import (
    PICK_POLICIES,
    TARGET_GROUPS,
    PickPolicy,
    TargetGroup,
    pick_threshold,
)
from calibration.report_html import write_report_html
from calibration.report_text import build_report, write_scores_csv
from calibration.scoring import build_name_to_vector, score_role, select_vectors
from indycat.decision import AGGREGATIONS, Aggregation, Gallery

#: Sentinel for a bare ``--html`` (no path given) -> auto-name into REPORTS_DIR.
_HTML_AUTO = "<auto>"

#: Sentinel for a bare ``--artifact`` (no path) -> auto-name into ARTIFACTS_DIR.
_ARTIFACT_AUTO = "<auto>"

#: Generation flags; if any is explicitly set, --manifest (replay) is rejected.
_GENERATION_FLAGS = (
    "gallery",
    "calibration",
    "test",
    "oxford_test_fraction",
    "prefer",
    "seed",
    "random_seed",
)


def build_parser() -> argparse.ArgumentParser:
    """The CLI: generation flags, a seed group, and the --manifest replay path."""
    parser = argparse.ArgumentParser(
        description="Calibrate the decide stage (currently: generate a split manifest)."
    )
    # Generation flags default to None so we can tell an explicit value from the
    # default, which is what lets --manifest reject being combined with them.
    parser.add_argument(
        "--gallery",
        type=int,
        default=None,
        help=f"Indy gallery size (default {DEFAULT_GALLERY})",
    )
    parser.add_argument(
        "--calibration",
        type=int,
        default=None,
        help=f"Indy calibration-positive size (default {DEFAULT_CALIBRATION})",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=None,
        help=f"Indy held-out test size (default {DEFAULT_TEST})",
    )
    parser.add_argument(
        "--oxford-test-fraction",
        type=float,
        default=None,
        help=f"fraction of each Oxford breed held out for test "
        f"(default {DEFAULT_OXFORD_TEST_FRACTION})",
    )
    parser.add_argument(
        "--prefer",
        choices=PREFER_CHOICES,
        default=None,
        help="bias the gallery toward head/tail-visible photos (off by default)",
    )
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seed", type=int, default=None, help=f"split seed (default {DEFAULT_SEED})"
    )
    seed_group.add_argument(
        "--random-seed",
        action="store_true",
        help="draw a fresh seed and record it in the written manifest",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="load + validate an existing manifest instead of generating "
        "(mutually exclusive with the generation flags)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output path for the manifest (default: data/splits/<auto-name>.yaml)",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="generate + save the manifest and stop (do not attempt calibration)",
    )
    # Scoring options -- not generation flags, so they compose with --manifest.
    parser.add_argument(
        "--aggregation",
        choices=AGGREGATIONS,
        default="max",
        help="how a query's per-gallery similarities collapse to one score "
        "(default: max)",
    )
    parser.add_argument(
        "--scores-out",
        default=None,
        help="optional CSV of per-image scores joined with provenance",
    )
    parser.add_argument(
        "--sweep-step",
        type=float,
        default=0.05,
        help="cutoff granularity for the threshold-sweep trade-off table "
        "(default: 0.05)",
    )
    parser.add_argument(
        "--html",
        nargs="?",
        const=_HTML_AUTO,
        default=None,
        help="also write an HTML report with embedded images; bare flag auto-names "
        "into data/reports/, or give an explicit path",
    )
    parser.add_argument(
        "--artifact",
        nargs="?",
        const=_ARTIFACT_AUTO,
        default=None,
        help="freeze the calibration artifact (V3): a calibration.yaml + companion "
        ".gallery.npy decide consumes; requires --policy. Bare flag auto-names into "
        "data/artifacts/, or give an explicit .yaml path",
    )
    # V2 threshold pick -- without --policy no cutoff is chosen (V1 behaviour).
    parser.add_argument(
        "--policy",
        choices=PICK_POLICIES,
        default=None,
        help="automated threshold-picking policy (V2); without it no cutoff is chosen",
    )
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.05,
        help="false-positive budget for --policy target-fpr (default: 0.05)",
    )
    parser.add_argument(
        "--target-fpr-group",
        choices=TARGET_GROUPS,
        default="look-alike",
        help="which negatives the target-fpr budget applies to (default: look-alike)",
    )
    return parser


def resolve_params(args: argparse.Namespace) -> tuple[GenerationParams, bool]:
    """Build params from parsed args; return ``(params, random_seed_drawn)``."""
    if args.random_seed:
        seed = random.SystemRandom().randrange(2**31)
        drawn = True
    else:
        seed = DEFAULT_SEED if args.seed is None else args.seed
        drawn = False
    params = GenerationParams(
        strategy=STRATEGY_THREE_WAY,
        seed=seed,
        gallery=DEFAULT_GALLERY if args.gallery is None else args.gallery,
        calibration=DEFAULT_CALIBRATION
        if args.calibration is None
        else args.calibration,
        test=DEFAULT_TEST if args.test is None else args.test,
        oxford_test_fraction=(
            DEFAULT_OXFORD_TEST_FRACTION
            if args.oxford_test_fraction is None
            else args.oxford_test_fraction
        ),
        prefer=args.prefer,
    )
    return params, drawn


def default_manifest_name(params: GenerationParams) -> str:
    """A deterministic, descriptive filename for an auto-saved manifest."""
    return (
        f"{params.strategy}-seed{params.seed}-g{params.gallery}"
        f"-c{params.calibration}-t{params.test}.yaml"
    )


def default_report_name(params: GenerationParams, aggregation: Aggregation) -> str:
    """A deterministic filename for an auto-saved HTML report.

    Parallels :func:`default_manifest_name` but also encodes the ``aggregation``,
    since that changes the report's scores.
    """
    return (
        f"report-{params.strategy}-seed{params.seed}-g{params.gallery}"
        f"-c{params.calibration}-t{params.test}-{aggregation}.html"
    )


def default_artifact_name(params: GenerationParams, policy: str) -> str:
    """A deterministic filename for an auto-saved calibration artifact.

    The operative aggregation is auto-selected (the winner of the max-vs-mean-top3
    run), so it is *not* in the name -- it is unknown until the run completes.
    """
    return (
        f"calibration-{params.strategy}-seed{params.seed}-g{params.gallery}"
        f"-c{params.calibration}-t{params.test}-{policy}.yaml"
    )


def summarize(manifest: SplitManifest) -> str:
    """A textual, screen-reader-friendly summary of a manifest's roles."""
    lines = [
        f"strategy: {manifest.params.strategy}   seed: {manifest.params.seed}"
        + ("   (drawn)" if manifest.random_seed_drawn else ""),
        f"Indy:   gallery {len(manifest.indy_gallery)}, "
        f"calibration {len(manifest.indy_calibration)}, test {len(manifest.indy_test)}",
        f"Oxford: setup {len(manifest.oxford_setup)} "
        f"({len(manifest.oxford_setup_breed_counts)} breeds), "
        f"test {len(manifest.oxford_test)} "
        f"({len(manifest.oxford_test_breed_counts)} breeds)",
    ]
    return "\n".join(lines)


def run_calibration(
    manifest: SplitManifest,
    label: str,
    aggregation: Aggregation,
    scores_out: Path | None,
    html_out: Path | None,
    sweep_step: float,
    policy: PickPolicy | None = None,
    target_fpr: float = 0.05,
    target_group: TargetGroup = "look-alike",
    artifact_out: Path | None = None,
) -> None:
    """Score the positives/negatives against the gallery and print the report.

    The V0 measurement step: build the gallery from the ``gallery`` role, score
    the held-back ``calibration`` positives and the Oxford ``setup`` negatives
    against it, and emit the textual distribution report. When ``policy`` is set
    (V2), an explicit threshold is picked and reported too. When ``artifact_out``
    is set (V3, requires ``policy``), the frozen calibration artifact is written.
    The ``test`` role is never read here -- that is ``evaluate.py``'s job.
    """
    indy_names, indy_vectors = load_cached_embeddings(INDY_METADATA, INDY_EMBEDDINGS)
    oxford_names, oxford_vectors = load_cached_embeddings(
        OXFORD_METADATA, OXFORD_EMBEDDINGS
    )
    indy_lookup = build_name_to_vector(indy_names, indy_vectors)
    oxford_lookup = build_name_to_vector(oxford_names, oxford_vectors)
    breeds = {record.source_filename: record.breed for record in load_oxford_metadata()}

    raw_gallery_vectors = select_vectors(manifest.indy_gallery, indy_lookup)
    gallery = Gallery.from_raw(manifest.indy_gallery, raw_gallery_vectors)
    positives = score_role(manifest.indy_calibration, indy_lookup, gallery, aggregation)
    negatives = score_role(
        manifest.oxford_setup, oxford_lookup, gallery, aggregation, breeds=breeds
    )

    choice = (
        pick_threshold(
            positives,
            negatives,
            policy,
            target_fpr=target_fpr,
            target_group=target_group,
        )
        if policy is not None
        else None
    )

    print()
    print(
        build_report(
            label,
            len(gallery.names),
            positives,
            negatives,
            aggregation,
            sweep_step=sweep_step,
            choice=choice,
        )
    )
    if scores_out is not None:
        write_scores_csv(scores_out, positives, negatives)
        print(f"\nPer-image scores written to {scores_out}")
    if html_out is not None:
        write_report_html(
            html_out,
            label,
            manifest.indy_gallery,
            positives,
            negatives,
            aggregation,
            sweep_step=sweep_step,
            choice=choice,
        )
        print(f"\nHTML report written to {html_out}")
    if artifact_out is not None:
        assert policy is not None  # guaranteed by the CLI; an artifact needs a pick
        _write_calibration_artifact(
            manifest,
            label,
            gallery,
            raw_gallery_vectors,
            indy_lookup,
            oxford_lookup,
            breeds,
            artifact_out,
            policy=policy,
            target_fpr=target_fpr,
            target_group=target_group,
            sweep_step=sweep_step,
        )


def _write_calibration_artifact(
    manifest: SplitManifest,
    label: str,
    gallery: Gallery,
    raw_gallery_vectors: NDArray[np.float32],
    indy_lookup: dict[str, NDArray[np.float32]],
    oxford_lookup: dict[str, NDArray[np.float32]],
    breeds: dict[str, str],
    artifact_out: Path,
    *,
    policy: PickPolicy,
    target_fpr: float,
    target_group: TargetGroup,
    sweep_step: float,
) -> None:
    """Score both aggregations, build the artifact, and write the YAML + npy pair.

    The operative aggregation is auto-selected as the FPR-first winner of the
    max-vs-mean-top3 comparison (``calibration.artifact.build_artifact``), so both
    are scored here regardless of the report's ``--aggregation``.
    """
    results: dict[Aggregation, AggregationResult] = {}
    for agg in AGGREGATIONS:
        pos = score_role(manifest.indy_calibration, indy_lookup, gallery, agg)
        neg = score_role(
            manifest.oxford_setup, oxford_lookup, gallery, agg, breeds=breeds
        )
        agg_choice = pick_threshold(
            pos, neg, policy, target_fpr=target_fpr, target_group=target_group
        )
        results[agg] = (pos, neg, agg_choice)

    artifact = build_artifact(
        manifest,
        label,
        manifest.indy_gallery,
        raw_gallery_vectors,
        load_indy_positions(),
        results,
        artifact_out.stem + GALLERY_VECTORS_SUFFIX,
        policy=policy,
        target_fpr=target_fpr,
        target_fpr_group=target_group,
        sweep_step=sweep_step,
    )
    vectors_path = write_artifact(artifact, raw_gallery_vectors, artifact_out)
    print(
        f"\nCalibration artifact written to {artifact_out} "
        f"(+ {vectors_path.name})\n"
        f"  winner: {artifact.winner}   threshold: {artifact.threshold:.4f}   "
        f"recall: {artifact.metrics_at_threshold.recall_indy:.3f}   "
        f"FPR(look-alike): {artifact.metrics_at_threshold.fpr_look_alike:.3f}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.manifest is not None:
        explicit = [
            f"--{flag.replace('_', '-')}"
            for flag in _GENERATION_FLAGS
            if getattr(args, flag) not in (None, False)
        ]
        if explicit:
            parser.error(
                f"--manifest cannot be combined with generation flags: {explicit}"
            )

    if args.artifact is not None and args.policy is None:
        parser.error(
            "--artifact requires --policy (an artifact must freeze a chosen threshold)"
        )

    try:
        if args.manifest is not None:
            manifest = load_manifest(Path(args.manifest))
            label = args.manifest
            print(f"Loaded manifest: {args.manifest}")
            print(summarize(manifest))
        else:
            params, drawn = resolve_params(args)
            indy = load_indy_metadata()
            oxford = load_oxford_metadata()
            manifest = generate_three_way(indy, oxford, params, random_seed_drawn=drawn)
            out_path = (
                Path(args.out)
                if args.out is not None
                else SPLITS_DIR / default_manifest_name(params)
            )
            write_manifest(manifest, out_path)
            label = str(out_path)
            print(f"Wrote manifest: {out_path}")
            print(summarize(manifest))

        if not args.generate_only:
            if args.html is None:
                html_out = None
            elif args.html == _HTML_AUTO:
                html_out = REPORTS_DIR / default_report_name(
                    manifest.params, args.aggregation
                )
            else:
                html_out = Path(args.html)
            if args.artifact is None:
                artifact_out = None
            elif args.artifact == _ARTIFACT_AUTO:
                artifact_out = ARTIFACTS_DIR / default_artifact_name(
                    manifest.params, args.policy
                )
            else:
                artifact_out = Path(args.artifact)
            run_calibration(
                manifest,
                label,
                args.aggregation,
                Path(args.scores_out) if args.scores_out is not None else None,
                html_out,
                args.sweep_step,
                policy=args.policy,
                target_fpr=args.target_fpr,
                target_group=args.target_fpr_group,
                artifact_out=artifact_out,
            )
    except SplitConfigError as err:
        raise SystemExit(str(err)) from err


if __name__ == "__main__":
    main()
