"""Calibrate the decide stage: split, score, and report the distributions (V0).

This is the V0 slice of ``docs/calibration_design.md`` Sec. 5 -- *distributions
only, no threshold chosen.* The command generates (or replays) a reproducible
split manifest, then scores the held-back Indy positives and the Oxford
negatives against the gallery and prints the textual distribution report. It
answers the first question worth answering -- do the positive and negative score
distributions separate at all? -- without picking any cutoff (that is V1/V2).

Generation is folded into this one command (no separate generate/calibrate
dance). The split logic lives in the reusable ``split_manifest`` module; the
scoring/report logic in ``calibration_report`` over the ``indycat.decision``
core. ``--generate-only`` stops after writing the manifest.

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

    # fresh random seed (the drawn seed is recorded in the written manifest)
    uv run python scripts/calibrate.py --random-seed

    # replay an exact prior split (scores it too)
    uv run python scripts/calibrate.py --manifest data/splits/run-<...>.yaml
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from _common import load_cached_embeddings
from calibration_report import (
    build_name_to_vector,
    build_report,
    score_role,
    select_vectors,
    write_scores_csv,
)
from indycat.decision import AGGREGATIONS, Aggregation, Gallery
from split_manifest import (
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
) -> None:
    """Score the positives/negatives against the gallery and print the report.

    The V0 measurement step: build the gallery from the ``gallery`` role, score
    the held-back ``calibration`` positives and the Oxford ``setup`` negatives
    against it, and emit the textual distribution report. The ``test`` role is
    never read here -- that is ``evaluate.py``'s job.
    """
    indy_names, indy_vectors = load_cached_embeddings(INDY_METADATA, INDY_EMBEDDINGS)
    oxford_names, oxford_vectors = load_cached_embeddings(
        OXFORD_METADATA, OXFORD_EMBEDDINGS
    )
    indy_lookup = build_name_to_vector(indy_names, indy_vectors)
    oxford_lookup = build_name_to_vector(oxford_names, oxford_vectors)
    breeds = {record.source_filename: record.breed for record in load_oxford_metadata()}

    gallery = Gallery.from_raw(
        manifest.indy_gallery, select_vectors(manifest.indy_gallery, indy_lookup)
    )
    positives = score_role(manifest.indy_calibration, indy_lookup, gallery, aggregation)
    negatives = score_role(
        manifest.oxford_setup, oxford_lookup, gallery, aggregation, breeds=breeds
    )

    print()
    print(build_report(label, len(gallery.names), positives, negatives, aggregation))
    if scores_out is not None:
        write_scores_csv(scores_out, positives, negatives)
        print(f"\nPer-image scores written to {scores_out}")


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
            run_calibration(
                manifest,
                label,
                args.aggregation,
                Path(args.scores_out) if args.scores_out is not None else None,
            )
    except SplitConfigError as err:
        raise SystemExit(str(err)) from err


if __name__ == "__main__":
    main()
