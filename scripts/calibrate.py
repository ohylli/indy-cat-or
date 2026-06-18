"""Calibrate the decide stage -- currently the split-manifest entry point.

The calibration step (scoring the held-back Indy positives and the Oxford
negatives against the gallery to locate the threshold) is not implemented yet --
this is the V0 generation slice of ``docs/calibration_design.md``. For now the
tool generates and saves a reproducible split manifest; the scoring step will be
added where the placeholder notice prints.

Generation is folded into this one command (no separate generate/calibrate
dance). The split logic itself lives in the reusable ``split_manifest`` module.

Usage::

    # zero-arg baseline: built-in defaults + built-in seed -> identical every time
    uv run python scripts/calibrate.py

    # generate only, then stop (stays useful once calibration exists)
    uv run python scripts/calibrate.py --generate-only

    # specify the split
    uv run python scripts/calibrate.py --gallery 15 --calibration 10 --test 10 --seed 42

    # fresh random seed (the drawn seed is recorded in the written manifest)
    uv run python scripts/calibrate.py --random-seed

    # replay / validate an exact prior split
    uv run python scripts/calibrate.py --manifest data/splits/run-<...>.yaml
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from split_manifest import (
    DEFAULT_CALIBRATION,
    DEFAULT_GALLERY,
    DEFAULT_OXFORD_TEST_FRACTION,
    DEFAULT_SEED,
    DEFAULT_TEST,
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

_NOT_IMPLEMENTED_NOTICE = (
    "Calibration scoring is not yet implemented -- only the split manifest was "
    "generated. See docs/calibration_design.md (V0)."
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
            print(f"Wrote manifest: {out_path}")
            print(summarize(manifest))
    except SplitConfigError as err:
        raise SystemExit(str(err)) from err

    if not args.generate_only and args.manifest is None:
        print()
        print(_NOT_IMPLEMENTED_NOTICE)


if __name__ == "__main__":
    main()
