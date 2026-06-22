"""Generate and load the split manifest for a calibration experiment.

A **split manifest** assigns every embedded image to a role for one reproducible
experiment of the decide stage (`image -> detect -> crop -> embed -> decide`):

    Indy    gallery / calibration / test
    Oxford  setup / test

It is the unit of an experiment. This module is the reusable generator the
``calibration.cli`` measurement step and a future ``evaluate.py`` import; it owns
no scoring or threshold logic (that is the decide stage). The CLI lives in
``calibration.cli`` (with the ``scripts/calibrate.py`` shim as its entry point).

Three design guarantees (see ``docs/calibration_design.md`` Sec. 3) are encoded
here, all guarding against silently-wrong numbers:

* **Test split first.** ``select_test_split`` draws the test set as a pure
  function of ``(filenames, seed, test_count)`` -- gallery/calibration counts are
  never passed in. So holding ``seed`` + ``test`` fixed and varying the setup
  split yields an *identical* exam; changing ``seed`` draws a new exam knowingly.
* **Counts validated against what is actually embedded.** Requested Indy role
  counts are checked against the rows present in ``metadata.csv`` (not an assumed
  35); an over-ask is a hard error, never a silently-truncated split.
* **Generate dynamically, materialize the result.** ``seed`` + counts +
  breed-stratify logic decide membership; the *resolved* filename lists are then
  written into the YAML body and used **verbatim** at load -- ``load_manifest``
  never recomputes from the seed. This keeps a frozen exam stable even if
  ``metadata.csv`` is re-embedded underneath it.
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from _common import EmbeddingsVariant

# This file is scripts/calibration/manifest.py, so the repo root is three levels up.
REPO_ROOT = Path(__file__).parent.parent.parent

#: Root under which every dataset's per-variant embeddings cache lives:
#: ``data/embeddings/<dataset>/<model_slug>/<crop_slug>/``. The flat
#: ``indy/metadata.csv`` layout is gone -- a cache is now identified by its model
#: and crop variant, never by a fixed path (see ``docs/embeddings_provenance.md``).
EMBEDDINGS_ROOT = REPO_ROOT / "data" / "embeddings"
INDY_MAPPING = REPO_ROOT / "images" / "indy" / "mapping.csv"
SPLITS_DIR = REPO_ROOT / "data" / "splits"

#: Baseline embedding variant: the no-arg calibrate run resolves to
#: ``facebook--dinov2-base/crop-m0.1`` under each dataset root.
DEFAULT_MODEL = "facebook/dinov2-base"
DEFAULT_DETECT = True
DEFAULT_MARGIN = 0.1


def indy_variant_dir(variant: EmbeddingsVariant) -> Path:
    """The Indy cache dir for ``variant`` under ``EMBEDDINGS_ROOT/indy``."""
    return variant.dir(EMBEDDINGS_ROOT / "indy")


def oxford_variant_dir(variant: EmbeddingsVariant) -> Path:
    """The Oxford cache dir for ``variant`` under ``EMBEDDINGS_ROOT/oxford``."""
    return variant.dir(EMBEDDINGS_ROOT / "oxford")


def catbreeds_variant_dir(variant: EmbeddingsVariant) -> Path:
    """The cat-breeds cache dir for ``variant`` under ``EMBEDDINGS_ROOT/catbreeds``."""
    return variant.dir(EMBEDDINGS_ROOT / "catbreeds")


#: Source-image directories, keyed to the ``source_filename`` columns: Indy's
#: ``metadata.csv`` names live directly in ``images/indy/`` and Oxford's under
#: ``images/oxford-iiit-pet/images/``. The HTML report embeds these full images.
INDY_IMAGE_DIR = REPO_ROOT / "images" / "indy"
OXFORD_IMAGE_DIR = REPO_ROOT / "images" / "oxford-iiit-pet" / "images"

#: Default output location for the optional HTML calibration report (gitignored,
#: like ``SPLITS_DIR``).
REPORTS_DIR = REPO_ROOT / "data" / "reports"

#: Default output location for the frozen calibration artifact pair (V3): the
#: ``calibration.yaml`` + companion ``.gallery.npy``. Gitignored, like the others.
ARTIFACTS_DIR = REPO_ROOT / "data" / "artifacts"

#: Built-in seed so a zero-arg baseline run is bit-for-bit repeatable.
DEFAULT_SEED = 20240601
DEFAULT_GALLERY = 15
DEFAULT_CALIBRATION = 10
DEFAULT_TEST = 10
DEFAULT_OXFORD_TEST_FRACTION = 0.30

MANIFEST_FORMAT_VERSION = 2
STRATEGY_THREE_WAY = "three_way"

#: Recognised ``prefer`` knobs: bias the gallery toward photos whose head or tail
#: is visible (the most identifying features). Off for the baseline.
PREFER_CHOICES = ("head_visible", "tail_visible")


class SplitConfigError(ValueError):
    """A requested or loaded split is impossible/invalid against the data."""


@dataclass(frozen=True)
class IndyRecord:
    """One embedded Indy photo plus the mapping attributes ``prefer`` uses."""

    source_filename: str
    head_visible: bool
    tail_visible: bool


@dataclass(frozen=True)
class OxfordRecord:
    """One embedded Oxford cat: filename and breed label (for stratification)."""

    source_filename: str
    breed: str


@dataclass(frozen=True)
class EmbeddingProvenance:
    """The embedding variant the manifest's frozen lists were drawn against.

    Recorded in the manifest header so a replay can be *cross-checked* against the
    caches it is scored over, not merely trusted by filename (invariant #1 of
    ``docs/embeddings_provenance.md``). ``margin``/``min_confidence`` are ``None``
    when ``detect`` is off, matching :class:`_common.EmbeddingsMeta`'s normalized
    :meth:`variant_key`. It is set at generation from the loaded sidecars' shared
    variant -- ``manifest.py`` owns no I/O, so the caller (calibrate) supplies it.
    """

    model_id: str
    detect: bool
    margin: float | None
    min_confidence: float | None

    def variant_key(self) -> tuple[str, bool, float | None, float | None]:
        """Normalized identity matching :meth:`EmbeddingsMeta.variant_key`.

        ``(model_id, detect, margin, min_confidence)`` with ``margin`` and
        ``min_confidence`` forced to ``None`` when ``detect`` is false, so a
        manifest header compares equal to the sidecars it was drawn against.
        """
        if not self.detect:
            return (self.model_id, self.detect, None, None)
        return (self.model_id, self.detect, self.margin, self.min_confidence)


@dataclass(frozen=True)
class GenerationParams:
    """The knobs that decide membership; recorded verbatim in the manifest header.

    ``seed`` is the *effective* seed actually used -- the drawn value when the CLI
    was invoked with ``--random-seed`` -- so a run is always reproducible after.
    """

    strategy: str
    seed: int
    gallery: int
    calibration: int
    test: int
    oxford_test_fraction: float
    prefer: str | None


@dataclass(frozen=True)
class SplitManifest:
    """A materialized split: provenance header plus the frozen role lists.

    The role lists are ``source_filename`` strings used **verbatim** -- loading a
    manifest never recomputes membership from the seed.
    """

    format_version: int
    params: GenerationParams
    embedding: EmbeddingProvenance
    generated_at: str
    random_seed_drawn: bool
    indy_gallery: list[str]
    indy_calibration: list[str]
    indy_test: list[str]
    oxford_setup: list[str]
    oxford_test: list[str]
    oxford_setup_breed_counts: dict[str, int]
    oxford_test_breed_counts: dict[str, int]

    def indy_role_lists(self) -> dict[str, list[str]]:
        """The three Indy roles keyed by name, for disjointness checks."""
        return {
            "gallery": self.indy_gallery,
            "calibration": self.indy_calibration,
            "test": self.indy_test,
        }

    def oxford_role_lists(self) -> dict[str, list[str]]:
        """The two Oxford roles keyed by name, for disjointness checks."""
        return {"setup": self.oxford_setup, "test": self.oxford_test}


# --------------------------------------------------------------------------- #
# Loading metadata / mapping
# --------------------------------------------------------------------------- #


def _parse_yes_no(value: str) -> bool:
    """Parse a ``yes``/``no`` mapping cell into a bool (case-insensitive)."""
    return value.strip().lower() == "yes"


def load_indy_metadata(
    metadata_path: Path,
    mapping_path: Path = INDY_MAPPING,
) -> list[IndyRecord]:
    """Load embedded Indy photos, joined to ``mapping.csv`` for head/tail flags.

    The metadata rows are the ground truth of what is embedded (and ``.npy``-row
    aligned); ``mapping.csv`` supplies ``head_visible``/``tail_visible``. A
    metadata filename absent from the mapping is a hard error -- the join must be
    total, and a gap is a data-integrity problem worth surfacing.
    """
    attributes: dict[str, tuple[bool, bool]] = {}
    with mapping_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            attributes[row["new_filename"]] = (
                _parse_yes_no(row["head_visible"]),
                _parse_yes_no(row["tail_visible"]),
            )

    records: list[IndyRecord] = []
    with metadata_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            name = row["source_filename"]
            if name not in attributes:
                raise SplitConfigError(
                    f"{name} is embedded in {metadata_path} but missing from "
                    f"{mapping_path}; cannot join its head/tail attributes"
                )
            head, tail = attributes[name]
            records.append(IndyRecord(name, head, tail))
    return records


def load_oxford_metadata(
    metadata_path: Path,
) -> list[OxfordRecord]:
    """Load embedded Oxford cats with breed labels (the negative pool)."""
    records: list[OxfordRecord] = []
    with metadata_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "breed" not in reader.fieldnames:
            raise SplitConfigError(
                f"{metadata_path} has no 'breed' column; not an Oxford metadata file"
            )
        for row in reader:
            records.append(OxfordRecord(row["source_filename"], row["breed"]))
    return records


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def validate_indy_counts(indy: list[IndyRecord], params: GenerationParams) -> None:
    """Raise ``SplitConfigError`` if the requested Indy split cannot be drawn.

    Over-asking is a hard error (never a silent truncation), checked against the
    rows *actually* embedded -- so if a photo ever fails to embed, the otherwise
    no-margin ``15 + 10 + 10`` baseline refuses to run rather than quietly
    shrinking.
    """
    for name, value in (
        ("gallery", params.gallery),
        ("calibration", params.calibration),
        ("test", params.test),
    ):
        if value < 0:
            raise SplitConfigError(f"--{name} cannot be negative (got {value})")
    if params.gallery < 1:
        raise SplitConfigError(
            "--gallery must be at least 1 (a gallery cannot be empty)"
        )

    requested = params.gallery + params.calibration + params.test
    available = len(indy)
    if requested > available:
        raise SplitConfigError(
            f"requested {params.gallery}+{params.calibration}+{params.test}="
            f"{requested} Indy photos but only {available} are embedded in "
            f"metadata.csv (over-ask by {requested - available})"
        )


def _seeded_shuffle(filenames: list[str], seed: int) -> list[str]:
    """A reproducible permutation of ``filenames`` driven only by ``seed``."""
    shuffled = list(filenames)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def select_test_split(
    filenames: list[str], seed: int, test_count: int
) -> tuple[list[str], list[str]]:
    """Draw the test set first: ``(test, remaining)`` from one seeded shuffle.

    A pure function of ``(filenames, seed, test_count)`` -- gallery/calibration
    counts never enter. ``test`` is the first ``test_count`` of the shuffle;
    ``remaining`` is the rest, in shuffled order. ``filenames`` must already be in
    a canonical order (metadata row order) so the same seed gives the same draw.
    """
    shuffled = _seeded_shuffle(filenames, seed)
    return shuffled[:test_count], shuffled[test_count:]


def _apply_prefer(
    indy: list[IndyRecord], remaining: list[str], prefer: str | None
) -> list[str]:
    """Stable-sort the post-test pool so ``prefer`` photos sort first.

    A stable sort keeps the seeded order within the preferred and non-preferred
    groups, so the gallery skims the preferred photos without disturbing the
    (already-drawn) test set or determinism. ``None`` is the identity (baseline).
    """
    if prefer is None:
        return remaining
    flags = {r.source_filename: getattr(r, prefer) for r in indy}
    return sorted(remaining, key=lambda name: not flags[name])


def stratified_oxford_split(
    records: list[OxfordRecord], seed: int, test_fraction: float
) -> tuple[list[str], list[str]]:
    """Split Oxford into ``(setup, test)`` stratified by breed.

    Each breed is split independently so every role gets a representative breed
    mix and the look-alike tail is never lopsided. Breeds are processed in sorted
    order with per-breed sub-seeds drawn from one ``Random(seed)`` (``hash()`` is
    avoided -- it is per-process salted). Within a breed, ``round(n * fraction)``
    images go to test, the rest to setup.
    """
    if not 0.0 <= test_fraction <= 1.0:
        raise SplitConfigError(
            f"--oxford-test-fraction must be in [0, 1] (got {test_fraction})"
        )
    by_breed: dict[str, list[str]] = {}
    for record in records:
        by_breed.setdefault(record.breed, []).append(record.source_filename)

    rng = random.Random(seed)
    setup: list[str] = []
    test: list[str] = []
    for breed in sorted(by_breed):
        sub_seed = rng.randrange(2**31)
        shuffled = _seeded_shuffle(by_breed[breed], sub_seed)
        n_test = round(len(shuffled) * test_fraction)
        test.extend(shuffled[:n_test])
        setup.extend(shuffled[n_test:])
    return setup, test


def _breed_counts(filenames: list[str], breeds: dict[str, str]) -> dict[str, int]:
    """Per-breed counts for a list of Oxford filenames, breed-sorted."""
    counts = Counter(breeds[name] for name in filenames)
    return {breed: counts[breed] for breed in sorted(counts)}


def generate_three_way(
    indy: list[IndyRecord],
    oxford: list[OxfordRecord],
    params: GenerationParams,
    embedding: EmbeddingProvenance,
    random_seed_drawn: bool = False,
) -> SplitManifest:
    """Generate a materialized ``three_way`` manifest from records + params.

    Order matters: the test set is drawn *before* gallery/calibration are sliced,
    so they cannot perturb the exam. Oxford is stratified by breed. ``embedding``
    is the shared variant the caller resolved the records' caches from; it is
    recorded in the header so a later replay can be cross-checked against the
    caches it scores over.
    """
    if params.strategy != STRATEGY_THREE_WAY:
        raise SplitConfigError(
            f"only the {STRATEGY_THREE_WAY!r} strategy is implemented "
            f"(got {params.strategy!r})"
        )
    if params.prefer is not None and params.prefer not in PREFER_CHOICES:
        raise SplitConfigError(
            f"--prefer must be one of {PREFER_CHOICES} (got {params.prefer!r})"
        )
    validate_indy_counts(indy, params)

    indy_names = [r.source_filename for r in indy]
    test, remaining = select_test_split(indy_names, params.seed, params.test)
    remaining = _apply_prefer(indy, remaining, params.prefer)
    gallery = remaining[: params.gallery]
    calibration = remaining[params.gallery : params.gallery + params.calibration]

    setup, oxford_test = stratified_oxford_split(
        oxford, params.seed, params.oxford_test_fraction
    )
    breeds = {r.source_filename: r.breed for r in oxford}

    return SplitManifest(
        format_version=MANIFEST_FORMAT_VERSION,
        params=params,
        embedding=embedding,
        generated_at=datetime.now(UTC).isoformat(),
        random_seed_drawn=random_seed_drawn,
        indy_gallery=gallery,
        indy_calibration=calibration,
        indy_test=test,
        oxford_setup=setup,
        oxford_test=oxford_test,
        oxford_setup_breed_counts=_breed_counts(setup, breeds),
        oxford_test_breed_counts=_breed_counts(oxford_test, breeds),
    )


# --------------------------------------------------------------------------- #
# YAML write / load
# --------------------------------------------------------------------------- #


def manifest_to_dict(manifest: SplitManifest) -> dict[str, Any]:
    """Render a manifest as the ordered, YAML-serialisable mapping (header + body)."""
    return {
        "format_version": manifest.format_version,
        "strategy": manifest.params.strategy,
        "generated_at": manifest.generated_at,
        "random_seed_drawn": manifest.random_seed_drawn,
        "params": {
            "seed": manifest.params.seed,
            "gallery": manifest.params.gallery,
            "calibration": manifest.params.calibration,
            "test": manifest.params.test,
            "oxford_test_fraction": manifest.params.oxford_test_fraction,
            "prefer": manifest.params.prefer,
        },
        "embedding": {
            "model_id": manifest.embedding.model_id,
            "detect": manifest.embedding.detect,
            "margin": manifest.embedding.margin,
            "min_confidence": manifest.embedding.min_confidence,
        },
        "oxford_breed_summary": {
            "setup": manifest.oxford_setup_breed_counts,
            "test": manifest.oxford_test_breed_counts,
        },
        "indy": {
            "gallery": manifest.indy_gallery,
            "calibration": manifest.indy_calibration,
            "test": manifest.indy_test,
        },
        "oxford": {
            "setup": manifest.oxford_setup,
            "test": manifest.oxford_test,
        },
    }


def write_manifest(manifest: SplitManifest, path: Path) -> None:
    """Write a manifest to ``path`` as YAML (header/provenance + body lists)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            manifest_to_dict(manifest),
            f,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )


def manifest_from_dict(data: dict[str, Any]) -> SplitManifest:
    """Reconstruct a manifest from a parsed YAML mapping (no validation)."""
    params_data = data["params"]
    params = GenerationParams(
        strategy=data["strategy"],
        seed=params_data["seed"],
        gallery=params_data["gallery"],
        calibration=params_data["calibration"],
        test=params_data["test"],
        oxford_test_fraction=params_data["oxford_test_fraction"],
        prefer=params_data["prefer"],
    )
    embedding_data = data["embedding"]
    embedding = EmbeddingProvenance(
        model_id=embedding_data["model_id"],
        detect=embedding_data["detect"],
        margin=embedding_data["margin"],
        min_confidence=embedding_data["min_confidence"],
    )
    summary = data["oxford_breed_summary"]
    return SplitManifest(
        format_version=data["format_version"],
        params=params,
        embedding=embedding,
        generated_at=data["generated_at"],
        random_seed_drawn=data["random_seed_drawn"],
        indy_gallery=list(data["indy"]["gallery"]),
        indy_calibration=list(data["indy"]["calibration"]),
        indy_test=list(data["indy"]["test"]),
        oxford_setup=list(data["oxford"]["setup"]),
        oxford_test=list(data["oxford"]["test"]),
        oxford_setup_breed_counts=dict(summary["setup"]),
        oxford_test_breed_counts=dict(summary["test"]),
    )


def assert_disjoint(named_lists: dict[str, list[str]], group: str) -> None:
    """Raise ``SplitConfigError`` if roles in ``group`` overlap or self-duplicate."""
    for role, names in named_lists.items():
        if len(names) != len(set(names)):
            dups = sorted({n for n in names if names.count(n) > 1})
            raise SplitConfigError(
                f"{group} role {role!r} contains duplicate filenames: {dups}"
            )
    roles = list(named_lists)
    for i, left in enumerate(roles):
        for right in roles[i + 1 :]:
            overlap = set(named_lists[left]) & set(named_lists[right])
            if overlap:
                raise SplitConfigError(
                    f"{group} roles {left!r} and {right!r} share filenames: "
                    f"{sorted(overlap)}"
                )


def load_manifest(path: Path) -> SplitManifest:
    """Load and validate a manifest from YAML; use its frozen lists verbatim.

    Validation is loud: a wrong ``format_version``, any overlap between role sets,
    or an ``oxford_breed_summary`` that disagrees with the materialized body (a
    hand-edit drift) all raise ``SplitConfigError``. Membership is never recomputed
    from the seed -- the body is authoritative.
    """
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    manifest = manifest_from_dict(data)

    if manifest.format_version != MANIFEST_FORMAT_VERSION:
        raise SplitConfigError(
            f"{path} has format_version {manifest.format_version}; "
            f"this tool writes/reads version {MANIFEST_FORMAT_VERSION}"
        )
    assert_disjoint(manifest.indy_role_lists(), "Indy")
    assert_disjoint(manifest.oxford_role_lists(), "Oxford")

    for role, names, stored in (
        ("setup", manifest.oxford_setup, manifest.oxford_setup_breed_counts),
        ("test", manifest.oxford_test, manifest.oxford_test_breed_counts),
    ):
        derived = dict(Counter(name.rsplit("_", 1)[0] for name in names))
        if derived != {k: v for k, v in stored.items()}:
            raise SplitConfigError(
                f"{path}: oxford_breed_summary[{role!r}] disagrees with the "
                f"materialized body; the manifest has been hand-edited inconsistently"
            )
    return manifest
