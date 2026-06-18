"""Tests for the split-manifest generator.

All pure and fast -- no GPU, no real (gitignored) data. Synthetic metadata/mapping
CSVs are written into ``tmp_path``; small record lists are built directly for the
pure-logic cases. The headline tests encode the design's three guarantees:
test-first invariance, fail-loud over-ask, and verbatim (never recomputed) load.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from split_manifest import (
    DEFAULT_SEED,
    MANIFEST_FORMAT_VERSION,
    STRATEGY_THREE_WAY,
    GenerationParams,
    IndyRecord,
    OxfordRecord,
    SplitConfigError,
    assert_disjoint,
    generate_three_way,
    load_indy_metadata,
    load_manifest,
    load_oxford_metadata,
    manifest_to_dict,
    select_test_split,
    stratified_oxford_split,
    write_manifest,
)

BASE_COLUMNS = [
    "row",
    "source_filename",
    "detect_used",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "area_fraction",
]


# --------------------------------------------------------------------------- #
# Builders / fixtures
# --------------------------------------------------------------------------- #


def make_indy_records(n: int = 35, n_tail: int = 9) -> list[IndyRecord]:
    """``n`` Indy photos, all head_visible, the first ``n_tail`` tail_visible."""
    return [
        IndyRecord(f"indy_{i:02d}.jpeg", head_visible=True, tail_visible=i < n_tail)
        for i in range(n)
    ]


def make_oxford_records(per_breed: dict[str, int]) -> list[OxfordRecord]:
    """Oxford records from a ``breed -> count`` map; filenames are ``Breed_k.jpg``."""
    records: list[OxfordRecord] = []
    for breed, count in per_breed.items():
        for k in range(1, count + 1):
            records.append(OxfordRecord(f"{breed}_{k}.jpg", breed))
    return records


def baseline_params(**overrides: object) -> GenerationParams:
    defaults: dict[str, object] = {
        "strategy": STRATEGY_THREE_WAY,
        "seed": 42,
        "gallery": 15,
        "calibration": 10,
        "test": 10,
        "oxford_test_fraction": 0.30,
        "prefer": None,
    }
    defaults.update(overrides)
    return GenerationParams(**defaults)  # type: ignore[arg-type]


def write_indy_csvs(
    tmp_path: Path, records: list[IndyRecord], *, drop_from_mapping: str | None = None
) -> tuple[Path, Path]:
    """Write synthetic Indy metadata + mapping CSVs; return their paths."""
    metadata_path = tmp_path / "indy_metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(BASE_COLUMNS)
        for i, r in enumerate(records):
            writer.writerow([i, r.source_filename, True, "0.9", 0, 0, 1, 1, "0.5"])

    mapping_path = tmp_path / "mapping.csv"
    with mapping_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "number",
                "new_filename",
                "original_filename",
                "position",
                "location",
                "view",
                "head_visible",
                "tail_visible",
                "notes",
            ]
        )
        for i, r in enumerate(records):
            if r.source_filename == drop_from_mapping:
                continue
            writer.writerow(
                [
                    f"{i:02d}",
                    r.source_filename,
                    f"orig_{i}.jpg",
                    "lying",
                    "bed",
                    "3q",
                    "yes" if r.head_visible else "no",
                    "yes" if r.tail_visible else "no",
                    "note",
                ]
            )
    return metadata_path, mapping_path


def write_oxford_csv(
    tmp_path: Path, records: list[OxfordRecord], *, include_breed: bool = True
) -> Path:
    """Write a synthetic Oxford metadata CSV; return its path."""
    path = tmp_path / "oxford_metadata.csv"
    columns = [*BASE_COLUMNS, "breed"] if include_breed else BASE_COLUMNS
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for i, r in enumerate(records):
            base = [i, r.source_filename, True, "0.9", 0, 0, 1, 1, "0.5"]
            writer.writerow([*base, r.breed] if include_breed else base)
    return path


# --------------------------------------------------------------------------- #
# Counts
# --------------------------------------------------------------------------- #


def test_three_way_role_counts_exact() -> None:
    manifest = generate_three_way(
        make_indy_records(), make_oxford_records({"A": 10, "B": 10}), baseline_params()
    )
    assert len(manifest.indy_gallery) == 15
    assert len(manifest.indy_calibration) == 10
    assert len(manifest.indy_test) == 10


def test_under_ask_leaves_leftovers_no_error() -> None:
    manifest = generate_three_way(
        make_indy_records(),
        make_oxford_records({"A": 10}),
        baseline_params(gallery=5, calibration=5, test=5),
    )
    used = (
        set(manifest.indy_gallery)
        | set(manifest.indy_calibration)
        | set(manifest.indy_test)
    )
    assert len(used) == 15  # 20 of the 35 left unused, no error


# --------------------------------------------------------------------------- #
# Over-ask: fail loud
# --------------------------------------------------------------------------- #


def test_overask_raises() -> None:
    with pytest.raises(SplitConfigError, match="37"):
        generate_three_way(
            make_indy_records(35),
            make_oxford_records({"A": 10}),
            baseline_params(gallery=15, calibration=10, test=12),
        )


def test_overask_validated_against_actual_rows() -> None:
    # Only 30 embedded -> the baseline 15+10+10=35 must refuse, not truncate.
    with pytest.raises(SplitConfigError, match="only 30"):
        generate_three_way(
            make_indy_records(30),
            make_oxford_records({"A": 10}),
            baseline_params(),
        )


def test_empty_gallery_rejected() -> None:
    with pytest.raises(SplitConfigError, match="gallery"):
        generate_three_way(
            make_indy_records(),
            make_oxford_records({"A": 10}),
            baseline_params(gallery=0),
        )


# --------------------------------------------------------------------------- #
# Test-first invariance (the headline guarantee)
# --------------------------------------------------------------------------- #


def test_test_split_invariant_to_gallery_calibration() -> None:
    indy = make_indy_records()
    oxford = make_oxford_records({"A": 10})
    a = generate_three_way(indy, oxford, baseline_params(gallery=15, calibration=10))
    b = generate_three_way(indy, oxford, baseline_params(gallery=5, calibration=5))
    assert set(a.indy_test) == set(b.indy_test)


def test_changing_seed_changes_test() -> None:
    indy = make_indy_records()
    oxford = make_oxford_records({"A": 10})
    a = generate_three_way(indy, oxford, baseline_params(seed=42))
    b = generate_three_way(indy, oxford, baseline_params(seed=43))
    assert set(a.indy_test) != set(b.indy_test)


def test_prefer_does_not_move_test() -> None:
    indy = make_indy_records()
    oxford = make_oxford_records({"A": 10})
    a = generate_three_way(indy, oxford, baseline_params(prefer=None))
    b = generate_three_way(indy, oxford, baseline_params(prefer="tail_visible"))
    assert set(a.indy_test) == set(b.indy_test)


def test_select_test_split_is_pure_function() -> None:
    names = [f"x_{i}" for i in range(20)]
    t1, r1 = select_test_split(names, seed=7, test_count=5)
    t2, r2 = select_test_split(names, seed=7, test_count=5)
    assert t1 == t2 and r1 == r2
    assert len(t1) == 5 and len(r1) == 15
    assert set(t1).isdisjoint(r1)


def test_prefer_biases_gallery_toward_tail_visible() -> None:
    indy = make_indy_records(n=35, n_tail=20)
    oxford = make_oxford_records({"A": 10})
    manifest = generate_three_way(indy, oxford, baseline_params(prefer="tail_visible"))
    tail = {r.source_filename for r in indy if r.tail_visible}
    # Gallery should be skimmed from the tail-visible pool that survived the test draw.
    in_gallery = sum(name in tail for name in manifest.indy_gallery)
    assert in_gallery > len(manifest.indy_gallery) // 2


# --------------------------------------------------------------------------- #
# Disjointness
# --------------------------------------------------------------------------- #


def test_generated_roles_are_disjoint() -> None:
    manifest = generate_three_way(
        make_indy_records(), make_oxford_records({"A": 10, "B": 10}), baseline_params()
    )
    assert_disjoint(manifest.indy_role_lists(), "Indy")
    assert_disjoint(manifest.oxford_role_lists(), "Oxford")


def test_assert_disjoint_detects_overlap() -> None:
    with pytest.raises(SplitConfigError, match="share filenames"):
        assert_disjoint({"a": ["x", "y"], "b": ["y", "z"]}, "Indy")


def test_assert_disjoint_detects_intra_role_duplicate() -> None:
    with pytest.raises(SplitConfigError, match="duplicate"):
        assert_disjoint({"a": ["x", "x"]}, "Indy")


# --------------------------------------------------------------------------- #
# Round-trip / verbatim load
# --------------------------------------------------------------------------- #


def test_write_then_load_roundtrip(tmp_path: Path) -> None:
    manifest = generate_three_way(
        make_indy_records(), make_oxford_records({"A": 10, "B": 10}), baseline_params()
    )
    path = tmp_path / "m.yaml"
    write_manifest(manifest, path)
    loaded = load_manifest(path)
    assert loaded == manifest


def test_yaml_is_ordered_and_well_formed(tmp_path: Path) -> None:
    manifest = generate_three_way(
        make_indy_records(), make_oxford_records({"A": 10}), baseline_params()
    )
    path = tmp_path / "m.yaml"
    write_manifest(manifest, path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert list(raw.keys())[:4] == [
        "format_version",
        "strategy",
        "generated_at",
        "random_seed_drawn",
    ]
    assert isinstance(raw["indy"]["gallery"], list)
    assert all(isinstance(name, str) for name in raw["indy"]["gallery"])


def test_load_uses_body_verbatim_not_recomputed(tmp_path: Path) -> None:
    manifest = generate_three_way(
        make_indy_records(), make_oxford_records({"A": 10}), baseline_params()
    )
    data = manifest_to_dict(manifest)
    # Replace the Indy body with a custom disjoint membership; load must honor it.
    data["indy"]["gallery"] = ["indy_30.jpeg", "indy_31.jpeg"]
    data["indy"]["calibration"] = ["indy_32.jpeg"]
    data["indy"]["test"] = ["indy_33.jpeg", "indy_34.jpeg"]
    path = tmp_path / "edited.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    loaded = load_manifest(path)
    assert loaded.indy_gallery == ["indy_30.jpeg", "indy_31.jpeg"]
    assert loaded.indy_test == ["indy_33.jpeg", "indy_34.jpeg"]


def test_load_rejects_overlapping_roles(tmp_path: Path) -> None:
    manifest = generate_three_way(
        make_indy_records(), make_oxford_records({"A": 10}), baseline_params()
    )
    data = manifest_to_dict(manifest)
    data["indy"]["test"] = [data["indy"]["gallery"][0]]  # force an overlap
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(SplitConfigError, match="share filenames"):
        load_manifest(path)


def test_load_rejects_wrong_format_version(tmp_path: Path) -> None:
    manifest = generate_three_way(
        make_indy_records(), make_oxford_records({"A": 10}), baseline_params()
    )
    data = manifest_to_dict(manifest)
    data["format_version"] = MANIFEST_FORMAT_VERSION + 1
    path = tmp_path / "v.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(SplitConfigError, match="format_version"):
        load_manifest(path)


def test_load_rejects_tampered_breed_summary(tmp_path: Path) -> None:
    manifest = generate_three_way(
        make_indy_records(), make_oxford_records({"A": 10, "B": 10}), baseline_params()
    )
    data = manifest_to_dict(manifest)
    # Bump a stored count so the summary disagrees with the body.
    breed = next(iter(data["oxford_breed_summary"]["setup"]))
    data["oxford_breed_summary"]["setup"][breed] += 5
    path = tmp_path / "drift.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(SplitConfigError, match="oxford_breed_summary"):
        load_manifest(path)


# --------------------------------------------------------------------------- #
# Breed stratification
# --------------------------------------------------------------------------- #


def test_oxford_each_breed_in_both_roles() -> None:
    oxford = make_oxford_records({"A": 10, "B": 10, "C": 10, "D": 10})
    setup, test = stratified_oxford_split(oxford, seed=42, test_fraction=0.30)
    setup_breeds = {n.rsplit("_", 1)[0] for n in setup}
    test_breeds = {n.rsplit("_", 1)[0] for n in test}
    assert setup_breeds == {"A", "B", "C", "D"}
    assert test_breeds == {"A", "B", "C", "D"}


def test_oxford_proportions_exact_round() -> None:
    oxford = make_oxford_records({"A": 10, "B": 10})
    setup, test = stratified_oxford_split(oxford, seed=42, test_fraction=0.30)
    assert len(test) == 6  # round(10*0.3)=3 per breed
    assert len(setup) == 14


def test_oxford_small_breed_not_dropped_from_setup() -> None:
    oxford = make_oxford_records({"A": 10, "Tiny": 3})
    setup, test = stratified_oxford_split(oxford, seed=42, test_fraction=0.30)
    # round(3*0.3)=1 test, 2 setup -> Tiny present in both roles.
    assert sum(n.startswith("Tiny_") for n in setup) == 2
    assert sum(n.startswith("Tiny_") for n in test) == 1


def test_oxford_split_is_disjoint_and_complete() -> None:
    oxford = make_oxford_records({"A": 10, "B": 7})
    setup, test = stratified_oxford_split(oxford, seed=1, test_fraction=0.30)
    assert set(setup).isdisjoint(test)
    assert set(setup) | set(test) == {r.source_filename for r in oxford}


def test_oxford_fraction_out_of_range_raises() -> None:
    with pytest.raises(SplitConfigError, match="oxford-test-fraction"):
        stratified_oxford_split(
            make_oxford_records({"A": 10}), seed=1, test_fraction=1.5
        )


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_default_seed_is_bit_for_bit_repeatable() -> None:
    indy = make_indy_records()
    oxford = make_oxford_records({"A": 10, "B": 10})
    params = baseline_params(seed=DEFAULT_SEED)
    a = generate_three_way(indy, oxford, params)
    b = generate_three_way(indy, oxford, params)
    assert a.indy_role_lists() == b.indy_role_lists()
    assert a.oxford_role_lists() == b.oxford_role_lists()


def test_same_seed_same_oxford_split() -> None:
    oxford = make_oxford_records({"A": 10, "B": 10})
    assert stratified_oxford_split(oxford, 5, 0.30) == stratified_oxford_split(
        oxford, 5, 0.30
    )


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def test_load_indy_joins_mapping_flags(tmp_path: Path) -> None:
    records = make_indy_records(n=3, n_tail=1)
    metadata_path, mapping_path = write_indy_csvs(tmp_path, records)
    loaded = load_indy_metadata(metadata_path, mapping_path)
    assert [r.source_filename for r in loaded] == [r.source_filename for r in records]
    assert loaded[0].tail_visible is True
    assert loaded[1].tail_visible is False
    assert all(r.head_visible for r in loaded)


def test_load_indy_missing_mapping_row_raises(tmp_path: Path) -> None:
    records = make_indy_records(n=3)
    metadata_path, mapping_path = write_indy_csvs(
        tmp_path, records, drop_from_mapping="indy_01.jpeg"
    )
    with pytest.raises(SplitConfigError, match="missing from"):
        load_indy_metadata(metadata_path, mapping_path)


def test_load_oxford_reads_breed(tmp_path: Path) -> None:
    records = make_oxford_records({"Persian": 2, "Ragdoll": 2})
    path = write_oxford_csv(tmp_path, records)
    loaded = load_oxford_metadata(path)
    assert {r.breed for r in loaded} == {"Persian", "Ragdoll"}


def test_load_oxford_missing_breed_column_raises(tmp_path: Path) -> None:
    records = make_oxford_records({"Persian": 2})
    path = write_oxford_csv(tmp_path, records, include_breed=False)
    with pytest.raises(SplitConfigError, match="breed"):
        load_oxford_metadata(path)
