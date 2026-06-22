"""Light CLI smoke tests for the calibrate command (``calibration.cli``).

The split logic is covered in ``test_split_manifest`` and the scoring/report in
``test_calibration_report``; here we only check the CLI wiring: mutual exclusion,
that ``--generate-only`` writes a loadable manifest and skips scoring, and that
the default and ``--manifest`` runs proceed to print the V0 distribution report.
The Indy/Oxford metadata loaders *and* the embeddings cache are stubbed via
monkeypatch so the tests never touch the gitignored real data or a GPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _common import EmbeddingsMeta, EmbeddingsVariant, write_embeddings_meta
from calibration import cli as calibrate
from calibration import manifest as manifest_mod
from calibration.artifact import load_artifact
from calibration.manifest import IndyRecord, OxfordRecord, load_manifest

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

#: The baseline variant the no-arg calibrate run selects.
BASELINE = EmbeddingsVariant(model_id="facebook/dinov2-base", detect=True, margin=0.1)
EMBED_DIM = 8

INDY_RECORDS = [
    IndyRecord(f"indy_{i:02d}.jpeg", head_visible=True, tail_visible=i < 9)
    for i in range(35)
]
OXFORD_RECORDS = [
    OxfordRecord(f"{breed}_{k}.jpg", breed)
    for breed in ("Persian", "Ragdoll")
    for k in range(1, 11)
]


def _write_metadata_csv(
    path: Path, names: list[str], *, breeds: list[str] | None
) -> None:
    """Write a minimal embeddings ``metadata.csv`` (loader reads source_filename)."""
    import csv

    columns = [*BASE_COLUMNS, "breed"] if breeds is not None else BASE_COLUMNS
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for i, name in enumerate(names):
            base = [i, name, True, "0.9", 0, 0, 1, 1, "0.5"]
            writer.writerow([*base, breeds[i]] if breeds is not None else base)


def write_variant_cache(
    out_dir: Path,
    names: list[str],
    *,
    breeds: list[str] | None,
    meta: EmbeddingsMeta,
) -> None:
    """Write a full variant cache: metadata.csv + embeddings.npy + the sidecar.

    Vectors are synthetic (seeded by row count) so the scoring path runs without a
    GPU or real weights; ``meta`` is written verbatim so tests can stamp a
    mismatched sidecar.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_metadata_csv(out_dir / "metadata.csv", names, breeds=breeds)
    rng = np.random.default_rng(len(names))
    vectors = rng.standard_normal((len(names), EMBED_DIM)).astype(np.float32)
    np.save(out_dir / "embeddings.npy", vectors)
    write_embeddings_meta(meta, out_dir)


def baseline_meta(row_count: int, **overrides: object) -> EmbeddingsMeta:
    """A sidecar matching the baseline variant, with optional field overrides."""
    defaults: dict[str, object] = {
        "format_version": 1,
        "model_id": "facebook/dinov2-base",
        "embedding_dim": EMBED_DIM,
        "normalized": False,
        "detect": True,
        "margin": 0.1,
        "min_confidence": 0.25,
        "row_count": row_count,
    }
    defaults.update(overrides)
    return EmbeddingsMeta(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def fake_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point calibrate at synthetic baseline variant caches under a tmp root.

    Writes real Indy + Oxford variant dirs (metadata.csv + embeddings.npy +
    sidecar) so the full ``load_embeddings_variant`` + scoring path runs without a
    GPU, real weights, or the gitignored data. The metadata *record* loaders (the
    head/tail join and breed read) are stubbed path-agnostically so the synthetic
    ``indy_NN`` names need no companion mapping.csv. Returns the tmp embeddings
    root so individual tests can overwrite a variant's sidecar to force a mismatch.
    """
    embeddings_root = tmp_path / "embeddings"
    monkeypatch.setattr(manifest_mod, "EMBEDDINGS_ROOT", embeddings_root)

    indy_names = [r.source_filename for r in INDY_RECORDS]
    oxford_names = [r.source_filename for r in OXFORD_RECORDS]
    oxford_breeds = [r.breed for r in OXFORD_RECORDS]

    write_variant_cache(
        BASELINE.dir(embeddings_root / "indy"),
        indy_names,
        breeds=None,
        meta=baseline_meta(len(indy_names)),
    )
    write_variant_cache(
        BASELINE.dir(embeddings_root / "oxford"),
        oxford_names,
        breeds=oxford_breeds,
        meta=baseline_meta(len(oxford_names)),
    )

    monkeypatch.setattr(calibrate, "load_indy_metadata", lambda _path: INDY_RECORDS)
    monkeypatch.setattr(calibrate, "load_oxford_metadata", lambda _path: OXFORD_RECORDS)
    return embeddings_root


def test_manifest_and_generation_flags_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        calibrate.main(["--manifest", "x.yaml", "--seed", "42"])


def test_generate_only_writes_loadable_manifest_and_skips_scoring(
    fake_data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "m.yaml"
    calibrate.main(["--generate-only", "--out", str(out)])
    assert out.exists()
    manifest = load_manifest(out)  # round-trips through validation
    assert len(manifest.indy_gallery) == 15
    assert "Score distribution" not in capsys.readouterr().out  # no scoring


def test_default_run_writes_manifest_and_prints_report(
    fake_data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "m.yaml"
    calibrate.main(["--out", str(out)])
    assert out.exists()
    printed = capsys.readouterr().out
    assert "Score distribution" in printed
    assert "Negatives by group" in printed
    assert "Threshold sweep" in printed  # V1 trade-off curve


def test_sweep_step_controls_grid(
    fake_data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "m.yaml"
    calibrate.main(["--out", str(out), "--sweep-step", "0.1"])
    printed = capsys.readouterr().out
    assert "Threshold sweep" in printed
    assert "Per-breed FPR by cutoff" in printed


def test_scores_out_writes_csv(
    fake_data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "m.yaml"
    scores = tmp_path / "scores.csv"
    calibrate.main(["--out", str(out), "--scores-out", str(scores)])
    assert scores.exists()
    assert "Per-image scores written" in capsys.readouterr().out


def test_no_policy_does_not_choose_a_threshold(
    fake_data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calibrate.main(["--out", str(tmp_path / "m.yaml")])
    assert "Chosen threshold" not in capsys.readouterr().out


def test_policy_target_fpr_prints_chosen_threshold(
    fake_data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calibrate.main(["--out", str(tmp_path / "m.yaml"), "--policy", "target-fpr"])
    printed = capsys.readouterr().out
    assert "Chosen threshold (policy=target-fpr)" in printed
    assert "look-alike" in printed  # default budget group reaches the rationale


def test_policy_youdens_j_prints_chosen_threshold(
    fake_data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calibrate.main(["--out", str(tmp_path / "m.yaml"), "--policy", "youdens-j"])
    assert "Chosen threshold (policy=youdens-j)" in capsys.readouterr().out


def test_target_fpr_group_overall_reaches_the_pick(
    fake_data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calibrate.main(
        [
            "--out",
            str(tmp_path / "m.yaml"),
            "--policy",
            "target-fpr",
            "--target-fpr-group",
            "overall",
            "--target-fpr",
            "0.1",
        ]
    )
    printed = capsys.readouterr().out
    assert "FPR(overall) <= 0.100" in printed


def test_overask_exits_loudly(fake_data: Path, tmp_path: Path) -> None:
    out = tmp_path / "m.yaml"
    with pytest.raises(SystemExit, match="over-ask"):
        calibrate.main(
            [
                "--gallery",
                "30",
                "--calibration",
                "10",
                "--test",
                "10",
                "--out",
                str(out),
            ]
        )


def test_load_manifest_path_scores_the_replayed_split(
    fake_data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "m.yaml"
    calibrate.main(["--generate-only", "--out", str(out)])
    capsys.readouterr()  # drop the generation output
    calibrate.main(["--manifest", str(out)])
    printed = capsys.readouterr().out
    assert "Loaded manifest" in printed
    assert "Score distribution" in printed  # replay scores too


def test_artifact_requires_policy(
    fake_data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        calibrate.main(["--out", str(tmp_path / "m.yaml"), "--artifact"])
    assert "requires --policy" in capsys.readouterr().err


def test_artifact_writes_yaml_and_vectors(
    fake_data: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The real mapping.csv has no synthetic indy_NN names, so stub position/view.
    monkeypatch.setattr(
        calibrate,
        "load_indy_positions",
        lambda: {f"indy_{i:02d}.jpeg": ("pos", "view") for i in range(35)},
    )
    out = tmp_path / "m.yaml"
    artifact_path = tmp_path / "calibration.yaml"
    calibrate.main(
        [
            "--out",
            str(out),
            "--policy",
            "target-fpr",
            "--artifact",
            str(artifact_path),
        ]
    )
    vectors_path = tmp_path / "calibration.gallery.npy"
    assert artifact_path.exists()
    assert vectors_path.exists()
    assert "Calibration artifact written to" in capsys.readouterr().out

    artifact, vectors = load_artifact(artifact_path)  # round-trips + validates
    assert artifact.aggregation in ("max", "mean-top3")
    assert artifact.gallery_count == 15
    assert vectors.shape[0] == 15
    # The artifact's embedding block records the variant the gallery was built with.
    assert artifact.embedding.model_id == "facebook/dinov2-base"
    assert artifact.embedding.embedding_dim == EMBED_DIM
    assert artifact.embedding.detect is True
    assert artifact.embedding.margin == 0.1
    assert artifact.embedding.min_confidence == 0.25


def test_generated_manifest_records_the_embedding_variant(
    fake_data: Path, tmp_path: Path
) -> None:
    out = tmp_path / "m.yaml"
    calibrate.main(["--generate-only", "--out", str(out)])
    manifest = load_manifest(out)
    assert manifest.embedding.variant_key() == (
        "facebook/dinov2-base",
        True,
        0.1,
        0.25,
    )


def test_requested_variant_vs_sidecar_mismatch_is_loud(
    fake_data: Path, tmp_path: Path
) -> None:
    # A cache sits in the crop-m0.2 folder but its sidecar still records margin 0.1
    # (a folder/sidecar disagreement). The CLI asks for margin 0.2 and must catch it.
    misplaced = EmbeddingsVariant(
        model_id="facebook/dinov2-base", detect=True, margin=0.2
    )
    indy_names = [r.source_filename for r in INDY_RECORDS]
    write_variant_cache(
        misplaced.dir(fake_data / "indy"),
        indy_names,
        breeds=None,
        meta=baseline_meta(len(indy_names)),  # sidecar says margin 0.1
    )
    oxford_names = [r.source_filename for r in OXFORD_RECORDS]
    write_variant_cache(
        misplaced.dir(fake_data / "oxford"),
        oxford_names,
        breeds=[r.breed for r in OXFORD_RECORDS],
        meta=baseline_meta(len(oxford_names)),
    )
    with pytest.raises(SystemExit, match="does not match the requested variant"):
        calibrate.main(["--out", str(tmp_path / "m.yaml"), "--margin", "0.2"])


def test_dual_sidecar_variant_mismatch_is_loud(fake_data: Path, tmp_path: Path) -> None:
    # Re-stamp the Oxford sidecar with a different min_confidence than Indy's.
    # min_confidence has no CLI flag, so only the indy-vs-oxford guard catches it.
    oxford_dir = BASELINE.dir(fake_data / "oxford")
    write_embeddings_meta(
        baseline_meta(len(OXFORD_RECORDS), min_confidence=0.5), oxford_dir
    )
    with pytest.raises(SystemExit, match="identical footing"):
        calibrate.main(["--out", str(tmp_path / "m.yaml")])


def test_manifest_replay_variant_mismatch_is_loud(
    fake_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Generate a manifest, then hand-edit its embedding header to a nocrop variant
    # and replay it against the crop-on caches: replay must fail cleanly.
    out = tmp_path / "m.yaml"
    calibrate.main(["--generate-only", "--out", str(out)])
    import yaml

    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    data["embedding"] = {
        "model_id": "facebook/dinov2-base",
        "detect": False,
        "margin": None,
        "min_confidence": None,
    }
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(SystemExit, match="generated against variant"):
        calibrate.main(["--manifest", str(out)])
