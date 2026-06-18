"""Light CLI smoke tests for ``calibrate.py``.

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

import calibrate
from split_manifest import IndyRecord, OxfordRecord, load_manifest

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


@pytest.fixture
def fake_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub calibrate's metadata loaders and embeddings cache with in-memory data.

    The metadata loaders return small record sets; the embeddings cache returns
    synthetic vectors aligned to those records' filenames, so the full scoring
    path runs without a GPU, real weights, or the gitignored data.
    """
    indy = [
        IndyRecord(f"indy_{i:02d}.jpeg", head_visible=True, tail_visible=i < 9)
        for i in range(35)
    ]
    oxford = [
        OxfordRecord(f"{breed}_{k}.jpg", breed)
        for breed in ("Persian", "Ragdoll")
        for k in range(1, 11)
    ]
    monkeypatch.setattr(calibrate, "load_indy_metadata", lambda: indy)
    monkeypatch.setattr(calibrate, "load_oxford_metadata", lambda: oxford)

    def fake_cache(
        metadata_path: Path, embeddings_path: Path
    ) -> tuple[list[str], object]:
        records = indy if metadata_path == calibrate.INDY_METADATA else oxford
        names = [r.source_filename for r in records]
        rng = np.random.default_rng(len(names))
        return names, rng.standard_normal((len(names), 8)).astype(np.float32)

    monkeypatch.setattr(calibrate, "load_cached_embeddings", fake_cache)


def test_manifest_and_generation_flags_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        calibrate.main(["--manifest", "x.yaml", "--seed", "42"])


def test_generate_only_writes_loadable_manifest_and_skips_scoring(
    fake_data: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "m.yaml"
    calibrate.main(["--generate-only", "--out", str(out)])
    assert out.exists()
    manifest = load_manifest(out)  # round-trips through validation
    assert len(manifest.indy_gallery) == 15
    assert "Score distribution" not in capsys.readouterr().out  # no scoring


def test_default_run_writes_manifest_and_prints_report(
    fake_data: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "m.yaml"
    calibrate.main(["--out", str(out)])
    assert out.exists()
    printed = capsys.readouterr().out
    assert "Score distribution" in printed
    assert "Negatives by group" in printed


def test_scores_out_writes_csv(
    fake_data: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "m.yaml"
    scores = tmp_path / "scores.csv"
    calibrate.main(["--out", str(out), "--scores-out", str(scores)])
    assert scores.exists()
    assert "Per-image scores written" in capsys.readouterr().out


def test_overask_exits_loudly(fake_data: None, tmp_path: Path) -> None:
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
    fake_data: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "m.yaml"
    calibrate.main(["--generate-only", "--out", str(out)])
    capsys.readouterr()  # drop the generation output
    calibrate.main(["--manifest", str(out)])
    printed = capsys.readouterr().out
    assert "Loaded manifest" in printed
    assert "Score distribution" in printed  # replay scores too
