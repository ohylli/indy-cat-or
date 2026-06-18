"""Light CLI smoke tests for ``calibrate.py``.

The split logic is covered in ``test_split_manifest``; here we only check the CLI
wiring: mutual exclusion, that ``--generate-only`` writes a loadable manifest, and
that a default run both writes a manifest and prints the not-implemented notice.
The Indy/Oxford loaders are pointed at synthetic CSVs via monkeypatch so the tests
never touch the gitignored real data.
"""

from __future__ import annotations

from pathlib import Path

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
    """Point calibrate's loaders at small in-memory record sets."""
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


def test_manifest_and_generation_flags_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        calibrate.main(["--manifest", "x.yaml", "--seed", "42"])


def test_generate_only_writes_loadable_manifest(
    fake_data: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "m.yaml"
    calibrate.main(["--generate-only", "--out", str(out)])
    assert out.exists()
    manifest = load_manifest(out)  # round-trips through validation
    assert len(manifest.indy_gallery) == 15
    notice = capsys.readouterr().out
    assert "not yet implemented" not in notice  # generate-only suppresses the notice


def test_default_run_writes_manifest_and_prints_notice(
    fake_data: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "m.yaml"
    calibrate.main(["--out", str(out)])
    assert out.exists()
    assert "not yet implemented" in capsys.readouterr().out


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


def test_load_manifest_path_does_not_print_notice(
    fake_data: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "m.yaml"
    calibrate.main(["--generate-only", "--out", str(out)])
    capsys.readouterr()  # drop the generation output
    calibrate.main(["--manifest", str(out)])
    printed = capsys.readouterr().out
    assert "Loaded manifest" in printed
    assert "not yet implemented" not in printed
