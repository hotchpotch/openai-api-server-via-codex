from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/release-notes.py")


def run_release_notes(tag: str, release_dir: Path) -> str:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), tag, str(release_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_release_notes_prefers_head_notes(tmp_path: Path) -> None:
    release_dir = tmp_path / "releases"
    release_dir.mkdir()
    (release_dir / "HEAD.md").write_text("# HEAD\n\n- Draft change\n", encoding="utf-8")
    (release_dir / "v1.2.3.md").write_text(
        "# v1.2.3\n\n- Final change\n", encoding="utf-8"
    )

    assert run_release_notes("v1.2.3", release_dir) == "- Draft change\n"


def test_release_notes_falls_back_to_tag_notes(tmp_path: Path) -> None:
    release_dir = tmp_path / "releases"
    release_dir.mkdir()
    (release_dir / "HEAD.md").write_text("# HEAD\n", encoding="utf-8")
    (release_dir / "v1.2.3.md").write_text(
        "# v1.2.3\n\nReleased: 2026-05-06\n\n- Final change\n",
        encoding="utf-8",
    )

    assert (
        run_release_notes("v1.2.3", release_dir)
        == "Released: 2026-05-06\n- Final change\n"
    )


def test_release_notes_uses_default_when_no_notes_exist(tmp_path: Path) -> None:
    release_dir = tmp_path / "releases"
    release_dir.mkdir()

    assert run_release_notes("v1.2.3", release_dir) == "Release v1.2.3\n"
