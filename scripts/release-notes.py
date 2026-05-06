#!/usr/bin/env python3
"""Print GitHub Release notes from draft or finalized release files."""

from __future__ import annotations

import argparse
from pathlib import Path


def read_notes(path: Path) -> str:
    if not path.exists():
        return ""

    lines = path.read_text(encoding="utf-8").splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(line for line in lines if line.strip()).strip()


def release_notes(tag: str, release_dir: Path) -> str:
    notes = read_notes(release_dir / "HEAD.md")
    if not notes:
        notes = read_notes(release_dir / f"{tag}.md")
    return notes or f"Release {tag}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print release notes from docs/releases/HEAD.md or a tag file."
    )
    parser.add_argument("tag", help="Release tag, for example v0.0.2")
    parser.add_argument(
        "release_dir",
        nargs="?",
        default="docs/releases",
        type=Path,
        help="Directory containing HEAD.md and vX.Y.Z.md release notes.",
    )
    args = parser.parse_args()

    print(release_notes(args.tag, args.release_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
