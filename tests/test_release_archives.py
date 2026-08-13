from __future__ import annotations

import hashlib
import io
import sys
import tarfile
import zipfile
from importlib import util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _load_script(path: Path, name: str) -> ModuleType:
    spec = util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build_release_archives = _load_script(
    Path("scripts/build-release-archives.py"), "build_release_archives"
)
build_platform_wheels = _load_script(
    Path("scripts/build-platform-wheels.py"), "release_archive_platform_wheels"
)


def _wheel(wheel_dir: Path, target: Any, version: str, binary: bytes) -> None:
    path = wheel_dir / (
        f"openai_api_server_via_codex-{version}-py3-none-{target.wheel_tag}.whl"
    )
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(
            f"openai_api_server_via_codex/bin/{target.executable}", binary
        )


def test_release_targets_match_platform_wheel_targets() -> None:
    archive_targets = {
        (target.goos, target.goarch, target.wheel_tag, target.executable)
        for target in build_release_archives.TARGETS
    }
    wheel_targets = {
        (target.goos, target.goarch, target.wheel_tag, target.executable)
        for target in build_platform_wheels.TARGETS
    }

    assert archive_targets == wheel_targets


def test_tar_archive_is_reproducible_and_contains_wheel_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    target = build_release_archives.TARGETS[0]
    binary = b"linux-binary"
    _wheel(wheel_dir, target, "1.2.3", binary)
    monkeypatch.setattr(build_release_archives, "ROOT", tmp_path)
    (tmp_path / "LICENSE").write_text("license\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("readme\n", encoding="utf-8")

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    first_archive = build_release_archives.build_archive(
        wheel_dir, first, target, "1.2.3", 1_700_000_000
    )
    second_archive = build_release_archives.build_archive(
        wheel_dir, second, target, "1.2.3", 1_700_000_000
    )

    assert first_archive.read_bytes() == second_archive.read_bytes()
    with tarfile.open(first_archive, "r:gz") as archive:
        assert archive.getnames() == [target.executable, "LICENSE", "README.md"]
        member = archive.getmember(target.executable)
        assert member.mode == 0o755
        extracted = archive.extractfile(member)
        assert extracted is not None
        assert extracted.read() == binary


def test_windows_archive_and_checksums(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wheel_dir = tmp_path / "wheels"
    output_dir = tmp_path / "release"
    wheel_dir.mkdir()
    output_dir.mkdir()
    target = next(
        target
        for target in build_release_archives.TARGETS
        if target.goos == "windows"
    )
    binary = b"windows-binary"
    _wheel(wheel_dir, target, "1.2.3", binary)
    monkeypatch.setattr(build_release_archives, "ROOT", tmp_path)
    (tmp_path / "LICENSE").write_text("license\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("readme\n", encoding="utf-8")

    archive = build_release_archives.build_archive(
        wheel_dir, output_dir, target, "1.2.3", 1_700_000_000
    )
    checksum_file = build_release_archives.write_checksums([archive], output_dir)

    with zipfile.ZipFile(io.BytesIO(archive.read_bytes())) as packaged:
        assert packaged.namelist() == [target.executable, "LICENSE", "README.md"]
        assert packaged.read(target.executable) == binary
    expected = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert checksum_file.read_text(encoding="utf-8") == (
        f"{expected}  {archive.name}\n"
    )


def test_read_binary_rejects_missing_platform_wheel(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="expected exactly one wheel"):
        build_release_archives.read_binary(
            tmp_path, build_release_archives.TARGETS[0], "1.2.3"
        )
