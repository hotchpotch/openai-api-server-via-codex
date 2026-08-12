from __future__ import annotations

import subprocess
import sys
from importlib import util
from pathlib import Path
from types import ModuleType

import pytest


def _load_script() -> ModuleType:
    path = Path("scripts/build-platform-wheels.py")
    spec = util.spec_from_file_location("build_platform_wheels", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build_platform_wheels = _load_script()


def _write_launcher_files(root: Path) -> Path:
    package = root / "openai_api_server_via_codex"
    package.mkdir()
    for name in build_platform_wheels.LAUNCHER_FILES:
        package.joinpath(name).write_text("", encoding="utf-8")
    return package


def test_validate_launcher_package_rejects_stale_binary(tmp_path: Path) -> None:
    package = _write_launcher_files(tmp_path)
    stale_binary = package / "bin" / "openai-api-server-via-codex"
    stale_binary.parent.mkdir()
    stale_binary.write_bytes(b"stale")

    with pytest.raises(RuntimeError, match="bin/openai-api-server-via-codex"):
        build_platform_wheels.validate_launcher_package(tmp_path)


def test_build_binary_requires_nonempty_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command: list[str] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        command.extend(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(build_platform_wheels.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="produced no usable binary"):
        build_platform_wheels.build_binary(
            build_platform_wheels.TARGETS[0], tmp_path / "missing", "1.2.3"
        )

    assert "-mod=readonly" in command


def test_validate_distribution_set_rejects_generic_wheel(tmp_path: Path) -> None:
    platform_wheel = tmp_path / "package-1-py3-none-win_amd64.whl"
    platform_wheel.write_bytes(b"wheel")
    (tmp_path / ".gitignore").write_text("*\n", encoding="utf-8")
    build_platform_wheels.validate_distribution_set(tmp_path, [platform_wheel])

    (tmp_path / "package-1-py3-none-any.whl").write_bytes(b"generic")

    with pytest.raises(RuntimeError, match="py3-none-any"):
        build_platform_wheels.validate_distribution_set(tmp_path, [platform_wheel])


def test_validate_distribution_set_rejects_unexpected_visible_file(
    tmp_path: Path,
) -> None:
    platform_wheel = tmp_path / "package-1-py3-none-win_amd64.whl"
    platform_wheel.write_bytes(b"wheel")
    (tmp_path / "checksums.txt").write_text("unexpected\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="checksums.txt"):
        build_platform_wheels.validate_distribution_set(tmp_path, [platform_wheel])


def test_release_targets_are_exactly_the_six_supported_platforms() -> None:
    assert {target.wheel_tag for target in build_platform_wheels.TARGETS} == {
        "manylinux_2_17_x86_64",
        "manylinux_2_17_aarch64",
        "macosx_10_15_x86_64",
        "macosx_11_0_arm64",
        "win_amd64",
        "win_arm64",
    }


@pytest.mark.parametrize("archive", ["package-1.tar.gz", "package-1.zip"])
def test_validate_distribution_set_rejects_source_archives(
    tmp_path: Path, archive: str
) -> None:
    platform_wheel = tmp_path / "package-1-py3-none-win_amd64.whl"
    platform_wheel.write_bytes(b"wheel")
    tmp_path.joinpath(archive).write_bytes(b"source")

    with pytest.raises(RuntimeError, match="source_archives"):
        build_platform_wheels.validate_distribution_set(tmp_path, [platform_wheel])
