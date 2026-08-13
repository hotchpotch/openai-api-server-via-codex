from __future__ import annotations

import os
from importlib import util
from pathlib import Path
from types import ModuleType

import pytest


def _load_script() -> ModuleType:
    path = Path("scripts/run-platform-wheel-e2e.py")
    spec = util.spec_from_file_location("run_platform_wheel_e2e", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_platform_wheel_e2e = _load_script()


def test_select_wheel_requires_exact_native_tag(tmp_path: Path) -> None:
    expected = tmp_path / "package-1-py3-none-win_amd64.whl"
    expected.write_bytes(b"wheel")
    (tmp_path / "package-1-py3-none-win_arm64.whl").write_bytes(b"other")

    assert run_platform_wheel_e2e.select_wheel(tmp_path, "win_amd64") == expected


def test_select_wheel_rejects_missing_or_duplicate(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="expected one"):
        run_platform_wheel_e2e.select_wheel(tmp_path, "win_amd64")

    (tmp_path / "one-py3-none-win_amd64.whl").write_bytes(b"one")
    (tmp_path / "two-py3-none-win_amd64.whl").write_bytes(b"two")
    with pytest.raises(RuntimeError, match="expected one"):
        run_platform_wheel_e2e.select_wheel(tmp_path, "win_amd64")


def test_environment_paths_match_host_platform(tmp_path: Path) -> None:
    scripts = "Scripts" if os.name == "nt" else "bin"
    python = "python.exe" if os.name == "nt" else "python"
    command = (
        "openai-api-server-via-codex.exe"
        if os.name == "nt"
        else "openai-api-server-via-codex"
    )

    assert run_platform_wheel_e2e.environment_python(tmp_path) == (
        tmp_path / scripts / python
    )
    assert run_platform_wheel_e2e.environment_command(tmp_path) == (
        tmp_path / scripts / command
    )
