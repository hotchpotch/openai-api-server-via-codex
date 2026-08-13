from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

from openai_api_server_via_codex import __version__


def test_package_version_metadata_is_consistent() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert pyproject["project"]["version"] == "0.1.6b2"
    assert __version__ == pyproject["project"]["version"]


def test_console_entry_point_is_declared() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert (
        pyproject["project"]["scripts"]["openai-api-server-via-codex"]
        == "openai_api_server_via_codex.launcher:main"
    )
    assert pyproject["project"]["dependencies"] == []


def test_python_package_contains_only_launcher_and_version_modules() -> None:
    modules = {
        path.relative_to("openai_api_server_via_codex").as_posix()
        for path in Path("openai_api_server_via_codex").rglob("*.py")
        if path.is_file()
    }

    assert modules == {"__init__.py", "__main__.py", "launcher.py"}
