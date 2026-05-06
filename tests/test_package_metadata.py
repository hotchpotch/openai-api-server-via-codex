from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

from openai_api_server_via_codex import __version__
from openai_api_server_via_codex import server


def test_package_version_metadata_is_consistent() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert pyproject["project"]["version"] == "0.0.4"
    assert __version__ == pyproject["project"]["version"]


def test_console_entry_point_is_declared() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert (
        pyproject["project"]["scripts"]["openai-api-server-via-codex"]
        == "openai_api_server_via_codex.server:main"
    )


def test_top_level_version_option(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        server.parse_args(["--version"])

    assert exc_info.value.code == 0
    assert re.fullmatch(r"0\.0\.4\n", capsys.readouterr().out)
