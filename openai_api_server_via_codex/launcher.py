from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import NoReturn


def bundled_binary(platform_name: str | None = None) -> Path:
    platform_name = os.name if platform_name is None else platform_name
    name = (
        "openai-api-server-via-codex.exe"
        if platform_name == "nt"
        else "openai-api-server-via-codex"
    )
    return Path(__file__).with_name("bin") / name


def _exec_go(binary: Path) -> NoReturn:
    os.execv(str(binary), [str(binary), *sys.argv[1:]])


def main() -> None:
    binary = bundled_binary()
    if not binary.is_file():
        raise SystemExit(
            "The Go server binary is not bundled for this platform. "
            "Install a supported platform wheel from PyPI or build "
            "./cmd/openai-api-server-via-codex from the source repository."
        )
    if os.name != "nt" and not os.access(binary, os.X_OK):
        raise SystemExit(f"The bundled Go server is not executable: {binary}")
    try:
        _exec_go(binary)
    except OSError as error:
        raise SystemExit(f"Failed to execute the bundled Go server: {error}") from error
