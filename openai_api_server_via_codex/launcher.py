from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import NoReturn

RUNTIME_ENV = "OPENAI_VIA_CODEX_RUNTIME"


def bundled_binary() -> Path:
    name = "openai-api-server-via-codex.exe" if os.name == "nt" else "openai-api-server-via-codex"
    return Path(__file__).with_name("bin") / name


def _exec_go(binary: Path) -> NoReturn:
    os.execv(str(binary), [str(binary), *sys.argv[1:]])


def main() -> None:
    runtime = os.environ.get(RUNTIME_ENV, "auto").strip().lower()
    if runtime not in {"auto", "go", "python"}:
        raise SystemExit(f"{RUNTIME_ENV} must be auto, go, or python")

    binary = bundled_binary()
    if runtime in {"auto", "go"} and binary.is_file():
        _exec_go(binary)
    if runtime == "go":
        raise SystemExit(
            "The Go binary is not bundled in this source installation. "
            "Install a supported platform wheel or build ./cmd/openai-api-server-via-codex."
        )

    from .server import main as python_main

    python_main()
