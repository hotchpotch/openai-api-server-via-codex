from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from .backend import CODEX_BASE_URL
from .compat import DEFAULT_MAX_STORED_ITEMS, DEFAULT_MODEL


APP_NAME = "openai-api-server-via-codex"
CONFIG_ENV = "OPENAI_VIA_CODEX_CONFIG"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18080
DEFAULT_TIMEOUT = 300.0
DEFAULT_MAX_CONCURRENT_REQUESTS = 10
DEFAULT_CLIENT_VERSION = "1.0.0"
DEFAULT_STOP_TIMEOUT = 10.0


def default_config_dir() -> Path:
    if xdg_config_home := os.environ.get("XDG_CONFIG_HOME"):
        root = Path(xdg_config_home)
    else:
        root = Path("~/.config")
    return (root / APP_NAME).expanduser().resolve()


def default_config_path() -> Path:
    return default_config_dir() / "config.toml"


def resolve_config_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()
    if env_path := os.environ.get(CONFIG_ENV):
        return Path(env_path).expanduser().resolve()
    return default_config_path()


def default_daemon_state_dir() -> Path:
    return default_config_dir() / "run"


def default_config_toml() -> str:
    state_dir = _toml_string(str(default_daemon_state_dir()))
    return f"""# OpenAI API Server via Codex configuration.
# CLI flags override environment variables, and environment variables override
# values in this file.

[server]
host = "{DEFAULT_HOST}"
port = {DEFAULT_PORT}
default_model = "{DEFAULT_MODEL}"
timeout = {DEFAULT_TIMEOUT}
verbose = false
max_stored_items = {DEFAULT_MAX_STORED_ITEMS}
max_concurrent_requests = {DEFAULT_MAX_CONCURRENT_REQUESTS}

[codex]
auth_json = "~/.codex/auth.json"
backend_base_url = "{CODEX_BASE_URL}"
client_version = "{DEFAULT_CLIENT_VERSION}"

[daemon]
state_dir = {state_dir}
# pid_file = "/path/to/openai-api-server-via-codex.pid"
# log_file = "/path/to/openai-api-server-via-codex.log"
stop_timeout = {DEFAULT_STOP_TIMEOUT}
"""


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_config_path(path)
    if not config_path.exists():
        return {}
    with config_path.open("rb") as file:
        loaded = tomllib.load(file)
    return loaded if isinstance(loaded, dict) else {}


def write_default_config(path: str | Path | None = None, *, force: bool = False) -> Path:
    config_path = resolve_config_path(path)
    if config_path.exists() and not force:
        raise FileExistsError(config_path)
    config_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    config_path.write_text(default_config_toml(), encoding="utf-8")
    config_path.chmod(0o600)
    return config_path


def _toml_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
