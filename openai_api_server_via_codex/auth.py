from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REFRESH_URL = "https://auth.openai.com/oauth/token"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
REFRESH_SKEW_SECONDS = 30


class BorrowKeyError(Exception):
    """Raised when local Codex ChatGPT OAuth credentials cannot be used."""


def borrow_codex_key() -> tuple[str, str | None]:
    """Return an access token and optional ChatGPT account id from Codex auth."""

    auth_path = _auth_path()
    data = _read_auth(auth_path)

    tokens = data.get("tokens")
    if not isinstance(tokens, dict) or not tokens.get("access_token"):
        raise BorrowKeyError("No ChatGPT tokens found. Run `codex login` first.")

    access_token = str(tokens["access_token"])
    account_id = tokens.get("account_id")
    exp = _jwt_exp(access_token)
    if exp is not None and time.time() < (exp - REFRESH_SKEW_SECONDS):
        return access_token, str(account_id) if account_id else None

    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        raise BorrowKeyError("No refresh token available. Run `codex login` again.")

    new_tokens = _refresh(str(refresh_token))
    for key in ("access_token", "id_token", "refresh_token"):
        if new_tokens.get(key):
            tokens[key] = new_tokens[key]

    data["tokens"] = tokens
    data["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    _write_auth(auth_path, data)

    access_token = str(tokens["access_token"])
    account_id = tokens.get("account_id")
    return access_token, str(account_id) if account_id else None


def _auth_path() -> Path:
    codex_home = Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()
    path = codex_home / "auth.json"
    if not path.exists():
        raise BorrowKeyError(f"Codex auth file not found at {path}.")
    return path


def _read_auth(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("auth_mode") != "chatgpt":
        raise BorrowKeyError(
            f"Expected Codex auth_mode 'chatgpt', got {data.get('auth_mode')!r}."
        )
    return data


def _write_auth(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)
    path.chmod(0o600)


def _jwt_exp(token: str) -> float | None:
    try:
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        exp = payload.get("exp")
        if isinstance(exp, int | float):
            return float(exp)
    except Exception:
        return None
    return None


def _refresh(refresh_token: str) -> dict[str, Any]:
    body = json.dumps(
        {
            "client_id": CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
    ).encode()
    req = urllib.request.Request(
        REFRESH_URL,
        data=body,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode(errors="replace")
        raise BorrowKeyError(f"Token refresh failed (HTTP {exc.code}): {error_body}")
    except urllib.error.URLError as exc:
        raise BorrowKeyError(f"Token refresh failed: {exc}") from exc
