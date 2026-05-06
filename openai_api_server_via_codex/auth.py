from __future__ import annotations

import base64
import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .redaction import redact_sensitive_text


REFRESH_URL = "https://auth.openai.com/oauth/token"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
REFRESH_SKEW_SECONDS = 30
AUTH_JSON_ENV = "OPENAI_VIA_CODEX_AUTH_JSON"


class BorrowKeyError(Exception):
    """Raised when local Codex ChatGPT OAuth credentials cannot be used."""


@dataclass(frozen=True)
class CodexAuthConfig:
    auth_json: Path | None = None


@dataclass(frozen=True)
class _CachedBorrowedKey:
    mtime_ns: int
    size: int
    access_token: str
    account_id: str | None
    exp: float | None


_AUTH_CACHE: dict[Path, _CachedBorrowedKey] = {}
_AUTH_CACHE_LOCK = threading.Lock()


def borrow_codex_key(auth_json: str | Path | None = None) -> tuple[str, str | None]:
    """Return an access token and optional ChatGPT account id from Codex auth."""

    auth_path = resolve_auth_path(auth_json)

    with _AUTH_CACHE_LOCK:
        stat = auth_path.stat()
        cached = _AUTH_CACHE.get(auth_path)
        if cached and _cache_matches(cached, stat) and _token_is_fresh(cached.exp):
            return cached.access_token, cached.account_id

        data = _read_auth(auth_path)

        tokens = data.get("tokens")
        if not isinstance(tokens, dict) or not tokens.get("access_token"):
            raise BorrowKeyError("No ChatGPT tokens found. Run `codex login` first.")

        access_token = str(tokens["access_token"])
        exp = _jwt_exp(access_token)
        if not _token_is_fresh(exp):
            refresh_token = tokens.get("refresh_token")
            if not refresh_token:
                raise BorrowKeyError("No refresh token available. Run `codex login` again.")

            new_tokens = _refresh(str(refresh_token))
            for key in ("access_token", "id_token", "refresh_token"):
                if new_tokens.get(key):
                    tokens[key] = new_tokens[key]

            data["tokens"] = tokens
            data["last_refresh"] = time.strftime(
                "%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()
            )
            _write_auth(auth_path, data)
            stat = auth_path.stat()

        access_token = str(tokens["access_token"])
        account_id = _account_id(tokens)
        exp = _jwt_exp(access_token)
        _AUTH_CACHE[auth_path] = _CachedBorrowedKey(
            mtime_ns=stat.st_mtime_ns,
            size=stat.st_size,
            access_token=access_token,
            account_id=account_id,
            exp=exp,
        )
        return access_token, account_id


def clear_codex_auth_cache() -> None:
    with _AUTH_CACHE_LOCK:
        _AUTH_CACHE.clear()


def resolve_auth_path(auth_json: str | Path | None = None) -> Path:
    if auth_json is not None:
        path = Path(auth_json)
    elif env_auth_json := os.environ.get(AUTH_JSON_ENV):
        path = Path(env_auth_json)
    else:
        codex_home = Path(os.environ.get("CODEX_HOME", "~/.codex"))
        path = codex_home / "auth.json"

    path = path.expanduser().resolve()
    if not path.exists():
        raise BorrowKeyError(f"Codex auth file not found at {path}.")
    return path


def _cache_matches(cached: _CachedBorrowedKey, stat: os.stat_result) -> bool:
    return cached.mtime_ns == stat.st_mtime_ns and cached.size == stat.st_size


def _token_is_fresh(exp: float | None) -> bool:
    return exp is None or time.time() < (exp - REFRESH_SKEW_SECONDS)


def _account_id(tokens: dict[str, Any]) -> str | None:
    account_id = tokens.get("account_id")
    if account_id:
        return str(account_id)
    id_token = tokens.get("id_token")
    if isinstance(id_token, str):
        payload = _jwt_payload(id_token)
        account_id = payload.get("chatgpt_account_id")
        if account_id:
            return str(account_id)

    access_token = tokens.get("access_token")
    if not isinstance(access_token, str):
        return None
    payload = _jwt_payload(access_token)
    openai_auth = payload.get("https://api.openai.com/auth")
    if isinstance(openai_auth, dict):
        account_id = openai_auth.get("chatgpt_account_id")
        if account_id:
            return str(account_id)
    account_id = payload.get("chatgpt_account_id")
    return str(account_id) if account_id else None


def _read_auth(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BorrowKeyError(f"Invalid Codex auth JSON at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise BorrowKeyError(f"Expected Codex auth JSON object at {path}.")
    if data.get("auth_mode") != "chatgpt":
        raise BorrowKeyError(
            f"Expected Codex auth_mode 'chatgpt', got {data.get('auth_mode')!r}."
        )
    return data


def _write_auth(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)
    path.chmod(0o600)


def _jwt_exp(token: str) -> float | None:
    payload = _jwt_payload(token)
    exp = payload.get("exp")
    if isinstance(exp, int | float):
        return float(exp)
    return None


def _jwt_payload(token: str) -> dict[str, Any]:
    try:
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


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
        raise BorrowKeyError(
            f"Token refresh failed (HTTP {exc.code}): "
            f"{redact_sensitive_text(error_body)}"
        )
    except urllib.error.URLError as exc:
        raise BorrowKeyError(f"Token refresh failed: {exc}") from exc
