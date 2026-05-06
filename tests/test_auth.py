from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

from openai_api_server_via_codex import auth


def test_borrow_codex_key_uses_explicit_auth_json_and_refreshes_same_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit_auth = tmp_path / "explicit-auth.json"
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    codex_home_auth = codex_home / "auth.json"
    explicit_auth.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": _jwt({"exp": time.time() - 60}),
                    "refresh_token": "old-refresh",
                    "account_id": "acct_explicit",
                },
            }
        )
    )
    codex_home_auth.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": _jwt({"exp": time.time() + 3600}),
                    "refresh_token": "codex-home-refresh",
                    "account_id": "acct_codex_home",
                },
            }
        )
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setattr(
        auth,
        "_refresh",
        lambda refresh_token: {
            "access_token": _jwt({"exp": time.time() + 3600}),
            "refresh_token": f"{refresh_token}-new",
            "id_token": _jwt({"chatgpt_account_id": "acct_from_id_token"}),
        },
    )
    auth.clear_codex_auth_cache()

    token, account_id = auth.borrow_codex_key(auth_json=explicit_auth)

    assert token
    assert account_id == "acct_explicit"
    explicit_data = json.loads(explicit_auth.read_text())
    assert explicit_data["tokens"]["refresh_token"] == "old-refresh-new"
    codex_home_data = json.loads(codex_home_auth.read_text())
    assert codex_home_data["tokens"]["refresh_token"] == "codex-home-refresh"


def test_borrow_codex_key_resolves_env_auth_json_before_codex_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_auth = tmp_path / "env-auth.json"
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    env_auth.write_text(_auth_json("acct_env"))
    (codex_home / "auth.json").write_text(_auth_json("acct_codex_home"))
    monkeypatch.setenv("OPENAI_VIA_CODEX_AUTH_JSON", str(env_auth))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    auth.clear_codex_auth_cache()

    _, account_id = auth.borrow_codex_key()

    assert account_id == "acct_env"


def test_borrow_codex_key_reads_account_id_from_id_token_when_needed(
    tmp_path: Path,
) -> None:
    auth_json = tmp_path / "auth.json"
    auth_json.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": _jwt({"exp": time.time() + 3600}),
                    "id_token": _jwt({"chatgpt_account_id": "acct_from_id_token"}),
                },
            }
        )
    )
    auth.clear_codex_auth_cache()

    _, account_id = auth.borrow_codex_key(auth_json=auth_json)

    assert account_id == "acct_from_id_token"


def test_borrow_codex_key_reads_account_id_from_access_token_openai_auth_claim(
    tmp_path: Path,
) -> None:
    auth_json = tmp_path / "auth.json"
    auth_json.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": _jwt(
                        {
                            "exp": time.time() + 3600,
                            "https://api.openai.com/auth": {
                                "chatgpt_account_id": "acct_from_access_token"
                            },
                        }
                    ),
                },
            }
        )
    )
    auth.clear_codex_auth_cache()

    _, account_id = auth.borrow_codex_key(auth_json=auth_json)

    assert account_id == "acct_from_access_token"


def test_borrow_codex_key_reports_invalid_json_as_borrow_key_error(
    tmp_path: Path,
) -> None:
    auth_json = tmp_path / "auth.json"
    auth_json.write_text("{not-json", encoding="utf-8")
    auth.clear_codex_auth_cache()

    with pytest.raises(auth.BorrowKeyError) as exc_info:
        auth.borrow_codex_key(auth_json=auth_json)

    message = str(exc_info.value)
    assert "Invalid Codex auth JSON" in message
    assert str(auth_json) in message


def test_borrow_codex_key_reports_non_object_json_as_borrow_key_error(
    tmp_path: Path,
) -> None:
    auth_json = tmp_path / "auth.json"
    auth_json.write_text("[]", encoding="utf-8")
    auth.clear_codex_auth_cache()

    with pytest.raises(auth.BorrowKeyError) as exc_info:
        auth.borrow_codex_key(auth_json=auth_json)

    assert "Expected Codex auth JSON object" in str(exc_info.value)


def test_borrow_codex_key_caches_until_auth_file_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auth_json = tmp_path / "auth.json"
    auth_json.write_text(_auth_json("acct_one"))
    original_read_auth = auth._read_auth
    read_count = 0

    def counting_read(path: Path) -> dict[str, Any]:
        nonlocal read_count
        read_count += 1
        return original_read_auth(path)

    monkeypatch.setattr(auth, "_read_auth", counting_read)
    auth.clear_codex_auth_cache()

    assert auth.borrow_codex_key(auth_json=auth_json)[1] == "acct_one"
    assert auth.borrow_codex_key(auth_json=auth_json)[1] == "acct_one"
    assert read_count == 1

    auth_json.write_text(_auth_json("acct_two", padding="changed"))
    os.utime(auth_json, ns=(time.time_ns(), time.time_ns()))

    assert auth.borrow_codex_key(auth_json=auth_json)[1] == "acct_two"
    assert read_count == 2


def test_concurrent_expired_token_refreshes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auth_json = tmp_path / "auth.json"
    auth_json.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": _jwt({"exp": time.time() - 60}),
                    "refresh_token": "refresh",
                    "account_id": "acct",
                },
            }
        )
    )
    refresh_count = 0

    def refresh(refresh_token: str) -> dict[str, Any]:
        nonlocal refresh_count
        refresh_count += 1
        return {
            "access_token": _jwt({"exp": time.time() + 3600}),
            "refresh_token": refresh_token,
        }

    monkeypatch.setattr(auth, "_refresh", refresh)
    auth.clear_codex_auth_cache()

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(
            executor.map(lambda _: auth.borrow_codex_key(auth_json=auth_json), range(5))
        )

    assert {account_id for _, account_id in results} == {"acct"}
    assert refresh_count == 1


def test_write_auth_uses_owner_only_permissions(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    auth._write_auth(auth_path, {"hello": "world"})

    mode = auth_path.stat().st_mode & 0o777
    assert mode == 0o600
    assert json.loads(auth_path.read_text(encoding="utf-8")) == {"hello": "world"}


def test_write_auth_overwrites_existing_with_owner_only_permissions(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text("{}", encoding="utf-8")
    auth_path.chmod(0o644)

    auth._write_auth(auth_path, {"replaced": True})

    mode = auth_path.stat().st_mode & 0o777
    assert mode == 0o600
    assert json.loads(auth_path.read_text(encoding="utf-8")) == {"replaced": True}


def test_refresh_error_redacts_auth_values(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_urlopen(request: object) -> object:
        raise urllib.error.HTTPError(
            url="https://auth.openai.com/oauth/token",
            code=401,
            msg="Unauthorized",
            hdrs=Message(),
            fp=io.BytesIO(
                b'{"error":"invalid","access_token":"tok_abcdefghijklmnopqrstuvwxyz"}'
            ),
        )

    monkeypatch.setattr(auth.urllib.request, "urlopen", fail_urlopen)

    with pytest.raises(auth.BorrowKeyError) as exc_info:
        auth._refresh("refresh_abcdefghijklmnopqrstuvwxyz")

    message = str(exc_info.value)
    assert "abcdefghijklmnopqrstuvwxyz" not in message
    assert '"access_token":"tok_ab******"' in message


def _auth_json(account_id: str, *, padding: str = "") -> str:
    return json.dumps(
        {
            "auth_mode": "chatgpt",
            "tokens": {
                "access_token": _jwt({"exp": time.time() + 3600}),
                "account_id": account_id,
                "padding": padding,
            },
        }
    )


def _jwt(payload: dict[str, Any]) -> str:
    header = {"alg": "none", "typ": "JWT"}
    return ".".join(
        [
            _b64(json.dumps(header).encode()),
            _b64(json.dumps(payload).encode()),
            "sig",
        ]
    )


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip("=")
