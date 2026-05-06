from __future__ import annotations

from openai_api_server_via_codex.redaction import (
    mask_secret,
    redact_sensitive_data,
    redact_sensitive_text,
)


def test_mask_secret_keeps_only_prefix() -> None:
    assert mask_secret("abcdefghijklmnopqrstuvwxyz") == "abcdef******"
    assert mask_secret("short") == "******"


def test_redact_sensitive_text_masks_auth_values() -> None:
    text = (
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz "
        '"access_token": "tok_abcdefghijklmnopqrstuvwxyz", '
        "refresh_token=ref_abcdefghijklmnopqrstuvwxyz&safe=value "
        "jwt=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )

    redacted = redact_sensitive_text(text)

    assert "abcdefghijklmnopqrstuvwxyz" not in redacted
    assert "Bearer abcdef******" in redacted
    assert '"access_token": "tok_ab******"' in redacted
    assert "refresh_token=ref_ab******&safe=value" in redacted
    assert "eyJhbG******" in redacted


def test_redact_sensitive_data_masks_nested_sensitive_fields() -> None:
    data = {
        "tokens": {
            "access_token": "access_abcdefghijklmnopqrstuvwxyz",
            "refresh_token": "refresh_abcdefghijklmnopqrstuvwxyz",
        },
        "headers": {"Authorization": "Bearer zyxwvutsrqponmlkjihgfedcba"},
        "safe": "value",
    }

    redacted = redact_sensitive_data(data)

    assert redacted["tokens"]["access_token"] == "access******"
    assert redacted["tokens"]["refresh_token"] == "refres******"
    assert redacted["headers"]["Authorization"] == "Bearer zyxwvu******"
    assert redacted["safe"] == "value"
