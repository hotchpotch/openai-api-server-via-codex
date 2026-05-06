from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any


MASK = "******"
DEFAULT_PREFIX_LENGTH = 6

_SENSITIVE_KEY_RE = re.compile(
    r"(?i)\b("
    r"access[_-]?token|refresh[_-]?token|id[_-]?token|api[_-]?key|"
    r"authorization|client[_-]?secret|secret|password"
    r")\b"
)
_AUTHORIZATION_BEARER_RE = re.compile(
    r"(?i)([\"']?\bAuthorization\b[\"']?\s*[:=]\s*[\"']?Bearer\s+)([^\"',&\s}]+)"
)
_BEARER_RE = re.compile(r"(?i)(\bBearer\s+)([A-Za-z0-9._~+/=-]{8,})")
_KEY_VALUE_RE = re.compile(
    r"(?i)([\"']?\b(?:"
    r"access[_-]?token|refresh[_-]?token|id[_-]?token|api[_-]?key|"
    r"client[_-]?secret|secret|password"
    r")\b[\"']?\s*[:=]\s*[\"']?)([^\"',&\s}]+)"
)
_JWT_RE = re.compile(
    r"\b([A-Za-z0-9_-]{8,})\.([A-Za-z0-9_-]{8,})\.([A-Za-z0-9_-]{8,})\b"
)


class RedactingLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_sensitive_text(record.getMessage())
        record.args = ()
        return True


def mask_secret(value: object, *, prefix_length: int = DEFAULT_PREFIX_LENGTH) -> str:
    text = str(value)
    if len(text) <= prefix_length:
        return MASK
    return f"{text[:prefix_length]}{MASK}"


def redact_sensitive_text(text: str) -> str:
    redacted = _AUTHORIZATION_BEARER_RE.sub(_mask_match_secret, text)
    redacted = _BEARER_RE.sub(_mask_match_secret, redacted)
    redacted = _KEY_VALUE_RE.sub(_mask_match_secret, redacted)
    return _JWT_RE.sub(lambda match: mask_secret(match.group(0)), redacted)


def redact_sensitive_data(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            if _is_sensitive_key(key):
                redacted[key] = _mask_sensitive_value(item)
            else:
                redacted[key] = redact_sensitive_data(item)
        return redacted
    if isinstance(value, tuple):
        return tuple(redact_sensitive_data(item) for item in value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [redact_sensitive_data(item) for item in value]
    if isinstance(value, str):
        return redact_sensitive_text(value)
    return value


def install_redacting_filter(target: logging.Filterer) -> None:
    if not any(isinstance(filter_, RedactingLogFilter) for filter_ in target.filters):
        target.addFilter(RedactingLogFilter())


def _is_sensitive_key(key: object) -> bool:
    return bool(_SENSITIVE_KEY_RE.search(str(key)))


def _mask_sensitive_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        bearer = re.match(r"(?i)^Bearer\s+(.+)$", value)
        if bearer:
            return f"Bearer {mask_secret(bearer.group(1))}"
        return mask_secret(value)
    return MASK


def _mask_match_secret(match: re.Match[str]) -> str:
    return f"{match.group(1)}{mask_secret(match.group(2))}"
