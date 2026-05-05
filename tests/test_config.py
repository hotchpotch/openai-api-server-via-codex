from __future__ import annotations

from pathlib import Path

from openai_api_server_via_codex import config


def test_default_config_path_uses_xdg_config_home(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))

    assert config.default_config_path() == (
        tmp_path
        / "xdg-config"
        / "openai-api-server-via-codex"
        / "config.toml"
    ).resolve()


def test_default_config_path_falls_back_to_home_config(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert config.default_config_path() == (
        tmp_path / ".config" / "openai-api-server-via-codex" / "config.toml"
    ).resolve()


def test_default_config_toml_documents_core_options() -> None:
    text = config.default_config_toml()

    assert "[server]" in text
    assert 'backend = "codex-http"' in text
    assert 'host = "127.0.0.1"' in text
    assert "[daemon]" in text
    assert "state_dir" in text
    assert "[codex]" in text
    assert "auth_json" in text
