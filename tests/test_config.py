from __future__ import annotations

from pathlib import Path

import pytest

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
    assert "backend =" not in text
    assert 'host = "127.0.0.1"' in text
    assert "port = 18080" in text
    assert 'default_model = "gpt-5.6-luna"' in text
    assert "timeout = 300.0" in text
    assert '# api_key = "change-me"' in text
    assert "max_stored_items = 1000" in text
    assert "max_concurrent_requests = 10" in text
    assert "[daemon]" in text
    assert "state_dir" in text
    assert "[codex]" in text
    assert "auth_json" in text
    assert "# [compat.drop_params_by_model]" in text
    assert '# "gpt-5.6-luna" = ["temperature"]' in text


def test_drop_params_by_model_defaults_to_no_rules() -> None:
    assert config.drop_params_by_model_from_config({}) == {}


def test_drop_params_by_model_loads_and_normalizes_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[compat.drop_params_by_model]
"gpt-5.6-luna" = ["temperature"]
"gpt-5.6-terra" = ["temperature", "top_p"]
""",
        encoding="utf-8",
    )

    loaded = config.load_config(config_path)

    assert config.drop_params_by_model_from_config(loaded) == {
        "gpt-5.6-luna": ("temperature",),
        "gpt-5.6-terra": ("temperature", "top_p"),
    }


@pytest.mark.parametrize(
    ("toml", "error_type", "message"),
    [
        (
            '[compat]\ndrop_params_by_model = "temperature"\n',
            TypeError,
            "compat.drop_params_by_model must be a TOML table",
        ),
        (
            '[compat.drop_params_by_model]\n"gpt-5.6-luna" = "temperature"\n',
            TypeError,
            "compat.drop_params_by_model.gpt-5.6-luna must be an array of strings",
        ),
        (
            '[compat.drop_params_by_model]\n"gpt-5.6-luna" = ["temperature", 1]\n',
            TypeError,
            "compat.drop_params_by_model.gpt-5.6-luna must contain only non-empty strings",
        ),
        (
            '[compat.drop_params_by_model]\n"gpt-5.6-luna" = [" "]\n',
            ValueError,
            "compat.drop_params_by_model.gpt-5.6-luna must contain only non-empty strings",
        ),
    ],
)
def test_drop_params_by_model_rejects_malformed_config(
    tmp_path: Path, toml: str, error_type: type[Exception], message: str
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(toml, encoding="utf-8")

    with pytest.raises(error_type, match=message):
        config.drop_params_by_model_from_config(config.load_config(config_path))


def test_load_config_reads_existing_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[server]
host = "127.0.0.9"
port = 8123

[codex]
auth_json = "/tmp/auth.json"

[daemon]
state_dir = "/tmp/state"
""",
        encoding="utf-8",
    )

    loaded = config.load_config(config_path)

    assert loaded["server"]["host"] == "127.0.0.9"
    assert loaded["server"]["port"] == 8123
    assert loaded["codex"]["auth_json"] == "/tmp/auth.json"
    assert loaded["daemon"]["state_dir"] == "/tmp/state"


def test_load_config_returns_empty_dict_for_missing_file(tmp_path: Path) -> None:
    assert config.load_config(tmp_path / "missing.toml") == {}


def test_write_default_config_uses_owner_only_permissions(tmp_path: Path) -> None:
    config_path = tmp_path / "nested" / "config.toml"

    written = config.write_default_config(config_path)

    assert written == config_path
    mode = config_path.stat().st_mode & 0o777
    assert mode == 0o600
    assert "[server]" in config_path.read_text(encoding="utf-8")


def test_write_default_config_force_overwrites_with_owner_only_permissions(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("# old", encoding="utf-8")
    config_path.chmod(0o644)

    config.write_default_config(config_path, force=True)

    mode = config_path.stat().st_mode & 0o777
    assert mode == 0o600
    assert "[server]" in config_path.read_text(encoding="utf-8")
