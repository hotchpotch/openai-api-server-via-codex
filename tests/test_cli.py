from __future__ import annotations

from pathlib import Path

import pytest

from openai_api_server_via_codex import server


def test_parse_args_defaults_to_foreground_serve() -> None:
    args = server.parse_args(["--host", "127.0.0.2", "--port", "8123"])

    assert args.command == "serve"
    assert args.host == "127.0.0.2"
    assert args.port == 8123


def test_parse_args_keeps_top_level_help(capsys) -> None:
    try:
        server.parse_args(["--help"])
    except SystemExit as exc:
        assert exc.code == 0

    output = capsys.readouterr().out
    assert "start" in output
    assert "stop" in output
    assert "status" in output
    assert "config-generate" in output


def test_server_settings_default_backend_is_codex_http() -> None:
    args = server.parse_args(["serve"])
    settings = server.server_settings_from_args(args)

    assert settings.backend == "codex-http"


def test_parse_args_rejects_removed_chatgpt_http_backend() -> None:
    with pytest.raises(SystemExit) as exc_info:
        server.parse_args(["serve", "--backend", "chatgpt-http"])

    assert exc_info.value.code == 2


def test_config_generate_writes_template(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "config.toml"

    result = server._main(["config-generate", "--config", str(config_path)])

    assert result == 0
    text = config_path.read_text(encoding="utf-8")
    assert "[server]" in text
    assert 'backend = "codex-http"' in text
    assert "Wrote config:" in capsys.readouterr().out


def test_config_generate_refuses_existing_file(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("existing = true\n", encoding="utf-8")

    result = server._main(["config-generate", "--config", str(config_path)])

    assert result == 1
    assert config_path.read_text(encoding="utf-8") == "existing = true\n"
    assert "already exists" in capsys.readouterr().err


def test_config_generate_stdout_does_not_write_default_config(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config-home"))

    result = server._main(["config-generate", "--stdout"])

    assert result == 0
    assert "[server]" in capsys.readouterr().out
    assert not (tmp_path / "config-home").exists()


def test_server_settings_prefer_cli_over_env(
    tmp_path: Path, monkeypatch
) -> None:
    env_auth = tmp_path / "env-auth.json"
    cli_auth = tmp_path / "cli-auth.json"
    monkeypatch.setenv("OPENAI_VIA_CODEX_HOST", "127.0.0.3")
    monkeypatch.setenv("OPENAI_VIA_CODEX_PORT", "8124")
    monkeypatch.setenv("OPENAI_VIA_CODEX_AUTH_JSON", str(env_auth))

    args = server.parse_args(
        [
            "serve",
            "--host",
            "127.0.0.4",
            "--port",
            "8125",
            "--auth-json",
            str(cli_auth),
        ]
    )
    settings = server.server_settings_from_args(args)

    assert settings.host == "127.0.0.4"
    assert settings.port == 8125
    assert settings.auth_json == cli_auth.resolve()


def test_server_settings_select_app_server_backend(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_VIA_CODEX_BACKEND", "codex-app-server")
    monkeypatch.setenv("OPENAI_VIA_CODEX_CODEX_BIN", "/tmp/codex")

    args = server.parse_args(["serve"])
    settings = server.server_settings_from_args(args)

    assert settings.backend == "codex-app-server"
    assert settings.codex_bin == "/tmp/codex"


def test_server_settings_read_config_file(tmp_path: Path) -> None:
    auth_json = tmp_path / "auth.json"
    app_cwd = tmp_path / "app-server-cwd"
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
[server]
backend = "codex-app-server"
host = "127.0.0.9"
port = 9009
default_model = "gpt-5.4-mini"
timeout = 12.5

[codex]
auth_json = "{auth_json}"
backend_base_url = "https://example.test/codex"
client_version = "2.0.0"
codex_bin = "/tmp/codex-bin"
app_server_cwd = "{app_cwd}"
""",
        encoding="utf-8",
    )

    args = server.parse_args(["serve", "--config", str(config_path)])
    loaded_config = server.load_config_for_args(args)
    settings = server.server_settings_from_args(args, loaded_config)

    assert settings.backend == "codex-app-server"
    assert settings.host == "127.0.0.9"
    assert settings.port == 9009
    assert settings.default_model == "gpt-5.4-mini"
    assert settings.timeout == 12.5
    assert settings.auth_json == auth_json.resolve()
    assert settings.backend_base_url == "https://example.test/codex"
    assert settings.client_version == "2.0.0"
    assert settings.codex_bin == "/tmp/codex-bin"
    assert settings.app_server_cwd == app_cwd.resolve()


def test_server_settings_precedence_is_cli_then_env_then_config(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[server]
host = "127.0.0.9"
port = 9009
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_VIA_CODEX_PORT", "9010")

    args = server.parse_args(
        ["serve", "--config", str(config_path), "--host", "127.0.0.11"]
    )
    loaded_config = server.load_config_for_args(args)
    settings = server.server_settings_from_args(args, loaded_config)

    assert settings.host == "127.0.0.11"
    assert settings.port == 9010


def test_daemon_paths_read_config_file(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    pid_file = tmp_path / "explicit.pid"
    log_file = tmp_path / "explicit.log"
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
[server]
host = "127.0.0.9"
port = 9009

[daemon]
state_dir = "{state_dir}"
pid_file = "{pid_file}"
log_file = "{log_file}"
""",
        encoding="utf-8",
    )

    args = server.parse_args(["start", "--config", str(config_path)])
    loaded_config = server.load_config_for_args(args)
    settings = server.server_settings_from_args(args, loaded_config)
    paths = server.daemon_paths_from_args(args, settings, loaded_config)

    assert paths.state_dir == state_dir.resolve()
    assert paths.pid_file == pid_file.resolve()
    assert paths.log_file == log_file.resolve()


def test_stop_timeout_reads_config_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[daemon]
stop_timeout = 2.5
""",
        encoding="utf-8",
    )

    args = server.parse_args(["stop", "--config", str(config_path)])
    loaded_config = server.load_config_for_args(args)

    assert server.stop_timeout_from_args(args, loaded_config) == 2.5


def test_serve_command_uses_current_python_module_and_selected_settings(
    tmp_path: Path, monkeypatch
) -> None:
    auth_json = tmp_path / "auth.json"
    monkeypatch.setattr(server.sys, "executable", "/tmp/python")
    args = server.parse_args(
        [
            "start",
            "--host",
            "127.0.0.1",
            "--port",
            "8126",
            "--auth-json",
            str(auth_json),
            "--default-model",
            "gpt-5.4-mini",
            "--timeout",
            "12.5",
            "--backend",
            "codex-app-server",
            "--codex-bin",
            "/tmp/codex-bin",
        ]
    )
    settings = server.server_settings_from_args(args)

    command = server.serve_command(settings)

    assert command[:4] == [
        "/tmp/python",
        "-m",
        "openai_api_server_via_codex.server",
        "serve",
    ]
    assert "--auth-json" in command
    assert str(auth_json.resolve()) in command
    assert "--default-model" in command
    assert "gpt-5.4-mini" in command
    assert "--timeout" in command
    assert "12.5" in command
    assert "--backend" in command
    assert "codex-app-server" in command
    assert "--codex-bin" in command
    assert "/tmp/codex-bin" in command
