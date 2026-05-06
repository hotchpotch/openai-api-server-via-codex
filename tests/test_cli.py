from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from openai_api_server_via_codex import server
from openai_api_server_via_codex.daemon import DaemonStatus, StopResult


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


def test_parse_args_rejects_removed_backend_option() -> None:
    with pytest.raises(SystemExit) as exc_info:
        server.parse_args(["serve", "--backend", "codex-http"])

    assert exc_info.value.code == 2


def test_parse_args_rejects_negative_max_stored_items() -> None:
    with pytest.raises(SystemExit) as exc_info:
        server.parse_args(["serve", "--max-stored-items", "-1"])

    assert exc_info.value.code == 2


def test_parse_args_accepts_verbose_for_stop_and_status() -> None:
    stop_args = server.parse_args(["stop", "--host", "0.0.0.0", "--verbose"])
    status_args = server.parse_args(["status", "--host", "0.0.0.0", "--verbose"])

    assert stop_args.command == "stop"
    assert stop_args.verbose is True
    assert status_args.command == "status"
    assert status_args.verbose is True


def test_config_generate_writes_template(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "config.toml"

    result = server._main(["config-generate", "--config", str(config_path)])

    assert result == 0
    text = config_path.read_text(encoding="utf-8")
    assert "[server]" in text
    assert "backend =" not in text
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


def test_server_settings_use_18080_as_default_port(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_VIA_CODEX_PORT", raising=False)
    monkeypatch.delenv("OPENAI_VIA_CODEX_TIMEOUT", raising=False)
    args = server.parse_args(["serve"])

    settings = server.server_settings_from_args(args)

    assert settings.port == 18080
    assert settings.timeout == 300.0


def test_server_settings_read_config_file(tmp_path: Path) -> None:
    auth_json = tmp_path / "auth.json"
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
[server]
host = "127.0.0.9"
port = 9009
default_model = "gpt-5.4-mini"
timeout = 12.5
verbose = true
max_stored_items = 123
max_concurrent_requests = 4
api_key = "config-secret"

[codex]
auth_json = "{auth_json}"
backend_base_url = "https://example.test/codex"
client_version = "2.0.0"
""",
        encoding="utf-8",
    )

    args = server.parse_args(["serve", "--config", str(config_path)])
    loaded_config = server.load_config_for_args(args)
    settings = server.server_settings_from_args(args, loaded_config)

    assert settings.host == "127.0.0.9"
    assert settings.port == 9009
    assert settings.default_model == "gpt-5.4-mini"
    assert settings.timeout == 12.5
    assert settings.verbose is True
    assert settings.max_stored_items == 123
    assert settings.max_concurrent_requests == 4
    assert settings.api_key == "config-secret"
    assert settings.auth_json == auth_json.resolve()
    assert settings.backend_base_url == "https://example.test/codex"
    assert settings.client_version == "2.0.0"


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


def test_server_settings_resolve_max_concurrent_requests_precedence(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[server]
max_concurrent_requests = 3
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_VIA_CODEX_MAX_CONCURRENT_REQUESTS", "4")

    env_args = server.parse_args(["serve", "--config", str(config_path)])
    env_settings = server.server_settings_from_args(
        env_args, server.load_config_for_args(env_args)
    )

    cli_args = server.parse_args(
        ["serve", "--config", str(config_path), "--max-concurrent-requests", "5"]
    )
    cli_settings = server.server_settings_from_args(
        cli_args, server.load_config_for_args(cli_args)
    )

    assert env_settings.max_concurrent_requests == 4
    assert cli_settings.max_concurrent_requests == 5


def test_server_settings_resolve_api_key_precedence(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[server]
api_key = "config-secret"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_VIA_CODEX_API_KEY", "env-secret")

    env_args = server.parse_args(["serve", "--config", str(config_path)])
    env_settings = server.server_settings_from_args(
        env_args, server.load_config_for_args(env_args)
    )

    cli_args = server.parse_args(
        ["serve", "--config", str(config_path), "--api-key", "cli-secret"]
    )
    cli_settings = server.server_settings_from_args(
        cli_args, server.load_config_for_args(cli_args)
    )

    assert env_settings.api_key == "env-secret"
    assert cli_settings.api_key == "cli-secret"


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


def test_status_discovers_single_pid_file_when_host_is_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    discovered_pid_file = state_dir / "server-0.0.0.0-18080.pid"
    discovered_pid_file.write_text("123\n")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
[daemon]
state_dir = "{state_dir}"
""",
        encoding="utf-8",
    )
    captured_paths = []

    def fake_daemon_status(paths):
        captured_paths.append(paths)
        return DaemonStatus(
            state="running",
            pid=123,
            pid_file=paths.pid_file,
            log_file=paths.log_file,
        )

    monkeypatch.setattr(server, "daemon_status", fake_daemon_status)

    result = server._main(["status", "--config", str(config_path)])

    assert result == 0
    assert captured_paths[0].pid_file == discovered_pid_file.resolve()
    assert captured_paths[0].log_file == discovered_pid_file.with_suffix(".log").resolve()
    assert "running: PID 123" in capsys.readouterr().out


def test_stop_discovers_single_pid_file_when_host_is_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    discovered_pid_file = state_dir / "server-0.0.0.0-18080.pid"
    discovered_pid_file.write_text("123\n")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
[daemon]
state_dir = "{state_dir}"
""",
        encoding="utf-8",
    )
    captured_paths = []

    def fake_stop_background(paths, *, timeout: float):
        captured_paths.append(paths)
        return StopResult(state="stopped", pid=123, pid_file=paths.pid_file)

    monkeypatch.setattr(server, "stop_background", fake_stop_background)

    result = server._main(["stop", "--config", str(config_path)])

    assert result == 0
    assert captured_paths[0].pid_file == discovered_pid_file.resolve()
    assert "Stopped PID 123" in capsys.readouterr().out


def test_status_reports_ambiguous_pid_files_when_host_is_omitted(
    tmp_path: Path, capsys
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "server-0.0.0.0-18080.pid").write_text("123\n")
    (state_dir / "server-127.0.0.2-18080.pid").write_text("456\n")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
[daemon]
state_dir = "{state_dir}"
""",
        encoding="utf-8",
    )

    result = server._main(["status", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert result == 1
    assert "Multiple PID files match port 18080" in captured.err
    assert "server-0.0.0.0-18080.pid" in captured.err
    assert "server-127.0.0.2-18080.pid" in captured.err


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
            "--max-stored-items",
            "222",
            "--max-concurrent-requests",
            "6",
            "--api-key",
            "local-secret",
        ]
    )
    settings = server.server_settings_from_args(args)

    command = server.serve_command(settings)
    env = server.serve_env(settings)

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
    assert "--max-stored-items" in command
    assert "222" in command
    assert "--max-concurrent-requests" in command
    assert "6" in command
    assert "--api-key" not in command
    assert "local-secret" not in command
    assert env is not None
    assert env["OPENAI_VIA_CODEX_API_KEY"] == "local-secret"


def test_serve_command_includes_verbose_flag_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(server.sys, "executable", "/tmp/python")
    args = server.parse_args(["start", "--verbose"])
    settings = server.server_settings_from_args(args)

    command = server.serve_command(settings)

    assert "--verbose" in command


def test_serve_uses_debug_log_level_when_verbose(monkeypatch) -> None:
    run_calls: list[dict[str, Any]] = []

    def fake_run(*args, **kwargs) -> None:
        run_calls.append(kwargs)

    monkeypatch.setattr(server.uvicorn, "run", fake_run)

    result = server._main(["serve", "--verbose"])

    assert result == 0
    assert run_calls[0]["log_level"] == "debug"
    assert run_calls[0]["app"].state.verbose is True


def test_create_app_uses_configured_max_stored_items() -> None:
    app = server.create_app(max_stored_items=7)

    assert app.state.max_stored_items == 7
    assert app.state.response_store.max_entries == 7
    assert app.state.chat_completion_store.max_entries == 7


def test_create_app_uses_configured_max_concurrent_requests() -> None:
    app = server.create_app(max_concurrent_requests=7)

    assert app.state.max_concurrent_requests == 7
    assert app.state.backend_semaphore is not None
