from __future__ import annotations

import signal
from pathlib import Path
from typing import Any

import pytest

from openai_api_server_via_codex import daemon


def test_resolve_daemon_paths_uses_uvx_safe_state_dir(tmp_path: Path) -> None:
    paths = daemon.resolve_daemon_paths(
        host="127.0.0.1",
        port=8123,
        state_dir=tmp_path,
    )

    assert paths.state_dir == tmp_path.resolve()
    assert paths.pid_file == tmp_path.resolve() / "server-127.0.0.1-8123.pid"
    assert paths.log_file == tmp_path.resolve() / "server-127.0.0.1-8123.log"


def test_resolve_daemon_paths_defaults_under_xdg_config_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))

    paths = daemon.resolve_daemon_paths(host="127.0.0.1", port=8123)

    assert paths.state_dir == (
        tmp_path / "xdg-config" / "openai-api-server-via-codex" / "run"
    ).resolve()
    assert paths.pid_file.parent == paths.state_dir
    assert paths.log_file.parent == paths.state_dir


def test_find_daemon_pid_files_lists_matching_port(tmp_path: Path) -> None:
    (tmp_path / "server-0.0.0.0-18080.pid").write_text("123\n")
    (tmp_path / "server-127.0.0.1-18080.pid").write_text("456\n")
    (tmp_path / "server-0.0.0.0-18081.pid").write_text("789\n")

    pid_files = daemon.find_daemon_pid_files(state_dir=tmp_path, port=18080)

    assert pid_files == [
        (tmp_path / "server-0.0.0.0-18080.pid").resolve(),
        (tmp_path / "server-127.0.0.1-18080.pid").resolve(),
    ]


def test_start_background_refuses_live_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = daemon.resolve_daemon_paths(
        host="127.0.0.1",
        port=8123,
        state_dir=tmp_path,
    )
    paths.pid_file.write_text("123\n")
    monkeypatch.setattr(daemon, "is_pid_alive", lambda pid: True)

    with pytest.raises(daemon.DaemonError, match="Already running"):
        daemon.start_background(["python", "-m", "fake"], paths)


def test_start_background_removes_stale_pid_and_writes_new_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = daemon.resolve_daemon_paths(
        host="127.0.0.1",
        port=8123,
        state_dir=tmp_path,
    )
    paths.pid_file.write_text("123\n")
    popen_calls: list[dict[str, Any]] = []

    class FakeProcess:
        pid = 456

    def fake_popen(command: list[str], **kwargs: Any) -> FakeProcess:
        popen_calls.append({"command": command, **kwargs})
        return FakeProcess()

    monkeypatch.setattr(daemon, "is_pid_alive", lambda pid: False)
    monkeypatch.setattr(daemon.subprocess, "Popen", fake_popen)

    pid = daemon.start_background(["python", "-m", "fake"], paths)

    assert pid == 456
    assert paths.pid_file.read_text() == "456\n"
    assert popen_calls[0]["command"] == ["python", "-m", "fake"]
    assert popen_calls[0]["start_new_session"] is True
    assert paths.log_file.exists()


def test_run_supervised_restarts_unexpected_child_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    popen_calls: list[list[str]] = []
    sleeps: list[float] = []
    return_codes = [1, 0]

    class FakeProcess:
        def __init__(self, returncode: int) -> None:
            self.pid = 1000 + len(popen_calls)
            self.returncode = returncode

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode

        def terminate(self) -> None:
            raise AssertionError("unexpected terminate")

        def kill(self) -> None:
            raise AssertionError("unexpected kill")

    def fake_popen(command: list[str], **kwargs: Any) -> FakeProcess:
        popen_calls.append(command)
        return FakeProcess(return_codes.pop(0))

    monkeypatch.setattr(daemon.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(daemon.time, "sleep", lambda delay: sleeps.append(delay))

    result = daemon.run_supervised(
        ["python", "-m", "fake"],
        restart_delay=0.25,
        restart_limit=1,
    )

    assert result == 0
    assert popen_calls == [["python", "-m", "fake"], ["python", "-m", "fake"]]
    assert sleeps == [0.25]


def test_run_supervised_reports_restart_limit_without_restarting(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    popen_calls: list[list[str]] = []
    sleeps: list[float] = []

    class FakeProcess:
        pid = 1001

        def wait(self, timeout: float | None = None) -> int:
            return 7

    def fake_popen(command: list[str], **kwargs: Any) -> FakeProcess:
        popen_calls.append(command)
        return FakeProcess()

    monkeypatch.setattr(daemon.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(daemon.time, "sleep", lambda delay: sleeps.append(delay))

    result = daemon.run_supervised(
        ["python", "-m", "fake"],
        restart_delay=0.25,
        restart_limit=0,
    )

    output = capsys.readouterr().out
    assert result == 7
    assert popen_calls == [["python", "-m", "fake"]]
    assert sleeps == []
    assert "restart limit reached" in output
    assert "restarting" not in output


def test_stop_background_terms_live_pid_and_removes_pid_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = daemon.resolve_daemon_paths(
        host="127.0.0.1",
        port=8123,
        state_dir=tmp_path,
    )
    paths.pid_file.write_text("456\n")
    sent_signals: list[tuple[int, int]] = []
    monkeypatch.setattr(daemon, "is_pid_alive", lambda pid: True)
    monkeypatch.setattr(
        daemon.os,
        "kill",
        lambda pid, sig: sent_signals.append((pid, sig)),
    )
    monkeypatch.setattr(daemon, "_wait_for_pid_exit", lambda pid, timeout: True)

    result = daemon.stop_background(paths, timeout=0.1)

    assert result.state == "stopped"
    assert result.pid == 456
    assert sent_signals == [(456, signal.SIGTERM)]
    assert not paths.pid_file.exists()


def test_status_reports_stale_pid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = daemon.resolve_daemon_paths(
        host="127.0.0.1",
        port=8123,
        state_dir=tmp_path,
    )
    paths.pid_file.write_text("789\n")
    monkeypatch.setattr(daemon, "is_pid_alive", lambda pid: False)

    status = daemon.daemon_status(paths)

    assert status.state == "stale"
    assert status.pid == 789
