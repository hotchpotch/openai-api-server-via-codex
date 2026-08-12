from __future__ import annotations

import os
from pathlib import Path

import pytest

from openai_api_server_via_codex import launcher


def test_launcher_execs_bundled_go_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "openai-api-server-via-codex"
    binary.write_bytes(b"binary")
    binary.chmod(0o755)
    calls: list[tuple[Path, list[str]]] = []
    monkeypatch.setattr(launcher, "bundled_binary", lambda: binary)

    class Executed(Exception):
        pass

    def fake_launch(path: Path) -> None:
        calls.append((path, [str(path), "--version"]))
        raise Executed

    monkeypatch.setattr(launcher, "_launch_go", fake_launch)
    monkeypatch.setattr(launcher.sys, "argv", ["command", "--version"])

    with pytest.raises(Executed):
        launcher.main()

    assert calls == [(binary, [str(binary), "--version"])]


def test_windows_launcher_waits_for_go_and_propagates_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "openai-api-server-via-codex.exe"
    calls: list[list[str]] = []
    monkeypatch.setattr(launcher.sys, "argv", ["command", "serve", "--port", "0"])

    def fake_call(arguments: list[str]) -> int:
        calls.append(arguments)
        return 23

    monkeypatch.setattr(launcher.subprocess, "call", fake_call)

    with pytest.raises(SystemExit) as raised:
        launcher._launch_go(binary, "nt")

    assert raised.value.code == 23
    assert calls == [[str(binary), "serve", "--port", "0"]]


def test_launcher_rejects_install_without_bundled_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launcher, "bundled_binary", lambda: tmp_path / "missing")

    with pytest.raises(SystemExit, match="supported platform wheel"):
        launcher.main()


def test_launcher_uses_windows_executable_name() -> None:
    assert launcher.bundled_binary("nt").name == "openai-api-server-via-codex.exe"
    assert launcher.bundled_binary("posix").name == "openai-api-server-via-codex"


@pytest.mark.skipif(os.name == "nt", reason="Windows does not use POSIX execute bits")
def test_launcher_rejects_non_executable_bundled_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "openai-api-server-via-codex"
    binary.write_bytes(b"binary")
    binary.chmod(0o644)
    monkeypatch.setattr(launcher, "bundled_binary", lambda: binary)

    with pytest.raises(SystemExit, match="not executable"):
        launcher.main()


def test_launcher_reports_exec_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "openai-api-server-via-codex"
    binary.write_bytes(b"binary")
    binary.chmod(0o755)
    monkeypatch.setattr(launcher, "bundled_binary", lambda: binary)
    monkeypatch.setattr(
        launcher,
        "_launch_go",
        lambda path: (_ for _ in ()).throw(OSError("exec format error")),
    )

    with pytest.raises(SystemExit, match="Failed to execute.*exec format error"):
        launcher.main()
