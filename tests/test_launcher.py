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

    def fake_exec(path: Path) -> None:
        calls.append((path, [str(path), "--version"]))
        raise Executed

    monkeypatch.setattr(launcher, "_exec_go", fake_exec)
    monkeypatch.setattr(launcher.sys, "argv", ["command", "--version"])

    with pytest.raises(Executed):
        launcher.main()

    assert calls == [(binary, [str(binary), "--version"])]


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
        "_exec_go",
        lambda path: (_ for _ in ()).throw(OSError("exec format error")),
    )

    with pytest.raises(SystemExit, match="Failed to execute.*exec format error"):
        launcher.main()
