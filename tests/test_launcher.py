from __future__ import annotations

from pathlib import Path

import pytest

from openai_api_server_via_codex import launcher


def test_launcher_execs_bundled_go_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "openai-api-server-via-codex"
    binary.write_bytes(b"binary")
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


def test_launcher_can_force_python_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setenv(launcher.RUNTIME_ENV, "python")
    monkeypatch.setattr(
        "openai_api_server_via_codex.server.main", lambda: calls.append("python")
    )

    launcher.main()

    assert calls == ["python"]


def test_launcher_rejects_explicit_go_without_bundled_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(launcher.RUNTIME_ENV, "go")
    monkeypatch.setattr(launcher, "bundled_binary", lambda: tmp_path / "missing")

    with pytest.raises(SystemExit, match="not bundled"):
        launcher.main()
