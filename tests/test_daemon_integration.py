from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
from pathlib import Path

import httpx


def test_cli_start_status_stop_runs_background_server(tmp_path: Path) -> None:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())

    try:
        start = subprocess.run(
            [
                sys.executable,
                "-m",
                "openai_api_server_via_codex.server",
                "start",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--state-dir",
                str(tmp_path),
            ],
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        assert start.returncode == 0, start.stderr + start.stdout
        assert "PID file:" in start.stdout
        assert "Log file:" in start.stdout
        asyncio.run(_wait_for_server(base_url))

        status = subprocess.run(
            [
                sys.executable,
                "-m",
                "openai_api_server_via_codex.server",
                "status",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--state-dir",
                str(tmp_path),
            ],
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        assert status.returncode == 0, status.stderr + status.stdout
        assert "running" in status.stdout
    finally:
        stop = subprocess.run(
            [
                sys.executable,
                "-m",
                "openai_api_server_via_codex.server",
                "stop",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--state-dir",
                str(tmp_path),
            ],
            env=env,
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
        assert stop.returncode == 0, stop.stderr + stop.stdout


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _wait_for_server(base_url: str) -> None:
    async with httpx.AsyncClient() as client:
        for _ in range(100):
            try:
                response = await client.get(f"{base_url}/healthz")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.1)
    raise AssertionError("background server did not become ready")
