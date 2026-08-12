from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMAND = "openai-api-server-via-codex"


def environment_python(environment: Path) -> Path:
    directory = "Scripts" if os.name == "nt" else "bin"
    executable = "python.exe" if os.name == "nt" else "python"
    return environment / directory / executable


def environment_command(environment: Path) -> Path:
    directory = "Scripts" if os.name == "nt" else "bin"
    executable = f"{COMMAND}.exe" if os.name == "nt" else COMMAND
    return environment / directory / executable


def installed_binary(python: Path) -> Path:
    result = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            (
                "from openai_api_server_via_codex.launcher import bundled_binary; "
                "print(bundled_binary())"
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    binary = Path(result.stdout.strip()).resolve()
    if not binary.is_file() or binary.stat().st_size == 0:
        raise RuntimeError(f"installed Go binary is missing or empty: {binary}")
    return binary


def select_wheel(directory: Path, wheel_tag: str) -> Path:
    wheels = list(directory.glob(f"*-py3-none-{wheel_tag}.whl"))
    if len(wheels) != 1:
        names = sorted(path.name for path in wheels)
        raise RuntimeError(f"expected one {wheel_tag} wheel, found {names}")
    return wheels[0].resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--wheel-tag", required=True)
    args = parser.parse_args()
    wheel = select_wheel(args.wheel_dir.resolve(), args.wheel_tag)

    with tempfile.TemporaryDirectory(prefix="openai-via-codex-wheel-e2e-") as temporary:
        environment = Path(temporary) / "venv"
        subprocess.run(
            ["uv", "venv", "--python", sys.executable, str(environment)],
            cwd=ROOT,
            check=True,
        )
        python = environment_python(environment)
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), str(wheel)],
            cwd=ROOT,
            check=True,
        )
        command = environment_command(environment)
        if not command.is_file():
            raise RuntimeError(f"installed console command is missing: {command}")
        subprocess.run([str(command), "--version"], cwd=ROOT, check=True)
        missing_auth = Path(temporary) / "missing-auth.json"
        preflight = subprocess.run(
            [
                str(command),
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--auth-json",
                str(missing_auth),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        preflight_output = preflight.stdout + preflight.stderr
        if (
            preflight.returncode == 0
            or "authentication preflight failed" not in preflight_output
        ):
            raise RuntimeError(
                "installed console command did not propagate preflight failure: "
                f"returncode={preflight.returncode} output={preflight_output!r}"
            )

        child_env = {
            **os.environ,
            "OPENAI_VIA_CODEX_E2E_EXECUTABLE": str(installed_binary(python)),
        }
        subprocess.run(
            [
                "go",
                "test",
                "./test/e2e",
                "-run",
                "TestGoBinaryForegroundE2E",
                "-v",
                "-count=1",
            ],
            cwd=ROOT,
            env=child_env,
            check=True,
        )


if __name__ == "__main__":
    main()
