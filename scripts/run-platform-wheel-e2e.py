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

        child_env = {
            **os.environ,
            "OPENAI_VIA_CODEX_E2E_EXECUTABLE": str(command),
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
