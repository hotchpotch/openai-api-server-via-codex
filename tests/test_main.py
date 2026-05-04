import subprocess
import sys
from pathlib import Path


def test_main_prints_project_greeting():
    project_root = Path(__file__).parents[1]

    result = subprocess.run(
        [sys.executable, str(project_root / "main.py")],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == "Hello from openai-api-server-via-codex!\n"
