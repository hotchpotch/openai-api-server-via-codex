import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def go_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    binary = tmp_path_factory.mktemp("go-bin") / "openai-api-server-via-codex"
    subprocess.run(
        ["go", "build", "-o", str(binary), "./cmd/openai-api-server-via-codex"],
        check=True,
    )
    return binary
