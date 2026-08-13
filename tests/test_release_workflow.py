from pathlib import Path


def test_release_workflow_publishes_wheels_and_standalone_archives() -> None:
    workflow = Path(".github/workflows/release.yml").read_text(encoding="utf-8")

    assert "scripts/build-release-archives.py" in workflow
    assert "scripts/render-homebrew-formula.py" in workflow
    assert "steps.version.outputs.prerelease == 'false'" in workflow
    assert "release/openai-api-server-via-codex.rb" in workflow
    assert "sha256sum --check checksums.txt" in workflow
    assert "name: platform-wheel-distributions" in workflow
    assert "name: standalone-go-distributions" in workflow
    assert 'gh release upload "${GITHUB_REF_NAME}" release-assets/* --clobber' in workflow
