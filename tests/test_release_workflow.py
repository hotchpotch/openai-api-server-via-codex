from pathlib import Path


def test_release_workflow_publishes_wheels_and_standalone_archives() -> None:
    workflow = Path(".github/workflows/release.yml").read_text(encoding="utf-8")

    assert "scripts/build-release-archives.py" in workflow
    assert "sha256sum --check checksums.txt" in workflow
    assert "name: platform-wheel-distributions" in workflow
    assert "name: standalone-go-distributions" in workflow
    assert 'gh release upload "${GITHUB_REF_NAME}" release-assets/* --clobber' in workflow


def test_release_workflow_uses_node24_actions() -> None:
    workflow = Path(".github/workflows/release.yml").read_text(encoding="utf-8")

    node24_actions = {
        "actions/upload-artifact@v7",
        "actions/download-artifact@v8",
        "docker/setup-qemu-action@96fe6ef7f33517b61c61be40b68a1882f3264fb8",
        "docker/setup-buildx-action@bb05f3f5519dd87d3ba754cc423b652a5edd6d2c",
        "docker/login-action@dbcb813823bdd20940b903addbd779551569679f",
        "docker/metadata-action@dc802804100637a589fabce1cb79ff13a1411302",
        "docker/build-push-action@53b7df96c91f9c12dcc8a07bcb9ccacbed38856a",
    }
    action_prefixes = tuple(action.split("@", 1)[0] for action in node24_actions)
    release_actions = {
        line.strip().removeprefix("uses: ").split(" #", 1)[0]
        for line in workflow.splitlines()
        if line.strip().startswith("uses: ")
        and line.strip().removeprefix("uses: ").startswith(action_prefixes)
    }

    assert release_actions == node24_actions
