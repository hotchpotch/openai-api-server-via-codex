import re
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

    # Minimum major versions that run on Node 24. Exact commits are pinned by
    # .github/workflows/actions.lock and updated by Dependabot.
    node24_minimum_majors = {
        "actions/upload-artifact": 7,
        "actions/download-artifact": 8,
        "docker/setup-qemu-action": 4,
        "docker/setup-buildx-action": 4,
        "docker/login-action": 4,
        "docker/metadata-action": 6,
        "docker/build-push-action": 7,
    }
    release_refs: dict[str, set[str]] = {}
    for line in workflow.splitlines():
        stripped = line.strip()
        if not stripped.startswith("uses: "):
            continue
        action, _, ref = stripped.removeprefix("uses: ").partition("@")
        if action in node24_minimum_majors:
            release_refs.setdefault(action, set()).add(ref)

    assert release_refs.keys() == node24_minimum_majors.keys()
    for action, refs in release_refs.items():
        for ref in refs:
            match = re.fullmatch(r"v(\d+)\.\d+\.\d+", ref)
            assert match, f"{action}@{ref} should use a full semver tag"
            assert int(match.group(1)) >= node24_minimum_majors[action], f"{action}@{ref}"
