from __future__ import annotations

import sys
from importlib import util
from pathlib import Path
from types import ModuleType

import pytest


def _load_script(path: Path, name: str) -> ModuleType:
    spec = util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


render_homebrew_formula = _load_script(
    Path("scripts/render-homebrew-formula.py"), "render_homebrew_formula"
)


def test_checked_in_formula_matches_stable_tag_renderer() -> None:
    digest = "e9d0f078b4430631dd18e60cb37700cd64bb61c9c983d20300550125bb019538"

    formula = render_homebrew_formula.render_formula("0.2.0", digest)

    assert formula == Path(
        "packaging/homebrew/openai-api-server-via-codex.rb"
    ).read_text(encoding="utf-8")
    assert 'depends_on "go" => :build' in formula
    assert "*std_go_args(ldflags:)" in formula
    assert "config-generate --stdout" in formula
    assert "bottle do" not in formula


@pytest.mark.parametrize("tag", ["v0.2.0b1", "0.2", "main", "v1.2.3-rc1"])
def test_renderer_rejects_non_stable_tags(tag: str) -> None:
    with pytest.raises(ValueError, match="stable vX.Y.Z"):
        render_homebrew_formula.stable_version(tag)


def test_renderer_normalizes_stable_tag() -> None:
    assert render_homebrew_formula.stable_version("v1.2.3") == "1.2.3"
    assert render_homebrew_formula.stable_version("1.2.3") == "1.2.3"


def test_renderer_rejects_invalid_digest() -> None:
    with pytest.raises(ValueError, match="invalid SHA-256"):
        render_homebrew_formula.render_formula("1.2.3", "not-a-digest")
