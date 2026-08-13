#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from urllib import request

PROJECT = "openai-api-server-via-codex"
REPOSITORY = f"hotchpotch/{PROJECT}"
STABLE_TAG = re.compile(r"^v?(\d+\.\d+\.\d+)$")


def stable_version(tag: str) -> str:
    match = STABLE_TAG.fullmatch(tag)
    if match is None:
        raise ValueError(
            f"Homebrew core Formulae require a stable vX.Y.Z tag, got {tag!r}"
        )
    return match.group(1)


def source_url(version: str) -> str:
    return f"https://github.com/{REPOSITORY}/archive/refs/tags/v{version}.tar.gz"


def download_sha256(url: str) -> str:
    digest = hashlib.sha256()
    with request.urlopen(url) as response:
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def render_formula(version: str, sha256: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise ValueError(f"invalid SHA-256 digest: {sha256!r}")
    return f'''class OpenaiApiServerViaCodex < Formula
  desc "OpenAI-compatible local API server backed by Codex credentials"
  homepage "https://github.com/{REPOSITORY}"
  url "{source_url(version)}"
  sha256 "{sha256}"
  license "Apache-2.0"

  livecheck do
    url :stable
    regex(/^v?(\\d+(?:\\.\\d+)+)$/i)
  end

  depends_on "go" => :build

  def install
    ldflags = "-s -w -X main.version=#{{version}}"
    system "go", "build", *std_go_args(ldflags:), "./cmd/{PROJECT}"
  end

  test do
    assert_equal version.to_s, shell_output("#{{bin}}/{PROJECT} --version").strip
    config = shell_output("#{{bin}}/{PROJECT} config-generate --stdout")
    assert_match "port = 18080", config
  end
end
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a homebrew/core-ready source Formula from an immutable release tag."
        )
    )
    parser.add_argument("tag", help="stable release tag, for example v0.2.0")
    parser.add_argument("--output", type=Path, help="write the Formula to this path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    version = stable_version(args.tag)
    formula = render_formula(version, download_sha256(source_url(version)))
    if args.output is None:
        print(formula, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(formula, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
