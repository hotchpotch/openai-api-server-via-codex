#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import os
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Target:
    goos: str
    goarch: str
    wheel_tag: str
    executable: str = "openai-api-server-via-codex"


TARGETS = (
    Target("linux", "amd64", "manylinux_2_17_x86_64"),
    Target("linux", "arm64", "manylinux_2_17_aarch64"),
    Target("darwin", "amd64", "macosx_10_15_x86_64"),
    Target("darwin", "arm64", "macosx_11_0_arm64"),
    Target("windows", "amd64", "win_amd64", "openai-api-server-via-codex.exe"),
    Target("windows", "arm64", "win_arm64", "openai-api-server-via-codex.exe"),
)
LAUNCHER_MODULES = {"__init__.py", "__main__.py", "launcher.py"}


def digest(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"sha256={encoded.decode()}", len(data)


def write_record(root: Path, dist_info: Path) -> None:
    record = dist_info / "RECORD"
    rows: list[tuple[str, str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == record:
            continue
        hashed, size = digest(path)
        rows.append((path.relative_to(root).as_posix(), hashed, str(size)))
    rows.append((record.relative_to(root).as_posix(), "", ""))
    with record.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file, lineterminator="\n").writerows(rows)


def archive_wheel(root: Path, output: Path) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as wheel:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo.from_file(path, relative)
            info.compress_type = zipfile.ZIP_DEFLATED
            wheel.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def validate_launcher_package(root: Path) -> None:
    package = root / "openai_api_server_via_codex"
    modules = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*.py")
        if path.is_file()
    }
    if modules != LAUNCHER_MODULES:
        unexpected = sorted(modules - LAUNCHER_MODULES)
        missing = sorted(LAUNCHER_MODULES - modules)
        raise RuntimeError(
            "base wheel must contain only the Python launcher modules; "
            f"unexpected={unexpected}, missing={missing}. "
            "Remove stale build/ output and rebuild the base wheel."
        )


def build_binary(target: Target, output: Path, version: str) -> None:
    env = {**os.environ, "CGO_ENABLED": "0", "GOOS": target.goos, "GOARCH": target.goarch}
    subprocess.run(
        [
            "go",
            "build",
            "-trimpath",
            "-ldflags",
            f"-s -w -X main.version={version}",
            "-o",
            str(output),
            "./cmd/openai-api-server-via-codex",
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    output.chmod(0o755)


def platform_wheel(base_wheel: Path, output_dir: Path, target: Target, version: str) -> Path:
    with tempfile.TemporaryDirectory(prefix="openai-via-codex-wheel-") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(base_wheel) as source:
            source.extractall(root)
        validate_launcher_package(root)
        dist_infos = list(root.glob("*.dist-info"))
        if len(dist_infos) != 1:
            raise RuntimeError(f"expected one dist-info directory in {base_wheel}")
        dist_info = dist_infos[0]
        wheel_metadata = dist_info / "WHEEL"
        lines = [
            line
            for line in wheel_metadata.read_text(encoding="utf-8").splitlines()
            if not line.startswith(("Root-Is-Purelib:", "Tag:"))
        ]
        lines.extend(["Root-Is-Purelib: false", f"Tag: py3-none-{target.wheel_tag}"])
        wheel_metadata.write_text("\n".join(lines) + "\n", encoding="utf-8")

        metadata = dist_info / "METADATA"
        metadata_lines = [
            line
            for line in metadata.read_text(encoding="utf-8").splitlines()
            if not line.startswith("Requires-Dist:")
        ]
        metadata.write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")

        binary = root / "openai_api_server_via_codex" / "bin" / target.executable
        binary.parent.mkdir(parents=True, exist_ok=True)
        build_binary(target, binary, version)
        write_record(root, dist_info)

        prefix = base_wheel.name.removesuffix("-py3-none-any.whl")
        output = output_dir / f"{prefix}-py3-none-{target.wheel_tag}.whl"
        archive_wheel(root, output)
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True, help="base py3-none-any wheel")
    parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    parser.add_argument("--version", required=True)
    parser.add_argument("--keep-base", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_wheel = args.wheel.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        platform_wheel(base_wheel, output_dir, target, args.version)
        for target in TARGETS
    ]
    if not args.keep_base and base_wheel.parent == output_dir:
        base_wheel.unlink()
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
