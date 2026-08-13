#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
import tarfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Target:
    goos: str
    goarch: str
    wheel_tag: str
    executable: str = "openai-api-server-via-codex"

    @property
    def archive_suffix(self) -> str:
        return ".zip" if self.goos == "windows" else ".tar.gz"


TARGETS = (
    Target("linux", "amd64", "manylinux_2_17_x86_64"),
    Target("linux", "arm64", "manylinux_2_17_aarch64"),
    Target("darwin", "amd64", "macosx_10_15_x86_64"),
    Target("darwin", "arm64", "macosx_11_0_arm64"),
    Target("windows", "amd64", "win_amd64", "openai-api-server-via-codex.exe"),
    Target("windows", "arm64", "win_arm64", "openai-api-server-via-codex.exe"),
)
PROJECT = "openai-api-server-via-codex"
PACKAGE_BINARY_DIR = "openai_api_server_via_codex/bin"
ROOT = Path(__file__).resolve().parents[1]


def read_binary(wheel_dir: Path, target: Target, version: str) -> bytes:
    pattern = f"openai_api_server_via_codex-{version}-py3-none-{target.wheel_tag}.whl"
    matches = list(wheel_dir.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one wheel matching {pattern!r}, found "
            f"{[path.name for path in matches]}"
        )
    member = f"{PACKAGE_BINARY_DIR}/{target.executable}"
    with zipfile.ZipFile(matches[0]) as wheel:
        try:
            binary = wheel.read(member)
        except KeyError as error:
            raise RuntimeError(f"{matches[0].name} does not contain {member}") from error
    if not binary:
        raise RuntimeError(f"{matches[0].name} contains an empty {member}")
    return binary


def archive_name(target: Target, version: str) -> str:
    return f"{PROJECT}_{version}_{target.goos}_{target.goarch}{target.archive_suffix}"


def tar_info(name: str, data: bytes, mode: int, epoch: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    info.mtime = epoch
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def write_tar_gz(output: Path, files: list[tuple[str, bytes, int]], epoch: int) -> None:
    with (
        output.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as compressed,
        tarfile.open(
            fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
        ) as archive,
    ):
        for name, data, mode in files:
            archive.addfile(tar_info(name, data, mode, epoch), io.BytesIO(data))


def zip_info(name: str, mode: int, epoch: int) -> zipfile.ZipInfo:
    # ZIP timestamps cannot represent dates before 1980-01-01.
    timestamp = max(epoch, 315532800)
    date_time = tuple(time.gmtime(timestamp)[:6])
    info = zipfile.ZipInfo(name, date_time=date_time)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (mode & 0xFFFF) << 16
    return info


def write_zip(output: Path, files: list[tuple[str, bytes, int]], epoch: int) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data, mode in files:
            archive.writestr(zip_info(name, mode, epoch), data, compresslevel=9)


def build_archive(
    wheel_dir: Path,
    output_dir: Path,
    target: Target,
    version: str,
    epoch: int,
) -> Path:
    files = [
        (target.executable, read_binary(wheel_dir, target, version), 0o755),
        ("LICENSE", ROOT.joinpath("LICENSE").read_bytes(), 0o644),
        ("README.md", ROOT.joinpath("README.md").read_bytes(), 0o644),
    ]
    output = output_dir / archive_name(target, version)
    if target.goos == "windows":
        write_zip(output, files, epoch)
    else:
        write_tar_gz(output, files, epoch)
    return output


def write_checksums(outputs: list[Path], output_dir: Path) -> Path:
    checksum_file = output_dir / "checksums.txt"
    lines = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}" for path in outputs]
    checksum_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksum_file


def validate_output_set(output_dir: Path, expected: list[Path]) -> None:
    expected_paths = {path.resolve() for path in expected}
    actual_paths = {path.resolve() for path in output_dir.iterdir() if path.is_file()}
    if expected_paths != actual_paths:
        missing = sorted(path.name for path in expected_paths - actual_paths)
        unexpected = sorted(path.name for path in actual_paths - expected_paths)
        raise RuntimeError(
            f"release archive directory is incomplete: missing={missing}, unexpected={unexpected}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package the Go executables embedded in platform wheels for GitHub Releases."
    )
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--source-date-epoch",
        type=int,
        default=int(os.environ.get("SOURCE_DATE_EPOCH", "0")),
        help="archive timestamp (defaults to SOURCE_DATE_EPOCH or 0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wheel_dir = args.wheel_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        build_archive(
            wheel_dir,
            output_dir,
            target,
            args.version,
            args.source_date_epoch,
        )
        for target in TARGETS
    ]
    checksum_file = write_checksums(outputs, output_dir)
    validate_output_set(output_dir, [*outputs, checksum_file])
    for output in [*outputs, checksum_file]:
        print(output)


if __name__ == "__main__":
    main()
