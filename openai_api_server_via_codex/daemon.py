from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import default_daemon_state_dir


STATE_DIR_ENV = "OPENAI_VIA_CODEX_STATE_DIR"
PID_FILE_ENV = "OPENAI_VIA_CODEX_PID_FILE"
LOG_FILE_ENV = "OPENAI_VIA_CODEX_LOG_FILE"


class DaemonError(Exception):
    """Raised when the background server cannot be managed safely."""


@dataclass(frozen=True)
class DaemonPaths:
    state_dir: Path
    pid_file: Path
    log_file: Path


@dataclass(frozen=True)
class DaemonStatus:
    state: str
    pid: int | None
    pid_file: Path
    log_file: Path


@dataclass(frozen=True)
class StopResult:
    state: str
    pid: int | None
    pid_file: Path


def resolve_daemon_paths(
    *,
    host: str,
    port: int,
    state_dir: str | Path | None = None,
    pid_file: str | Path | None = None,
    log_file: str | Path | None = None,
) -> DaemonPaths:
    state = _path_from_arg_env_default(state_dir, STATE_DIR_ENV, _default_state_dir())
    stem = f"server-{_sanitize_path_part(host)}-{port}"
    pid = _path_from_arg_env_default(pid_file, PID_FILE_ENV, state / f"{stem}.pid")
    log = _path_from_arg_env_default(log_file, LOG_FILE_ENV, state / f"{stem}.log")
    return DaemonPaths(state_dir=state, pid_file=pid, log_file=log)


def find_daemon_pid_files(*, state_dir: str | Path, port: int) -> list[Path]:
    state = Path(state_dir).expanduser().resolve()
    if not state.exists():
        return []
    return sorted(
        path.resolve()
        for path in state.glob(f"server-*-{port}.pid")
        if path.is_file()
    )


def start_background(
    command: list[str], paths: DaemonPaths, *, env: dict[str, str] | None = None
) -> int:
    paths.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    paths.log_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    paths.pid_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

    existing_pid = _read_pid(paths.pid_file)
    if existing_pid is not None:
        if is_pid_alive(existing_pid):
            raise DaemonError(
                f"Already running with PID {existing_pid} ({paths.pid_file})."
            )
        paths.pid_file.unlink(missing_ok=True)

    with paths.log_file.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            env=env,
        )

    _write_pid(paths.pid_file, process.pid)
    return int(process.pid)


def run_supervised(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    restart_delay: float = 1.0,
    restart_limit: int | None = None,
    terminate_timeout: float = 10.0,
) -> int:
    stop_requested = False
    child: subprocess.Popen[bytes] | None = None
    previous_handlers: dict[int, Any] = {}

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        if child is not None and _process_is_running(child):
            child.terminate()

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_stop)

    restarts = 0
    try:
        while True:
            print(
                f"daemon supervisor starting child: {' '.join(command)}",
                flush=True,
            )
            child = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                env=env,
            )
            return_code = child.wait()
            if stop_requested:
                return int(return_code or 0)

            print(
                f"daemon supervisor child exited with status {return_code}; restarting",
                flush=True,
            )
            if restart_limit is not None and restarts >= restart_limit:
                return int(return_code or 0)
            restarts += 1

            if restart_delay > 0:
                time.sleep(restart_delay)
            if stop_requested:
                return 0
    finally:
        if child is not None and _process_is_running(child):
            _terminate_process(child, timeout=terminate_timeout)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def stop_background(paths: DaemonPaths, *, timeout: float = 10.0) -> StopResult:
    pid = _read_pid(paths.pid_file)
    if pid is None:
        return StopResult(state="not_running", pid=None, pid_file=paths.pid_file)

    if not is_pid_alive(pid):
        _remove_pid_file_if_matching(paths.pid_file, pid)
        return StopResult(state="stale", pid=pid, pid_file=paths.pid_file)

    os.kill(pid, signal.SIGTERM)
    if _wait_for_pid_exit(pid, timeout):
        _remove_pid_file_if_matching(paths.pid_file, pid)
        return StopResult(state="stopped", pid=pid, pid_file=paths.pid_file)

    os.kill(pid, signal.SIGKILL)
    _wait_for_pid_exit(pid, 5.0)
    _remove_pid_file_if_matching(paths.pid_file, pid)
    return StopResult(state="killed", pid=pid, pid_file=paths.pid_file)


def daemon_status(paths: DaemonPaths) -> DaemonStatus:
    pid = _read_pid(paths.pid_file)
    if pid is None:
        return DaemonStatus(
            state="stopped",
            pid=None,
            pid_file=paths.pid_file,
            log_file=paths.log_file,
        )
    if is_pid_alive(pid):
        return DaemonStatus(
            state="running",
            pid=pid,
            pid_file=paths.pid_file,
            log_file=paths.log_file,
        )
    return DaemonStatus(
        state="stale",
        pid=pid,
        pid_file=paths.pid_file,
        log_file=paths.log_file,
    )


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_pid_exit(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_pid_alive(pid):
            return True
        time.sleep(0.05)
    return not is_pid_alive(pid)


def _process_is_running(process: subprocess.Popen[bytes]) -> bool:
    poll = getattr(process, "poll", None)
    return bool(callable(poll) and poll() is None)


def _terminate_process(process: subprocess.Popen[bytes], *, timeout: float) -> None:
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def _default_state_dir() -> Path:
    return default_daemon_state_dir()


def _path_from_arg_env_default(
    arg_value: str | Path | None, env_name: str, default: Path
) -> Path:
    if arg_value is not None:
        return Path(arg_value).expanduser().resolve()
    if env_value := os.environ.get(env_name):
        return Path(env_value).expanduser().resolve()
    return default.expanduser().resolve()


def _sanitize_path_part(value: str) -> str:
    return "".join(char if char.isalnum() or char in ".-" else "_" for char in value)


def _read_pid(pid_file: Path) -> int | None:
    try:
        text = pid_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    try:
        pid = int(text)
    except ValueError:
        return None
    return pid if pid > 0 else None


def _write_pid(pid_file: Path, pid: int) -> None:
    tmp = pid_file.with_suffix(pid_file.suffix + ".tmp")
    tmp.write_text(f"{pid}\n", encoding="utf-8")
    tmp.replace(pid_file)


def _remove_pid_file_if_matching(pid_file: Path, pid: int) -> None:
    if _read_pid(pid_file) == pid:
        pid_file.unlink(missing_ok=True)
