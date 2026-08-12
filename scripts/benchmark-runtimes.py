#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import httpx

ROOT = Path(__file__).resolve().parents[1]


class FakeCodexHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    counter = 0
    counter_lock = threading.Lock()

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def do_GET(self) -> None:
        if self.path.startswith("/backend-api/codex/models"):
            self._json(
                {
                    "models": [
                        {
                            "slug": "gpt-5.6-luna",
                            "supported_in_api": True,
                            "visibility": "list",
                        }
                    ]
                }
            )
            return
        self._json({"object": "list", "data": []})

    def do_POST(self) -> None:
        size = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(size))
        if self.path != "/backend-api/codex/responses":
            self._json({"ok": True})
            return
        with self.counter_lock:
            type(self).counter += 1
            number = type(self).counter
        response_id = f"resp_bench_{number}"
        message_id = f"msg_bench_{number}"
        text = "benchmark response"
        created = time.time()
        message = {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "phase": "final_answer",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
        response = {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "status": "completed",
            "model": payload.get("model"),
            "output": [message],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
        }
        events = [
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {**response, "status": "in_progress", "output": []},
            },
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "output_index": 0,
                "content_index": 0,
                "item_id": message_id,
                "delta": text,
                "logprobs": [],
            },
            {
                "type": "response.output_item.done",
                "sequence_number": 2,
                "output_index": 0,
                "item": message,
            },
            {
                "type": "response.completed",
                "sequence_number": 3,
                "response": response,
            },
        ]
        encoded = (
            "".join(
                f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
                for event in events
            )
            + "data: [DONE]\n\n"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _json(self, value: Any) -> None:
        encoded = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


@contextmanager
def fake_codex(host: str, port: int) -> Iterator[str]:
    server = ThreadingHTTPServer((host, port), FakeCodexHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}/backend-api/codex"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def build_go_binary(directory: Path) -> Path:
    binary = directory / "openai-api-server-via-codex-go"
    subprocess.run(
        ["go", "build", "-trimpath", "-o", str(binary), "./cmd/openai-api-server-via-codex"],
        cwd=ROOT,
        check=True,
    )
    return binary


def write_auth(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": "header.payload.signature",
                    "account_id": "benchmark-account",
                },
            }
        ),
        encoding="utf-8",
    )


def runtime_command(runtime: str, go_binary: Path) -> list[str]:
    if runtime == "python":
        return [sys.executable, "-m", "openai_api_server_via_codex"]
    if runtime == "go":
        return [str(go_binary)]
    raise ValueError(runtime)


async def wait_healthy(base_url: str, process: subprocess.Popen[str]) -> float:
    started = time.perf_counter()
    async with httpx.AsyncClient(trust_env=False) as client:
        for _ in range(300):
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                raise RuntimeError(f"server exited during startup: {output}")
            try:
                response = await client.get(f"{base_url}/healthz", timeout=0.2)
                if response.status_code == 200:
                    return time.perf_counter() - started
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.01)
    raise RuntimeError(f"server did not become healthy: {base_url}")


def proc_metrics(pid: int) -> tuple[int, int, float]:
    status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    values: dict[str, int] = {}
    for line in status.splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0])
    stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    fields = stat[stat.rfind(")") + 2 :].split()
    ticks = int(fields[11]) + int(fields[12])
    cpu_seconds = ticks / float(os.sysconf("SC_CLK_TCK"))
    return values.get("VmRSS", 0), values.get("VmHWM", 0), cpu_seconds


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


async def request_once(client: httpx.AsyncClient, scenario: str) -> float:
    started = time.perf_counter()
    if scenario == "responses_nonstream":
        response = await client.post(
            "/v1/responses",
            json={"model": "gpt-5.6-luna", "input": "benchmark marker"},
        )
        if response.status_code != 200 or response.json().get("status") != "completed":
            raise RuntimeError(f"bad Responses result: {response.status_code} {response.text}")
    elif scenario == "chat_stream":
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "gpt-5.6-luna",
                "messages": [{"role": "user", "content": "benchmark marker"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ) as response:
            body = b"".join([chunk async for chunk in response.aiter_bytes()])
            if response.status_code != 200 or b"data: [DONE]" not in body:
                raise RuntimeError(f"bad Chat stream result: {response.status_code}")
    else:
        raise ValueError(scenario)
    return time.perf_counter() - started


async def run_load(base_url: str, scenario: str, requests: int, concurrency: int) -> list[float]:
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    timeout = httpx.Timeout(30)
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(
        base_url=base_url,
        headers={"Authorization": "Bearer benchmark"},
        limits=limits,
        timeout=timeout,
        trust_env=False,
    ) as client:
        async def bounded() -> float:
            async with semaphore:
                return await request_once(client, scenario)

        return await asyncio.gather(*(bounded() for _ in range(requests)))


async def benchmark_one(
    runtime: str,
    scenario: str,
    command: list[str],
    host: str,
    port: int,
    upstream_url: str,
    auth_json: Path,
    requests: int,
    concurrency: int,
) -> dict[str, Any]:
    process = cast(
        subprocess.Popen[str],
        await asyncio.to_thread(
            subprocess.Popen,
            [
                *command,
                "serve",
                "--host",
                host,
                "--port",
                str(port),
                "--backend-base-url",
                upstream_url,
                "--auth-json",
                str(auth_json),
                "--max-stored-items",
                "0",
                "--max-concurrent-requests",
                "0",
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "NO_PROXY": host, "no_proxy": host},
        ),
    )
    base_url = f"http://{host}:{port}"
    try:
        startup = await wait_healthy(base_url, process)
        idle_rss, _, _ = proc_metrics(process.pid)
        await run_load(base_url, scenario, min(20, requests), min(concurrency, 8))
        _, _, cpu_before = proc_metrics(process.pid)
        started = time.perf_counter()
        latencies = await run_load(base_url, scenario, requests, concurrency)
        elapsed = time.perf_counter() - started
        rss, peak_rss, cpu_after = proc_metrics(process.pid)
        cpu_seconds = cpu_after - cpu_before
        return {
            "runtime": runtime,
            "scenario": scenario,
            "requests": requests,
            "concurrency": concurrency,
            "startup_ms": round(startup * 1000, 2),
            "idle_rss_mib": round(idle_rss / 1024, 2),
            "final_rss_mib": round(rss / 1024, 2),
            "peak_rss_mib": round(peak_rss / 1024, 2),
            "cpu_seconds": round(cpu_seconds, 4),
            "cpu_percent_one_core": round(cpu_seconds / elapsed * 100, 2),
            "elapsed_seconds": round(elapsed, 4),
            "throughput_rps": round(requests / elapsed, 2),
            "latency_p50_ms": round(percentile(latencies, 0.50) * 1000, 2),
            "latency_p95_ms": round(percentile(latencies, 0.95) * 1000, 2),
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    if platform.system() != "Linux" or not Path("/proc/self/status").exists():
        raise RuntimeError("this benchmark currently requires Linux /proc metrics")
    with tempfile.TemporaryDirectory(prefix="openai-via-codex-benchmark-") as temp:
        temp_dir = Path(temp)
        auth_json = temp_dir / "auth.json"
        write_auth(auth_json)
        go_binary = await asyncio.to_thread(build_go_binary, temp_dir)
        go_version = await asyncio.to_thread(
            subprocess.run,
            ["go", "version"],
            check=True,
            capture_output=True,
            text=True,
        )
        results: list[dict[str, Any]] = []
        with fake_codex(args.host, args.upstream_port) as upstream_url:
            index = 0
            for scenario in ("responses_nonstream", "chat_stream"):
                for runtime in ("python", "go"):
                    result = await benchmark_one(
                        runtime,
                        scenario,
                        runtime_command(runtime, go_binary),
                        args.host,
                        args.server_port + index,
                        upstream_url,
                        auth_json,
                        args.requests,
                        args.concurrency,
                    )
                    results.append(result)
                    print(json.dumps(result, sort_keys=True), flush=True)
                    index += 1
        return {
            "schema_version": 1,
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "machine": {
                "os": platform.platform(),
                "architecture": platform.machine(),
                "cpu": cpu_model(),
                "logical_cpus": os.cpu_count(),
                "python": platform.python_version(),
                "go": go_version.stdout.strip(),
            },
            "configuration": {
                "requests_per_scenario": args.requests,
                "concurrency": args.concurrency,
                "max_stored_items": 0,
                "max_concurrent_requests": 0,
                "backend": "deterministic local SSE fake",
            },
            "results": results,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=18100)
    parser.add_argument("--upstream-port", type=int, default=18110)
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = asyncio.run(async_main(args))
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
