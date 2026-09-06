from __future__ import annotations

"""Safe discovery and cleanup of stale LNL web listeners.

The cleanup command deliberately identifies a process twice: it must own a
listening socket *and* its command line must look like an LNL service.  This
keeps a Python process belonging to another application out of the kill set.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Iterable


@dataclass(frozen=True)
class Listener:
    pid: int
    port: int
    address: str
    commandline: str


_LNL_COMMAND_MARKERS = (
    "lnl_toolbox",
    "command_console.py",
    "scratch/web/server.py",
    "scratch\\web\\server.py",
    "-m web.command_console",
)
_DEFAULT_LNL_PORTS = frozenset({8765})


def _port_from_endpoint(endpoint: str) -> int | None:
    value = endpoint.rsplit(":", 1)[-1].strip("[]")
    try:
        port = int(value)
    except ValueError:
        return None
    return port if 1 <= port <= 65535 else None


def _windows_listeners() -> list[tuple[int, int, str]]:
    result = subprocess.run(
        ["netstat", "-ano", "-p", "tcp"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    listeners: list[tuple[int, int, str]] = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 5 or fields[0].upper() != "TCP":
            continue
        if fields[3].upper() != "LISTENING":
            continue
        port = _port_from_endpoint(fields[1])
        try:
            pid = int(fields[4])
        except ValueError:
            continue
        if port is not None and pid > 0:
            listeners.append((pid, port, fields[1]))
    return listeners


def _posix_listeners() -> list[tuple[int, int, str]]:
    """Best-effort fallback for development environments without netstat.

    ``ss -ltnp`` is preferred because it includes owning PIDs.  If process
    details are hidden, returning no candidates is safer than guessing.
    """

    result = subprocess.run(
        ["ss", "-ltnp"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    listeners: list[tuple[int, int, str]] = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 5 or fields[0].upper() != "LISTEN":
            continue
        port = _port_from_endpoint(fields[3])
        if port is None:
            continue
        marker = next((field for field in fields[5:] if "pid=" in field), "")
        try:
            pid = int(marker.split("pid=", 1)[1].split(",", 1)[0])
        except (IndexError, ValueError):
            continue
        if pid > 0:
            listeners.append((pid, port, fields[3]))
    return listeners


def _process_commandlines(pids: Iterable[int]) -> dict[int, str]:
    wanted = {int(pid) for pid in pids if int(pid) > 0}
    if not wanted:
        return {}
    if os.name == "nt":
        # A single query avoids spawning PowerShell once per socket.  Failure
        # is intentionally treated as "unknown", never as permission to kill.
        script = (
            "Get-CimInstance Win32_Process | "
            "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
        )
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return {}
            payload = json.loads(result.stdout)
        except (OSError, json.JSONDecodeError):
            return {}
        entries = payload if isinstance(payload, list) else [payload]
        commandlines: dict[int, str] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                pid = int(entry.get("ProcessId"))
            except (TypeError, ValueError):
                continue
            if pid in wanted and entry.get("CommandLine"):
                commandlines[pid] = str(entry["CommandLine"])
        return commandlines

    commandlines = {}
    for pid in wanted:
        try:
            commandlines[pid] = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (OSError, UnicodeDecodeError):
            continue
    return commandlines


def _windows_process_images(pids: Iterable[int]) -> dict[int, str]:
    """Return executable paths when CIM command-line access is restricted."""

    wanted = {int(pid) for pid in pids if int(pid) > 0}
    if not wanted:
        return {}
    script = (
        "Get-Process -Id "
        + ",".join(str(pid) for pid in sorted(wanted))
        + " -ErrorAction SilentlyContinue | "
        "Select-Object Id,Path | ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {}
        payload = json.loads(result.stdout)
    except (OSError, json.JSONDecodeError):
        return {}
    entries = payload if isinstance(payload, list) else [payload]
    images: dict[int, str] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("Path"):
            continue
        try:
            pid = int(entry.get("Id"))
        except (TypeError, ValueError):
            continue
        if pid in wanted:
            images[pid] = str(entry["Path"])
    return images


def find_lnl_listeners(ports: Iterable[int] | None = None) -> tuple[Listener, ...]:
    requested = None if ports is None else {int(port) for port in ports}
    raw = _windows_listeners() if os.name == "nt" else _posix_listeners()
    raw_pids = {pid for pid, _, _ in raw}
    commandlines = _process_commandlines(raw_pids)
    images = (
        _windows_process_images(raw_pids - commandlines.keys())
        if os.name == "nt" and raw_pids - commandlines.keys()
        else {}
    )
    listeners: list[Listener] = []
    seen: set[tuple[int, int]] = set()
    for pid, port, address in raw:
        commandline = commandlines.get(pid, "")
        if requested is not None and port not in requested:
            continue
        if commandline:
            if not any(marker in commandline.lower() for marker in _LNL_COMMAND_MARKERS):
                continue
        else:
            # Some locked-down Windows sessions deny Win32_Process command-line
            # access.  The fallback is deliberately narrow: only a Python LNL
            # service on the documented default port is eligible.
            fallback_ports = _DEFAULT_LNL_PORTS if requested is None else requested
            image = images.get(pid, "")
            if port not in fallback_ports or Path(image).name.lower() not in {
                "python.exe",
                "pythonw.exe",
                "lnl.exe",
            }:
                continue
            commandline = f"{image} (command line unavailable; matched LNL port {port})"
        key = (pid, port)
        if key in seen:
            continue
        seen.add(key)
        listeners.append(Listener(pid, port, address, commandline))
    return tuple(sorted(listeners, key=lambda item: (item.port, item.pid)))


def _terminate(pid: int) -> tuple[bool, str]:
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except OSError as exc:
            return False, str(exc)
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, (result.stderr or result.stdout).strip()
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        return False, str(exc)
    return True, "SIGTERM sent"


def cleanup_ports(ports: Iterable[int] | None = None, *, dry_run: bool = False) -> int:
    listeners = find_lnl_listeners(ports)
    if not listeners:
        scope = "指定端口" if ports is not None else "LNL 监听端口"
        print(f"未发现可清理的{scope}进程。")
        return 0

    failures = 0
    for listener in listeners:
        prefix = "将清理" if dry_run else "清理"
        print(f"{prefix} PID {listener.pid} · {listener.address} · {listener.port}")
        if dry_run:
            continue
        ok, detail = _terminate(listener.pid)
        if ok:
            print(f"  已终止 PID {listener.pid}")
        else:
            print(f"  无法终止 PID {listener.pid}: {detail}", file=sys.stderr)
            failures += 1
    return 0 if dry_run or failures == 0 else 1
