"""Subprocess execution with live terminal output and capture."""

from __future__ import annotations

import codecs
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


@dataclass
class RunResult:
    command: List[str]
    returncode: int
    stdout: str
    stderr: str
    duration_sec: float

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0

    @property
    def combined_output(self) -> str:
        parts = []
        if self.stdout.strip():
            parts.append(f"--- STDOUT ---\n{self.stdout}")
        if self.stderr.strip():
            parts.append(f"--- STDERR ---\n{self.stderr}")
        return "\n".join(parts)


def _default_python_executable() -> str:
    """Choose the active virtualenv Python if set, otherwise sys.executable."""
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        venv_path = Path(venv)
        for rel in ("bin/python", "bin/python3", "Scripts/python.exe"):
            cand = venv_path / rel
            if cand.is_file() and os.access(cand, os.X_OK):
                return str(cand)
    return sys.executable


def resolve_command(
    args: Sequence[str],
    cwd: Optional[Path] = None,
    python_executable: Optional[str] = None,
) -> List[str]:
    """Resolve user CLI args into a Python interpreter command.

    Only Python scripts (`.py` files) are supported.
    """
    if not args:
        raise ValueError("No Python script provided.")

    cmd = list(args)
    base_dir = cwd or Path.cwd()
    first = cmd[0]

    if not first.endswith(".py"):
        raise ValueError(
            f"Unsupported target '{first}'. aidebug only runs Python scripts (.py files)."
        )

    candidate = Path(first)
    if not candidate.is_absolute():
        candidate = base_dir / candidate

    if not candidate.is_file():
        raise FileNotFoundError(f"Python script not found: {first}")

    python_bin = python_executable or _default_python_executable()
    return [python_bin, *cmd]


def _pump_stream(stream, target, buffer: List[str]) -> None:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    fd = stream.fileno()
    try:
        while True:
            raw = os.read(fd, 4096)
            if not raw:
                break
            chunk = decoder.decode(raw, final=False)
            if chunk:
                buffer.append(chunk)
                if target is not None:
                    target.write(chunk)
                    target.flush()
        tail = decoder.decode(b"", final=True)
        if tail:
            buffer.append(tail)
            if target is not None:
                target.write(tail)
                target.flush()
    finally:
        stream.close()


def run_command(
    command: Sequence[str],
    cwd: Optional[Path] = None,
    stream_output: bool = True,
    timeout: Optional[float] = None,
) -> RunResult:
    """Execute a command, optionally streaming stdout/stderr live while capturing both."""
    cmd_list = list(command)
    start = time.monotonic()

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    proc = subprocess.Popen(
        cmd_list,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    stdout_buf: List[str] = []
    stderr_buf: List[str] = []

    t_out = threading.Thread(
        target=_pump_stream,
        args=(proc.stdout, sys.stdout if stream_output else None, stdout_buf),
        daemon=True,
    )
    t_err = threading.Thread(
        target=_pump_stream,
        args=(proc.stderr, sys.stderr if stream_output else None, stderr_buf),
        daemon=True,
    )
    t_out.start()
    t_err.start()

    try:
        returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        returncode = proc.wait()
        stderr_buf.append(f"\n[aidebug] Command timed out after {timeout}s.\n")
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise

    t_out.join(timeout=2.0)
    t_err.join(timeout=2.0)

    elapsed = time.monotonic() - start
    return RunResult(
        command=cmd_list,
        returncode=returncode,
        stdout="".join(stdout_buf),
        stderr="".join(stderr_buf),
        duration_sec=elapsed,
    )
