"""Extract relevant source files and error context from program output."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Matches Python tracebacks: File "/path/to/foo.py", line 12
_PY_TRACEBACK_RE = re.compile(r'File "([^"]+\.py)", line (\d+)')

# Matches Python file:line references (e.g., syntax/linter output: app.py:12:5)
_PY_FILE_LINE_RE = re.compile(
    r'(?:^|[\s(\["\'])((?:/|\.{1,2}/)?[\w./-]+\.py):(\d+)(?::\d+)?'
)

# Matches bare .py filenames mentioned in error messages
_PY_BARE_FILE_RE = re.compile(
    r'(?:^|[\s(\["\'`])((?:\.{1,2}/)?[\w./-]+\.py)(?=$|[\s)\]}"\'`:,;])'
)

_IGNORED_DIR_PARTS = {
    "site-packages",
    "dist-packages",
    ".venv",
    "venv",
    "__pycache__",
    ".git",
}


@dataclass
class SourceContext:
    path: Path
    rel_path: str
    content: str
    focus_lines: List[int]

    @property
    def total_lines(self) -> int:
        return len(self.content.splitlines())

    def is_truncated(self, max_lines: int = 250) -> bool:
        return self.total_lines > max_lines

    def numbered_content(self, max_lines: int = 250) -> str:
        lines = self.content.splitlines()
        if len(lines) <= max_lines:
            return "\n".join(f"{i + 1:4d} | {line}" for i, line in enumerate(lines))

        # Keep windows around focus lines if the file is large.
        keep = set()
        if self.focus_lines:
            for ln in self.focus_lines:
                for idx in range(max(0, ln - 40), min(len(lines), ln + 40)):
                    keep.add(idx)
        else:
            for idx in range(min(len(lines), max_lines)):
                keep.add(idx)

        out: List[str] = []
        last_idx = -1
        for idx in sorted(keep):
            if last_idx != -1 and idx > last_idx + 1:
                out.append("     | ... (lines omitted) ...")
            out.append(f"{idx + 1:4d} | {lines[idx]}")
            last_idx = idx
        return "\n".join(out)


def _is_ignored(path: Path) -> bool:
    return any(part in _IGNORED_DIR_PARTS for part in path.parts)


def _resolve_existing_file(raw_path: str, cwd: Path) -> Optional[Path]:
    p = Path(raw_path)
    candidates = [p] if p.is_absolute() else [cwd / p, p]
    for cand in candidates:
        try:
            resolved = cand.resolve()
            if resolved.is_file() and not _is_ignored(resolved):
                return resolved
        except OSError:
            continue
    return None


def extract_referenced_files(
    output: str,
    command: Sequence[str],
    cwd: Path,
    max_files: int = 4,
    allow_outside_cwd: bool = False,
) -> List[SourceContext]:
    """Find source files referenced in error output and the invoked command."""
    cwd_resolved = cwd.resolve()
    file_lines: Dict[Path, List[int]] = {}
    ordered_paths: List[Path] = []

    def record(raw_path: str, line_no: Optional[int] = None) -> None:
        resolved = _resolve_existing_file(raw_path, cwd_resolved)
        if resolved is None:
            return
        # Prefer files inside the workspace to avoid reading system/stdlib files
        try:
            resolved.relative_to(cwd_resolved)
        except ValueError:
            if not allow_outside_cwd:
                return
        if resolved not in file_lines:
            file_lines[resolved] = []
            ordered_paths.append(resolved)
        if line_no is not None and line_no not in file_lines[resolved]:
            file_lines[resolved].append(line_no)

    # 1. Parse Python tracebacks (bottom-most frame is usually the root cause)
    py_matches: List[Tuple[str, int]] = [
        (m.group(1), int(m.group(2))) for m in _PY_TRACEBACK_RE.finditer(output)
    ]
    for raw_path, ln in reversed(py_matches):
        record(raw_path, ln)

    # 2. Parse Python file:line references
    for m in _PY_FILE_LINE_RE.finditer(output):
        record(m.group(1), int(m.group(2)))

    # 3. Parse bare .py filenames mentioned in error messages
    for m in _PY_BARE_FILE_RE.finditer(output):
        record(m.group(1), None)

    # 4. Include any .py script explicitly passed on the command line
    for token in command:
        if token.startswith("-") or not token.endswith(".py"):
            continue
        record(token, None)

    contexts: List[SourceContext] = []
    for path in ordered_paths[:max_files]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        try:
            rel = str(path.relative_to(cwd_resolved))
        except ValueError:
            rel = str(path)
        contexts.append(
            SourceContext(
                path=path,
                rel_path=rel,
                content=text,
                focus_lines=sorted(file_lines[path]),
            )
        )

    return contexts


def describe_workspace(cwd: Path, max_entries: int = 30) -> str:
    """Return a concise summary of the working directory and its top-level files."""
    resolved = cwd.resolve()
    try:
        entries = sorted(
            p.name + ("/" if p.is_dir() else "")
            for p in resolved.iterdir()
            if p.name not in _IGNORED_DIR_PARTS
        )
    except OSError:
        entries = []

    shown = entries[:max_entries]
    extra = f" ... (+{len(entries) - max_entries} more)" if len(entries) > max_entries else ""
    listing = ", ".join(shown) + extra if shown else "(empty)"
    return f"Working directory: {resolved}\nTop-level entries: {listing}"
