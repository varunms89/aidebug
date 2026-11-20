"""Safe patch staging, unified diff formatting, application, and rollback."""

from __future__ import annotations

import difflib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .ollama_client import ProposedEdit


class PatchError(RuntimeError):
    """Raised when a proposed edit cannot be safely applied."""


@dataclass
class StagedChange:
    path: Path
    rel_path: str
    original: str
    updated: str

    def unified_diff(self, color: bool = True) -> str:
        diff_lines = list(
            difflib.unified_diff(
                self.original.splitlines(keepends=True),
                self.updated.splitlines(keepends=True),
                fromfile=f"a/{self.rel_path}",
                tofile=f"b/{self.rel_path}",
            )
        )
        if not diff_lines:
            return "(no changes)"

        if not color or not sys.stdout.isatty():
            return "".join(diff_lines)

        colored: List[str] = []
        for line in diff_lines:
            if line.startswith("+++") or line.startswith("---"):
                colored.append(f"\033[1m{line}\033[0m")
            elif line.startswith("@@"):
                colored.append(f"\033[36m{line}\033[0m")
            elif line.startswith("+"):
                colored.append(f"\033[32m{line}\033[0m")
            elif line.startswith("-"):
                colored.append(f"\033[31m{line}\033[0m")
            else:
                colored.append(line)
        return "".join(colored)


def _resolve_safe_path(raw_file: str, cwd: Path, allow_outside_cwd: bool = False) -> Tuple[Path, str]:
    cwd_resolved = cwd.resolve()
    candidate = Path(raw_file)
    if not candidate.is_absolute():
        candidate = cwd_resolved / candidate
    resolved = candidate.resolve()

    try:
        rel = str(resolved.relative_to(cwd_resolved))
    except ValueError as exc:
        if not allow_outside_cwd:
            raise PatchError(
                f"Refusing to edit file outside working directory: {raw_file}"
            ) from exc
        rel = str(resolved)

    return resolved, rel


def _find_all_occurrences(text: str, sub: str) -> List[int]:
    positions: List[int] = []
    start = 0
    while True:
        idx = text.find(sub, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + 1
    return positions


def _pick_match_index(
    match_lines: Sequence[int],
    focus_lines: Optional[Sequence[int]],
    rel_path: str,
) -> int:
    """Return index into match_lines to patch, using focus_lines to disambiguate."""
    if len(match_lines) == 1:
        return 0
    if focus_lines:
        scored = [
            (min(abs(ln - fl) for fl in focus_lines), idx)
            for idx, ln in enumerate(match_lines)
        ]
        scored.sort()
        if scored[0][0] < scored[1][0]:
            return scored[0][1]
    raise PatchError(
        f"Ambiguous 'search' block in {rel_path} (matched {len(match_lines)} locations). "
        "Include more surrounding lines in 'search'."
    )


def _apply_search_replace(
    original: str,
    search: str,
    replace: str,
    rel_path: str,
    focus_lines: Optional[Sequence[int]] = None,
) -> str:
    if not search:
        raise PatchError(f"Edit for {rel_path} has an empty 'search' block.")

    positions = _find_all_occurrences(original, search)
    if positions:
        match_line_nos = [original.count("\n", 0, pos) + 1 for pos in positions]
        chosen_idx = _pick_match_index(match_line_nos, focus_lines, rel_path)
        pos = positions[chosen_idx]
        return original[:pos] + replace + original[pos + len(search) :]

    # Second chance: match ignoring trailing whitespace on each line.
    orig_lines = original.splitlines(keepends=True)
    search_lines = search.splitlines()
    if not search_lines:
        raise PatchError(f"Edit for {rel_path} has an empty 'search' block.")

    n = len(search_lines)
    target = [ln.rstrip() for ln in search_lines]
    matched_starts: List[int] = []
    for i in range(len(orig_lines) - n + 1):
        window = [ln.rstrip("\r\n").rstrip() for ln in orig_lines[i : i + n]]
        if window == target:
            matched_starts.append(i)

    if matched_starts:
        match_line_nos = [i + 1 for i in matched_starts]
        chosen_idx = _pick_match_index(match_line_nos, focus_lines, rel_path)
        i = matched_starts[chosen_idx]
        newline = "\r\n" if orig_lines[i].endswith("\r\n") else "\n"
        rep_lines = replace.splitlines()
        replacement_block = "".join(ln + newline for ln in rep_lines)
        if not orig_lines[i + n - 1].endswith(("\n", "\r\n")) and replacement_block.endswith(newline):
            replacement_block = replacement_block[: -len(newline)]
        return "".join(orig_lines[:i]) + replacement_block + "".join(orig_lines[i + n :])

    # Third chance: match ignoring leading/trailing whitespace and preserve base indentation.
    stripped_target = [ln.strip() for ln in search_lines]
    if any(stripped_target):
        stripped_starts: List[int] = []
        for i in range(len(orig_lines) - n + 1):
            window = [ln.strip() for ln in orig_lines[i : i + n]]
            if window == stripped_target:
                stripped_starts.append(i)
        if stripped_starts:
            match_line_nos = [i + 1 for i in stripped_starts]
            chosen_idx = _pick_match_index(match_line_nos, focus_lines, rel_path)
            i = stripped_starts[chosen_idx]
            newline = "\r\n" if orig_lines[i].endswith("\r\n") else "\n"
            orig_indent = len(orig_lines[i]) - len(orig_lines[i].lstrip(" "))
            search_indent = len(search_lines[0]) - len(search_lines[0].lstrip(" "))
            indent_delta = orig_indent - search_indent
            rep_lines = replace.splitlines()
            adjusted: List[str] = []
            for ln in rep_lines:
                if not ln.strip():
                    adjusted.append("")
                elif indent_delta > 0:
                    adjusted.append((" " * indent_delta) + ln)
                elif indent_delta < 0:
                    drop = min(-indent_delta, len(ln) - len(ln.lstrip(" ")))
                    adjusted.append(ln[drop:])
                else:
                    adjusted.append(ln)
            replacement_block = "".join(ln + newline for ln in adjusted)
            if not orig_lines[i + n - 1].endswith(("\n", "\r\n")) and replacement_block.endswith(newline):
                replacement_block = replacement_block[: -len(newline)]
            return "".join(orig_lines[:i]) + replacement_block + "".join(orig_lines[i + n :])

    raise PatchError(
        f"Could not find target snippet in {rel_path}. The model's 'search' block did not match."
    )


def _apply_line_range_replace(
    original: str,
    start_line: int,
    end_line: Optional[int],
    replace: str,
    rel_path: str,
) -> str:
    orig_lines = original.splitlines(keepends=True)
    end = start_line if end_line is None else end_line
    if start_line < 1 or start_line > len(orig_lines) + 1:
        raise PatchError(f"Invalid start_line {start_line} for {rel_path}.")
    if end < start_line - 1 or end > len(orig_lines):
        raise PatchError(f"Invalid end_line {end} for {rel_path}.")
    newline = "\r\n" if (orig_lines and orig_lines[0].endswith("\r\n")) else "\n"
    rep_lines = replace.splitlines()
    if start_line <= len(orig_lines) and rep_lines:
        orig_first = orig_lines[start_line - 1]
        orig_indent = len(orig_first) - len(orig_first.lstrip(" "))
        rep_indent = len(rep_lines[0]) - len(rep_lines[0].lstrip(" "))
        if orig_indent > 0 and rep_indent == 0 and not rep_lines[0].startswith(("def ", "class ", "@")):
            rep_lines = [(" " * orig_indent + ln) if ln.strip() else "" for ln in rep_lines]
    replacement_block = "".join(ln + newline for ln in rep_lines)
    return "".join(orig_lines[: start_line - 1]) + replacement_block + "".join(orig_lines[end:])


def _prepend_snippet(original: str, snippet: str) -> str:
    """Insert a top-of-file snippet (e.g. missing import) after shebang/docstring/__future__."""
    newline = "\r\n" if "\r\n" in original else "\n"
    block = snippet if snippet.endswith(("\n", "\r\n")) else snippet + newline
    lines = original.splitlines(keepends=True)
    insert_at = 0

    if lines and lines[0].startswith("#!"):
        insert_at = 1

    # Skip leading blank lines or comments before module docstring
    idx = insert_at
    while idx < len(lines) and (not lines[idx].strip() or lines[idx].lstrip().startswith("#")):
        idx += 1

    # Skip module docstring if present
    if idx < len(lines) and lines[idx].lstrip().startswith(('"""', "'''")):
        quote = lines[idx].lstrip()[:3]
        rest = lines[idx].lstrip()[3:]
        if quote in rest:
            insert_at = idx + 1
        else:
            idx += 1
            while idx < len(lines):
                if quote in lines[idx]:
                    insert_at = idx + 1
                    break
                idx += 1

    # Skip `from __future__ import ...` statements
    idx = insert_at
    while idx < len(lines) and (
        not lines[idx].strip() or lines[idx].lstrip().startswith("from __future__ import ")
    ):
        if lines[idx].lstrip().startswith("from __future__ import "):
            insert_at = idx + 1
        idx += 1

    prefix = "".join(lines[:insert_at])
    suffix = "".join(lines[insert_at:])
    if prefix and not prefix.endswith("\n"):
        prefix += newline
    if prefix and not suffix.startswith(("\n", "\r\n")):
        block = newline + block
    return prefix + block + suffix


def stage_edits(
    edits: Sequence[ProposedEdit],
    cwd: Path,
    allow_outside_cwd: bool = False,
    focus_lines_by_path: Optional[Dict[Path, Sequence[int]]] = None,
) -> Tuple[List[StagedChange], List[str]]:
    """Stage proposed edits in memory without modifying disk yet.

    Returns `(staged_changes, warnings)`.
    """
    working_texts: Dict[Path, str] = {}
    originals: Dict[Path, str] = {}
    rel_paths: Dict[Path, str] = {}
    warnings: List[str] = []

    for edit in edits:
        raw_file = edit.file
        try:
            path, rel = _resolve_safe_path(raw_file, cwd, allow_outside_cwd=allow_outside_cwd)
        except PatchError as exc:
            warnings.append(str(exc))
            continue

        if not path.is_file() and (raw_file.startswith("a/") or raw_file.startswith("b/")):
            try:
                alt_path, alt_rel = _resolve_safe_path(
                    raw_file[2:], cwd, allow_outside_cwd=allow_outside_cwd
                )
                if alt_path.is_file():
                    path, rel = alt_path, alt_rel
            except PatchError:
                pass

        if path not in originals:
            if not path.is_file():
                warnings.append(f"File does not exist: {edit.file}")
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            originals[path] = text
            working_texts[path] = text
            rel_paths[path] = rel

        current = working_texts[path]
        if edit.search.strip():
            try:
                file_focus = focus_lines_by_path.get(path) if focus_lines_by_path else None
                working_texts[path] = _apply_search_replace(
                    current, edit.search, edit.replace, rel, focus_lines=file_focus
                )
            except PatchError as exc:
                if edit.start_line is not None:
                    try:
                        working_texts[path] = _apply_line_range_replace(
                            current, edit.start_line, edit.end_line, edit.replace, rel
                        )
                    except PatchError:
                        warnings.append(str(exc))
                else:
                    warnings.append(str(exc))
        elif edit.start_line is not None:
            try:
                working_texts[path] = _apply_line_range_replace(
                    current, edit.start_line, edit.end_line, edit.replace, rel
                )
            except PatchError as exc:
                warnings.append(str(exc))
        elif edit.content is not None and edit.content.strip():
            working_texts[path] = edit.content
        elif edit.replace.strip():
            working_texts[path] = _prepend_snippet(current, edit.replace)
        else:
            warnings.append(f"Edit for {rel} has neither 'search' nor 'content'.")

    staged: List[StagedChange] = []
    for path, orig in originals.items():
        updated = working_texts[path]
        if updated != orig:
            staged.append(
                StagedChange(
                    path=path,
                    rel_path=rel_paths[path],
                    original=orig,
                    updated=updated,
                )
            )

    return staged, warnings


def validate_python_syntax(changes: Sequence[StagedChange]) -> List[str]:
    """Compile-check staged `.py` changes and return any SyntaxError descriptions."""
    errors: List[str] = []
    for change in changes:
        if change.path.suffix != ".py":
            continue
        try:
            compile(change.updated, change.rel_path, "exec")
        except SyntaxError as exc:
            line_info = f" line {exc.lineno}" if exc.lineno else ""
            bad_line = f": {exc.text.strip()!r}" if exc.text else ""
            errors.append(f"{change.rel_path}{line_info}: {exc.msg}{bad_line}")
    return errors


def validate_no_truncation(changes: Sequence[StagedChange]) -> List[str]:
    """Detect when a model accidentally truncated a file instead of returning all lines."""
    errors: List[str] = []
    for change in changes:
        orig_nonempty = [ln for ln in change.original.splitlines() if ln.strip()]
        new_nonempty = [ln for ln in change.updated.splitlines() if ln.strip()]
        if len(orig_nonempty) >= 6 and len(new_nonempty) < max(3, int(len(orig_nonempty) * 0.65)):
            errors.append(
                f"{change.rel_path}: proposed edit shrank file from "
                f"{len(orig_nonempty)} to {len(new_nonempty)} non-empty lines (likely truncated)"
            )
    return errors


def apply_changes(changes: Sequence[StagedChange]) -> None:
    """Write staged changes to disk."""
    for change in changes:
        change.path.write_text(change.updated, encoding="utf-8")


def rollback_changes(changes: Sequence[StagedChange]) -> None:
    """Restore files to their pre-patch contents."""
    for change in changes:
        change.path.write_text(change.original, encoding="utf-8")
