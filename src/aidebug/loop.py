"""Interactive Run -> Diagnose -> Propose -> Confirm -> Apply -> Re-run loop."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from .context import describe_workspace, extract_referenced_files
from .ollama_client import OllamaClient, OllamaError
from .patcher import (
    StagedChange,
    apply_changes,
    rollback_changes,
    stage_edits,
    validate_no_truncation,
    validate_python_syntax,
)
from .runner import resolve_command, run_command


def _use_color() -> bool:
    return sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    if not _use_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def _banner(title: str, code: str = "1;36") -> None:
    print(f"\n{_c('=== ' + title + ' ===', code)}")


def _safe_input(input_fn: Callable[[str], str], prompt: str) -> Optional[str]:
    try:
        return input_fn(prompt)
    except EOFError:
        return None


def _normalize_suggested_command(raw_cmd: str, python_exe: str) -> List[str]:
    parts = shlex.split(raw_cmd)
    if parts and parts[0] in ("pip", "pip3"):
        return [python_exe, "-m", "pip", *parts[1:]]
    return parts


def run_debug_loop(
    raw_args: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    model: Optional[str] = None,
    host: Optional[str] = None,
    max_retries: Optional[int] = None,
    timeout: Optional[float] = None,
    ollama_timeout: float = 120.0,
    dry_run: bool = False,
    auto_yes: bool = False,
    auto_pull: bool = True,
    allow_outside_cwd: bool = False,
    python_executable: Optional[str] = None,
    client: Optional[OllamaClient] = None,
    input_fn: Callable[[str], str] = input,
) -> int:
    """Execute a Python script and interactively debug failures with local Ollama."""
    work_dir = (cwd or Path.cwd()).resolve()
    command = resolve_command(raw_args, cwd=work_dir, python_executable=python_executable)
    ollama = client or OllamaClient(host=host, timeout=ollama_timeout)
    selected_model: Optional[str] = None
    applied_history: List[List[StagedChange]] = []
    attempt_history: List[str] = []

    limit = max_retries if (max_retries is not None and max_retries >= 0) else None
    attempt = 0
    while limit is None or attempt <= limit:
        attempt += 1
        _banner(f"Run #{attempt}: {' '.join(command)}", "1;34")
        result = run_command(command, cwd=work_dir, stream_output=True, timeout=timeout)

        if result.succeeded:
            print(
                _c(
                    f"\n[aidebug] Program exited cleanly (code 0) in {result.duration_sec:.2f}s.",
                    "1;32",
                )
            )
            return 0

        print(
            _c(
                f"\n[aidebug] Program failed with exit code {result.returncode} "
                f"({result.duration_sec:.2f}s).",
                "1;31",
            )
        )

        if limit is not None and attempt > limit:
            print(_c(f"[aidebug] Reached max retries ({limit}). Stopping.", "1;33"))
            return result.returncode

        # Resolve (and auto-pull if missing) the Ollama model on first failure.
        try:
            if not selected_model:
                if hasattr(ollama, "ensure_model"):
                    selected_model = ollama.ensure_model(preferred=model, auto_pull=auto_pull)
                else:
                    selected_model = ollama.select_model(preferred=model)
        except OllamaError as exc:
            print(_c(f"[aidebug] {exc}", "1;31"), file=sys.stderr)
            return result.returncode

        files = extract_referenced_files(
            result.combined_output,
            command,
            work_dir,
            allow_outside_cwd=allow_outside_cwd,
        )
        focus_lines_by_path = {src.path: src.focus_lines for src in files}
        truncated_paths = {src.path for src in files if src.is_truncated()}
        workspace_summary = describe_workspace(work_dir)
        extra_hint: Optional[str] = None
        syntax_retries = 0

        while True:
            print(
                _c(
                    f"[aidebug] Analyzing error with local Ollama model '{selected_model}'...",
                    "36",
                )
            )
            try:
                diagnosis = ollama.diagnose(
                    model=selected_model,
                    command=command,
                    returncode=result.returncode,
                    output=result.combined_output,
                    files=files,
                    workspace_summary=workspace_summary,
                    previous_attempts=attempt_history,
                    extra_hint=extra_hint,
                )
            except OllamaError as exc:
                print(_c(f"[aidebug] Diagnosis failed: {exc}", "1;31"), file=sys.stderr)
                return result.returncode

            staged, warnings = stage_edits(
                diagnosis.edits,
                cwd=work_dir,
                allow_outside_cwd=allow_outside_cwd,
                focus_lines_by_path=focus_lines_by_path,
            )

            syntax_errors = validate_python_syntax(staged)
            truncation_errors = validate_no_truncation(staged)
            validation_errors = syntax_errors + truncation_errors
            staging_failed = bool(warnings)

            if (validation_errors or staging_failed) and syntax_retries < 2:
                syntax_retries += 1
                problems = validation_errors if validation_errors else warnings
                err_summary = "; ".join(problems)
                print(
                    _c(
                        f"[aidebug] Proposed patch failed validation ({err_summary}). "
                        f"Asking Ollama to fix it ({syntax_retries}/2)...",
                        "33",
                    )
                )
                touches_truncated = (
                    any(c.path in truncated_paths for c in staged)
                    or bool(truncated_paths)
                    or bool(truncation_errors)
                )
                if touches_truncated or staging_failed:
                    extra_hint = (
                        f"Your previous proposed edit failed: {err_summary}. "
                        "Use 'search' and 'replace' (with 'content': null) using exact contiguous lines "
                        "copied verbatim from the file. NEVER use '...' placeholders."
                    )
                else:
                    extra_hint = (
                        f"Your previous proposed edit was rejected: {err_summary}. "
                        "Fix the syntax error using 'search' and 'replace' (with 'content': null) "
                        "and make sure the resulting Python code is 100% syntactically valid."
                    )
                continue

            _banner("Diagnosis", "1;35")
            print(f"{_c('Root cause:', '1')} {diagnosis.root_cause}")
            if diagnosis.fix_summary:
                print(f"{_c('Proposed fix:', '1')} {diagnosis.fix_summary}")

            for warn in warnings:
                print(_c(f"[aidebug] Warning: {warn}", "33"))
            for err in validation_errors:
                print(_c(f"[aidebug] Validation warning in proposed patch: {err}", "1;31"))

            if staged:
                _banner("Proposed Diff", "1;36")
                for change in staged:
                    print(change.unified_diff(color=True))
            elif diagnosis.command_suggestion:
                print(
                    f"\n{_c('Suggested command:', '1;33')} {diagnosis.command_suggestion}"
                )
            else:
                print(
                    _c(
                        "\n[aidebug] No applicable file edits were proposed.",
                        "33",
                    )
                )

            if dry_run:
                return result.returncode

            if not staged and not diagnosis.command_suggestion:
                if auto_yes:
                    return result.returncode
                raw_choice = _safe_input(
                    input_fn,
                    "\n[r]etry with hint / [u]ndo last patch / [q]uit: "
                    if applied_history
                    else "\n[r]etry with hint / [q]uit: ",
                )
                if raw_choice is None:
                    return result.returncode
                choice = raw_choice.strip().lower()
                if choice in ("r", "retry"):
                    raw_hint = _safe_input(input_fn, "Hint for Ollama (optional): ")
                    extra_hint = raw_hint.strip() if raw_hint else None
                    continue
                if choice in ("u", "undo") and applied_history:
                    rollback_changes(applied_history.pop())
                    print(_c("[aidebug] Reverted last patch.", "33"))
                    break
                return result.returncode

            # Confirm before applying edits or running suggested command.
            if auto_yes:
                if staged and validation_errors:
                    print(
                        _c(
                            "[aidebug] Refusing to auto-apply patch that failed validation.",
                            "1;31",
                        ),
                        file=sys.stderr,
                    )
                    return result.returncode
                choice = "y"
            else:
                action_label = (
                    "Apply patch and re-run?"
                    if staged
                    else f"Run suggested command `{diagnosis.command_suggestion}` and re-run?"
                )
                undo_opt = " / [u]ndo" if applied_history else ""
                raw_choice = _safe_input(
                    input_fn,
                    f"\n{action_label} [y]es / [n]o / [r]etry{undo_opt}: ",
                )
                if raw_choice is None:
                    print(_c("[aidebug] Non-interactive input closed. Exiting.", "33"))
                    return result.returncode
                choice = raw_choice.strip().lower()

            if choice in ("y", "yes"):
                if staged:
                    apply_changes(staged)
                    applied_history.append(staged)
                    patched_files = ", ".join(c.rel_path for c in staged)
                    attempt_history.append(
                        f"Patched {patched_files} ({diagnosis.fix_summary or 'code edit'})"
                    )
                    print(_c(f"[aidebug] Applied patch to: {patched_files}", "1;32"))
                elif diagnosis.command_suggestion:
                    try:
                        cmd_parts = _normalize_suggested_command(
                            diagnosis.command_suggestion, command[0]
                        )
                    except ValueError as exc:
                        print(
                            _c(f"[aidebug] Invalid suggested command: {exc}", "1;31"),
                            file=sys.stderr,
                        )
                        return result.returncode
                    if not cmd_parts:
                        print(_c("[aidebug] Empty suggested command.", "1;31"), file=sys.stderr)
                        return result.returncode
                    _banner(f"Running suggested command: {' '.join(cmd_parts)}", "1;33")
                    try:
                        cmd_res = run_command(cmd_parts, cwd=work_dir, stream_output=True)
                    except OSError as exc:
                        print(
                            _c(f"[aidebug] Failed to run suggested command: {exc}", "1;31"),
                            file=sys.stderr,
                        )
                        attempt_history.append(
                            f"Failed to run `{diagnosis.command_suggestion}` ({exc})"
                        )
                    else:
                        attempt_history.append(
                            f"Ran `{diagnosis.command_suggestion}` -> exit code {cmd_res.returncode}"
                        )
                        if not cmd_res.succeeded:
                            print(
                                _c(
                                    f"[aidebug] Suggested command exited with {cmd_res.returncode}.",
                                    "1;31",
                                )
                            )
                break  # Re-run the program
            elif choice in ("r", "retry"):
                raw_hint = _safe_input(input_fn, "Hint for Ollama (optional): ")
                extra_hint = raw_hint.strip() if raw_hint else None
                continue
            elif choice in ("u", "undo") and applied_history:
                rollback_changes(applied_history.pop())
                print(_c("[aidebug] Reverted last patch. Re-running...", "33"))
                break
            else:
                print(_c("[aidebug] Fix declined. Exiting.", "33"))
                return result.returncode

    return 1
