"""Command-line interface for aidebug."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from . import __version__
from .loop import run_debug_loop
from .ollama_client import DEFAULT_MODEL, DEFAULT_OLLAMA_HOST, OllamaClient, OllamaError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aidebug",
        description="Run Python scripts and interactively debug errors using local Ollama.",
    )
    parser.add_argument(
        "-V", "--version", action="version", version=f"aidebug {__version__}"
    )

    sub = parser.add_subparsers(dest="subcommand")

    # `aidebug run`
    run_p = sub.add_parser(
        "run",
        help="Run a Python script (.py) and debug on failure.",
    )
    run_p.add_argument(
        "-m",
        "--model",
        default=None,
        help="Ollama model name (defaults to AIDEBUG_MODEL or best installed coding model).",
    )
    run_p.add_argument(
        "--host",
        default=None,
        help=f"Ollama server URL (default: OLLAMA_HOST or {DEFAULT_OLLAMA_HOST}).",
    )
    run_p.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Maximum debug/fix iterations (default: unlimited until clean exit).",
    )
    run_p.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Timeout in seconds for each program run.",
    )
    run_p.add_argument(
        "--ollama-timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for Ollama API responses (default: 120).",
    )
    run_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Diagnose and display diff without prompting to apply.",
    )
    run_p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Automatically apply proposed fixes without confirmation.",
    )
    run_p.add_argument(
        "--allow-outside-cwd",
        action="store_true",
        help="Allow patching files outside the current working directory.",
    )
    run_p.add_argument(
        "--python",
        default=None,
        help="Path to the Python interpreter to run the script (defaults to active virtualenv or current Python).",
    )
    run_p.add_argument(
        "--no-auto-pull",
        action="store_true",
        help="Do not automatically pull missing Ollama models.",
    )
    run_p.add_argument(
        "target",
        nargs=argparse.REMAINDER,
        help="Python script and optional arguments (e.g. app.py --port 8000).",
    )

    # `aidebug setup`
    setup_p = sub.add_parser(
        "setup",
        help=f"Ensure a local Ollama model is installed (auto-pulls {DEFAULT_MODEL} if missing).",
    )
    setup_p.add_argument(
        "-m",
        "--model",
        default=None,
        help=f"Model to ensure/pull (default: AIDEBUG_MODEL or {DEFAULT_MODEL}).",
    )
    setup_p.add_argument("--host", default=None, help="Ollama server URL.")

    # `aidebug models`
    models_p = sub.add_parser(
        "models",
        help="List locally installed Ollama models.",
    )
    models_p.add_argument("--host", default=None, help="Ollama server URL.")

    # `aidebug doctor`
    doctor_p = sub.add_parser(
        "doctor",
        help="Check Python environment and local Ollama connectivity.",
    )
    doctor_p.add_argument("--host", default=None, help="Ollama server URL.")

    return parser


def _cmd_models(host: Optional[str]) -> int:
    client = OllamaClient(host=host)
    try:
        models = client.list_models()
    except OllamaError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not models:
        print(f"No models installed on {client.host}.")
        print("Try: ollama pull qwen2.5-coder:7b")
        return 0

    default_model = client.ensure_model(auto_pull=False)
    print(f"Installed Ollama models at {client.host}:")
    for name in models:
        marker = " (default)" if name == default_model else ""
        print(f"  - {name}{marker}")
    return 0


def _cmd_setup(model: Optional[str], host: Optional[str]) -> int:
    client = OllamaClient(host=host)
    try:
        chosen = client.ensure_model(preferred=model, auto_pull=True, stream_progress=True)
    except OllamaError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Ready to debug with local Ollama model: {chosen}")
    return 0


def _cmd_doctor(host: Optional[str]) -> int:
    print(f"aidebug version: {__version__}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    client = OllamaClient(host=host)
    print(f"Ollama host: {client.host}")
    try:
        models = client.list_models()
    except OllamaError as exc:
        print(f"Ollama status: UNREACHABLE ({exc})", file=sys.stderr)
        return 1

    print(f"Ollama status: OK ({len(models)} model(s) installed)")
    if models:
        print(f"Selected default model: {client.ensure_model(auto_pull=False)}")
    else:
        print(f"No models pulled yet. Run `aidebug setup` (or `aidebug run`) to auto-pull {DEFAULT_MODEL}.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "models":
        return _cmd_models(args.host)

    if args.subcommand == "setup":
        return _cmd_setup(args.model, args.host)

    if args.subcommand == "doctor":
        return _cmd_doctor(args.host)

    if args.subcommand == "run":
        target: List[str] = list(args.target)
        if target and target[0] == "--":
            target = target[1:]
        if not target:
            parser.error("`aidebug run` requires a Python script, e.g. `aidebug run app.py`.")
        try:
            return run_debug_loop(
                target,
                cwd=Path.cwd(),
                model=args.model,
                host=args.host,
                max_retries=args.max_retries,
                timeout=args.timeout,
                ollama_timeout=args.ollama_timeout,
                dry_run=args.dry_run,
                auto_yes=args.yes,
                auto_pull=not args.no_auto_pull,
                allow_outside_cwd=args.allow_outside_cwd,
                python_executable=args.python,
            )
        except (ValueError, FileNotFoundError) as exc:
            print(f"[aidebug] Error: {exc}", file=sys.stderr)
            return 2
        except KeyboardInterrupt:
            print("\n[aidebug] Interrupted by user.", file=sys.stderr)
            return 130

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
