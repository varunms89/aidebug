# aidebug

Local AI-powered CLI Python runner and interactive debugger backed by [Ollama](https://ollama.com) — zero API keys and zero third-party Python dependencies.

## Features

- **Python-only runner**: `aidebug run script.py [args...]` runs your `.py` script and streams stdout/stderr live (including interactive `input()` prompts).
- **Automatic Ollama model setup**: Automatically pulls the default local coding model (`qwen2.5-coder:7b`, or `AIDEBUG_MODEL`) on first run or via `aidebug setup`.
- **Local Ollama diagnosis**: Parses Python tracebacks, loads relevant `.py` workspace files (with partial-window support for large files), and uses structured JSON Schema outputs to propose targeted fixes.
- **Interactive confirmation & colored diffs**: Displays root cause, fix summary, and a unified diff before modifying any file (`[y]es / [n]o / [r]etry / [u]ndo`).
- **Safe patching & validation**: Syntax-checks and truncation-checks every patch before applying, refuses to edit files outside the working directory by default, and supports rolling back applied patches.

## Prerequisites

- Python `>= 3.9`
- [Ollama](https://ollama.com) running locally (`ollama serve`)

## Installation

```bash
pip install ollama-aidebug
```

Or install globally as an isolated CLI tool with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install ollama-aidebug
```

## Quick Start

1. Verify Ollama connectivity and ensure a coding model is installed:
   ```bash
   aidebug doctor
   aidebug setup
   ```

2. Run and debug a Python script:
   ```bash
   aidebug run app.py
   aidebug run app.py --port 8000
   ```

3. Use a specific Ollama model or Python interpreter:
   ```bash
   aidebug run -m qwen2.5-coder:7b script.py
   aidebug run --python /path/to/python3.12 script.py
   ```

## Commands

| Command | Description |
| --- | --- |
| `aidebug run <script.py> [args...]` | Run a Python script and interactively debug failures |
| `aidebug setup [-m MODEL]` | Pull/verify a local Ollama coding model |
| `aidebug models` | List locally installed Ollama models |
| `aidebug doctor` | Check Python environment and local Ollama connectivity |

## Options (`aidebug run`)

| Flag | Description |
| --- | --- |
| `-m, --model` | Ollama model name (or set `AIDEBUG_MODEL`) |
| `--host` | Ollama URL (default: `OLLAMA_HOST` or `http://127.0.0.1:11434`) |
| `--max-retries` | Maximum fix/re-run attempts (default: unlimited until clean exit; `0` = run once) |
| `--timeout` | Per-run execution timeout in seconds |
| `--ollama-timeout` | Ollama API response timeout in seconds (default: `120`) |
| `--dry-run` | Show diagnosis and diff without modifying files |
| `-y, --yes` | Auto-apply validated fixes without confirmation |
| `--no-auto-pull` | Disable automatic pulling of missing Ollama models |
| `--python` | Path to Python interpreter (defaults to active `VIRTUAL_ENV` or current Python) |
| `--allow-outside-cwd` | Allow reading and patching files outside the current working directory |

## Development & Tests

```bash
git clone https://github.com/varunms89/aidebug.git
cd aidebug
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## License

MIT

---

*Disclaimer: `ollama-aidebug` is an independent open-source project and is not affiliated with, sponsored by, or endorsed by Ollama.*
