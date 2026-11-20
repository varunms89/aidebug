"""Local Ollama HTTP client for error diagnosis and patch generation."""

from __future__ import annotations

import http.client
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .context import SourceContext

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5-coder:7b"

# Preference order when auto-selecting an installed local model.
PREFERRED_MODEL_PREFIXES = (
    "qwen2.5-coder",
    "qwen3-coder",
    "deepseek-coder",
    "codellama",
    "starcoder",
    "qwen2.5",
    "qwen3",
    "llama3",
    "gemma",
    "mistral",
)


class OllamaError(RuntimeError):
    """Raised when communicating with the local Ollama server fails."""


@dataclass
class ProposedEdit:
    file: str
    search: str = ""
    replace: str = ""
    content: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None


@dataclass
class Diagnosis:
    root_cause: str
    fix_summary: str
    edits: List[ProposedEdit] = field(default_factory=list)
    command_suggestion: Optional[str] = None
    raw_response: str = ""


SYSTEM_PROMPT = """You are an expert Python debugger. You are given:
1. The Python script command that was executed and failed.
2. The exit code, stdout, and stderr (including the Python traceback).
3. Relevant `.py` source files from the workspace (with line numbers for reference).

Diagnose the root cause and fix the bug(s) in the relevant file(s) so the script runs cleanly and produces correct results.
Respond ONLY with a valid JSON object matching this schema:
{
  "root_cause": "Clear, concise explanation of why the Python error happened.",
  "fix_summary": "Short summary of what the proposed fix changes.",
  "edits": [
    {
      "file": "relative/path/to/file.py",
      "search": "exact contiguous lines from the existing file (NO line numbers)",
      "replace": "corrected replacement lines (NO line numbers)",
      "content": null
    }
  ],
  "command_suggestion": null
}

Rules:
- ALWAYS prefer "search" and "replace" (with "content": null) so only the broken lines change.
- "search" MUST be copied verbatim from lines that ALREADY EXIST in the file (never put a new line that doesn't exist yet into "search").
- To INSERT a new line (such as a missing `import`), put an EXISTING line from the file in "search", and put the new line plus that existing line in "replace" (e.g., `"search": "def load_tasks(raw: str) -> list:"`, `"replace": "import json\\n\\ndef load_tasks(raw: str) -> list:"`).
- NEVER include the "   1 | " line number prefixes in "search", "replace", or "content".
- NEVER use "..." or placeholders to abbreviate code.
- NEVER wrap code in markdown code fences (no ```python).
- Make sure the resulting Python file has 100% valid Python syntax and proper indentation.
- If the error is a missing third-party package (`ModuleNotFoundError`), leave "edits" as [] and set "command_suggestion" to `pip install <package>`.
- NEVER repeat a previously attempted command or edit that already failed.
"""


DIAGNOSIS_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "root_cause": {"type": "string"},
        "fix_summary": {"type": "string"},
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "search": {"type": "string"},
                    "replace": {"type": "string"},
                    "content": {"type": ["string", "null"]},
                    "start_line": {"type": ["integer", "null"]},
                    "end_line": {"type": ["integer", "null"]},
                },
                "required": ["file", "search", "replace"],
            },
        },
        "command_suggestion": {"type": ["string", "null"]},
    },
    "required": ["root_cause", "fix_summary", "edits", "command_suggestion"],
}


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Parse JSON from model output, tolerating markdown code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: locate the outermost {...} block.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    raise OllamaError(f"Model did not return valid JSON. Raw output:\n{text[:1000]}")


_LINE_NUMBER_PREFIX_RE = re.compile(r"^\s*\d+\s\|\s?")


def _strip_line_number_prefixes(text: str) -> str:
    """Strip accidental '  12 | ' line-number prefixes copied from prompt context."""
    lines = text.splitlines(keepends=True)
    non_empty = [ln for ln in lines if ln.strip()]
    if not non_empty:
        return text
    matched = sum(1 for ln in non_empty if _LINE_NUMBER_PREFIX_RE.match(ln))
    if matched == len(non_empty):
        return "".join(_LINE_NUMBER_PREFIX_RE.sub("", ln, count=1) for ln in lines)
    return text


def _normalize_host(raw_host: Optional[str]) -> str:
    host = (raw_host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).strip().rstrip("/")
    if not host:
        host = DEFAULT_OLLAMA_HOST
    if "://" not in host:
        host = f"http://{host}"
    return host


def parse_diagnosis(raw_text: str) -> Diagnosis:
    """Convert raw Ollama JSON text into a structured Diagnosis."""
    data = _extract_json_object(raw_text)
    edits: List[ProposedEdit] = []

    raw_edits = data.get("edits") or []
    if isinstance(raw_edits, list):
        for item in raw_edits:
            if not isinstance(item, dict):
                continue
            file_path = str(item.get("file") or "").strip()
            if not file_path:
                continue
            content = item.get("content")
            if content is not None:
                cleaned_content = str(content)
                if cleaned_content.lstrip().startswith("```"):
                    cleaned_content = re.sub(r"^\s*```(?:python|py)?\s*\n?", "", cleaned_content)
                    cleaned_content = re.sub(r"\n?```\s*$", "", cleaned_content)
                cleaned_content = _strip_line_number_prefixes(cleaned_content)
                if cleaned_content and not cleaned_content.endswith("\n"):
                    cleaned_content += "\n"
                content = cleaned_content
            search_str = _strip_line_number_prefixes(str(item.get("search") or ""))
            replace_str = _strip_line_number_prefixes(str(item.get("replace") or ""))
            raw_start = item.get("start_line")
            raw_end = item.get("end_line")
            start_line = int(raw_start) if isinstance(raw_start, int) else None
            end_line = int(raw_end) if isinstance(raw_end, int) else None
            edits.append(
                ProposedEdit(
                    file=file_path,
                    search=search_str,
                    replace=replace_str,
                    content=content,
                    start_line=start_line,
                    end_line=end_line,
                )
            )

    cmd_suggestion = data.get("command_suggestion")
    if cmd_suggestion is not None:
        cmd_suggestion = str(cmd_suggestion).strip() or None

    return Diagnosis(
        root_cause=str(data.get("root_cause") or "Unable to determine root cause.").strip(),
        fix_summary=str(data.get("fix_summary") or "").strip(),
        edits=edits,
        command_suggestion=cmd_suggestion,
        raw_response=raw_text,
    )


class OllamaClient:
    def __init__(
        self,
        host: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        self.host = _normalize_host(host)
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.host}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                parsed = json.loads(body)
                if not isinstance(parsed, dict):
                    raise OllamaError(f"Unexpected non-object JSON response from Ollama at {url}.")
                return parsed
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama HTTP {exc.code} at {url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"Could not connect to Ollama at {self.host}. "
                f"Make sure Ollama is running (e.g., `ollama serve`). Details: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise OllamaError(f"Ollama request timed out after {self.timeout}s.") from exc
        except json.JSONDecodeError as exc:
            raise OllamaError(f"Invalid JSON received from Ollama at {url}: {exc}") from exc
        except (http.client.HTTPException, OSError) as exc:
            raise OllamaError(f"Error communicating with Ollama at {url}: {exc}") from exc

    def list_models(self) -> List[str]:
        data = self._request("GET", "/api/tags")
        models = data.get("models") or []
        return [m["name"] for m in models if isinstance(m, dict) and "name" in m]

    @staticmethod
    def _is_model_installed(model: str, installed: Sequence[str]) -> bool:
        target = model.strip().lower()
        candidates = {target}
        if ":" not in target:
            candidates.add(f"{target}:latest")
        return any(name.lower() in candidates for name in installed)

    def pull_model(self, model: str, stream_progress: bool = True) -> None:
        """Download a model into the local Ollama server via POST /api/pull."""
        url = f"{self.host}/api/pull"
        data = json.dumps({"name": model, "stream": True}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if stream_progress:
            print(f"[aidebug] Pulling local Ollama model '{model}' (first-time setup)...")

        last_status = ""
        try:
            with urllib.request.urlopen(req, timeout=600.0) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "error" in event:
                        raise OllamaError(f"Failed to pull '{model}': {event['error']}")
                    status = str(event.get("status") or "")
                    total = event.get("total")
                    completed = event.get("completed")
                    if stream_progress:
                        if total and completed is not None and total > 0:
                            pct = int((completed / total) * 100)
                            msg = f"[aidebug] {status}: {pct}%"
                            if msg != last_status:
                                print(msg, end="\r", flush=True)
                                last_status = msg
                        elif status and status != last_status:
                            if last_status.endswith("%"):
                                print()
                            print(f"[aidebug] {status}", flush=True)
                            last_status = status
            if stream_progress and last_status.endswith("%"):
                print()
            if stream_progress:
                print(f"[aidebug] Model '{model}' is ready.")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama HTTP {exc.code} pulling '{model}': {detail}") from exc
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"Could not connect to Ollama at {self.host} to pull '{model}'. "
                f"Make sure Ollama is running (e.g., `ollama serve`). Details: {exc.reason}"
            ) from exc

    def ensure_model(
        self,
        preferred: Optional[str] = None,
        auto_pull: bool = True,
        stream_progress: bool = True,
    ) -> str:
        """Return an available local Ollama model, pulling it automatically if missing."""
        installed = self.list_models()
        requested = preferred or os.environ.get("AIDEBUG_MODEL") or os.environ.get("OLLAMA_MODEL")

        if requested:
            if not self._is_model_installed(requested, installed):
                if not auto_pull:
                    raise OllamaError(
                        f"Ollama model '{requested}' is not installed. Run `ollama pull {requested}`."
                    )
                self.pull_model(requested, stream_progress=stream_progress)
            return requested

        for prefix in PREFERRED_MODEL_PREFIXES:
            for name in installed:
                if name.lower().startswith(prefix):
                    return name

        if installed:
            return installed[0]

        if not auto_pull:
            raise OllamaError(
                f"No local Ollama models found. Run `ollama pull {DEFAULT_MODEL}`."
            )

        self.pull_model(DEFAULT_MODEL, stream_progress=stream_progress)
        return DEFAULT_MODEL

    def select_model(self, preferred: Optional[str] = None) -> str:
        return self.ensure_model(preferred=preferred, auto_pull=True)

    def diagnose(
        self,
        model: str,
        command: Sequence[str],
        returncode: int,
        output: str,
        files: Sequence[SourceContext],
        workspace_summary: Optional[str] = None,
        previous_attempts: Optional[Sequence[str]] = None,
        extra_hint: Optional[str] = None,
    ) -> Diagnosis:
        # Cap error output so small local models stay within context limits.
        trimmed_output = output[-6000:] if len(output) > 6000 else output

        sections = [
            f"Command: {' '.join(command)}",
            f"Exit code: {returncode}",
        ]

        if workspace_summary:
            sections.append(workspace_summary)

        sections.append(f"Output:\n{trimmed_output or '(no output)'}")

        if files:
            sections.append("Relevant source files:")
            for src in files:
                focus = f" (error lines: {src.focus_lines})" if src.focus_lines else ""
                partial_note = (
                    f" [PARTIAL VIEW of {src.total_lines} total lines — MUST use search/replace, DO NOT use content]"
                    if src.is_truncated()
                    else ""
                )
                body = src.numbered_content() if src.content else "(empty file)"
                sections.append(
                    f"\n=== FILE: {src.rel_path}{focus}{partial_note} ===\n{body}"
                )

        if previous_attempts:
            history_text = "\n".join(f"- {item}" for item in previous_attempts)
            sections.append(
                f"Previous fix attempts that already failed (DO NOT repeat these):\n{history_text}"
            )

        if extra_hint:
            sections.append(f"\nAdditional user note: {extra_hint}")

        user_prompt = "\n\n".join(sections)

        payload = {
            "model": model,
            "stream": False,
            "format": DIAGNOSIS_JSON_SCHEMA,
            "keep_alive": "10m",
            "options": {
                "temperature": 0.1,
                "num_predict": 2048,
                "repeat_penalty": 1.1,
            },
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }

        response = self._request("POST", "/api/chat", payload)
        message = response.get("message") or {}
        content = str(message.get("content") or "")
        return parse_diagnosis(content)
