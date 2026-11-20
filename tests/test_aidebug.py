"""Unit and end-to-end tests for aidebug."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from aidebug.context import describe_workspace, extract_referenced_files
from aidebug.loop import _normalize_suggested_command, run_debug_loop
from aidebug.ollama_client import (
    DEFAULT_MODEL,
    Diagnosis,
    OllamaClient,
    OllamaError,
    ProposedEdit,
    parse_diagnosis,
)
from aidebug.patcher import (
    apply_changes,
    rollback_changes,
    stage_edits,
    validate_no_truncation,
)
from aidebug.runner import resolve_command, run_command


class FakeOllamaClient:
    def __init__(self, diagnosis: Diagnosis) -> None:
        self._diagnosis = diagnosis
        self.calls = 0

    def select_model(self, preferred=None) -> str:
        return preferred or "fake-coder:7b"

    def diagnose(self, **kwargs) -> Diagnosis:
        self.calls += 1
        return self._diagnosis


class TestRunnerAndContext(unittest.TestCase):
    def test_resolve_command_detects_python_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script = tmp_path / "hello.py"
            script.write_text("print('hi')\n", encoding="utf-8")
            cmd = resolve_command(["hello.py", "--flag"], cwd=tmp_path)
            self.assertEqual(cmd, [sys.executable, "hello.py", "--flag"])

    def test_resolve_command_rejects_non_python_command(self) -> None:
        with self.assertRaises(ValueError):
            resolve_command(["npm", "run", "dev"])

    def test_resolve_command_rejects_missing_python_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                resolve_command(["missing.py"], cwd=Path(tmp))

    def test_extract_referenced_files_from_python_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            app = tmp_path / "app.py"
            app.write_text("x = 1 / 0\n", encoding="utf-8")
            output = (
                "Traceback (most recent call last):\n"
                f'  File "{app}", line 1, in <module>\n'
                "ZeroDivisionError: division by zero\n"
            )
            contexts = extract_referenced_files(output, ["python", "app.py"], tmp_path)
            self.assertEqual(len(contexts), 1)
            self.assertEqual(contexts[0].rel_path, "app.py")
            self.assertEqual(contexts[0].focus_lines, [1])

    def test_extract_bare_py_filename_and_workspace_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            helper = tmp_path / "helper.py"
            helper.write_text("def f():\n    return 1\n", encoding="utf-8")
            output = "SyntaxError in helper.py: unexpected EOF\n"
            contexts = extract_referenced_files(output, ["python", "main.py"], tmp_path)
            self.assertEqual(len(contexts), 1)
            self.assertEqual(contexts[0].rel_path, "helper.py")
            summary = describe_workspace(tmp_path)
            self.assertIn("helper.py", summary)

    def test_extract_referenced_files_outside_cwd_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            work_dir = Path(tmp1)
            ext_script = Path(tmp2) / "external.py"
            ext_script.write_text("x = 1 / 0\n", encoding="utf-8")
            output = f'  File "{ext_script}", line 1, in <module>\nZeroDivisionError\n'

            denied = extract_referenced_files(output, ["python", str(ext_script)], work_dir)
            self.assertEqual(denied, [])

            allowed = extract_referenced_files(
                output,
                ["python", str(ext_script)],
                work_dir,
                allow_outside_cwd=True,
            )
            self.assertEqual(len(allowed), 1)
            self.assertEqual(allowed[0].path, ext_script.resolve())
            self.assertEqual(allowed[0].focus_lines, [1])

    def test_run_command_streams_prompt_without_newline(self) -> None:
        res = run_command(
            [sys.executable, "-c", "import sys; sys.stdout.write('Enter: '); sys.stdout.flush()"],
            stream_output=False,
        )
        self.assertTrue(res.succeeded)
        self.assertEqual(res.stdout, "Enter: ")


class TestOllamaParsingAndPatching(unittest.TestCase):
    def test_parse_diagnosis_with_code_fences(self) -> None:
        raw = """```json
        {
          "root_cause": "Division by zero",
          "fix_summary": "Guard denominator",
          "edits": [
            {"file": "calc.py", "search": "return a / b", "replace": "return a / b if b else 0"}
          ],
          "command_suggestion": null
        }
        ```"""
        diag = parse_diagnosis(raw)
        self.assertEqual(diag.root_cause, "Division by zero")
        self.assertEqual(len(diag.edits), 1)
        self.assertEqual(diag.edits[0].file, "calc.py")

    def test_stage_apply_and_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            target = tmp_path / "calc.py"
            target.write_text("def div(a, b):\n    return a / b\n", encoding="utf-8")

            edits = [
                ProposedEdit(
                    file="calc.py",
                    search="    return a / b",
                    replace="    return a / b if b else 0",
                )
            ]
            staged, warnings = stage_edits(edits, cwd=tmp_path)
            self.assertEqual(warnings, [])
            self.assertEqual(len(staged), 1)
            self.assertIn("-    return a / b", staged[0].unified_diff(color=False))
            self.assertIn("+    return a / b if b else 0", staged[0].unified_diff(color=False))

            apply_changes(staged)
            self.assertIn("if b else 0", target.read_text(encoding="utf-8"))

            rollback_changes(staged)
            self.assertEqual(target.read_text(encoding="utf-8"), "def div(a, b):\n    return a / b\n")

    def test_refuses_edit_outside_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            edits = [ProposedEdit(file="/etc/hosts", search="a", replace="b")]
            staged, warnings = stage_edits(edits, cwd=tmp_path)
            self.assertEqual(staged, [])
            self.assertEqual(len(warnings), 1)
            self.assertIn("outside working directory", warnings[0])

    def test_validate_no_truncation_catches_dropped_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            target = tmp_path / "script.py"
            original = "\n".join(f"x{i} = {i}" for i in range(10)) + "\n"
            target.write_text(original, encoding="utf-8")

            edits = [ProposedEdit(file="script.py", content="x0 = 0\nx1 = 1\n")]
            staged, _ = stage_edits(edits, cwd=tmp_path)
            errors = validate_no_truncation(staged)
            self.assertEqual(len(errors), 1)
            self.assertIn("likely truncated", errors[0])

    def test_ensure_model_auto_pulls_default_when_none_installed(self) -> None:
        pulled = []

        class StubClient(OllamaClient):
            def list_models(self):
                return []

            def pull_model(self, model: str, stream_progress: bool = True) -> None:
                pulled.append(model)

        client = StubClient()
        chosen = client.ensure_model(auto_pull=True, stream_progress=False)
        self.assertEqual(chosen, DEFAULT_MODEL)
        self.assertEqual(pulled, [DEFAULT_MODEL])

    def test_ensure_model_skips_pull_when_already_installed(self) -> None:
        pulled = []

        class StubClient(OllamaClient):
            def list_models(self):
                return ["qwen2.5-coder:7b", "llama3:latest"]

            def pull_model(self, model: str, stream_progress: bool = True) -> None:
                pulled.append(model)

        client = StubClient()
        chosen = client.ensure_model(auto_pull=True, stream_progress=False)
        self.assertEqual(chosen, "qwen2.5-coder:7b")
        self.assertEqual(pulled, [])

    def test_ensure_model_pulls_requested_missing_model(self) -> None:
        pulled = []

        class StubClient(OllamaClient):
            def list_models(self):
                return ["llama3:latest"]

            def pull_model(self, model: str, stream_progress: bool = True) -> None:
                pulled.append(model)

        client = StubClient()
        chosen = client.ensure_model(preferred="deepseek-coder:6.7b", auto_pull=True, stream_progress=False)
        self.assertEqual(chosen, "deepseek-coder:6.7b")
        self.assertEqual(pulled, ["deepseek-coder:6.7b"])

    def test_ensure_model_raises_when_auto_pull_disabled(self) -> None:
        class StubClient(OllamaClient):
            def list_models(self):
                return []

        client = StubClient()
        with self.assertRaises(OllamaError):
            client.ensure_model(auto_pull=False)

    def test_ollama_host_normalization_prepends_http(self) -> None:
        client = OllamaClient(host="127.0.0.1:11434")
        self.assertEqual(client.host, "http://127.0.0.1:11434")

    def test_parse_diagnosis_strips_line_number_prefixes(self) -> None:
        raw = """{
          "root_cause": "Missing colon",
          "fix_summary": "Add colon",
          "edits": [
            {
              "file": "a.py",
              "search": "   1 | def f(x)\\n   2 |     return x",
              "replace": "   1 | def f(x):\\n   2 |     return x"
            }
          ],
          "command_suggestion": null
        }"""
        diag = parse_diagnosis(raw)
        self.assertEqual(diag.edits[0].search, "def f(x)\n    return x")
        self.assertEqual(diag.edits[0].replace, "def f(x):\n    return x")

    def test_ambiguous_search_uses_focus_lines_or_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            target = tmp_path / "dup.py"
            target.write_text(
                "def a():\n    return x\n\ndef b():\n    return x\n",
                encoding="utf-8",
            )
            edit = ProposedEdit(file="dup.py", search="    return x", replace="    return 42")

            # Without focus_lines, duplicate match is rejected as ambiguous
            staged, warnings = stage_edits([edit], cwd=tmp_path)
            self.assertEqual(staged, [])
            self.assertEqual(len(warnings), 1)
            self.assertIn("Ambiguous", warnings[0])

            # With focus_lines pointing at line 5, second occurrence is patched
            staged2, warnings2 = stage_edits(
                [edit],
                cwd=tmp_path,
                focus_lines_by_path={target.resolve(): [5]},
            )
            self.assertEqual(warnings2, [])
            self.assertEqual(len(staged2), 1)
            self.assertEqual(
                staged2[0].updated,
                "def a():\n    return x\n\ndef b():\n    return 42\n",
            )

    def test_stage_edits_handles_diff_prefix_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            target = tmp_path / "app.py"
            target.write_text("x = 1 / 0\n", encoding="utf-8")
            staged, warnings = stage_edits(
                [ProposedEdit(file="a/app.py", search="1 / 0", replace="1 / 2")],
                cwd=tmp_path,
            )
            self.assertEqual(warnings, [])
            self.assertEqual(len(staged), 1)
            self.assertEqual(staged[0].updated, "x = 1 / 2\n")

    def test_stage_edits_indentation_normalized_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            target = tmp_path / "app.py"
            target.write_text("def f(x):\n    return x / 0\n", encoding="utf-8")
            # Model omitted leading 4 spaces in search/replace
            staged, warnings = stage_edits(
                [ProposedEdit(file="app.py", search="return x / 0", replace="return x / 2")],
                cwd=tmp_path,
            )
            self.assertEqual(warnings, [])
            self.assertEqual(len(staged), 1)
            self.assertEqual(staged[0].updated, "def f(x):\n    return x / 2\n")

    def test_stage_edits_line_range_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            target = tmp_path / "app.py"
            target.write_text("a = 1\nb = 2\n", encoding="utf-8")
            staged, warnings = stage_edits(
                [
                    ProposedEdit(
                        file="app.py",
                        search="nonexistent",
                        replace="import json\na = 1",
                        start_line=1,
                        end_line=1,
                    )
                ],
                cwd=tmp_path,
            )
            self.assertEqual(warnings, [])
            self.assertEqual(len(staged), 1)
            self.assertEqual(staged[0].updated, "import json\na = 1\nb = 2\n")

    def test_stage_edits_empty_search_with_replace_inserts_after_docstring(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            target = tmp_path / "app.py"
            target.write_text('"""Docstring."""\n\nx = json.loads("{}")\n', encoding="utf-8")
            staged, warnings = stage_edits(
                [ProposedEdit(file="app.py", search="", replace="import json")],
                cwd=tmp_path,
            )
            self.assertEqual(warnings, [])
            self.assertEqual(len(staged), 1)
            self.assertEqual(
                staged[0].updated,
                '"""Docstring."""\nimport json\n\nx = json.loads("{}")\n',
            )


class TestDebugLoopEndToEnd(unittest.TestCase):
    def test_run_diagnose_confirm_apply_and_rerun_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script = tmp_path / "buggy.py"
            script.write_text("x = 10 / 0\nprint(x)\n", encoding="utf-8")

            fake_diag = Diagnosis(
                root_cause="Division by zero on line 1.",
                fix_summary="Divide by 2 instead of 0.",
                edits=[
                    ProposedEdit(
                        file="buggy.py",
                        search="x = 10 / 0",
                        replace="x = 10 / 2",
                    )
                ],
            )
            client = FakeOllamaClient(fake_diag)

            rc = run_debug_loop(
                ["buggy.py"],
                cwd=tmp_path,
                client=client,
                input_fn=lambda _prompt: "y",
            )
            self.assertEqual(rc, 0)
            self.assertEqual(client.calls, 1)
            self.assertIn("x = 10 / 2", script.read_text(encoding="utf-8"))

    def test_declining_fix_leaves_file_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script = tmp_path / "buggy.py"
            original = "x = 10 / 0\n"
            script.write_text(original, encoding="utf-8")

            fake_diag = Diagnosis(
                root_cause="Division by zero.",
                fix_summary="Change 0 to 2.",
                edits=[ProposedEdit(file="buggy.py", search="10 / 0", replace="10 / 2")],
            )
            client = FakeOllamaClient(fake_diag)

            rc = run_debug_loop(
                ["buggy.py"],
                cwd=tmp_path,
                client=client,
                input_fn=lambda _prompt: "n",
            )
            self.assertNotEqual(rc, 0)
            self.assertEqual(script.read_text(encoding="utf-8"), original)

    def test_unlimited_retries_runs_until_clean_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script = tmp_path / "countdown.py"
            script.write_text("n = 6\nassert n == 0\nprint('done')\n", encoding="utf-8")

            class CountdownClient:
                def __init__(self) -> None:
                    self.calls = 0

                def select_model(self, preferred=None) -> str:
                    return "fake-coder:7b"

                def diagnose(self, **kwargs) -> Diagnosis:
                    self.calls += 1
                    cur = 6 - (self.calls - 1)
                    nxt = cur - 1
                    return Diagnosis(
                        root_cause=f"n is {cur}, expected 0",
                        fix_summary=f"Decrement n to {nxt}",
                        edits=[
                            ProposedEdit(
                                file="countdown.py",
                                search=f"n = {cur}",
                                replace=f"n = {nxt}",
                            )
                        ],
                    )

            client = CountdownClient()
            rc = run_debug_loop(
                ["countdown.py"],
                cwd=tmp_path,
                client=client,
                auto_yes=True,
            )
            self.assertEqual(rc, 0)
            self.assertEqual(client.calls, 6)
            self.assertIn("n = 0", script.read_text(encoding="utf-8"))

    def test_auto_retries_when_model_produces_invalid_syntax(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script = tmp_path / "broken.py"
            script.write_text("def f(x)\n    return x\nprint(f(5))\n", encoding="utf-8")

            class FlakySyntaxClient:
                def __init__(self) -> None:
                    self.calls = 0

                def select_model(self, preferred=None) -> str:
                    return "fake-coder:7b"

                def diagnose(self, **kwargs) -> Diagnosis:
                    self.calls += 1
                    if self.calls == 1:
                        return Diagnosis(
                            root_cause="Missing colon",
                            fix_summary="Bad colon patch",
                            edits=[
                                ProposedEdit(
                                    file="broken.py",
                                    content="def f(x):\n    :\n    return x\nprint(f(5))\n",
                                )
                            ],
                        )
                    return Diagnosis(
                        root_cause="Missing colon",
                        fix_summary="Valid colon patch",
                        edits=[
                            ProposedEdit(
                                file="broken.py",
                                content="def f(x):\n    return x\nprint(f(5))\n",
                            )
                        ],
                    )

            client = FlakySyntaxClient()
            rc = run_debug_loop(
                ["broken.py"],
                cwd=tmp_path,
                client=client,
                auto_yes=True,
            )
            self.assertEqual(rc, 0)
            self.assertEqual(client.calls, 2)
            self.assertEqual(
                script.read_text(encoding="utf-8"),
                "def f(x):\n    return x\nprint(f(5))\n",
            )

    def test_max_retries_zero_runs_once_without_diagnosing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script = tmp_path / "fail.py"
            script.write_text("1 / 0\n", encoding="utf-8")

            client = FakeOllamaClient(Diagnosis(root_cause="x", fix_summary="y"))
            rc = run_debug_loop(
                ["fail.py"],
                cwd=tmp_path,
                max_retries=0,
                client=client,
                auto_yes=True,
            )
            self.assertNotEqual(rc, 0)
            self.assertEqual(client.calls, 0)

    def test_auto_retries_when_search_block_fails_to_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script = tmp_path / "buggy.py"
            script.write_text("x = 10 / 0\nprint(x)\n", encoding="utf-8")

            class MismatchedSearchClient:
                def __init__(self) -> None:
                    self.calls = 0
                    self.hints = []

                def select_model(self, preferred=None) -> str:
                    return "fake-coder:7b"

                def diagnose(self, **kwargs) -> Diagnosis:
                    self.calls += 1
                    self.hints.append(kwargs.get("extra_hint"))
                    if self.calls == 1:
                        return Diagnosis(
                            root_cause="Zero division",
                            fix_summary="Bad search",
                            edits=[
                                ProposedEdit(
                                    file="buggy.py",
                                    search="nonexistent_line = 0",
                                    replace="x = 10 / 2",
                                )
                            ],
                        )
                    return Diagnosis(
                        root_cause="Zero division",
                        fix_summary="Good search",
                        edits=[
                            ProposedEdit(
                                file="buggy.py",
                                search="x = 10 / 0",
                                replace="x = 10 / 2",
                            )
                        ],
                    )

            client = MismatchedSearchClient()
            rc = run_debug_loop(
                ["buggy.py"],
                cwd=tmp_path,
                client=client,
                auto_yes=True,
            )
            self.assertEqual(rc, 0)
            self.assertEqual(client.calls, 2)
            self.assertIn("did not match", client.hints[1] or "")

    def test_auto_yes_refuses_to_apply_persistently_invalid_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script = tmp_path / "broken.py"
            original = "x = 1 / 0\n"
            script.write_text(original, encoding="utf-8")

            bad_diag = Diagnosis(
                root_cause="Broken",
                fix_summary="Still invalid syntax",
                edits=[ProposedEdit(file="broken.py", content="def broken(:\n")],
            )
            client = FakeOllamaClient(bad_diag)
            rc = run_debug_loop(
                ["broken.py"],
                cwd=tmp_path,
                client=client,
                auto_yes=True,
            )
            self.assertNotEqual(rc, 0)
            self.assertEqual(client.calls, 3)  # 1 initial + 2 validation retries
            self.assertEqual(script.read_text(encoding="utf-8"), original)

    def test_eof_on_input_exits_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script = tmp_path / "buggy.py"
            script.write_text("1 / 0\n", encoding="utf-8")
            client = FakeOllamaClient(
                Diagnosis(
                    root_cause="div0",
                    fix_summary="fix",
                    edits=[ProposedEdit(file="buggy.py", search="1 / 0", replace="1 / 1")],
                )
            )

            def raise_eof(_prompt: str) -> str:
                raise EOFError

            rc = run_debug_loop(["buggy.py"], cwd=tmp_path, client=client, input_fn=raise_eof)
            self.assertNotEqual(rc, 0)

    def test_normalize_suggested_pip_command(self) -> None:
        self.assertEqual(
            _normalize_suggested_command("pip install requests", "/venv/bin/python"),
            ["/venv/bin/python", "-m", "pip", "install", "requests"],
        )
        self.assertEqual(
            _normalize_suggested_command("pip3 install -U numpy", "/venv/bin/python"),
            ["/venv/bin/python", "-m", "pip", "install", "-U", "numpy"],
        )


if __name__ == "__main__":
    unittest.main()
