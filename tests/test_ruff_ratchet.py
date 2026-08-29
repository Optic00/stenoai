import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import ruff_ratchet


class RuffRatchetTests(unittest.TestCase):
    FINDING_A = "a" * 64
    FINDING_B = "b" * 64

    def _baseline(self, root: Path, content: dict) -> Path:
        path = root / "baseline.json"
        path.write_text(json.dumps(content), encoding="utf-8")
        return path

    def test_matches_the_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            findings = {"src/example.py": {"F401": {self.FINDING_A: 2}}}
            baseline = self._baseline(root, findings)
            with patch.object(ruff_ratchet, "ruff_findings", return_value=findings):
                self.assertEqual(ruff_ratchet.check(baseline_path=baseline, update=False, root=root), 0)

    def test_new_finding_fails_without_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = self._baseline(root, {"src/example.py": {"F401": {self.FINDING_A: 1}}})
            findings = {"src/example.py": {"F401": {self.FINDING_A: 1, self.FINDING_B: 1}}}
            with patch.object(ruff_ratchet, "ruff_findings", return_value=findings):
                self.assertEqual(ruff_ratchet.check(baseline_path=baseline, update=False, root=root), 1)

    def test_burn_down_also_requires_an_explicit_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = self._baseline(root, {"src/example.py": {"F401": {self.FINDING_A: 2}}})
            findings = {"src/example.py": {"F401": {self.FINDING_A: 1}}}
            with patch.object(ruff_ratchet, "ruff_findings", return_value=findings):
                self.assertEqual(ruff_ratchet.check(baseline_path=baseline, update=False, root=root), 1)

    def test_update_writes_stably_sorted_current_findings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = self._baseline(root, {})
            findings = {
                "z.py": {"F841": {self.FINDING_B: 1}},
                "a.py": {
                    "F401": {self.FINDING_B: 1, self.FINDING_A: 2},
                    "E401": {self.FINDING_A: 1},
                },
            }
            with patch.object(ruff_ratchet, "ruff_findings", return_value=findings):
                self.assertEqual(ruff_ratchet.check(baseline_path=baseline, update=True, root=root), 0)
            self.assertEqual(json.loads(baseline.read_text(encoding="utf-8")), {
                "a.py": {
                    "E401": {self.FINDING_A: 1},
                    "F401": {self.FINDING_A: 2, self.FINDING_B: 1},
                },
                "z.py": {"F841": {self.FINDING_B: 1}},
            })

    def test_absolute_ruff_paths_are_normalized_to_posix_repo_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            filename = root / "src" / "example.py"
            filename.parent.mkdir()
            filename.write_text("x = 1\n", encoding="utf-8")
            version = unittest.mock.Mock(returncode=0, stdout="ruff 0.15.21\n", stderr="")
            diagnostics = unittest.mock.Mock(
                returncode=1,
                stdout=json.dumps([{
                    "filename": str(filename),
                    "code": "F401",
                    "message": "example finding",
                    "location": {"row": 1, "column": 1},
                    "end_location": {"row": 1, "column": 2},
                }]),
                stderr="",
            )
            with patch.object(ruff_ratchet, "_run", side_effect=[version, diagnostics]):
                findings = ruff_ratchet.ruff_findings(root)
            identities = findings["src/example.py"]["F401"]
            self.assertEqual(sum(identities.values()), 1)
            self.assertRegex(next(iter(identities)), r"^[0-9a-f]{64}$")

    def test_clean_ruff_exit_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version = unittest.mock.Mock(returncode=0, stdout="ruff 0.15.21\n", stderr="")
            clean = unittest.mock.Mock(returncode=0, stdout="[]", stderr="")
            with patch.object(ruff_ratchet, "_run", side_effect=[version, clean]):
                self.assertEqual(ruff_ratchet.ruff_findings(root), {})

    def test_baseline_rejects_non_posix_or_absolute_filenames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = self._baseline(root, {"C:\\repo\\file.py": {"F401": 1}})
            with self.assertRaisesRegex(RuntimeError, "POSIX relative"):
                ruff_ratchet.read_baseline(baseline)

    def test_wrong_ruff_version_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_version = unittest.mock.Mock(returncode=0, stdout="ruff 0.16.0\n", stderr="")
            with patch.object(ruff_ratchet, "_run", return_value=bad_version):
                with self.assertRaisesRegex(RuntimeError, "Expected ruff 0.15.21"):
                    ruff_ratchet.ruff_findings(root)

    def test_same_count_with_a_different_concrete_finding_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            filename = root / "src" / "example.py"
            filename.parent.mkdir()

            def findings(source: str, message: str):
                filename.write_text(source, encoding="utf-8")
                version = unittest.mock.Mock(returncode=0, stdout="ruff 0.15.21\n", stderr="")
                diagnostics = unittest.mock.Mock(
                    returncode=1,
                    stdout=json.dumps([{
                        "filename": str(filename),
                        "code": "F401",
                        "message": message,
                        "location": {"row": 1, "column": 8},
                        "end_location": {"row": 1, "column": len(source)},
                    }]),
                    stderr="",
                )
                with patch.object(ruff_ratchet, "_run", side_effect=[version, diagnostics]):
                    return ruff_ratchet.ruff_findings(root)

            before = findings("import os\n", "`os` imported but unused")
            after = findings("import sys\n", "`sys` imported but unused")

            self.assertNotEqual(before, after)
            self.assertTrue(ruff_ratchet.differences(before, after))

    def test_pure_line_shift_keeps_the_same_concrete_finding_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            filename = root / "src" / "example.py"
            filename.parent.mkdir()

            def findings(source: str, row: int):
                filename.write_text(source, encoding="utf-8")
                version = unittest.mock.Mock(returncode=0, stdout="ruff 0.15.21\n", stderr="")
                diagnostics = unittest.mock.Mock(
                    returncode=1,
                    stdout=json.dumps([{
                        "filename": str(filename),
                        "code": "F401",
                        "message": "`os` imported but unused",
                        "location": {"row": row, "column": 8},
                        "end_location": {"row": row, "column": 10},
                    }]),
                    stderr="",
                )
                with patch.object(ruff_ratchet, "_run", side_effect=[version, diagnostics]):
                    return ruff_ratchet.ruff_findings(root)

            before = findings("import os\n", 1)
            after = findings("\n\nimport os\n", 3)

            self.assertEqual(before, after)
            self.assertEqual(ruff_ratchet.differences(before, after), [])

    def test_same_finding_text_moved_between_function_scopes_is_a_swap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            filename = root / "src" / "example.py"
            filename.parent.mkdir()

            def findings(source: str, row: int):
                filename.write_text(source, encoding="utf-8")
                version = unittest.mock.Mock(returncode=0, stdout="ruff 0.15.21\n", stderr="")
                diagnostics = unittest.mock.Mock(
                    returncode=1,
                    stdout=json.dumps([{
                        "filename": str(filename),
                        "code": "F841",
                        "message": "Local variable `unused` is assigned to but never used",
                        "location": {"row": row, "column": 5},
                        "end_location": {"row": row, "column": 11},
                    }]),
                    stderr="",
                )
                with patch.object(ruff_ratchet, "_run", side_effect=[version, diagnostics]):
                    return ruff_ratchet.ruff_findings(root)

            before = findings(
                "def old_scope():\n"
                "    unused = 1\n\n"
                "def new_scope():\n"
                "    return 1\n",
                2,
            )
            after = findings(
                "def old_scope():\n"
                "    return 1\n\n"
                "def new_scope():\n"
                "    unused = 1\n",
                5,
            )

            self.assertNotEqual(before, after)
            self.assertTrue(ruff_ratchet.differences(before, after))

    def test_formatting_outside_ruff_span_keeps_the_same_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            filename = root / "src" / "example.py"
            filename.parent.mkdir()

            def findings(source: str):
                filename.write_text(source, encoding="utf-8")
                version = unittest.mock.Mock(returncode=0, stdout="ruff 0.15.21\n", stderr="")
                diagnostics = unittest.mock.Mock(
                    returncode=1,
                    stdout=json.dumps([{
                        "filename": str(filename),
                        "code": "F841",
                        "message": "Local variable `unused` is assigned to but never used",
                        "location": {"row": 2, "column": 5},
                        "end_location": {"row": 2, "column": 11},
                    }]),
                    stderr="",
                )
                with patch.object(ruff_ratchet, "_run", side_effect=[version, diagnostics]):
                    return ruff_ratchet.ruff_findings(root)

            before = findings("def f():\n    unused = dict(a = 1)\n")
            after = findings("def f():\n    unused = dict(a=1)\n")

            self.assertEqual(before, after)
            self.assertEqual(ruff_ratchet.differences(before, after), [])

    def test_spacing_change_inside_e401_span_keeps_the_same_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            filename = root / "src" / "example.py"
            filename.parent.mkdir()

            def findings(source: str):
                filename.write_text(source, encoding="utf-8")
                line = source.rstrip("\n")
                version = unittest.mock.Mock(returncode=0, stdout="ruff 0.15.21\n", stderr="")
                diagnostics = unittest.mock.Mock(
                    returncode=1,
                    stdout=json.dumps([{
                        "filename": str(filename),
                        "code": "E401",
                        "message": "Multiple imports on one line",
                        "location": {"row": 1, "column": 1},
                        "end_location": {"row": 1, "column": len(line) + 1},
                    }]),
                    stderr="",
                )
                with patch.object(ruff_ratchet, "_run", side_effect=[version, diagnostics]):
                    return ruff_ratchet.ruff_findings(root)

            before = findings("import os,sys\n")
            after = findings("import os, sys\n")

            self.assertEqual(before, after)
            self.assertEqual(ruff_ratchet.differences(before, after), [])

    def test_quote_change_inside_f541_span_keeps_the_same_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            filename = root / "src" / "example.py"
            filename.parent.mkdir()

            def findings(source: str):
                filename.write_text(source, encoding="utf-8")
                line = source.splitlines()[1]
                version = unittest.mock.Mock(returncode=0, stdout="ruff 0.15.21\n", stderr="")
                diagnostics = unittest.mock.Mock(
                    returncode=1,
                    stdout=json.dumps([{
                        "filename": str(filename),
                        "code": "F541",
                        "message": "f-string without any placeholders",
                        "location": {"row": 2, "column": line.index("f") + 1},
                        "end_location": {"row": 2, "column": line.rindex(")") + 1},
                    }]),
                    stderr="",
                )
                with patch.object(ruff_ratchet, "_run", side_effect=[version, diagnostics]):
                    return ruff_ratchet.ruff_findings(root)

            before = findings('def f():\n    print(f"constant")\n')
            after = findings("def f():\n    print(f'constant')\n")

            self.assertEqual(before, after)
            self.assertEqual(ruff_ratchet.differences(before, after), [])

    def test_same_named_platform_scopes_have_distinct_identities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            filename = root / "src" / "example.py"
            filename.parent.mkdir()

            def findings(source: str, row: int):
                filename.write_text(source, encoding="utf-8")
                version = unittest.mock.Mock(returncode=0, stdout="ruff 0.15.21\n", stderr="")
                diagnostics = unittest.mock.Mock(
                    returncode=1,
                    stdout=json.dumps([{
                        "filename": str(filename),
                        "code": "F841",
                        "message": "Local variable `unused` is assigned to but never used",
                        "location": {"row": row, "column": 13},
                        "end_location": {"row": row, "column": 19},
                    }]),
                    stderr="",
                )
                with patch.object(ruff_ratchet, "_run", side_effect=[version, diagnostics]):
                    return ruff_ratchet.ruff_findings(root)

            before = findings(
                "import sys\n"
                "if sys.platform == 'win32':\n"
                "    class Backend:\n"
                "        def start(self):\n"
                "            unused = 1\n"
                "else:\n"
                "    class Backend:\n"
                "        def start(self):\n"
                "            return 1\n",
                5,
            )
            after = findings(
                "import sys\n"
                "if sys.platform == 'win32':\n"
                "    class Backend:\n"
                "        def start(self):\n"
                "            return 1\n"
                "else:\n"
                "    class Backend:\n"
                "        def start(self):\n"
                "            unused = 1\n",
                9,
            )

            self.assertNotEqual(before, after)
            self.assertTrue(ruff_ratchet.differences(before, after))

    def test_protected_t1_job_directly_runs_both_new_lint_gates(self):
        workflow = (Path(__file__).parents[1] / ".github" / "workflows" / "e2e.yml").read_text()
        t1_body = workflow.split("  t1-renderer:\n", 1)[1].split("\n  lint-renderer:\n", 1)[0]

        self.assertIn("npm run lint:main", t1_body)
        self.assertIn("python scripts/ruff_ratchet.py", t1_body)
        self.assertNotIn("continue-on-error:", t1_body)
