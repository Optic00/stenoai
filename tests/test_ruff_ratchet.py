import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import ruff_ratchet


class RuffRatchetTests(unittest.TestCase):
    def _baseline(self, root: Path, content: dict) -> Path:
        path = root / "baseline.json"
        path.write_text(json.dumps(content), encoding="utf-8")
        return path

    def test_matches_the_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = self._baseline(root, {"src/example.py": {"F401": 2}})
            with patch.object(ruff_ratchet, "ruff_findings", return_value={"src/example.py": {"F401": 2}}):
                self.assertEqual(ruff_ratchet.check(baseline_path=baseline, update=False, root=root), 0)

    def test_new_finding_fails_without_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = self._baseline(root, {"src/example.py": {"F401": 1}})
            with patch.object(ruff_ratchet, "ruff_findings", return_value={"src/example.py": {"F401": 2}}):
                self.assertEqual(ruff_ratchet.check(baseline_path=baseline, update=False, root=root), 1)

    def test_burn_down_also_requires_an_explicit_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = self._baseline(root, {"src/example.py": {"F401": 2}})
            with patch.object(ruff_ratchet, "ruff_findings", return_value={"src/example.py": {"F401": 1}}):
                self.assertEqual(ruff_ratchet.check(baseline_path=baseline, update=False, root=root), 1)

    def test_update_writes_stably_sorted_current_findings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = self._baseline(root, {})
            findings = {
                "z.py": {"F841": 1},
                "a.py": {"F401": 2, "E401": 1},
            }
            with patch.object(ruff_ratchet, "ruff_findings", return_value=findings):
                self.assertEqual(ruff_ratchet.check(baseline_path=baseline, update=True, root=root), 0)
            self.assertEqual(json.loads(baseline.read_text(encoding="utf-8")), {
                "a.py": {"E401": 1, "F401": 2},
                "z.py": {"F841": 1},
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
                stdout=json.dumps([{"filename": str(filename), "code": "F401"}]),
                stderr="",
            )
            with patch.object(ruff_ratchet, "_run", side_effect=[version, diagnostics]):
                self.assertEqual(ruff_ratchet.ruff_findings(root), {"src/example.py": {"F401": 1}})

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
