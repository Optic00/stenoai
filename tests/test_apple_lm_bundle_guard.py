import tempfile
import unittest
from pathlib import Path

from scripts.apple_lm_bundle_guard import resolve_apple_lm_sidecar


class AppleLMBundleGuardTests(unittest.TestCase):
    def test_release_build_fails_when_sidecar_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "steno-apple-lm"

            with self.assertRaisesRegex(FileNotFoundError, "steno-apple-lm"):
                resolve_apple_lm_sidecar(
                    missing,
                    platform="darwin",
                    required=True,
                )

    def test_development_build_may_omit_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "steno-apple-lm"

            self.assertIsNone(
                resolve_apple_lm_sidecar(
                    missing,
                    platform="darwin",
                    required=False,
                )
            )

    def test_existing_sidecar_must_be_executable(self):
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / "steno-apple-lm"
            sidecar.write_bytes(b"not executable")

            with self.assertRaisesRegex(PermissionError, "executable"):
                resolve_apple_lm_sidecar(sidecar, platform="darwin")

    def test_macos_build_accepts_executable_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / "steno-apple-lm"
            sidecar.write_bytes(b"executable")
            sidecar.chmod(0o755)

            self.assertEqual(
                resolve_apple_lm_sidecar(sidecar, platform="darwin"),
                sidecar,
            )

    def test_other_platforms_ignore_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "steno-apple-lm"

            self.assertIsNone(
                resolve_apple_lm_sidecar(
                    missing,
                    platform="win32",
                    required=True,
                )
            )


if __name__ == "__main__":
    unittest.main()
