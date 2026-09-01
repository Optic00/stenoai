"""Build-time guard for the optional macOS Apple language-model sidecar."""

import os
import sys
from pathlib import Path
from typing import Optional


def resolve_apple_lm_sidecar(
    sidecar_path: Path,
    *,
    platform: str = sys.platform,
    required: bool = False,
) -> Optional[Path]:
    """Validate and return the sidecar included in a macOS bundle.

    Development builds may omit the helper when their Xcode lacks the macOS 26
    SDK. Release builds pass ``required=True`` so a missing feature cannot ship
    silently. Other platforms never bundle the Darwin-only executable.
    """
    if platform != "darwin":
        return None
    if not sidecar_path.is_file():
        if required:
            raise FileNotFoundError(
                f"Missing required Apple LM sidecar: {sidecar_path}. "
                "Run scripts/build-apple-lm-sidecar.sh before PyInstaller."
            )
        return None
    if not os.access(sidecar_path, os.X_OK):
        raise PermissionError(
            f"Apple LM sidecar is not executable: {sidecar_path}"
        )
    return sidecar_path
