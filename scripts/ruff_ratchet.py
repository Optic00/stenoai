"""Keep the selected Ruff debt stable until an intentional baseline update."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any


EXPECTED_RUFF_VERSION = "ruff 0.15.21"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "scripts" / "ruff_baseline.json"
RUFF_ARGS = [
    "check",
    ".",
    "--isolated",
    "--no-cache",
    "--select",
    "E4,E7,E9,F",
    "--target-version",
    "py311",
    "--output-format",
    "json",
]

Baseline = dict[str, dict[str, dict[str, int]]]
FINGERPRINT_RE = re.compile(r"[0-9a-f]{64}")


def _run(command: list[str], root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)


def _relative_posix_path(filename: str, root: Path) -> str:
    candidate = Path(filename)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        return candidate.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Ruff reported a path outside the repository: {filename}") from error


def _finding_fingerprint(
    diagnostic: dict[str, Any],
    *,
    filename: str,
    code: str,
    root: Path,
) -> str:
    """Identify one finding without depending on its absolute line number."""
    message = diagnostic.get("message")
    location = diagnostic.get("location")
    end_location = diagnostic.get("end_location")
    if (
        not isinstance(message, str)
        or not message
        or not isinstance(location, dict)
        or not isinstance(end_location, dict)
    ):
        raise RuntimeError("Ruff returned a diagnostic without stable identity fields")

    start_row = location.get("row")
    end_row = end_location.get("row")
    end_column = end_location.get("column")
    if (
        not isinstance(start_row, int)
        or isinstance(start_row, bool)
        or not isinstance(end_row, int)
        or isinstance(end_row, bool)
        or not isinstance(end_column, int)
        or isinstance(end_column, bool)
        or start_row < 1
        or end_row < start_row
        or end_column < 1
    ):
        raise RuntimeError("Ruff returned a diagnostic with invalid source coordinates")

    # Ruff end locations are exclusive. A multi-line span ending at column 1
    # therefore ends on the previous physical line; excluding that following
    # line keeps the identity stable when unrelated lines move around it.
    last_row = end_row - 1 if end_row > start_row and end_column == 1 else end_row
    source_path = root / PurePosixPath(filename)
    try:
        source_lines = source_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"Could not read Ruff source file {filename}") from error
    if last_row > len(source_lines):
        raise RuntimeError("Ruff returned a diagnostic outside its source file")
    source_span = "\n".join(source_lines[start_row - 1:last_row]).strip()
    if not source_span:
        raise RuntimeError("Ruff returned a diagnostic without source text")

    payload = json.dumps(
        [code, message, source_span],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ruff_findings(root: Path) -> Baseline:
    version = _run([sys.executable, "-m", "ruff", "--version"], root)
    if version.returncode != 0 or version.stdout.strip() != EXPECTED_RUFF_VERSION:
        actual = version.stdout.strip() or version.stderr.strip() or "not available"
        raise RuntimeError(f"Expected {EXPECTED_RUFF_VERSION}; found {actual}")

    result = _run([sys.executable, "-m", "ruff", *RUFF_ARGS], root)
    if result.returncode not in {0, 1}:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise RuntimeError(f"Ruff did not complete successfully: {detail}")
    try:
        diagnostics: list[dict[str, Any]] = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Ruff did not produce JSON diagnostics") from error

    grouped: defaultdict[str, defaultdict[str, defaultdict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    for diagnostic in diagnostics:
        filename = diagnostic.get("filename")
        code = diagnostic.get("code")
        if not isinstance(filename, str) or not isinstance(code, str):
            raise RuntimeError("Ruff returned a diagnostic without filename or rule code")
        relative_filename = _relative_posix_path(filename, root)
        fingerprint = _finding_fingerprint(
            diagnostic,
            filename=relative_filename,
            code=code,
            root=root,
        )
        grouped[relative_filename][code][fingerprint] += 1
    return {
        filename: {
            code: dict(sorted(identities.items()))
            for code, identities in sorted(rules.items())
        }
        for filename, rules in sorted(grouped.items())
    }


def read_baseline(path: Path) -> Baseline:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read Ruff baseline {path}: {error}") from error
    if not isinstance(parsed, dict):
        raise RuntimeError("Ruff baseline must be a JSON object")
    baseline: Baseline = {}
    for filename, rules in parsed.items():
        posix_path = PurePosixPath(filename) if isinstance(filename, str) else None
        if (
            not isinstance(filename, str)
            or "\\" in filename
            or posix_path is None
            or posix_path.is_absolute()
            or ".." in posix_path.parts
            or posix_path.as_posix() != filename
        ):
            raise RuntimeError("Ruff baseline filenames must be POSIX relative paths")
        if not isinstance(rules, dict):
            raise RuntimeError(f"Ruff baseline rules for {filename} must be an object")
        normalized: dict[str, dict[str, int]] = {}
        for code, identities in rules.items():
            if not isinstance(code, str) or not isinstance(identities, dict):
                raise RuntimeError(f"Invalid Ruff baseline entry for {filename}")
            normalized_identities: dict[str, int] = {}
            for fingerprint, count in identities.items():
                if (
                    not isinstance(fingerprint, str)
                    or FINGERPRINT_RE.fullmatch(fingerprint) is None
                    or not isinstance(count, int)
                    or isinstance(count, bool)
                    or count < 1
                ):
                    raise RuntimeError(f"Invalid Ruff baseline entry for {filename}")
                normalized_identities[fingerprint] = count
            normalized[code] = normalized_identities
        baseline[filename] = normalized
    return baseline


def differences(expected: Baseline, actual: Baseline) -> list[str]:
    messages: list[str] = []
    for filename in sorted(set(expected) | set(actual)):
        expected_rules = expected.get(filename, {})
        actual_rules = actual.get(filename, {})
        for code in sorted(set(expected_rules) | set(actual_rules)):
            expected_identities = expected_rules.get(code, {})
            actual_identities = actual_rules.get(code, {})
            for fingerprint in sorted(set(expected_identities) | set(actual_identities)):
                before = expected_identities.get(fingerprint, 0)
                after = actual_identities.get(fingerprint, 0)
                if before == after:
                    continue
                direction = "increased" if after > before else "decreased"
                messages.append(
                    f"{filename} {code} {fingerprint[:12]}: "
                    f"{direction} from {before} to {after}"
                )
    return messages


def write_baseline(path: Path, baseline: Baseline) -> None:
    stable = {
        filename: {
            code: dict(sorted(baseline[filename][code].items()))
            for code in sorted(baseline[filename])
        }
        for filename in sorted(baseline)
    }
    path.write_text(json.dumps(stable, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check(*, baseline_path: Path, update: bool, root: Path = ROOT) -> int:
    actual = ruff_findings(root)
    if update:
        write_baseline(baseline_path, actual)
        print("Updated Ruff baseline.")
        return 0
    expected = read_baseline(baseline_path)
    changes = differences(expected, actual)
    if not changes:
        print("Ruff ratchet matches baseline.")
        return 0
    print("Ruff findings differ from the baseline. Run with --update after reviewing the change:")
    print("\n".join(changes))
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="write the reviewed current findings as baseline")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        return check(baseline_path=args.baseline, update=args.update)
    except RuntimeError as error:
        print(f"ruff ratchet: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
