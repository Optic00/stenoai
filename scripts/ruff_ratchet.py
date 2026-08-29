"""Keep the selected Ruff debt stable until an intentional baseline update."""

from __future__ import annotations

import argparse
import json
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

Baseline = dict[str, dict[str, int]]


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

    grouped: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    for diagnostic in diagnostics:
        filename = diagnostic.get("filename")
        code = diagnostic.get("code")
        if not isinstance(filename, str) or not isinstance(code, str):
            raise RuntimeError("Ruff returned a diagnostic without filename or rule code")
        grouped[_relative_posix_path(filename, root)][code] += 1
    return {
        filename: dict(sorted(rules.items()))
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
        normalized: dict[str, int] = {}
        for code, count in rules.items():
            if not isinstance(code, str) or not isinstance(count, int) or count < 0:
                raise RuntimeError(f"Invalid Ruff baseline entry for {filename}")
            normalized[code] = count
        baseline[filename] = normalized
    return baseline


def differences(expected: Baseline, actual: Baseline) -> list[str]:
    messages: list[str] = []
    for filename in sorted(set(expected) | set(actual)):
        expected_rules = expected.get(filename, {})
        actual_rules = actual.get(filename, {})
        for code in sorted(set(expected_rules) | set(actual_rules)):
            before = expected_rules.get(code, 0)
            after = actual_rules.get(code, 0)
            if before == after:
                continue
            direction = "increased" if after > before else "decreased"
            messages.append(f"{filename} {code}: {direction} from {before} to {after}")
    return messages


def write_baseline(path: Path, baseline: Baseline) -> None:
    stable = {filename: baseline[filename] for filename in sorted(baseline)}
    path.write_text(json.dumps(stable, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check(*, baseline_path: Path, update: bool, root: Path = ROOT) -> int:
    actual = ruff_findings(root)
    expected = read_baseline(baseline_path)
    changes = differences(expected, actual)
    if not changes:
        print("Ruff ratchet matches baseline.")
        return 0
    if update:
        write_baseline(baseline_path, actual)
        print("Updated Ruff baseline.")
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
