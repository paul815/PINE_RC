#!/usr/bin/env python3
"""Guard the line endings the launchers depend on.

cmd.exe reads a batch file by byte offset. In an LF-only .bat it loses the
offsets and label lookup fails outright: Setup_WIN.bat dies at
"call :open_browser" with "The system cannot find the batch label specified",
the console window closes, and the browser is never opened -- the backend then
shuts itself down 45s later for want of a browser lease. The mirror image is
true on macOS: a .command with CRLF fails with "bad interpreter".

.gitattributes pins both (`*.bat text eol=crlf`, `*.command text eol=lf`), but
an editor, a plain `copy` from another checkout, or a tool that writes '\\n'
can still land the wrong bytes in the working tree. This checker is the
backstop; it runs from CI and from the test suite.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# suffix -> required ending ("crlf" or "lf")
REQUIRED_ENDINGS: dict[str, str] = {
    ".bat": "crlf",
    ".cmd": "crlf",
    ".command": "lf",
    ".sh": "lf",
}

EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "node_modules",
    "models",
    "projects",
    "backups",
    "logs",
    "data",
    ".tmp_vox",
    ".pytest_cache",
}

# Same idea, for directories whose name is only known at runtime. pytest names
# its temp root after the current user (pytest-of-paul/), and the launcher tests
# copy real .command and .bat files into it -- so a tree left behind by an
# earlier run turns this check red while every launcher in the repo is fine.
EXCLUDED_DIR_PREFIXES = ("pytest-of-", "pytest-cache-files-")


def _in_excluded_dir(path: Path) -> bool:
    return any(
        part in EXCLUDED_DIR_NAMES or part.startswith(EXCLUDED_DIR_PREFIXES)
        for part in path.parts
    )


def iter_target_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if _in_excluded_dir(path):
            continue
        if path.suffix.lower() in REQUIRED_ENDINGS:
            files.append(path)
    return sorted(files)


def check_file(path: Path) -> str | None:
    """Return a human-readable problem, or None when the file is fine."""
    want = REQUIRED_ENDINGS[path.suffix.lower()]
    data = path.read_bytes()
    crlf = data.count(b"\r\n")
    lone_lf = data.count(b"\n") - crlf
    if want == "crlf" and lone_lf:
        return f"{lone_lf} LF line(s) in a CRLF file (cmd.exe cannot find labels here)"
    if want == "lf" and crlf:
        return f"{crlf} CRLF line(s) in an LF file (bad interpreter on macOS)"
    return None


def fix_file(path: Path) -> bool:
    want = REQUIRED_ENDINGS[path.suffix.lower()]
    data = path.read_bytes()
    normalised = data.replace(b"\r\n", b"\n")
    if want == "crlf":
        normalised = normalised.replace(b"\n", b"\r\n")
    if normalised == data:
        return False
    path.write_bytes(normalised)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Check launcher script line endings.")
    parser.add_argument("--fix", action="store_true", help="Rewrite offending files in place.")
    parser.add_argument("--root", default=str(REPO_ROOT), help="Repository root to scan.")
    args = parser.parse_args()

    root = Path(args.root)
    problems: list[tuple[Path, str]] = []
    fixed = 0

    for path in iter_target_files(root):
        problem = check_file(path)
        if problem is None:
            continue
        if args.fix and fix_file(path):
            fixed += 1
            continue
        problems.append((path, problem))

    if problems:
        print("Wrong line endings:")
        for path, problem in problems:
            print(f"  - {path.relative_to(root).as_posix()}: {problem}")
        print("\nFix with: python backend/scripts/check_line_endings.py --fix")
        return 1

    if args.fix and fixed:
        print(f"Normalised {fixed} file(s).")
    else:
        print("Line endings OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
