#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# Use unicode escape sequences so this checker is not sensitive to editor/display encoding.
REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("\u00e2\u20ac\u00a6", "\u2026"),  # mojibake ellipsis -> ellipsis
    ("\u00e2\u20ac\u201d", "\u2014"),  # mojibake em dash -> em dash
    ("\u00e2\u20ac\u201c", "\u2013"),  # mojibake en dash -> en dash
    ("\u00c2\u00b7", "\u00b7"),        # mojibake middle dot -> middle dot
    ("\u00e2\u2020\u2014", "\u2197"),  # mojibake up-right arrow -> up-right arrow
    ("\u00e2\u02c6\u2019", "\u2212"),  # mojibake minus -> minus sign
    ("\u00e2\u201d\u20ac", "\u2500"),  # mojibake box drawing char -> box drawing char
)

TARGET_EXTENSIONS = {
    ".py",
    ".html",
    ".js",
    ".css",
    ".md",
    ".txt",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".bat",
    ".ps1",
    ".ts",
    ".tsx",
    ".jsx",
}

EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "node_modules",
    "models",
    "projects",
    "backups",
    ".tmp_vox",
}


@dataclass
class Finding:
    file_path: Path
    line_number: int
    token: str


def printable_token(token: str) -> str:
    return token.encode("unicode_escape").decode("ascii")


def iter_target_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
                continue
            if path.suffix.lower() not in TARGET_EXTENSIONS:
                continue
            files.append(path)
    return files


def find_mojibake(text: str, file_path: Path) -> list[Finding]:
    findings: list[Finding] = []
    lines = text.splitlines()
    for idx, line in enumerate(lines, start=1):
        for bad, _good in REPLACEMENTS:
            if bad in line:
                findings.append(Finding(file_path=file_path, line_number=idx, token=bad))
    return findings


def fix_text(text: str) -> str:
    fixed = text
    for bad, good in REPLACEMENTS:
        fixed = fixed.replace(bad, good)
    return fixed


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect and optionally fix mojibake in text files.")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply safe replacements and rewrite files as UTF-8 without BOM.",
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        default=["backend", "frontend", "documentation"],
        help="Root paths to scan.",
    )
    args = parser.parse_args()

    roots = [Path(p) for p in args.paths]
    files = iter_target_files(roots)

    all_findings: list[Finding] = []
    touched_files = 0

    for file_path in files:
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        findings = find_mojibake(text, file_path)
        if findings:
            all_findings.extend(findings)
            if args.fix:
                fixed = fix_text(text)
                if fixed != text:
                    file_path.write_text(fixed, encoding="utf-8", newline="\n")
                    touched_files += 1

    if all_findings:
        print("Mojibake findings:")
        for f in all_findings:
            rel = f.file_path.as_posix()
            print(f"  - {rel}:{f.line_number}: token '{printable_token(f.token)}'")
        if args.fix:
            print(f"\nApplied fixes in {touched_files} file(s).")
            # Verify clean state after fix.
            post_findings: list[Finding] = []
            for file_path in files:
                try:
                    text = file_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                post_findings.extend(find_mojibake(text, file_path))
            if post_findings:
                print("\nSome mojibake tokens still remain after --fix.")
                return 1
            print("No mojibake tokens remain after --fix.")
            return 0
        return 1

    print("No mojibake tokens found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
