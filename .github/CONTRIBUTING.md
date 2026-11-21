# Contributing to PINE

Thanks for taking the time. PINE handles research material that often cannot be
re-collected — interviews recorded once, under NDA, with participants who will
not sit down a second time. That shapes how changes are reviewed: **data
integrity beats everything else**, including elegance and speed.

## The most useful contribution

Bug reports from people running PINE on real interviews. Transcription is a long,
hardware-dependent pipeline and no test suite reproduces every machine. If
something went wrong, the report is valuable even if you cannot debug it —
attach the relevant file from `backend/logs/` and the System check output from
onboarding.

## Setting up

```bash
git clone https://github.com/paul815/PINE.git
cd pine
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\activate
pip install -r backend/requirements.txt
pip install pytest pytest-cov ruff
```

Test tooling is intentionally missing from `requirements.txt`: a user's install
should carry no test runner or linter. Install it by hand in the venv.

Running the app:

```bash
cd backend && python supervisor.py
```

`python run.py` starts the backend alone, without the supervisor — useful when
you want the process in the foreground, but it is not how the app normally runs.

Heavy ML packages (torch, whisperx, pyannote, onnx) are **not** in
`requirements.txt` either. They are installed at runtime during onboarding by
`app/services/pip_installer.py`. Tests that need them skip when they are absent,
which is also what happens in CI.

## Before you open a pull request

```bash
cd backend && pytest
ruff check backend
```

Both must be clean. If you touched anything that reads or writes transcripts,
annotations or project folders, add a test — that code path is the reason the
suite exists.

## Rules that are not obvious from the code

**The process boundary is load-bearing.** `backend/app/` must never import
`torch`, `whisperx` or `pyannote`. Those belong to `backend/ml_worker/`
alone, so that a CUDA crash or an out-of-memory kill takes down the worker and
not the app. An import that crosses this line looks harmless and quietly undoes
the split.

**Writes to user data are atomic.** Transcripts and annotations are the source of
truth and live in project folders, not in SQLite. Write to a temp file and
replace; preserve unknown JSON fields; keep backward compatibility with existing
files; fail closed on corruption rather than silently truncating.

**Line endings are pinned per file type.** `.bat` and `.cmd` must be CRLF, and
`.command` and `.sh` must be LF — cmd.exe reads batch files by byte offset and
macOS rejects a `.command` with CRLF. `.gitattributes` enforces it,
`backend/scripts/check_line_endings.py` is the backstop, and it runs from both
CI and the test suite.

**No mojibake.** `backend/scripts/check_mojibake.py` catches text that has been
through a wrong-encoding round trip. Write files as UTF-8. CI runs it; there is
no pre-commit hook to install.

**Style.** `ruff` config lives in `backend/pyproject.toml`. Line length is not
enforced — the codebase wraps by meaning, not by column. Comments explain *why*,
and the surrounding code is the style guide: match its density and idiom rather
than importing your own.

**Terminology.** The UI says *codes* and *themes* (qualitative research
vocabulary). The API, database and JSON files still say `tags` and `groups`.
Keep user-facing strings on the research side of that split and do not rename the
storage format to match.

## Documentation that travels with a change

| If you change | Update |
|---|---|
| An endpoint or a payload shape | [documentation/API.md](../documentation/API.md) |
| The pipeline, data model or process layout | [documentation/ARCHITECTURE.md](../documentation/ARCHITECTURE.md) |
| Tokens, components or UI patterns | [documentation/DESIGN.md](../documentation/DESIGN.md) |
| Anything a user can see or click | [README.md](../README.md) |

## Commits and pull requests

- One concern per pull request; a refactor and a fix in the same diff are hard to
  review and harder to revert.
- Say what breaks if the change is wrong. That sentence is more useful than a
  summary of the diff.
- Note what you tested manually, on which OS, and with which model. The suite
  mocks the models, so real transcription quality, diarization, onboarding and a
  clean install are only ever covered by a human — say which of those you touched.

## Security

Do not open a public issue for a vulnerability — see [SECURITY.md](SECURITY.md).
