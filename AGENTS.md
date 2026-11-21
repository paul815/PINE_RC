# AGENTS.md

Guidance for any coding agent working in this repository — Claude Code, Codex,
Cursor, and whatever comes next. It replaces the former `documentation/AGENTS.md`
and `documentation/CLAUDE.md`, which said much of the same thing in two places
and, sitting one directory down, were loaded by neither tool.

Last updated: 2026-11-22.

## 1) Mission

PINE (Private Interview & Notes Environment) is a fully local, offline Flask
desktop web app for product researchers. Audio and transcripts are processed on
the user's machine; nothing leaves the PC. Speech-to-text is WhisperX (on Apple
Silicon through CTranslate2's Metal backend), speaker diarization is pyannote.

The top engineering goal is data integrity for research assets:

- Transcripts (`*_transcript.json`)
- Annotations (`*_annotations.json` — tags, comments, speaker labels)
- Project files (audio/video/uploads and exported documents)

When speed conflicts with safety, choose safety.

## 2) Priority rules (highest → lowest)

1. System/tooling safety constraints
2. This file
3. The user's request in the current task
4. Agent preferences and defaults

If rules conflict, say which won and why.

## 3) Data safety (critical)

Treat every change that can affect transcript, tag, or file persistence as high
risk.

Never do these without an explicit user request:

- Delete or overwrite project directories or recording files
- Mass-rewrite files under `projects/`
- Run the reset scripts (`backend/reset_win.bat`, `backend/reset.command`)
- Execute destructive git or filesystem operations

Risky operations require all three: explicit confirmation, minimal scope, and a
rollback path (backup, snapshot, or a reversible write strategy).

When editing code that writes transcript/tag/annotation files:

- Prefer atomic writes — write a temp file, then replace
- Preserve unknown fields when reading and updating JSON
- Keep backward compatibility with the existing JSON shape
- Fail closed on corruption; never silently truncate or drop user content

Read carefully before editing:

- `backend/app/api/projects/` (the package — `crud`, `recordings`, `segments`,
  `tags`, `export`, `attachments`, `common`)
- `backend/app/services/annotations.py`
- `backend/app/services/export_service.py`
- `backend/app/services/backup_service.py`
- `backend/app/services/file_utils.py`
- `backend/app/services/transcript_edit.py`
- `backend/app/services/multitrack_ingest.py` — registers files by absolute path;
  never copies, never moves user media
- `backend/app/models/` — schema implications

Data-bearing runtime directories, not to be touched casually: `projects/`,
`backend/data/`, `backups/`.

## 4) Running the app

```bash
# From the repo root, activate the venv first:
source .venv/bin/activate          # macOS/Linux
.\.venv\Scripts\activate           # Windows

# Then start the server:
python backend/run.py
# Opens at http://127.0.0.1:5000
```

On Windows `Setup_WIN.bat` does both steps; on macOS/Linux use
`Setup_MAC.command`. Both live in the repo root, create `.venv` and install
dependencies on first run, then start the server on later launches.

## 5) Running tests

Run from `backend/` with the venv active. 2422 tests, all green as of 2026-11-21
(Windows). In an installed copy (after onboarding) the 9 installer tests fail
because `Setup_*` has moved under `backend/`; that is expected there, not in a
clone. The suite currently leaves `tmp*` and `pytest-api-quit-*` directories in
the working tree; they are gitignored.

```bash
cd backend && pytest                                    # all
cd backend && pytest tests/test_api_projects.py         # one file
cd backend && pytest tests/test_api_projects.py::test_create_project
cd backend && pytest --cov=app
cd backend && python -m ruff check .
```

The suite uses an in-memory SQLite DB and temp directories — it never touches
real data. `PINE_TESTING=1` signals test mode to the app.

Minimum expectation per change:

- Data-layer or API changes: run the impacted test modules
- Annotation/export/project-file logic: targeted tests plus one integration-style
  API test file
- Anything touching durability: prioritise the durability, annotation and export
  tests

If tests cannot be run, report the exact reason, what went unvalidated, and the
concrete manual checks required.

## 6) Architecture

### Stack

- **Backend:** Flask 3 + Flask-SQLAlchemy + Flask-SocketIO + Flask-CORS
- **Templates:** server-rendered Jinja2, no build step and no JS framework
- **DB:** SQLite (`backend/data/pine.db`) through SQLAlchemy
- **Real-time:** SocketIO for transcription progress

### Layout

```
backend/
  app/            # Web process. Never import torch/whisperx/pyannote here
    api/          # onboarding_bp, settings_bp, utils_bp, backup_bp, and projects/ —
                  # a package (common, crud, recordings, segments, tags, export,
                  # attachments) whose modules all register on one projects_bp
    models/       # SQLAlchemy: Project, Recording, RecordingTrack, Segment, Setting, MLModel
    ports.py      # backend/supervisor ports and the CORS origin list, in one place
    services/     # transcription/ (job_runner + worker_client), export, model_manager,
                  # launcher_layout, pip_installer, annotations, multitrack_ingest, pii,
                  # backup, system_check, ffmpeg_setup, transcript_format
  ml_worker/      # ML process. torch/whisperx/pyannote live only here
    engines/      # whisperx_engine (CUDA/Metal/CPU), behind base.py
  templates/      # main, recording, tags, settings, onboarding
  scripts/        # check_mojibake.py, check_line_endings.py, build_cyrillic_fonts.py
  data/           # pine.db (created at runtime)
  tests/          # pytest suite with conftest.py fixtures
  tools/          # dev helpers: dev_reset.*, design/contrast audits, reset allowlists
models/           # HuggingFace model files (configurable path, gitignored)
projects/         # Per-project folders (configurable path)
```

### Data model (hybrid)

**SQLite** holds metadata: projects, recordings (status, paths), segments
(screener groups), settings (key-value), the ML model registry.

**Filesystem JSON** holds transcripts (`<name>_transcript.json`), annotations
(`<name>_annotations.json`) and attachments (`attachments.json` plus an
`attachments/` directory) inside `projects/<folder_name>/`. This is what keeps a
project portable in a transfer package — and why breaking consistency between the
two halves is a data-loss event, not a cosmetic bug.

### Process boundary

`backend/app/` must **never** import `torch`, `whisperx` or `pyannote`.
Those belong to `backend/ml_worker/` alone, so a CUDA crash or an OOM kills the
worker rather than the backend. An import across this line looks harmless and
quietly undoes the split.

### Transcription pipeline

- `services/transcription/job_runner.py` owns the queue inside Flask;
  `worker_client.py` spawns the `ml_worker` process and talks to it over a pipe
  (JSON messages, `ml_worker/protocol.py`)
- One background worker thread, one recording at a time (GPU-bound). It starts in
  `create_app()`
- `requeue_interrupted()` on startup reschedules recordings left in `transcribing`
  by a previous crash
- Progress arrives as SocketIO `transcription:status` events —
  `{ recording_id, status, stage, message, percent? }`; every event name uses the
  `namespace:event` form. `ml_worker/progress.py` folds all stages into one 0–100
  scale and learns per-machine throughput, stored as
  `progress_scale:<model>:<single|multi>` in Settings
- ASR chunks files ≥30 min (30-min chunks, 30 s overlap). Diarization runs
  whole-file and overlaps the ASR pass; its own chunking sits behind a 4-hour
  threshold because chunked pyannote flipped speakers across seams. Thresholds
  live in `ml_worker/constants.py`
- Multi-track recordings (Zoom folders, multi-channel files) skip diarization
  entirely — `services/multitrack_ingest.py` and `ml_worker/tracks.py`

### Model management

- `MODEL_REGISTRY` in `services/model_manager.py` lists every available model.
  Required: pyannote-diarization, pyannote-segmentation, one transcription model.
  Optional: gliner-pii, offered during onboarding and installable or removable
  from Settings afterwards
- **One transcription model on every platform:** `whisperx-large-v3` — on a Mac it
  runs on Metal (CTranslate2 4.9+), CPU int8 without it. The retired
  `mlx-whisper-large-v3` id maps onto it; `normalize_stt_model_id()` rejects the rest
- Heavy ML deps are **not** in `requirements.txt` — `pip_installer.py` installs
  them at download time
- Models come from HuggingFace Hub during onboarding; pyannote needs an HF token,
  stored in SQLite

### Dependency files

- `backend/requirements.txt` — minimal base deps, installed by the launchers on
  first run. No ML packages
- `backend/requirements-lock.txt` — full pinned `pip freeze` after a complete
  install. Not used by the launchers; kept for reproducibility and auditing.
  Regenerate with `pip freeze > backend/requirements-lock.txt` from a clean
  post-onboarding venv

Two traps for anyone tidying the ML layer:

- **`torchvision` looks unused — it is not.** PINE never imports it, but whisperx
  pins `torchvision~=0.23.0`, and `install_pip_packages()` installs whisperx with
  `--no-deps` and adds its dependencies by hand. It is there on purpose. Its
  wheel channel must also match torch's (`+cu128` vs `+cpu`) — so must
  torchaudio's, which PINE does import; `_torch_companion_channels_aligned()`
  checks both, and a mismatch fails later as an undefined-symbol crash
- **~13 packages of dead weight cannot be removed.** `pyannote-audio 4.x`
  declares `pyannoteai-sdk`, eight `opentelemetry-*`, `grpcio`, `protobuf` and
  `googleapis-common-protos` — a cloud SDK PINE never calls, roughly 25 MB. pip
  cannot drop them while pyannote declares them. The telemetry itself is stubbed
  out before import in `app/__init__.py`, `ml_worker/compat.py` and `run.py`;
  leave those stubs alone

### Schema migrations

No Alembic. `_migrate_db()` in `app/__init__.py` runs `ALTER TABLE ADD COLUMN`
migrations at startup. New migrations go there.

### Frontend

- Five standalone Jinja2 templates with inline `<style>` and `<script>`. None uses
  `{% extends %}`
- Shared CSS in `app/static/css/`: `tokens.css` (palette, spacing and radius
  scales, z-index ladder), `fonts.css`, `button-system.css`, `ui-system.css`,
  `save-status.css`. Every template links them *before* its own `<style>` block
  (onboarding skips `save-status.css`), so a page rule overrides a shared one at
  equal specificity
- `python backend/tools/design_audit.py --rules` checks the CSS against
  `documentation/DESIGN.md` — spacing scale, radii, colours, z-index, motion,
  shadows, line height, tracking, duplicated classes — and names each line
- A colour or font is defined in exactly one place. What remains in a template's
  `:root` is page-only (layout metrics, player colours)
- Never write a raw `z-index` — use the `--z-*` ladder. Scope theme rules to
  `:root[data-theme="dark"]`, never a bare `[data-theme="dark"]`: the latter also
  matches the theme-picker buttons, which carry that attribute

### Export

`services/export_service.py` produces Markdown or ODT (via `odfpy`), with
`include_comments`, `include_tags` and `remove_pii` options.

## 7) Change strategy

- Make the smallest change that solves the request
- Do not refactor unrelated code in the same task
- Keep API contracts stable unless a breaking change was asked for
- Preserve existing export formatting and annotation semantics unless asked
- A migration must be forward-safe and handle the fallback explicitly

## 8) Git and workspace

- Do not commit or push unless asked
- Do not rewrite history unless asked
- Do not revert unrelated local changes; assume the worktree is dirty and isolate
  your edits
- Forbidden without an explicit request: `git reset --hard`,
  `git checkout -- <path>`, recursive deletes touching user data paths

Two things that bite here specifically: a finished install tidies the repo root
(`launcher_layout.finalize_root_layout_after_onboarding()` moves `DESIGN.md` and
the root `CLAUDE.md` into `documentation/`, drops both installers under
`backend/` and puts `Launch Pine` in their place — it runs when the user presses
Launch PINE at the end of onboarding, not before), and a reset deletes everything
at the root that is not listed in `backend/tools/reset_preserve_root.txt`. A new
root-level file needs an entry in that allowlist or the first reset takes it.

### Repositories, branches and machines

- **`origin` = PINE_RC** (`paul815/PINE_RC`) is where all work happens: branch
  off `main`, merge back into `main`.
- **`pine` = PINE** (`paul815/PINE`, public) receives `main` only once it has
  been verified on both Windows and macOS. Never push there without an explicit
  request. A release is a version bump plus a tag; the in-app update check reads
  `releases/latest`, so an untagged push reaches no users.
- Machines: a Windows desktop (CUDA) and a personal Mac each have a single clone
  of PINE_RC and run Claude Code. A work Mac has no Claude Code and is used only
  for testing: `git pull`, run, and file anything found as a PINE_RC issue.
- Open work lives in PINE_RC GitHub Issues, not in an agent's local memory, so it
  is visible from every machine. Hand-off between machines is `git push` /
  `git pull` of a branch plus the issue number.
- An installed copy that holds real projects (`projects/`, `backend/data/`) is
  not a development checkout. Do not develop in it and do not delete it.

## 9) Security and privacy

- Never expose tokens or secrets in logs or responses
- Never send transcript or project content to an external service
- Keep the local-first model intact; any change to network behaviour must be
  called out in the final report

## 10) Communication contract

Keep progress updates short. The final response states:

1. What changed
2. Files touched
3. Validation run, and its result
4. Data-safety impact — why transcripts, tags and files remain safe
5. Residual risk, if any

If uncertainty could cause data loss, stop and ask.

## 11) Definition of done

- The requested behaviour is implemented
- Impacted tests ran, or the report says plainly that they did not
- No silent data-destructive path was introduced
- The report states the validation and data-safety outcome

## Known gaps

The ones worth knowing before you start:

- **No download cancellation** — a model download cannot be stopped once started
- **No cross-transcript search** — no `/search` endpoint exists
- **No formal migrations** — the `ALTER TABLE ADD COLUMN` loop cannot rename,
  drop, or retype
- **No JSON-schema validation** on transcript/annotation files — a hand-edited file
  propagates silently. Transcripts *are* version-stamped
  (`services/transcript_format.py`: `schema_version`, `engine`, `model`,
  `created_at`; a file without the key reads as version 1), but nothing validates
  their shape

## Reference docs

- [documentation/API.md](documentation/API.md) — every endpoint, data shapes,
  SocketIO events
- [documentation/ARCHITECTURE.md](documentation/ARCHITECTURE.md) — the same ground
  as section 6, in more detail
- [documentation/DESIGN.md](documentation/DESIGN.md) — UI tokens and rules
