# PINE API Reference

Base URL: `http://127.0.0.1:5000`

All JSON responses use `Content-Type: application/json`. Errors return `{ "error": "message" }` with appropriate HTTP status.

---

## Onboarding (`/api/onboarding`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/status` | `completed`, `modules`, `models_path`, `projects_path` |
| GET | `/device` | Transcription device (cuda/cpu), GPU name, VRAM |
| GET | `/system-check` | Python, FFmpeg, torch, disk checks |
| GET | `/defaults` | Default paths, disk usage |
| POST | `/modules` | Set modules; body `{ "modules": ["pii"] }` |
| POST | `/storage` | Set paths; body `{ "models_path", "projects_path" }` |
| POST | `/hf-token` | Validate and store HF token |
| POST | `/download/start` | Start model download |
| GET | `/download/status` | Model download status per model |
| POST | `/stt-model` | Pick the transcription model to download; body `{ "stt_model_id" }`. Unknown ids fall back to the platform default |
| GET | `/browse-folder` | Browse the filesystem for a storage path |
| POST | `/handoff/prepare` | Hand the finished onboarding over to the main app |
| POST | `/play-install-sound` | Play the install-complete chime |

---

## Projects (`/api/projects`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `` | List active + archived projects |
| POST | `` | Create project; body `{ "name" }` |
| GET | `/<id>` | Get project with recordings |
| PATCH | `/<id>` | Update any subset of the project fields (see below). Renaming also renames the folder on disk and rewrites `folder_name` |
| DELETE | `/<id>` | Delete project and folder |
| POST | `/<id>/archive` | Archive project |
| POST | `/<id>/unarchive` | Unarchive project |
| GET | `/<id>/transfer` | Download project as ZIP (no audio) — holds `project.json`, `project_tags.json`, `project_themes.json`, the project `README.md`, and every transcript under `transcripts/` with its annotations under `annotations/` |

`PATCH /<id>` accepts, in three groups:

- **Plain text** — `name`, `description`, `objective`, `summary`, `icon`,
  `results_recommendations`, `further_steps`, `methodology`, `interview_guide`,
  `key_findings`, `recommendations` (each stripped; an empty string clears it)
- **Lists**, stored as JSON — `research_questions`, `hypotheses`, `stakeholders`,
  `enabled_sections`, `enabled_summary_sections`, `custom_sections`,
  `custom_summary_sections`
- **`default_transcription_language`** — a code such as `ru`, matched against
  `^[a-z][a-z0-9_]{0,15}$` (`""` clears it); anything else returns **400**. When
  set, it answers the language prompt ahead of time for every recording in the
  project

Unknown keys are ignored. The response is the full project.

### Recordings

`num_speakers` is optional on every endpoint that accepts it. Without it,
diarization uses the `PINE_MIN_SPEAKERS`/`PINE_MAX_SPEAKERS` range, which
defaults to **2..4** — PINE's material is interviews with two to four
participants. A monologue or a group of five and up falls outside that
range and has to pass an explicit count (or run with different env values).

The spoken language is detected by the worker, not asked for at upload. When the
detector is unsure — confidence below `PINE_LANG_CONFIDENCE_MIN` (0.82) — the job
stops mid-flight and parks the recording in `awaiting_language`. The
`transcription:status` event for that state carries the guess and the shortlist,
and the job thread waits until `POST …/transcription/language` answers it. A
project that sets `default_transcription_language` never sees the question;
`PINE_SKIP_LANG_CONFIRM=1` turns the prompt off globally.

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/<id>/recordings` | Upload file (multipart `file`, optional `num_speakers` 1–10) |
| POST | `/<id>/recordings/link` | Link external file; body `{ "path": "/absolute/path", "num_speakers": 2 }` (count optional) |
| POST | `/<id>/recordings/multitrack` | Register material with one track per speaker; body `{ "folder": "/path/to/zoom/meeting" }` or `{ "path": "/file/with/one/channel/per/speaker" }`. Never copies — paths are stored as-is. Diarization is skipped for these. |
| GET | `/<id>/recordings/<rid>` | Recording + transcript + tags + annotations |
| PATCH | `/<id>/recordings/<rid>` | Update recording; body `{ "segment_id", "participant_notes", "original_name", "num_speakers" }` |
| DELETE | `/<id>/recordings/<rid>` | Delete recording and files |
| DELETE | `/<id>/recordings/<rid>/transcription` | Cancel in-progress transcription (409 if not transcribing) |
| POST | `/<id>/recordings/<rid>/transcription/language` | Answer the language prompt and let the job continue; body `{ "language": "ru" }`, matched against `^[a-z][a-z0-9_]{0,15}$`. **409** unless the recording sits in `awaiting_language`, **404** when the prompt has already been withdrawn |
| GET | `/<id>/recordings/<rid>/media` | Stream media file |
| GET | `/<id>/recordings/<rid>/transcript` | Raw transcript JSON |
| POST | `/<id>/recordings/<rid>/transcript/replace` | Find & replace across the transcript text. Tag spans and comments live in a separate file and reference character offsets, so `services/transcript_edit.py` remaps every offset when a replacement changes a segment's length |
| POST | `/<id>/recordings/<rid>/transcript/block` | Rewrite one speaker block in place; body `{ "indices": [int], "original_text", "new_text" }`. `indices` names the block as `services/speaker_blocks.py` folded it — not a range, since an interrupted speaker resumes into the block they opened. `original_text` is what the editor started from: **409** `{"error": "stale"}` when it no longer matches, so a second tab's edit is refused rather than overwritten. **400** when `new_text` is empty or `indices` is not a non-empty list of ints, **404** unless the recording is `transcribed`. Returns `{changed, transcript, annotations, indices, block_text}`; `changed: false` means the text came back identical and nothing was written |
| GET/POST | `/<id>/recordings/<rid>/export` | Export single recording; body `{ "format": "markdown" \| "odt", "include_comments", "include_tags", "include_participant_details", "include_project_details", "include_prompt", "remove_pii", "use_recording_screen_prompt" }`. The last one swaps the project's LLM prompt for the recording-screen one (`export_default_prompt_recording` in Settings) |
| GET/POST | `/<id>/export` | Export project (multiple recordings); body `{ "recording_ids", "format", "include_comments", "include_tags", "include_participant_details", "include_project_details", "include_prompt", "remove_pii", "separate_files" }`; with `separate_files: true` the response is a ZIP holding one document per recording |

### Tags & Annotations

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/<id>/tags` | Project tags |
| PATCH | `/<id>/tags` | Update tags; body `{ "tags": [...] }` |
| GET | `/<id>/tags/quotes` | Tagged quotes grouped by recording (cached by transcript mtime — only changed files are re-read) |
| GET/PATCH | `/<id>/themes` | Tag themes: a two-level theme → tags hierarchy, stored per project |
| GET | `/<id>/recordings/<rid>/annotations` | Annotations (tag_spans, comments, speaker_labels) |
| PATCH | `/<id>/recordings/<rid>/annotations` | Update annotations |

### Segments

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/<id>/segments` | List segments for project |
| POST | `/<id>/segments` | Create segment; body `{ "name", "description?", "screener_questions?", "target_count?" }` |
| PATCH | `/<id>/segments/<sid>` | Update segment; body `{ "name?", "description?", "screener_questions?", "target_count?" }` |
| DELETE | `/<id>/segments/<sid>` | Delete segment |

### Attachments

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/<id>/attachments` | List attachments for project |
| POST | `/<id>/attachments` | Upload attachment (multipart `file`) |
| PATCH | `/<id>/attachments/<aid>` | Rename attachment; body `{ "display_name" }` |
| DELETE | `/<id>/attachments/<aid>` | Delete attachment |
| GET | `/<id>/attachments/<aid>/download` | Download attachment file |

### Single Transcriptions (`/api/projects/single-transcriptions`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/single-transcriptions` | List standalone recordings (no project) |
| POST | `/single-transcriptions` | Upload file for standalone transcription (multipart `file`) |
| DELETE | `/single-transcriptions/<rid>` | Delete standalone recording |

---

## Backups (`/api/backup`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `` | Create backup; body `{ "include_audio?" }` |
| GET | `` | List available backups |
| GET | `/<filename>/manifest` | Get backup manifest (project list) |
| POST | `/restore` | Restore from backup; body `{ "filename", "project_folders?", "restore_settings?", "conflict_strategy?" }` |
| POST | `/upload` | Upload backup ZIP (multipart `file`) |
| POST | `/open-folder` | Reveal the backup folder in Explorer/Finder |
| DELETE | `/<filename>` | Delete backup file |

---

## Settings (`/api/settings`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `` | All settings + `hf_token_masked`, `models`, `stt_models` |
| PATCH | `` | Update any of the writable keys below. Anything outside that set is ignored, and a value that fails validation is dropped rather than erroring — except `stt_model_id`, which returns **409** when the model is not on disk |
| POST | `/stt-model/install` | Download the transcription model; body `{ "model_id" }` |
| POST | `/pii-model/install` | Download the optional GLiNER PII model after onboarding |
| POST | `/pii-model/remove` | Delete it and free the disk space |
| POST | `/check-update` | Ask GitHub whether a newer PINE release exists |
| POST | `/reset` | Reset settings to defaults |
| GET | `/app-launch` | Both shortcut states in one call: `{ "prompt_dismissed", "start_menu": { "supported", "added", "label", "description" }, "desktop": {…} }`. `prompt_dismissed` drives the first-run offer in the sidebar; adding either shortcut sets it |
| GET/POST | `/start-menu`, `/start-menu/add`, `/start-menu/remove` | Windows Start-menu shortcut |
| GET/POST | `/desktop`, `/desktop/add`, `/desktop/remove` | Desktop shortcut |

Writable keys (`ALLOWED_KEYS` in `api/settings.py`), stored as strings:

| Group | Keys |
|-------|------|
| Appearance | `font_size`, `font_family` (whitelist), `theme` |
| Export defaults | `export_default_format`, `export_default_comments`, `export_default_tags`, `export_default_project_details`, `export_default_participant_details`, `export_default_remove_pii`, `export_default_include_prompt`, `export_default_prompt`, `export_default_prompt_recording` |
| Transcription | `stt_model_id` (409 if not installed), `transcription_timeout_secs`, `link_recordings`, `transcription_complete_sound_enabled`, `transcription_complete_sound_volume` (0–100) |
| Backup | `backup_path`, `backup_include_audio`, `auto_backup_enabled`, `auto_backup_interval_hours`, `backup_retention_count` — touching any of these reschedules the auto-backup timer |
| Other | `pii_threshold`, `last_open_project_id` (digits only), `app_launch_prompt_dismissed` |

---

## Other

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Readiness probe for the launcher; returns `{ "ok": true, "version" }` |
| POST | `/api/quit` | Graceful server shutdown |
| POST | `/api/utils/pick-file` | Native file picker; returns `{ "path" }` |
| POST | `/api/utils/pick-files` | Native multi-file picker; returns `{ "paths" }` |
| POST | `/api/utils/pick-folder` | Native folder picker; returns `{ "path" }` |
| POST | `/api/utils/inspect-multitrack` | Report per-speaker tracks without importing; body `{ "folder" }` or `{ "path" }`, returns `{ "kind", "tracks": [{ "path", "speaker_name", "channel"? }], "media" }` |
| GET | `/api/utils/runtime-status` | Backend runtime state for the launcher |
| POST | `/api/utils/restart` | Restart the backend process |
| POST | `/api/internal/quit-backend` | Internal shutdown signal used by the supervisor — not for UI use |

---

## Data Shapes

### Project

```json
{
  "id": 1,
  "name": "Project Name",
  "description": "",
  "objective": "",
  "research_questions": [],
  "hypotheses": [],
  "summary": "",
  "results_recommendations": "",
  "further_steps": "",
  "stakeholders": [],
  "enabled_sections": [],
  "methodology": "",
  "interview_guide": "",
  "key_findings": "",
  "recommendations": "",
  "enabled_summary_sections": ["key_findings"],
  "custom_sections": [],
  "custom_summary_sections": [],
  "icon": "",
  "is_archived": false,
  "archived_at": null,
  "is_system": false,
  "folder_name": "Project_Name",
  "default_transcription_language": "",
  "recording_count": 3,
  "created_at": "...",
  "updated_at": "..."
}
```

The seven list fields hold JSON in the database and come back parsed; a corrupt
value reads as `[]` rather than raising. `recordings` is **not** part of this
shape — only `GET /<id>` and the transfer package ask for it
(`to_dict(include_recordings=True)`), everywhere else you get `recording_count`.

### Recording

```json
{
  "id": 1,
  "project_id": 1,
  "original_name": "interview.mp3",
  "stored_name": "interview.mp3",
  "transcription_status": "transcribed",
  "transcript_path": "interview_transcript.json",
  "duration_seconds": 3600,
  "file_format": "mp3",
  "file_size_bytes": 45000000,
  "language": "en",
  "error_message": null,
  "segment_id": null,
  "participant_notes": "",
  "num_speakers": 2,
  "is_linked": false,
  "source_kind": "single",
  "track_count": 0,
  "created_at": "..."
}
```

`source_kind` is `"single"` or `"multitrack"`; `track_count` is 0 for single-file
recordings and the number of speaker tracks otherwise. `GET /<id>` adds a
`segment_name` to each recording it returns; no other endpoint does.

`transcription_status` is one of:

| Value | Meaning |
|-------|---------|
| `pending` | Queued, or waiting for the single worker |
| `transcribing` | The worker is on it |
| `awaiting_language` | Stopped mid-job for a language answer; the job thread is parked, not dead |
| `transcribed` | Done — `transcript_path` is written |
| `error` | Failed; `error_message` says why |
| `cancelling` | Cancellation asked for, worker not yet stopped |
| `cancelled` | The worker confirmed the stop |

### Segment

```json
{
  "id": 1,
  "project_id": 1,
  "name": "Power Users",
  "description": "Daily active users with 6+ months tenure",
  "screener_questions": "How often do you use the product?",
  "target_count": 5,
  "assigned_count": 2,
  "created_at": "..."
}
```

### Attachment

```json
{
  "id": "a1b2c3d4",
  "display_name": "research_brief.pdf",
  "stored_name": "a1b2c3d4.pdf",
  "uploaded_at": "2025-04-12T10:30:00.000000+00:00"
}
```

### Transcript segment

```json
{
  "start": 0.5,
  "end": 3.2,
  "text": "Hello world",
  "speaker": "SPEAKER_00"
}
```

### Tag span

```json
{
  "segment_idx": 0,
  "tag_id": "pain",
  "start_char": 0,
  "end_char": 5
}
```

---

## SocketIO Events

All events are server → client. Names use a `namespace:event` form.

| Event | Payload |
|-------|---------|
| `transcription:status` | `{ "recording_id", "status", "stage", "message", "percent"? }` — plus any extra keys the emitting stage adds. `status` is the recording's state, `stage` the step within it (`queued`, `asr`, `diarize`, `language`, …). Queue-position pings reuse this event with `stage: "queued"` and a `#k of N` message. With `status: "awaiting_language"` it also carries `guessed_language`, `language_confidence` (`-1.0` when the detector returned none) and `language_options` |
| `download:model_start` / `download:progress` / `download:model_complete` | Per-model download of weights |
| `download:all_complete` / `download:error` | End of the whole download batch |
| `install:start` / `install:log` / `install:complete` / `install:error` | pip install of the ML stack during onboarding — `install:log` carries raw output lines |
| `backup:progress` / `backup:complete` / `backup:error` | Backup creation |
| `backup:restore_progress` / `backup:restore_complete` / `backup:restore_error` | Restore |
