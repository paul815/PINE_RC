"""Backup and restore service for PINE data."""

import hashlib
import json
import logging
import ntpath
import os
import posixpath
import re
import shutil
import tempfile
import threading
import time
import zipfile
from datetime import UTC, datetime
from glob import glob

from ..extensions import db, socketio
from ..models.project import Project
from ..models.recording import Recording
from ..models.segment import Segment
from ..models.setting import Setting
from .annotations import annotations_filename
from .file_utils import safe_child_path, safe_project_dir
from .multitrack_ingest import TRACK_EXTENSIONS
from .setting_rules import RESTORABLE_SETTINGS, clean_setting

log = logging.getLogger(__name__)

_backup_lock = threading.Lock()
_auto_backup_thread = None
_auto_backup_stop = None
_auto_backup_lock = threading.Lock()

# Settings keys that belong to *this* installation, not to the backup.
#
# Left out on write. (A restore goes further and takes only the preferences in
# setting_rules.RESTORABLE_SETTINGS, so a backup cannot repoint this install
# at another machine's paths.) A backup ZIP travels: onto a USB stick, into a cloud-synced folder, to whoever the user
# asks for help. hf_token is a live HuggingFace credential, and models_path /
# projects_path are local filesystem layout — neither has any business leaving
# the machine inside an archive nobody expects to hold secrets.
_SENSITIVE_KEYS = frozenset({
    'hf_token', 'models_path', 'projects_path', 'onboarding_complete',
})

BACKUP_VERSION = 2


def _backup_dir(app):
    """Resolve backup directory path, creating it if needed."""
    default = os.path.join(app.config.get('ROOT_DIR', ''), 'backups')
    path = Setting.get('backup_path', default) or default
    if not os.path.isabs(path):
        path = os.path.join(app.config.get('ROOT_DIR', ''), path)
    os.makedirs(path, exist_ok=True)
    return path


def _external_safety_backup_dir():
    path = os.path.join(tempfile.gettempdir(), 'pine_safety_backups')
    os.makedirs(path, mode=0o700, exist_ok=True)
    # Snapshots hold every project with its audio, and on Linux this sits in a
    # shared /tmp; makedirs' mode does not touch a folder that already exists.
    # On Windows the user's own %TEMP% already keeps others out.
    try:
        os.chmod(path, 0o700)
    except OSError:
        log.warning('Could not restrict permissions on %s', path)
    return path


def _projects_root(app):
    return Setting.get('projects_path', app.config['DEFAULT_PROJECTS_PATH'])


def _emit(event, data):
    try:
        socketio.emit(event, data)
    except Exception:
        pass


def _sanitize_backup_label(label):
    safe = re.sub(r'[^a-z0-9_-]+', '_', (label or '').strip().lower())
    return safe.strip('_')[:48]


def _build_backup_filename(now, include_audio, label=None):
    audio_suffix = '_with_audio' if include_audio else ''
    label_suffix = ''
    safe_label = _sanitize_backup_label(label)
    if safe_label:
        label_suffix = f'_{safe_label}'
    return f'pine_backup_{now.strftime("%Y-%m-%dT%H-%M-%S")}{audio_suffix}{label_suffix}.zip'


def _writestr_with_checksum(zf, arc_name, data, checksums):
    payload = data.encode('utf-8') if isinstance(data, str) else data
    zf.writestr(arc_name, payload)
    checksums[arc_name] = {
        'sha256': hashlib.sha256(payload).hexdigest(),
        'size_bytes': len(payload),
    }


def _write_file_with_checksum(zf, src_path, arc_name, checksums):
    digest = hashlib.sha256()
    size = 0
    with open(src_path, 'rb') as src, zf.open(arc_name, 'w', force_zip64=True) as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            dst.write(chunk)
    checksums[arc_name] = {
        'sha256': digest.hexdigest(),
        'size_bytes': size,
    }


def _safe_archive_leaf(name, fallback='file'):
    leaf = os.path.basename(name or '').strip()
    if not leaf:
        leaf = fallback
    leaf = re.sub(r'[^A-Za-z0-9._-]+', '_', leaf).strip('._') or fallback
    return leaf[:180]


def _backup_audio_name(recording):
    base = os.path.splitext(
        os.path.basename(recording.stored_name or recording.original_name or 'recording')
    )[0] or 'recording'
    ext = os.path.splitext(
        os.path.basename(recording.stored_name or recording.original_name or '')
    )[1]
    if not ext and recording.file_format:
        ext = f'.{recording.file_format.lstrip(".")}'
    return f'{_safe_archive_leaf(base, "recording")}_{recording.id}{ext.lower()}'


def _is_linkable_media(path):
    """Whether *path* can be the target of a linked recording.

    Linking only ever accepts absolute paths to media; holding a restored or
    stored link to the same rule keeps a crafted row from turning the backup
    into a way to copy any file off the machine. Absolute by either OS's
    rules, because backups move between the Windows and Mac machines and a
    link from the other one must not sink the whole restore.
    """
    path = path or ''
    ext = os.path.splitext(path)[1].lstrip('.').lower()
    is_abs = ntpath.isabs(path) or posixpath.isabs(path)
    return bool(path) and is_abs and ext in TRACK_EXTENSIONS


def _inside_or_none(root, rel_path):
    try:
        return safe_child_path(root, rel_path)
    except ValueError:
        log.warning('Backup skipped a path outside %s: %r', root, rel_path)
        return None


# What a recording row may be restored as. In-flight states are not among
# them: requeue_interrupted() re-runs 'transcribing' and 'awaiting_language' at
# the next start, so a backup could make the worker open a path on its own.
_RESTORABLE_STATUSES = frozenset({'pending', 'transcribed', 'error', 'cancelled'})
_FILE_FORMAT_RE = re.compile(r'[a-z0-9][a-z0-9_-]{0,15}')
_LANGUAGE_RE = re.compile(r'[a-z]{2,8}(-[a-z0-9]{2,8})?')


def _is_network_path(path):
    r"""A UNC share, a device path (\\?\, \\.\) or //host — anything off this disk.

    Opening one sends an SMB request, with the user's NTLM hash, to whatever
    host it names.
    """
    return (path or '').replace('\\', '/').startswith('//')


def _restored_status(value):
    status = str(value or '').strip()
    return status if status in _RESTORABLE_STATUSES else 'pending'


def _restored_token(value, pattern):
    """A short code (file format, language) in its expected shape, else ''."""
    text = str(value or '').strip().lower()
    return text if pattern.fullmatch(text) else ''


def _restored_number(value, cast):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return cast(0)
    return cast(number) if number == number and abs(number) != float('inf') else cast(0)


def _restored_flag(value):
    return value is True or value == 1 or str(value).strip().lower() == 'true'


def _serialize_recording_for_backup(recording, include_audio):
    data = recording.to_dict()
    if recording.is_linked:
        data['linked_source_path'] = recording.stored_name
        if (include_audio and _is_linkable_media(recording.stored_name)
                and os.path.isfile(recording.stored_name)):
            data['backup_archived_audio_name'] = _backup_audio_name(recording)
    return data


def _serialize_project_for_backup(project, include_audio):
    data = project.to_dict(include_recordings=False)
    data['recordings'] = [
        _serialize_recording_for_backup(recording, include_audio)
        for recording in project.recordings
    ]
    return data


def _read_manifest_from_zip(zf):
    if 'manifest.json' not in zf.namelist():
        return None
    return json.loads(zf.read('manifest.json'))


def validate_backup_zip(zip_path):
    """Validate a backup ZIP and return ``(manifest, error_message)``."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            manifest = _read_manifest_from_zip(zf)
            if manifest is None:
                return None, 'Backup manifest is missing.'

            if manifest.get('backup_version', 0) >= 2:
                checksums = manifest.get('checksums')
                if not isinstance(checksums, dict) or not checksums:
                    return manifest, 'Backup manifest is missing checksums.'

                for arc_name, meta in checksums.items():
                    try:
                        entry = zf.getinfo(arc_name)
                    except KeyError:
                        return manifest, f'Backup is missing archive entry: {arc_name}'

                    expected_size = int(meta.get('size_bytes', -1))
                    if entry.file_size != expected_size:
                        return manifest, f'Backup entry has unexpected size: {arc_name}'

                    digest = hashlib.sha256()
                    with zf.open(entry, 'r') as src:
                        while True:
                            chunk = src.read(1024 * 1024)
                            if not chunk:
                                break
                            digest.update(chunk)
                    if digest.hexdigest() != meta.get('sha256'):
                        return manifest, f'Backup checksum mismatch: {arc_name}'

            return manifest, None
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
        return None, f'Invalid backup archive: {exc}'


def _safe_restore_path(root, rel_path):
    return safe_child_path(root, rel_path)


def _validate_restore_paths(projects_data, projects_root):
    """Refuse an archive whose paths would reach outside the projects root.

    Everything in ``db/projects.json`` is the archive author's to choose, and
    the manifest checksums are too, so a backup is checked like any other
    outside input. ``folder_name`` decides where files are written and, on a
    failed restore, what gets removed; ``stored_name`` and ``transcript_path``
    become DB rows that the media, delete and backup code later follow. Runs
    before anything is written, so a refused archive changes nothing.
    """
    for pdata in projects_data:
        folder = pdata.get('folder_name')
        project_dir = safe_project_dir(projects_root, folder)

        for rdata in pdata.get('recordings', []):
            stored_name = rdata.get('stored_name') or ''
            archived = (rdata.get('backup_archived_audio_name') or '').strip()
            if rdata.get('is_linked', False) and not archived:
                # A link to media outside PINE (a Zoom folder, an external
                # drive) is legitimate — but only to media, never to a file
                # the next backup would then sweep up and carry off.
                if not _is_linkable_media(stored_name):
                    raise ValueError(
                        f'Unsafe linked recording in project {folder!r}: {stored_name!r}'
                    )
            else:
                safe_child_path(project_dir, archived or stored_name)

            transcript_path = rdata.get('transcript_path') or ''
            if transcript_path:
                safe_child_path(project_dir, transcript_path)


def _stage_project_files(zf, original_folder, staging_root):
    stage_dir = os.path.join(staging_root, original_folder)
    os.makedirs(stage_dir, exist_ok=True)
    prefix = f'projects/{original_folder}/'
    for name in zf.namelist():
        if not name.startswith(prefix) or name.endswith('/'):
            continue
        rel = name[len(prefix):]
        dest = _safe_restore_path(stage_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with zf.open(name, 'r') as src, open(dest, 'wb') as dst:
            shutil.copyfileobj(src, dst)
    return stage_dir


def _reserve_temp_path(base_dir, prefix):
    path = tempfile.mkdtemp(prefix=prefix, dir=base_dir)
    os.rmdir(path)
    return path


def _set_setting_value(key, value):
    setting = db.session.get(Setting, key)
    if setting:
        setting.value = str(value)
    else:
        db.session.add(Setting(key=key, value=str(value)))


def create_safety_snapshot(app, reason, include_audio=False, project_folders=None,
                           backup_dir=None):
    """Create a synchronous backup before a destructive action."""
    if not _backup_lock.acquire(timeout=5):
        raise RuntimeError('Could not acquire backup lock for safety snapshot.')
    try:
        with app.app_context():
            return _create_backup_inner(
                app,
                include_audio=include_audio,
                project_folders=project_folders,
                label=f'safety_{reason}',
                backup_dir=backup_dir,
                verify=True,
            )
    finally:
        _backup_lock.release()


def create_backup(app, include_audio=False, project_folders=None, label=None,
                  backup_dir=None):
    """Create a backup ZIP in a background thread."""
    if not _backup_lock.acquire(blocking=False):
        _emit('backup:error', {'message': 'Another backup/restore is already running.'})
        return None

    def _run():
        try:
            _emit('backup:progress', {'stage': 'init', 'percent': 0, 'message': 'Starting backup...'})
            with app.app_context():
                result = _create_backup_inner(
                    app,
                    include_audio=include_audio,
                    project_folders=project_folders,
                    label=label,
                    backup_dir=backup_dir,
                    verify=True,
                )
            if result:
                _emit('backup:complete', result)
        except Exception as exc:
            log.exception('Backup failed')
            _emit('backup:error', {'message': str(exc)})
        finally:
            _backup_lock.release()

    thread = threading.Thread(target=_run, daemon=True, name='backup-create')
    thread.start()
    return thread


def _create_backup_inner(app, include_audio, project_folders=None, label=None,
                         backup_dir=None, verify=True):
    backup_path = backup_dir or _backup_dir(app)
    os.makedirs(backup_path, exist_ok=True)
    projects_root = _projects_root(app)
    now = datetime.now(UTC)
    filename = _build_backup_filename(now, include_audio, label=label)
    zip_path = os.path.join(backup_path, filename)

    query = Project.query
    if project_folders is not None:
        query = query.filter(Project.folder_name.in_(list(project_folders)))
    projects = query.all()
    project_ids = [project.id for project in projects]
    if project_ids:
        segments = Segment.query.filter(Segment.project_id.in_(project_ids)).all()
    else:
        segments = []

    manifest_projects = []
    for project in projects:
        manifest_projects.append({
            'id': project.id,
            'name': project.name,
            'folder_name': project.folder_name,
            'recording_count': len(project.recordings),
        })

    total_recordings = sum(len(project.recordings) for project in projects)
    total_steps = len(projects) + total_recordings + len(projects)
    done_steps = 0

    def progress(stage, message):
        nonlocal done_steps
        done_steps += 1
        pct = min(int(done_steps / max(total_steps, 1) * 100), 99)
        _emit('backup:progress', {'stage': stage, 'percent': pct, 'message': message})

    checksums = {}

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        _emit('backup:progress', {'stage': 'db', 'percent': 5, 'message': 'Exporting database...'})

        all_settings = {
            row.key: row.value
            for row in Setting.query.all()
            if row.key not in _SENSITIVE_KEYS
        }
        _writestr_with_checksum(
            zf,
            'db/settings.json',
            json.dumps(all_settings, ensure_ascii=False, indent=2),
            checksums,
        )

        projects_data = [_serialize_project_for_backup(project, include_audio) for project in projects]
        _writestr_with_checksum(
            zf,
            'db/projects.json',
            json.dumps(projects_data, ensure_ascii=False, indent=2),
            checksums,
        )

        segments_data = [segment.to_dict() for segment in segments]
        _writestr_with_checksum(
            zf,
            'db/segments.json',
            json.dumps(segments_data, ensure_ascii=False, indent=2),
            checksums,
        )

        for project in projects:
            try:
                project_dir = safe_project_dir(projects_root, project.folder_name)
            except ValueError:
                log.warning('Backup skipped files of project %s: unsafe folder_name %r',
                            project.id, project.folder_name)
                progress('files', f'Skipping {project.name} (unsafe folder)')
                continue
            if not os.path.isdir(project_dir):
                progress('files', f'Skipping {project.name} (no folder)')
                continue

            arc_prefix = f'projects/{project.folder_name}'

            tags_file = os.path.join(project_dir, 'project_tags.json')
            if os.path.isfile(tags_file):
                _write_file_with_checksum(zf, tags_file, f'{arc_prefix}/project_tags.json', checksums)

            att_meta = os.path.join(project_dir, 'attachments.json')
            if os.path.isfile(att_meta):
                _write_file_with_checksum(zf, att_meta, f'{arc_prefix}/attachments.json', checksums)
            att_dir = os.path.join(project_dir, 'attachments')
            if os.path.isdir(att_dir):
                for att_file in os.listdir(att_dir):
                    att_path = os.path.join(att_dir, att_file)
                    if os.path.isfile(att_path):
                        _write_file_with_checksum(
                            zf,
                            att_path,
                            f'{arc_prefix}/attachments/{_safe_archive_leaf(att_file, "attachment")}',
                            checksums,
                        )

            for recording in project.recordings:
                if recording.transcript_path:
                    transcript_path = _inside_or_none(project_dir, recording.transcript_path)
                    if transcript_path and os.path.isfile(transcript_path):
                        _write_file_with_checksum(
                            zf,
                            transcript_path,
                            f'{arc_prefix}/{recording.transcript_path}',
                            checksums,
                        )

                ann_name = annotations_filename(recording.transcript_path or recording.stored_name)
                ann_file = os.path.join(project_dir, ann_name)
                if os.path.isfile(ann_file):
                    _write_file_with_checksum(zf, ann_file, f'{arc_prefix}/{ann_name}', checksums)

                if include_audio:
                    archived_audio_name = recording.stored_name
                    if recording.is_linked:
                        audio_file = (recording.stored_name
                                      if _is_linkable_media(recording.stored_name) else None)
                        archived_audio_name = _backup_audio_name(recording)
                    else:
                        audio_file = _inside_or_none(project_dir, recording.stored_name)
                    if audio_file and os.path.isfile(audio_file):
                        _write_file_with_checksum(
                            zf,
                            audio_file,
                            f'{arc_prefix}/{archived_audio_name}',
                            checksums,
                        )

                progress('files', f'{project.name}: {recording.original_name}')

        _emit('backup:progress', {'stage': 'exports', 'percent': 80, 'message': 'Generating ODT exports...'})
        from .export_service import export_project_odt
        for project in projects:
            recording_ids = [
                recording.id
                for recording in project.recordings
                if recording.transcription_status == 'transcribed'
            ]
            if not recording_ids:
                progress('exports', f'Skipping export for {project.name} (no transcripts)')
                continue
            try:
                odt_bytes, err = export_project_odt(
                    app, project.id, recording_ids,
                    {
                        'include_comments': True,
                        'include_tags': True,
                        'include_project_details': True,
                        'include_participant_details': True,
                        'remove_pii': False,
                    },
                )
                if odt_bytes and not err:
                    safe_name = project.folder_name or f'project_{project.id}'
                    _writestr_with_checksum(zf, f'exports/{safe_name}.odt', odt_bytes, checksums)
            except Exception as exc:
                log.warning('ODT export failed for project %s: %s', project.name, exc)
            progress('exports', f'Exported {project.name}')

        manifest = {
            'pine_version': _get_version(),
            'backup_version': BACKUP_VERSION,
            'created_at': now.isoformat(),
            'include_audio': include_audio,
            'project_count': len(projects),
            'recording_count': total_recordings,
            'projects': manifest_projects,
            'checksums': checksums,
        }
        zf.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))

    size = os.path.getsize(zip_path)
    manifest_out = manifest
    if verify:
        validated_manifest, error = validate_backup_zip(zip_path)
        if error:
            try:
                os.remove(zip_path)
            except OSError:
                pass
            raise ValueError(f'Backup verification failed: {error}')
        manifest_out = validated_manifest or manifest

    if backup_dir is None:
        prune_backups(app)

    _emit('backup:progress', {'stage': 'done', 'percent': 100, 'message': 'Backup complete.'})
    return {
        'filename': filename,
        'size_bytes': size,
        'project_count': len(projects),
        'created_at': manifest_out.get('created_at'),
        'backup_dir': backup_path,
    }


def _get_version():
    try:
        from .. import __version__
        return __version__
    except Exception:
        return 'unknown'


def list_backups(app):
    """Return metadata for all backup ZIPs in the backup directory."""
    backup_path = _backup_dir(app)
    results = []
    pattern = os.path.join(backup_path, 'pine_backup_*.zip')
    for path in sorted(glob(pattern), reverse=True):
        try:
            manifest = _read_manifest(path)
            if manifest:
                results.append({
                    'filename': os.path.basename(path),
                    'size_bytes': os.path.getsize(path),
                    'created_at': manifest.get('created_at', ''),
                    'include_audio': manifest.get('include_audio', False),
                    'project_count': manifest.get('project_count', 0),
                    'recording_count': manifest.get('recording_count', 0),
                    'backup_version': manifest.get('backup_version', 0),
                })
        except Exception:
            log.warning('Could not read manifest from %s', path)
    return results


def get_manifest(app, filename):
    """Read and return the full manifest from a backup ZIP."""
    backup_path = _backup_dir(app)
    zip_path = os.path.join(backup_path, os.path.basename(filename))
    if not os.path.isfile(zip_path):
        return None
    return _read_manifest(zip_path)


def _read_manifest(zip_path):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            return _read_manifest_from_zip(zf)
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError):
        return None


def restore_backup(app, filename, project_folders=None, restore_settings=False,
                   conflict_strategy='skip'):
    """Restore data from a backup ZIP in a background thread."""
    if not _backup_lock.acquire(blocking=False):
        _emit('backup:restore_error', {'message': 'Another backup/restore is already running.'})
        return None

    def _run():
        try:
            _emit('backup:restore_progress', {'stage': 'init', 'percent': 0, 'message': 'Starting restore...'})
            with app.app_context():
                result = _restore_inner(
                    app,
                    filename=filename,
                    project_folders=project_folders,
                    restore_settings=restore_settings,
                    conflict_strategy=conflict_strategy,
                )
            if result:
                _emit('backup:restore_complete', result)
        except Exception as exc:
            log.exception('Restore failed')
            _emit('backup:restore_error', {'message': str(exc)})
        finally:
            _backup_lock.release()

    thread = threading.Thread(target=_run, daemon=True, name='backup-restore')
    thread.start()
    return thread


def _restore_inner(app, filename, project_folders, restore_settings, conflict_strategy):
    backup_path = _backup_dir(app)
    zip_path = os.path.join(backup_path, os.path.basename(filename))
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f'Backup file not found: {filename}')

    manifest, error = validate_backup_zip(zip_path)
    if error:
        raise ValueError(error)
    if manifest is None:
        raise ValueError('Backup manifest is missing.')
    if manifest.get('backup_version', 0) > BACKUP_VERSION:
        raise ValueError(
            f'Backup version {manifest["backup_version"]} is newer than supported ({BACKUP_VERSION})'
        )

    projects_root = _projects_root(app)
    os.makedirs(projects_root, exist_ok=True)
    staging_root = tempfile.mkdtemp(prefix='pine_restore_stage_', dir=projects_root)
    restored_count = 0
    settings_restored = False
    safety_backup = None
    project_file_actions = []
    old_dirs_to_cleanup = []

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            projects_data = json.loads(zf.read('db/projects.json'))
            segments_data = json.loads(zf.read('db/segments.json'))
            if project_folders is not None:
                wanted = set(project_folders)
                projects_data = [pdata for pdata in projects_data if pdata['folder_name'] in wanted]

            _validate_restore_paths(projects_data, projects_root)

            existing_targets = []
            for pdata in projects_data:
                if Project.query.filter_by(folder_name=pdata['folder_name']).first() is not None:
                    existing_targets.append(pdata['folder_name'])

            if existing_targets or restore_settings:
                safety_backup = _create_backup_inner(
                    app,
                    include_audio=True,
                    project_folders=list(set(existing_targets)) or None,
                    label='pre_restore',
                    verify=True,
                )

            if restore_settings and 'db/settings.json' in zf.namelist():
                settings_json = json.loads(zf.read('db/settings.json'))
                if not isinstance(settings_json, dict):
                    settings_json = {}
                for key, value in settings_json.items():
                    # Preferences only, and only in a valid shape: see
                    # RESTORABLE_SETTINGS for what stays with this machine.
                    if key not in RESTORABLE_SETTINGS:
                        continue
                    cleaned = clean_setting(key, value)
                    if cleaned is None:
                        log.warning('Restore skipped setting %s: invalid value', key)
                        continue
                    _set_setting_value(key, cleaned)
                settings_restored = True
                _emit('backup:restore_progress', {
                    'stage': 'settings',
                    'percent': 10,
                    'message': 'Settings staged.',
                })

            total = len(projects_data)

            for index, pdata in enumerate(projects_data):
                original_folder = pdata['folder_name']
                folder = original_folder
                existing = Project.query.filter_by(folder_name=folder).first()
                final_dir = safe_project_dir(projects_root, folder)
                # A folder can sit on disk with no DB row behind it: a project
                # deleted from the UI whose files survived, a hand-copied
                # folder, or the leftovers of an interrupted restore. That
                # desync is exactly what a restore is run to repair, so it has
                # to count as a conflict — otherwise no strategy applies and
                # the staging loop below dies on FileExistsError.
                orphan_dir = existing is None and os.path.isdir(final_dir)
                pct = 10 + int((index + 1) / max(total, 1) * 85)
                _emit('backup:restore_progress', {
                    'stage': 'projects',
                    'percent': pct,
                    'message': f'Staging {pdata["name"]}...',
                })

                if existing or orphan_dir:
                    if conflict_strategy == 'skip':
                        continue
                    if conflict_strategy == 'overwrite':
                        if existing:
                            db.session.delete(existing)
                            db.session.flush()
                    elif conflict_strategy == 'rename':
                        counter = 1
                        candidate = f'{folder}_restored_{counter}'
                        while Project.query.filter_by(folder_name=candidate).first() or \
                                os.path.isdir(os.path.join(projects_root, candidate)):
                            counter += 1
                            candidate = f'{folder}_restored_{counter}'
                        folder = candidate
                        final_dir = safe_project_dir(projects_root, folder)

                stage_dir = _stage_project_files(zf, original_folder, staging_root)

                old_project_id = pdata['id']
                old_segment_ids = {
                    sdata['id']
                    for sdata in segments_data
                    if sdata['project_id'] == old_project_id
                }
                seg_id_map = {}
                for sdata in segments_data:
                    if sdata['id'] not in old_segment_ids:
                        continue
                    segment = Segment(
                        project_id=0,
                        name=sdata.get('name', ''),
                        description=sdata.get('description', ''),
                        screener_questions=sdata.get('screener_questions', ''),
                        target_count=_restored_number(sdata.get('target_count', 0), int),
                    )
                    db.session.add(segment)
                    db.session.flush()
                    seg_id_map[sdata['id']] = segment.id

                project = Project(
                    name=pdata['name'],
                    description=pdata.get('description', ''),
                    objective=pdata.get('objective', ''),
                    research_questions=json.dumps(pdata.get('research_questions', [])),
                    hypotheses=json.dumps(pdata.get('hypotheses', [])),
                    summary=pdata.get('summary', ''),
                    results_recommendations=pdata.get('results_recommendations', ''),
                    further_steps=pdata.get('further_steps', ''),
                    stakeholders=json.dumps(pdata.get('stakeholders', [])),
                    enabled_sections=json.dumps(pdata.get('enabled_sections', [])),
                    methodology=pdata.get('methodology', ''),
                    interview_guide=pdata.get('interview_guide', ''),
                    key_findings=pdata.get('key_findings', ''),
                    recommendations=pdata.get('recommendations', ''),
                    enabled_summary_sections=json.dumps(
                        pdata.get('enabled_summary_sections', ['key_findings'])
                    ),
                    custom_sections=json.dumps(pdata.get('custom_sections', [])),
                    custom_summary_sections=json.dumps(pdata.get('custom_summary_sections', [])),
                    icon=pdata.get('icon', ''),
                    is_archived=_restored_flag(pdata.get('is_archived', False)),
                    is_system=_restored_flag(pdata.get('is_system', False)),
                    folder_name=folder,
                    default_transcription_language=_restored_token(
                        pdata.get('default_transcription_language'), _LANGUAGE_RE),
                )
                db.session.add(project)
                db.session.flush()

                for new_segment_id in seg_id_map.values():
                    segment = db.session.get(Segment, new_segment_id)
                    if segment:
                        segment.project_id = project.id

                for rdata in pdata.get('recordings', []):
                    old_seg_id = rdata.get('segment_id')
                    new_seg_id = seg_id_map.get(old_seg_id) if old_seg_id else None
                    stored_name = rdata['stored_name']
                    is_linked = rdata.get('is_linked', False)
                    archived_audio_name = (rdata.get('backup_archived_audio_name') or '').strip()
                    if is_linked and archived_audio_name:
                        stored_name = archived_audio_name
                        is_linked = False
                    elif is_linked and _is_network_path(stored_name):
                        # The recording, transcript and codes come back; the
                        # link to a share does not. A NAS link in one's own
                        # backup is legitimate, but nothing here can tell it
                        # from a host waiting for this machine's NTLM hash.
                        log.warning('Restore dropped a network link for recording %r',
                                    rdata.get('original_name'))
                        stored_name = ''

                    recording = Recording(
                        project_id=project.id,
                        original_name=rdata['original_name'],
                        stored_name=stored_name,
                        file_format=_restored_token(rdata.get('file_format'), _FILE_FORMAT_RE),
                        file_size_bytes=_restored_number(rdata.get('file_size_bytes', 0), int),
                        duration_seconds=_restored_number(rdata.get('duration_seconds', 0), float),
                        transcription_status=_restored_status(rdata.get('transcription_status')),
                        language=_restored_token(rdata.get('language'), _LANGUAGE_RE),
                        transcript_path=rdata.get('transcript_path', ''),
                        error_message=rdata.get('error_message', ''),
                        segment_id=new_seg_id,
                        participant_notes=rdata.get('participant_notes', ''),
                        is_linked=is_linked,
                    )
                    db.session.add(recording)

                project_file_actions.append({
                    'final_dir': final_dir,
                    'stage_dir': stage_dir,
                    'backup_existing_dir': None,
                    'existing_dir': os.path.join(projects_root, original_folder)
                    if (existing or orphan_dir) and conflict_strategy == 'overwrite'
                    else None,
                    # Overwriting an orphan means replacing files no DB row
                    # described, so the user never saw it in the conflict list.
                    # Kept out of the post-commit delete below.
                    'was_orphan': orphan_dir,
                })
                restored_count += 1

            for action in project_file_actions:
                final_dir = action['final_dir']
                existing_dir = action['existing_dir']
                if existing_dir and os.path.isdir(final_dir):
                    backup_existing_dir = _reserve_temp_path(projects_root, 'pine_restore_old_')
                    os.replace(final_dir, backup_existing_dir)
                    action['backup_existing_dir'] = backup_existing_dir

                if os.path.exists(final_dir):
                    raise FileExistsError(f'Restore target already exists: {final_dir}')
                if os.path.isdir(action['stage_dir']):
                    os.replace(action['stage_dir'], final_dir)
                    # Only a folder this restore put in place is its to remove
                    # on rollback; anything else at final_dir belongs to the user.
                    action['placed'] = True

            db.session.commit()

        for action in project_file_actions:
            backup_existing_dir = action.get('backup_existing_dir')
            if backup_existing_dir and os.path.isdir(backup_existing_dir):
                if action.get('was_orphan'):
                    # No DB row pointed at this folder, so it never appeared in
                    # the conflict list and the user never knowingly chose to
                    # replace it. Leave it on disk rather than deleting data
                    # PINE cannot describe.
                    log.warning(
                        'Restore replaced an untracked project folder; previous '
                        'contents kept at %s', backup_existing_dir,
                    )
                    continue
                old_dirs_to_cleanup.append(backup_existing_dir)

        for old_dir in old_dirs_to_cleanup:
            shutil.rmtree(old_dir, ignore_errors=True)

        _emit('backup:restore_progress', {'stage': 'done', 'percent': 100, 'message': 'Restore complete.'})
        return {
            'projects_restored': restored_count,
            'settings_restored': settings_restored,
            'safety_backup': safety_backup,
        }
    except Exception:
        db.session.rollback()
        for action in reversed(project_file_actions):
            final_dir = action.get('final_dir')
            backup_existing_dir = action.get('backup_existing_dir')
            if action.get('placed') and final_dir and os.path.isdir(final_dir):
                shutil.rmtree(final_dir, ignore_errors=True)
            if backup_existing_dir and os.path.isdir(backup_existing_dir):
                if os.path.exists(final_dir):
                    log.warning(
                        'Restore rollback could not clear %s; previous contents kept at %s',
                        final_dir, backup_existing_dir,
                    )
                else:
                    os.replace(backup_existing_dir, final_dir)
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def delete_backup(app, filename):
    """Delete a specific backup file."""
    backup_path = _backup_dir(app)
    zip_path = os.path.join(backup_path, os.path.basename(filename))
    if os.path.isfile(zip_path):
        os.remove(zip_path)
        return True
    return False


def prune_backups(app):
    """Keep only the N most recent backups per retention setting."""
    retention = int(Setting.get('backup_retention_count', '10') or '10')
    if retention <= 0:
        return
    backup_path = _backup_dir(app)
    pattern = os.path.join(backup_path, 'pine_backup_*.zip')
    files = sorted(glob(pattern), reverse=True)
    for old_file in files[retention:]:
        try:
            os.remove(old_file)
            log.info('Pruned old backup: %s', old_file)
        except OSError as exc:
            log.warning('Failed to prune backup %s: %s', old_file, exc)


def start_auto_backup(app):
    """Start the auto-backup daemon thread if enabled."""
    global _auto_backup_thread, _auto_backup_stop

    with app.app_context():
        enabled = Setting.get('auto_backup_enabled', 'false')
    if enabled != 'true':
        return None

    with _auto_backup_lock:
        if _auto_backup_thread is not None and _auto_backup_thread.is_alive():
            return _auto_backup_thread
        stop_event = threading.Event()
        _auto_backup_stop = stop_event

    def _scheduler(local_stop):
        while not local_stop.is_set():
            try:
                with app.app_context():
                    interval_h = int(Setting.get('auto_backup_interval_hours', '24') or '24')
                    include_audio = Setting.get('backup_include_audio', 'false') == 'true'
                    backup_path = _backup_dir(app)

                pattern = os.path.join(backup_path, 'pine_backup_*.zip')
                files = sorted(glob(pattern), reverse=True)
                should_backup = True
                if files:
                    newest = os.path.getmtime(files[0])
                    age_hours = (time.time() - newest) / 3600
                    if age_hours < interval_h:
                        should_backup = False

                if should_backup:
                    log.info('Auto-backup: creating backup...')
                    thread = create_backup(app, include_audio=include_audio)
                    if thread:
                        thread.join(timeout=3600)
            except Exception:
                log.exception('Auto-backup scheduler error')

            with app.app_context():
                interval_h = max(1, int(Setting.get('auto_backup_interval_hours', '24') or '24'))
            for _ in range(interval_h * 60):
                if local_stop.is_set():
                    return
                local_stop.wait(60)

    thread = threading.Thread(
        target=_scheduler,
        args=(_auto_backup_stop,),
        daemon=True,
        name='auto-backup-scheduler',
    )
    thread.start()
    with _auto_backup_lock:
        _auto_backup_thread = thread
    log.info('Auto-backup scheduler started.')
    return thread


def stop_auto_backup():
    """Signal the auto-backup thread to stop."""
    global _auto_backup_thread, _auto_backup_stop

    with _auto_backup_lock:
        thread = _auto_backup_thread
        stop_event = _auto_backup_stop
        _auto_backup_thread = None
        _auto_backup_stop = None

    if stop_event is not None:
        stop_event.set()
    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=2)


def refresh_auto_backup(app):
    """Apply backup setting changes immediately."""
    stop_auto_backup()
    return start_auto_backup(app)
