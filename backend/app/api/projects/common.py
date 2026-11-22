"""The projects blueprint, and the helpers its domains share.

The blueprint object lives here rather than in ``__init__`` so the route
modules can import it without importing the package that imports them.
"""

import logging
import os
import re
import subprocess
from datetime import UTC, datetime
from flask import Blueprint, current_app
from ...extensions import db
from ...models.project import Project
from ...models.recording import Recording
from ...models.setting import Setting
from ...services.annotations import (
    annotation_recording_ref,
    annotations_filename,
)
from ...services.file_utils import atomic_write_text, safe_child_path, safe_project_dir

log = logging.getLogger(__name__)

projects_bp = Blueprint('projects', __name__)

ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'm4a', 'wav', 'mkv', 'webm', 'ogg', 'flac'}

def _write_project_readme(project):
    """Write a README.md with project metadata into the project folder."""
    root = _projects_root()
    folder = os.path.join(root, project.folder_name)
    if not os.path.isdir(folder):
        return

    lines = [f'# {project.name}', '']

    if project.description:
        lines += [project.description, '']

    if project.objective:
        lines += ['## Objective', '', project.objective, '']

    if project.methodology:
        lines += ['## Methodology', '', project.methodology, '']

    if project.interview_guide:
        lines += ['## Interview Guide', '', project.interview_guide, '']

    questions = project.get_questions()
    if questions:
        lines += ['## Research Questions', '']
        for i, q in enumerate(questions, 1):
            lines.append(f'{i}. {q}')
        lines.append('')

    hypotheses = project.get_hypotheses()
    if hypotheses:
        lines += ['## Hypotheses', '']
        for i, h in enumerate(hypotheses, 1):
            lines.append(f'{i}. {h}')
        lines.append('')

    stakeholders = project.get_stakeholders()
    if stakeholders:
        lines += ['## Stakeholders', '']
        for i, s in enumerate(stakeholders, 1):
            lines.append(f'{i}. {s}')
        lines.append('')

    summary_sections = project.get_enabled_summary_sections()
    summary_content = []
    for key in summary_sections:
        if key == 'key_findings' and project.key_findings:
            summary_content.append(('Key Findings', project.key_findings))
        elif key == 'further_steps' and project.further_steps:
            summary_content.append(('Further Steps', project.further_steps))
        elif key == 'recommendations' and project.recommendations:
            summary_content.append(('Recommendations', project.recommendations))
    if summary_content:
        lines += ['## Summary', '']
        for title, content in summary_content:
            lines += [f'### {title}', '', content, '']

    recs = project.recordings
    if recs:
        lines += ['## Recordings', '']
        for r in recs:
            status = r.transcription_status or 'pending'
            lines.append(f'- **{r.original_name}** ({status})')
        lines.append('')

    lines += [
        '---',
        f'*Created: {project.created_at.strftime("%Y-%m-%d") if project.created_at else "—"}*  ',
        f'*Updated: {project.updated_at.strftime("%Y-%m-%d") if project.updated_at else "—"}*',
    ]

    readme_path = os.path.join(folder, 'README.md')
    atomic_write_text(readme_path, '\n'.join(lines) + '\n')

def _projects_root():
    return Setting.get('projects_path', current_app.config['DEFAULT_PROJECTS_PATH'])

def _project_dir_or_none(project):
    """The project's folder, or None if its folder_name leads out of the root.

    A row like that can only come from a restore made before archive paths
    were checked. Deleting, moving or serving through it would act on
    whatever the archive named, so the callers skip the file work instead.
    """
    try:
        return safe_project_dir(_projects_root(), project.folder_name)
    except ValueError:
        log.warning('Project %s has an unsafe folder_name %r; skipping its files',
                    project.id, project.folder_name)
        return None

def _project_file_or_none(project_dir, rel_path):
    """*rel_path* inside *project_dir*, or None if it would leave it."""
    if not project_dir:
        return None
    try:
        return safe_child_path(project_dir, rel_path)
    except ValueError:
        log.warning('Refusing a path outside %s: %r', project_dir, rel_path)
        return None

def _recording_filepath(project, recording):
    """Return the absolute path to the media file for a recording.

    Linked recordings store the absolute path directly in stored_name.
    Copied recordings store just the filename relative to the project folder;
    one whose stored_name points outside that folder returns None. So does a
    link to anything but media, which linking itself never creates.
    """
    if recording.is_linked:
        if _ext_of(recording.stored_name) not in ALLOWED_EXTENSIONS:
            log.warning('Recording %s links to a non-media path %r; not serving it',
                        recording.id, recording.stored_name)
            return None
        return recording.stored_name
    return _project_file_or_none(_project_dir_or_none(project), recording.stored_name)

def _delete_recording_annotation_files(project_dir, recording):
    """Delete both current and legacy annotation files for a recording."""
    refs = {
        annotation_recording_ref(recording),
        (recording.stored_name or '').strip(),
    }
    for ref in refs:
        if not ref:
            continue
        ann_path = _project_file_or_none(project_dir, annotations_filename(ref))
        if ann_path and os.path.isfile(ann_path):
            os.remove(ann_path)

def _safe_folder_name(name, existing_names=None):
    """Turn a project name into a filesystem-safe folder name, avoiding collisions."""
    slug = re.sub(r'[^\w\s\-]', '', name).strip()
    slug = re.sub(r'\s+', '_', slug) or 'project'
    slug = slug[:80]

    if existing_names is None:
        existing_names = set()

    candidate = slug
    counter = 1
    while candidate.lower() in {n.lower() for n in existing_names}:
        counter += 1
        candidate = f'{slug}_{counter}'
    return candidate

def _mp4_faststart(filepath):
    """Move MP4 moov atom to front for instant browser playback."""
    tmp = filepath + '.faststart.mp4'
    try:
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', filepath, '-c', 'copy',
             '-movflags', '+faststart', tmp],
            capture_output=True, timeout=120,
        )
        if result.returncode == 0 and os.path.isfile(tmp):
            os.replace(tmp, filepath)
    except Exception:
        pass
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

def _ext_of(filepath):
    """Lowercase extension without the dot, or '' when there is none."""
    name = os.path.basename(filepath or '')
    return name.rsplit('.', 1)[-1].lower() if '.' in name else ''

def _probe_duration(filepath):
    """Use ffprobe to get duration in seconds. Returns 0 on failure."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', filepath],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0

SYSTEM_PROJECT_NAME = '__single_transcriptions'

def _get_or_create_system_project():
    """Return the hidden system project, creating it on first use."""
    proj = Project.query.filter_by(is_system=True).first()
    if proj:
        return proj
    root = _projects_root()
    proj = Project(
        name=SYSTEM_PROJECT_NAME,
        folder_name=SYSTEM_PROJECT_NAME,
        is_system=True,
    )
    db.session.add(proj)
    db.session.flush()
    os.makedirs(os.path.join(root, SYSTEM_PROJECT_NAME), exist_ok=True)
    db.session.commit()
    return proj

def _segment_assigned_count(segment_id):
    return Recording.query.filter_by(segment_id=segment_id).count()

def _touch_project(project):
    if project is None:
        return
    project.updated_at = datetime.now(UTC)
    db.session.add(project)

def _parse_bool(val, default):
    """Parse bool from JSON or query string (true/false/1/0)."""
    if val is None or val == '':
        return default
    if isinstance(val, bool):
        return val
    return str(val).lower() in ('true', '1', 'yes')
