"""Exporting a recording or a whole project, and the transfer package."""

import json
import os
import re
from flask import Response, current_app, jsonify, request, send_file
from ...extensions import db
from ...models.project import Project
from ...models.recording import Recording
from ...services.annotations import (
    annotation_recording_ref,
    annotations_filename,
    get_project_tags,
    get_project_themes,
)
from .common import (
    _parse_bool,
    _project_dir_or_none,
    _project_file_or_none,
    projects_bp,
)

def _safe_ascii(name):
    """ASCII-only fallback name for Content-Disposition filename= param."""
    if not name:
        return 'export'
    safe = ''.join(c if ord(c) < 128 else '_' for c in str(name))
    return re.sub(r'[^\w\s\-.]', '', safe).strip() or 'export'

def _content_disposition(filename):
    """Build a Content-Disposition header value with RFC 5987 Unicode support.
    Modern browsers use filename*=UTF-8''<encoded>; old clients fall back to filename=."""
    from urllib.parse import quote
    ascii_name = _safe_ascii(filename)
    encoded = quote(str(filename).encode('utf-8'), safe=' -()')
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{encoded}"

def _build_recording_export_filename(project_name, recording_name, ext):
    """Build export filename as <project>-<recording>.<ext>."""
    project_part = (project_name or '').strip()
    recording_part = os.path.splitext(recording_name or '')[0].strip() or 'recording'
    if project_part:
        return f'{project_part}-{recording_part}{ext}'
    return f'{recording_part}{ext}'

def _unique_zip_name(name, used):
    """Keep names unique inside the archive: a.md, a (2).md, a (3).md."""
    base, ext = os.path.splitext(name)
    candidate, n = name, 1
    while candidate.lower() in used:
        n += 1
        candidate = f'{base} ({n}){ext}'
    used.add(candidate.lower())
    return candidate

def _zip_recording_exports(project, recordings, fmt, opts):
    """Bundle one export file per recording into a ZIP, returned as bytes."""
    import io
    import zipfile

    from ...services.export_service import export_recording_markdown, export_recording_odt

    ext = '.odt' if fmt == 'odt' else '.md'
    buf = io.BytesIO()
    used = set()
    written = 0
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for rec in recordings:
            if fmt == 'odt':
                content, err = export_recording_odt(current_app, project.id, rec.id, opts)
            else:
                content, err = export_recording_markdown(current_app, project.id, rec.id, opts)
            if err or content is None:
                continue
            name = _unique_zip_name(
                _build_recording_export_filename(project.name, rec.original_name, ext), used
            )
            zf.writestr(name, content)
            written += 1
    if not written:
        return None, 'No transcripts to export'
    return buf.getvalue(), None

@projects_bp.route('/<int:project_id>/recordings/<int:recording_id>/export', methods=['GET', 'POST'])
def export_recording(project_id, recording_id):
    recording = db.session.get(Recording, recording_id)
    if not recording or recording.project_id != project_id:
        return jsonify({'error': 'Recording not found'}), 404

    if request.method == 'POST':
        data = request.get_json(force=True, silent=True) or {}
    else:
        data = request.args
    fmt = (data.get('format') or 'markdown').strip().lower()
    opts = {
        'include_comments': _parse_bool(data.get('include_comments'), True),
        'include_tags': _parse_bool(data.get('include_tags'), True),
        'include_participant_details': _parse_bool(data.get('include_participant_details'), True),
        'include_project_details': _parse_bool(data.get('include_project_details'), True),
        'include_prompt': _parse_bool(data.get('include_prompt'), True),
        'remove_pii': _parse_bool(data.get('remove_pii'), False),
        'use_recording_screen_prompt': _parse_bool(data.get('use_recording_screen_prompt'), False),
    }

    from ...services.export_service import export_recording_markdown, export_recording_odt
    from ...services.pii_service import PIIError

    try:
        if fmt == 'odt':
            content, err = export_recording_odt(current_app, project_id, recording_id, opts)
        else:
            content, err = export_recording_markdown(current_app, project_id, recording_id, opts)
    except PIIError as exc:
        return jsonify({'error': str(exc)}), 422

    if err:
        return jsonify({'error': err}), 404

    project_name = (recording.project.name if recording.project else '').strip()
    if fmt == 'odt':
        name = _build_recording_export_filename(project_name, recording.original_name, '.odt')
        return Response(
            content,
            mimetype='application/vnd.oasis.opendocument.text',
            headers={'Content-Disposition': _content_disposition(name)},
        )

    name = _build_recording_export_filename(project_name, recording.original_name, '.md')
    return Response(
        content,
        mimetype='text/markdown; charset=utf-8',
        headers={'Content-Disposition': _content_disposition(name)},
    )

@projects_bp.route('/<int:project_id>/export', methods=['GET', 'POST'])
def export_project(project_id):
    """Export selected recordings as a single Markdown or ODT document."""
    project = db.session.get(Project, project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    if request.method == 'POST':
        data = request.get_json(force=True, silent=True) or {}
    else:
        data = request.args
    recording_ids_raw = data.get('recording_ids', [])
    if isinstance(recording_ids_raw, str):
        recording_ids = [int(x.strip()) for x in recording_ids_raw.split(',') if x.strip()]
    else:
        recording_ids = recording_ids_raw
    fmt = (data.get('format') or 'markdown').strip().lower()
    opts = {
        'include_comments': _parse_bool(data.get('include_comments'), True),
        'include_tags': _parse_bool(data.get('include_tags'), True),
        'include_participant_details': _parse_bool(data.get('include_participant_details'), True),
        'include_project_details': _parse_bool(data.get('include_project_details'), True),
        'include_prompt': _parse_bool(data.get('include_prompt'), True),
        'remove_pii': _parse_bool(data.get('remove_pii'), False),
    }

    separate_files = _parse_bool(data.get('separate_files'), False)

    if not recording_ids:
        return jsonify({'error': 'No recordings selected'}), 400

    from ...services.export_service import export_project_markdown, export_project_odt
    from ...services.pii_service import PIIError

    if separate_files:
        recordings = []
        for rid in recording_ids:
            try:
                rec = db.session.get(Recording, int(rid))
            except (TypeError, ValueError):
                rec = None
            if rec and rec.project_id == project_id:
                recordings.append(rec)
        try:
            content, err = _zip_recording_exports(project, recordings, fmt, opts)
        except PIIError as exc:
            return jsonify({'error': str(exc)}), 422
        if err:
            return jsonify({'error': err}), 400
        zip_name = project.name.replace(' ', '_')[:50] + '_export.zip'
        return Response(
            content,
            mimetype='application/zip',
            headers={'Content-Disposition': _content_disposition(zip_name)},
        )

    try:
        if fmt == 'odt':
            content, err = export_project_odt(current_app, project_id, recording_ids, opts)
        else:
            content, err = export_project_markdown(current_app, project_id, recording_ids, opts)
    except PIIError as exc:
        return jsonify({'error': str(exc)}), 422

    if err:
        return jsonify({'error': err}), 400

    ext = 'odt' if fmt == 'odt' else 'md'
    export_name = project.name.replace(' ', '_')[:50] + f'_export.{ext}'
    mimetype = 'application/vnd.oasis.opendocument.text' if fmt == 'odt' else 'text/markdown; charset=utf-8'

    return Response(
        content,
        mimetype=mimetype,
        headers={'Content-Disposition': _content_disposition(export_name)},
    )

@projects_bp.route('/<int:project_id>/transfer', methods=['GET'])
def transfer_project(project_id):
    """Package a project as a ZIP file (transcripts + annotations + metadata, no audio)."""
    import io
    import zipfile

    project = db.session.get(Project, project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    project_dir = _project_dir_or_none(project)
    if not project_dir:
        return jsonify({'error': 'Project folder is outside the projects directory'}), 409

    # Build project metadata
    meta = project.to_dict(include_recordings=True)
    tags = get_project_tags(project_dir)
    themes = get_project_themes(project_dir)

    # In memory, not a temp file: there is no audio in here, and the temp file
    # this used to be was never removed — a copy of every transcript per export.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Project metadata
        zf.writestr('project.json', json.dumps(meta, ensure_ascii=False, indent=2))
        zf.writestr('project_tags.json', json.dumps(tags, ensure_ascii=False, indent=2))
        zf.writestr('project_themes.json', json.dumps(themes, ensure_ascii=False, indent=2))

        # Each recording's transcript + annotations
        for rec in project.recordings:
            # Transcript JSON
            if rec.transcript_path:
                tx_file = _project_file_or_none(project_dir, rec.transcript_path)
                if tx_file and os.path.isfile(tx_file):
                    zf.write(tx_file, f'transcripts/{rec.transcript_path}')

            # Annotations JSON
            ann_name = annotations_filename(annotation_recording_ref(rec))
            ann_file = os.path.join(project_dir, ann_name)
            if os.path.isfile(ann_file):
                zf.write(ann_file, f'annotations/{ann_name}')

        # README
        readme_path = os.path.join(project_dir, 'README.md')
        if os.path.isfile(readme_path):
            zf.write(readme_path, 'README.md')

    safe_name = _safe_ascii(project.name).replace(' ', '_')[:50]
    buf.seek(0)
    return send_file(
        buf,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'{safe_name}_transfer.zip',
    )
