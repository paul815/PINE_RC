"""Projects themselves: create, read, update, archive, delete."""

import os
import re
from datetime import UTC, datetime
from flask import current_app, jsonify, request
from ...extensions import db
from ...models.project import Project
from ...models.segment import Segment
from ...models.setting import Setting
from ...services.annotations import (
    get_project_tags,
)
from .common import (
    _project_dir_or_none,
    _projects_root,
    _safe_folder_name,
    _write_project_readme,
    projects_bp,
)

@projects_bp.route('', methods=['GET'])
def list_projects():
    projects = Project.query.filter(
        (Project.is_system.is_(False)) | (Project.is_system.is_(None))
    ).order_by(Project.updated_at.desc()).all()
    raw_inc = request.args.get('include_tags') or request.args.get('includeTags') or ''
    include_tags = str(raw_inc).lower() in ('1', 'true', 'yes')
    root = _projects_root() if include_tags else None

    def to_entry(p):
        d = p.to_dict()
        if include_tags:
            project_dir = os.path.normpath(os.path.join(root, p.folder_name))
            d['tags'] = get_project_tags(project_dir)
        return d

    active = [to_entry(p) for p in projects if not p.is_archived]
    archived = [to_entry(p) for p in projects if p.is_archived]
    return jsonify({'active': active, 'archived': archived})

@projects_bp.route('', methods=['POST'])
def create_project():
    data = request.get_json(force=True) if request.is_json else {}
    name = (data.get('name') or 'Untitled project').strip()

    root = _projects_root()
    existing = {p.folder_name for p in Project.query.all()}
    folder = _safe_folder_name(name, existing)

    project = Project(name=name, folder_name=folder)
    db.session.add(project)
    db.session.flush()

    os.makedirs(os.path.join(root, folder), exist_ok=True)

    db.session.commit()
    _write_project_readme(project)
    return jsonify(project.to_dict(include_recordings=True)), 201

@projects_bp.route('/<int:project_id>', methods=['GET'])
def get_project(project_id):
    project = db.session.get(Project, project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404
    data = project.to_dict(include_recordings=True)
    # Add segment names to recordings
    if data.get('recordings'):
        seg_ids = {r.segment_id for r in project.recordings if r.segment_id}
        segments = Segment.query.filter(Segment.id.in_(seg_ids)).all() if seg_ids else []
        seg_map = {s.id: s.name for s in segments}
        for rec in data['recordings']:
            rec['segment_name'] = seg_map.get(rec['segment_id']) if rec.get('segment_id') else None
    return jsonify(data)

@projects_bp.route('/<int:project_id>', methods=['PATCH'])
def update_project(project_id):
    project = db.session.get(Project, project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    data = request.get_json(force=True)

    if 'name' in data:
        new_name = (data['name'] or '').strip() or 'Untitled project'
        if new_name != project.name:
            project.name = new_name
            root = _projects_root()
            old_folder = _project_dir_or_none(project)
            existing = {p.folder_name for p in Project.query.filter(Project.id != project.id).all()}
            new_folder = _safe_folder_name(new_name, existing)
            new_path = os.path.join(root, new_folder)
            if old_folder and os.path.isdir(old_folder) and old_folder != new_path:
                os.rename(old_folder, new_path)
            elif not os.path.isdir(new_path):
                os.makedirs(new_path, exist_ok=True)
            project.folder_name = new_folder

    for field in ('description', 'objective', 'summary', 'icon', 'results_recommendations', 'further_steps',
                  'methodology', 'interview_guide', 'key_findings', 'recommendations'):
        if field in data:
            setattr(project, field, (data[field] or '').strip())

    if 'enabled_summary_sections' in data:
        project.set_enabled_summary_sections(data['enabled_summary_sections'])

    if 'research_questions' in data:
        project.set_questions(data['research_questions'])

    if 'hypotheses' in data:
        project.set_hypotheses(data['hypotheses'])

    if 'stakeholders' in data:
        project.set_stakeholders(data['stakeholders'])

    if 'enabled_sections' in data:
        project.set_enabled_sections(data['enabled_sections'])

    if 'custom_sections' in data:
        project.set_custom_sections(data['custom_sections'])

    if 'custom_summary_sections' in data:
        project.set_custom_summary_sections(data['custom_summary_sections'])

    if 'default_transcription_language' in data:
        raw = (data['default_transcription_language'] or '').strip().lower().replace('-', '_')
        if raw == '':
            project.default_transcription_language = ''
        elif re.match(r'^[a-z][a-z0-9_]{0,15}$', raw):
            project.default_transcription_language = raw
        else:
            return jsonify({'error': 'Invalid default_transcription_language'}), 400

    db.session.commit()
    _write_project_readme(project)
    return jsonify(project.to_dict())

@projects_bp.route('/<int:project_id>/archive', methods=['POST'])
def archive_project(project_id):
    project = db.session.get(Project, project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404
    project.is_archived = True
    project.archived_at = datetime.now(UTC)
    db.session.commit()
    return jsonify(project.to_dict())

@projects_bp.route('/<int:project_id>/unarchive', methods=['POST'])
def unarchive_project(project_id):
    project = db.session.get(Project, project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404
    project.is_archived = False
    project.archived_at = None
    db.session.commit()
    return jsonify(project.to_dict())

@projects_bp.route('/<int:project_id>', methods=['DELETE'])
def delete_project(project_id):
    project = db.session.get(Project, project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404
    if project.is_system:
        return jsonify({'error': 'Cannot delete system project'}), 403

    from ...services.backup_service import create_safety_snapshot

    safety_backup = create_safety_snapshot(
        current_app._get_current_object(),
        reason=f'delete_project_{project.folder_name}',
        include_audio=True,
        project_folders=[project.folder_name],
    )

    import shutil
    folder_path = _project_dir_or_none(project)
    if folder_path and os.path.isdir(folder_path):
        shutil.rmtree(folder_path, ignore_errors=True)

    lp = db.session.get(Setting, 'last_open_project_id')
    if lp and lp.value == str(project_id):
        lp.value = ''
    db.session.delete(project)
    db.session.commit()
    return jsonify({'ok': True, 'safety_backup': safety_backup})
