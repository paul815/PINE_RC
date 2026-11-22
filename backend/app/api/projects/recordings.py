"""Recordings: upload, link, multi-track ingest, playback, transcript.

Also the standalone "single transcriptions" that live in the hidden
system project rather than a research project of their own.
"""

import json
import os
import re
from flask import jsonify, request
from werkzeug.utils import secure_filename
from ...extensions import db
from ...models.project import Project
from ...models.recording import Recording
from ...models.segment import Segment
from ...services.annotations import (
    annotation_recording_ref,
    get_annotations,
    get_project_tags,
    get_project_themes,
    mark_anchor_drift,
    save_annotations,
)
from ...services.speaker_blocks import with_speaker_blocks
from ...services.file_utils import (
    atomic_read_json,
    atomic_write_text,
    claim_free_path,
    save_upload_atomically,
)
from .common import (
    ALLOWED_EXTENSIONS,
    _delete_recording_annotation_files,
    _ext_of,
    _get_or_create_system_project,
    _mp4_faststart,
    _parse_bool,
    _probe_duration,
    _project_dir_or_none,
    _project_file_or_none,
    _projects_root,
    _recording_filepath,
    _segment_assigned_count,
    _touch_project,
    _write_project_readme,
    projects_bp,
)

@projects_bp.route('/single-transcriptions', methods=['GET'])
def list_single_transcriptions():
    proj = Project.query.filter_by(is_system=True).first()
    if not proj:
        return jsonify({'recordings': [], 'project_id': None})
    recs = Recording.query.filter_by(project_id=proj.id)\
        .order_by(Recording.created_at.desc()).all()
    return jsonify({
        'recordings': [r.to_dict() for r in recs],
        'project_id': proj.id,
    })

@projects_bp.route('/single-transcriptions', methods=['POST'])
def upload_single_transcription():
    proj = _get_or_create_system_project()

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400

    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'error': f'Unsupported format: .{ext}'}), 400

    original_name = file.filename
    safe_name = secure_filename(original_name)
    if not safe_name:
        safe_name = f'recording.{ext}'

    project_dir = os.path.join(_projects_root(), proj.folder_name)
    os.makedirs(project_dir, exist_ok=True)

    dest, safe_name = claim_free_path(project_dir, safe_name)
    save_upload_atomically(file, dest)

    file_size = os.path.getsize(dest)
    duration = _probe_duration(dest)

    recording = Recording(
        project_id=proj.id,
        original_name=original_name,
        stored_name=safe_name,
        file_format=ext,
        file_size_bytes=file_size,
        duration_seconds=duration,
        transcription_status='pending',
    )
    db.session.add(recording)
    db.session.commit()

    from ...services.transcription import enqueue
    enqueue(recording.id)

    return jsonify(recording.to_dict()), 201

@projects_bp.route('/single-transcriptions/<int:recording_id>', methods=['DELETE'])
def delete_single_transcription(recording_id):
    proj = Project.query.filter_by(is_system=True).first()
    if not proj:
        return jsonify({'error': 'Not found'}), 404
    recording = db.session.get(Recording, recording_id)
    if not recording or recording.project_id != proj.id:
        return jsonify({'error': 'Recording not found'}), 404

    project_dir = _project_dir_or_none(proj)
    filepath = _recording_filepath(proj, recording)
    if not recording.is_linked and filepath and os.path.isfile(filepath):
        os.remove(filepath)
    if recording.transcript_path:
        tf = _project_file_or_none(project_dir, recording.transcript_path)
        if tf and os.path.isfile(tf):
            os.remove(tf)
    _delete_recording_annotation_files(project_dir, recording)

    db.session.delete(recording)
    db.session.commit()
    return jsonify({'ok': True})

def _parse_num_speakers(raw):
    """Clamp a user-supplied speaker count to 1..10, or None when unset/invalid.

    None means "no explicit count" — diarization falls back to its default
    speaker-range heuristic.
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    if n < 1:
        return None
    return min(n, 10)

@projects_bp.route('/<int:project_id>/recordings', methods=['POST'])
def upload_recording(project_id):
    project = db.session.get(Project, project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400

    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'error': f'Unsupported format: .{ext}'}), 400

    original_name = file.filename
    safe_name = secure_filename(original_name)
    if not safe_name:
        safe_name = f'recording.{ext}'

    project_dir = os.path.join(_projects_root(), project.folder_name)
    os.makedirs(project_dir, exist_ok=True)

    # Claims the name atomically — see claim_free_path on why exists() is not enough
    dest, safe_name = claim_free_path(project_dir, safe_name)
    save_upload_atomically(file, dest)

    # Move MP4 moov atom to front for instant browser playback
    if ext == 'mp4':
        _mp4_faststart(dest)

    file_size = os.path.getsize(dest)
    duration = _probe_duration(dest)

    recording = Recording(
        project_id=project.id,
        original_name=original_name,
        stored_name=safe_name,
        file_format=ext,
        file_size_bytes=file_size,
        duration_seconds=duration,
        transcription_status='pending',
        num_speakers=_parse_num_speakers(request.form.get('num_speakers')),
    )
    db.session.add(recording)
    _touch_project(project)
    db.session.commit()

    from ...services.transcription import enqueue
    enqueue(recording.id)

    _write_project_readme(project)
    return jsonify(recording.to_dict()), 201

@projects_bp.route('/<int:project_id>/recordings/link', methods=['POST'])
def link_recording(project_id):
    """Register an existing file as a recording without copying it.

    The file stays at its original path on disk. PINE stores that absolute
    path in stored_name and sets is_linked=True so the path is used directly
    when serving media or starting transcription.
    """
    project = db.session.get(Project, project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    data = request.get_json(force=True) or {}
    file_path = (data.get('path') or '').strip()
    if not file_path:
        return jsonify({'error': 'path is required'}), 400

    if not os.path.isfile(file_path):
        return jsonify({'error': 'File not found at the given path'}), 400

    ext = file_path.rsplit('.', 1)[-1].lower() if '.' in file_path else ''
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'error': f'Unsupported format: .{ext}'}), 400

    original_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    duration = _probe_duration(file_path)

    os.makedirs(os.path.join(_projects_root(), project.folder_name), exist_ok=True)

    recording = Recording(
        project_id=project.id,
        original_name=original_name,
        stored_name=file_path,          # absolute path IS the stored reference
        file_format=ext,
        file_size_bytes=file_size,
        duration_seconds=duration,
        transcription_status='pending',
        is_linked=True,
        num_speakers=_parse_num_speakers(data.get('num_speakers')),
    )
    db.session.add(recording)
    _touch_project(project)
    db.session.commit()

    from ...services.transcription import enqueue
    enqueue(recording.id)

    _write_project_readme(project)
    return jsonify(recording.to_dict()), 201

@projects_bp.route('/<int:project_id>/recordings/multitrack', methods=['POST'])
def create_multitrack_recording(project_id):
    """Add a recording whose speakers already have their own tracks.

    Nothing is copied: like the link flow, the material stays where it is and
    PINE stores absolute paths. The recording itself points at whatever should
    play — the Zoom video, the mixed audio Zoom also writes, or a mixdown built
    here when the folder holds neither.

    Body: { folder: str }  — a Zoom meeting folder, or
          { path: str }    — one file whose channels are the speakers.
    """
    from ...models.recording_track import RecordingTrack
    from ...services.multitrack_ingest import (
        build_mixdown,
        describe_multichannel,
        detect_zoom_folder,
        meeting_name_from_folder,
    )

    project = db.session.get(Project, project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    data = request.get_json(force=True, silent=True) or {}
    folder = (data.get('folder') or '').strip()
    path = (data.get('path') or '').strip()

    project_dir = os.path.join(_projects_root(), project.folder_name)
    os.makedirs(project_dir, exist_ok=True)

    if folder:
        found = detect_zoom_folder(folder)
        if not found:
            return jsonify({'error': 'No per-participant tracks in this folder'}), 400
        original_name = meeting_name_from_folder(folder)
        media = found['media']
        if not media:
            media = build_mixdown(
                [t['path'] for t in found['tracks']],
                os.path.join(project_dir, f'{secure_filename(original_name)}_mixdown.m4a'))
            if not media:
                return jsonify({'error': 'Could not build a playable mixdown'}), 500
    elif path:
        found = describe_multichannel(path)
        if not found:
            return jsonify({'error': 'This file has only one audio channel'}), 400
        original_name = os.path.basename(path)
        media = found['media']
    else:
        return jsonify({'error': 'folder or path is required'}), 400

    ext = _ext_of(media)
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'error': f'Unsupported format: .{ext}'}), 400

    recording = Recording(
        project_id=project.id,
        original_name=original_name,
        stored_name=media,              # absolute path, as for linked recordings
        file_format=ext,
        file_size_bytes=os.path.getsize(media) if os.path.isfile(media) else 0,
        duration_seconds=_probe_duration(media),
        transcription_status='pending',
        is_linked=True,
        source_kind='multitrack',
        # The tracks are the speakers — there is nothing left to estimate.
        num_speakers=len(found['tracks']),
    )
    db.session.add(recording)
    db.session.flush()

    for i, track in enumerate(found['tracks']):
        db.session.add(RecordingTrack(
            recording_id=recording.id,
            track_index=i,
            source_path=track['path'],
            speaker_name=track.get('speaker_name') or '',
            channel_index=track.get('channel'),
            duration_seconds=_probe_duration(track['path']),
        ))

    _touch_project(project)
    db.session.commit()

    from ...services.transcription import enqueue
    enqueue(recording.id)

    _write_project_readme(project)
    return jsonify(recording.to_dict()), 201

@projects_bp.route('/<int:project_id>/recordings/<int:recording_id>', methods=['GET'])
def get_recording(project_id, recording_id):
    recording = db.session.get(Recording, recording_id)
    if not recording or recording.project_id != project_id:
        return jsonify({'error': 'Recording not found'}), 404

    project = db.session.get(Project, project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    data = {
        'recording': recording.to_dict(),
        'project': project.to_dict(),
    }

    project_dir = os.path.join(_projects_root(), project.folder_name)

    if recording.transcription_status == 'transcribed' and recording.transcript_path:
        transcript_file = _project_file_or_none(_project_dir_or_none(project), recording.transcript_path)
        if transcript_file and os.path.isfile(transcript_file):
            data['transcript'] = with_speaker_blocks(atomic_read_json(transcript_file))

    data['tags'] = get_project_tags(project_dir)
    data['themes'] = get_project_themes(project_dir)
    ann = get_annotations(project_dir, annotation_recording_ref(recording))
    # Every anchor is checked against the words it was taken from on the way
    # out: a highlight whose offsets no longer cover its quote is reported
    # rather than drawn as though nothing had happened.
    data['annotations'] = mark_anchor_drift(
        ann, (data.get('transcript') or {}).get('segments') or [])

    # Include segment info if assigned
    if recording.segment_id:
        seg = db.session.get(Segment, recording.segment_id)
        if seg:
            data['recording']['segment'] = seg.to_dict(
                assigned_count=_segment_assigned_count(seg.id)
            )

    return jsonify(data)

@projects_bp.route('/<int:project_id>/recordings/<int:recording_id>', methods=['PATCH'])
def update_recording(project_id, recording_id):
    recording = db.session.get(Recording, recording_id)
    if not recording or recording.project_id != project_id:
        return jsonify({'error': 'Recording not found'}), 404

    data = request.get_json(force=True, silent=True) or {}

    if 'segment_id' in data:
        sid = data['segment_id']
        if sid is None:
            recording.segment_id = None
        else:
            segment = db.session.get(Segment, int(sid))
            if not segment or segment.project_id != project_id:
                return jsonify({'error': 'Segment not found'}), 404
            recording.segment_id = segment.id

    if 'participant_notes' in data:
        recording.participant_notes = (data['participant_notes'] or '').strip()

    if 'num_speakers' in data:
        recording.num_speakers = _parse_num_speakers(data.get('num_speakers'))

    if 'original_name' in data:
        new_name = (data['original_name'] or '').strip()
        if new_name:
            # Preserve extension if the new name doesn't have one
            ext = ''
            if recording.original_name and '.' in recording.original_name:
                ext = '.' + recording.original_name.rsplit('.', 1)[-1]
            if ext and not new_name.lower().endswith(ext.lower()):
                new_name = new_name + ext
            recording.original_name = new_name[:500]

    _touch_project(db.session.get(Project, project_id))
    db.session.commit()

    result = recording.to_dict()
    if recording.segment_id:
        seg = db.session.get(Segment, recording.segment_id)
        if seg:
            result['segment'] = seg.to_dict(assigned_count=_segment_assigned_count(seg.id))
    return jsonify(result)

@projects_bp.route('/<int:project_id>/recordings/<int:recording_id>/media', methods=['GET'])
def stream_media(project_id, recording_id):
    recording = db.session.get(Recording, recording_id)
    if not recording or recording.project_id != project_id:
        return jsonify({'error': 'Recording not found'}), 404

    project = db.session.get(Project, project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    filepath = _recording_filepath(project, recording)
    if not filepath or not os.path.isfile(filepath):
        return jsonify({'error': 'Media file not found'}), 404

    from flask import send_file as flask_send_file
    return flask_send_file(filepath, conditional=True)

@projects_bp.route('/<int:project_id>/recordings/<int:recording_id>/transcript', methods=['GET'])
def get_transcript(project_id, recording_id):
    recording = db.session.get(Recording, recording_id)
    if not recording or recording.project_id != project_id:
        return jsonify({'error': 'Recording not found'}), 404

    if recording.transcription_status != 'transcribed' or not recording.transcript_path:
        return jsonify({'error': 'Transcript not available', 'status': recording.transcription_status}), 404

    project = db.session.get(Project, project_id)
    transcript_file = _project_file_or_none(_project_dir_or_none(project), recording.transcript_path)
    if not transcript_file or not os.path.isfile(transcript_file):
        return jsonify({'error': 'Transcript file missing'}), 404

    return jsonify(with_speaker_blocks(atomic_read_json(transcript_file)))

@projects_bp.route('/<int:project_id>/recordings/<int:recording_id>/transcript/replace', methods=['POST'])
def replace_transcript_text(project_id, recording_id):
    """Find & replace across a recording's transcript, migrating tag/comment offsets.

    Used to mass-fix recurring ASR errors (names, jargon). Existing annotations
    stay anchored: their char offsets are shifted to follow the edited text.
    """
    recording = db.session.get(Recording, recording_id)
    if not recording or recording.project_id != project_id:
        return jsonify({'error': 'Recording not found'}), 404
    if recording.transcription_status != 'transcribed' or not recording.transcript_path:
        return jsonify({'error': 'Transcript not available'}), 404

    body = request.get_json(force=True, silent=True) or {}
    find = str(body.get('find') or '')
    replace = str(body.get('replace') or '')
    if not find:
        return jsonify({'error': 'Nothing to find'}), 400
    match_case = _parse_bool(body.get('match_case'), False)
    whole_word = _parse_bool(body.get('whole_word'), False)

    project = db.session.get(Project, project_id)
    transcript_file = _project_file_or_none(_project_dir_or_none(project), recording.transcript_path)
    if not transcript_file or not os.path.isfile(transcript_file):
        return jsonify({'error': 'Transcript file missing'}), 404

    transcript = atomic_read_json(transcript_file)

    project_dir = os.path.join(_projects_root(), project.folder_name)
    ann_ref = annotation_recording_ref(recording)
    annotations = get_annotations(project_dir, ann_ref)

    from ...services.transcript_edit import apply_find_replace
    count = apply_find_replace(
        transcript.get('segments') or [], annotations,
        find, replace, match_case=match_case, whole_word=whole_word,
    )

    if count:
        atomic_write_text(transcript_file, json.dumps(transcript, ensure_ascii=False, indent=2))
        save_annotations(project_dir, ann_ref, annotations)
        _touch_project(project)
        db.session.commit()

    return jsonify({'count': count, 'transcript': with_speaker_blocks(transcript),
                    'annotations': annotations})

# Said when the transcript on disk no longer matches what the editor started
# from -- another tab, or another window of the same one, got there first.
_BLOCK_EDIT_STALE_MESSAGE = 'This transcript changed elsewhere. Reload the page to continue editing.'


@projects_bp.route('/<int:project_id>/recordings/<int:recording_id>/transcript/block', methods=['POST'])
def edit_transcript_block(project_id, recording_id):
    """Rewrite one speaker block's text, the way it reads on the recording screen.

    The caller sends the block's own ``indices`` (not a range -- an interrupted
    speaker resumes into the block they opened), the text it had when editing
    started, and the text it has now. Sending the original back is what lets a
    second tab's edit be refused instead of silently overwritten.
    """
    recording = db.session.get(Recording, recording_id)
    if not recording or recording.project_id != project_id:
        return jsonify({'error': 'Recording not found'}), 404
    if recording.transcription_status != 'transcribed' or not recording.transcript_path:
        return jsonify({'error': 'Transcript not available'}), 404

    body = request.get_json(force=True, silent=True) or {}
    raw_indices = body.get('indices')
    if not isinstance(raw_indices, list) or not raw_indices or \
            not all(isinstance(i, int) and not isinstance(i, bool) for i in raw_indices):
        return jsonify({'error': 'Nothing to save'}), 400
    original_text = str(body.get('original_text') or '')
    new_text = str(body.get('new_text') or '')

    project = db.session.get(Project, project_id)
    transcript_file = _project_file_or_none(_project_dir_or_none(project), recording.transcript_path)
    if not transcript_file or not os.path.isfile(transcript_file):
        return jsonify({'error': 'Transcript file missing'}), 404

    transcript = atomic_read_json(transcript_file)
    segments = transcript.get('segments') or []
    if any(i < 0 or i >= len(segments) for i in raw_indices):
        return jsonify({'error': 'stale', 'message': _BLOCK_EDIT_STALE_MESSAGE}), 409

    project_dir = os.path.join(_projects_root(), project.folder_name)
    ann_ref = annotation_recording_ref(recording)
    annotations = get_annotations(project_dir, ann_ref)

    from ...services.speaker_blocks import block_text
    from ...services.transcript_edit import (
        BLOCK_EDIT_EMPTY,
        BLOCK_EDIT_OK,
        BLOCK_EDIT_STALE,
        apply_block_edit,
    )
    status = apply_block_edit(segments, annotations, raw_indices, original_text, new_text)

    if status == BLOCK_EDIT_STALE:
        return jsonify({'error': 'stale', 'message': _BLOCK_EDIT_STALE_MESSAGE}), 409
    if status == BLOCK_EDIT_EMPTY:
        return jsonify({'error': 'Nothing to save'}), 400

    changed = status == BLOCK_EDIT_OK
    if changed:
        atomic_write_text(transcript_file, json.dumps(transcript, ensure_ascii=False, indent=2))
        save_annotations(project_dir, ann_ref, annotations)
        _touch_project(project)
        db.session.commit()

    return jsonify({
        'changed': changed,
        'transcript': with_speaker_blocks(transcript),
        'annotations': annotations,
        'indices': raw_indices,
        'block_text': block_text(segments, raw_indices),
    })

@projects_bp.route('/<int:project_id>/recordings/<int:recording_id>', methods=['DELETE'])
def delete_recording(project_id, recording_id):
    recording = db.session.get(Recording, recording_id)
    if not recording or recording.project_id != project_id:
        return jsonify({'error': 'Recording not found'}), 404

    project = db.session.get(Project, project_id)
    if project:
        project_dir = _project_dir_or_none(project)
        filepath = _recording_filepath(project, recording)
        # Only delete the media file if it was copied into the project folder
        if not recording.is_linked and filepath and os.path.isfile(filepath):
            os.remove(filepath)
        if recording.transcript_path:
            transcript_file = _project_file_or_none(project_dir, recording.transcript_path)
            if transcript_file and os.path.isfile(transcript_file):
                os.remove(transcript_file)
        _delete_recording_annotation_files(project_dir, recording)

    db.session.delete(recording)
    _touch_project(project)
    db.session.commit()

    if project:
        _write_project_readme(project)
    return jsonify({'ok': True})

@projects_bp.route('/<int:project_id>/recordings/<int:recording_id>/transcription', methods=['DELETE'])
def cancel_transcription_route(project_id, recording_id):
    """Cancel an in-progress transcription job."""
    recording = db.session.get(Recording, recording_id)
    if not recording or recording.project_id != project_id:
        return jsonify({'error': 'Recording not found'}), 404
    if recording.transcription_status not in (
            'transcribing', 'cancelling', 'awaiting_language'):
        return jsonify({'error': 'Recording is not being transcribed'}), 409

    from ...services.transcription import cancel_transcription
    found = cancel_transcription(recording_id)
    if found:
        recording.transcription_status = 'cancelling'
        _touch_project(db.session.get(Project, project_id))
        db.session.commit()
        return jsonify({'ok': True, 'status': 'cancelling'})
    return jsonify({'error': 'No active job found for this recording'}), 404

@projects_bp.route(
    '/<int:project_id>/recordings/<int:recording_id>/transcription/language',
    methods=['POST'])
def submit_transcription_language(project_id, recording_id):
    """Resume transcription after the worker asked for a spoken language (low detector confidence)."""
    recording = db.session.get(Recording, recording_id)
    if not recording or recording.project_id != project_id:
        return jsonify({'error': 'Recording not found'}), 404
    if recording.transcription_status != 'awaiting_language':
        return jsonify({'error': 'This recording is not waiting for a language'}), 409

    data = request.get_json(silent=True) or {}
    lang = (data.get('language') or '').strip().lower()
    if not lang or not re.match(r'^[a-z][a-z0-9_]{0,15}$', lang):
        return jsonify({'error': 'Invalid language code (e.g. en, ru, de)'}), 400

    from ...services.transcription import notify_language_chosen
    if not notify_language_chosen(recording_id, lang):
        return jsonify({'error': 'No active language prompt for this recording'}), 404
    return jsonify({'ok': True, 'language': lang})
