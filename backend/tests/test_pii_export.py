"""Export with "Remove PII" on: what it must cover, and that it fails closed.

Before, a detector error came back as "no entities" and the transcript left
unredacted without a word. And only the transcript text was ever redacted: a
renamed speaker, the recording's file name, comments and participant notes —
the places a respondent's name most often sits — went out as they were.

A stand-in for GLiNER finds one name, so these run without the model.
"""

import json
import os

import pytest

NAME = 'Anna Petrova'


class _FakeGliner:
    def __init__(self, fail=False):
        self.fail = fail

    def predict_entities(self, text, labels, threshold=0.5):
        if self.fail:
            raise RuntimeError('model exploded')
        found, i = [], text.find(NAME)
        while i >= 0:
            found.append({'start': i, 'end': i + len(NAME), 'text': NAME, 'label': 'person'})
            i = text.find(NAME, i + 1)
        return found


@pytest.fixture
def fake_pii(monkeypatch):
    from app.services import pii_service

    def install(fail=False):
        monkeypatch.setattr(pii_service, '_gliner_model', _FakeGliner(fail=fail))
        monkeypatch.setattr(pii_service, '_load_model', lambda app: None)
    return install


@pytest.fixture
def recording(app):
    """A transcribed recording whose name, speaker, comment and notes hold NAME."""
    from app.extensions import db
    from app.models.project import Project
    from app.models.recording import Recording
    from app.models.setting import Setting

    with app.app_context():
        root = app.config['DEFAULT_PROJECTS_PATH']
        Setting.set('projects_path', root)
        proj = Project(name='PII Project', folder_name='pii_project')
        db.session.add(proj)
        db.session.flush()
        rec = Recording(
            project_id=proj.id,
            original_name='Interview_Anna_Petrova.mp3',
            stored_name='interview.mp3',
            transcript_path='interview_transcript.json',
            transcription_status='transcribed',
            participant_notes=f'Recruited via {NAME}, call back on Monday.',
        )
        db.session.add(rec)
        db.session.commit()
        pid, rid = proj.id, rec.id

    project_dir = os.path.join(root, 'pii_project')
    os.makedirs(project_dir, exist_ok=True)
    with open(os.path.join(project_dir, 'interview_transcript.json'), 'w', encoding='utf-8') as fh:
        json.dump({'segments': [
            {'start': 0, 'end': 2, 'speaker': 'SPEAKER_00', 'text': 'Tell me about your week.'},
            {'start': 2, 'end': 4, 'speaker': 'SPEAKER_01', 'text': 'It was busy.'},
        ]}, fh)
    with open(os.path.join(project_dir, 'interview_annotations.json'), 'w', encoding='utf-8') as fh:
        json.dump({
            'tag_spans': [],
            'comments': [{
                'id': 'c1', 'segment_idx': 1, 'start_char': 0, 'end_char': 6,
                'text': f'{NAME} hesitated here',
            }],
            'speaker_labels': {'SPEAKER_00': 'Moderator', 'SPEAKER_01': NAME},
        }, fh)
    return pid, rid


OPTS = {
    'include_comments': True, 'include_tags': True,
    'include_project_details': True, 'include_participant_details': True,
    'include_prompt': False, 'remove_pii': True,
}


def _export_recording(app, pid, rid, **overrides):
    from app.services.export_service import export_recording_markdown
    body, err = export_recording_markdown(app, pid, rid, {**OPTS, **overrides})
    assert err is None
    return body


class TestRemovePiiCoversEverything:
    def test_no_trace_of_the_name_anywhere(self, app, recording, fake_pii):
        fake_pii()

        body = _export_recording(app, *recording)

        assert NAME not in body
        assert 'Anna_Petrova' not in body
        assert 'Petrova' not in body

    def test_renamed_speaker_becomes_a_stable_pseudonym(self, app, recording, fake_pii):
        fake_pii()

        body = _export_recording(app, *recording)

        assert '**Speaker 2**' in body

    def test_a_role_label_is_kept(self, app, recording, fake_pii):
        fake_pii()

        body = _export_recording(app, *recording)

        assert '**Moderator**' in body

    def test_the_comment_survives_with_the_name_taken_out(self, app, recording, fake_pii):
        fake_pii()

        body = _export_recording(app, *recording)

        assert 'hesitated here' in body

    def test_project_export_covers_the_same_places(self, app, recording, fake_pii):
        from app.services.export_service import export_project_markdown

        fake_pii()
        pid, rid = recording

        body, err = export_project_markdown(app, pid, [rid], OPTS)

        assert err is None
        assert 'Petrova' not in body

    def test_without_remove_pii_nothing_is_touched(self, app, recording, fake_pii):
        fake_pii()

        body = _export_recording(app, *recording, remove_pii=False)

        assert f'**{NAME}**' in body
        assert 'Interview_Anna_Petrova' in body


class TestRemovePiiFailsClosed:
    def test_a_detector_error_stops_the_export(self, app, recording, fake_pii):
        from app.services.pii_service import PIIError

        fake_pii(fail=True)

        with pytest.raises(PIIError):
            _export_recording(app, *recording)

    def test_the_api_answers_with_an_error_not_the_transcript(self, client, recording, fake_pii):
        fake_pii(fail=True)
        pid, rid = recording

        res = client.get(
            f'/api/projects/{pid}/recordings/{rid}/export',
            query_string={'format': 'markdown', 'remove_pii': 'true'},
        )

        assert res.status_code == 422
        assert b'busy' not in res.data
