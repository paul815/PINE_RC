"""Second pass over what a backup may put into the database, and what reads it.

A follow-up review found three things the first pass left open:

- A linked recording from someone else's backup could point at a network
  share (\\\\host\\share\\x.mp3). With its status left at 'transcribing', the
  next start re-queued it and the worker opened the share on its own — an SMB
  connection, and the user's NTLM hash, sent to whoever made the backup.
- file_format, language and transcription_status went in as given;
  file_format lands inside a JS string on the recording page.
- The transcript read/replace/edit routes and the tag views joined
  transcript_path without checking it stays inside the project — the edit
  routes write the file back.
"""

import hashlib
import json
import os
import zipfile

import pytest


def _projects_root(app):
    from app.models.setting import Setting
    return Setting.get('projects_path', app.config['DEFAULT_PROJECTS_PATH'])


def _backup_with_recording(app, client, **recording_overrides):
    """A backup of one project with one transcribed recording, rewritten as asked,
    with the original project gone so the restore applies."""
    import shutil

    from app.extensions import db
    from app.models.project import Project
    from app.models.recording import Recording
    from app.services.backup_service import _create_backup_inner

    proj = client.post('/api/projects', json={'name': 'Hardening'}).get_json()
    with app.app_context():
        project_dir = os.path.join(_projects_root(app), proj['folder_name'])
        os.makedirs(project_dir, exist_ok=True)
        with open(os.path.join(project_dir, 'talk_transcript.json'), 'w') as fh:
            json.dump({'segments': [{'start': 0, 'end': 1, 'text': 'hi'}]}, fh)
        db.session.add(Recording(
            project_id=proj['id'], original_name='talk.mp3', stored_name='talk.mp3',
            file_format='mp3', transcription_status='transcribed',
            transcript_path='talk_transcript.json',
        ))
        db.session.commit()
        filename = _create_backup_inner(app, include_audio=False)['filename']
        Recording.query.delete()
        Project.query.delete()
        db.session.commit()
        shutil.rmtree(project_dir)

    path = os.path.join(app.config['ROOT_DIR'], 'backups', filename)
    with zipfile.ZipFile(path) as src:
        entries = {n: src.read(n) for n in src.namelist()}
    projects = json.loads(entries['db/projects.json'])
    projects[0]['recordings'][0].update(recording_overrides)
    if recording_overrides.get('is_linked'):
        projects[0]['recordings'][0].pop('backup_archived_audio_name', None)
    entries['db/projects.json'] = json.dumps(projects).encode('utf-8')
    manifest = json.loads(entries['manifest.json'])
    manifest['checksums'] = {
        n: {'sha256': hashlib.sha256(d).hexdigest(), 'size_bytes': len(d)}
        for n, d in entries.items() if n != 'manifest.json'
    }
    entries['manifest.json'] = json.dumps(manifest).encode('utf-8')
    with zipfile.ZipFile(path, 'w') as dst:
        for n, d in entries.items():
            dst.writestr(n, d)
    return filename


def _restored_recording(app, filename):
    from app.models.recording import Recording
    from app.services.backup_service import _restore_inner
    with app.app_context():
        result = _restore_inner(app, filename, None, False, 'skip')
        rec = Recording.query.one()
        return result, {
            'stored_name': rec.stored_name, 'is_linked': rec.is_linked,
            'transcription_status': rec.transcription_status,
            'file_format': rec.file_format, 'language': rec.language,
        }


# ---------------------------------------------------------------------------
# Network links and in-flight status
# ---------------------------------------------------------------------------


class TestLinkedRecordingsFromABackup:
    @pytest.mark.parametrize('share', [
        r'\\attacker.example\share\x.mp3',
        '//attacker.example/share/x.mp3',
        r'\\?\UNC\attacker.example\share\x.mp3',
        r'\\.\attacker\x.mp3',
    ])
    def test_a_network_link_is_not_kept(self, app, client, share):
        """The project still restores — a NAS link in someone's own backup is
        legitimate — but the path to the share does not come with it."""
        filename = _backup_with_recording(app, client, is_linked=True, stored_name=share)

        result, rec = _restored_recording(app, filename)

        assert result['projects_restored'] == 1
        assert 'attacker' not in (rec['stored_name'] or '')

    def test_a_network_link_is_never_opened_by_the_backend(self, app, client, monkeypatch):
        """Status, media, backup: nothing may reach for the share afterwards."""
        from app.services.backup_service import _create_backup_inner

        share = r'\\attacker.example\share\x.mp3'
        filename = _backup_with_recording(app, client, is_linked=True, stored_name=share,
                                          transcription_status='transcribing')
        _restored_recording(app, filename)

        touched = []
        real_isfile = os.path.isfile
        monkeypatch.setattr(os.path, 'isfile',
                            lambda p: touched.append(str(p)) or real_isfile(p))
        with app.app_context():
            _create_backup_inner(app, include_audio=True)

        assert not [p for p in touched if 'attacker' in p]

    def test_a_local_media_link_still_restores(self, app, client, tmp_path):
        media = tmp_path / 'external' / 'call.m4a'
        media.parent.mkdir()
        media.write_bytes(b'\x00' * 8)
        filename = _backup_with_recording(app, client, is_linked=True, stored_name=str(media))

        _, rec = _restored_recording(app, filename)

        assert rec['is_linked'] is True
        assert rec['stored_name'] == str(media)


class TestStatusFromABackup:
    @pytest.mark.parametrize('status', ['transcribing', 'awaiting_language', 'cancelling'])
    def test_an_in_flight_status_is_not_restored(self, app, client, status):
        """requeue_interrupted() would pick it up at the next start."""
        filename = _backup_with_recording(app, client, transcription_status=status)

        _, rec = _restored_recording(app, filename)

        assert rec['transcription_status'] not in ('transcribing', 'awaiting_language', 'cancelling')

    def test_an_unknown_status_is_not_restored(self, app, client):
        filename = _backup_with_recording(app, client, transcription_status='<img src=x>')

        _, rec = _restored_recording(app, filename)

        assert rec['transcription_status'] in ('pending', 'transcribed', 'error', 'cancelled')

    def test_a_finished_transcript_stays_transcribed(self, app, client):
        filename = _backup_with_recording(app, client)

        _, rec = _restored_recording(app, filename)

        assert rec['transcription_status'] == 'transcribed'


class TestOtherFieldsFromABackup:
    def test_file_format_and_language_must_look_like_what_they_are(self, app, client):
        filename = _backup_with_recording(
            app, client, file_format="mp3\\';alert(1)//\n", language='en"><b>',
        )

        _, rec = _restored_recording(app, filename)

        assert rec['file_format'] in ('', 'mp3')
        assert rec['language'] == ''

    def test_ordinary_values_survive(self, app, client):
        filename = _backup_with_recording(app, client, file_format='m4a', language='pt-br')

        _, rec = _restored_recording(app, filename)

        assert rec['file_format'] == 'm4a'
        assert rec['language'] == 'pt-br'


class TestRecordingPageFileFormat:
    def test_file_format_reaches_the_script_as_a_json_string(self, app, client, project_with_recording):
        from app.extensions import db
        from app.models.recording import Recording
        from app.models.setting import Setting

        pid, rid, _ = project_with_recording({'segments': []}, file_format='mp3\\\n')
        with app.app_context():
            Setting.set('onboarding_complete', 'true')
            db.session.commit()
            assert db.session.get(Recording, rid).file_format == 'mp3\\\n'

        html = client.get(f'/project/{pid}/recording/{rid}').get_data(as_text=True)

        line = next(ln for ln in html.splitlines() if 'const FILE_FORMAT' in ln)
        assert json.loads(line.split('=', 1)[1].strip().rstrip(';')) == 'mp3\\\n'


# ---------------------------------------------------------------------------
# Transcript routes stay inside the project
# ---------------------------------------------------------------------------


class TestTranscriptRoutesStayInside:
    @pytest.fixture
    def poisoned(self, project_with_recording, tmp_path):
        outside = tmp_path / 'outside_transcript.json'
        outside.write_text(json.dumps({'segments': [
            {'start': 0, 'end': 1, 'text': 'secret words', 'speaker': 'S'},
        ]}), encoding='utf-8')
        pid, rid, project_dir = project_with_recording(transcript_path=str(outside))
        self.project_dir = project_dir
        return pid, rid, outside

    def test_read_refuses(self, client, poisoned):
        pid, rid, _ = poisoned

        res = client.get(f'/api/projects/{pid}/recordings/{rid}/transcript')

        assert res.status_code == 404
        assert b'secret' not in res.data

    def test_recording_view_does_not_include_it(self, client, poisoned):
        pid, rid, _ = poisoned

        res = client.get(f'/api/projects/{pid}/recordings/{rid}')

        assert b'secret' not in res.data

    def test_replace_does_not_write_outside(self, client, poisoned):
        pid, rid, outside = poisoned
        before = outside.read_bytes()

        client.post(f'/api/projects/{pid}/recordings/{rid}/transcript/replace',
                    json={'find': 'secret', 'replace': 'public'})

        assert outside.read_bytes() == before

    def test_block_edit_does_not_write_outside(self, client, poisoned):
        pid, rid, outside = poisoned
        before = outside.read_bytes()

        client.post(f'/api/projects/{pid}/recordings/{rid}/transcript/block',
                    json={'indices': [0], 'original_text': 'secret words', 'new_text': 'x'})

        assert outside.read_bytes() == before

    def test_startup_check_flags_it_instead_of_following_it(self, app, poisoned):
        """The one pass over every row at start: a transcript outside the
        project is reported like a missing one, so no route treats it as ready."""
        from app import _run_startup_checks
        from app.extensions import db
        from app.models.recording import Recording

        _, rid, outside = poisoned
        before = outside.read_bytes()

        with app.app_context():
            _run_startup_checks(app)
            rec = db.session.get(Recording, rid)
            assert rec.transcription_status == 'error'
            assert 'outside' in rec.error_message
        assert outside.read_bytes() == before

    def test_tags_view_does_not_read_it(self, client, poisoned):
        """A coded quote shows its words; the words must not come from outside."""
        pid, _, _ = poisoned
        project_dir = self.project_dir
        with open(os.path.join(project_dir, 'project_tags.json'), 'w') as fh:
            json.dump([{'id': 't1', 'name': 'pain', 'color': '#ff0000'}], fh)
        with open(os.path.join(project_dir, 'outside_annotations.json'), 'w') as fh:
            json.dump({'tag_spans': [{'id': 's1', 'tag_id': 't1', 'segment_idx': 0,
                                      'start_char': 0, 'end_char': 6}],
                       'comments': []}, fh)

        res = client.get(f'/api/projects/{pid}/tags/quotes')

        assert res.status_code == 200
        assert b'secret' not in res.data
