"""Paths that come from a backup, or from a DB row a backup wrote, must stay
inside the projects directory.

A backup ZIP is data from outside: an archive a colleague sent, a file in a
synced folder. Its ``db/projects.json`` names the project folder and every
recording's ``stored_name`` / ``transcript_path``. Taken at face value, an
absolute path or a ``..`` there let a restore write — and, on rollback, delete —
anything the user can touch, and the poisoned row then drove ``/media``,
recording delete and project delete at arbitrary paths.

Also here: an upload must never clobber an existing backup of the same name,
and Reset must only remove folders PINE itself created.
"""

import hashlib
import io
import json
import os
import zipfile

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _projects_root(app):
    from app.models.setting import Setting
    return Setting.get('projects_path', app.config['DEFAULT_PROJECTS_PATH'])


def _backup_dir(app):
    return os.path.join(app.config['ROOT_DIR'], 'backups')


def _make_backup(app, client, name='Source Project'):
    """One project with one copied recording and a transcript, backed up."""
    from app.extensions import db
    from app.models.recording import Recording
    from app.services.backup_service import _create_backup_inner

    proj = client.post('/api/projects', json={'name': name}).get_json()
    with app.app_context():
        project_dir = os.path.join(_projects_root(app), proj['folder_name'])
        os.makedirs(project_dir, exist_ok=True)
        with open(os.path.join(project_dir, 'talk.mp3'), 'wb') as fh:
            fh.write(b'\x00' * 64)
        with open(os.path.join(project_dir, 'talk_transcript.json'), 'w') as fh:
            json.dump({'segments': []}, fh)
        db.session.add(Recording(
            project_id=proj['id'],
            original_name='talk.mp3',
            stored_name='talk.mp3',
            file_format='mp3',
            transcription_status='transcribed',
            transcript_path='talk_transcript.json',
        ))
        db.session.commit()
        result = _create_backup_inner(app, include_audio=True)
    return result['filename'], proj


def _rewrite_backup(app, filename, mutate, extra_entries=None):
    """Edit db/projects.json in a backup and re-sign it, as an attacker would.

    The checksums in the manifest are computed by whoever writes the archive,
    so they say nothing about where its paths point.
    """
    zip_path = os.path.join(_backup_dir(app), filename)
    with zipfile.ZipFile(zip_path) as src:
        entries = {name: src.read(name) for name in src.namelist()}

    projects = json.loads(entries['db/projects.json'])
    mutate(projects)
    entries['db/projects.json'] = json.dumps(projects).encode('utf-8')
    entries.update(extra_entries or {})

    manifest = json.loads(entries['manifest.json'])
    manifest['checksums'] = {
        name: {'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}
        for name, data in entries.items()
        if name != 'manifest.json'
    }
    entries['manifest.json'] = json.dumps(manifest).encode('utf-8')

    with zipfile.ZipFile(zip_path, 'w') as dst:
        for name, data in entries.items():
            dst.writestr(name, data)


def _restore(app, filename, strategy='skip'):
    from app.services.backup_service import _restore_inner
    with app.app_context():
        return _restore_inner(app, filename, None, False, strategy)


def _set_first(projects, key, value):
    projects[0][key] = value


def _set_first_recording(projects, key, value):
    projects[0]['recordings'][0][key] = value


# ---------------------------------------------------------------------------
# Restore: folder_name
# ---------------------------------------------------------------------------


class TestRestoreProjectFolder:
    def test_absolute_folder_name_is_refused_and_nothing_is_written(self, app, client, temp_dir):
        filename, _ = _make_backup(app, client)
        outside = os.path.join(temp_dir, 'outside_target')
        # Forward slashes: ZIP entry names always use them, and Windows accepts
        # them in a path, so this is the spelling that actually lands.
        folder = outside.replace(os.sep, '/')
        _rewrite_backup(
            app, filename,
            lambda p: _set_first(p, 'folder_name', folder),
            extra_entries={f'projects/{folder}/evil.txt': b'planted'},
        )

        with pytest.raises(ValueError, match='(?i)unsafe'):
            _restore(app, filename)

        assert not os.path.exists(outside)

    def test_parent_traversal_folder_name_is_refused(self, app, client, temp_dir):
        filename, _ = _make_backup(app, client)
        _rewrite_backup(
            app, filename,
            lambda p: _set_first(p, 'folder_name', '../escaped'),
            extra_entries={'projects/../escaped/evil.txt': b'planted'},
        )

        with pytest.raises(ValueError):
            _restore(app, filename)

        assert not os.path.exists(os.path.join(temp_dir, 'escaped'))

    def test_nested_folder_name_is_refused(self, app, client):
        filename, _ = _make_backup(app, client)
        _rewrite_backup(app, filename, lambda p: _set_first(p, 'folder_name', 'a/b'))

        with pytest.raises(ValueError):
            _restore(app, filename)

    def test_overwrite_never_deletes_a_folder_that_contains_the_projects_root(
            self, app, client, temp_dir):
        """The rmtree-on-rollback case: folder_name names a parent of PINE.

        Moving a directory into its own child fails, and the rollback used to
        answer that by deleting the 'restored' folder — the parent itself.
        """
        filename, _ = _make_backup(app, client)
        marker = os.path.join(temp_dir, 'keep_me.txt')
        with open(marker, 'w') as fh:
            fh.write('user data next to PINE')
        _rewrite_backup(app, filename, lambda p: _set_first(p, 'folder_name', temp_dir))

        with pytest.raises(ValueError):
            _restore(app, filename, strategy='overwrite')

        assert os.path.isfile(marker)

    def test_refused_restore_leaves_the_database_untouched(self, app, client):
        from app.models.project import Project

        filename, _ = _make_backup(app, client)
        _rewrite_backup(app, filename, lambda p: _set_first(p, 'folder_name', '../escaped'))
        with app.app_context():
            before = Project.query.count()

        with pytest.raises(ValueError):
            _restore(app, filename, strategy='rename')

        with app.app_context():
            assert Project.query.count() == before

    def test_unicode_folder_name_still_restores(self, app, client):
        """Guard: the check must not reject the names PINE itself produces."""
        import shutil

        from app.extensions import db
        from app.models.project import Project
        from app.models.recording import Recording

        filename, proj = _make_backup(app, client, name='Интервью клиентов')
        with app.app_context():
            project_dir = os.path.join(_projects_root(app), proj['folder_name'])
            Recording.query.delete()
            Project.query.delete()
            db.session.commit()
            shutil.rmtree(project_dir)

        result = _restore(app, filename)

        assert result['projects_restored'] == 1
        assert os.path.isfile(os.path.join(project_dir, 'talk_transcript.json'))


# ---------------------------------------------------------------------------
# Restore: recording paths
# ---------------------------------------------------------------------------


class TestRestoreRecordingPaths:
    def _fresh_restore_target(self, app, client):
        """A backup whose project no longer exists, so the restore would apply."""
        import shutil

        from app.extensions import db
        from app.models.project import Project
        from app.models.recording import Recording

        filename, proj = _make_backup(app, client)
        with app.app_context():
            Recording.query.delete()
            Project.query.delete()
            db.session.commit()
            shutil.rmtree(os.path.join(_projects_root(app), proj['folder_name']))
        return filename

    def test_stored_name_outside_the_project_is_refused(self, app, client, temp_dir):
        filename = self._fresh_restore_target(app, client)
        target = os.path.join(temp_dir, 'victim.mp3')
        _rewrite_backup(app, filename, lambda p: _set_first_recording(p, 'stored_name', target))

        with pytest.raises(ValueError):
            _restore(app, filename)

    def test_stored_name_with_parent_traversal_is_refused(self, app, client):
        filename = self._fresh_restore_target(app, client)
        _rewrite_backup(
            app, filename,
            lambda p: _set_first_recording(p, 'stored_name', '../../victim.mp3'),
        )

        with pytest.raises(ValueError):
            _restore(app, filename)

    def test_transcript_path_outside_the_project_is_refused(self, app, client, temp_dir):
        filename = self._fresh_restore_target(app, client)
        target = os.path.join(temp_dir, 'victim_transcript.json')
        _rewrite_backup(
            app, filename,
            lambda p: _set_first_recording(p, 'transcript_path', target),
        )

        with pytest.raises(ValueError):
            _restore(app, filename)

    def test_linked_recording_to_a_non_media_file_is_refused(self, app, client, temp_dir):
        filename = self._fresh_restore_target(app, client)
        secret = os.path.join(temp_dir, 'id_rsa')
        with open(secret, 'w') as fh:
            fh.write('private key')

        def poison(projects):
            rec = projects[0]['recordings'][0]
            rec['is_linked'] = True
            rec['stored_name'] = secret
            rec.pop('backup_archived_audio_name', None)

        _rewrite_backup(app, filename, poison)

        with pytest.raises(ValueError):
            _restore(app, filename)

    def test_linked_media_recording_still_restores_as_linked(self, app, client, temp_dir):
        """Guard: a real linked recording (Zoom folder, external drive) survives."""
        from app.models.recording import Recording

        filename = self._fresh_restore_target(app, client)
        media = os.path.join(temp_dir, 'external', 'call.m4a')
        os.makedirs(os.path.dirname(media))
        with open(media, 'wb') as fh:
            fh.write(b'\x00' * 16)

        def link(projects):
            rec = projects[0]['recordings'][0]
            rec['is_linked'] = True
            rec['stored_name'] = media
            rec.pop('backup_archived_audio_name', None)

        _rewrite_backup(app, filename, link)

        result = _restore(app, filename)

        assert result['projects_restored'] == 1
        with app.app_context():
            rec = Recording.query.one()
            assert rec.is_linked is True
            assert rec.stored_name == media

    @pytest.mark.parametrize('link', [
        '/Users/researcher/Zoom/call.m4a',     # made on a Mac
        'D:\\Zoom\\2026-11-01 call\\call.mp4',  # made on Windows
    ])
    def test_linked_recording_from_the_other_os_still_restores(self, app, client, link):
        """A backup travels between the Windows and Mac machines.

        The link will not resolve there, but the recording, its transcript and
        its annotations must still come back; refusing the archive would lose
        all of it over a path that was never going to be opened.
        """
        from app.models.recording import Recording

        filename = self._fresh_restore_target(app, client)

        def link_to(projects):
            rec = projects[0]['recordings'][0]
            rec['is_linked'] = True
            rec['stored_name'] = link
            rec.pop('backup_archived_audio_name', None)

        _rewrite_backup(app, filename, link_to)

        result = _restore(app, filename)

        assert result['projects_restored'] == 1
        with app.app_context():
            assert Recording.query.one().stored_name == link


# ---------------------------------------------------------------------------
# Rows already in the DB: the routes must not follow them out of the project
# ---------------------------------------------------------------------------


class TestRoutesStayInsideProject:
    def test_delete_recording_leaves_files_outside_the_project(
            self, client, project_with_recording, temp_dir):
        outside_media = os.path.join(temp_dir, 'outside.mp3')
        outside_transcript = os.path.join(temp_dir, 'outside_transcript.json')
        for path in (outside_media, outside_transcript):
            with open(path, 'w') as fh:
                fh.write('not PINE data')

        pid, rid, _ = project_with_recording(
            stored_name=outside_media,
            transcript_path=outside_transcript,
            is_linked=False,
        )

        res = client.delete(f'/api/projects/{pid}/recordings/{rid}')

        assert res.status_code == 200
        assert os.path.isfile(outside_media)
        assert os.path.isfile(outside_transcript)

    def test_media_route_refuses_unlinked_path_outside_the_project(
            self, client, project_with_recording, temp_dir):
        outside = os.path.join(temp_dir, 'outside.mp3')
        with open(outside, 'wb') as fh:
            fh.write(b'secret bytes')

        pid, rid, _ = project_with_recording(stored_name=outside, is_linked=False)

        res = client.get(f'/api/projects/{pid}/recordings/{rid}/media')

        assert res.status_code == 404

    def test_media_route_refuses_a_linked_path_that_is_not_media(
            self, client, project_with_recording, temp_dir):
        secret = os.path.join(temp_dir, 'id_rsa')
        with open(secret, 'w') as fh:
            fh.write('private key')

        pid, rid, _ = project_with_recording(stored_name=secret, is_linked=True)

        res = client.get(f'/api/projects/{pid}/recordings/{rid}/media')

        assert res.status_code == 404

    def test_media_route_still_serves_linked_media(
            self, client, project_with_recording, temp_dir):
        media = os.path.join(temp_dir, 'external', 'call.mp3')
        os.makedirs(os.path.dirname(media))
        with open(media, 'wb') as fh:
            fh.write(b'\xff\xfb' * 32)

        pid, rid, _ = project_with_recording(stored_name=media, is_linked=True)

        res = client.get(f'/api/projects/{pid}/recordings/{rid}/media')

        assert res.status_code == 200
        res.close()

    def test_delete_project_leaves_a_folder_outside_the_projects_root(self, app, client, temp_dir):
        from app.extensions import db
        from app.models.project import Project

        outside = os.path.join(temp_dir, 'not_a_project')
        os.makedirs(outside)
        with open(os.path.join(outside, 'keep.txt'), 'w') as fh:
            fh.write('user data')

        with app.app_context():
            proj = Project(name='Poisoned', folder_name=outside)
            db.session.add(proj)
            db.session.commit()
            pid = proj.id

        res = client.delete(f'/api/projects/{pid}')

        assert res.status_code == 200
        assert os.path.isfile(os.path.join(outside, 'keep.txt'))

    def test_rename_project_does_not_move_a_folder_from_outside_the_root(
            self, app, client, temp_dir):
        from app.extensions import db
        from app.models.project import Project

        outside = os.path.join(temp_dir, 'not_a_project')
        os.makedirs(outside)
        with open(os.path.join(outside, 'keep.txt'), 'w') as fh:
            fh.write('user data')

        with app.app_context():
            proj = Project(name='Poisoned', folder_name=outside)
            db.session.add(proj)
            db.session.commit()
            pid = proj.id

        res = client.patch(f'/api/projects/{pid}', json={'name': 'Renamed'})

        assert res.status_code == 200
        assert os.path.isfile(os.path.join(outside, 'keep.txt'))


# ---------------------------------------------------------------------------
# Backup: a poisoned row must not carry a file out inside the next archive
# ---------------------------------------------------------------------------


def _archive_payloads(app, filename):
    with zipfile.ZipFile(os.path.join(_backup_dir(app), filename)) as zf:
        return [zf.read(name) for name in zf.namelist()]


class TestBackupStaysInsideProjects:
    SECRET = b'-----BEGIN OPENSSH PRIVATE KEY-----'

    def _backup(self, app):
        from app.services.backup_service import _create_backup_inner
        with app.app_context():
            return _create_backup_inner(app, include_audio=True)['filename']

    def test_project_folder_outside_the_root_is_not_archived(self, app, temp_dir):
        from app.extensions import db
        from app.models.project import Project

        outside = os.path.join(temp_dir, 'outside_project')
        os.makedirs(outside)
        with open(os.path.join(outside, 'project_tags.json'), 'wb') as fh:
            fh.write(self.SECRET)
        with app.app_context():
            db.session.add(Project(name='Poisoned', folder_name=outside))
            db.session.commit()

        payloads = _archive_payloads(app, self._backup(app))

        assert self.SECRET not in payloads

    def test_recording_files_outside_the_project_are_not_archived(
            self, app, project_with_recording, temp_dir):
        media = os.path.join(temp_dir, 'outside.mp3')
        transcript = os.path.join(temp_dir, 'outside_transcript.json')
        for path in (media, transcript):
            with open(path, 'wb') as fh:
                fh.write(self.SECRET)
        project_with_recording(stored_name=media, transcript_path=transcript, is_linked=False)

        payloads = _archive_payloads(app, self._backup(app))

        assert self.SECRET not in payloads

    def test_linked_recording_to_a_non_media_file_is_not_archived(
            self, app, project_with_recording, temp_dir):
        secret = os.path.join(temp_dir, 'id_rsa')
        with open(secret, 'wb') as fh:
            fh.write(self.SECRET)
        project_with_recording(stored_name=secret, transcript_path='', is_linked=True)

        payloads = _archive_payloads(app, self._backup(app))

        assert self.SECRET not in payloads


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


class TestUploadKeepsExistingBackups:
    def _existing_backup(self, app):
        from app.services.backup_service import _create_backup_inner
        with app.app_context():
            result = _create_backup_inner(app, include_audio=False)
        path = os.path.join(_backup_dir(app), result['filename'])
        with open(path, 'rb') as fh:
            return result['filename'], path, fh.read()

    def test_invalid_upload_does_not_delete_a_backup_with_the_same_name(self, app, client):
        filename, path, original = self._existing_backup(app)

        res = client.post(
            '/api/backup/upload',
            data={'file': (io.BytesIO(b'not a zip'), filename)},
            content_type='multipart/form-data',
        )

        assert res.status_code == 400
        assert os.path.isfile(path)
        with open(path, 'rb') as fh:
            assert fh.read() == original

    def test_valid_upload_with_the_same_name_is_stored_alongside(self, app, client):
        filename, path, original = self._existing_backup(app)

        res = client.post(
            '/api/backup/upload',
            data={'file': (io.BytesIO(original), filename)},
            content_type='multipart/form-data',
        )

        assert res.status_code == 200
        stored = res.get_json()['filename']
        assert stored != filename
        assert stored.startswith('pine_backup_') and stored.endswith('.zip')
        assert os.path.isfile(os.path.join(_backup_dir(app), stored))
        with open(path, 'rb') as fh:
            assert fh.read() == original

    def test_rejected_upload_leaves_no_partial_file(self, app, client):
        os.makedirs(_backup_dir(app), exist_ok=True)
        before = set(os.listdir(_backup_dir(app)))

        res = client.post(
            '/api/backup/upload',
            data={'file': (io.BytesIO(b'not a zip'), 'fresh.zip')},
            content_type='multipart/form-data',
        )

        assert res.status_code == 400
        assert set(os.listdir(_backup_dir(app))) == before


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestResetScope:
    """The risk is a projects path the user picked at onboarding: an existing
    folder that already held other things. The default ``<root>/projects`` is
    PINE's own and goes with the workspace cleanup, which is stubbed out here
    along with the launcher shortcuts so the test touches nothing real.
    """

    @pytest.fixture
    def custom_projects_root(self, app, tmp_path, monkeypatch):
        from app.api import settings as settings_api
        from app.models.setting import Setting

        monkeypatch.setattr(settings_api, '_cleanup_reset_workspace', lambda _root: [])
        monkeypatch.setattr(settings_api, '_remove_all_app_launcher_entries', lambda: None)
        monkeypatch.setattr(
            settings_api, 'restore_default_launcher_layout_after_reset', lambda _root: None,
        )

        root = tmp_path / 'chosen at onboarding'
        root.mkdir()
        with app.app_context():
            Setting.set('projects_path', str(root))
        return str(root)

    def test_reset_removes_project_folders_but_not_foreign_ones(
            self, client, custom_projects_root):
        proj = client.post('/api/projects', json={'name': 'Doomed'}).get_json()
        project_dir = os.path.join(custom_projects_root, proj['folder_name'])
        os.makedirs(project_dir, exist_ok=True)

        foreign = os.path.join(custom_projects_root, 'My Photos')
        os.makedirs(foreign)
        with open(os.path.join(foreign, 'holiday.jpg'), 'wb') as fh:
            fh.write(b'\xff\xd8')

        res = client.post('/api/settings/reset', json={'confirm': 'Yes'})

        assert res.status_code == 200
        assert not os.path.exists(project_dir)
        assert os.path.isfile(os.path.join(foreign, 'holiday.jpg'))

    def test_reset_never_follows_a_folder_name_out_of_the_projects_root(
            self, app, client, custom_projects_root, tmp_path):
        from app.extensions import db
        from app.models.project import Project

        outside = tmp_path / 'outside_reset'
        outside.mkdir()
        (outside / 'keep.txt').write_text('user data', encoding='utf-8')
        with app.app_context():
            db.session.add(Project(name='Poisoned', folder_name=str(outside)))
            db.session.commit()

        res = client.post('/api/settings/reset', json={'confirm': 'Yes'})

        assert res.status_code == 200
        assert (outside / 'keep.txt').is_file()
