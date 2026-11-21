"""Tests for backup & restore functionality."""

import json
import os
import zipfile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_project(client, name='Test Project'):
    """Create a project via API and return its JSON dict."""
    res = client.post('/api/projects', json={'name': name})
    assert res.status_code == 201 or res.status_code == 200
    return res.get_json()


def _seed_project_files(app, project):
    """Write dummy transcript, annotations, and tags for a project."""
    from app.models.setting import Setting
    with app.app_context():
        projects_root = Setting.get('projects_path', app.config['DEFAULT_PROJECTS_PATH'])
    folder = project['folder_name']
    project_dir = os.path.join(projects_root, folder)
    os.makedirs(project_dir, exist_ok=True)

    # Transcript
    transcript = {'segments': [{'speaker': 'SPEAKER_00', 'text': 'Hello world', 'start': 0, 'end': 1.5}]}
    transcript_name = 'test_audio_transcript.json'
    with open(os.path.join(project_dir, transcript_name), 'w') as f:
        json.dump(transcript, f)

    # Annotations
    annotations = {'tag_spans': [], 'comments': [], 'speaker_labels': {}}
    with open(os.path.join(project_dir, 'test_audio_annotations.json'), 'w') as f:
        json.dump(annotations, f)

    # Tags
    tags = [{'id': 'tag1', 'name': 'pain', 'color': '#ff0000'}]
    with open(os.path.join(project_dir, 'project_tags.json'), 'w') as f:
        json.dump(tags, f)

    # Fake audio file
    with open(os.path.join(project_dir, 'test_audio.mp3'), 'wb') as f:
        f.write(b'\x00' * 1024)

    return transcript_name


def _add_recording_to_db(app, project_id, transcript_name='test_audio_transcript.json'):
    """Insert a recording row into the DB."""
    from app.extensions import db
    from app.models.recording import Recording
    with app.app_context():
        rec = Recording(
            project_id=project_id,
            original_name='test_audio.mp3',
            stored_name='test_audio.mp3',
            file_format='mp3',
            file_size_bytes=1024,
            duration_seconds=90.0,
            transcription_status='transcribed',
            transcript_path=transcript_name,
        )
        db.session.add(rec)
        db.session.commit()
        return rec.id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCreateBackup:
    def test_create_backup_empty(self, app, client):
        """Backup with no projects creates a valid ZIP with manifest."""
        from app.services.backup_service import _create_backup_inner
        with app.app_context():
            result = _create_backup_inner(app, include_audio=False)
        assert result is not None
        assert result['project_count'] == 0
        assert result['filename'].startswith('pine_backup_')
        assert result['filename'].endswith('.zip')

        # Verify ZIP structure
        backup_dir = os.path.join(app.config['ROOT_DIR'], 'backups')
        zip_path = os.path.join(backup_dir, result['filename'])
        assert os.path.isfile(zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = zf.namelist()
            assert 'manifest.json' in names
            assert 'db/settings.json' in names
            assert 'db/projects.json' in names
            assert 'db/segments.json' in names

    def test_backup_carries_no_credentials_or_local_paths(self, app, client):
        """A backup ZIP travels; the HuggingFace token must not travel with it.

        Backups get copied to USB sticks, dropped into synced folders and sent
        to whoever is helping. Restore already refuses to apply these keys, but
        that only protects the machine reading the archive -- the secret was
        still written into it. Filter at the point of writing instead.
        """
        from app.models.setting import Setting
        with app.app_context():
            Setting.set('hf_token', 'hf_thisisalivecredential')
            Setting.set('models_path', r'C:\Users\someone\models')
            Setting.set('theme', 'dark')

        from app.services.backup_service import _create_backup_inner
        with app.app_context():
            result = _create_backup_inner(app, include_audio=False)

        zip_path = os.path.join(app.config['ROOT_DIR'], 'backups', result['filename'])
        with zipfile.ZipFile(zip_path, 'r') as zf:
            settings = json.loads(zf.read('db/settings.json'))
            raw = b''.join(zf.read(name) for name in zf.namelist())

        assert 'hf_token' not in settings
        assert 'models_path' not in settings
        # ...and nowhere else in the archive either.
        assert b'hf_thisisalivecredential' not in raw

        # The settings a backup exists to carry are still there.
        assert settings['theme'] == 'dark'

    def test_create_backup_with_project(self, app, client):
        """Backup includes project files (transcripts, annotations, tags)."""
        proj = _create_project(client, 'My Research')
        transcript_name = _seed_project_files(app, proj)
        _add_recording_to_db(app, proj['id'], transcript_name)

        from app.services.backup_service import _create_backup_inner
        with app.app_context():
            result = _create_backup_inner(app, include_audio=False)

        assert result['project_count'] == 1
        backup_dir = os.path.join(app.config['ROOT_DIR'], 'backups')
        zip_path = os.path.join(backup_dir, result['filename'])
        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = zf.namelist()
            folder = proj['folder_name']
            assert f'projects/{folder}/project_tags.json' in names
            assert f'projects/{folder}/{transcript_name}' in names
            assert f'projects/{folder}/test_audio_annotations.json' in names
            # No audio by default
            assert f'projects/{folder}/test_audio.mp3' not in names

    def test_create_backup_with_audio(self, app, client):
        """Backup with include_audio=True includes audio files."""
        proj = _create_project(client, 'Audio Project')
        transcript_name = _seed_project_files(app, proj)
        _add_recording_to_db(app, proj['id'], transcript_name)

        from app.services.backup_service import _create_backup_inner
        with app.app_context():
            result = _create_backup_inner(app, include_audio=True)

        assert '_with_audio' in result['filename']
        backup_dir = os.path.join(app.config['ROOT_DIR'], 'backups')
        zip_path = os.path.join(backup_dir, result['filename'])
        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = zf.namelist()
            folder = proj['folder_name']
            assert f'projects/{folder}/test_audio.mp3' in names

    def test_create_backup_includes_odt_export(self, app, client):
        """Backup includes ODT export in exports/ folder."""
        proj = _create_project(client, 'ODT Test')
        transcript_name = _seed_project_files(app, proj)
        _add_recording_to_db(app, proj['id'], transcript_name)

        from app.services.backup_service import _create_backup_inner
        with app.app_context():
            result = _create_backup_inner(app, include_audio=False)

        backup_dir = os.path.join(app.config['ROOT_DIR'], 'backups')
        zip_path = os.path.join(backup_dir, result['filename'])
        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = zf.namelist()
            folder = proj['folder_name']
            assert f'exports/{folder}.odt' in names


class TestListBackups:
    def test_list_backups_empty(self, app, client):
        """List backups returns empty list when no backups exist."""
        res = client.get('/api/backup')
        assert res.status_code == 200
        assert res.get_json() == []

    def test_list_backups_after_create(self, app, client):
        """List backups returns backup metadata after creating one."""
        from app.services.backup_service import _create_backup_inner
        with app.app_context():
            _create_backup_inner(app, include_audio=False)

        res = client.get('/api/backup')
        assert res.status_code == 200
        data = res.get_json()
        assert len(data) == 1
        assert data[0]['project_count'] == 0
        assert data[0]['filename'].startswith('pine_backup_')


class TestManifest:
    def test_get_manifest(self, app, client):
        """Reading manifest from a backup returns correct data."""
        from app.services.backup_service import _create_backup_inner
        with app.app_context():
            result = _create_backup_inner(app, include_audio=False)

        res = client.get(f'/api/backup/{result["filename"]}/manifest')
        assert res.status_code == 200
        manifest = res.get_json()
        assert manifest['backup_version'] == 2
        assert manifest['project_count'] == 0
        assert 'checksums' in manifest

    def test_manifest_not_found(self, client):
        """Requesting manifest for nonexistent backup returns 404."""
        res = client.get('/api/backup/nonexistent.zip/manifest')
        assert res.status_code == 404


class TestRestore:
    def _create_and_get_backup(self, app, client):
        """Helper: create a project with data, back it up, return filename + project data."""
        proj = _create_project(client, 'Restore Me')
        transcript_name = _seed_project_files(app, proj)
        _add_recording_to_db(app, proj['id'], transcript_name)

        from app.services.backup_service import _create_backup_inner
        with app.app_context():
            result = _create_backup_inner(app, include_audio=True)
        return result['filename'], proj

    def test_restore_skip_conflict(self, app, client):
        """Restore with skip strategy does not duplicate existing project."""
        filename, proj = self._create_and_get_backup(app, client)

        from app.services.backup_service import _restore_inner
        with app.app_context():
            result = _restore_inner(app, filename, None, False, 'skip')
        assert result['projects_restored'] == 0  # skipped because it already exists

    def test_restore_rename_conflict(self, app, client):
        """Restore with rename strategy creates a new project with _restored suffix."""
        filename, proj = self._create_and_get_backup(app, client)

        from app.models.project import Project
        from app.services.backup_service import _restore_inner
        with app.app_context():
            result = _restore_inner(app, filename, None, False, 'rename')
            assert result['projects_restored'] == 1
            # Should have 2 projects now
            projects = Project.query.all()
            assert len(projects) == 2
            folders = {p.folder_name for p in projects}
            assert proj['folder_name'] in folders
            restored = [f for f in folders if '_restored_' in f]
            assert len(restored) == 1

    def test_restore_overwrite_conflict(self, app, client):
        """Restore with overwrite strategy replaces the existing project."""
        filename, proj = self._create_and_get_backup(app, client)

        from app.models.project import Project
        from app.services.backup_service import _restore_inner
        with app.app_context():
            result = _restore_inner(app, filename, None, False, 'overwrite')
            assert result['projects_restored'] == 1
            projects = Project.query.all()
            assert len(projects) == 1
            assert projects[0].folder_name == proj['folder_name']

    def test_restore_selective(self, app, client):
        """Restore only selected projects."""
        proj1 = _create_project(client, 'Project A')
        _seed_project_files(app, proj1)
        _add_recording_to_db(app, proj1['id'])
        proj2 = _create_project(client, 'Project B')
        _seed_project_files(app, proj2)
        _add_recording_to_db(app, proj2['id'])

        from app.extensions import db
        from app.models.project import Project
        from app.services.backup_service import _create_backup_inner, _restore_inner
        with app.app_context():
            result = _create_backup_inner(app, include_audio=False)

            # Delete both projects to simulate fresh restore. delete_project
            # rmtree's the folder as well as the row, so drop the directories
            # too — leaving them behind would be a DB/disk desync, which the
            # restore now treats as a conflict in its own right.
            import shutil

            from app.models.setting import Setting
            projects_root = Setting.get('projects_path', app.config['DEFAULT_PROJECTS_PATH'])
            from app.models.recording import Recording
            for proj in (proj1, proj2):
                shutil.rmtree(os.path.join(projects_root, proj['folder_name']), ignore_errors=True)
            Recording.query.delete()
            Project.query.delete()
            db.session.commit()

            # Restore only project A
            restore_result = _restore_inner(app, result['filename'],
                                            [proj1['folder_name']], False, 'skip')
            assert restore_result['projects_restored'] == 1
            projects = Project.query.all()
            assert len(projects) == 1
            assert projects[0].name == 'Project A'

    def test_restore_treats_untracked_folder_as_a_conflict(self, app, client):
        """A folder with no DB row still blocks the target path.

        Left unhandled this raised FileExistsError from the staging loop
        instead of going through the chosen conflict strategy, which is the
        one scenario a restore is most likely to meet: the DB was reset or
        lost while the projects directory survived.
        """
        import shutil

        from app.extensions import db
        from app.models.project import Project
        from app.models.recording import Recording
        from app.models.setting import Setting
        from app.services.backup_service import _create_backup_inner, _restore_inner

        proj = _create_project(client, 'Orphan Proj')
        _seed_project_files(app, proj)
        _add_recording_to_db(app, proj['id'])

        with app.app_context():
            result = _create_backup_inner(app, include_audio=False)
            projects_root = Setting.get('projects_path', app.config['DEFAULT_PROJECTS_PATH'])
            project_dir = os.path.join(projects_root, proj['folder_name'])

            # Drop only the rows: the folder stays behind, untracked.
            Recording.query.delete()
            Project.query.delete()
            db.session.commit()
            assert os.path.isdir(project_dir)

            # 'skip' must decline rather than crash on the orphan.
            skipped = _restore_inner(app, result['filename'],
                                     [proj['folder_name']], False, 'skip')
            assert skipped['projects_restored'] == 0
            assert Project.query.count() == 0

            # 'rename' restores alongside it, leaving the orphan untouched.
            marker = os.path.join(project_dir, 'untracked.txt')
            with open(marker, 'w') as fh:
                fh.write('not in the database')

            renamed = _restore_inner(app, result['filename'],
                                     [proj['folder_name']], False, 'rename')
            assert renamed['projects_restored'] == 1
            assert os.path.isfile(marker), 'rename must not disturb the orphan'
            restored = Project.query.one()
            assert restored.folder_name != proj['folder_name']
            assert os.path.isdir(os.path.join(projects_root, restored.folder_name))

            shutil.rmtree(project_dir, ignore_errors=True)

    def test_restore_settings(self, app, client):
        """Restore settings from backup (excluding sensitive keys)."""
        from app.models.setting import Setting
        with app.app_context():
            Setting.set('font_size', '17')
            Setting.set('theme', 'dark')
            Setting.set('hf_token', 'secret-token')

        from app.services.backup_service import _create_backup_inner, _restore_inner
        with app.app_context():
            result = _create_backup_inner(app, include_audio=False)

            # Change settings
            Setting.set('font_size', '12')
            Setting.set('theme', 'light')

            # Restore with settings
            _restore_inner(app, result['filename'], [], True, 'skip')

            # font_size and theme should be restored
            assert Setting.get('font_size') == '17'
            assert Setting.get('theme') == 'dark'
            # hf_token should NOT be restored (sensitive)
            assert Setting.get('hf_token') == 'secret-token'  # unchanged

    def test_restore_extracts_files(self, app, client):
        """Restore extracts transcript and annotation files to disk."""
        filename, proj = self._create_and_get_backup(app, client)

        from app.extensions import db
        from app.models.project import Project
        from app.models.recording import Recording
        from app.services.backup_service import _restore_inner
        with app.app_context():
            projects_root = app.config['DEFAULT_PROJECTS_PATH']
            # Delete existing project
            Recording.query.delete()
            Project.query.delete()
            db.session.commit()
            import shutil
            project_dir = os.path.join(projects_root, proj['folder_name'])
            if os.path.isdir(project_dir):
                shutil.rmtree(project_dir)

            _restore_inner(app, filename, None, False, 'skip')

            # Files should be back on disk
            assert os.path.isfile(os.path.join(project_dir, 'project_tags.json'))
            assert os.path.isfile(os.path.join(project_dir, 'test_audio_transcript.json'))
            assert os.path.isfile(os.path.join(project_dir, 'test_audio_annotations.json'))
            assert os.path.isfile(os.path.join(project_dir, 'test_audio.mp3'))  # audio was in backup

    def test_restore_linked_recording_with_audio_becomes_local_copy(self, app, client, temp_dir):
        """Linked audio included in backup restores as a local project file."""
        proj = _create_project(client, 'Linked Restore')
        audio_bytes = b'\xff\xfb\x90\x00' * 128
        linked_audio = os.path.join(temp_dir, 'linked_source.mp3')
        with open(linked_audio, 'wb') as f:
            f.write(audio_bytes)

        from app.extensions import db
        from app.models.project import Project
        from app.models.recording import Recording
        from app.models.setting import Setting
        from app.services.backup_service import _create_backup_inner, _restore_inner

        with app.app_context():
            db.session.add(Recording(
                project_id=proj['id'],
                original_name='linked_source.mp3',
                stored_name=linked_audio,
                file_format='mp3',
                file_size_bytes=len(audio_bytes),
                duration_seconds=42.0,
                transcription_status='pending',
                is_linked=True,
            ))
            db.session.commit()

            result = _create_backup_inner(app, include_audio=True)

            projects_root = Setting.get('projects_path', app.config['DEFAULT_PROJECTS_PATH'])
            project_dir = os.path.join(projects_root, proj['folder_name'])
            Recording.query.delete()
            Project.query.delete()
            db.session.commit()

            import shutil
            if os.path.isdir(project_dir):
                shutil.rmtree(project_dir)

            restore_result = _restore_inner(app, result['filename'], None, False, 'skip')
            assert restore_result['projects_restored'] == 1

            restored_project = Project.query.filter_by(folder_name=proj['folder_name']).one()
            restored_recording = Recording.query.filter_by(project_id=restored_project.id).one()
            assert restored_recording.is_linked is False
            assert not os.path.isabs(restored_recording.stored_name)

            restored_audio = os.path.join(
                projects_root,
                restored_project.folder_name,
                restored_recording.stored_name,
            )
            assert os.path.isfile(restored_audio)
            with open(restored_audio, 'rb') as f:
                assert f.read() == audio_bytes


class TestDeleteBackup:
    def test_delete_backup(self, app, client):
        """Deleting a backup removes the file."""
        from app.services.backup_service import _create_backup_inner
        with app.app_context():
            result = _create_backup_inner(app, include_audio=False)

        res = client.delete(f'/api/backup/{result["filename"]}')
        assert res.status_code == 200

        # Should be gone
        res = client.get('/api/backup')
        assert res.get_json() == []

    def test_delete_nonexistent(self, client):
        """Deleting nonexistent backup returns 404."""
        res = client.delete('/api/backup/nope.zip')
        assert res.status_code == 404


class TestPruneBackups:
    def test_prune_keeps_n(self, app, client):
        """Prune removes older backups beyond retention count."""
        import json

        from app.models.setting import Setting
        from app.services.backup_service import (
            BACKUP_VERSION,
            _backup_dir,
            list_backups,
            prune_backups,
        )

        with app.app_context():
            Setting.set('backup_retention_count', '2')
            backup_path = _backup_dir(app)

            # Create 3 backup files manually with distinct names
            for i in range(3):
                fname = f'pine_backup_2026-01-0{i+1}T00-00-00.zip'
                fpath = os.path.join(backup_path, fname)
                manifest = {'backup_version': BACKUP_VERSION, 'created_at': f'2026-01-0{i+1}T00:00:00+00:00',
                            'include_audio': False, 'project_count': 0, 'recording_count': 0,
                            'projects': [], 'checksums': {}}
                with zipfile.ZipFile(fpath, 'w') as zf:
                    zf.writestr('manifest.json', json.dumps(manifest))

            prune_backups(app)
            backups = list_backups(app)
            assert len(backups) == 2

    def test_prune_default_keeps_ten(self, app, client):
        """With no retention saved, prune keeps 10 and Settings reports 10."""
        import json

        from app.services.backup_service import (
            BACKUP_VERSION,
            _backup_dir,
            list_backups,
            prune_backups,
        )

        with app.app_context():
            backup_path = _backup_dir(app)
            for i in range(12):
                fname = f'pine_backup_2026-01-{i+1:02d}T00-00-00.zip'
                manifest = {'backup_version': BACKUP_VERSION, 'created_at': f'2026-01-{i+1:02d}T00:00:00+00:00',
                            'include_audio': False, 'project_count': 0, 'recording_count': 0,
                            'projects': [], 'checksums': {}}
                with zipfile.ZipFile(os.path.join(backup_path, fname), 'w') as zf:
                    zf.writestr('manifest.json', json.dumps(manifest))

            prune_backups(app)
            names = sorted(b['filename'] for b in list_backups(app))
            assert len(names) == 10
            assert names[0] == 'pine_backup_2026-01-03T00-00-00.zip'

        assert client.get('/api/settings').get_json()['backup_retention_count'] == '10'


class TestUploadBackup:
    def test_upload_valid_backup(self, app, client):
        """Upload a valid backup ZIP."""
        from app.services.backup_service import _create_backup_inner
        with app.app_context():
            result = _create_backup_inner(app, include_audio=False)
            backup_dir = os.path.join(app.config['ROOT_DIR'], 'backups')
            zip_path = os.path.join(backup_dir, result['filename'])

        # Read the ZIP and upload it with a different name
        with open(zip_path, 'rb') as f:
            data = {'file': (f, 'uploaded_backup.zip')}
            res = client.post('/api/backup/upload', data=data, content_type='multipart/form-data')
        assert res.status_code == 200
        assert res.get_json()['ok'] is True

    def test_upload_invalid_file(self, client):
        """Upload a non-ZIP file is rejected."""
        import io
        data = {'file': (io.BytesIO(b'not a zip'), 'bad.zip')}
        res = client.post('/api/backup/upload', data=data, content_type='multipart/form-data')
        assert res.status_code == 400

    def test_upload_rejects_checksum_mismatch(self, app, client):
        """Upload rejects archives whose content no longer matches the manifest checksum."""
        from app.services.backup_service import _create_backup_inner

        with app.app_context():
            result = _create_backup_inner(app, include_audio=False)
            backup_dir = os.path.join(app.config['ROOT_DIR'], 'backups')
            zip_path = os.path.join(backup_dir, result['filename'])

        tampered_path = os.path.join(backup_dir, 'tampered.zip')
        with zipfile.ZipFile(zip_path, 'r') as src, zipfile.ZipFile(tampered_path, 'w') as dst:
            for name in src.namelist():
                data = src.read(name)
                if name == 'db/projects.json':
                    # Corrupt in place, preserving byte length: the manifest
                    # size check is cheaper and runs first, so a length change
                    # would trip that instead and the checksum comparison
                    # this test exists for would never run.
                    data = data[::-1]
                dst.writestr(name, data)

        with open(tampered_path, 'rb') as fh:
            data = {'file': (fh, 'tampered.zip')}
            res = client.post('/api/backup/upload', data=data, content_type='multipart/form-data')

        assert res.status_code == 400
        assert 'checksum' in res.get_json()['error'].lower()


class TestBackupAPI:
    def test_create_endpoint(self, app, client):
        """POST /api/backup returns ok and starts backup."""
        res = client.post('/api/backup', json={})
        assert res.status_code == 200
        assert res.get_json()['ok'] is True
