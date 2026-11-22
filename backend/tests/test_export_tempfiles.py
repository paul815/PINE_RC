"""Exports leave no copy of project data behind in the temp folder.

The transfer ZIP and every ODT were written to NamedTemporaryFile(delete=False)
and never removed — each export left another copy of transcripts in %TEMP%.
Both are small (the transfer package carries no audio), so they are now built
in memory. The transfer package also only reads files inside the project.
"""

import io
import os
import tempfile
import zipfile

import pytest


@pytest.fixture
def watched_tempdir(tmp_path, monkeypatch):
    path = tmp_path / 'system_temp'
    path.mkdir()
    monkeypatch.setattr(tempfile, 'tempdir', str(path))
    return path


def test_odt_export_writes_nothing_to_temp(watched_tempdir):
    from app.services.export_service import _markdown_to_odt

    data = _markdown_to_odt('# Interview\n\nSome text.')

    assert zipfile.ZipFile(io.BytesIO(data)).read('mimetype') == b'application/vnd.oasis.opendocument.text'
    assert os.listdir(watched_tempdir) == []


def test_transfer_package_writes_nothing_to_temp(client, project_with_recording, watched_tempdir):
    pid, _, _ = project_with_recording({'segments': [{'start': 0, 'end': 1, 'text': 'hi'}]})

    res = client.get(f'/api/projects/{pid}/transfer')
    body = res.get_data()
    res.close()

    assert res.status_code == 200
    assert 'transcripts/rec_transcript.json' in zipfile.ZipFile(io.BytesIO(body)).namelist()
    assert os.listdir(watched_tempdir) == []


def test_transfer_package_never_reads_outside_the_project(client, project_with_recording, tmp_path):
    secret = tmp_path / 'id_rsa.json'
    secret.write_text('{"private": "key"}', encoding='utf-8')
    pid, _, _ = project_with_recording(transcript_path=str(secret))

    res = client.get(f'/api/projects/{pid}/transfer')
    archive = zipfile.ZipFile(io.BytesIO(res.get_data()))

    assert res.status_code == 200
    assert all(b'private' not in archive.read(n) for n in archive.namelist())
