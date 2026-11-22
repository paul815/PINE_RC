"""The safety-snapshot folder in the system temp dir is the user's alone.

Snapshots before Reset and project delete hold every project with its audio.
They live in <tmp>/pine_safety_backups, created with default permissions —
on Linux, /tmp is shared and the folder came out world-readable.
"""

import os
import stat
import sys
import tempfile

import pytest


@pytest.fixture
def temp_root(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    return tmp_path


def test_the_folder_is_restricted_to_its_owner(temp_root, monkeypatch):
    from app.services import backup_service

    modes = []
    real_chmod = os.chmod
    monkeypatch.setattr(backup_service.os, 'chmod',
                        lambda p, m: (modes.append((p, m)), real_chmod(p, m))[1])

    path = backup_service._external_safety_backup_dir()

    assert (path, 0o700) in modes


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX permission bits')
def test_an_existing_open_folder_is_tightened(temp_root):
    from app.services import backup_service

    existing = temp_root / 'pine_safety_backups'
    existing.mkdir(mode=0o777)
    os.chmod(existing, 0o777)

    path = backup_service._external_safety_backup_dir()

    assert stat.S_IMODE(os.stat(path).st_mode) & 0o077 == 0
