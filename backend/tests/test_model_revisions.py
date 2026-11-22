"""Every model PINE downloads comes from a fixed commit, not whatever main is today.

A model repo is code-adjacent: pyannote and GLiNER ship pickle weights, and
GLiNER and the WeSpeaker embedding live on accounts PINE does not control. With
no revision, a single pushed commit reached every new install. The registry now
names the commit each model was verified at; downloads ask for exactly that.
"""

import os
import re

import pytest

SHA = re.compile(r'^[0-9a-f]{40}$')


def test_every_registry_model_names_a_commit():
    from app.services.model_manager import MODEL_REGISTRY

    for model_id, info in MODEL_REGISTRY.items():
        assert SHA.match(info.get('revision', '')), f'{model_id} has no pinned revision'
        if info.get('tokenizer_repo'):
            assert SHA.match(info.get('tokenizer_revision', '')), \
                f'{model_id} pins its weights but not its tokenizer'


class _Hub:
    """Records hub calls; snapshot_download lays out a real HF cache entry."""

    def __init__(self):
        self.calls = []

    def hf_hub_download(self, **kwargs):
        self.calls.append(('file', kwargs))
        return os.path.join(kwargs['local_dir'], kwargs['filename'])

    def snapshot_download(self, **kwargs):
        self.calls.append(('snapshot', kwargs))
        if 'cache_dir' in kwargs:
            folder = 'models--' + kwargs['repo_id'].replace('/', '--')
            snap = os.path.join(kwargs['cache_dir'], folder, 'snapshots', kwargs['revision'])
            os.makedirs(snap, exist_ok=True)
            with open(os.path.join(snap, 'config.yaml'), 'w') as fh:
                fh.write('pipeline: {}\n')
            return snap
        return kwargs['local_dir']


@pytest.fixture
def hub():
    return _Hub()


class TestDownloadsUseThePinnedCommit:
    def test_a_local_dir_model(self, hub, tmp_path):
        from app.services.model_manager import MODEL_REGISTRY, _fetch_model_files

        info = MODEL_REGISTRY['gliner-pii']
        path = _fetch_model_files(info, str(tmp_path / 'gliner'), str(tmp_path / 'pyc'), 'tok',
                                  hub.hf_hub_download, hub.snapshot_download)

        kind, kwargs = hub.calls[0]
        assert kind == 'snapshot'
        assert kwargs['revision'] == info['revision']
        assert kwargs['local_dir'] == path

    def test_a_single_file_model(self, hub, tmp_path):
        from app.services.model_manager import _fetch_model_files

        info = {'repo_id': 'org/m', 'filename': 'm.bin', 'revision': 'a' * 40}
        _fetch_model_files(info, str(tmp_path / 'm'), str(tmp_path / 'pyc'), None,
                           hub.hf_hub_download, hub.snapshot_download)

        assert hub.calls[0][1]['revision'] == 'a' * 40

    def test_a_pyannote_hub_cache_model_is_findable_offline_afterwards(self, hub, tmp_path):
        """A download by commit writes no refs/main, yet pyannote and the
        'already downloaded?' check both resolve 'main' in that cache. Without
        the ref the model would be fetched again on every launch and could not
        be found offline at all."""
        from app.services.model_manager import (
            MODEL_REGISTRY,
            _fetch_model_files,
            snapshot_dir_if_repo_cached,
        )

        info = MODEL_REGISTRY['pyannote-diarization']
        pyc = str(tmp_path / 'pyannote_cache')

        path = _fetch_model_files(info, pyc, pyc, 'tok', hub.hf_hub_download, hub.snapshot_download)

        kind, kwargs = hub.calls[0]
        assert kwargs['revision'] == info['revision'] and kwargs['cache_dir'] == pyc
        assert snapshot_dir_if_repo_cached(info['repo_id'], pyc) == path
        assert path.endswith(info['revision'])

    def test_tokenizer_files_come_from_their_pinned_commit(self, hub, tmp_path):
        from app.services.model_manager import MODEL_REGISTRY, _fetch_tokenizer_files

        info = MODEL_REGISTRY['gliner-pii']
        _fetch_tokenizer_files(info, str(tmp_path), 'tok', hub.hf_hub_download)

        assert hub.calls, 'no tokenizer files were requested'
        assert {kw['revision'] for _, kw in hub.calls} == {info['tokenizer_revision']}
        assert {kw['repo_id'] for _, kw in hub.calls} == {info['tokenizer_repo']}
