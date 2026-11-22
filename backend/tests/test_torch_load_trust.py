"""The worker's torch.load patch trusts PINE's own checkpoints, not every download.

pyannote's Lightning checkpoints need weights_only=False under torch 2.6+, so
the worker forced it on every torch.load in the process. That also covered the
alignment models whisperx fetches on first use of a language — from personal
Hugging Face accounts (anton-l, jonatasgrosman), stored as pytorch_model.bin,
i.e. pickle — so one pushed commit to such a repo meant code execution.

Now the relaxation skips files in the shared download caches (the Hugging Face
hub cache, torch.hub). transformers asks for weights_only=True there and gets
it; pyannote, loading from PINE's own models folder, is untouched.
"""

import io
import os
import sys
import types

import pytest


class NotATensor:
    """Anything a weights_only unpickler must refuse."""

    def __init__(self):
        self.payload = 'would have been code'


@pytest.fixture
def caches(tmp_path, monkeypatch):
    """A fake HF hub cache, torch.hub dir and PINE models folder."""
    import huggingface_hub.constants as hf_constants

    roots = {
        'hf': tmp_path / 'hf_hub',
        'torch_hub': tmp_path / 'torch_hub',
        'models': tmp_path / 'pine_models',
    }
    for root in roots.values():
        root.mkdir()
    monkeypatch.setattr(hf_constants, 'HF_HUB_CACHE', str(roots['hf']))
    monkeypatch.setenv('PYANNOTE_CACHE', str(roots['models'] / 'pyannote_cache'))
    monkeypatch.delenv('PINE_TORCH_STRICT_WEIGHTS_ONLY', raising=False)
    return roots


@pytest.fixture
def fake_torch(monkeypatch, caches):
    """A torch whose load() records the weights_only it was called with."""
    from ml_worker import compat

    calls = []
    torch = types.ModuleType('torch')
    hub = types.ModuleType('torch.hub')
    hub.get_dir = lambda: str(caches['torch_hub'])
    torch.hub = hub

    def load(f, *args, **kwargs):
        calls.append(kwargs.get('weights_only', 'unset'))
        return 'checkpoint'

    torch.load = load
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'torch.hub', hub)
    monkeypatch.setattr(compat, '_TORCH_LOAD_TRUST_PATCHED', False)
    compat.patch_torch_load_for_trusted_checkpoints()
    return torch, calls


class TestWhichLoadsAreRelaxed:
    def test_pyannote_checkpoint_in_the_models_folder_is_relaxed(self, fake_torch, caches):
        torch, calls = fake_torch

        torch.load(str(caches['models'] / 'pyannote_cache' / 'm' / 'pytorch_model.bin'))

        assert calls == [False]

    def test_a_stream_without_a_name_keeps_the_old_behaviour(self, fake_torch):
        """Lightning may hand torch an open stream; unknown origin stays as before."""
        torch, calls = fake_torch

        torch.load(io.BytesIO(b''))

        assert calls == [False]

    def test_hf_cache_load_keeps_what_the_caller_asked_for(self, fake_torch, caches):
        torch, calls = fake_torch
        align = caches['hf'] / 'models--anton-l--wav2vec2-large-xlsr-53-russian' / 'pytorch_model.bin'

        torch.load(str(align), weights_only=True)
        torch.load(str(align))

        assert calls == [True, 'unset']

    def test_hf_cache_stream_with_a_name_is_not_relaxed(self, fake_torch, caches):
        torch, calls = fake_torch
        path = caches['hf'] / 'align.bin'
        path.write_bytes(b'')

        with open(path, 'rb') as fh:
            torch.load(fh)

        assert calls == ['unset']

    def test_torch_hub_download_is_not_relaxed(self, fake_torch, caches):
        torch, calls = fake_torch

        torch.load(os.path.join(caches['torch_hub'], 'checkpoints', 'wav2vec2.pth'))

        assert calls == ['unset']

    def test_pyannote_cache_inside_the_hf_cache_is_still_trusted(self, fake_torch, caches, monkeypatch):
        torch, calls = fake_torch
        inside = caches['hf'] / 'pyannote_cache'
        monkeypatch.setenv('PYANNOTE_CACHE', str(inside))

        torch.load(str(inside / 'm' / 'pytorch_model.bin'))

        assert calls == [False]


class TestWithRealTorch:
    """The point of it all, with the real unpickler: a pickle in the HF cache
    cannot run, a pickle in PINE's models folder still loads."""

    @pytest.fixture
    def real_torch(self, monkeypatch, caches):
        torch = pytest.importorskip('torch')
        from ml_worker import compat

        monkeypatch.setattr(torch, 'load', torch.load)   # restored at teardown
        monkeypatch.setattr(compat, '_TORCH_LOAD_TRUST_PATCHED', False)
        monkeypatch.setattr(torch.hub, 'get_dir', lambda: str(caches['torch_hub']))
        compat.patch_torch_load_for_trusted_checkpoints()
        return torch

    def test_a_pickle_payload_in_the_hf_cache_is_refused(self, real_torch, caches):
        import pickle

        path = caches['hf'] / 'models--someone--align' / 'pytorch_model.bin'
        path.parent.mkdir(parents=True)
        real_torch.save({'w': NotATensor()}, path)

        with pytest.raises(pickle.UnpicklingError):
            real_torch.load(str(path), weights_only=True)
        with pytest.raises(pickle.UnpicklingError):
            real_torch.load(str(path))

    def test_plain_weights_in_the_hf_cache_still_load(self, real_torch, caches):
        path = caches['hf'] / 'state_dict.bin'
        real_torch.save({'w': real_torch.zeros(2)}, path)

        assert real_torch.load(str(path), weights_only=True)['w'].shape == (2,)

    def test_a_pyannote_style_checkpoint_in_the_models_folder_loads(self, real_torch, caches):
        path = caches['models'] / 'pyannote_cache' / 'segmentation.bin'
        path.parent.mkdir(parents=True)
        real_torch.save({'hparams': NotATensor()}, path)

        assert isinstance(real_torch.load(str(path))['hparams'], NotATensor)
