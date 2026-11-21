"""Shutdown flush, align-model loading offline, GPU cache after diarization, torchcodec shim.

Four small fixes found reading the 2026-11-21 transcription logs from Windows
and the Mac; none of them changes a transcript.
"""
import contextlib
import sys
import threading
import types

import pytest


# ── SQLite flush on shutdown ──

def _wal_app(tmp_path):
    from flask import Flask

    from app.extensions import db

    app = Flask('flush-test')
    app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{(tmp_path / 't.db').as_posix()}"
    db.init_app(app)
    with app.app_context():
        with db.engine.connect() as conn:
            conn.execute(db.text('PRAGMA journal_mode=WAL'))
            conn.execute(db.text('CREATE TABLE t (x INTEGER)'))
            conn.execute(db.text('INSERT INTO t VALUES (1)'))
            conn.commit()
    return app


def test_flush_reaches_the_database_from_a_bare_thread(tmp_path, monkeypatch):
    # Every graceful_exit caller is a daemon thread with no app context; that
    # used to raise "Working outside of application context" and skip the flush.
    from app import shutdown

    app = _wal_app(tmp_path)
    wal = tmp_path / 't.db-wal'
    assert wal.exists() and wal.stat().st_size > 0
    monkeypatch.setattr(shutdown, '_app', None)
    shutdown.register_app(app)

    result = {}
    t = threading.Thread(target=lambda: result.update(ok=shutdown.flush_database()))
    t.start()
    t.join(10)

    assert result['ok'] is True
    assert not wal.exists() or wal.stat().st_size == 0


def test_flush_without_an_app_reports_instead_of_raising(monkeypatch):
    from app import shutdown

    monkeypatch.setattr(shutdown, '_app', None)
    result = {}
    t = threading.Thread(target=lambda: result.update(ok=shutdown.flush_database()))
    t.start()
    t.join(10)
    assert result['ok'] is False


def test_create_app_registers_itself_for_shutdown(app):
    from app import shutdown
    assert shutdown._app is app


# ── align model: local cache first ──

def _engine(hf_offline=True):
    from ml_worker.engines.whisperx_engine import WhisperXEngine
    return WhisperXEngine(env=types.SimpleNamespace(hf_offline=hf_offline))


def _network_recorder(monkeypatch):
    from ml_worker import compat
    state = {'open': False, 'opened': 0}

    @contextlib.contextmanager
    def fake_allow(hf_offline):
        state['open'] = True
        state['opened'] += 1
        try:
            yield
        finally:
            state['open'] = False

    monkeypatch.setattr(compat, 'allow_hf_network_for_align', fake_allow)
    monkeypatch.setattr(compat, 'hf_network_allowed', lambda: False)
    return state


def test_cached_align_model_loads_without_opening_the_network(monkeypatch):
    net = _network_recorder(monkeypatch)
    calls = []

    class FakeWhisperX:
        @staticmethod
        def load_align_model(language_code, device, model_name=None):
            calls.append(net['open'])
            return object(), {'language': language_code}

    model, meta = _engine()._get_align_model(FakeWhisperX, 'en', 'cpu')
    assert meta['language'] == 'en'
    assert calls == [False]
    assert net['opened'] == 0


def test_missing_align_model_is_fetched_once_with_the_network_open(monkeypatch):
    net = _network_recorder(monkeypatch)
    calls = []

    class FakeWhisperX:
        @staticmethod
        def load_align_model(language_code, device, model_name=None):
            calls.append(net['open'])
            if not net['open']:
                raise OSError('offline and not in the cache')
            return object(), {'language': language_code}

    model, meta = _engine()._get_align_model(FakeWhisperX, 'ru', 'cpu')
    assert meta['language'] == 'ru'
    assert calls == [False, True]
    assert net['opened'] == 1


def test_online_install_loads_once_as_before(monkeypatch):
    net = _network_recorder(monkeypatch)
    calls = []

    class FakeWhisperX:
        @staticmethod
        def load_align_model(language_code, device, model_name=None):
            calls.append(net['open'])
            return object(), {'language': language_code}

    _engine(hf_offline=False)._get_align_model(FakeWhisperX, 'en', 'cpu')
    assert calls == [False]
    assert net['opened'] == 0


# ── GPU cache after diarization ──

@pytest.mark.parametrize('device', ['cuda', 'mps', 'cpu'])
def test_diarizer_returns_the_allocator_cache(monkeypatch, device):
    from ml_worker.diarize import Diarizer

    emptied = []
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True,
                                   empty_cache=lambda: emptied.append('cuda')),
        mps=types.SimpleNamespace(empty_cache=lambda: emptied.append('mps')),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    d = Diarizer.__new__(Diarizer)
    d.device = device
    d._return_cached_memory()
    assert emptied == ([] if device == 'cpu' else [device])


# ── torchcodec shim ──

def test_torchcodec_shim_exports_every_name_pyannote_imports():
    # pyannote.audio.core.io: `from torchcodec import AudioSamples` and
    # `from torchcodec.decoders import AudioDecoder, AudioStreamMetadata`.
    # A missing name fails the whole import and pyannote drops its decoder.
    from ml_worker import compat

    def tc_keys():
        return [k for k in sys.modules if k == 'torchcodec' or k.startswith('torchcodec.')]

    saved = {k: sys.modules[k] for k in tc_keys()}
    sys.modules['torchcodec'] = None        # the real one unusable, as on the Mac
    try:
        compat.stub_torchcodec()
        from torchcodec import AudioSamples  # noqa: F401
        from torchcodec.decoders import AudioDecoder, AudioStreamMetadata  # noqa: F401
        assert getattr(sys.modules['torchcodec'], '_pine_stub', False)
    finally:
        for k in tc_keys():
            del sys.modules[k]
        sys.modules.update(saved)


def test_mac_engine_installs_the_shim_before_whisperx(monkeypatch):
    from ml_worker import compat
    from ml_worker.engines import whisperx_engine

    order = []
    monkeypatch.setattr(sys, 'platform', 'darwin')
    monkeypatch.setattr(compat, 'stub_torchcodec', lambda: order.append('shim'))
    monkeypatch.setattr(whisperx_engine, 'detect_torch_device', lambda: ('cpu', 'int8'))
    fake = types.ModuleType('whisperx')
    fake.load_model = lambda *a, **k: order.append('whisperx') or object()
    monkeypatch.setitem(sys.modules, 'whisperx', fake)

    eng = whisperx_engine.WhisperXEngine(env=types.SimpleNamespace(model_dir='m', hf_offline=False))
    eng.load()
    assert order == ['shim', 'whisperx']
