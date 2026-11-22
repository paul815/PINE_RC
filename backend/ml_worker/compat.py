"""Runtime shims for the ML dependency stack.

Every patch here mutates process-global state (env vars, ``sys.modules``,
``torch.load``). That is exactly why the ML pipeline runs in its own worker
process: these shims never leak into the Flask backend.

The exception is onboarding's pyannote warm-up, which imports the same stack
inside Flask (``app.services.model_manager``). It borrows the attribute-level
shims — the ones that only add back an API a dependency dropped — and the
scoped ``trusted_torch_load`` below, never the permanent ``torch.load`` patch.

All patches are defensive no-ops when the condition they guard against is
absent, so they are safe to keep across dependency upgrades.
"""

import logging
import os
import sys
import types
from contextlib import contextmanager

log = logging.getLogger(__name__)


def ensure_ca_bundle():
    """Give urllib a CA bundle on a Mac whose Python came without one.

    python.org's Python ships its own OpenSSL and no certificates until its
    "Install Certificates.command" is run; every urllib download then fails
    with CERTIFICATE_VERIFY_FAILED. whisperx's alignment fetches NLTK's
    punkt_tab that way when it is missing, and the failure turns alignment off
    for the job (mac-bench, 2026-11-21). Onboarding puts punkt_tab in the venv;
    this keeps the fallback fetch working too. Leaves a set SSL_CERT_FILE alone.
    """
    if sys.platform != 'darwin' or os.environ.get('SSL_CERT_FILE'):
        return
    try:
        import certifi
        os.environ['SSL_CERT_FILE'] = certifi.where()
    except ImportError:
        if os.path.isfile('/etc/ssl/cert.pem'):
            os.environ['SSL_CERT_FILE'] = '/etc/ssl/cert.pem'


def disable_model_telemetry():
    """Disable telemetry/analytics toggles for model libraries before imports."""
    os.environ['PYANNOTE_METRICS_ENABLED'] = 'false'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['DISABLE_TELEMETRY'] = '1'
    os.environ['DO_NOT_TRACK'] = '1'
    os.environ['WANDB_DISABLED'] = 'true'
    os.environ['OTEL_SDK_DISABLED'] = 'true'
    os.environ['OTEL_METRICS_EXPORTER'] = 'none'
    os.environ['OTEL_TRACES_EXPORTER'] = 'none'
    os.environ['OTEL_LOGS_EXPORTER'] = 'none'

    # pyannote.audio 4.x creates its OTLP telemetry exporter during module
    # import. Replace that module up front so no outbound telemetry can start.
    telemetry_stub = types.ModuleType('pyannote.audio.telemetry')

    def _noop(*args, **kwargs):
        return None

    telemetry_stub.set_opentelemetry_log_level = _noop
    telemetry_stub.set_telemetry_metrics = _noop
    telemetry_stub.track_model_init = _noop
    telemetry_stub.track_pipeline_init = _noop
    telemetry_stub.track_pipeline_apply = _noop
    sys.modules['pyannote.audio.telemetry'] = telemetry_stub
    sys.modules['pyannote.audio.telemetry.metrics'] = telemetry_stub

    try:
        import onnxruntime as ort
        ort.disable_telemetry_events()
    except Exception:
        pass


def patch_torchaudio_for_pyannote():
    """Shim torchaudio APIs removed in 2.9+ that older pyannote code paths expect.

    No-op on torchaudio <= 2.8 where both attributes still exist.
    """
    try:
        import torchaudio
    except ImportError:
        return

    if not hasattr(torchaudio, 'AudioMetaData'):
        from typing import NamedTuple

        class AudioMetaData(NamedTuple):
            sample_rate: int
            num_frames: int
            num_channels: int
            bits_per_sample: int
            encoding: str

        torchaudio.AudioMetaData = AudioMetaData  # type: ignore[attr-defined]
        log.debug('Patched torchaudio.AudioMetaData for pyannote compatibility')

    if not hasattr(torchaudio, 'list_audio_backends'):
        def list_audio_backends():
            # Order matches pyannote: prefer soundfile when installed, else ffmpeg/sox.
            names = []
            try:
                import soundfile as _sf  # noqa: F401
                names.append('soundfile')
            except ImportError:
                pass
            names.append('ffmpeg')
            return names

        torchaudio.list_audio_backends = list_audio_backends  # type: ignore[attr-defined]
        log.debug('Patched torchaudio.list_audio_backends for pyannote compatibility')


_HF_HUB_LEGACY_AUTH_PATCHED = False


def patch_hf_hub_legacy_use_auth_token():
    """Map use_auth_token -> token on hf_hub_download (legacy callers + new huggingface_hub)."""
    global _HF_HUB_LEGACY_AUTH_PATCHED
    if _HF_HUB_LEGACY_AUTH_PATCHED:
        return
    try:
        import huggingface_hub
        import huggingface_hub.file_download as fd
    except ImportError:
        return

    _orig = fd.hf_hub_download

    def _hf_hub_download(*args, **kwargs):
        legacy = kwargs.pop('use_auth_token', None)
        if legacy is not None and kwargs.get('token') is None:
            kwargs['token'] = legacy
        return _orig(*args, **kwargs)

    fd.hf_hub_download = _hf_hub_download  # type: ignore[assignment]
    huggingface_hub.hf_hub_download = _hf_hub_download  # type: ignore[assignment]
    _HF_HUB_LEGACY_AUTH_PATCHED = True
    log.debug('Patched hf_hub_download for legacy use_auth_token kwarg')


_HF_HUB_OFFLINE_MODE_PATCHED = False


def patch_hf_hub_is_offline_mode():
    """Restore ``huggingface_hub.is_offline_mode`` for callers that still import it.

    huggingface_hub moved the flag to ``constants.HF_HUB_OFFLINE`` and dropped the
    helper from the package namespace. Parts of the pyannote/speechbrain stack
    still do ``from huggingface_hub import is_offline_mode`` at import time, and
    that ImportError takes down the whole import — it is what made onboarding's
    diarization warm-up fail on macOS while the worker, which never hits that
    path, went on transcribing.

    Reads the constant on every call rather than snapshotting it, so
    ``apply_hf_offline`` and ``allow_hf_network_for_align`` stay authoritative.
    """
    global _HF_HUB_OFFLINE_MODE_PATCHED
    if _HF_HUB_OFFLINE_MODE_PATCHED:
        return
    try:
        import huggingface_hub
    except ImportError:
        return
    if hasattr(huggingface_hub, 'is_offline_mode'):
        _HF_HUB_OFFLINE_MODE_PATCHED = True
        return

    def is_offline_mode():
        try:
            import huggingface_hub.constants as _const
            return bool(getattr(_const, 'HF_HUB_OFFLINE', False))
        except Exception:
            return bool(os.environ.get('HF_HUB_OFFLINE', '').strip())

    huggingface_hub.is_offline_mode = is_offline_mode  # type: ignore[attr-defined]
    _HF_HUB_OFFLINE_MODE_PATCHED = True
    log.debug('Restored huggingface_hub.is_offline_mode for legacy importers')


_TORCH_LOAD_TRUST_PATCHED = False


def _checkpoint_path(f):
    """The filesystem path torch.load was given, directly or as an open file's name."""
    if isinstance(f, (str, os.PathLike)):
        return os.fspath(f)
    name = getattr(f, 'name', None)
    return name if isinstance(name, str) else None


def _is_inside(path, root):
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:  # different drives on Windows
        return False


def _shared_download_caches():
    """Caches any library fills from the internet on its own, outside PINE's models folder."""
    roots = []
    try:
        from huggingface_hub import constants as hf_constants
        roots.append(hf_constants.HF_HUB_CACHE)
    except Exception:
        pass
    try:
        import torch.hub
        roots.append(torch.hub.get_dir())
    except Exception:
        pass
    return [os.path.abspath(r) for r in roots if r]


def _from_shared_download_cache(f):
    """True for a checkpoint that came from the HF hub cache or torch.hub.

    Those hold what whisperx fetches on first use of a language — alignment
    models from personal Hugging Face accounts, as pickle — so they keep the
    caller's weights_only. PINE's own pyannote cache wins even if a user put
    it inside the HF cache. A stream with no name is not one of these: it keeps
    the old, relaxed behaviour so a pyannote load handed over as a file object
    still works.
    """
    path = _checkpoint_path(f)
    if not path:
        return False
    path = os.path.abspath(path)
    pyannote_cache = os.environ.get('PYANNOTE_CACHE', '').strip()
    if pyannote_cache and _is_inside(path, os.path.abspath(pyannote_cache)):
        return False
    return any(_is_inside(path, root) for root in _shared_download_caches())


def patch_torch_load_for_trusted_checkpoints():
    """Force weights_only=False on torch.load for pyannote/Lightning checkpoints.

    PyTorch 2.6+ tightened unpickling; pyannote checkpoints embed many custom
    types. PINE trusts the models it installed during onboarding into its own
    models folder, and this patch lives in the ML worker process only — the
    Flask backend keeps the strict default.

    It does not reach the shared download caches (see
    ``_from_shared_download_cache``): forcing it there let an alignment model
    from a personal HF account run code through its pickle. transformers asks
    for weights_only=True on those and now gets it.

    Set PINE_TORCH_STRICT_WEIGHTS_ONLY=1 to skip this patch (not recommended for pyannote).
    """
    global _TORCH_LOAD_TRUST_PATCHED
    if _TORCH_LOAD_TRUST_PATCHED:
        return
    if os.environ.get('PINE_TORCH_STRICT_WEIGHTS_ONLY', '').strip() == '1':
        return
    import functools

    import torch

    _orig = torch.load
    if getattr(_orig, '_pine_trusted_checkpoint_wrap', False):
        _TORCH_LOAD_TRUST_PATCHED = True
        return

    @functools.wraps(_orig)
    def _load(*args, **kwargs):
        f = args[0] if args else kwargs.get('f')
        if not _from_shared_download_cache(f):
            kwargs['weights_only'] = False
        return _orig(*args, **kwargs)

    _load._pine_trusted_checkpoint_wrap = True
    torch.load = _load
    _TORCH_LOAD_TRUST_PATCHED = True
    log.debug('torch.load → weights_only=False for trusted onboarding checkpoints '
              '(not for the shared HF / torch.hub caches)')


@contextmanager
def trusted_torch_load():
    """Relax ``torch.load`` for one checkpoint read, then put it back.

    Same reasoning as the permanent patch above, but for the Flask process,
    which loads a pyannote checkpoint exactly once — during the onboarding
    warm-up — and keeps the strict default for everything else it ever
    unpickles. A no-op when the permanent patch is already in place.
    """
    try:
        import torch
    except ImportError:
        yield
        return

    if os.environ.get('PINE_TORCH_STRICT_WEIGHTS_ONLY', '').strip() == '1':
        yield
        return

    orig = torch.load
    if getattr(orig, '_pine_trusted_checkpoint_wrap', False):
        yield
        return

    import functools

    @functools.wraps(orig)
    def _load(*args, **kwargs):
        kwargs['weights_only'] = False
        return orig(*args, **kwargs)

    _load._pine_trusted_checkpoint_wrap = True
    torch.load = _load
    try:
        yield
    finally:
        torch.load = orig


def stub_torchcodec():
    """Replace torchcodec with an ffmpeg-backed shim when its native libs are broken.

    pyannote-audio uses torchcodec.decoders.AudioDecoder for all audio I/O,
    including internal pipeline steps.  torchcodec's native lib requires FFmpeg
    shared libraries which may not be present (common on macOS).

    This injects a drop-in AudioDecoder that decodes via an ffmpeg subprocess.
    """
    import importlib.machinery

    if getattr(sys.modules.get('torchcodec'), '_pine_stub', False):
        return  # already stubbed

    # Try the real torchcodec — if native libs load OK, keep it.
    # A bare import is not enough: the package may install as a
    # pyannote-audio dependency but its native FFmpeg bindings can
    # fail at runtime.  We instantiate a throw-away decoder on a tiny
    # silent WAV to verify it actually works.
    try:
        import io
        import os as _os
        import struct
        import tempfile

        from torchcodec.decoders import AudioDecoder as _real
        # Build a minimal 16-bit PCM WAV (44 bytes header + 2 bytes data)
        _buf = io.BytesIO()
        _sr = 16000
        _nch = 1
        _bps = 16
        _nsamp = 1
        _data_sz = _nsamp * _nch * (_bps // 8)
        _buf.write(b'RIFF')
        _buf.write(struct.pack('<I', 36 + _data_sz))
        _buf.write(b'WAVEfmt ')
        _buf.write(struct.pack('<IHHIIHH', 16, 1, _nch, _sr, _sr * _nch * (_bps // 8), _nch * (_bps // 8), _bps))
        _buf.write(b'data')
        _buf.write(struct.pack('<I', _data_sz))
        _buf.write(b'\x00\x00')
        _tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        _tmp.write(_buf.getvalue())
        _tmp.close()
        try:
            _real(_tmp.name)        # actual native-lib smoke test
        finally:
            _os.unlink(_tmp.name)
        return  # real torchcodec works end-to-end, no shim needed
    except Exception:
        pass  # broken or missing — fall through to install shim

    # Remove any partially-loaded real torchcodec modules
    for key in list(sys.modules):
        if key == 'torchcodec' or key.startswith('torchcodec.'):
            del sys.modules[key]

    # --- Functional AudioDecoder backed by ffmpeg ---
    from collections import namedtuple

    _AudioStreamMetadata = namedtuple(
        'AudioStreamMetadata',
        ['sample_rate', 'num_channels', 'num_frames',
         'codec', 'bit_rate', 'duration_seconds_from_header',
         'begin_stream_seconds_from_header',
         'end_stream_seconds_from_header',
         'bit_depth'])

    _AudioSamples = namedtuple(
        'AudioSamples', ['data', 'pts_seconds', 'sample_rate'])

    class AudioDecoder:
        """ffmpeg-backed drop-in for torchcodec.decoders.AudioDecoder.

        Uses ffmpeg subprocess directly — never torchaudio — to avoid
        recursion (torchaudio may try its torchcodec backend which calls
        back into this shim).  Per-file caching ensures ffmpeg runs at
        most once per source file even when pyannote creates hundreds of
        AudioDecoder instances for chunk-level processing.
        """
        _cache = {}  # source_path → (waveform, sample_rate)

        def __init__(self, source, *args, **kwargs):
            source = str(source)
            if source not in AudioDecoder._cache:
                AudioDecoder._cache[source] = AudioDecoder._decode(source)
            self._waveform, self._sr = AudioDecoder._cache[source]
            self._num_frames = self._waveform.shape[-1]
            self._num_channels = self._waveform.shape[0]
            self._duration = self._num_frames / self._sr

        @staticmethod
        def _decode(path, target_sr=16000):
            """Decode audio to 16 kHz mono float32 tensor via ffmpeg."""
            import struct
            import subprocess

            import torch
            proc = subprocess.run(
                ['ffmpeg', '-y', '-i', path,
                 '-ar', str(target_sr), '-ac', '1',
                 '-f', 's16le', 'pipe:1'],
                capture_output=True, check=True,
            )
            raw = proc.stdout
            n = len(raw) // 2
            samples = struct.unpack(f'<{n}h', raw)
            waveform = (torch.tensor(samples, dtype=torch.float32)
                        .unsqueeze(0) / 32768.0)
            return waveform, target_sr

        @property
        def metadata(self):
            return _AudioStreamMetadata(
                sample_rate=self._sr,
                num_channels=self._num_channels,
                num_frames=self._num_frames,
                codec='unknown',
                bit_rate=0,
                duration_seconds_from_header=self._duration,
                begin_stream_seconds_from_header=0.0,
                end_stream_seconds_from_header=self._duration,
                bit_depth=16,
            )

        def get_all_samples(self):
            import torch
            pts = torch.arange(self._num_frames, dtype=torch.float64) / self._sr
            return _AudioSamples(
                data=self._waveform, pts_seconds=pts,
                sample_rate=self._sr)

        def get_samples_played_in_range(self, start_seconds=0.0,
                                        end_seconds=None):
            import torch
            s = max(0, int(start_seconds * self._sr))
            e = self._num_frames if end_seconds is None \
                else min(self._num_frames, int(end_seconds * self._sr))
            s = min(s, self._num_frames)
            e = max(s, e)
            chunk = self._waveform[:, s:e]
            pts = (torch.arange(e - s, dtype=torch.float64) / self._sr
                   + start_seconds)
            return _AudioSamples(
                data=chunk, pts_seconds=pts, sample_rate=self._sr)

    # --- Wire up the stub module hierarchy ---
    # Python 3.13+: importlib.util.find_spec() raises if the name is in
    # sys.modules but __spec__ is None (transformers checks torchcodec at import).
    def _pkg(name):
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(
            name, loader=None, is_package=True)
        m.__path__ = []
        return m

    def _mod(name):
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(
            name, loader=None, is_package=False)
        return m

    root = _pkg('torchcodec')
    root._pine_stub = True
    core = _pkg('torchcodec._core')
    core._pine_stub = True
    ops = _mod('torchcodec._core.ops')
    ops._pine_stub = True
    decoders = _pkg('torchcodec.decoders')
    decoders._pine_stub = True

    decoders.AudioDecoder = AudioDecoder
    # pyannote.audio.core.io does ``from torchcodec import AudioSamples`` at
    # import time and, when that fails, warns that built-in decoding is broken
    # and falls back. The shim decodes fine; it was only missing the name the
    # real package exports next to the decoder, so pyannote read our stub as a
    # broken install. Same namedtuple the decoder returns, so isinstance holds.
    root.AudioSamples = _AudioSamples
    decoders.AudioSamples = _AudioSamples
    # pyannote-audio 4 imports this one from torchcodec.decoders as well; without
    # it the whole import failed and pyannote set its decoder to None, shim or
    # no shim (checked 2026-11-21 with torchcodec forced off on Windows).
    decoders.AudioStreamMetadata = _AudioStreamMetadata
    root.decoders = decoders
    root._core = core
    core.ops = ops

    sys.modules.update({
        'torchcodec': root,
        'torchcodec._core': core,
        'torchcodec._core.ops': ops,
        'torchcodec.decoders': decoders,
    })


def clear_torchcodec_decode_cache():
    """Drop the shim AudioDecoder's per-file waveform cache (frees RAM after diarization)."""
    try:
        from torchcodec.decoders import AudioDecoder as _AD
        if hasattr(_AD, '_cache'):
            _AD._cache.clear()
    except Exception:
        pass


def hf_network_allowed():
    return os.environ.get('PINE_ALLOW_HF_NETWORK', '').strip() == '1'


def apply_hf_offline(hf_offline: bool):
    """Do not contact Hugging Face during transcription once onboarding finished.

    Pipeline config may still reference hub repo ids; those files must already
    be under the app's models dir or HF cache. Override with PINE_ALLOW_HF_NETWORK=1.

    Exception: while ``whisperx.load_align_model`` runs (first load per language),
    Hub is briefly allowed again so language-specific wav2vec2 weights can be
    cached; see ``allow_hf_network_for_align``.
    """
    if not hf_offline or hf_network_allowed():
        return
    os.environ['HF_HUB_OFFLINE'] = '1'
    # Also patch the in-memory constant — huggingface_hub reads the env var
    # at import time and won't see later changes to os.environ.
    try:
        import huggingface_hub.constants as _hf_const
        _hf_const.HF_HUB_OFFLINE = True
    except Exception:
        pass
    # transformers snapshots HF_HUB_OFFLINE at import in utils.hub._is_offline_mode;
    # keep it in sync so from_pretrained respects Hub offline during transcription.
    try:
        import transformers.utils.hub as _tf_hub
        _tf_hub._is_offline_mode = True
    except Exception:
        pass
    log.info('Hugging Face Hub offline for transcription (onboarding complete).')


@contextmanager
def allow_hf_network_for_align(hf_offline: bool):
    """Allow Hugging Face Hub briefly so whisperx can fetch align-model weights.

    After onboarding, transcription runs with HF_HUB_OFFLINE; align models are
    language-specific and may not be cached yet. Restore offline when done.
    Skipped when PINE_ALLOW_HF_NETWORK=1 (already unrestricted) or offline mode
    was never applied.
    """
    if not hf_offline or hf_network_allowed():
        yield
        return

    os.environ.pop('HF_HUB_OFFLINE', None)
    hf_const = None
    try:
        import huggingface_hub.constants as hf_const
        hf_const.HF_HUB_OFFLINE = False
    except Exception:
        pass
    tf_hub = None
    tf_offline_prev = True
    try:
        import transformers.utils.hub as tf_hub
        tf_offline_prev = tf_hub._is_offline_mode
        tf_hub._is_offline_mode = False
    except Exception:
        pass
    log.info('Hugging Face Hub: network allowed for align model load.')
    try:
        yield
    finally:
        os.environ['HF_HUB_OFFLINE'] = '1'
        try:
            if hf_const is not None:
                hf_const.HF_HUB_OFFLINE = True
        except Exception:
            pass
        try:
            if tf_hub is not None:
                tf_hub._is_offline_mode = tf_offline_prev
        except Exception:
            pass
        log.info('Hugging Face Hub: restored offline mode after align load.')
