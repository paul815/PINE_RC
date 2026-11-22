"""The model registry and everything about getting models onto disk.

Owns `MODEL_REGISTRY` (what PINE can download, and how big each one is), the
`MLModel` rows that mirror it, HuggingFace downloads with progress, token
validation for the gated pyannote repos, and removal.

Two neighbours carry what used to live here:
  * `launcher_layout` — the installer/launcher file shuffle,
  * `pip_installer`   — installing torch and the rest of the ML stack.
"""

import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from ..extensions import db
from ..extensions import safe_emit as _safe_emit
from ..models.ml_model import MLModel
from .launcher_state import mark_onboarding_complete
from .pip_installer import (
    _check_packages,
    _install_emit,
    _install_step,
    _install_model_specific_packages,
    _realign_torch_companions,
    install_pip_packages,
)

log = logging.getLogger(__name__)

IS_MAC = sys.platform == 'darwin'


def _begin_windows_hf_hub_no_symlinks():
    """Force huggingface_hub to copy files instead of symlinking (blobs→snapshots).

    On Windows without Developer Mode or SeCreateSymbolicLinkPrivilege, ``os.symlink`` raises
    WinError 1314 ("A required privilege is not held by the client") during model download.
    HF's probe can still report symlinks supported, so we override the check for this thread's downloads.
    """
    if sys.platform != 'win32':
        return None
    try:
        import huggingface_hub.file_download as fd
        orig = fd.are_symlinks_supported

        def _no_symlinks(cache_dir=None):
            return False

        fd.are_symlinks_supported = _no_symlinks
        return orig
    except Exception as exc:
        log.warning('Windows HF symlink workaround unavailable: %s', exc)
        return None

def _end_windows_hf_hub_no_symlinks(orig):
    if orig is None:
        return
    try:
        import huggingface_hub.file_download as fd
        fd.are_symlinks_supported = orig
    except Exception:
        pass

# Every model is fetched at a fixed commit (`revision`), not at whatever the
# repo's main is that day: pyannote and GLiNER ship pickle weights, and some
# repos sit on accounts PINE does not control. Each SHA below was main on
# 2026-11-22 and matches the files PINE was verified with. To move a model on,
# change its SHA deliberately and re-verify transcription on both platforms.
MODEL_REGISTRY = {
    'whisperx-large-v3': {
        'name': 'WhisperX + faster-whisper large-v3',
        'function': 'stt',
        'repo_id': 'Systran/faster-whisper-large-v3',
        'revision': 'edaa852ec7e145841d8ffdb056a99866b5f0a478',
        'size_bytes': 3_000_000_000,
        'required': True,
        'language': 'multi',
        # Every platform: on Apple Silicon it runs on Metal (CTranslate2 4.9+),
        # the same engine and weights as Windows. The Mac's former model,
        # mlx-whisper-large-v3, is retired — see LEGACY_STT_MODEL_ID_ALIASES.
        'platform': 'all',
    },
    'pyannote-diarization': {
        'name': 'pyannote speaker-diarization-community-1',
        'function': 'diarization',
        'repo_id': 'pyannote/speaker-diarization-community-1',
        'revision': '3533c8cf8e369892e6b79ff1bf80f7b0286a54ee',
        # Repo is pipeline config + small assets only (weights live in other pyannote_* entries).
        'size_bytes': 8_000_000,
        'required': True,
        'language': None,
        # pyannote loads checkpoints via hf_hub_download(..., cache_dir=PYANNOTE_CACHE)
        'use_pyannote_hub_cache': True,
    },
    'pyannote-wespeaker-voxceleb-resnet34-LM': {
        'name': 'pyannote WeSpeaker embedding (diarization)',
        'function': 'diarization',
        'repo_id': 'pyannote/wespeaker-voxceleb-resnet34-LM',
        'revision': '837717ddb9ff5507820346191109dc79c958d614',
        'size_bytes': 55_000_000,
        'required': True,
        'language': None,
        'use_pyannote_hub_cache': True,
    },
    'gliner-pii': {
        'name': 'GLiNER multilingual (PII removal)',
        'function': 'pii',
        'repo_id': 'urchade/gliner_multi_pii-v1',
        'revision': '1fcf13e85f4eef5394e1fcd406cf2ca9ea82351d',
        'size_bytes': 1_800_000_000,
        'required': False,
        'language': None,
        # GLiNER needs the mdeberta tokenizer files locally to avoid runtime HF fetches
        'tokenizer_repo': 'microsoft/mdeberta-v3-base',
        'tokenizer_revision': 'a0484667b22365f84929a935b5e50a51f71f159d',
        'tokenizer_files': [
            'spm.model', 'tokenizer_config.json', 'config.json',
        ],
    },
}

def _model_for_platform(info):
    """Return True if this model entry applies to the current platform."""
    plat = info.get('platform')
    if plat is None or plat == 'all':
        return True
    return plat == sys.platform

def init_model_registry():
    """Populate the ml_models table with known models (idempotent).

    Rows for ids the registry no longer knows are dropped. Without that, an entry
    retired from MODEL_REGISTRY keeps showing up in the onboarding and settings
    lists of every install that ever saw it, since both read the table directly.
    Platform-filtered entries are still in the registry and so survive a move
    between machines; only genuinely unknown ids go.
    """
    for stale in MLModel.query.filter(MLModel.id.notin_(list(MODEL_REGISTRY))).all():
        log.info('Dropping ml_models row for retired model %s', stale.id)
        db.session.delete(stale)

    for model_id, info in MODEL_REGISTRY.items():
        if not _model_for_platform(info):
            continue
        m = db.session.get(MLModel, model_id)
        if m:
            # Always sync size_bytes from registry (catches stale values)
            if info.get('size_bytes') and m.size_bytes != info['size_bytes']:
                m.size_bytes = info['size_bytes']
        else:
            db.session.add(MLModel(
                id=model_id,
                name=info['name'],
                function=info['function'],
                repo_id=info['repo_id'],
                filename=info.get('filename'),
                size_bytes=info['size_bytes'],
                status='not_downloaded',
                language=info.get('language'),
                required=info['required'],
            ))
    db.session.commit()

def pyannote_hub_cache_root(models_path):
    """Directory where pyannote HF repos are snapshotted (matches PYANNOTE_CACHE in transcription)."""
    return os.path.join(models_path, 'pyannote_cache')

def _hub_cache_marker_file(repo_id):
    # Pipeline-config repos (no weights) use config.yaml as the cache marker.
    _pipeline_repos = ('speaker-diarization-3.1', 'speaker-diarization-community-1')
    if any(repo_id.endswith(r) for r in _pipeline_repos):
        return 'config.yaml'
    return 'pytorch_model.bin'

def snapshot_dir_if_repo_cached(repo_id, cache_dir):
    """Return snapshot directory for repo if HF hub cache under cache_dir has the marker file."""
    from huggingface_hub import _CACHED_NO_EXIST, try_to_load_from_cache

    marker = _hub_cache_marker_file(repo_id)
    fp = try_to_load_from_cache(repo_id, marker, cache_dir=cache_dir)
    if fp is None or fp is _CACHED_NO_EXIST:
        return None
    return os.path.dirname(fp)

def model_storage_dir(models_path, m, info):
    """Resolved on-disk location for a model row (hub-cache snapshot, legacy folder, or STT path)."""
    if m.path and os.path.isdir(m.path):
        return m.path
    if info.get('use_pyannote_hub_cache'):
        pyc = pyannote_hub_cache_root(models_path)
        snap = snapshot_dir_if_repo_cached(info['repo_id'], pyc)
        if snap:
            return snap
        legacy = os.path.join(models_path, m.id)
        if os.path.isdir(legacy):
            return legacy
        return None
    return os.path.join(models_path, m.id)

def reconcile_model_statuses(models_path):
    """On startup, fix stale model statuses by checking what is actually on disk.

    - Models stuck in 'downloading' are reset to 'not_downloaded' (re-shows the Download button).
    - Models marked 'not_downloaded' that already have files on disk are promoted to 'ready'.
    """
    changed = False
    for m in MLModel.query.all():
        info = MODEL_REGISTRY.get(m.id, {})
        if not info:
            continue
        dest = model_storage_dir(models_path, m, info)
        expected = m.size_bytes or info.get('size_bytes', 0)

        if m.status == 'downloading':
            # Server restarted mid-download; decide based on what's actually on disk
            if dest and _model_already_on_disk(dest, expected):
                log.info('Model %s found complete on disk after restart, marking ready', m.id)
                m.status = 'ready'
                m.path = dest
                m.downloaded_bytes = expected
            else:
                log.info('Model %s was mid-download at restart, resetting to not_downloaded', m.id)
                m.status = 'not_downloaded'
                m.downloaded_bytes = 0
            changed = True

        elif m.status == 'not_downloaded':
            # Model may have been downloaded in a previous session whose DB was lost/reset
            if dest and _model_already_on_disk(dest, expected):
                log.info('Model %s already on disk but DB shows not_downloaded, marking ready', m.id)
                m.status = 'ready'
                m.path = dest
                m.downloaded_bytes = expected
                changed = True

    if changed:
        db.session.commit()

def _gated_pyannote_repos():
    """Repo IDs that require accepting a HuggingFace license before download.

    Derived from MODEL_REGISTRY so it stays in sync with what we actually
    download — the pyannote diarization stack is gated and the user must accept
    each model's conditions on huggingface.co first.
    """
    return [
        info['repo_id']
        for info in MODEL_REGISTRY.values()
        if info.get('use_pyannote_hub_cache') and info.get('repo_id')
    ]

def validate_hf_token(token):
    """Validate a HuggingFace token AND verify it can access the gated pyannote repos.

    Returns a dict:
      - valid:       token authenticates (whoami succeeds)
      - username:    HF account name (when valid)
      - gated_ok:    True if every gated pyannote repo is accessible
      - gated_repos: per-repo access detail [{repo_id, url, accessible, reason}]

    A token that authenticates but is missing a license acceptance returns
    ``valid=True, gated_ok=False`` so the UI can tell the user exactly which
    licenses to accept — instead of letting the failure surface deep inside the
    multi-GB model download.
    """
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        return {'valid': False, 'error': f'huggingface_hub unavailable: {exc}'}

    try:
        api = HfApi(token=token)
        user = api.whoami()
    except Exception as exc:
        return {'valid': False, 'error': str(exc)}

    try:
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except Exception:  # pragma: no cover - defensive
        GatedRepoError = RepositoryNotFoundError = ()

    gated_repos = []
    auth_check = getattr(api, 'auth_check', None)
    for repo_id in _gated_pyannote_repos():
        entry = {
            'repo_id': repo_id,
            'url': f'https://huggingface.co/{repo_id}',
            'accessible': True,
            'reason': '',
        }
        if auth_check is None:
            # Older huggingface_hub without auth_check — cannot pre-verify.
            gated_repos.append(entry)
            continue
        try:
            auth_check(repo_id)
        except GatedRepoError:
            entry['accessible'] = False
            entry['reason'] = 'license_not_accepted'
        except RepositoryNotFoundError:
            entry['accessible'] = False
            entry['reason'] = 'not_found_or_no_access'
        except Exception as exc:
            # Network/transient error: surface but do not hard-block onboarding.
            entry['accessible'] = None
            entry['reason'] = f'check_failed: {exc}'
        gated_repos.append(entry)

    blocked = [g for g in gated_repos if g['accessible'] is False]
    return {
        'valid': True,
        'username': user.get('name', ''),
        'gated_ok': not blocked,
        'gated_repos': gated_repos,
    }

STT_MODEL_QUALITY = 'whisperx-large-v3'

#: The Mac's model until 2026-11: mlx-whisper in 120-s windows. Retired — on
#: an 88-min interview it matched Windows in 94.4% of words with 8 holes, where
#: WhisperX on Metal matched 99.9% (mac-bench, 2026-11-21). Kept as an alias
#: target only, so a stored setting naming it lands on the current model.
MLX_STT_MODEL_QUALITY = 'mlx-whisper-large-v3'

LEGACY_STT_MODEL_ID_ALIASES = {
    'whisperx-large-v3-turbo': STT_MODEL_QUALITY,
    'mlx-whisper-large-v3-turbo': STT_MODEL_QUALITY,
    MLX_STT_MODEL_QUALITY: STT_MODEL_QUALITY,
}

def get_default_stt_model():
    """Return the default STT model ID — the same on every platform."""
    return STT_MODEL_QUALITY

def supported_stt_models():
    """STT model IDs a user may choose on this platform, best-quality first."""
    return (get_default_stt_model(),)

def normalize_stt_model_id(stt_model_id):
    """Map legacy/foreign/invalid STT IDs onto one this platform can run.

    One model everywhere: ``whisperx-large-v3``, on Metal on a Mac. The Mac's
    retired mlx ids map onto it, as does anything unknown.
    """
    model_id = (stt_model_id or '').strip()
    if model_id in LEGACY_STT_MODEL_ID_ALIASES:
        model_id = LEGACY_STT_MODEL_ID_ALIASES[model_id]

    if model_id not in supported_stt_models():
        return get_default_stt_model()
    return model_id

#: Which registry entries an optional setup module brings with it. The setup
#: screen sizes its rows from this too, so a module gains a model in one place.
OPTIONAL_MODULE_MODELS = {
    'pii': ('gliner-pii',),
}

def get_models_for_setup(modules, stt_model_id=None):
    """Return list of model IDs to download based on user choices."""
    stt_model = normalize_stt_model_id(stt_model_id) if stt_model_id else get_default_stt_model()
    ids = [
        stt_model,
        'pyannote-diarization',
    ]

    for module in modules:
        for model_id in OPTIONAL_MODULE_MODELS.get(module, ()):
            if model_id not in ids:
                ids.append(model_id)

    return ids

def registry_size_bytes(model_ids):
    """Download size of these registry entries, in bytes. Unknown ids count 0."""
    return sum((MODEL_REGISTRY.get(model_id) or {}).get('size_bytes') or 0 for model_id in model_ids)

def setup_size_breakdown(stt_model_id=None):
    """Per-row download sizes for the setup Modules step, from the registry.

    The screen used to carry its own constants, which drifted: diarization was
    printed as 1 GB and PII as 500 MB against a registry saying 63 MB and 1.8 GB.
    Serving the numbers from here keeps the estimate and the download in step.

    Diarization sums every entry with that function rather than the ids in
    ``get_models_for_setup``: the download list names the community-1 entrypoint
    alone, but the warm-up that follows resolves the pipeline's sub-components
    and fetches the WeSpeaker checkpoint as well, so the user pays for both.
    """
    stt_id = normalize_stt_model_id(stt_model_id) if stt_model_id else get_default_stt_model()
    diarization_ids = [
        model_id for model_id, info in MODEL_REGISTRY.items()
        if info.get('function') == 'diarization' and _model_for_platform(info)
    ]
    return {
        'stt_bytes': registry_size_bytes([stt_id]),
        'diarization_bytes': registry_size_bytes(diarization_ids),
        'modules': {
            module: registry_size_bytes(model_ids)
            for module, model_ids in OPTIONAL_MODULE_MODELS.items()
        },
    }

def remove_model(model_id, models_path):
    """Remove a model from disk and reset its DB status."""
    m = db.session.get(MLModel, model_id)
    info = MODEL_REGISTRY.get(model_id, {})
    dest = None
    if m and m.path and os.path.isdir(m.path):
        dest = m.path
    elif info.get('use_pyannote_hub_cache'):
        pyc = pyannote_hub_cache_root(models_path)
        dest = snapshot_dir_if_repo_cached(info['repo_id'], pyc)
    if dest is None:
        dest = os.path.join(models_path, model_id)
    if os.path.isdir(dest):
        try:
            shutil.rmtree(dest)
            log.info('Removed model %s from %s', model_id, dest)
        except OSError as exc:
            log.warning('Could not remove model dir %s: %s', dest, exc)
    if m:
        m.status = 'not_downloaded'
        m.path = None
        m.downloaded_bytes = 0
        db.session.commit()

def _dir_size(path):
    total = 0
    try:
        for f in Path(path).rglob('*'):
            if f.is_file():
                total += f.stat().st_size
    except Exception:
        pass
    return total

def _model_already_on_disk(dest, expected_bytes):
    """Return True if the model directory exists and has substantial content (≥80% of expected)."""
    if not expected_bytes or expected_bytes < 1000:
        return os.path.isdir(dest) and _dir_size(dest) > 1000
    return _dir_size(dest) >= 0.8 * expected_bytes

def _monitor_progress(model_id, dest, expected, completed_so_far, overall_total, stop, app):
    """Poll directory size every second and emit SocketIO progress."""
    prev_size = 0
    prev_time = time.time()
    while not stop.is_set():
        cur_size = min(_dir_size(dest), expected) if expected else _dir_size(dest)
        now = time.time()
        dt = now - prev_time
        speed = int((cur_size - prev_size) / dt) if dt > 0 else 0

        _safe_emit('download:progress', {
            'model_id': model_id,
            'downloaded_bytes': cur_size,
            'total_bytes': expected,
            'speed_bps': max(speed, 0),
            'overall_downloaded': completed_so_far + cur_size,
            'overall_total': overall_total,
        })

        with app.app_context():
            m = db.session.get(MLModel, model_id)
            if m:
                m.downloaded_bytes = cur_size
                db.session.commit()

        prev_size = cur_size
        prev_time = now
        stop.wait(1.0)

def play_install_complete_sound(app, volume_pct=50):
    """Play install-complete sound from the terminal (works regardless of browser tab focus).

    ``volume_pct`` is clamped to 0–100; 0 or below skips playback. Prefer ``ffplay`` when
    available so volume applies on all platforms; otherwise decode with ffmpeg and use the
    OS default player path (volume applied when ffmpeg runs).
    """
    try:
        vol = max(0, min(100, int(volume_pct)))
        if vol <= 0:
            return
        static_dir = Path(app.static_folder) if app.static_folder else Path(app.root_path) / 'static'
        sound_path = static_dir / 'sounds' / 'install-complete.mp3'
        if not sound_path.is_file():
            log.warning('Install-complete sound not found: %s', sound_path)
            return
        path_str = str(sound_path.resolve())

        def _play():
            try:
                ffplay = shutil.which('ffplay')
                if ffplay:
                    subprocess.Popen(
                        [ffplay, '-nodisp', '-autoexit', '-volume', str(vol),
                         '-loglevel', 'quiet', path_str],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    return

                import tempfile
                fd, wav_path = tempfile.mkstemp(suffix='.wav')
                os.close(fd)
                try:
                    vol_lin = vol / 100.0
                    r = subprocess.run(
                        [
                            'ffmpeg', '-y', '-i', path_str,
                            '-filter:a', f'volume={vol_lin}',
                            '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1', wav_path,
                        ],
                        capture_output=True, timeout=60,
                    )
                    if r.returncode != 0 or not os.path.isfile(wav_path):
                        raise RuntimeError('ffmpeg wav decode failed')

                    if sys.platform == 'win32':
                        import winsound
                        winsound.PlaySound(wav_path, winsound.SND_FILENAME)
                    elif sys.platform == 'darwin':
                        subprocess.run(
                            ['afplay', wav_path], capture_output=True, timeout=120,
                        )
                    else:
                        aplay = shutil.which('aplay') or shutil.which('paplay')
                        if not aplay:
                            raise RuntimeError('no aplay/paplay')
                        subprocess.run(
                            [aplay, wav_path], capture_output=True, timeout=120,
                        )
                finally:
                    try:
                        os.remove(wav_path)
                    except OSError:
                        pass
            except Exception as exc:
                log.warning('Play sound failed: %s', exc)
                if sys.platform == 'win32':
                    try:
                        os.startfile(path_str)
                    except Exception:
                        pass

        threading.Thread(target=_play, daemon=True).start()
    except Exception as exc:
        log.warning('Failed to play install-complete sound: %s', exc)

def _preload_alignment_model():
    """Pre-download WhisperX alignment model (English) so first transcription isn't delayed."""
    try:
        _install_step('Pre-downloading alignment model (English)', 'Downloading alignment model')
        # On a Mac, torch.hub's urllib download needs the CA bundle that
        # app._ensure_ca_bundle() sets at import.
        import torchaudio
        bundle = getattr(torchaudio.pipelines, 'WAV2VEC2_ASR_BASE_960H', None)
        if bundle:
            bundle.get_model()
            _install_emit('Alignment model cached.')
        else:
            _install_emit('Alignment model not found in torchaudio, will download on first use.')
    except Exception as exc:
        log.warning('Alignment model preload failed: %s', exc)
        _install_emit(f'Alignment preload skipped: {exc}')

def _warmup_pyannote_community1(models_path, hf_token=None):
    """Warm up pyannote community-1 so first diarization does not cold-start.

    This keeps the onboarding model list minimal (community-1 entrypoint only)
    while still pulling any internal artifacts the pipeline needs.
    """
    try:
        _install_step('Preloading pyannote diarization pipeline (community-1)',
                      'Preloading diarization model')
        import inspect

        import torch

        # The worker applies these before it touches pyannote (ml_worker/diarize.py);
        # this warm-up imports the same stack inside Flask and needs them just as
        # much. Without the is_offline_mode shim the import itself raises and
        # onboarding silently skips the warm-up it just told the user it was doing.
        from ml_worker import compat
        compat.patch_hf_hub_is_offline_mode()
        compat.patch_torchaudio_for_pyannote()
        compat.patch_hf_hub_legacy_use_auth_token()

        from pyannote.audio import Pipeline

        pyc = pyannote_hub_cache_root(models_path)
        os.makedirs(pyc, exist_ok=True)
        os.environ['PYANNOTE_CACHE'] = pyc

        sig = inspect.signature(Pipeline.from_pretrained)
        kwargs = {}
        auth = hf_token or None
        if 'token' in sig.parameters:
            kwargs['token'] = auth
        elif 'use_auth_token' in sig.parameters:
            kwargs['use_auth_token'] = auth
        if 'cache_dir' in sig.parameters:
            kwargs['cache_dir'] = pyc
        if 'revision' in sig.parameters:
            # Network is still open during onboarding: without the pin the hub
            # client would look up today's main instead of the commit downloaded.
            kwargs['revision'] = MODEL_REGISTRY['pyannote-diarization']['revision']

        # torch 2.6+ refuses to unpickle the custom types pyannote checkpoints
        # carry. The worker relaxes torch.load for good; Flask borrows it for
        # this one read of a model the user just downloaded and hands it back.
        with compat.trusted_torch_load():
            pipeline = Pipeline.from_pretrained(
                'pyannote/speaker-diarization-community-1', **kwargs)
            pipeline.to(torch.device('cpu'))

            # Tiny dry-run to force lazy sub-component resolution during
            # onboarding — which reads more checkpoints, so it stays inside.
            waveform = torch.zeros((1, 16000 * 3), dtype=torch.float32)
            try:
                pipeline({'waveform': waveform, 'sample_rate': 16000}, min_speakers=1, max_speakers=2)
            except Exception as run_exc:
                log.info('pyannote warm-up inference skipped/partial: %s', run_exc)

        # To the app log as well as the install stream: by the time the warm-up
        # runs, the pip log it would otherwise land in has been closed, and only
        # the failure had a line of its own. A silent success and a warm-up that
        # never ran read the same in the log afterwards.
        log.info('pyannote community-1 warm-up complete')
        _install_emit('pyannote community-1 is warmed up and cached.')
    except Exception as exc:
        log.warning('pyannote community-1 warm-up failed: %s', exc)
        _install_emit(f'pyannote warm-up skipped: {exc}')

def _point_main_at_revision(cache_dir, repo_id, revision):
    """Make 'main' in PINE's own hub cache mean the pinned commit.

    A download by commit hash writes no refs/main, but pyannote and
    snapshot_dir_if_repo_cached both resolve 'main' in this cache — offline,
    once onboarding is done. Without the ref the model could not be found and
    would be fetched again on every launch.
    """
    refs = os.path.join(cache_dir, 'models--' + repo_id.replace('/', '--'), 'refs')
    os.makedirs(refs, exist_ok=True)
    from .file_utils import atomic_write_text
    atomic_write_text(os.path.join(refs, 'main'), revision)


def _fetch_model_files(info, dest, pyc_root, hf_token, hf_hub_download, snapshot_download):
    """Download one registry model at its pinned revision; return where it landed."""
    revision = info['revision']
    if info.get('filename'):
        hf_hub_download(
            repo_id=info['repo_id'],
            filename=info['filename'],
            local_dir=dest,
            token=hf_token,
            revision=revision,
        )
        return dest
    snap_kwargs = {'repo_id': info['repo_id'], 'token': hf_token, 'revision': revision}
    if info.get('ignore_patterns'):
        snap_kwargs['ignore_patterns'] = info['ignore_patterns']
    if info.get('use_pyannote_hub_cache'):
        final_path = snapshot_download(cache_dir=pyc_root, **snap_kwargs)
        _point_main_at_revision(pyc_root, info['repo_id'], revision)
        return final_path
    snapshot_download(local_dir=dest, **snap_kwargs)
    return dest


def _fetch_tokenizer_files(info, dest, hf_token, hf_hub_download):
    """Fetch the tokenizer files a model needs locally, at their pinned revision."""
    tok_repo = info.get('tokenizer_repo')
    for tf in info.get('tokenizer_files', []) if tok_repo else []:
        if os.path.isfile(os.path.join(dest, tf)):
            continue
        try:
            hf_hub_download(
                repo_id=tok_repo,
                filename=tf,
                local_dir=dest,
                token=hf_token,
                revision=info['tokenizer_revision'],
            )
        except Exception as tok_exc:
            log.warning('Tokenizer file %s download failed: %s', tf, tok_exc)


def download_models(app, model_ids, models_path, hf_token=None, finish_onboarding=True, block=False):
    """Download all requested models in a background thread.

    finish_onboarding: if True, sets onboarding_complete and preloads alignment model.
    Set False when re-downloading (e.g. STT model switch) after onboarding is complete.
    block: if True, wait for download to complete before returning.
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    def _run():
        # Transcription sets HF_HUB_OFFLINE after onboarding; allow network here.
        _prev_hf_offline = os.environ.pop('HF_HUB_OFFLINE', None)
        # Also reset the module-level constant — huggingface_hub caches it at import
        # time, so clearing the env var alone is not enough.
        import huggingface_hub.constants as _hf_const
        _prev_hf_const_offline = getattr(_hf_const, 'HF_HUB_OFFLINE', False)
        _hf_const.HF_HUB_OFFLINE = False
        _hf_symlink_orig = None
        try:
            _hf_symlink_orig = _begin_windows_hf_hub_no_symlinks()
            time.sleep(1.5)

            # Skip full pip install when downloading only optional models and core packages are already installed
            requested_models = [MODEL_REGISTRY.get(mid, {}) for mid in model_ids]
            only_optional = all(info.get('required', True) is False for info in requested_models if info)
            all_core_installed = all(p['installed'] for p in _check_packages())

            if only_optional and all_core_installed:
                log.info('Skipping pip install: only optional models requested and core packages ready.')
            else:
                install_pip_packages()

            # ffmpeg rides along with the models. Its first fetch is ~190 MB from
            # GitHub and used to run at the top of run.py, blocking the browser
            # from opening at all; here it is inside the step the user already
            # waits on. No-op once the binaries are in place.
            from .ffmpeg_setup import ensure_ffmpeg, ffmpeg_on_path
            if not ffmpeg_on_path():
                _install_emit('Installing ffmpeg...')
                if ensure_ffmpeg():
                    _install_emit('ffmpeg ready.')
                else:
                    # Not fatal: the models are the point of this step, and
                    # transcription retries the fetch when it starts.
                    _install_emit('ffmpeg install failed -- audio features may not work.')
                    log.warning('ffmpeg could not be prepared during model download')

            # Fix 2: Install model-specific packages (e.g. gliner for gliner-pii)
            optional_pip = _install_model_specific_packages(model_ids)
            # Optional pip installs can alter torch; keep its companions on the same channel.
            if not IS_MAC and optional_pip and not _realign_torch_companions():
                log.warning(
                    'torch companion re-align failed after model-specific pip installs; '
                    'transcription may fail until you run: python -m pip install torchaudio torchvision'
                )

            with app.app_context():
                rows = [db.session.get(MLModel, mid) for mid in model_ids]
                rows = [r for r in rows if r and r.status != 'ready']
                overall_total = sum(r.size_bytes for r in rows)
                completed = 0

                pyc_root = pyannote_hub_cache_root(models_path)

                for model in rows:
                    info = MODEL_REGISTRY.get(model.id, {})
                    expected = model.size_bytes or info.get('size_bytes', 0)
                    use_pyc = bool(info.get('use_pyannote_hub_cache'))

                    if use_pyc:
                        os.makedirs(pyc_root, exist_ok=True)
                        os.environ['PYANNOTE_CACHE'] = pyc_root
                        dest = pyc_root
                        snap_existing = snapshot_dir_if_repo_cached(info['repo_id'], pyc_root)
                        if snap_existing and _model_already_on_disk(snap_existing, expected):
                            log.info('Model %s already in pyannote cache, skipping download', model.id)
                            model.status = 'ready'
                            model.path = snap_existing
                            model.downloaded_bytes = expected
                            db.session.commit()
                            completed += expected
                            _safe_emit('download:model_complete', {
                                'model_id': model.id,
                                'overall_downloaded': completed,
                                'overall_total': overall_total,
                            })
                            continue
                    else:
                        dest = os.path.join(models_path, model.id)
                        if _model_already_on_disk(dest, expected):
                            log.info('Model %s already on disk, skipping download', model.id)
                            model.status = 'ready'
                            model.path = dest
                            model.downloaded_bytes = expected
                            db.session.commit()
                            completed += expected
                            _safe_emit('download:model_complete', {
                                'model_id': model.id,
                                'overall_downloaded': completed,
                                'overall_total': overall_total,
                            })
                            continue

                    if not use_pyc:
                        os.makedirs(dest, exist_ok=True)
                    model.status = 'downloading'
                    model.downloaded_bytes = 0
                    db.session.commit()

                    _safe_emit('download:model_start', {
                        'model_id': model.id,
                        'name': model.name,
                        'size_bytes': model.size_bytes,
                    })

                    stop_evt = threading.Event()
                    mon = threading.Thread(
                        target=_monitor_progress,
                        args=(model.id, dest, model.size_bytes, completed, overall_total, stop_evt, app),
                        daemon=True,
                    )
                    mon.start()

                    try:
                        final_path = _fetch_model_files(
                            info, dest, pyc_root, hf_token, hf_hub_download, snapshot_download,
                        )

                        stop_evt.set()
                        mon.join(timeout=3)

                        # Download tokenizer files if the model needs them locally
                        if info.get('tokenizer_repo') and info.get('tokenizer_files'):
                            _install_emit(f'Downloading tokenizer for {model.name}...')
                            _fetch_tokenizer_files(info, dest, hf_token, hf_hub_download)

                        model.status = 'ready'
                        model.path = final_path if use_pyc else dest
                        model.downloaded_bytes = model.size_bytes
                        db.session.commit()
                        completed += model.size_bytes

                        _safe_emit('download:model_complete', {
                            'model_id': model.id,
                            'overall_downloaded': completed,
                            'overall_total': overall_total,
                        })

                    except Exception as exc:
                        stop_evt.set()
                        mon.join(timeout=3)

                        model.status = 'error'
                        model.error_message = str(exc)
                        db.session.commit()

                        _safe_emit('download:error', {
                            'model_id': model.id,
                            'error': str(exc),
                        })

                if finish_onboarding:
                    # The Mac aligns with WhisperX too now (on Metal), so it
                    # preloads the same alignment model as Windows.
                    _preload_alignment_model()
                    _warmup_pyannote_community1(models_path, hf_token=hf_token)
                    from ..models.setting import Setting
                    Setting.set('onboarding_complete', 'true')
                    mark_onboarding_complete()
                    # The repo-root shuffle deliberately does NOT happen here.
                    # Downloads finishing is not the end of onboarding: the user
                    # still has a Launch PINE button to press, and until they do
                    # the root installer stays put as the way back in. See
                    # launcher_layout.finalize_root_layout_after_onboarding,
                    # called from the /handoff/prepare endpoint.
                    # Enable auto-backup by default for new installs
                    if not Setting.get('auto_backup_enabled'):
                        Setting.set('auto_backup_enabled', 'true')

                _safe_emit('download:all_complete', {
                    'overall_total': overall_total,
                })

        finally:
            _end_windows_hf_hub_no_symlinks(_hf_symlink_orig)
            _hf_const.HF_HUB_OFFLINE = _prev_hf_const_offline
            if _prev_hf_offline is not None:
                os.environ['HF_HUB_OFFLINE'] = _prev_hf_offline

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    if block:
        t.join()
