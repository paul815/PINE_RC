"""Installing the ML stack: torch and everything that must match it.

PINE ships without torch, whisperx, pyannote or mlx — together they are several
gigabytes, and the right torch build depends on the machine. Onboarding
installs them here, into the app's own venv, streaming pip's output to the UI.

Two things make this more than `pip install`:

* **The right torch.** On Windows/Linux `torchruntime` picks a CUDA or CPU
  wheel index for the detected GPU; on Apple Silicon the plain wheel is right.
* **Keeping the companions aligned.** torchaudio and torchvision must come from
  the same channel as torch (`+cu128` vs `+cpu`). whisperx pins both; torchaudio
  is PINE's own besides — it ships the wav2vec2 alignment bundle and backs
  pyannote — and pip will happily install a CPU torchaudio next to a CUDA torch,
  which then fails at import with an unhelpful symbol error. So
  `repair_torch_companion_wheels_if_needed()` checks the channels on startup and
  reinstalls the odd one out.

Everything runs with `PIP_REQUIRE_VIRTUALENV=1` and `PYTHONNOUSERSITE=1`, so a
mistake here can never write into the user's system or `~/.local` packages.
"""

import logging
import os
import subprocess
import sys
from datetime import datetime

from ..extensions import safe_emit as _safe_emit

log = logging.getLogger(__name__)

IS_MAC = sys.platform == 'darwin'


# Subprocess environment that prevents pip from ever touching system or user packages.
# PIP_REQUIRE_VIRTUALENV  — pip refuses to run if not inside a venv.
# PYTHONNOUSERSITE        — disables ~/.local / %APPDATA%\Python user site-packages.
_PIP_ENV = {**os.environ, 'PIP_REQUIRE_VIRTUALENV': '1', 'PYTHONNOUSERSITE': '1'}

if IS_MAC:
    REQUIRED_PACKAGES = [
        ('torch', 'torch', 'PyTorch ML runtime'),
        ('torchaudio', 'torchaudio', 'Audio backend for pyannote'),
        ('mlx_whisper', 'mlx-whisper', 'Transcription (mlx-whisper, Metal-accelerated)'),
        ('pyannote.audio', 'pyannote-audio', 'Speaker diarization (pyannote)'),
    ]
    # torchaudio 2.9+ dropped AudioMetaData; patched at runtime in ml_worker.compat.patch_torchaudio_for_pyannote().
    PIP_INSTALL_TARGETS = ['torch', 'torchaudio', 'mlx-whisper', 'pyannote-audio']
else:
    REQUIRED_PACKAGES = [
        ('torchruntime', 'torchruntime', 'GPU auto-detection'),
        ('torch', 'torch', 'PyTorch ML runtime'),
        ('torchaudio', 'torchaudio', 'Alignment model + pyannote audio backend'),
        ('torchvision', 'torchvision', 'Torch companion pinned by WhisperX'),
        ('whisperx', 'whisperx', 'Transcription + alignment (WhisperX)'),
        ('faster_whisper', 'faster-whisper', 'Whisper inference engine'),
        ('pyannote.audio', 'pyannote-audio', 'Speaker diarization (pyannote)'),
    ]
    PIP_INSTALL_TARGETS = ['torchruntime', 'torchaudio', 'whisperx']

# Optional pip packages required by specific models (beyond REQUIRED_PACKAGES).
# A runtime lives here rather than in REQUIRED_PACKAGES so an install that never
# uses the model it belongs to is not blocked on a package it does not need.
MODEL_OPTIONAL_PACKAGES = {
    'gliner-pii': [('gliner', 'gliner', 'PII detection')],
    # Parakeet's runtime, and the reason it is the one engine both platforms can
    # run: onnxruntime needs neither torch nor Metal.
    #
    # The wheel differs by platform and the difference is not cosmetic. There is
    # no ``onnxruntime-gpu`` for macOS at all — asking for it there fails the
    # install outright — while the plain wheel carries the CoreML provider that
    # the Mac path depends on. On Windows/Linux ``onnxruntime-gpu`` brings its
    # own CUDA and cuDNN; loading those beside torch is deliberate, they are
    # separate runtimes and share no state.
    'parakeet-tdt-0.6b-v3': [
        ('onnx_asr', 'onnx-asr[hub]', 'Parakeet speech recognition'),
        ('onnxruntime',
         'onnxruntime' if IS_MAC else 'onnxruntime-gpu[cuda,cudnn]',
         'ONNX Runtime'),
    ],
}

def _is_package_installed(import_name):
    """True if *import_name* is importable (e.g. ``torch``, ``pyannote.audio``)."""
    import importlib
    import importlib.util
    try:
        if importlib.util.find_spec(import_name) is not None:
            return True
    except (ModuleNotFoundError, ValueError):
        pass
    # find_spec can miss some layouts; importing is the ground truth (slightly slower).
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def _check_packages():
    """Return list of dicts with install status for each required package."""
    results = []
    for import_name, pip_name, description in REQUIRED_PACKAGES:
        results.append({
            'name': pip_name,
            'description': description,
            'installed': _is_package_installed(import_name),
        })
    return results

# The subset of REQUIRED_PACKAGES that must be importable before transcription.
_MAC_TRANSCRIPTION_IMPORTS = [
    ('torch',          'torch'),
    ('mlx_whisper',    'mlx-whisper'),
    ('pyannote.audio', 'pyannote-audio'),
]

_TRANSCRIPTION_IMPORTS = [
    ('torch',          'torch'),
    ('whisperx',       'whisperx'),
    ('pyannote.audio', 'pyannote-audio'),
]

def _transcription_imports():
    """Imports that must resolve before transcription can start."""
    return list(_MAC_TRANSCRIPTION_IMPORTS if IS_MAC else _TRANSCRIPTION_IMPORTS)

def check_ml_deps():
    """Return a list of pip package names that are missing.

    Call this before starting transcription to surface a clear error instead
    of a cryptic ImportError.  Returns an empty list when all deps are present.
    """
    return [pip for imp, pip in _transcription_imports()
            if not _is_package_installed(imp)]

def ensure_transcription_dependencies():
    """Install any missing transcription pip packages, then re-check.

    Upgrades can add new required packages (e.g. pyannote-audio on Mac after dropping
    WhisperX) while onboarding stays marked complete; this self-heals on first transcribe.
    """
    for imp, pip in _transcription_imports():
        if not _is_package_installed(imp):
            log.info('Installing missing transcription dependency: %s', pip)
            _install_package_if_missing(imp, pip)
    return check_ml_deps()

def _install_package_if_missing(import_name, pip_name):
    """Install pip package if not already installed. Returns True if installed or already present."""
    if _is_package_installed(import_name):
        return True
    if os.environ.get('PINE_TESTING'):
        # Reachable from the job runner via ensure_transcription_dependencies,
        # so an unmocked test would shell out to pip and pull ~116 MB of torch
        # into whatever interpreter is running the suite — slow, networked, and
        # it mutates the developer's environment. Tests mock the install path.
        log.warning('PINE_TESTING set — skipping pip install of %s', pip_name)
        return False
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--no-input', pip_name],
            env=_PIP_ENV,
            capture_output=True,
            timeout=300,
            check=True,
        )
        log.info('Installed %s', pip_name)
        return True
    except subprocess.CalledProcessError as exc:
        log.warning('Failed to install %s: %s', pip_name, exc)
        return False
    except Exception as exc:
        log.warning('Failed to install %s: %s', pip_name, exc)
        return False

def _install_model_specific_packages(model_ids):
    """Install any pip packages required by the requested models.

    Returns True if a pip install was attempted (optional deps can alter torch; caller may re-align).
    """
    attempted = False
    for mid in model_ids:
        for import_name, pip_name, desc in MODEL_OPTIONAL_PACKAGES.get(mid, []):
            if not _is_package_installed(import_name):
                attempted = True
                _install_emit(f'Installing {pip_name} for {desc}...')
                _install_package_if_missing(import_name, pip_name)
    return attempted

def _base_python_torch_version():
    """Return torch version string if torch is installed in the base (non-venv) Python, else None."""
    # sys.base_exec_prefix is the real Python prefix, even when running inside a venv
    if sys.base_exec_prefix == sys.exec_prefix:
        return None  # not in a venv, no separate base to check
    base_exe = os.path.join(sys.base_exec_prefix, 'python.exe' if sys.platform == 'win32' else os.path.join('bin', 'python3'))
    if not os.path.isfile(base_exe):
        return None
    try:
        r = subprocess.run(
            [base_exe, '-c', 'import torch; print(torch.__version__)'],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None

def _is_torch_cuda_available():
    """Return True if torch in the venv has CUDA support enabled."""
    try:
        r = subprocess.run(
            [sys.executable, '-c', 'import torch; print(torch.cuda.is_available())'],
            capture_output=True, text=True, timeout=30,
        )
        return r.stdout.strip() == 'True'
    except Exception:
        return False

def _run_torchruntime_install():
    """Run 'torchruntime install' to get the correct PyTorch variant for the GPU."""
    if _is_package_installed('torch') and _is_torch_cuda_available():
        log.info('PyTorch with CUDA already installed, skipping torchruntime install.')
        _install_emit('PyTorch (CUDA) already installed, skipping GPU detection step.')
        return

    if _is_package_installed('torch'):
        _install_emit('PyTorch installed but CUDA not available — running GPU detection to install correct variant...')

    if os.environ.get('PINE_TESTING'):
        log.warning('PINE_TESTING set — skipping torchruntime install')
        return

    try:
        _install_step('Detecting GPU and installing PyTorch variant', 'Step 2/5: detecting GPU')
        proc = subprocess.Popen(
            [sys.executable, '-m', 'torchruntime', 'install'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=_PIP_ENV,
        )
        for line in proc.stdout:
            line = line.rstrip('\n\r')
            if line:
                _install_emit(line)
        proc.wait()
        if proc.returncode != 0:
            log.warning('torchruntime install exited with code %d', proc.returncode)
            _install_emit('WARNING: torchruntime install had issues. CPU mode may be used.')
        else:
            log.info('torchruntime install succeeded.')
    except Exception as exc:
        log.warning('torchruntime install failed: %s', exc)
        _install_emit(f'WARNING: {exc}')

def _pytorch_wheel_index_url():
    """Wheel repo that matches the installed torch build (cpu vs cu128, etc.).

    Default PyPI ships torchaudio with no channel tag at all, so pip will drop an
    untagged wheel next to a ``+cu128`` torch; the pair then fails at import with an
    undefined-symbol error rather than anything that names the real cause.
    """
    if IS_MAC:
        return None
    try:
        import torch
        ver = torch.__version__
    except Exception:
        return 'https://download.pytorch.org/whl/cpu'
    if '+' in ver:
        tag = ver.split('+', 1)[1]
        return f'https://download.pytorch.org/whl/{tag}'
    return 'https://download.pytorch.org/whl/cpu'

def _companion_pip_extra_args(force_reinstall=False):
    """Pip options to install the torch companions from the same channel as torch."""
    if IS_MAC:
        return []
    url = _pytorch_wheel_index_url()
    args = ['--index-url', url]
    if force_reinstall:
        args.insert(0, '--force-reinstall')
    return args

def _local_version_tag(version):
    """The ``+cu128`` / ``+cpu`` suffix of a wheel version, or None for an untagged one."""
    return version.split('+', 1)[1] if '+' in version else None

def _torch_companion_channels_aligned():
    """True if torch and its companions report the same +cpu / +cu* local version tag.

    Both companions are checked. torchaudio is the one PINE imports and the one the
    old check ignored, so a CPU torchaudio next to a CUDA torch used to slip through
    and fail later at import with an undefined-symbol error.
    """
    if IS_MAC:
        return True
    try:
        import torch
        th_tag = _local_version_tag(torch.__version__)
    except Exception:
        return True
    try:
        import torchaudio
    except Exception:
        return False
    if _local_version_tag(torchaudio.__version__) != th_tag:
        return False
    try:
        import torchvision
    except Exception:
        return False
    return _local_version_tag(torchvision.__version__) == th_tag

def _evict_torch_companion_modules():
    """Drop the torch companions from sys.modules after pip reinstall in-process."""
    to_drop = [
        k for k in list(sys.modules)
        if k in ('torchvision', 'torchaudio')
        or k.startswith(('torchvision.', 'torchaudio.'))
    ]
    for k in to_drop:
        sys.modules.pop(k, None)

def repair_torch_companion_wheels_if_needed():
    """If a torch companion's wheel channel mismatches torch, reinstall from PyTorch's index.

    Call before ``import whisperx`` so broken mixed installs self-heal without re-onboarding.
    """
    if IS_MAC:
        return True
    if _torch_companion_channels_aligned():
        return True
    try:
        import torch as _torch
        _th_ver = _torch.__version__
    except Exception:
        _th_ver = '?'
    log.warning(
        'torch (%s) and its companions are on different wheel channels; reinstalling from %s',
        _th_ver,
        _pytorch_wheel_index_url(),
    )
    _install_step('Repairing torch companions to match the PyTorch wheel channel',
                  'Repairing torch companions')
    ok = _realign_torch_companions()
    if ok:
        _evict_torch_companion_modules()
        log.info('torch companions repaired to match torch.')
    return ok

def _realign_torch_companions():
    """Ensure torch and its companions all match the targeted wheel channel.

    whisperx 3.8.5 pins ``torch~=2.8.0`` and its pip-install step will happily downgrade
    torch to the CPU-only 2.8.0 wheel from PyPI (~250 MB), leaving a ``+cu128`` companion
    behind to fail at import with an undefined-symbol error.

    We reinstall them together from the PyTorch wheel channel derived from the previously
    installed torch version (e.g. ``+cu128``). ``--no-deps`` is intentional: the companion
    wheels declare their own exact torch pin, so without it pip cascades and re-downloads
    torch a second time (~2.75 GB). With it, only the explicit list is fetched, which also
    bypasses whisperx's ``torch~=2.8.0`` pin; that leaves ``pip check`` noisy about
    whisperx, but runtime is compatible.
    """
    if IS_MAC:
        return True
    packages = ['torch', 'torchaudio', 'torchvision']
    _install_step('Re-aligning torch companions with installed PyTorch',
                  'Re-aligning torch companions')
    ok = _run_pip(
        packages,
        extra_args=[*_companion_pip_extra_args(force_reinstall=True), '--no-deps'],
    )
    if not ok:
        log.error('Failed to re-align %s', ', '.join(packages))
    return ok


_INSTALL_LOG_FH = None  # open file handle for the active install_pip_packages() session

def _install_log_path():
    """Return a per-session install log path under the same tree as app logs.

    Mirrors the convention in backend/app/__init__.py (log_dir = PINE_LOG_DIR or
    <backend>/logs, daily subdir YYYYMMDD). File name: install-{stamp}-{pid}.log.
    """
    base = os.environ.get('PINE_LOG_DIR')
    if not base:
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        base = os.path.join(backend_dir, 'logs')
    now = datetime.now()
    daily = os.path.join(base, now.strftime('%Y%m%d'))
    try:
        os.makedirs(daily, exist_ok=True)
    except Exception:
        # Fall back to the base dir if the daily subdir can't be created.
        daily = base
        try:
            os.makedirs(daily, exist_ok=True)
        except Exception:
            pass
    return os.path.join(daily, f'install-{now.strftime("%Y%m%d-%H%M%S")}-{os.getpid()}.log')

# Marks a line as a named step rather than pip's own output. Onboarding shows the
# text after the prefix as the current status; without it the UI guessed the step
# from pip's wording and turned the "Phase 1/5" header into "Installing: GPU".
_STEP_PREFIX = 'PINE-STEP: '

def _install_emit(line, log_line=None):
    """Emit an install log line to SocketIO clients AND append it to the install log file.

    *log_line* overrides what reaches the file, so a UI marker never lands in the log.
    If no install session is active, behaves like a plain _safe_emit('install:log', ...).
    """
    _safe_emit('install:log', {'line': line})
    fh = _INSTALL_LOG_FH
    if fh is None:
        return
    try:
        fh.write((line if log_line is None else log_line).rstrip('\r\n') + '\n')
        fh.flush()
    except Exception:
        pass

def _install_step(label, status=None):
    """Name the step the install is on: a section header in the log, a marker for the UI.

    *status* is the short form the UI shows — the status sits on one nowrap line next
    to the card title, so the full header would crowd it out.
    """
    _install_emit(_STEP_PREFIX + (status or label), log_line='\n--- ' + label + ' ---')

def _run_pip(packages, extra_args=None):
    """Run pip install with streaming output. Returns True on success."""
    cmd = [sys.executable, '-m', 'pip', 'install', '--no-input']
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(packages)
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=_PIP_ENV,
        )
        for line in proc.stdout:
            line = line.rstrip('\n\r')
            if line:
                _install_emit(line)
        proc.wait()
        if proc.returncode != 0:
            log.error('pip exited with code %d for %s', proc.returncode, packages)
        return proc.returncode == 0
    except Exception as exc:
        log.exception('Failed to run pip for %s: %s', packages, exc)
        _install_emit(f'ERROR: {exc}')
        return False

def install_pip_packages():
    """Install required pip packages if missing. Streams output via SocketIO and to
    a per-session install log file (``backend/logs/YYYYMMDD/install-{stamp}-{pid}.log``)."""
    global _INSTALL_LOG_FH

    status = _check_packages()
    all_installed = all(p['installed'] for p in status)

    if all_installed:
        log.info('All required packages already installed.')
        _safe_emit('install:complete', {'skipped': True, 'packages': status})
        return True

    log.info('Some packages missing, installing...')
    _safe_emit('install:start', {'packages': status})

    log_path = _install_log_path()
    log.info('Install log: %s', log_path)
    try:
        _INSTALL_LOG_FH = open(log_path, 'w', encoding='utf-8', buffering=1)
    except Exception as exc:
        log.warning('Could not open install log %s: %s', log_path, exc)
        _INSTALL_LOG_FH = None

    try:
        _install_emit(f'Install log: {log_path}')
        _install_emit(f'Started: {datetime.now().isoformat(timespec="seconds")}')
        _install_emit(f'Python:   {sys.executable}')
        _install_emit(f'Platform: {sys.platform}')
        missing = [p['name'] for p in status if not p['installed']]
        _install_emit(f'Missing:  {missing}')

        if IS_MAC:
            _install_step('Installing PyTorch, mlx-whisper and pyannote', 'Installing ML packages\u2026')
            success = _run_pip(PIP_INSTALL_TARGETS)
        else:
            # Five-phase install:
            #   1. torchruntime (--no-deps to avoid CPU torch)
            #   2. GPU-detected torch via torchruntime install
            #   3. torchaudio/torchvision from matching wheel channel
            #   4. whisperx (--no-deps to keep GPU torch)
            #   5. whisperx's non-torch runtime deps
            # Re-align safety net runs only if channels actually diverged.
            _install_step('Phase 1/5: Installing GPU detection tool', 'Step 1/5: GPU detection tool')
            success = _run_pip(['torchruntime'], extra_args=['--no-deps'])

            if success:
                _install_step('Phase 2/5: Detecting GPU and installing PyTorch', 'Step 2/5: PyTorch')
                _run_torchruntime_install()

                # whisperx declares torchaudio~=2.8.0 and torchvision~=0.23.0 outright,
                # and Phase 4 installs it --no-deps, so both companions are ours to place.
                # They come from the PyTorch wheel index (same +cpu / +cu* as torch);
                # --no-deps avoids re-resolving numpy / pillow / sympy which torchruntime
                # already installed in Phase 2.
                _install_step('Phase 3/5: Installing torchaudio and torchvision',
                              'Step 3/5: torchaudio, torchvision')
                success = _run_pip(
                    ['torchaudio', 'torchvision'],
                    extra_args=[*_companion_pip_extra_args(), '--no-deps'],
                )

            if success:
                # --no-deps prevents whisperx's ``torch~=2.8.0`` pin from
                # downgrading the GPU-enabled torch we installed in Phase 2.
                _install_step('Phase 4/5: Installing whisperx (no-deps)', 'Step 4/5: whisperx')
                success = _run_pip(['whisperx'], extra_args=['--no-deps'])

            if success:
                # Now pull whisperx's non-torch runtime deps.  faster-whisper
                # transitively brings ctranslate2, onnxruntime, tokenizers, av;
                # pyannote-audio brings lightning, scikit-learn, etc.
                _install_step('Phase 5/5: Installing whisperx dependencies', 'Step 5/5: whisperx deps')
                success = _run_pip([
                    'faster-whisper',
                    'pyannote-audio',
                    'transformers',
                    'nltk',
                    'pandas',
                    'omegaconf',
                    'huggingface-hub<1.0.0',
                ])

            if success and not _torch_companion_channels_aligned():
                # Safety net: re-align only when channels actually diverged.
                _install_step('Re-aligning torch stack (channels diverged)', 'Re-aligning torch stack')
                success = _realign_torch_companions()

        if success:
            log.info('Package installation succeeded.')
            _install_emit('\nPackage installation succeeded.')
            _install_emit(f'Finished: {datetime.now().isoformat(timespec="seconds")}')
            _safe_emit('install:complete', {
                'skipped': False,
                'packages': _check_packages(),
            })
            return True
        else:
            msg = 'Package installation failed'
            log.error(msg)
            _install_emit(f'\nERROR: {msg}')
            _install_emit(f'Finished: {datetime.now().isoformat(timespec="seconds")}')
            _safe_emit('install:error', {'error': msg})
            return False
    finally:
        fh, _INSTALL_LOG_FH = _INSTALL_LOG_FH, None
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass
