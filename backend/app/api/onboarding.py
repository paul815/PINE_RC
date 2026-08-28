import json
import os
import secrets
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request

from ..models.ml_model import MLModel
from ..models.setting import Setting
from ..ports import read_supervisor_port_file, supervisor_port, supervisor_url
from ..services.model_manager import (
    IS_MAC,
    MODEL_REGISTRY,
    download_models,
    get_default_stt_model,
    get_models_for_setup,
    normalize_stt_model_id,
    play_install_complete_sound,
    supported_stt_models,
    validate_hf_token,
)
from ..services.system_check import _detect_nvidia_gpu, _existing_ancestor, run_system_check

onboarding_bp = Blueprint('onboarding', __name__)

def _supervisor_status_url() -> str:
    return supervisor_url('status')


def _resolve_supervisor_python() -> Path:
    """Reuse the current interpreter, preferring console python over pythonw."""
    executable = Path(sys.executable)
    if executable.name.lower() == 'pythonw.exe':
        console_exe = executable.with_name('python.exe')
        if console_exe.exists():
            return console_exe
    return executable


def _probe_supervisor_status(timeout: float = 0.6) -> dict | None:
    try:
        with urllib.request.urlopen(_supervisor_status_url(), timeout=timeout) as resp:
            payload = json.loads(resp.read().decode('utf-8'))
            return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _ensure_supervisor_running(timeout: float = 8.0) -> dict | None:
    """Start supervisor if not running; return its token/port info, or None on failure."""
    status = _probe_supervisor_status()
    if status and status.get('supervisor_running'):
        # Supervisor is already up — reuse the token we already know about.
        existing_token = os.environ.get('PINE_SUPERVISOR_TOKEN', '')
        if not existing_token:
            # This backend did not spawn that supervisor, so the token was never
            # in our environment; before it was published, the handoff simply
            # handed the page an empty one and every control call 403'd.
            existing_token = (read_supervisor_port_file() or {}).get('token', '')
            if existing_token:
                os.environ['PINE_SUPERVISOR_TOKEN'] = existing_token
        sup_port = status.get('supervisor_port') or supervisor_port()
        return {'token': existing_token, 'port': int(sup_port)}

    token = secrets.token_urlsafe(32)
    backend_dir = Path(__file__).resolve().parents[2]
    cmd = [str(_resolve_supervisor_python()), 'supervisor.py']
    env = os.environ.copy()
    env['PINE_SUPERVISOR_TOKEN'] = token
    popen_kwargs = {
        'cwd': str(backend_dir),
        'stdout': subprocess.DEVNULL,
        'stderr': subprocess.DEVNULL,
        'env': env,
    }
    if os.name == 'nt':
        flags = subprocess.CREATE_NEW_PROCESS_GROUP
        flags |= getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        popen_kwargs['creationflags'] = flags
    else:
        popen_kwargs['start_new_session'] = True
        popen_kwargs['close_fds'] = True

    try:
        subprocess.Popen(cmd, **popen_kwargs)
    except Exception:
        return None

    deadline = time.time() + timeout
    while time.time() < deadline:
        status = _probe_supervisor_status()
        if status and status.get('supervisor_running'):
            sup_port = status.get('supervisor_port') or supervisor_port()
            # Propagate for this backend process so later requests use the right port.
            os.environ['PINE_SUPERVISOR_PORT'] = str(sup_port)
            os.environ['PINE_SUPERVISOR_TOKEN'] = token
            return {'token': token, 'port': int(sup_port)}
        time.sleep(0.2)
    return None


@onboarding_bp.route('/status', methods=['GET'])
def status():
    completed = Setting.get('onboarding_complete', 'false') == 'true'
    raw_modules = Setting.get('onboarding_modules', '')
    modules = [m for m in raw_modules.split(',') if m] if raw_modules else []
    stt_model_id = normalize_stt_model_id(Setting.get('stt_model_id', get_default_stt_model()))
    if stt_model_id != Setting.get('stt_model_id', get_default_stt_model()):
        Setting.set('stt_model_id', stt_model_id)
    return jsonify({
        'completed': completed,
        'modules': modules,
        'stt_model_id': stt_model_id,
        # What the model step offers, best-quality first.
        'stt_models': [{
            'id': model_id,
            'name': MODEL_REGISTRY.get(model_id, {}).get('name', model_id),
            'size_bytes': MODEL_REGISTRY.get(model_id, {}).get('size_bytes', 0),
        } for model_id in supported_stt_models()],
        'is_mac': IS_MAC,
        'models_path': Setting.get('models_path', current_app.config['DEFAULT_MODELS_PATH']),
        'projects_path': Setting.get('projects_path', current_app.config['DEFAULT_PROJECTS_PATH']),
    })


@onboarding_bp.route('/device', methods=['GET'])
def device_status():
    """Return transcription device info with GPU diagnostics."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_gb = (getattr(props, 'total_memory', 0) or getattr(props, 'total_mem', 0)) / (1024 ** 3)
            warnings = []
            if vram_gb <= 4:
                warnings.append(f'VRAM is {vram_gb:.0f} GB — transcription may be slow or fail on large files')
            return jsonify({
                'device': 'cuda',
                'gpu_name': name,
                'vram_gb': round(vram_gb, 1),
                'has_nvidia_gpu': True,
                'has_cuda': True,
                'warnings': warnings,
                'message': f'Using GPU: {name} ({vram_gb:.0f} GB)',
            })
        if getattr(torch.backends.mps, 'is_available', lambda: False)():
            return jsonify({
                'device': 'mps',
                'has_nvidia_gpu': False,
                'has_cuda': False,
                'warnings': [],
                'message': 'Apple Silicon detected — using Metal acceleration via mlx-whisper',
            })
        nvidia_gpu = _detect_nvidia_gpu()
        if nvidia_gpu:
            return jsonify({
                'device': 'cpu',
                'gpu_name': nvidia_gpu,
                'has_nvidia_gpu': True,
                'has_cuda': False,
                'warnings': [],
                'cuda_download_url': 'https://developer.nvidia.com/cuda-downloads',
                'message': f'{nvidia_gpu} detected but CUDA not available — install CUDA Toolkit to enable GPU acceleration.',
            })
        return jsonify({
            'device': 'cpu',
            'has_nvidia_gpu': False,
            'has_cuda': False,
            'warnings': [],
            'message': 'No compatible GPU detected — using CPU. Transcription will be slow.',
        })
    except ImportError:
        return jsonify({'device': 'unknown', 'has_nvidia_gpu': False, 'has_cuda': False, 'warnings': [], 'message': 'PyTorch not installed'}), 500


@onboarding_bp.route('/system-check', methods=['GET'])
def system_check():
    models_path = Setting.get('models_path', current_app.config['DEFAULT_MODELS_PATH'])
    checks = run_system_check(models_path)
    has_blockers = any(c['name'] == 'Python' and c['status'] in ('err', 'warn') for c in checks)
    return jsonify({'checks': checks, 'has_blockers': has_blockers})


@onboarding_bp.route('/modules', methods=['POST'])
def set_modules():
    data = request.get_json(force=True)
    modules = data.get('modules', [])
    Setting.set('onboarding_modules', ','.join(modules))

    # The recognition engine is chosen on this step too. An absent or unrunnable
    # id normalizes to the platform default, so an older client that posts only
    # modules behaves exactly as before.
    stt_model_id = normalize_stt_model_id(
        data.get('stt_model_id') or Setting.get('stt_model_id', get_default_stt_model()))
    Setting.set('stt_model_id', stt_model_id)
    model_ids = get_models_for_setup(modules, stt_model_id)
    total = sum(
        (db.session.get(MLModel, mid).size_bytes or 0)
        for mid in model_ids
        if db.session.get(MLModel, mid)
    )
    return jsonify({'ok': True, 'total_size_bytes': total, 'model_ids': model_ids})


@onboarding_bp.route('/stt-model', methods=['POST'])
def set_stt_model():
    """Record which transcription model setup should download."""
    data = request.get_json(force=True) or {}
    requested = str(data.get('stt_model_id', '')).strip()
    # An unsupported id is not an error here: normalize falls back to the
    # platform default, which is what setup would have downloaded anyway.
    stt_model_id = normalize_stt_model_id(requested) if requested else get_default_stt_model()
    Setting.set('stt_model_id', stt_model_id)
    raw_modules = Setting.get('onboarding_modules', '')
    modules = [m for m in raw_modules.split(',') if m]
    model_ids = get_models_for_setup(modules, stt_model_id)
    total = sum(
        (db.session.get(MLModel, mid).size_bytes or 0)
        for mid in model_ids
        if db.session.get(MLModel, mid)
    )
    return jsonify({'ok': True, 'stt_model_id': stt_model_id, 'total_size_bytes': total, 'model_ids': model_ids})


@onboarding_bp.route('/storage', methods=['POST'])
def set_storage():
    data = request.get_json(force=True)
    models_path = data.get('models_path', '').strip()
    projects_path = data.get('projects_path', '').strip()

    if not models_path or not projects_path:
        return jsonify({'error': 'Both paths are required'}), 400

    try:
        os.makedirs(models_path, exist_ok=True)
        os.makedirs(projects_path, exist_ok=True)
    except OSError as exc:
        return jsonify({'error': f'Cannot create directories: {exc}'}), 400

    Setting.set('models_path', models_path)
    Setting.set('projects_path', projects_path)

    drive = os.path.splitdrive(models_path)[0]
    disk_path = (drive + os.sep) if drive else _existing_ancestor(models_path)
    try:
        disk = shutil.disk_usage(disk_path)
        return jsonify({
            'ok': True,
            'disk_free_bytes': disk.free,
            'disk_total_bytes': disk.total,
        })
    except Exception:
        return jsonify({'ok': True})


@onboarding_bp.route('/hf-token', methods=['POST'])
def set_hf_token():
    data = request.get_json(force=True)
    token = data.get('token', '').strip()
    if not token:
        return jsonify({'valid': False, 'error': 'Token is empty'}), 400

    result = validate_hf_token(token)
    if result['valid']:
        Setting.set('hf_token', token)
    return jsonify(result)


@onboarding_bp.route('/download/start', methods=['POST'])
def start_download():
    raw_modules = Setting.get('onboarding_modules', '')
    modules = [m for m in raw_modules.split(',') if m]
    models_path = Setting.get('models_path', current_app.config['DEFAULT_MODELS_PATH'])
    hf_token = Setting.get('hf_token')

    if not models_path:
        return jsonify({'error': 'Models path not configured'}), 400

    stt_model_id = normalize_stt_model_id(Setting.get('stt_model_id', get_default_stt_model()))
    Setting.set('stt_model_id', stt_model_id)
    model_ids = get_models_for_setup(modules, stt_model_id)
    download_models(current_app._get_current_object(), model_ids, models_path, hf_token)
    return jsonify({'ok': True, 'model_ids': model_ids})


@onboarding_bp.route('/download/status', methods=['GET'])
def download_status():
    models = MLModel.query.all()
    return jsonify({'models': [m.to_dict() for m in models]})


@onboarding_bp.route('/play-install-sound', methods=['POST'])
def play_install_sound():
    """Play the install-complete sound from the terminal. Called by frontend when download completes and user has the checkbox checked."""
    play_install_complete_sound(current_app._get_current_object())
    return jsonify({'ok': True})


@onboarding_bp.route('/handoff/prepare', methods=['POST'])
def prepare_handoff():
    """Start supervisor so the first-launch backend can restart under supervision."""
    if Setting.get('onboarding_complete', 'false') != 'true':
        return jsonify({'ok': False, 'error': 'Onboarding is not complete yet'}), 409

    info = _ensure_supervisor_running()
    if not info:
        return jsonify({'ok': False, 'error': 'Supervisor did not start in time'}), 503

    return jsonify({
        'ok': True,
        'supervisor_running': True,
        'supervisor_token': info.get('token', ''),
        'supervisor_port': info.get('port'),
        'main_url': '/',
    })


@onboarding_bp.route('/browse-folder', methods=['GET'])
def browse_folder():
    """Open native OS folder picker dialog, return selected path."""
    raw = (request.args.get('initial') or '').strip()
    if raw:
        cand = os.path.normpath(os.path.expanduser(raw))
        initialdir = cand if os.path.isdir(cand) else str(Path.home())
    else:
        initialdir = str(Path.home())
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', True)
        selected = filedialog.askdirectory(initialdir=initialdir, title='Select Folder')
        root.destroy()
        return jsonify({'path': selected or None})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@onboarding_bp.route('/defaults', methods=['GET'])
def defaults():
    """Return default paths and disk info for the storage step."""
    mp = current_app.config['DEFAULT_MODELS_PATH']
    pp = current_app.config['DEFAULT_PROJECTS_PATH']

    drive = os.path.splitdrive(mp)[0]
    disk_path = (drive + os.sep) if drive else _existing_ancestor(mp)
    try:
        disk = shutil.disk_usage(disk_path)
        free = disk.free
        total = disk.total
        used_pct = int((disk.used / disk.total) * 100)
    except Exception:
        free = total = used_pct = 0

    return jsonify({
        'models_path': mp,
        'projects_path': pp,
        'disk_free_bytes': free,
        'disk_total_bytes': total,
        'disk_used_percent': used_pct,
    })


# Need db import for the modules endpoint
from ..extensions import db  # noqa: E402
