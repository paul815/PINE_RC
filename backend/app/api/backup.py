"""Backup API for create, list, restore, delete, and upload."""

import os
import subprocess
import sys
import tempfile
import threading

from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from ..models.setting import Setting
from ..services.file_utils import claim_free_path
from ..services.backup_service import (
    _backup_dir,
    create_backup,
    delete_backup,
    get_manifest,
    list_backups,
    restore_backup,
    validate_backup_zip,
)

backup_bp = Blueprint('backup', __name__)


@backup_bp.route('', methods=['POST'])
def create():
    """Create a new backup. Optional body: ``{ include_audio: bool }``."""
    data = request.get_json(silent=True) or {}
    include_audio_setting = Setting.get('backup_include_audio', 'false') == 'true'
    include_audio = data.get('include_audio', include_audio_setting)
    app = current_app._get_current_object()
    create_backup(app, include_audio=bool(include_audio))
    return jsonify({'ok': True, 'message': 'Backup started.'})


@backup_bp.route('', methods=['GET'])
def index():
    """List all existing backups with metadata."""
    app = current_app._get_current_object()
    return jsonify(list_backups(app))


_SW_RESTORE = 9
_EXPLORER_CLASSES = ('CabinetWClass', 'ExploreWClass')


def _raise_windows_folder(folder_name, timeout=4.0):
    """Pull the freshly opened Explorer window in front of the browser.

    Windows refuses to let a background process hand focus to a new window, so
    the folder opens behind PINE and only blinks in the taskbar. Attaching to
    the foreground window's input queue lifts that restriction for the duration
    of the call. Runs off-thread: finding the window takes a moment.
    """
    import ctypes
    import time

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    user32.GetForegroundWindow.restype = ctypes.c_void_p
    user32.GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    target = folder_name.lower()
    found = []

    enum_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def _visit(hwnd, _lparam):
        cls = ctypes.create_unicode_buffer(64)
        user32.GetClassNameW(hwnd, cls, 64)
        if cls.value not in _EXPLORER_CLASSES:
            return True
        title = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(hwnd, title, 512)
        if target and target in title.value.lower():
            found.append(hwnd)
            return False
        return True

    callback = enum_proc(_visit)
    deadline = time.monotonic() + timeout
    while not found and time.monotonic() < deadline:
        user32.EnumWindows(callback, 0)
        if found:
            break
        time.sleep(0.15)
    if not found:
        return False

    hwnd = found[0]
    foreground = user32.GetForegroundWindow()
    fg_thread = user32.GetWindowThreadProcessId(ctypes.c_void_p(foreground), None)
    cur_thread = kernel32.GetCurrentThreadId()
    attached = bool(user32.AttachThreadInput(fg_thread, cur_thread, True))
    try:
        user32.ShowWindow(ctypes.c_void_p(hwnd), _SW_RESTORE)
        user32.BringWindowToTop(ctypes.c_void_p(hwnd))
        user32.SetForegroundWindow(ctypes.c_void_p(hwnd))
    finally:
        if attached:
            user32.AttachThreadInput(fg_thread, cur_thread, False)
    return True


@backup_bp.route('/open-folder', methods=['POST'])
def open_folder():
    """Reveal the backup folder in the OS file manager, in front of the app."""
    app = current_app._get_current_object()
    try:
        path = _backup_dir(app)
        if sys.platform == 'darwin':
            subprocess.Popen(['open', path])
        elif os.name == 'nt':
            path = os.path.normpath(path)
            # explorer.exe is the shell itself, so it places the window better
            # than an in-process ShellExecute from this background server would.
            subprocess.Popen(['explorer', path])
            threading.Thread(
                target=_raise_windows_folder,
                args=(os.path.basename(path.rstrip('\\/')),),
                daemon=True,
            ).start()
        else:
            subprocess.Popen(['xdg-open', path])
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
    return jsonify({'ok': True, 'path': path})


@backup_bp.route('/<filename>/manifest', methods=['GET'])
def manifest(filename):
    """Read manifest from a specific backup ZIP."""
    app = current_app._get_current_object()
    manifest_data = get_manifest(app, filename)
    if manifest_data is None:
        return jsonify({'error': 'Backup not found or invalid.'}), 404
    return jsonify(manifest_data)


@backup_bp.route('/restore', methods=['POST'])
def restore():
    """Restore from a backup ZIP."""
    data = request.get_json(force=True)
    filename = data.get('filename')
    if not filename:
        return jsonify({'error': 'filename is required.'}), 400

    app = current_app._get_current_object()
    restore_backup(
        app,
        filename=filename,
        project_folders=data.get('project_folders'),
        restore_settings=data.get('restore_settings', False),
        conflict_strategy=data.get('conflict_strategy', 'skip'),
    )
    return jsonify({'ok': True, 'message': 'Restore started.'})


@backup_bp.route('/<filename>', methods=['DELETE'])
def delete(filename):
    """Delete a specific backup file."""
    app = current_app._get_current_object()
    deleted = delete_backup(app, filename)
    if not deleted:
        return jsonify({'error': 'Backup not found.'}), 404
    return jsonify({'ok': True})


@backup_bp.route('/upload', methods=['POST'])
def upload():
    """Upload an external backup ZIP file."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
    file_obj = request.files['file']
    if not file_obj.filename or not file_obj.filename.endswith('.zip'):
        return jsonify({'error': 'File must be a .zip archive.'}), 400

    app = current_app._get_current_object()
    default = os.path.join(app.config.get('ROOT_DIR', ''), 'backups')
    backup_path = Setting.get('backup_path', default) or default
    if not os.path.isabs(backup_path):
        backup_path = os.path.join(app.config.get('ROOT_DIR', ''), backup_path)
    os.makedirs(backup_path, exist_ok=True)

    safe_name = secure_filename(file_obj.filename) or 'uploaded_backup.zip'

    # Validate under a temporary name and only then claim a free final one.
    # Saving straight to the uploaded name replaced any backup already called
    # that, and a failed check then deleted the file — taking the original
    # with it.
    fd, tmp_path = tempfile.mkstemp(dir=backup_path, prefix='.upload-', suffix='.part')
    os.close(fd)
    try:
        file_obj.save(tmp_path)
        manifest_data, error = validate_backup_zip(tmp_path)
        if error:
            return jsonify({'error': error}), 400
        dest, safe_name = claim_free_path(backup_path, safe_name)
        os.replace(tmp_path, dest)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return jsonify({
        'ok': True,
        'filename': safe_name,
        'project_count': manifest_data.get('project_count', 0),
    })
