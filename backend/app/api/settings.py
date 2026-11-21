"""Settings API — read and update application settings."""

import ctypes
import json
import os
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request

from flask import Blueprint, current_app, jsonify, request

from .. import __version__
from ..extensions import db
from ..models.ml_model import MLModel
from ..models.project import Project
from ..models.recording import Recording
from ..models.segment import Segment
from ..models.setting import Setting
from ..ports import supervisor_port
from ..services.export_service import DEFAULT_EXPORT_PROMPT, DEFAULT_EXPORT_PROMPT_RECORDING
from ..services.launcher_layout import (
    INSTALLER_STORAGE_DIR,
    restore_default_launcher_layout_after_reset,
)
from ..services.model_manager import (
    MODEL_REGISTRY,
    download_models,
    get_default_stt_model,
    normalize_stt_model_id,
    remove_model,
    supported_stt_models,
)

GITHUB_REPO = "paul815/PINE"

settings_bp = Blueprint('settings', __name__)

PRESERVE_LIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tools',
)
ROOT_PRESERVE_LIST = 'reset_preserve_root.txt'
BACKEND_PRESERVE_LIST = 'reset_preserve_backend.txt'


def load_preserve_list(filename):
    """Read one reset allowlist from backend/tools/.

    Those two files are the only copy of the lists — reset_win.bat and
    reset.command read the same ones, so the three resets cannot drift apart.
    Read at reset time rather than at import: a missing list must fail the
    reset, not take the whole app down at startup. A truncated list is worse
    than no list at all (everything unlisted is deleted), so refuse it.
    """
    path = os.path.join(PRESERVE_LIST_DIR, filename)
    with open(path, encoding='utf-8') as fh:
        entries = {line.split('#', 1)[0].strip() for line in fh}
    entries.discard('')
    if len(entries) < 8:
        raise RuntimeError(
            f'{path} holds only {len(entries)} entries - refusing to reset '
            'against a truncated allowlist.'
        )
    return entries

# Keys that can be read/written via this API
ALLOWED_KEYS = {
    'font_size', 'font_family', 'theme',
    'export_default_format',
    'export_default_comments',
    'export_default_tags',
    'export_default_project_details',
    'export_default_participant_details',
    'export_default_remove_pii',
    'export_default_include_prompt',
    'export_default_prompt',
    'export_default_prompt_recording',
    'stt_model_id',
    'transcription_timeout_secs',
    'link_recordings',
    'backup_path',
    'backup_include_audio',
    'auto_backup_enabled',
    'auto_backup_interval_hours',
    'backup_retention_count',
    'pii_threshold',
    'last_open_project_id',
    'transcription_complete_sound_enabled',
    'transcription_complete_sound_volume',
    'app_launch_prompt_dismissed',
}

ALLOWED_FONT_FAMILIES = {'system', 'inter', 'source-sans-3', 'open-sans', 'roboto'}

DEFAULTS = {
    'font_size': '13',
    'font_family': 'inter',
    'theme': 'system',
    'export_default_format': 'odt',
    'export_default_comments': 'true',
    'export_default_tags': 'true',
    'export_default_project_details': 'true',
    'export_default_participant_details': 'true',
    'export_default_remove_pii': 'false',
    'export_default_include_prompt': 'false',
    'export_default_prompt': DEFAULT_EXPORT_PROMPT,
    'export_default_prompt_recording': DEFAULT_EXPORT_PROMPT_RECORDING,
    'stt_model_id': get_default_stt_model(),
    'transcription_timeout_secs': '7200',
    'link_recordings': 'false',
    'backup_path': '',
    'backup_include_audio': 'false',
    'auto_backup_enabled': 'false',
    'auto_backup_interval_hours': '24',
    'backup_retention_count': '10',
    'pii_threshold': '0.8',
    'last_open_project_id': '',
    'transcription_complete_sound_enabled': 'true',
    'transcription_complete_sound_volume': '70',
    'app_launch_prompt_dismissed': 'false',
}


def _remove_tree_if_present(path):
    """Best-effort directory removal used by reset cleanup."""
    if not os.path.isdir(path):
        return True
    try:
        shutil.rmtree(path)
    except OSError:
        return False
    return not os.path.exists(path)


def _remove_path_if_present(path):
    if not path:
        return True
    if os.path.isdir(path):
        try:
            shutil.rmtree(path)
        except OSError:
            return False
        return True
    if os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            return False
    return True


def _cleanup_except_preserved(base_dir, preserved_names):
    """Remove all top-level entries except an allowlist."""
    failed_paths = []
    if not os.path.isdir(base_dir):
        return failed_paths

    for entry in os.scandir(base_dir):
        if entry.name in preserved_names:
            continue
        if entry.is_dir(follow_symlinks=False):
            ok = _remove_tree_if_present(entry.path)
        else:
            try:
                os.remove(entry.path)
                ok = not os.path.exists(entry.path)
            except OSError:
                ok = False
        if not ok:
            failed_paths.append(entry.path)
    return failed_paths


def _cleanup_reset_workspace(root_dir):
    """Reset the repo workspace while preserving application source and models."""
    failed_paths = _cleanup_except_preserved(
        root_dir,
        load_preserve_list(ROOT_PRESERVE_LIST),
    )
    failed_paths.extend(
        _cleanup_except_preserved(
            os.path.join(root_dir, 'backend'),
            load_preserve_list(BACKEND_PRESERVE_LIST),
        )
    )
    return failed_paths


def _shutdown_backend_after_reset():
    """Terminate the running app shortly after the reset response is sent."""

    def _exit():
        import time

        try:
            sup_port = supervisor_port()
            req = urllib.request.Request(
                f'http://127.0.0.1:{sup_port}/shutdown',
                method='POST',
                headers={
                    'X-Pine-Supervisor-Token': os.environ.get('PINE_SUPERVISOR_TOKEN', ''),
                },
            )
            urllib.request.urlopen(req, timeout=1.5)
        except Exception:
            pass

        time.sleep(0.8)
        from ..shutdown import graceful_exit
        graceful_exit(0)

    threading.Thread(target=_exit, daemon=True).start()


def _schedule_windows_post_reset_cleanup(root_dir, failed_paths):
    """Hand reset leftovers to the shipped cleanup helper after the app exits.

    Files locked by the running process (chiefly the virtual environment that
    hosts the interpreter) cannot be removed in-process on Windows. We therefore
    write the target paths to a plain text file and launch the committed
    ``backend/tools/reset_post_cleanup.cmd`` as a detached process: it waits for
    the app to exit, then deletes the listed paths. The helper script is shipped
    with the app rather than generated at runtime, and nothing here emits or
    self-deletes batch code.
    """
    cleanup_targets = []
    seen = set()

    for path in list(failed_paths or []) + [
        os.path.join(root_dir, '.venv'),
        os.path.join(root_dir, 'venv'),
    ]:
        norm = os.path.normcase(os.path.abspath(path))
        if norm in seen or not os.path.exists(path):
            continue
        seen.add(norm)
        cleanup_targets.append(os.path.abspath(path))

    if not cleanup_targets:
        return False

    tools_dir = os.path.join(root_dir, 'backend', 'tools')
    script_path = os.path.join(tools_dir, 'reset_post_cleanup.cmd')
    if not os.path.isfile(script_path):
        return False

    targets_path = os.path.join(tools_dir, 'reset_cleanup_targets.txt')
    try:
        with open(targets_path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(cleanup_targets) + '\n')
        # One string, not an argv list. From a list, "cmd /c" receives two quoted
        # paths, and cmd then strips the first and last quote of the line and
        # cuts the command at the first space: under an install path such as
        # "F:\AI Stuff\..." the helper never started and the venv was left
        # half-deleted. With /s cmd strips exactly the outer pair added here.
        subprocess.Popen(
            f'cmd.exe /d /s /c ""{script_path}" "{targets_path}""',
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            | getattr(subprocess, 'DETACHED_PROCESS', 0),
            close_fds=True,
        )
    except OSError:
        return False
    return True

WIN_INSTALL_LAUNCHER_NAME = 'Setup_WIN.bat'
WIN_APP_LAUNCHER_NAME = 'Launch Pine.bat'
WIN_HIDDEN_LAUNCHER_NAME = 'Launch Pine.vbs'
LEGACY_WIN_LAUNCHER_NAMES = ('Launch_WIN.bat',)
START_MENU_LAUNCHER_NAME = 'PINE.lnk'
DESKTOP_LAUNCHER_NAME = 'Launch Pine.lnk'
LEGACY_START_MENU_SHORTCUT_NAMES = ('Launch Pine.lnk',)
LEGACY_DESKTOP_SHORTCUT_NAMES = ('PINE.lnk',)
MAC_INSTALL_LAUNCHER_NAME = 'Setup_MAC.command'
MAC_APP_LAUNCHER_NAME = 'Launch Pine.command'
LEGACY_MAC_LAUNCHER_NAMES = ('Launch_MAC.command',)
MAC_START_MENU_LAUNCHER_NAME = 'Launch Pine.app'
MAC_DESKTOP_LAUNCHER_NAME = 'Launch Pine.app'
LEGACY_MAC_SHORTCUT_NAMES = ('PINE.command', 'MAC_Install.command', 'Launch Pine.command')
# INSTALLER_STORAGE_DIR is imported from launcher_layout — that module owns the layout.
START_MENU_ENABLED_KEY = 'start_menu_launcher_enabled'
DESKTOP_ENABLED_KEY = 'desktop_launcher_enabled'
# Set once the user has answered the first-run shortcut prompt for good, either
# by adding a shortcut or by choosing not to be asked again. Closing the card
# deliberately leaves it alone, so the offer comes back on the next launch.
APP_LAUNCH_PROMPT_DISMISSED_KEY = 'app_launch_prompt_dismissed'


def _is_windows():
    return os.name == 'nt'


def _is_macos():
    return sys.platform == 'darwin'


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def _repo_launcher_path(*launcher_names):
    root = _repo_root()
    backend_dir = os.path.join(root, INSTALLER_STORAGE_DIR)
    for launcher_name in launcher_names:
        for path in (
            os.path.join(backend_dir, launcher_name),
            os.path.join(root, launcher_name),
        ):
            if os.path.isfile(path):
                return path
    return os.path.join(backend_dir, launcher_names[0])


def _launch_win_bat_path():
    return _repo_launcher_path(
        WIN_APP_LAUNCHER_NAME,
        WIN_INSTALL_LAUNCHER_NAME,
        *LEGACY_WIN_LAUNCHER_NAMES,
    )


def _launch_win_vbs_path():
    return _repo_launcher_path(WIN_HIDDEN_LAUNCHER_NAME)


def _ensure_windows_shortcut_target():
    root = _repo_root()
    backend_dir = os.path.join(root, INSTALLER_STORAGE_DIR)
    root_launcher = os.path.join(root, WIN_APP_LAUNCHER_NAME)
    backend_launcher = os.path.join(backend_dir, WIN_APP_LAUNCHER_NAME)
    backend_installer = os.path.join(backend_dir, WIN_INSTALL_LAUNCHER_NAME)
    root_installer = os.path.join(root, WIN_INSTALL_LAUNCHER_NAME)

    if os.path.isfile(root_launcher):
        return root_launcher

    if os.path.isfile(backend_launcher):
        return backend_launcher

    if os.path.isfile(backend_installer):
        return backend_installer

    if os.path.isfile(root_installer):
        return root_installer

    return _repo_launcher_path(
        WIN_APP_LAUNCHER_NAME,
        WIN_INSTALL_LAUNCHER_NAME,
        *LEGACY_WIN_LAUNCHER_NAMES,
    )


def _launch_mac_command_path():
    return _repo_launcher_path(
        MAC_APP_LAUNCHER_NAME,
        MAC_INSTALL_LAUNCHER_NAME,
        *LEGACY_MAC_LAUNCHER_NAMES,
    )


def _launcher_target_path():
    if _is_windows():
        return _ensure_windows_shortcut_target()
    if _is_macos():
        return _launch_mac_command_path()
    return None


def _icon_file_path():
    return os.path.join(_repo_root(), 'backend', 'app', 'static', 'icons', 'pine.ico')


def _mac_icon_file_path():
    return os.path.join(_repo_root(), 'backend', 'app', 'static', 'icons', 'pine.icns')


def _launcher_icon_location():
    icon_path = _icon_file_path()
    if os.path.isfile(icon_path):
        return icon_path
    return os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'System32', 'shell32.dll') + ',220'


def _start_menu_launcher_path():
    if _is_windows():
        appdata = os.environ.get('APPDATA', '').strip()
        if not appdata:
            return None
        return os.path.join(
            appdata,
            'Microsoft',
            'Windows',
            'Start Menu',
            'Programs',
            START_MENU_LAUNCHER_NAME,
        )
    if _is_macos():
        home = os.path.expanduser('~')
        if not home:
            return None
        return os.path.join(home, 'Applications', MAC_START_MENU_LAUNCHER_NAME)
    return None


def _legacy_start_menu_launcher_paths():
    if _is_windows():
        appdata = os.environ.get('APPDATA', '').strip()
        if not appdata:
            return []
        base = os.path.join(
            appdata,
            'Microsoft',
            'Windows',
            'Start Menu',
            'Programs',
        )
        return [os.path.join(base, name) for name in LEGACY_START_MENU_SHORTCUT_NAMES]
    if _is_macos():
        home = os.path.expanduser('~')
        if not home:
            return []
        base = os.path.join(home, 'Applications')
        return [os.path.join(base, name) for name in LEGACY_MAC_SHORTCUT_NAMES]
    return []


def _desktop_shortcut_name():
    return MAC_DESKTOP_LAUNCHER_NAME if _is_macos() else DESKTOP_LAUNCHER_NAME


def _desktop_candidate_dirs(include_nonexistent=False):
    candidates = []

    if _is_windows():
        try:
            # CSIDL_DESKTOPDIRECTORY = 0x0010
            buf = ctypes.create_unicode_buffer(260)
            if ctypes.windll.shell32.SHGetFolderPathW(None, 0x0010, None, 0, buf) == 0:
                known = buf.value.strip()
                if known:
                    candidates.append(known)
        except Exception:
            pass

        one_drive = os.environ.get('OneDrive', '').strip()
        user_profile = os.environ.get('USERPROFILE', '').strip()
        if one_drive:
            candidates.append(os.path.join(one_drive, 'Desktop'))
        if user_profile:
            candidates.append(os.path.join(user_profile, 'Desktop'))
            candidates.append(os.path.join(user_profile, 'OneDrive', 'Desktop'))
    elif _is_macos():
        home = os.path.expanduser('~')
        if home:
            candidates.append(os.path.join(home, 'Desktop'))

    unique = []
    for path in candidates:
        norm = os.path.normcase(os.path.normpath(path))
        if any(os.path.normcase(os.path.normpath(p)) == norm for p in unique):
            continue
        if include_nonexistent or os.path.isdir(path):
            unique.append(path)
    return unique


def _desktop_directory():
    for path in _desktop_candidate_dirs():
        if os.path.isdir(path):
            return path
    candidates = _desktop_candidate_dirs(include_nonexistent=True)
    return candidates[0] if candidates else None


def _desktop_launcher_path():
    desktop_dir = _desktop_directory()
    if not desktop_dir:
        return None
    return os.path.join(desktop_dir, _desktop_shortcut_name())


def _desktop_launcher_paths():
    return [os.path.join(d, _desktop_shortcut_name()) for d in _desktop_candidate_dirs(include_nonexistent=True)]


def _legacy_desktop_launcher_paths():
    legacy_names = LEGACY_MAC_SHORTCUT_NAMES if _is_macos() else LEGACY_DESKTOP_SHORTCUT_NAMES
    return [os.path.join(d, name) for d in _desktop_candidate_dirs(include_nonexistent=True) for name in legacy_names]


def _legacy_start_menu_launcher_path():
    if not _is_windows():
        return None
    appdata = os.environ.get('APPDATA', '').strip()
    if not appdata:
        return None
    return os.path.join(
        appdata,
        'Microsoft',
        'Windows',
        'Start Menu',
        'Programs',
        'PINE',
        'Open PINE.bat',
    )


def _legacy_start_menu_bat_path():
    if not _is_windows():
        return None
    appdata = os.environ.get('APPDATA', '').strip()
    if not appdata:
        return None
    return os.path.join(
        appdata,
        'Microsoft',
        'Windows',
        'Start Menu',
        'Programs',
        'PINE.bat',
    )


def _legacy_desktop_bat_path():
    if not _is_windows():
        return None
    desktop_dir = _desktop_directory()
    if not desktop_dir:
        return None
    return os.path.join(desktop_dir, 'PINE.bat')


def _legacy_desktop_bat_paths():
    if not _is_windows():
        return []
    return [os.path.join(d, 'PINE.bat') for d in _desktop_candidate_dirs(include_nonexistent=True)]


def _cleanup_legacy_start_menu_entry():
    for legacy_link in _legacy_start_menu_launcher_paths():
        _remove_path_if_present(legacy_link)
    legacy_path = _legacy_start_menu_launcher_path()
    if legacy_path and os.path.isfile(legacy_path):
        os.remove(legacy_path)
    legacy_bat = _legacy_start_menu_bat_path()
    if legacy_bat and os.path.isfile(legacy_bat):
        os.remove(legacy_bat)
    if legacy_path:
        legacy_dir = os.path.dirname(legacy_path)
        if os.path.isdir(legacy_dir):
            try:
                if not os.listdir(legacy_dir):
                    os.rmdir(legacy_dir)
            except OSError:
                pass


def _cleanup_legacy_desktop_entry():
    for legacy_link in _legacy_desktop_launcher_paths():
        _remove_path_if_present(legacy_link)
    for legacy_bat in _legacy_desktop_bat_paths():
        if legacy_bat and os.path.isfile(legacy_bat):
            os.remove(legacy_bat)


def _remove_all_app_launcher_entries():
    """Remove app launchers from Start Menu and Desktop, including legacy variants."""
    launcher_path = _start_menu_launcher_path()
    _remove_path_if_present(launcher_path)
    _cleanup_legacy_start_menu_entry()
    _set_launcher_enabled(START_MENU_ENABLED_KEY, False)

    for desktop_path in _desktop_launcher_paths():
        _remove_path_if_present(desktop_path)
    _cleanup_legacy_desktop_entry()
    _set_launcher_enabled(DESKTOP_ENABLED_KEY, False)


def _start_menu_supported():
    target = _launcher_target_path()
    return _start_menu_launcher_path() is not None and bool(target) and os.path.isfile(target)


def _desktop_supported():
    target = _launcher_target_path()
    return _desktop_launcher_path() is not None and bool(target) and os.path.isfile(target)


def _ps_single_quote(value):
    return str(value).replace("'", "''")


def _ps_double_quote(value):
    return str(value).replace('"', '""')


def _create_macos_app_bundle(app_path, launch_cmd):
    contents = os.path.join(app_path, 'Contents')
    macos_dir = os.path.join(contents, 'MacOS')
    resources_dir = os.path.join(contents, 'Resources')
    os.makedirs(macos_dir, exist_ok=True)
    os.makedirs(resources_dir, exist_ok=True)

    launcher_script = os.path.join(macos_dir, 'launch_pine')
    launch_quoted = launch_cmd.replace('\\', '\\\\').replace('"', '\\"')
    with open(launcher_script, 'w', encoding='utf-8', newline='\n') as f:
        f.write('#!/bin/bash\n')
        f.write('set -e\n')
        f.write(f'open -a Terminal "{launch_quoted}"\n')
    os.chmod(launcher_script, 0o755)

    icon_bundle_name = None
    mac_icon = _mac_icon_file_path()
    if os.path.isfile(mac_icon):
        icon_bundle_name = 'AppIcon.icns'
        shutil.copy2(mac_icon, os.path.join(resources_dir, icon_bundle_name))

    bundle_name = os.path.splitext(os.path.basename(app_path))[0]
    plist_path = os.path.join(contents, 'Info.plist')
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDisplayName</key>
  <string>{bundle_name}</string>
  <key>CFBundleExecutable</key>
  <string>launch_pine</string>
  <key>CFBundleIdentifier</key>
  <string>local.pine.launcher</string>
  <key>CFBundleName</key>
  <string>{bundle_name}</string>
"""
    if icon_bundle_name:
        plist += f"""  <key>CFBundleIconFile</key>
  <string>{icon_bundle_name}</string>
"""
    plist += """  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>1.0</string>
  <key>CFBundleVersion</key>
  <string>1</string>
  <key>LSMinimumSystemVersion</key>
  <string>10.13</string>
</dict>
</plist>
"""
    with open(plist_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(plist)


def _write_launcher_file(launcher_path):
    os.makedirs(os.path.dirname(launcher_path), exist_ok=True)
    if _is_windows():
        launch_target = _launch_win_vbs_path()
        root = _repo_root()
        icon_location = _launcher_icon_location()
        if os.path.isfile(launch_target):
            target_path = os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'System32', 'wscript.exe')
            arguments_line = f"$Shortcut.Arguments = '//nologo \"{_ps_double_quote(launch_target)}\"'"
            window_style = 7
        else:
            target_path = _ensure_windows_shortcut_target()
            arguments_line = ''
            window_style = 1
        script = f"""
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut('{_ps_single_quote(launcher_path)}')
$Shortcut.TargetPath = '{_ps_single_quote(target_path)}'
{arguments_line}
$Shortcut.WorkingDirectory = '{_ps_single_quote(root)}'
$Shortcut.WindowStyle = {window_style}
$Shortcut.Description = 'Launch Pine'
$Shortcut.IconLocation = '{_ps_single_quote(icon_location)}'
$Shortcut.Save()
""".strip()
        try:
            subprocess.run(
                ['powershell', '-NoProfile', '-NonInteractive', '-Command', script],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired as e:
            raise OSError('Launcher shortcut creation timed out') from e
        return

    if _is_macos():
        launch_cmd = _launch_mac_command_path()
        if os.path.exists(launcher_path):
            if os.path.isdir(launcher_path):
                shutil.rmtree(launcher_path)
            else:
                os.remove(launcher_path)
        if launcher_path.lower().endswith('.app'):
            _create_macos_app_bundle(launcher_path, launch_cmd)
        else:
            shutil.copy2(launch_cmd, launcher_path)
            os.chmod(launcher_path, os.stat(launcher_path).st_mode | 0o111)
        return

    raise OSError('Unsupported platform for launcher creation')


def _launcher_is_added(launcher_path):
    return bool(launcher_path and os.path.exists(launcher_path))


def _any_launcher_added(paths):
    return any(p and os.path.exists(p) for p in paths)


def _launcher_enabled(setting_key):
    return Setting.get(setting_key, 'false') == 'true'


def _set_launcher_enabled(setting_key, enabled):
    Setting.set(setting_key, 'true' if enabled else 'false')


def _mark_app_launch_prompt_answered():
    """Retire the first-run shortcut offer once a shortcut has been created.

    Set from the add endpoints rather than from the card alone, so adding a
    shortcut in Settings counts as an answer too. Removing it later does not
    bring the offer back — the user has seen the controls and knows where
    they live.
    """
    Setting.set(APP_LAUNCH_PROMPT_DISMISSED_KEY, 'true')


def _launcher_added_state(setting_key, paths):
    added = _any_launcher_added(paths)
    if _launcher_enabled(setting_key) != added:
        _set_launcher_enabled(setting_key, added)
    return added


@settings_bp.route('/app-launch', methods=['GET'])
def app_launch_status():
    start_menu_path = _start_menu_launcher_path()
    start_menu_legacy_paths = _legacy_start_menu_launcher_paths()
    desktop_paths = _desktop_launcher_paths()
    desktop_legacy_paths = _legacy_desktop_launcher_paths() + _legacy_desktop_bat_paths()
    return jsonify({
        # Carried here rather than left to a second /api/settings round trip:
        # the first-run card needs the flag and the two states together before
        # it can decide whether to show at all.
        'prompt_dismissed': Setting.get(APP_LAUNCH_PROMPT_DISMISSED_KEY, 'false') == 'true',
        'start_menu': {
            'supported': _start_menu_supported(),
            'added': _launcher_added_state(
                START_MENU_ENABLED_KEY,
                ([start_menu_path] if start_menu_path else []) + start_menu_legacy_paths,
            ),
            'label': 'Applications' if _is_macos() else 'Start Menu',
            'description': 'Shortcut in Applications folder' if _is_macos() else 'Shortcut in Start Menu',
        },
        'desktop': {
            'supported': _desktop_supported(),
            'added': _launcher_added_state(DESKTOP_ENABLED_KEY, desktop_paths + desktop_legacy_paths),
            'label': 'Desktop shortcut',
            'description': 'Shortcut on your Desktop',
        },
    })


@settings_bp.route('/start-menu', methods=['GET'])
def start_menu_status():
    launcher_path = _start_menu_launcher_path()
    supported = _start_menu_supported()
    return jsonify({
        'supported': supported,
        'added': _launcher_added_state(
            START_MENU_ENABLED_KEY,
            ([launcher_path] if launcher_path else []) + _legacy_start_menu_launcher_paths(),
        ),
    })


@settings_bp.route('/start-menu/add', methods=['POST'])
def add_start_menu_entry():
    if not _start_menu_supported():
        return jsonify({'error': 'App launcher is unavailable on this system.'}), 400

    launcher_path = _start_menu_launcher_path()
    assert launcher_path is not None

    try:
        _write_launcher_file(launcher_path)
    except (subprocess.CalledProcessError, OSError):
        return jsonify({'error': 'Could not create app launcher shortcut.'}), 500
    if not _launcher_is_added(launcher_path):
        _set_launcher_enabled(START_MENU_ENABLED_KEY, False)
        return jsonify({'error': 'Could not create app launcher shortcut.'}), 500

    _cleanup_legacy_start_menu_entry()
    _set_launcher_enabled(START_MENU_ENABLED_KEY, True)
    _mark_app_launch_prompt_answered()

    return jsonify({'ok': True, 'added': True})


@settings_bp.route('/start-menu/remove', methods=['POST'])
def remove_start_menu_entry():
    if not (_is_windows() or _is_macos()):
        return jsonify({'error': 'App launcher is unavailable on this system.'}), 400

    launcher_path = _start_menu_launcher_path()
    _remove_path_if_present(launcher_path)
    _cleanup_legacy_start_menu_entry()
    _set_launcher_enabled(START_MENU_ENABLED_KEY, False)

    return jsonify({'ok': True, 'added': False})


@settings_bp.route('/desktop', methods=['GET'])
def desktop_status():
    launcher_paths = _desktop_launcher_paths()
    legacy_paths = _legacy_desktop_launcher_paths() + _legacy_desktop_bat_paths()
    supported = _desktop_supported()
    return jsonify({
        'supported': supported,
        'added': _launcher_added_state(DESKTOP_ENABLED_KEY, launcher_paths + legacy_paths),
    })


@settings_bp.route('/desktop/add', methods=['POST'])
def add_desktop_entry():
    if not _desktop_supported():
        return jsonify({'error': 'Desktop shortcut is unavailable on this system.'}), 400

    launcher_path = _desktop_launcher_path()
    assert launcher_path is not None
    try:
        _write_launcher_file(launcher_path)
    except (subprocess.CalledProcessError, OSError):
        return jsonify({'error': 'Could not create Desktop shortcut.'}), 500
    if not _launcher_is_added(launcher_path):
        _set_launcher_enabled(DESKTOP_ENABLED_KEY, False)
        return jsonify({'error': 'Could not create Desktop shortcut.'}), 500
    _cleanup_legacy_desktop_entry()
    _set_launcher_enabled(DESKTOP_ENABLED_KEY, True)
    _mark_app_launch_prompt_answered()
    return jsonify({'ok': True, 'added': True})


@settings_bp.route('/desktop/remove', methods=['POST'])
def remove_desktop_entry():
    if not (_is_windows() or _is_macos()):
        return jsonify({'error': 'Desktop shortcut is unavailable on this system.'}), 400

    for launcher_path in _desktop_launcher_paths():
        _remove_path_if_present(launcher_path)
    _cleanup_legacy_desktop_entry()
    _set_launcher_enabled(DESKTOP_ENABLED_KEY, False)

    return jsonify({'ok': True, 'added': False})


@settings_bp.route('', methods=['GET'])
def get_settings():
    result = {}
    for key in ALLOWED_KEYS:
        default = DEFAULTS.get(key, '')
        val = Setting.get(key, default)
        if key == 'stt_model_id':
            normalized = normalize_stt_model_id(val)
            if normalized != val:
                Setting.set('stt_model_id', normalized)
            val = normalized
        result[key] = val

    # Masked HF token
    token = Setting.get('hf_token', '')
    if token and len(token) > 8:
        result['hf_token_masked'] = token[:3] + '****' + token[-4:]
    else:
        result['hf_token_masked'] = ''

    # Models list (only models in registry — excludes removed features like LLM)
    models = MLModel.query.all()
    result['models'] = [m.to_dict() for m in models if m.id in MODEL_REGISTRY]

    # The transcription models this platform can offer, best-quality first, with
    # the download size the settings UI quotes.
    result['stt_models'] = [{
        'id': model_id,
        'name': MODEL_REGISTRY.get(model_id, {}).get('name', model_id),
        'size_bytes': MODEL_REGISTRY.get(model_id, {}).get('size_bytes', 0),
        'installed': _stt_model_installed(model_id),
    } for model_id in supported_stt_models()]
    # The same model runs on Metal on a Mac, where there is no VRAM to quote.
    result['is_mac'] = _is_macos()

    result['app_version'] = __version__

    return jsonify(result)


@settings_bp.route('', methods=['PATCH'])
def update_settings():
    data = request.get_json(force=True)
    updated = {}

    for key in ALLOWED_KEYS:
        if key in data:
            value = str(data[key]).strip()
            if key == 'font_family' and value not in ALLOWED_FONT_FAMILIES:
                continue  # reject invalid font_family
            if key == 'last_open_project_id' and value and not value.isdigit():
                continue  # reject non-numeric project id
            if key == 'stt_model_id':
                value = normalize_stt_model_id(value)
                if not _stt_model_installed(value):
                    # Switching to a model that is not on disk would only fail at
                    # transcribe time; the UI installs first, then switches.
                    return jsonify({
                        'error': f'Model "{value}" is not installed yet',
                        'stt_model_id': value,
                    }), 409
            if key == 'transcription_complete_sound_volume':
                try:
                    iv = int(value)
                    if iv < 0 or iv > 100:
                        continue
                except ValueError:
                    continue
                value = str(iv)
            Setting.set(key, value)
            updated[key] = value

    if updated and {
        'auto_backup_enabled',
        'auto_backup_interval_hours',
        'backup_include_audio',
        'backup_path',
    } & set(updated.keys()):
        from ..services.backup_service import refresh_auto_backup
        refresh_auto_backup(current_app._get_current_object())

    return jsonify({'ok': True, 'updated': updated})



def _parse_version(tag):
    """Strip leading 'v' and split into tuple of ints for comparison."""
    tag = tag.lstrip("vV")
    parts = []
    for p in tag.split("."):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    return tuple(parts)


@settings_bp.route('/check-update', methods=['POST'])
def check_update():
    """Check GitHub for a newer release."""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "PINE-UpdateChecker",
    })
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return jsonify({
                'current_version': __version__,
                'update_available': False,
            'message': 'You’re using the latest version.',
            })
        return jsonify({'error': f'GitHub API error (HTTP {e.code})'}), 502
    except (urllib.error.URLError, OSError):
        return jsonify({'error': 'Could not reach GitHub. Check your internet connection.'}), 502

    latest_tag = data.get("tag_name", "")
    latest_ver = _parse_version(latest_tag)
    current_ver = _parse_version(__version__)
    update_available = latest_ver > current_ver

    return jsonify({
        'current_version': __version__,
        'latest_version': latest_tag.lstrip("vV"),
        'update_available': update_available,
        'release_url': data.get("html_url", f"https://github.com/{GITHUB_REPO}/releases"),
        'release_notes': data.get("body", ""),
    })


def _stt_model_installed(stt_model_id):
    """True when the model is downloaded and ready."""
    row = db.session.get(MLModel, stt_model_id)
    return row is not None and row.status == 'ready'


@settings_bp.route('/stt-model/install', methods=['POST'])
def install_stt_model():
    """Download a transcription model the user has not installed yet."""
    data = request.get_json(force=True) or {}
    model_id = str(data.get('model_id', '')).strip()
    if model_id not in supported_stt_models():
        return jsonify({'error': f'Unknown transcription model "{model_id}"'}), 400

    models_path = Setting.get('models_path', current_app.config['DEFAULT_MODELS_PATH'])
    if not models_path:
        return jsonify({'error': 'Models path not configured'}), 400

    hf_token = Setting.get('hf_token')
    app = current_app._get_current_object()
    model_ids = [model_id]
    download_models(app, model_ids, models_path, hf_token=hf_token, finish_onboarding=False)
    return jsonify({'ok': True, 'model_ids': model_ids})


@settings_bp.route('/pii-model/install', methods=['POST'])
def install_pii_model():
    """Download and install the GLiNER PII model."""
    models_path = Setting.get('models_path', current_app.config['DEFAULT_MODELS_PATH'])
    if not models_path:
        return jsonify({'error': 'Models path not configured'}), 400
    hf_token = Setting.get('hf_token')
    app = current_app._get_current_object()
    download_models(app, ['gliner-pii'], models_path, hf_token=hf_token, finish_onboarding=False)
    return jsonify({'ok': True})


@settings_bp.route('/pii-model/remove', methods=['POST'])
def remove_pii_model():
    """Remove the GLiNER PII model from disk."""
    models_path = Setting.get('models_path', current_app.config['DEFAULT_MODELS_PATH'])
    if not models_path:
        return jsonify({'error': 'Models path not configured'}), 400
    remove_model('gliner-pii', models_path)
    return jsonify({'ok': True})


@settings_bp.route('/reset', methods=['POST'])
def reset_all_data():
    """Permanently delete all projects, recordings, segments, and project folders.
    Requires confirm: 'Yes' in the request body."""
    data = request.get_json(force=True) or {}
    if data.get('confirm') != 'Yes':
        return jsonify({'error': 'Reset requires confirm: "Yes"'}), 400

    projects_path = Setting.get('projects_path', current_app.config['DEFAULT_PROJECTS_PATH'])
    if not os.path.isabs(projects_path):
        root = current_app.config.get(
            'ROOT_DIR',
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        )
        projects_path = os.path.join(root, projects_path)
    else:
        root = current_app.config.get(
            'ROOT_DIR',
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        )

    from ..services.backup_service import _external_safety_backup_dir, create_safety_snapshot

    safety_backup = create_safety_snapshot(
        current_app._get_current_object(),
        reason='reset_all_data',
        include_audio=True,
        backup_dir=_external_safety_backup_dir(),
    )

    # Delete all project folders from disk
    if os.path.isdir(projects_path):
        for name in os.listdir(projects_path):
            path = os.path.join(projects_path, name)
            if os.path.isdir(path):
                try:
                    shutil.rmtree(path)
                except OSError:
                    pass

    # Delete all DB records: clear segment refs, then segments, recordings, then projects
    Recording.query.update({Recording.segment_id: None})
    Segment.query.delete()
    # Bulk deletes skip the ORM cascade, so the per-speaker tracks have to go
    # explicitly or they outlive the recordings they belong to.
    from ..models.recording_track import RecordingTrack
    RecordingTrack.query.delete()
    Recording.query.delete()
    Project.query.delete()
    db.session.commit()

    # Reset onboarding so user goes through setup again
    Setting.set('onboarding_complete', 'false')
    Setting.set('last_open_project_id', '')
    _remove_all_app_launcher_entries()
    restore_default_launcher_layout_after_reset(root)

    failed_temp_paths = _cleanup_reset_workspace(root)
    scheduled_followup_cleanup = False
    if _is_windows():
        scheduled_followup_cleanup = _schedule_windows_post_reset_cleanup(root, failed_temp_paths)
        if scheduled_followup_cleanup:
            _shutdown_backend_after_reset()

    message = 'All project data and non-essential workspace files have been removed. Reload to continue.'
    if scheduled_followup_cleanup:
        message = 'All project data has been removed. PINE will close to finish reset cleanup, including the virtual environment.'
    elif failed_temp_paths:
        message = (
            'All project data has been removed, but some non-essential files could not be deleted. '
            'Close any running tools and try again.'
        )

    return jsonify({
        'ok': True,
        'message': message,
        'shutdown': scheduled_followup_cleanup,
        'safety_backup': safety_backup,
    })
