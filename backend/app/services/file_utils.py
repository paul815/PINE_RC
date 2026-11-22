"""Shared file I/O utilities for safe, atomic writes and reads."""

import errno
import json
import os
import tempfile
import time
from pathlib import PurePosixPath

from . import windows_io

# A rename is one atomic step on POSIX: readers see the old inode or the new
# one, and a second renamer simply wins or loses. Windows goes through
# MoveFileExW instead, and for the moment it takes to swap the file in it
# refuses everybody — the other replacer and every would-be reader alike get
# ERROR_ACCESS_DENIED (5) or ERROR_SHARING_VIOLATION (32). The search indexer
# and antivirus scanners open these files too, so the window is not only ours.
#
# windows_io closes the window for the handles we own; this stays as the
# backstop, for the handles we do not and for volumes it cannot help on.
_TRANSIENT_WINDOWS_ERRORS = frozenset({5, 32})

# The gap to aim for is microseconds wide, so a 1ms sleep steps straight over
# it — spin first, then back off for the case where the holder is something
# slower than us, like a scanner reading what we have only just written.
_SPINS_BEFORE_SLEEPING = 5

# Writers need the longer budget: they are the side being blocked, and the
# side whose failure reaches the user as a 500 or a lost edit. A read that
# gives up is refetched by the next request.
_WRITE_ATTEMPTS = 60
_READ_ATTEMPTS = 20


def _is_transient(exc):
    """Is this the replace window, or a real refusal?

    The two halves of the problem report it differently, which is easy to get
    wrong: os.replace fails from the Win32 API and carries .winerror, so 5 and
    32 are visible. open() fails from the C runtime, which has already
    flattened both to EACCES and sets no .winerror at all — matching on
    .winerror alone silently never retries a single read.

    When .winerror is there it is the authoritative answer, so an error that
    is genuinely something else is not retried on an errno coincidence. POSIX
    never reaches the errno branch: EACCES there means what it says.
    """
    winerror = getattr(exc, 'winerror', None)
    if winerror is not None:
        return winerror in _TRANSIENT_WINDOWS_ERRORS
    return os.name == 'nt' and exc.errno == errno.EACCES


def _through_the_replace_window(operation, attempts):
    """Run *operation*, retrying the moment Windows spends swapping a file.

    A real permission problem still raises, just about a second later at the
    write budget. On POSIX no exception carries .winerror, so nothing here is
    ever retried.
    """
    for attempt in range(attempts):
        try:
            return operation()
        except OSError as exc:
            if attempt == attempts - 1 or not _is_transient(exc):
                raise
            if attempt >= _SPINS_BEFORE_SLEEPING:
                backoff = attempt - _SPINS_BEFORE_SLEEPING
                time.sleep(min(0.0005 * (2 ** backoff), 0.02))


def _rename(tmp_path, path):
    """Rename with POSIX semantics where Windows offers them, os.replace where not.

    Three recordings uploaded into one project at once all rewrite that
    project's README.md, and two of the three replaces would collide: one
    upload returned 500 with WinError 5 roughly one run in forty.

    The fallback is not a formality — a project folder on an exFAT stick
    cannot do FileRenameInfoEx, and neither can Windows before 1607. Falling
    through leaves exactly the behaviour this module had before, retries and
    all, so the worst case is no worse than it was.
    """
    if windows_io.AVAILABLE:
        try:
            windows_io.rename_posix(tmp_path, path)
            return
        except OSError as exc:
            if getattr(exc, 'winerror', None) not in windows_io.RENAME_UNSUPPORTED:
                raise
    os.replace(tmp_path, path)


def _replace_atomically(tmp_path, path):
    """_rename, retried through whatever transient refusals are left."""
    _through_the_replace_window(lambda: _rename(tmp_path, path), _WRITE_ATTEMPTS)


def _open_for_read(path, encoding):
    """The one place that decides how these files get opened."""
    if windows_io.AVAILABLE:
        return windows_io.open_shared(path, encoding=encoding)
    return open(path, encoding=encoding)


def atomic_read_text(path, encoding='utf-8'):
    """Read a file that atomic_write_* may be replacing underneath us.

    The mirror of _replace_atomically, and the more dangerous half. An open()
    that lands in the window raises PermissionError, and every caller of these
    files reads defensively: attachments come back empty, tags and themes
    vanish from the analysis screen, and update_annotations merges into a
    default dict and writes *that* back over real speaker labels. A transient
    failure to read therefore turned into permanent data loss.

    Reads whole, so a caller can never be handed half a file. On Windows it
    also shares delete, so holding the file for the length of the read does
    not push the writer into its retries.
    """
    def read():
        with _open_for_read(path, encoding) as fh:
            return fh.read()

    return _through_the_replace_window(read, _READ_ATTEMPTS)


def atomic_read_json(path, encoding='utf-8'):
    """atomic_read_text, parsed. Raises json.JSONDecodeError on a bad file."""
    return json.loads(atomic_read_text(path, encoding=encoding))


def atomic_write_json(path, data):
    """Write JSON to a temp file then atomically replace the target.

    Uses tempfile.mkstemp in the same directory as the target so that
    the replace stays on one filesystem.  If anything goes wrong, the temp
    file is cleaned up and the original is untouched.
    """
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        _replace_atomically(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_text(path, text):
    """Write text to a temp file then atomically replace the target."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text)
        _replace_atomically(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def save_upload_atomically(file_storage, path):
    """Stream an upload into a sibling temp file, then swap it into *path*.

    ``file_storage.save(path)`` writes straight to the final name, so a backend
    that dies mid-upload — or a browser that drops a 300 MB request — leaves a
    truncated file under the real recording name. Everything downstream then
    treats that stump as the recording: on 2026-08-26 a restart re-queued one
    and ffmpeg failed on it with a CalledProcessError the user had to decode.
    Writing beside the destination and replacing it means the final name only
    ever appears complete; an interrupted upload leaves a .part nothing reads.
    """
    directory = os.path.dirname(path) or '.'
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix='.upload-', suffix='.part')
    os.close(fd)
    try:
        file_storage.save(tmp_path)
        _replace_atomically(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def safe_child_path(root, rel_path):
    """Resolve *rel_path* under *root*, or raise ``ValueError`` if it leaves it.

    For names that come from outside the process — a backup archive, a DB row
    a restore wrote — where an absolute path, a drive letter or a ``..`` would
    otherwise walk ``os.path.join`` straight out of *root*. Both separators
    count; an empty *rel_path* resolves to *root* itself.
    """
    parts = []
    for part in PurePosixPath((rel_path or '').replace('\\', '/')).parts:
        if part in ('', '.'):
            continue
        if part == '..' or ':' in part or '\0' in part:
            raise ValueError(f'Unsafe path: {rel_path!r}')
        parts.append(part)

    root_abs = os.path.abspath(root)
    dest = os.path.abspath(os.path.join(root_abs, *parts))
    try:
        inside = os.path.commonpath([root_abs, dest]) == root_abs
    except ValueError:  # different drives on Windows
        inside = False
    if not inside:
        raise ValueError(f'Path escapes {root_abs!r}: {rel_path!r}')
    return dest


def safe_project_dir(projects_root, folder_name):
    """Return the directory of the project folder *folder_name*, or raise.

    A project folder is exactly one path component naming a direct child of
    *projects_root*. Anything else — empty, ``.``/``..``, a separator, a drive,
    or a trailing dot or space, which Windows strips so that ``'.. '`` becomes
    ``'..'`` — raises ``ValueError``. Callers that delete or move the folder
    must go through this rather than ``os.path.join``.
    """
    name = folder_name if isinstance(folder_name, str) else ''
    if (not name or name in ('.', '..') or name != name.rstrip('. ')
            or any(ch in name for ch in '/\\:\0')):
        raise ValueError(f'Unsafe project folder name: {folder_name!r}')

    root_abs = os.path.abspath(projects_root)
    dest = os.path.abspath(os.path.join(root_abs, name))
    if os.path.dirname(dest) != root_abs:
        raise ValueError(f'Unsafe project folder name: {folder_name!r}')
    return dest


def claim_free_path(directory, filename, max_tries=10000):
    """Reserve a non-colliding path inside *directory* and return it.

    Returns ``(path, name)`` for an empty file this call just created. The
    caller is expected to overwrite it.

    The obvious version of this — walk a counter while ``os.path.exists`` is
    true, then write — has a window between the check and the write. Two
    uploads of the same filename arriving together both see the name as free
    and the second one overwrites the first recording's audio, leaving two DB
    rows pointing at one file. ``O_CREAT | O_EXCL`` closes that window: the
    filesystem hands the name to exactly one caller and the loser retries with
    the next counter.

    Names go ``interview.mp3``, ``interview_2.mp3``, ``interview_3.mp3`` …,
    matching what the check-then-write version produced.
    """
    os.makedirs(directory, exist_ok=True)
    base, extension = os.path.splitext(filename)
    name = filename
    for counter in range(1, max_tries + 1):
        if counter > 1:
            name = f'{base}_{counter}{extension}'
        path = os.path.join(directory, name)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            continue
        os.close(fd)
        return path, name
    raise OSError(f'Could not find a free filename for {filename!r} in {directory!r}')
