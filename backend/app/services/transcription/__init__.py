"""Transcription orchestration (Flask side).

This package owns everything that touches the database, SocketIO or the
UI: the job queue, the watchdog, status events, cancellation and the
language-confirmation flow. The ML pipeline itself (WhisperX / pyannote)
lives in the Flask-free ``ml_worker`` package and runs in a separate worker
process — see ``worker_client.py`` and ``job_runner.py``.

Module layout:
    __init__.py      — queue, worker thread, watchdog, status/cancel/language API
    job_runner.py    — resolves a job from the DB, drives the worker, persists results
    worker_client.py — ML worker subprocess lifecycle + JSON-lines RPC
"""

import logging
import queue
import threading
import time as _time

# Re-exports: parts of the ML package that the API layer and tests reach
# through this module for convenience/compatibility.
from ml_worker.audio import fmt_elapsed as _fmt_elapsed
from ml_worker.audio import write_wav as _write_wav  # noqa: F401
from ml_worker.constants import (  # noqa: F401
    CHUNK_OVERLAP_SEC,
    CHUNK_SIZE_SEC,
    CHUNK_THRESHOLD_SEC,
    SPEAKER_LABELS,
)
from ml_worker.errors import TranscriptionCancelled  # noqa: F401

from ...extensions import db, socketio
from ...models.recording import Recording

log = logging.getLogger(__name__)

_job_queue = queue.Queue()
_queued_ids = []                 # recording ids waiting in the queue, FIFO (for position display)
_queue_lock = threading.Lock()
_worker_thread = None
_worker_generation = 0
_current_recording_id = None
_last_heartbeat = 0.0
_watchdog_thread = None
_cancel_flags = {}          # recording_id -> threading.Event
_cancel_flags_lock = threading.Lock()

# Job thread blocks on language confirmation (low confidence); HTTP sets language + Event.
_language_waiters = {}       # recording_id -> {'event': Event, 'language': str|None, 'cancelled': bool}
_language_waiters_lock = threading.Lock()


def _build_transcript_filename(recording) -> str:
    """Build a stable, collision-safe transcript filename for a recording."""
    import os
    existing = (getattr(recording, 'transcript_path', '') or '').strip()
    if existing:
        return existing

    stored_name = getattr(recording, 'stored_name', '') or ''
    stored_basename = os.path.splitext(os.path.basename(stored_name))[0] or 'recording'
    recording_id = getattr(recording, 'id', None)
    if recording_id is not None:
        return f'{stored_basename}_{recording_id}_transcript.json'
    return f'{stored_basename}_transcript.json'


# ── status / heartbeat ──

def _touch_heartbeat():
    """Update the last-activity timestamp for the watchdog."""
    global _last_heartbeat
    _last_heartbeat = _time.monotonic()


def _emit_status(recording_id, status, stage='', message='', percent=None,
                 touch=True, **extra):
    # Queue-position pings (touch=False) must not reset the watchdog heartbeat,
    # which only tracks progress of the recording actively being transcribed.
    if touch:
        _touch_heartbeat()
    payload = {
        'recording_id': recording_id,
        'status': status,
        'stage': stage,
        'message': message,
    }
    if percent is not None:
        payload['percent'] = percent
    if extra:
        payload.update(extra)
    try:
        socketio.emit('transcription:status', payload)
    except Exception:
        pass


def _emit_queue_positions():
    """Broadcast each waiting recording's place in line so the UI can show '#k of N'."""
    with _queue_lock:
        ids = list(_queued_ids)
    total = len(ids)
    for pos, rid in enumerate(ids, start=1):
        _emit_status(rid, 'pending', stage='queued',
                     message=f'In queue (#{pos} of {total})',
                     percent=0, touch=False,
                     queue_position=pos, queue_total=total)


# ── cancellation ──

def _register_cancel(recording_id):
    ev = threading.Event()
    with _cancel_flags_lock:
        _cancel_flags[recording_id] = ev
    return ev


def _unregister_cancel(recording_id):
    with _cancel_flags_lock:
        _cancel_flags.pop(recording_id, None)


def _check_cancel(recording_id):
    """Raise TranscriptionCancelled if a cancel was requested for this recording."""
    with _cancel_flags_lock:
        ev = _cancel_flags.get(recording_id)
    if ev is not None and ev.is_set():
        raise TranscriptionCancelled(recording_id)


def cancel_transcription(recording_id):
    """Signal recording_id to stop. Returns True if an active job was found."""
    woke_language_wait = False
    with _language_waiters_lock:
        lw = _language_waiters.get(recording_id)
        if lw is not None:
            lw['cancelled'] = True
            lw['event'].set()
            woke_language_wait = True
    found_active = False
    with _cancel_flags_lock:
        ev = _cancel_flags.get(recording_id)
        if ev is not None:
            ev.set()
            found_active = True
    if found_active:
        # Forward to the worker process so it can stop at the next checkpoint
        # instead of waiting for the job thread to notice the flag.
        from .worker_client import notify_cancel
        notify_cancel(recording_id)
    return found_active or woke_language_wait


# ── language confirmation ──

def notify_language_chosen(recording_id: int, language_code: str) -> bool:
    """Called from Flask when the user submits a language. Returns False if nothing was waiting."""
    code = (language_code or '').strip().lower()
    if not code:
        return False
    with _language_waiters_lock:
        state = _language_waiters.get(recording_id)
        if state is None:
            return False
        state['language'] = code
        state['event'].set()
    log.info('Language chosen for recording %s: %s', recording_id, code)
    return True


# ── queue / worker ──

def enqueue(recording_id):
    """Add a recording to the transcription queue."""
    with _queue_lock:
        _queued_ids.append(recording_id)
    _job_queue.put(recording_id)
    log.info('Enqueued recording %d for transcription', recording_id)
    _emit_queue_positions()


def busy_snapshot():
    """What the worker is doing right now, for callers outside this module.

    The supervisor reads this over ``/api/health`` before it acts on an expired
    browser lease. A backgrounded tab stops heartbeating within seconds, and a
    long recording outlives the grace that follows — killing the backend there
    throws away however much of the transcription had already run.
    """
    with _queue_lock:
        queued = len(_queued_ids)
    recording_id = _current_recording_id
    return {
        'busy': recording_id is not None or queued > 0,
        'recording_id': recording_id,
        'queued': queued,
    }


def _worker_loop(app, generation):
    """Background thread that processes transcription jobs one at a time.

    The heavy lifting happens in the ML worker subprocess; this thread only
    resolves the job, relays events and persists results (see job_runner).
    """
    global _current_recording_id
    while True:
        if generation != _worker_generation:
            log.info('Worker gen %d superseded by %d, exiting.',
                     generation, _worker_generation)
            return

        recording_id = _job_queue.get()
        with _queue_lock:
            try:
                _queued_ids.remove(recording_id)
            except ValueError:
                pass
        # The job left the queue — refresh everyone else's position.
        _emit_queue_positions()

        if generation != _worker_generation:
            log.info('Worker gen %d superseded after dequeue, exiting.',
                     generation, _worker_generation)
            _job_queue.task_done()
            return

        _current_recording_id = recording_id
        _touch_heartbeat()
        _register_cancel(recording_id)

        try:
            from .job_runner import run_transcription_job
            run_transcription_job(app, recording_id)
        except TranscriptionCancelled:
            log.info('Recording %d transcription cancelled.', recording_id)
            with app.app_context():
                recording = db.session.get(Recording, recording_id)
                if recording:
                    recording.transcription_status = 'cancelled'
                    recording.error_message = ''
                    db.session.commit()
            _emit_status(recording_id, 'cancelled')
        except Exception as exc:
            log.exception('Transcription failed for recording %d: %s',
                          recording_id, exc)
            with app.app_context():
                recording = db.session.get(Recording, recording_id)
                if recording:
                    recording.transcription_status = 'error'
                    recording.error_message = str(exc)[:500]
                    db.session.commit()
            _emit_status(recording_id, 'error', message=str(exc)[:200])
        finally:
            _unregister_cancel(recording_id)
            _current_recording_id = None
            _job_queue.task_done()


def start_worker(app):
    """Start the background transcription worker thread (idempotent)."""
    global _worker_thread
    if _worker_thread is not None and _worker_thread.is_alive():
        return
    _worker_thread = threading.Thread(
        target=_worker_loop, args=(app, _worker_generation),
        daemon=True, name=f'transcription-worker-gen{_worker_generation}')
    _worker_thread.start()
    log.info('Transcription worker started (generation %d).', _worker_generation)


# ── watchdog ──

def _get_watchdog_timeout(app):
    """Get timeout from settings, defaulting to 7200 seconds (2 hours)."""
    try:
        with app.app_context():
            from ...models.setting import Setting
            val = Setting.get('transcription_timeout_secs', '7200')
            return max(300, int(val))
    except Exception:
        return 7200


def _watchdog_loop(app):
    """Monitor the worker for hangs. Runs in a separate daemon thread."""
    global _worker_generation, _worker_thread, _current_recording_id

    while True:
        _time.sleep(60)

        rec_id = _current_recording_id
        if rec_id is None:
            continue

        timeout = _get_watchdog_timeout(app)
        elapsed = _time.monotonic() - _last_heartbeat

        if elapsed <= timeout:
            continue

        elapsed_fmt = _fmt_elapsed(elapsed)
        timeout_fmt = _fmt_elapsed(timeout)
        log.error(
            'Watchdog: worker hung for %s (timeout=%s) on recording %d. '
            'Marking as error and restarting worker.',
            elapsed_fmt, timeout_fmt, rec_id)

        try:
            with app.app_context():
                recording = db.session.get(Recording, rec_id)
                if recording and recording.transcription_status == 'transcribing':
                    recording.transcription_status = 'error'
                    recording.error_message = (
                        f'Transcription timed out after {elapsed_fmt} '
                        f'(limit: {timeout_fmt})')
                    db.session.commit()
        except Exception as exc:
            log.error('Watchdog: failed to mark recording %d as error: %s',
                      rec_id, exc)

        _emit_status(rec_id, 'error',
                     message=f'Transcription timed out after {elapsed_fmt}')

        # Kill the ML worker process: a hung CUDA/Metal op cannot be
        # interrupted in-process, but the process boundary can.
        try:
            from .worker_client import kill_worker
            kill_worker()
        except Exception as exc:
            log.warning('Watchdog: failed to kill ML worker: %s', exc)

        _current_recording_id = None
        _worker_generation += 1

        _worker_thread = threading.Thread(
            target=_worker_loop, args=(app, _worker_generation),
            daemon=True, name=f'transcription-worker-gen{_worker_generation}')
        _worker_thread.start()
        log.info('Watchdog: new worker started (generation %d).',
                 _worker_generation)


def start_watchdog(app):
    """Start the watchdog thread (idempotent). Call after start_worker."""
    global _watchdog_thread
    if _watchdog_thread is not None and _watchdog_thread.is_alive():
        return
    _watchdog_thread = threading.Thread(
        target=_watchdog_loop, args=(app,),
        daemon=True, name='transcription-watchdog')
    _watchdog_thread.start()
    log.info('Transcription watchdog started.')


def requeue_interrupted(app):
    """Re-queue recordings stuck in 'transcribing' or 'awaiting_language' (from a crash)."""
    with app.app_context():
        stuck = Recording.query.filter(
            Recording.transcription_status.in_(('transcribing', 'awaiting_language'))
        ).all()
        for rec in stuck:
            rec.transcription_status = 'pending'
            db.session.commit()
            enqueue(rec.id)
        if stuck:
            log.info('Re-queued %d interrupted transcriptions', len(stuck))
