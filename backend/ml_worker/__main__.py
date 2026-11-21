"""ML worker process entry point: ``python -m ml_worker``.

Reads ops as JSON lines on stdin, emits events as JSON lines on stdout,
logs to stderr. One job at a time; models stay loaded between jobs.

The protocol channel is a private duplicate of fd 1: ``sys.stdout`` itself is
redirected to stderr so stray ``print()`` calls inside ML libraries (e.g.
whisperx progress) can never corrupt the protocol stream.
"""

import logging
import os
import sys
import threading
import time
import traceback

from . import __version__, compat, protocol
from .errors import TranscriptionCancelled
from .pipeline import JobEnv, JobRequest, MLPipeline, PipelineEvents

log = logging.getLogger('ml_worker')


def _stdin_lines():
    """Yield protocol lines from stdin without ever blocking in a CRT read.

    On Windows, a thread sitting in a blocking read of fd 0 stalls concurrent
    DLL loading in this process: ``import numpy``/torch/cudnn hang for minutes
    (CRT lock vs. loader-lock interaction; measured empirically — a plain CPU
    loop is unaffected, imports crawl ~100x). Since the whole point of this
    process is to load big native libraries while staying responsive to
    cancel/shutdown ops, poll the pipe with PeekNamedPipe and only call
    ``os.read`` when bytes are already available.

    POSIX stdin has no such problem — plain blocking iteration is used there.
    """
    if os.name != 'nt':
        for line in sys.stdin:
            yield line
        return

    import ctypes
    import msvcrt
    kernel32 = ctypes.windll.kernel32
    handle = msvcrt.get_osfhandle(0)
    buf = b''
    while True:
        avail = ctypes.c_ulong(0)
        ok = kernel32.PeekNamedPipe(
            ctypes.c_void_p(handle), None, 0, None, ctypes.byref(avail), None)
        if not ok:
            break  # pipe broken / parent gone → EOF
        if avail.value == 0:
            time.sleep(0.05)
            continue
        chunk = os.read(0, 65536)
        if not chunk:
            break
        buf += chunk
        while b'\n' in buf:
            line, buf = buf.split(b'\n', 1)
            yield line.decode('utf-8', 'replace')
    if buf:
        yield buf.decode('utf-8', 'replace')


def _configure_logging():
    level_name = os.environ.get('PINE_LOG_LEVEL', '').strip().upper() or 'INFO'
    level = getattr(logging, level_name, logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)


class WorkerServer:
    def __init__(self):
        # Private protocol channel on a dup of fd 1; line-buffered UTF-8.
        self._proto = os.fdopen(
            os.dup(sys.stdout.fileno()), 'w',
            encoding='utf-8', buffering=1, newline='\n')
        # Anything that prints to stdout from here on lands in the log instead.
        sys.stdout = sys.stderr

        self._emit_lock = threading.Lock()
        self._pipeline = MLPipeline()
        self._job_thread = None
        self._current_recording_id = None
        self._cancel_event = threading.Event()
        self._language_lock = threading.Lock()
        self._language_code = None
        self._language_event = threading.Event()

    # ── outgoing ──

    def emit(self, **payload):
        with self._emit_lock:
            self._proto.write(protocol.dumps_line(payload))

    # ── job execution ──

    def _events_for(self, recording_id):
        def status(stage='', message='', percent=None, eta_secs=None, **extra):
            payload = dict(ev='status', recording_id=recording_id,
                           stage=stage, message=message, **extra)
            if percent is not None:
                payload['percent'] = percent
            if eta_secs is not None:
                payload['eta_secs'] = eta_secs
            self.emit(**payload)

        def check_cancel():
            if self._cancel_event.is_set():
                raise TranscriptionCancelled(recording_id)

        def request_language(probe):
            with self._language_lock:
                self._language_code = None
                self._language_event.clear()
            self.emit(
                ev='language_request', recording_id=recording_id,
                guessed=probe.code or '',
                confidence=float(probe.confidence),
                options=probe.options or [],
            )
            while not self._language_event.wait(timeout=1.0):
                check_cancel()
            check_cancel()
            with self._language_lock:
                return self._language_code

        return PipelineEvents(
            status=status,
            request_language=request_language,
            check_cancel=check_cancel,
        )

    def _run_job(self, env_dict, job_dict):
        recording_id = job_dict.get('recording_id')
        try:
            env = JobEnv(**env_dict)
            job = JobRequest(**job_dict)
            transcript = self._pipeline.run(env, job, self._events_for(recording_id))
            self.emit(ev='result', recording_id=recording_id, transcript=transcript)
        except TranscriptionCancelled:
            log.info('Job for recording %s cancelled.', recording_id)
            self.emit(ev='cancelled', recording_id=recording_id)
        except BaseException as exc:  # noqa: BLE001 — everything must become an event
            log.error('Job for recording %s failed:\n%s',
                      recording_id, traceback.format_exc())
            self.emit(ev='error', recording_id=recording_id,
                      message=str(exc)[:500] or type(exc).__name__,
                      error_type=type(exc).__name__)
        finally:
            self._current_recording_id = None
            self._cancel_event.clear()

    # ── incoming ──

    def handle(self, msg) -> bool:
        """Dispatch one op; returns False when the server should exit."""
        op = msg.get('op')
        if op == 'job':
            if self._job_thread is not None and self._job_thread.is_alive():
                self.emit(ev='error',
                          recording_id=(msg.get('job') or {}).get('recording_id'),
                          message='Worker busy: a job is already running',
                          error_type='WorkerBusy')
                return True
            self._cancel_event.clear()
            self._current_recording_id = (msg.get('job') or {}).get('recording_id')
            self._job_thread = threading.Thread(
                target=self._run_job,
                args=(msg.get('env') or {}, msg.get('job') or {}),
                daemon=True, name='ml-worker-job')
            self._job_thread.start()
        elif op == 'language':
            with self._language_lock:
                self._language_code = (msg.get('code') or '').strip().lower()
            self._language_event.set()
        elif op == 'cancel':
            self._cancel_event.set()
            self._language_event.set()   # wake a pending language wait
        elif op == 'ping':
            self.emit(ev='pong')
        elif op == 'shutdown':
            return False
        return True

    def serve(self):
        self.emit(ev='ready', pid=os.getpid(), version=__version__)
        for line in _stdin_lines():
            msg = protocol.parse_line(line)
            if msg is None:
                continue
            if not self.handle(msg):
                break
        # stdin closed → the backend is gone (or asked us to stop).
        # A job may still hold GPU state; nothing needs flushing — exit hard.
        if self._job_thread is not None and self._job_thread.is_alive():
            log.warning('Backend disappeared mid-job — aborting worker.')
            os._exit(1)


def main():
    _configure_logging()
    try:
        sys.stdin.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    if os.environ.get('PINE_WORKER_DEBUG_STACKS', '').strip() == '1':
        # Diagnose hangs: dump every thread's stack to stderr periodically.
        import faulthandler
        faulthandler.dump_traceback_later(30, repeat=True, file=sys.stderr)
    compat.disable_model_telemetry()
    compat.ensure_ca_bundle()
    log.info('ML worker starting (pid=%s, python=%s)', os.getpid(), sys.version.split()[0])
    WorkerServer().serve()
    log.info('ML worker exiting.')


if __name__ == '__main__':
    main()
