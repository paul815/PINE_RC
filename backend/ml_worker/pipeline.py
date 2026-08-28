"""MLPipeline — probe → transcribe ∥ diarize → clean/map, no Flask, no DB.

The caller (worker ``__main__`` or the in-process fallback) supplies a JobEnv
(paths and model choices resolved from settings), a JobRequest (one recording)
and PipelineEvents (status/progress, language confirmation, cancellation).
The result is the transcript JSON payload, byte-identical in shape to what
PINE has always written.
"""

import logging
import sys
import threading
import time
from dataclasses import dataclass, replace

from . import compat
from .audio import get_duration_secs, load_audio_range
from .constants import PARALLEL_STAGES, PROGRESS_SCALE_KEY, SPEAKER_LABELS
from .diarize import Diarizer, detect_diarize_device
from .engines import TranscribeContext, create_engine, engine_kind_for_model
from .errors import TranscriptionCancelled
from .multitrack import run_multitrack
from .progress import ProgressMapper

log = logging.getLogger(__name__)



@dataclass
class JobEnv:
    """Settings-derived context, resolved by the Flask side per job."""
    stt_model_id: str
    model_dir: str
    diarize_dir: str
    pyannote_cache: str
    hf_token: str = ''
    hf_offline: bool = False
    # How far off the shipped cost model this machine has been measured
    # running, for this model and mode. 1.0 = never measured; the pipeline
    # reports a fresh figure back on every job it finishes.
    progress_scale: float = 1.0


@dataclass
class TrackSpec:
    """One speaker's own audio, for a multi-track recording."""
    index: int
    path: str
    speaker_name: str = ''
    channel: int | None = None      # set when the track is a channel of `path`


@dataclass
class JobRequest:
    """One recording to transcribe."""
    recording_id: int
    audio_path: str
    num_speakers: int | None = None
    forced_language: str | None = None   # project preset; skips probing
    confirm_language: bool = True        # False → auto-detect, never ask
    # When set, every speaker already has their own track: each is transcribed
    # on its own and diarization is skipped. Empty → the single-file path, with
    # pyannote, exactly as before.
    tracks: list | None = None

    def track_specs(self):
        """``tracks`` as TrackSpec objects — they arrive as dicts over the wire."""
        out = []
        for i, raw in enumerate(self.tracks or []):
            if isinstance(raw, TrackSpec):
                out.append(raw)
                continue
            out.append(TrackSpec(
                index=int(raw.get('index', i)),
                path=raw.get('path', ''),
                speaker_name=raw.get('speaker_name', '') or '',
                channel=raw.get('channel'),
            ))
        out.sort(key=lambda t: t.index)
        return out


def _noop_status(**kwargs):
    return None


def _noop_check_cancel():
    return None


def _no_language_ui(probe):
    raise RuntimeError('Language confirmation requested but no UI is attached')


@dataclass
class PipelineEvents:
    """Callbacks from the pipeline to whoever runs it.

    status(stage=..., message=..., percent=..., eta_secs=..., **extra)
    request_language(probe: LanguageProbe) -> str — block until the user picks
        a language; raise TranscriptionCancelled to abort.
    check_cancel() — raise TranscriptionCancelled when the job must stop.
    """
    status: callable = _noop_status
    request_language: callable = _no_language_ui
    check_cancel: callable = _noop_check_cancel


def map_speakers(raw_segments, names=None):
    """Map raw speaker ids to display labels, in order of first appearance.

    ``names`` supplies real names where they are known — multi-track recordings
    carry the participant in the filename, so there is no reason to show
    "Participant 1" for someone the recording already named. Anything not in
    ``names`` falls back to the positional label, which is the whole of the
    single-file behaviour.
    """
    names = names or {}
    seen = {}
    mapping = {}
    used = set()
    for seg in raw_segments:
        spk = seg.get('speaker')
        if spk and spk not in seen:
            idx = len(seen)
            label = names.get(spk) or (
                SPEAKER_LABELS[idx] if idx < len(SPEAKER_LABELS)
                else f'Speaker {idx + 1}')
            # Two tracks can parse to the same name; labels have to stay
            # distinct or the transcript merges two people into one.
            if label in used:
                label = f'{label} ({idx + 1})'
            used.add(label)
            seen[spk] = label
            mapping[spk] = label

    for seg in raw_segments:
        spk = seg.get('speaker')
        if spk and spk in mapping:
            seg['speaker'] = mapping[spk]

    return mapping


def clean_transcript_segments(segments):
    """Drop empty/degenerate segments and words before persisting transcript JSON."""
    clean_segments = []
    for seg in segments or []:
        start = float(seg.get('start', 0) or 0)
        end = float(seg.get('end', 0) or 0)
        had_words = bool(seg.get('words'))
        words_out = []
        for w in seg.get('words', []) or []:
            w_start = float(w.get('start', 0) or 0)
            w_end = float(w.get('end', 0) or 0)
            word = str(w.get('word', '') or '').strip()
            if not word or w_end <= w_start:
                continue
            words_out.append({
                'word': word,
                'start': w_start,
                'end': w_end,
            })

        text = str(seg.get('text', '') or '').strip()
        if words_out:
            text = ' '.join(w['word'] for w in words_out)
        elif had_words:
            # MLX artifact: keep neither empty nor all-zero-word segments.
            text = ''

        if not text or end <= start:
            continue

        clean_seg = {
            'start': start,
            'end': end,
            'text': text,
            'speaker': seg.get('speaker', ''),
        }
        if words_out:
            clean_seg['words'] = words_out
        clean_segments.append(clean_seg)
    return clean_segments


def _log_diarization_failure(exc):
    """Diarization is best-effort: a failure costs speaker labels, not the job."""
    cause = exc
    for _ in range(3):
        nxt = getattr(cause, '__cause__', None) or getattr(cause, '__context__', None)
        if nxt is None:
            break
        cause = nxt
    try:
        msg = str(cause)[:300]
    except Exception:
        msg = type(cause).__name__
    log.warning('Diarization failed: %s - skipping speaker labels', msg)


class _DiarizeTask:
    """The audio-only half of diarization, optionally run under the STT stage.

    Owns the thread so the pipeline never leaves one behind: the ML worker
    process is reused between jobs rather than killed, so a diarization still
    running after its job ended would compete with the next one for the GPU.
    Every exit path joins.

    Errors are held rather than raised in the thread, and handed back on
    ``result()``, which keeps diarization best-effort exactly as it was when
    it ran inline — except cancellation, which is the job ending and must
    propagate.
    """

    def __init__(self, diarizer, job, total_duration, check_cancel):
        self._diarizer = diarizer
        self._job = job
        self._total_duration = total_duration
        self._check_cancel = check_cancel
        self._thread = None
        self._result = None
        self._error = None
        self.elapsed = 0.0

    @property
    def started(self):
        return self._thread is not None

    def start(self):
        self._thread = threading.Thread(
            target=self._work, name='diarize', daemon=True)
        self._thread.start()

    def run(self, diarize_input=None, on_status=_noop_status):
        """Run it here and now — the serial path."""
        self._work(diarize_input, on_status=on_status)

    def _work(self, diarize_input=None, on_status=_noop_status):
        t0 = time.monotonic()
        try:
            self._result = self._diarizer.compute(
                diarize_input if diarize_input is not None else self._job.audio_path,
                self._job.recording_id,
                audio_path=self._job.audio_path,
                total_duration=self._total_duration,
                num_speakers=self._job.num_speakers,
                on_status=on_status,
                check_cancel=self._check_cancel)
        except BaseException as exc:      # noqa: BLE001 — re-raised in result()
            self._error = exc
        finally:
            self.elapsed = time.monotonic() - t0

    def join(self):
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join()

    def result(self):
        """Join and return the diarization, or None if it did not produce one."""
        self.join()
        log.info('PERF: diarization took %.1fs', self.elapsed)
        if self._error is not None:
            if isinstance(self._error, TranscriptionCancelled):
                raise self._error
            _log_diarization_failure(self._error)
            return None
        return self._result


class MLPipeline:
    """Holds loaded models between jobs and runs the full pipeline per job."""

    def __init__(self):
        self._engine = None
        self._engine_key = None       # (stt_model_id, model_dir)
        self._diarizer = None
        self._diarizer_key = None     # (engine_kind, diarize_dir)

    def _ensure_models(self, env: JobEnv, need_diarizer=True):
        compat.patch_torch_load_for_trusted_checkpoints()

        engine_kind = engine_kind_for_model(env.stt_model_id)
        prefer_mps = (sys.platform == 'darwin'
                      and detect_diarize_device() != 'cpu')

        engine_key = (env.stt_model_id, env.model_dir)
        if self._engine is None or self._engine_key != engine_key:
            self._engine = create_engine(env, prefer_mps=prefer_mps)
            self._engine.load()
            self._engine_key = engine_key
        else:
            self._engine.env = env

        if not need_diarizer:
            # Multi-track recordings answer the speaker question themselves.
            # Loading pyannote anyway would cost seconds and a GPU allocation
            # for a model that is never called.
            return

        diarizer_key = (engine_kind, env.diarize_dir)
        if self._diarizer is None or self._diarizer_key != diarizer_key:
            self._diarizer = Diarizer(env, engine_kind=engine_kind)
            self._diarizer.load()
            self._diarizer_key = diarizer_key
        else:
            self._diarizer.env = env

    def _resolve_language(self, env: JobEnv, job: JobRequest,
                          events: PipelineEvents, total_duration: float):
        """Optional probe + user confirmation. Returns whisper language code or None for auto."""
        if job.forced_language:
            log.info('Using preset transcription language for recording %s: %s',
                     job.recording_id, job.forced_language)
            return job.forced_language
        if not job.confirm_language:
            return None

        probe_secs = min(30.0, max(total_duration, 0.5) if total_duration > 0 else 30.0)
        try:
            probe_audio = load_audio_range(job.audio_path, 0, probe_secs)
        except Exception as exc:
            log.warning('Language probe: could not load audio sample: %s', exc)
            return None

        try:
            probe = self._engine.probe_language(job.audio_path, probe_audio)
        except Exception as exc:
            log.warning('Language probe (%s) failed: %s', self._engine.id, exc)
            return None
        if probe is None:
            return None
        if probe.uncertain:
            log.info(
                'Language uncertain (%s): best=%s p=%.2f second=%.2f — asking user',
                self._engine.id, probe.code, probe.confidence, probe.second_confidence)
            chosen = events.request_language(probe)
            if not chosen:
                raise RuntimeError('Language confirmation ended without a language')
            return chosen.strip().lower()
        return probe.code

    def run(self, env: JobEnv, job: JobRequest, events: PipelineEvents) -> dict:
        """Run the full pipeline for one recording; returns the transcript payload."""
        compat.apply_hf_offline(env.hf_offline)

        pipeline_t0 = time.monotonic()
        total_duration = get_duration_secs(job.audio_path)

        # Everything downstream reports its own local 0→100; the mapper folds
        # those into the single scale the UI shows. Nothing below this line
        # should emit a percent of its own.
        multitrack = bool(job.tracks)
        progress = ProgressMapper(total_duration, events.status,
                                  parallel=PARALLEL_STAGES and not multitrack,
                                  multitrack=multitrack,
                                  scale=env.progress_scale).start()
        events = replace(events, status=progress)
        events.status(stage='loading')

        try:
            return self._run(env, job, events, progress,
                             pipeline_t0, total_duration)
        finally:
            progress.stop()

    def _run(self, env, job, events, progress, pipeline_t0, total_duration):
        recording_id = job.recording_id
        specs = job.track_specs()

        self._ensure_models(env, need_diarizer=not specs)
        log.info('PERF: model loading took %.1fs', time.monotonic() - pipeline_t0)

        # The language prompt blocks on the user — their thinking time is not
        # the job being slow, so keep it out of the ETA.
        progress.pause()
        try:
            forced_lang = self._resolve_language(env, job, events, total_duration)
        finally:
            progress.resume()

        if specs:
            segments, speaker_names, detected_lang = run_multitrack(
                self._engine, specs, forced_lang, events)
        else:
            segments, detected_lang, total_duration = self._transcribe_single(
                job, events, forced_lang, pipeline_t0, total_duration)
            speaker_names = None

        events.status(stage='finalizing', message='Finishing up...')
        speaker_map = map_speakers(segments, speaker_names)
        clean_segments = clean_transcript_segments(segments)
        progress.finish()

        elapsed_total = time.monotonic() - pipeline_t0
        log.info('PERF: total pipeline took %.1fs for %.0fs audio (%.2fx realtime)',
                 elapsed_total, total_duration, elapsed_total / max(total_duration, 1))

        payload = {
            'recording_id': recording_id,
            'language': detected_lang,
            'duration_seconds': total_duration,
            'speakers': speaker_map,
            'segments': clean_segments,
            # Which engine and model produced this file. Recorded so a future
            # format change, or a quality regression, can be traced to what
            # actually ran — see app/services/transcript_format.py.
            'engine': getattr(self._engine, 'id', ''),
            'model': env.stt_model_id or '',
        }
        # How this job actually compared to the plan, for the next one to start
        # from. Rides along in the payload so both execution paths carry it
        # without a protocol change; the Flask side pops it before the
        # transcript is written, leaving the file's shape untouched.
        measured = progress.observed_scale()
        if measured is not None:
            payload[PROGRESS_SCALE_KEY] = measured
            log.info('PERF: progress plan was off by %.2fx (was %.2fx)',
                     measured, env.progress_scale)
        return payload

    def _transcribe_single(self, job, events, forced_lang, pipeline_t0,
                           total_duration):
        """One file, speakers told apart afterwards by pyannote."""
        ctx = TranscribeContext(
            on_status=events.status,
            check_cancel=events.check_cancel,
        )

        diarize = _DiarizeTask(self._diarizer, job, total_duration,
                               events.check_cancel)
        if PARALLEL_STAGES:
            # Started before transcription rather than after it: pyannote reads
            # only the audio. Its progress is deliberately not reported while it
            # runs underneath the STT stage — ProgressMapper gives each stage a
            # disjoint band and clamps monotonically, so two stages reporting at
            # once would pin the bar to the diarize band and silently swallow
            # transcription's progress.
            diarize.start()

        try:
            # The single-file path hands over the recording untouched. Stated
            # rather than assumed: the engine is cached across jobs, and a
            # multitrack job before this one leaves the flag set.
            self._engine.presegmented = False
            self._engine.presegmented_regions = None
            out = self._engine.transcribe(
                job.audio_path, total_duration, forced_lang, ctx)
            total_duration = out.duration_seconds
            detected_lang = out.language

            stt_elapsed = time.monotonic() - pipeline_t0
            log.info('PERF: STT (%s) took %.1fs', self._engine.id, stt_elapsed)

            events.check_cancel()

            diarize_t0 = time.monotonic()
            events.status(stage='diarizing', message='Identifying speakers...')

            if not diarize.started:
                # Serial path: hand it the waveform the engine already decoded.
                diarize.run(out.audio, on_status=events.status)
            diarization = diarize.result()
            log.info('PERF: waited %.1fs for diarization after STT',
                     time.monotonic() - diarize_t0)
        finally:
            diarize.join()

        segments = out.segments
        if diarization is not None:
            try:
                segments = self._diarizer.assign(diarization, segments)
            except Exception as exc:      # noqa: BLE001 — best-effort labels
                _log_diarization_failure(exc)

        if out.audio is not None:
            out.audio = None
            import gc
            gc.collect()

        return segments, detected_lang, total_duration
