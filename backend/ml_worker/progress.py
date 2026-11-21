"""One 0→100 progress track for the whole job.

Engines and the diarizer report their own local 0→100 for whatever they happen
to be doing — chunk 2 of 3, diarize chunk 5 of 8. This module is the only place
that knows how those local scales add up to one job: every stage owns a band of
the global scale, sized by how long that stage is expected to take, and local
progress is mapped into its band.

Because band widths are proportional to expected cost, the global percentage is
(approximately) the fraction of the job's work that is done — which is what
makes the ETA a plain extrapolation of elapsed time.

Two properties the UI depends on:

* Monotonic. The number never goes backwards, so a stage that is skipped or
  fails (diarization is best-effort) reads as a jump forward, never a reset.
* Continuous. Engines only report on chunk boundaries, and short recordings
  report nothing at all, so a ticker interpolates between real milestones.

Both properties assume the stages are sequential, which is what makes the bands
disjoint. When the pipeline overlaps diarization with transcription it does not
report two stages at once — it keeps the bar on the stage the user is waiting
for and passes ``parallel=True`` here, which shrinks the diarize band to the
part that outlasts transcription. The overlapped work is then simply time the
bar does not have to account for twice.
"""

import logging
import threading
import time

from .constants import (
    CHUNK_THRESHOLD_SEC,
    PROGRESS_FINALIZE_SEC,
    PROGRESS_LEARN_MIN_SEC,
    PROGRESS_LOAD_SEC,
    PROGRESS_RTF_ALIGN,
    PROGRESS_RTF_DIARIZE,
    PROGRESS_RTF_TRANSCRIBE,
    PROGRESS_SCALE_MAX,
    PROGRESS_SCALE_MIN,
    PROGRESS_TICK_SEC,
)

log = logging.getLogger(__name__)

# Global scale, in order. Stages the active engine never emits (one without a
# separate 'aligning' pass, say) simply get skipped over.
STAGES = ('loading', 'transcribing', 'aligning', 'diarizing', 'finalizing')

# Don't let the ticker run all the way to the end of a band — the real
# milestone has to stay ahead of the guess.
_BAND_TICK_CAP = 0.97
# How much signal is needed before the job's observed pace beats the plan.
_PACE_MIN_FRACTION = 0.03
_PACE_MIN_ELAPSED_SEC = 15.0
# The plan is a prior: observation may scale it, but only this far. Without
# this a percent that arrives early (a cached model, a resumed job) reads as
# an absurdly fast machine and the bar rockets to the end of its band.
_PACE_CORRECTION_MIN = 0.25
_PACE_CORRECTION_MAX = 4.0
# With stages overlapped, diarization may well finish before transcription does.
# Keep it a sliver of a band anyway: the bar has to have somewhere to sit while
# "Identifying speakers…" is on screen, and a zero-width band reads as a freeze
# at whatever percent transcription happened to end on.
_PARALLEL_MIN_BAND = 0.15


class ProgressMapper:
    """Wraps a status callback and rewrites per-stage progress into one scale.

    Call it exactly like ``PipelineEvents.status``; it forwards every event with
    ``percent``/``eta_secs`` replaced by the global figures.
    """

    def __init__(self, total_duration, status, tick_secs=PROGRESS_TICK_SEC,
                 clock=time.monotonic, parallel=False, multitrack=False,
                 scale=1.0):
        self._status = status
        self._clock = clock
        self._tick_secs = tick_secs
        self._lock = threading.RLock()

        scale = _sane_scale(scale)
        self._expected = _expected_seconds(total_duration, parallel=parallel,
                                           multitrack=multitrack, scale=scale)
        # The same plan unscaled, so ``observed_scale`` can report an absolute
        # figure rather than one relative to whatever scale this job ran with.
        unscaled = _expected_seconds(total_duration, parallel=parallel,
                                     multitrack=multitrack, scale=1.0)
        self._flat_expected = PROGRESS_LOAD_SEC + PROGRESS_FINALIZE_SEC
        self._variable_expected = max(
            sum(unscaled.values()) - self._flat_expected, 0.0)
        self._audio_secs = max(float(total_duration or 0.0), 0.0)
        total = sum(self._expected.values()) or 1.0
        self._bands = {}
        acc = 0.0
        for stage in STAGES:
            width = self._expected[stage] / total * 100.0
            self._bands[stage] = (acc, acc + width)
            acc += width
        self._total_expected = total

        self._t0 = clock()
        self._paused_at = None
        self._paused_total = 0.0

        self._stage = STAGES[0]
        self._local = 0.0
        self._pct = 0.0
        self._message = ''
        self._anchor_pct = 0.0
        self._anchor_elapsed = 0.0

        self._stop = threading.Event()
        self._thread = None

    # ── lifecycle ──

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._tick_loop, name='progress', daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=self._tick_secs * 2)

    def pause(self):
        """Stop the clock while the job waits on the user (language prompt)."""
        with self._lock:
            if self._paused_at is None:
                self._paused_at = self._clock()

    def resume(self):
        with self._lock:
            if self._paused_at is not None:
                self._paused_total += self._clock() - self._paused_at
                self._paused_at = None

    def finish(self):
        with self._lock:
            self._stage = STAGES[-1]
            self._local = 100.0
            self._emit(self._message)

    def observed_scale(self):
        """What the plan's scale should have been for this job, or None.

        Measured against the duration-proportional part of the plan only — model
        loading and finalizing are flat costs that do not shrink on a faster
        machine, so folding them in would bias the figure on short recordings.

        Returns None when the recording is too short to measure anything
        trustworthy: on a one-minute clip the fixed costs dominate and the
        ratio says more about disk cache than about how fast this machine
        transcribes.
        """
        with self._lock:
            if (self._variable_expected <= 0
                    or self._audio_secs < PROGRESS_LEARN_MIN_SEC):
                return None
            spent = self._elapsed() - self._flat_expected
            if spent <= 0:
                return None
            return _sane_scale(spent / self._variable_expected)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    # ── the wrapped status callback ──

    def __call__(self, stage='', message='', percent=None, eta_secs=None,
                 **extra):
        with self._lock:
            if stage in self._bands:
                if stage != self._stage:
                    self._stage = stage
                    self._local = 0.0
                if percent is not None:
                    # Guard against an engine walking its own scale backwards.
                    self._local = max(self._local, min(float(percent), 100.0))
            if message:
                self._message = message
            self._emit(message or self._message, extra)

    # ── internals ──

    def _elapsed(self):
        paused = self._paused_total
        if self._paused_at is not None:
            paused += self._clock() - self._paused_at
        return max(self._clock() - self._t0 - paused, 0.0)

    def _milestone_pct(self):
        """Global percent implied by the last real event."""
        start, end = self._bands[self._stage]
        return start + (end - start) * self._local / 100.0

    def _advance(self, pct):
        """Clamp monotonically and re-anchor the interpolation."""
        pct = min(max(pct, self._pct), 100.0)
        self._pct = pct
        return pct

    def _predicted_total(self):
        """Best estimate of the job's total wall time, in seconds.

        Starts as the plan (stage weights x audio duration) and hands over to
        the job's own observed pace once there is enough of it to trust.

        Pace is measured against the last real milestone, never against the
        interpolated percent: feeding the ticker's own guess back in would let
        it accelerate itself to the end of the band.
        """
        elapsed = self._anchor_elapsed
        if (self._anchor_pct < _PACE_MIN_FRACTION * 100
                or elapsed < _PACE_MIN_ELAPSED_SEC):
            return self._total_expected
        observed = elapsed / (self._anchor_pct / 100.0)
        return min(max(observed, self._total_expected * _PACE_CORRECTION_MIN),
                   self._total_expected * _PACE_CORRECTION_MAX)

    def _drift_pct(self):
        """Best estimate of where the job actually is, between two milestones.

        Not the same number as the one on screen: the display is clamped so it
        never goes backwards, and when the ticker has run ahead of reality that
        clamp holds it still while this catches up. The ETA is derived from
        this one, because a remaining time computed from an inflated percent is
        inflated in exactly the same proportion.
        """
        start, end = self._bands[self._stage]
        cap = start + (end - start) * _BAND_TICK_CAP
        if self._anchor_pct >= cap:
            return self._anchor_pct
        predicted_total = self._predicted_total()
        if predicted_total <= 0:
            return self._anchor_pct
        gained = (self._elapsed() - self._anchor_elapsed) / predicted_total * 100.0
        return min(self._anchor_pct + max(gained, 0.0), cap)

    def _eta_secs(self, pct):
        if pct >= 100:
            return None
        return max(int(round(self._predicted_total() * (1.0 - pct / 100.0))), 0)

    def _emit(self, message, extra=None, from_tick=False):
        """Caller holds the lock."""
        if from_tick:
            actual = self._drift_pct()
        else:
            actual = self._milestone_pct()
            # Anchor on the milestone itself, never on what is displayed. When
            # the ticker has drifted past it, _advance holds the display where
            # it was — and recording *that* here would make the observed pace
            # come out equal to the plan by construction, so an estimate that
            # started out too optimistic could never correct itself upward.
            self._anchor_pct = actual
            self._anchor_elapsed = self._elapsed()
        pct = self._advance(actual)
        self._status(stage=self._stage, message=message or '',
                     percent=int(round(pct)), eta_secs=self._eta_secs(actual),
                     **(extra or {}))

    def _tick_loop(self):
        while not self._stop.wait(self._tick_secs):
            try:
                with self._lock:
                    if self._paused_at is not None:
                        continue
                    before = self._pct
                    if self._drift_pct() - before < 0.5:
                        continue
                    self._emit(self._message, from_tick=True)
            except Exception:
                log.debug('Progress tick failed', exc_info=True)


def _sane_scale(scale):
    """Keep a learned scale inside believable bounds."""
    try:
        value = float(scale)
    except (TypeError, ValueError):
        return 1.0
    if value <= 0:
        return 1.0
    return min(max(value, PROGRESS_SCALE_MIN), PROGRESS_SCALE_MAX)


def _expected_seconds(total_duration, parallel=False, multitrack=False,
                      scale=1.0):
    """How long each stage should take, in seconds, for this recording.

    With ``parallel``, diarization has been running underneath transcription
    and alignment, so it only adds wall time where it outlasts them.

    With ``multitrack`` there is no diarization at all — the speakers came in on
    separate tracks — and alignment happens once per track inside the
    transcribing stage rather than after it, so both bands fold into that one.
    A stage with no width is simply never visited.
    """
    secs = max(float(total_duration or 0.0), 0.0)
    # ``scale`` is what this machine has actually been measured doing, relative
    # to the shipped cost model. The flat costs stay put: loading a model off
    # disk does not get cheaper because the GPU is fast.
    expected = {
        'loading': PROGRESS_LOAD_SEC,
        'transcribing': secs * PROGRESS_RTF_TRANSCRIBE * scale,
        'aligning': secs * PROGRESS_RTF_ALIGN * scale,
        'diarizing': secs * PROGRESS_RTF_DIARIZE * scale,
        'finalizing': PROGRESS_FINALIZE_SEC,
    }
    if multitrack:
        expected['transcribing'] += expected['aligning']
        expected['aligning'] = 0.0
        expected['diarizing'] = 0.0
        return expected
    if secs >= CHUNK_THRESHOLD_SEC:
        # Long files are transcribed in chunks, and the chunked path aligns each
        # chunk as it goes while reporting the whole thing as 'transcribing'.
        # The aligning stage is therefore never visited on exactly the
        # recordings that need an ETA most — leaving it a band of its own put
        # real work in the denominator that no milestone ever paid off, so the
        # estimate ran short and the bar jumped when transcription ended.
        expected['transcribing'] += expected['aligning']
        expected['aligning'] = 0.0
    if parallel:
        hidden = expected['transcribing'] + expected['aligning']
        expected['diarizing'] = max(expected['diarizing'] - hidden,
                                    expected['diarizing'] * _PARALLEL_MIN_BAND)
    return expected
